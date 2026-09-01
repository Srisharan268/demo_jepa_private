# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time
import wandb

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.utils.single_gpu import wrap_ddp

from app.vjepa_2_1_dreamer_ac.dataset import init_data
from app.vjepa_2_1_dreamer_ac.transforms import make_transforms
from app.vjepa_2_1_dreamer_ac.utils import (
    init_encoder_predictor,
    init_dreamer_predictor,
    load_pretrained_encoder_predictor,
    load_dreamer_predictor,
    init_opt,
    vjepa_2_1_encoder_args_from_cfg,
)

from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


import warnings

warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message="`torch.backends.cuda.sdp_kernel",
    category=FutureWarning,
)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    dreamer_file = cfgs_meta.get("dreamer_predictor_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", True)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    unfreeze_dreamer_predictor = cfgs_model.get("unfreeze_dreamer_predictor", False)
    dreamer_predictor_fusion_type = cfgs_model.get("dreamer_predictor_fusion_type", "conv3d")

    # -- DATA
    cfgs_data = args.get("data")
    dataset = cfgs_data.get("dataset", None)
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    camera_frame = cfgs_data.get("camera_frame", False)
    camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])
    stereo_view = cfgs_data.get("stereo_view", False)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    data_fps = cfgs_data.get("data_fps")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    data_type = cfgs_data.get("data_type", "sim")

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    normalize_reps = cfgs_loss.get("normalize_reps")
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
    # --
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    wandb_config = args.get("wandb")
    project = wandb_config.get("project")
    name = wandb_config.get("name")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank == 0:
        # The config ships `name: test`, so every run would be called "test" and
        # a batch-size sweep would produce N indistinguishable runs. Suffix the
        # things that actually vary, and log the full config so you can tell
        # afterwards WHICH settings produced a curve.
        _bs = args["data"]["batch_size"]
        _ac = args["optimization"].get("accum_steps", 1)
        _run = f"{name}-b{_bs}xa{_ac}-{time.strftime('%m%d-%H%M%S')}"
        wandb.init(project=project, name=_run, config=args)
        logger.info(f"wandb run: {_run}  (mode={os.environ.get('WANDB_MODE', 'online')})")
    else:
        wandb.init(mode="disabled")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    # -- init model (V-JEPA 2.1 encoder)
    encoder, predictor = init_encoder_predictor(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        **vjepa_2_1_encoder_args_from_cfg(cfgs_model),
    )
    # cloud-test: target_encoder is never updated (no EMA, no momentum write) and
    # loads from the same checkpoint key as encoder — deepcopy wastes ~4.05 GB.
    # Both are frozen for the duration of stage 2, so sharing is safe.
    target_encoder = encoder
    dreamer_predictor = init_dreamer_predictor(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        dreamer_predictor_fusion_type=dreamer_predictor_fusion_type,
    )

    if compile_model:
        logger.info("Compiling encoder/target_encoder and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()   # target_encoder is the same object — no duplicate compile
        predictor.compile()
        dreamer_predictor.compile()

    video_collator = torch.utils.data.default_collate
    transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        dataset=dataset,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        data_fps=data_fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        stereo_view=stereo_view,
        transform=transform,
        collator=video_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        data_type=data_type,
    )
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        dreamer_predictor=dreamer_predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        unfreeze_dreamer_predictor=unfreeze_dreamer_predictor,
    )
    encoder = wrap_ddp(encoder, world_size, static_graph=True)
    # cloud-test: target_encoder is the same object — alias after wrap so both
    # names refer to the same (already-wrapped) module. No second wrap needed.
    target_encoder = encoder
    predictor = wrap_ddp(predictor, world_size, static_graph=False, find_unused_parameters=True)
    logger.info("Encoder (shared as target_encoder) and predictor have been wrapped with DDP.")
    dreamer_predictor = wrap_ddp(dreamer_predictor, world_size)
    logger.info("Dreamer has been wrapped with DDP.")

    # -- load pretrained weights
    logger.info("Loading pretrained weights...")
    encoder, predictor, _ = load_pretrained_encoder_predictor(
        r_path=p_file,
        encoder=encoder,
        predictor=predictor,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
        target_encoder=None,    # cloud-test: same object as encoder, skip redundant load
        load_predictor=load_predictor,
        load_encoder=load_encoder,
    )
    # Re-alias so target_encoder stays in sync (encoder may be reassigned by load fn)
    target_encoder = encoder
    for p in encoder.parameters():
        p.requires_grad = False

    # -- load dreamer checkpoint
    logger.info("Loading dreamer checkpoint...")
    dreamer_predictor = load_dreamer_predictor(
        dreamer_predictor_path=dreamer_file,
        dreamer_predictor=dreamer_predictor,
    )
    if unfreeze_dreamer_predictor:
        for p in dreamer_predictor.parameters():
            p.requires_grad = True
    else:
        for p in dreamer_predictor.parameters():
            p.requires_grad = False

    start_epoch = 0

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            # cloud-test: encoder and target_encoder are the same object.
            # Save only under "target_encoder"; make_deploy_ckpt.py synthesises
            # "encoder" from it, so nothing downstream breaks.
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            def load_clips():
                primary_images, reference_images, primary_actions, primary_states = sample
                return primary_images.to(device, non_blocking=True), reference_images.to(device, non_blocking=True), primary_actions.to(device, dtype=torch.float, non_blocking=True), primary_states.to(device, dtype=torch.float, non_blocking=True), None

            primary_images, reference_images, actions, states, extrinsics = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(c):
                    with torch.no_grad():
                        c = (
                            c.permute(0, 2, 1, 3, 4)
                            .flatten(0, 1)
                            .unsqueeze(2)
                            .repeat(1, 1, tubelet_size, 1, 1)
                        )
                        # V-JEPA 2.1: training=False returns embed_dim output
                        h = target_encoder(c, masks=None, training=False)
                        h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
                        if normalize_reps:
                            h = F.layer_norm(h, (h.size(-1),))
                        return h

                def forward_dreamer_predictor_target(current_feature, current_reference_feature, target_reference_feature):
                    with torch.no_grad():
                        dreamer_output = dreamer_predictor(xt=current_feature, yt=current_reference_feature, yt_plus_1=target_reference_feature)
                    return dreamer_output

                def forward_predictions(z):

                    def _step_predictor(_z, _a, _s, _e):
                        _z = predictor(_z, _a, _s, _e)
                        if normalize_reps:
                            _z = F.layer_norm(_z, (_z.size(-1),))
                        return _z

                    # -- one step of predictor with teacher forcing
                    if use_extrinsics:
                        _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
                    else:
                        _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], None
                    z_tf = _step_predictor(_z, _a, _s, _e)

                    # -- full auto-regressive rollouts of predictor
                    _z = torch.cat([z[:, : tokens_per_frame], z_tf[:, : tokens_per_frame]], dim=1)
                    for n in range(1, auto_steps):
                        if use_extrinsics:
                            _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], extrinsics[:, : n + 1]
                        else:
                            _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], None
                        _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                        _z = torch.cat([_z, _z_nxt], dim=1)
                    z_ar = _z[:, tokens_per_frame:]

                    return z_tf, z_ar

                def loss_fn(z, h):
                    if z.shape[1] != h.shape[1]:
                        _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
                    else:
                        _h = h
                    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

                def loss_fn_dreamer(z, dreamer_h):
                    if z.shape[1] == dreamer_h.shape[1]:
                        _h = dreamer_h
                    else:
                        _h = dreamer_h[:, : z.size(1)]
                    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(primary_images)
                    # current_ref = forward_target(reference_images[:-1])
                    # ref_feature = forward_target(reference_images[1:])
                    ref_feature = forward_target(reference_images)
                    dreamer_target_feature = forward_dreamer_predictor_target(
                        h[:, :-tokens_per_frame],
                        ref_feature[:, :-tokens_per_frame],
                        ref_feature[:, tokens_per_frame:],
                    )
                    z_tf, z_ar = forward_predictions(h)
                    jloss = loss_fn_dreamer(z_tf, dreamer_target_feature)
                    sloss = loss_fn_dreamer(z_ar, dreamer_target_feature)
                    loss = jloss + sloss

                # Step 2. Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return (
                    float(loss),
                    float(jloss),
                    float(sloss),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                jloss,
                sloss,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [%.2f, %.2f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jloss_meter.avg,
                            sloss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
            def wandb_log_stats():
                wandb.log({
                    "epoch": epoch + 1,
                    "itr": itr,
                    "loss": loss_meter.avg,
                    "jloss": jloss_meter.avg,
                    "sloss": sloss_meter.avg,
                    # sloss is the AUTOREGRESSIVE rollout (predictor consuming
                    # its own output) -- which is exactly what MPC does at
                    # deploy time. jloss is teacher-forced and therefore easy.
                    # A widening gap means one-step prediction is fine but error
                    # compounds under autoregression, i.e. good jloss curves that
                    # will still roll out badly. This is the number that predicts
                    # rollout success.
                    "sloss_minus_jloss": sloss_meter.avg - jloss_meter.avg,
                    "sloss_over_jloss": sloss_meter.avg / max(jloss_meter.avg, 1e-9),
                    "wd": _new_wd,
                    "lr": _new_lr,
                    "mem": torch.cuda.max_memory_allocated() / 1024.0**2,
                    "iter": iter_time_meter.avg,
                    "gpu": gpu_time_meter.avg,
                    "data": data_elapsed_time_meter.avg,
                })

            log_stats()
            wandb_log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
