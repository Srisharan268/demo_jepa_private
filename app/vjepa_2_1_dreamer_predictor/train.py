# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time
import wandb
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_2_1_dreamer_predictor.transforms import make_transforms
from app.vjepa_2_1_dreamer_predictor.utils import init_opt, init_video_model, load_checkpoint
from app.vjepa_2_1_dreamer_predictor.dataset import init_data
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


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    dreamer_predictor_path = cfgs_meta.get("dreamer_predictor_checkpoint", None)
    load_path = cfgs_meta.get("pretrain_checkpoint", None)
    pretrain_encoder_key = cfgs_meta.get("pretrain_encoder_key", "encoder")
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

    # -- MASK

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    unfreeze_vit = cfgs_model.get("unfreeze_vit", False)
    fusion_type = cfgs_model.get("dreamer_predictor_fusion_type", "conv3d")
    # -- DATA
    cfgs_data = args.get("data")
    dataset = cfgs_data.get("dataset")
    batch_size = cfgs_data.get("batch_size")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    camera_views = cfgs_data.get("camera_views")
    data_type = cfgs_data.get("data_type", "real")
    variance_step = cfgs_data.get("variance_step", 0)

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
    normalize_reps = cfgs_loss.get("normalize_reps", True)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    # Number of micro-batches accumulated per optimizer step. Effective batch is
    # batch_size * world_size * accum_steps. Default 1 == upstream behaviour.
    accum_steps = int(cfgs_opt.get("accum_steps", 1))
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")

    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    wandb_cfg = args.get("wandb")
    project = wandb_cfg.get("project")
    name = wandb_cfg.get("name")
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

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

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    if rank == 0:
        wandb.init(project=project, name=name)
    else:
        wandb.init(mode="disabled")

    
    # -- init model
    encoder, dreamer_predictor = init_video_model(
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
        model_cfg=cfgs_model,
        fusion_type=fusion_type,
    )
    if unfreeze_vit:
        target_encoder = deepcopy(encoder)
    else:
        target_encoder = None

    if compile_model:
        logger.info("Compiling encoder and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        dreamer_predictor.compile()


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
        transform=transform,
        rank=rank,
        world_size=world_size,
        camera_views=camera_views,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
        pin_mem=pin_mem,
        data_type=data_type,
        variance_step=variance_step,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")


    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        dreamer_predictor=dreamer_predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        unfreeze_vit=unfreeze_vit,
    )

    # for n, p in encoder.named_parameters():
    #     p.requires_grad = False
    # encoder.eval()

    #siglip loss
    #t_prime, b, siglip = init_siglip_loss(optimizer, lr, device)


    #encoder = DistributedDataParallel(encoder, static_graph=True)
    #encoder = torch.nn.DataParallel(encoder).to(device)

    # encoder = encoder.to(device)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    if target_encoder is not None:
        target_encoder = DistributedDataParallel(target_encoder, static_graph=True)
    dreamer_predictor = DistributedDataParallel(dreamer_predictor, static_graph=True)
    
    

    start_epoch = 0
    # -- load training checkpoint
    if load_path or dreamer_predictor_path:
        (
            encoder,
            target_encoder,
            dreamer_predictor,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            dreamer_predictor_path=dreamer_predictor_path,
            encoder=encoder,
            target_encoder=target_encoder,
            dreamer_predictor=dreamer_predictor,
            opt=optimizer,
            scaler=scaler,
            encoder_key=pretrain_encoder_key,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
    
    if target_encoder is not None:
        for param in target_encoder.parameters():
            param.requires_grad = False
        target_encoder.eval()
    

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            # "dreamer": {k: v for k, v in dreamer.state_dict().items() if k.startswith("module.adapter")},
            "dreamer_predictor": {k: v for k, v in dreamer_predictor.state_dict().items()},
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
        # -- update distributed-data-loader epoch

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
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            # -- collect `accum_steps` micro-batches; one optimizer step consumes all of them
            micro_batches = []
            for _ in range(accum_steps):
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
                            logger.warning(
                                f"Encountered exception when loading data (num retries {iter_retries}):\n{e}"
                            )
                            iter_retries += 1
                            time.sleep(5)
                        else:
                            logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                            raise e

                current_frame, target_frame, current_reference, target_reference = sample
                micro_batches.append(
                    (
                        current_frame.to(device, dtype=dtype),
                        target_frame.to(device, dtype=dtype),
                        current_reference.to(device, dtype=dtype),
                        target_reference.to(device, dtype=dtype),
                    )
                )
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_feature(frame_encoder, frame):
                    with torch.no_grad():
                        feature = frame_encoder(frame)
                        if normalize_reps:
                            feature = F.layer_norm(feature, (feature.size(-1),))
                        return feature
                

                def forward_dreamer(current_feature, current_reference_feature, target_reference_feature):
                    dreamer_output = dreamer_predictor(xt = current_feature, yt = current_reference_feature, yt_plus_1 = target_reference_feature)
                    return dreamer_output

                def loss_fn(dreamer_output, target_feature):
                    # loss = siglip(dreamer_output, target_feature, t_prime, b)
                    loss = F.l1_loss(dreamer_output, target_feature, reduction='mean')
                    #loss = F.mse_loss(dreamer_output, target_feature)
                    #loss = 1 - F.cosine_similarity(dreamer_output, target_feature)
                    #loss = loss.mean()
                    return loss

                # Step 1+2. Forward/backward over each micro-batch. Gradients accumulate
                # into .grad; the optimizer steps once below, so the effective batch is
                # batch_size * world_size * accum_steps.
                optimizer.zero_grad()
                n_micro = len(micro_batches)
                micro_losses = []
                for _i, (current_frame, target_frame, current_reference, target_reference) in enumerate(micro_batches):
                    # Suppress DDP's all-reduce until the final micro-batch: syncing on
                    # every one is correct but wastes n_micro-1 collectives per step.
                    _sync = nullcontext() if _i == n_micro - 1 else dreamer_predictor.no_sync()
                    with _sync:
                        with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                            if unfreeze_vit:
                                target_feature = forward_feature(target_encoder, target_frame)
                            else:
                                target_feature = forward_feature(encoder, target_frame)
                            current_feature = forward_feature(encoder, current_frame)
                            current_reference_feature = forward_feature(encoder, current_reference)
                            target_reference_feature = forward_feature(encoder, target_reference)
                            dreamer_output = forward_dreamer(current_feature, current_reference_feature, target_reference_feature)
                            # Divide so the accumulated gradient equals the gradient of the
                            # mean loss over the full effective batch.
                            loss = loss_fn(dreamer_output, target_feature) / n_micro

                        if mixed_precision:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    micro_losses.append(float(loss) * n_micro)

                # Report the unscaled mean loss, matching upstream's logged scale.
                loss = sum(micro_losses) / n_micro

                # Unscale once, on the fully-accumulated gradient
                if mixed_precision:
                    scaler.unscale_(optimizer)

                # Gradient clipping for training stability
                # clip_grad_norm_ returns the original (unclipped) gradient norm
                max_grad_norm = 1.0
                grad_norm_unclipped = torch.nn.utils.clip_grad_norm_(
                    dreamer_predictor.parameters(), 
                    max_norm=max_grad_norm
                )
                # The effective clipped norm
                grad_norm_clipped = min(float(grad_norm_unclipped), max_grad_norm)
                
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()



                return (
                    float(loss),
                    _new_lr,
                    _new_wd,
                    float(grad_norm_unclipped),
                    grad_norm_clipped,
                )

            (
                loss,
                _new_lr,
                _new_wd,
                _grad_norm_unclipped,
                _grad_norm_clipped,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "[wd: %.2e] [lr: %.2e] "
                        "[grad: %.2e -> %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            _new_wd,
                            _new_lr,
                            _grad_norm_unclipped,
                            _grad_norm_clipped,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
            def wandb_log_stats():
                #csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                #if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                wandb.log({
                    "epoch": epoch + 1,
                    "itr": itr,
                    "loss": loss_meter.avg,
                    "wd": _new_wd,
                    "lr": _new_lr,
                    "grad_norm_unclipped": _grad_norm_unclipped,
                    "grad_norm_clipped": _grad_norm_clipped,
                    "mem": torch.cuda.max_memory_allocated() / 1024.0**2,
                    "iter": iter_time_meter.avg,
                    "gpu": gpu_time_meter.avg,
                    "data": data_elapsed_time_meter.avg,
                })

            log_stats()
            wandb_log_stats()
            # assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
