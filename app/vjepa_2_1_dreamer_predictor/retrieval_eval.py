import os

import copy
import gc
import random
import time
from pydantic_core.core_schema import NoneSchema
import wandb
import argparse
import yaml
import gc
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
from src.utils.checkpoint_loader import robust_checkpoint_loader


__all__ = ["compute_cosine_similarity_matrix", "retrieval_eval"]


def compute_cosine_similarity_matrix(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two batches of vectors.

    Both inputs must have the same shape [batch_size, feature_dim].

    Args:
        queries: Tensor of shape [B, D]
        candidates: Tensor of shape [B, D]
        eps: Small constant to avoid division by zero

    Returns:
        sim: Tensor of shape [B, B], where sim[i, j] = cosine(q_i, c_j)
    """
    if queries.dim() != 2 or candidates.dim() != 2:
        raise ValueError("queries and candidates must be 2D tensors of shape [B, D]")
    if queries.shape != candidates.shape:
        raise ValueError("queries and candidates must have the same shape [B, D]")

    queries = F.normalize(queries, p=2, dim=1, eps=eps)
    candidates = F.normalize(candidates, p=2, dim=1, eps=eps)
    # [B, D] @ [D, B] -> [B, B]
    sim = queries @ candidates.t()
    return sim


def retrieval_eval(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    topk=(1,),
    eps: float = 1e-8,
):
    """
    Evaluate retrieval by cosine similarity between two batches of embeddings.

    Given queries Q and candidates C with the same shape [B, D], compute the
    cosine similarity matrix S = Q*C^T. For each i in [0..B-1], a retrieval is
    considered correct at top-k if index i is within the top-k most similar
    candidates for query i.

    Args:
        queries: Tensor of shape [B, D]
        candidates: Tensor of shape [B, D]
        topk: Iterable of k values to compute accuracy@k (e.g., (1, 5, 10))
        eps: Small constant to avoid division by zero in normalization

    Returns:
        result: dict with fields
            - "acc@{k}": float accuracy for each requested k
            - "correct@1": Bool tensor of shape [B] for top-1 correctness
            - "top1_index": Long tensor of shape [B] with predicted indices
            - "similarity": Tensor [B, B] cosine similarity matrix
    """
    if not isinstance(topk, (list, tuple)):
        topk = (int(topk),)

    # 将输入展平成 [B, D] 形状
    queries = queries.flatten(1)
    candidates = candidates.flatten(1)
    sim = compute_cosine_similarity_matrix(queries, candidates, eps=eps)

    batch_size = sim.shape[0]
    device = sim.device
    target = torch.arange(batch_size, device=device)

    # Top-1
    top1_vals, top1_idx = sim.max(dim=1)
    correct_top1 = top1_idx.eq(target)

    result = {
        "acc@1": correct_top1.float().mean().item(),
        "correct@1": correct_top1,
        "top1_index": top1_idx,
        "similarity": sim,
    }

    # Additional top-k
    unique_topk = sorted({int(k) for k in topk if int(k) > 1})
    for k in unique_topk:
        k = min(k, batch_size)
        topk_idx = sim.topk(k, dim=1).indices
        correct_at_k = topk_idx.eq(target.unsqueeze(1)).any(dim=1)
        result[f"acc@{k}"] = correct_at_k.float().mean().item()

    return result

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    dreamer_predictor_path = "/mnt/jayyoung/codeHub/vjepa2/exp/jepa_2_1_dreamer_predictor/e150.pt"
    data_path = "/mnt/log2r/jepa_data/stage1/40tasks"

    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    # dreamer_predictor_path = cfgs_meta.get("dreamer_predictor_checkpoint", "/mnt/jayyoung/codeHub/vjepa2/exp/dreamer_predictor_515_resume_slr/e100.pt")
    load_model = True
    load_path = cfgs_meta.get("pretrain_checkpoint", None)
    r_file = cfgs_meta.get("read_checkpoint", None)
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

    # -- DATA
    cfgs_data = args.get("data")
    # data_path = cfgs_data.get("data_path")
    batch_size = 64
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    camera_views = cfgs_data.get("camera_views")
    target_robot_type = cfgs_data.get("target_robot_type", "panda")
    reference_robot_type = cfgs_data.get("reference_robot_type", "sawyer")
    fps = cfgs_data.get("fps")

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    normalize_reps = cfgs_loss.get("normalize_reps", True)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")

    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    wandb.init(project="vjepa_2_1_dreamer_predictor_retrieval_eval", name=f"vjepa_2_1_retrieval_eval", config=args)

    
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
    )


    # transform = make_transforms(
    #     random_horizontal_flip=True,
    #     random_resize_aspect_ratio=ar_range,
    #     random_resize_scale=rr_scale,
    #     reprob=reprob,
    #     auto_augment=use_aa,
    #     motion_shift=motion_shift,
    #     crop_size=crop_size,
    # )

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    (unsupervised_loader, unsupervised_sampler) = init_data(
        data_path=data_path,
        batch_size=batch_size,
        transform=transform,
        camera_views=camera_views,
        frames_per_clip=16,
        frameskip=2,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
        pin_mem=pin_mem,
        fps=fps,
    )

    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # encoder = DistributedDataParallel(encoder, static_graph=True)
    # dreamer = DistributedDataParallel(dreamer, static_graph=True)

    # (
    #     encoder,
    #     dreamer,
    #     _,
    #     _,
    #     start_epoch,
    # ) = load_checkpoint(
    #     r_path=load_path,
    #     dreamer_path=dreamer_path,
    #     encoder=encoder,
    #     dreamer=dreamer,
    #     opt=None,
    #     scaler=None,
    #     replace_kw = ["module.", "backbone."]
    # )

    (
        encoder,
        target_encoder,
        dreamer_predictor,
        _,
        _,
        _
    ) = load_checkpoint(
        r_path=load_path,
        dreamer_predictor_path=dreamer_predictor_path,
        encoder=encoder,
        target_encoder=None,
        dreamer_predictor=dreamer_predictor,
        opt=None,
        scaler=None,
        replace_kw = ["module.", "backbone."]
    )


    start_epoch = 0
    # -- load training checkpoint

    # checkpoint = robust_checkpoint_loader(load_path, map_location=torch.device("cpu"))

    # pretrained_dict = checkpoint["encoder"]
    # pretrained_dict = {k.replace("module.backbone.", ""): v for k, v in pretrained_dict.items()}
    # msg = encoder.load_state_dict(pretrained_dict)

    # dreamer_checkpoint = robust_checkpoint_loader(dreamer_path, map_location=torch.device("cpu"))
    # msg = dreamer.load_state_dict(pretrained_dict, strict=False)
    # pretrained_dreamer = dreamer_checkpoint["dreamer"]
    # pretrained_dreamer = {k.replace("module.", ""): v for k, v in pretrained_dreamer.items()}
    # msg = dreamer.load_state_dict(pretrained_dreamer, strict=False)
    
    # pretrained_dreamer = dreamer_checkpoint["dreamer"]
    # pretrained_dreamer = {k.replace("module.", ""): v for k, v in pretrained_dreamer.items()}
    # msg = dreamer.load_state_dict(pretrained_dreamer, strict=False)
    
    # del dreamer_checkpoint
    # del checkpoint
    # del pretrained_dict
    # del pretrained_dreamer

    # encoder.to(device)
    # dreamer.to(device)
    # encoder.to(dtype=dtype)
    # dreamer.to(dtype=dtype)
    gc.collect()
    torch.cuda.empty_cache()
    encoder.eval()
    dreamer_predictor.eval()
    
    

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)


    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):

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

            # def prepare_data():
            #     first_frame, target_frame, reference_frame = sample
            #     first_frame = first_frame.to(device, non_blocking=True)
            #     target_frame = target_frame.to(device, non_blocking=True)
            #     reference_frame = reference_frame.to(device, non_blocking=True)
            #     #reference_feature = encoder(reference_frame)
            #     return first_frame, target_frame, reference_frame

            current_frame, target_frame, current_reference, target_reference = sample
            current_frame = current_frame.to(device, dtype=dtype)
            target_frame = target_frame.to(device, dtype=dtype)
            current_reference = current_reference.to(device, dtype=dtype)
            target_reference = target_reference.to(device, dtype=dtype)

            def forward_feature(frame):
                with torch.no_grad():
                    feature = encoder(frame)
                    if normalize_reps:
                        feature = F.layer_norm(feature, (feature.size(-1),))
                    return feature

            def forward_dreamer(current_feature, current_reference_feature, target_reference_feature):
                    dreamer_output = dreamer_predictor(xt = current_feature, yt = current_reference_feature, yt_plus_1 = target_reference_feature)
                    return dreamer_output

            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=dtype):
                    target_feature = forward_feature(target_frame)
                    current_feature = forward_feature(current_frame)
                    current_reference_feature = forward_feature(current_reference)
                    target_reference_feature = forward_feature(target_reference)
                    dreamer_output = forward_dreamer(current_feature, current_reference_feature, target_reference_feature)
                    result = retrieval_eval(dreamer_output,target_feature)

            def wandb_log_stats():
                wandb.log({
                    "step": itr,
                    "acc@1": result["acc@1"],
                })
            wandb_log_stats()
    gc.collect()
    torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument("--debugmode", type=bool, default=False, help="Setting this to true will not spin up new processes. The main code runs the main process, which makes it easier to debug with checkpointing.")

if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    
    
    main(params)
