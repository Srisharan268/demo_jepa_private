# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import app.vjepa_2_1_dreamer_ac.models.vision_transformer as video_vit_enc
import src.models.ac_predictor as vit_ac_pred
from src.models.dreamer_predictor import get_dreamer_predictor
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule, WarmupCosineSchedule


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

MAX_RETRIES = 3


def vjepa_2_1_encoder_args_from_cfg(cfgs_model):
    """V-JEPA 2.1 vision encoder kwargs from yaml config."""
    return dict(
        init_type=cfgs_model.get("init_type", "default"),
        img_temporal_dim_size=cfgs_model.get("img_temporal_dim_size", None),
        n_registers=cfgs_model.get("n_registers", 0),
        has_cls_first=cfgs_model.get("has_cls_first", False),
        interpolate_rope=cfgs_model.get("interpolate_rope", False),
        modality_embedding=cfgs_model.get("modality_embedding", False),
        n_output_distillation=cfgs_model.get("n_output_distillation", 4),
        is_causal=cfgs_model.get("is_causal_encoder", False),
    )


def _normalize_pretrained_keys(state_dict, replace_kw=("backbone.",)):
    """Strip backbone. (and other) prefixes from checkpoint keys."""
    out = state_dict
    for kw in replace_kw:
        out = {k.replace(kw, ""): v for k, v in out.items()}
    return out


def load_pretrained_encoder_predictor(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=True,
    load_encoder=True,
    replace_kw=("backbone.",),
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint.get("epoch", 0)

    if load_encoder and encoder is not None:
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = _normalize_pretrained_keys(pretrained_dict, replace_kw)
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_predictor and predictor is not None:
        pretrained_dict = checkpoint["predictor"]
        pretrained_dict = _normalize_pretrained_keys(pretrained_dict, replace_kw)
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    if load_encoder and target_encoder is not None:
        pretrained_dict = checkpoint[target_encoder_key]
        pretrained_dict = _normalize_pretrained_keys(pretrained_dict, replace_kw)
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
    )


def load_dreamer_predictor(
    dreamer_predictor_path,
    dreamer_predictor,
    replace_kw=("backbone.",),
):
    logger.info(f"Loading checkpoint from {dreamer_predictor_path}")
    dreamer_predictor_checkpoint = robust_checkpoint_loader(
        dreamer_predictor_path, map_location=torch.device("cpu")
    )

    pretrained_dreamer_predictor = dreamer_predictor_checkpoint["dreamer_predictor"]
    pretrained_dreamer_predictor = _normalize_pretrained_keys(
        pretrained_dreamer_predictor, replace_kw
    )
    msg = dreamer_predictor.load_state_dict(pretrained_dreamer_predictor)
    logger.info(f"loaded pretrained dreamer_predictor with msg: {msg}")
    del dreamer_predictor_checkpoint

    return dreamer_predictor


def init_encoder_predictor(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    action_embed_dim=7,
    use_extrinsics=False,
    old_pred=False,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
    n_output_distillation=4,
    is_causal=False,
    **kwargs,
):
    enc_kwargs = dict(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        n_output_distillation=n_output_distillation,
        is_causal=is_causal,
    )

    encoder = video_vit_enc.__dict__[model_name](**enc_kwargs)

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_dreamer_predictor(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    dreamer_predictor_fusion_type="conv3d",
):
    dreamer_predictor = get_dreamer_predictor(
        embed_dim=1408,
        num_heads=16,
        mlp_ratio=4.0,
        patch_h=16,
        patch_w=16,
        conv_kernel_size=3,
        up_dim=64,
        num_self_attn_blocks=16,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        fusion_type=dreamer_predictor_fusion_type,
    )

    dreamer_predictor.to(device)
    logger.info(dreamer_predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Dreamer predictor number of parameters: {count_parameters(dreamer_predictor)}")

    return dreamer_predictor


def init_opt(
    encoder,
    predictor,
    dreamer_predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
    dreamer_predictor_lr_scale=1.0,
    unfreeze_dreamer_predictor=False,
):
    if unfreeze_dreamer_predictor:
        param_groups = [
            {
                "params": (p for n, p in dreamer_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
                "lr_scale": dreamer_predictor_lr_scale,
            },
            {
                "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            },
            {
                "params": (p for n, p in dreamer_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
                "lr_scale": dreamer_predictor_lr_scale,
            },
            {
                "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
        ]
    else:
        param_groups = [
            {
                "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            },
            {
                "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler