# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch
import torch.nn as nn

import app.vjepa_2_1_dreamer_predictor.models.vision_transformer as video_vit
from src.models.dreamer_predictor import get_dreamer_predictor
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

MAX_RETRIES = 3


def vjepa_2_1_encoder_args_from_cfg(cfgs_model):
    """Same as app.vjepa_2_1_ac.utils — V-JEPA 2.1 vision encoder kwargs from yaml."""
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
    """Match vjepa_2_1_ac load_pretrained / load_checkpoint: strip backbone. prefixes."""
    out = state_dict
    for kw in replace_kw:
        out = {k.replace(kw, ""): v for k, v in out.items()}
    return out


def load_pretrained(
    r_path,
    encoder=None,
    dreamer_predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_dreamer_predictor=False,
    load_encoder=True,
):
    """Same contract as app.vjepa_2_1_ac.utils.load_pretrained; dreamer head key is dreamer_predictor."""
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint.get("epoch", 0)

    if load_encoder and encoder is not None:
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = _normalize_pretrained_keys(pretrained_dict)
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_dreamer_predictor and dreamer_predictor is not None:
        pretrained_dict = checkpoint["dreamer_predictor"]
        pretrained_dict = _normalize_pretrained_keys(pretrained_dict)
        msg = dreamer_predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained dreamer_predictor from epoch {epoch} with msg: {msg}")

    if load_encoder and target_encoder is not None:
        pretrained_dict = checkpoint[target_encoder_key]
        pretrained_dict = _normalize_pretrained_keys(pretrained_dict)
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target_encoder from epoch {epoch} with msg: {msg}")

    del checkpoint

    return encoder, dreamer_predictor, target_encoder


def load_checkpoint(
    r_path,
    dreamer_predictor_path,
    encoder,
    target_encoder,
    dreamer_predictor,
    opt,
    scaler,
    replace_kw=["backbone."],
    encoder_key="target_encoder",
):
    """Dreamer checkpoint layout matches app.dreamer_predictor; encoder load follows vjepa_2_1_ac (strict=False)."""
    logger.info(f"Loading checkpoint from {r_path}")
    encoder_checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    encoder_dict = encoder_checkpoint.get(encoder_key) or encoder_checkpoint.get("encoder")
    if encoder_dict is None:
        raise KeyError(
            f"Checkpoint must have '{encoder_key}' or 'encoder' key. Found: {list(encoder_checkpoint.keys())}"
        )

    encoder_dict = _normalize_pretrained_keys(encoder_dict, replace_kw=tuple(replace_kw))
    msg = encoder.load_state_dict(encoder_dict, strict=False)
    logger.info(f"loaded pretrained encoder with msg: {msg}")
    if target_encoder is not None:
        msg = target_encoder.load_state_dict(encoder_dict, strict=False)
        logger.info(f"loaded pretrained target encoder with msg: {msg}")
    del encoder_checkpoint

    if dreamer_predictor_path is not None:
        dreamer_predictor_checkpoint = robust_checkpoint_loader(
            dreamer_predictor_path, map_location=torch.device("cpu")
        )
        dreamer_predictor_dict = dreamer_predictor_checkpoint["dreamer_predictor"]
        dreamer_predictor_dict = _normalize_pretrained_keys(dreamer_predictor_dict, replace_kw=tuple(replace_kw))

        msg = dreamer_predictor.load_state_dict(dreamer_predictor_dict)
        logger.info(f"loaded pretrained dreamer predictor with msg: {msg}")
        del dreamer_predictor_checkpoint

    return (
        encoder,
        target_encoder,
        dreamer_predictor,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
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
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
    n_output_distillation=4,
    is_causal_encoder=False,
    model_cfg=None,
    fusion_type="conv3d",
    **kwargs,
):
    if model_cfg is not None:
        ecfg = vjepa_2_1_encoder_args_from_cfg(model_cfg)
        init_type = ecfg["init_type"]
        img_temporal_dim_size = ecfg["img_temporal_dim_size"]
        n_registers = ecfg["n_registers"]
        has_cls_first = ecfg["has_cls_first"]
        interpolate_rope = ecfg["interpolate_rope"]
        modality_embedding = ecfg["modality_embedding"]
        n_output_distillation = ecfg["n_output_distillation"]
        is_causal_encoder = ecfg["is_causal"]

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
        is_causal=is_causal_encoder,
    )
    encoder = video_vit.__dict__[model_name](**enc_kwargs)

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
        fusion_type=fusion_type,
    )

    encoder.to(device)
    dreamer_predictor.to(device)
    logger.info(encoder)
    logger.info(dreamer_predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Dreamer number of parameters: {count_parameters(dreamer_predictor)}")

    return encoder, dreamer_predictor


def init_opt(
    encoder,
    dreamer_predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    unfreeze_vit=False,
):
    if unfreeze_vit:
        param_groups = [
            {
                "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            },
            {
                "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
            {
                "params": (
                    p for n, p in dreamer_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
                ),
            },
            {
                "params": (p for n, p in dreamer_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
        ]
    else:
        param_groups = [
            {
                "params": (
                    p for n, p in dreamer_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
                ),
            },
            {
                "params": (p for n, p in dreamer_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
