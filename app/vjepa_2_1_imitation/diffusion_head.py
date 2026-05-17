"""
Conditional diffusion on a trajectory tensor x (e.g. actions). Condition cond is supplied by the
caller — no observation encoder or task-specific parsing.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion


def _default_ddpm_scheduler(
    num_train_timesteps: int = 100,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "squaredcos_cap_v2",
    variance_type: str = "fixed_small",
    clip_sample: bool = True,
    prediction_type: str = "epsilon",
) -> DDPMScheduler:
    """Defaults aligned with ``train_diffusion_transformer_lowdim_workspace.yaml``."""
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        variance_type=variance_type,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )


class DiffusionHead(ModuleAttrMixin):
    """
    - Builds TransformerForDiffusion over sequences of shape (B, T, trajectory_dim).
    - Training: epsilon (or sample) prediction with optional cross-attention condition
      cond of shape (B, n_cond_steps, cond_dim).
    - Inference: DDPM denoising loop with the same cond.
    """

    def __init__(
        self,
        trajectory_dim: int = 7,
        cond_dim: int = 1408,
        horizon: int = 1,
        n_cond_steps: int = 256,
        noise_scheduler: Optional[DDPMScheduler] = None,
        num_inference_steps: Optional[int] = None,
        # used only when ``noise_scheduler`` is None (see ``_default_ddpm_scheduler``)
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        n_layer: int = 8,
        n_cond_layers: int = 0,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        **scheduler_step_kwargs,
    ):
        """
        Args:
            trajectory_dim: D_x, per-timestep dimension of the trajectory being diffused.
            cond_dim: D_c, last dim of cond (set 0 for unconditional; pass cond=None).
            horizon: T, length of the noisy trajectory sequence.
            n_cond_steps: number of condition tokens (maps to TransformerForDiffusion n_obs_steps).
            noise_scheduler: Optional ``DDPMScheduler``. If omitted, one is built from the
                ``num_train_timesteps`` / ``beta_*`` / ``prediction_type`` arguments below.
        """
        super().__init__()

        if noise_scheduler is None:
            noise_scheduler = _default_ddpm_scheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                variance_type=variance_type,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )

        self.model = TransformerForDiffusion(
            input_dim=trajectory_dim,
            output_dim=trajectory_dim,
            horizon=horizon,
            n_obs_steps=n_cond_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=(cond_dim > 0),
            n_cond_layers=n_cond_layers,
        )
        self.noise_scheduler = noise_scheduler
        self.trajectory_dim = trajectory_dim
        self.cond_dim = cond_dim
        self.horizon = horizon
        self.n_cond_steps = n_cond_steps

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.scheduler_step_kwargs = scheduler_step_kwargs

    def forward(
        self,
        cond: Optional[torch.Tensor],
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cond: shape (B, n_cond_steps, cond_dim), or None if cond_dim == 0.
            trajectory: clean x_0, shape (B, T, trajectory_dim).
        """
        if self.cond_dim > 0:
            assert cond is not None
            assert cond.shape[-1] == self.cond_dim
            assert cond.shape[1] == self.n_cond_steps
        else:
            assert cond is None

        noise = torch.randn_like(trajectory)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.model(noisy, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction_type {pred_type}")

        loss = F.mse_loss(pred, target)
        return loss

    @torch.no_grad()
    def denoise_inference(
        self,
        cond: Optional[torch.Tensor] = None,
        *,
        batch_size: int = 1,
        init_template: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        DDPM 采样：从标准高斯噪声迭代去噪得到轨迹。

        Args:
            cond: (B, n_cond_steps, cond_dim)；无条件时 ``cond_dim==0`` 则传 None。
            batch_size: 仅在无条件且未提供 ``init_template`` 时使用。
            init_template: 若给定，用其 ``shape`` / ``dtype`` / ``device`` 生成初始噪声；
                否则形状为 ``(batch_size, horizon, trajectory_dim)``，设备与 dtype 与模块参数一致。
        """
        if self.cond_dim > 0:
            assert cond is not None
            assert cond.shape[1] == self.n_cond_steps
            assert cond.shape[2] == self.cond_dim
            batch_size = cond.shape[0]
        else:
            assert cond is None

        if init_template is not None:
            x = torch.randn(
                init_template.shape,
                dtype=init_template.dtype,
                device=init_template.device,
                generator=generator,
            )
        else:
            x = torch.randn(
                batch_size,
                self.horizon,
                self.trajectory_dim,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            model_output = self.model(x, t, cond)
            x = scheduler.step(
                model_output,
                t,
                x,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample

        return x

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas),
        )
