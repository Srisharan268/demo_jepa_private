import torch
import torch.nn as nn

from src.models.utils.modules import Expert
from src.models.utils.modules import CrossAttentionBlock, Block
from src.utils.tensors import trunc_normal_

def _spatial_to_conv3d(x):
    """
    Convert fusion block input from spatial layout to PyTorch Conv3d layout.
    Input:  [B, H, W, D, 3]  (B, patch_h, patch_w, embed_dim, fusion_dim)
    Output: (B, C, D, H, W)  with C=3 (fusion target), D=embed_dim (kept)
    """
    # B H W D 3 -> B 3 D H W
    return x.permute(0, 4, 3, 1, 2)  # (B, C=3, D, H, W)


def _conv3d_to_spatial(x):
    """
    Convert from PyTorch Conv3d layout back to spatial layout.
    Input:  (B, C, D, H, W)  e.g. (B, 1, D, H, W) after fusion
    Output: [B, H, W, D]    (B, patch_h, patch_w, embed_dim)
    """
    # (B, C, D, H, W) -> squeeze C if C=1 -> (B, D, H, W) -> (B, H, W, D)
    if x.shape[1] == 1:
        x = x.squeeze(1)  # (B, D, H, W)
    return x.permute(0, 2, 3, 1)  # (B, H, W, D)


class Conv3dFusionBlock(nn.Module):
    """
    Simple 3D conv block on up_dim channels.
    Layout: (B, C, D, H, W); kernel is (1, k, k) so D is not mixed.
    """

    def __init__(self, channels: int, kernel_size: int = 3, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size // 2, kernel_size // 2),
            bias=True,
        )
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: (B, C, D, H, W) -> (B, C, D, H, W)
        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        if self.use_residual:
            out = out + x
        return out


class Conv3dFusionNetwork(nn.Module):
    """
    Fuse the dim-3 in [B, H, W, D, 3] via Conv3d: scale up (3 -> up_dim) then scale down (up_dim -> 1).
    Uses PyTorch Conv3d layout (B, C, D, H, W) where C is the fusion target, D is kept (kernel 1 along D).
    """
    def __init__(
        self,
        embed_dim,
        up_dim=64,
        kernel_size=3,
        num_layers=4,
        use_residual=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.up_dim = up_dim
        self.num_layers = num_layers
        
        # Scale up: 3 -> up_dim
        self.scale_up = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=up_dim,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2),
                bias=True,
            ),
            nn.GroupNorm(num_groups=min(32, up_dim), num_channels=up_dim),
            nn.GELU(),
        )

        # Mid fusion blocks on up_dim channels to better capture channel info
        self.mid_blocks = nn.ModuleList(
            [
                Conv3dFusionBlock(
                    channels=up_dim,
                    kernel_size=kernel_size,
                    use_residual=use_residual,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Scale down: up_dim -> 1
        self.scale_down = nn.Sequential(
            nn.Conv3d(
                in_channels=up_dim,
                out_channels=1,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2),
                bias=True,
            ),
        )

    def forward(self, x):
        """
        x: [B, H, W, D, 3] -> (B, 3, D, H, W) -> (B, up_dim, D, H, W) -> (B, 1, D, H, W) -> [B, H, W, D]
        """
        x = _spatial_to_conv3d(x)  # (B, 3, D, H, W)
        x = self.scale_up(x)        # (B, up_dim, D, H, W)
        for block in self.mid_blocks:
            x = block(x)            # (B, up_dim, D, H, W)
        x = self.scale_down(x)      # (B, 1, D, H, W)
        x = _conv3d_to_spatial(x)   # [B, H, W, D]
        return x

class DreamerPredictor(nn.Module):
    """
    Dreamer Predictor: cross attention + Conv3d fusion + self-attention.
    Process:
        1. Cross attention: x_embodiment = cross(yt, xt), y_motion = cross(yt, yt+1)
        2. Stack -> B H W D 3 -> Conv3d fusion (scale up 3->up_dim, scale down up_dim->1) -> B N D
        3. Self-attention blocks (configurable depth) -> B N D
    """
    def __init__(
        self,
        embed_dim=1408,
        num_heads=12,
        mlp_ratio=4.0,
        patch_h=16,
        patch_w=16,
        conv_kernel_size=3,
        up_dim=64,
        num_self_attn_blocks=4,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        fusion_type="conv3d", # "conv3d" or "mean pooling"
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.fusion_type = fusion_type

        # Cross attention blocks
        self.embodiment_cross_attn = CrossAttentionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer,
        )
        self.motion_cross_attn = CrossAttentionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer,
        )
        
        # Conv3d fusion: scale up (3 -> up_dim) then scale down (up_dim -> 1)
        self.fusion_conv = Conv3dFusionNetwork(
            embed_dim=embed_dim,
            up_dim=up_dim,
            kernel_size=conv_kernel_size,
        ) if fusion_type == "conv3d" else None
        
        # Self-attention blocks for scaling
        self.self_attn_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                use_sdpa=True,
                is_causal=False,
            )
            for _ in range(num_self_attn_blocks)
        ])
        
        self.output_norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)

    def forward(self, xt, yt, yt_plus_1, patch_h=None, patch_w=None):
        """
        Args:
            xt: [B, N, D]
            yt: [B, N, D]
            yt_plus_1: [B, N, D]
            patch_h, patch_w: optional overrides
        Returns:
            [B, N, D]
        """
        B, N, D = xt.shape
        H = patch_h if patch_h is not None else self.patch_h
        W = patch_w if patch_w is not None else self.patch_w
        # assert N == H * W, f"N ({N}) must equal H*W ({H}*{W}={H*W})"
        token_flag = N != H * W # if token_flag is True, it means the input is a batch of temporal token seqs [B, T*N, D]
        if token_flag:
            import einops
            xt = einops.rearrange(xt, 'b (t n) d -> (b t) n d', n = H * W)
            yt = einops.rearrange(yt, 'b (t n) d -> (b t) n d', n = H * W)
            yt_plus_1 = einops.rearrange(yt_plus_1, 'b (t n) d -> (b t) n d', n = H * W)
            T = xt.shape[0] // B
            B, N, D = xt.shape

        # Cross attention operations
        x_embodiment = self.embodiment_cross_attn(yt, xt)  # [B, N, D]
        y_motion = self.motion_cross_attn(yt, yt_plus_1)  # [B, N, D]
        
        # Stack and fuse via Conv3d
        stacked = torch.stack([xt, x_embodiment, y_motion], dim=-1)  # [B, N, D, 3]
        stacked = stacked.view(B, H, W, D, 3)  # [B, H, W, D, 3]
        fused = self.fusion_conv(stacked) if self.fusion_type == "conv3d" else stacked.mean(dim=-1) # [B, H, W, D]
        x = fused.reshape(B, N, D)  # [B, N, D]
        
        # Self-attention blocks
        for block in self.self_attn_blocks:
            x = block(x)
        
        x = self.output_norm(x)
        
        if token_flag:
            x = einops.rearrange(x, '(b t) n d -> b (t n) d', t = T)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_dreamer_predictor(
    embed_dim=1408,
    num_heads=16,
    mlp_ratio=4.0,
    patch_h=16,
    patch_w=16,
    conv_kernel_size=3,
    up_dim=64,
    num_self_attn_blocks=4,
    norm_layer=nn.LayerNorm,
    init_std=0.02,
    fusion_type="conv3d", # "conv3d" or "mean pooling"
):
    return DreamerPredictor(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_h=patch_h,
        patch_w=patch_w,
        conv_kernel_size=conv_kernel_size,
        up_dim=up_dim,
        num_self_attn_blocks=num_self_attn_blocks,
        norm_layer=norm_layer,
        init_std=init_std,
        fusion_type=fusion_type,
    )


if __name__ == "__main__":
    patch_h, patch_w = 16, 16
    N = patch_h * patch_w
    embed_dim = 768

    print("=" * 50)
    print("Testing DreamerPredictor (Cross-attn + Conv3d fusion + Self-attn)")
    print("=" * 50)

    # Test with different self-attention depths
    configs = [
        {"num_self_attn_blocks": 2, "name": "2 self-attn blocks"},
        {"num_self_attn_blocks": 4, "name": "4 self-attn blocks"},
        {"num_self_attn_blocks": 6, "name": "6 self-attn blocks"},
    ]

    for config in configs:
        print(f"\n--- Config: {config['name']} ---")
        model = DreamerPredictor(
            embed_dim=embed_dim,
            patch_h=patch_h,
            patch_w=patch_w,
            up_dim=64,
            num_self_attn_blocks=config["num_self_attn_blocks"],
        ).to("cuda")

        xt = torch.randn(2, N, embed_dim).to("cuda")
        yt = torch.randn(2, N, embed_dim).to("cuda")
        yt_plus_1 = torch.randn(2, N, embed_dim).to("cuda")

        out = model(xt, yt, yt_plus_1)
        print(f"Input shape: {xt.shape}")
        print(f"Output shape: {out.shape}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

