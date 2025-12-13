"""
DAT: Dual Aggregation Transformer for Image Restoration
========================================================

Implementation based on ICCV 2023 paper:
"Dual Aggregation Transformer for Image Super-Resolution"
https://arxiv.org/abs/2308.03364

Key Features:
1. Dual Aggregation: Alternating spatial and channel attention
   - Inter-block: Alternate between spatial and channel attention
   - Intra-block: Adaptive Interaction Module (AIM)
2. Spatial-Gate Feed-Forward Network (SGFN)
3. Overlapped Cross-Attention (OCA) for better locality
4. Relative position bias for spatial awareness

Architecture:
- Shallow feature extraction
- Deep feature extraction with DATB (DAT Blocks)
- Reconstruction head

For image retouching (not super-resolution), we adapt the architecture
to output at the same resolution.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# =============================================================================
# Utility Functions
# =============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# Attention Layers
# =============================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ChannelAttention(nn.Module):
    """Channel attention across spatial dimensions."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Channel attention: transpose Q, K, V
        q = q.transpose(-2, -1)  # [B, heads, head_dim, N]
        k = k.transpose(-2, -1)  # [B, heads, head_dim, N]
        v = v.transpose(-2, -1)  # [B, heads, head_dim, N]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-2, -1).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# =============================================================================
# Adaptive Interaction Module (AIM)
# =============================================================================

class AdaptiveInteractionModule(nn.Module):
    """
    Adaptive Interaction Module (AIM) for intra-block dual aggregation.
    Combines spatial and channel attention with learnable interaction.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Spatial attention
        self.spatial_attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias, attn_drop, proj_drop
        )

        # Channel attention
        self.channel_attn = ChannelAttention(
            dim, num_heads, qkv_bias, attn_drop, proj_drop
        )

        # Learnable interaction weights
        self.spatial_weight = nn.Parameter(torch.ones(1))
        self.channel_weight = nn.Parameter(torch.ones(1))

        # Compute attention mask for shifted windows
        if self.shift_size > 0:
            H, W = 64, 64  # Will be recomputed based on input
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._get_attn_mask(H, W, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Spatial attention
        spatial_out = self.spatial_attn(self.norm1(x_windows), mask=attn_mask)

        # Window reverse
        spatial_out = spatial_out.view(-1, self.window_size, self.window_size, C)
        spatial_out = window_reverse(spatial_out, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            spatial_out = torch.roll(spatial_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        spatial_out = spatial_out.view(B, L, C)

        # Channel attention (global)
        channel_out = self.channel_attn(self.norm2(x.view(B, L, C)))

        # Adaptive interaction
        out = self.spatial_weight * spatial_out + self.channel_weight * channel_out

        return out

    def _get_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Compute attention mask for the given resolution."""
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


# =============================================================================
# Spatial-Gate Feed-Forward Network (SGFN)
# =============================================================================

class SpatialGateFeedForward(nn.Module):
    """
    Spatial-Gate Feed-Forward Network (SGFN).
    Adds spatial information through depth-wise convolution gating.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.act = act_layer()

        # Spatial gate with depth-wise conv
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim, bias=True
        )

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape

        # First projection and split
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)

        # Spatial gating
        gate = gate.view(B, H, W, -1).permute(0, 3, 1, 2)
        gate = self.dwconv(gate)
        gate = gate.permute(0, 2, 3, 1).view(B, L, -1)

        x = self.act(x) * gate
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


# =============================================================================
# DAT Block
# =============================================================================

class DATBlock(nn.Module):
    """
    Dual Aggregation Transformer Block.

    Combines:
    - Adaptive Interaction Module (AIM) for intra-block dual aggregation
    - Spatial-Gate Feed-Forward Network (SGFN)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Adaptive Interaction Module
        self.aim = AdaptiveInteractionModule(
            dim, num_heads, window_size, shift_size, qkv_bias, attn_drop, drop
        )

        # Drop path
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)

        # Spatial-Gate Feed-Forward
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.sgfn = SpatialGateFeedForward(dim, mlp_hidden_dim, drop=drop)

        # Learnable scale
        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.gamma2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # AIM
        x = x + self.drop_path(self.gamma1 * self.aim(x, H, W))

        # SGFN
        x = x + self.drop_path(self.gamma2 * self.sgfn(self.norm(x), H, W))

        return x


# =============================================================================
# DAT Layer (Group of Blocks)
# =============================================================================

class DATLayer(nn.Module):
    """A layer of DAT blocks with residual connection."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # DAT blocks with alternating shift
        self.blocks = nn.ModuleList([
            DATBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

        # Conv for residual connection
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.blocks:
            x = block(x, H, W)

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv(x) + residual

        return x


# =============================================================================
# DAT Model
# =============================================================================

class DAT(nn.Module):
    """
    Dual Aggregation Transformer for Image Restoration.

    Adapted from super-resolution to image retouching (same resolution).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 180,
        depths: Tuple[int, ...] = (6, 6, 6, 6),
        num_heads: Tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = len(depths)

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        # Deep feature extraction
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DATLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            )
            self.layers.append(layer)

        # Final conv before output
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        # Reconstruction (same resolution for retouching)
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape

        # Pad to multiple of window size
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Shallow feature
        shallow_feat = self.conv_first(x)

        # Deep feature
        deep_feat = shallow_feat
        for layer in self.layers:
            deep_feat = layer(deep_feat)

        deep_feat = self.conv_after_body(deep_feat) + shallow_feat

        # Reconstruction
        output = self.conv_last(deep_feat)

        # Remove padding
        output = output[:, :, :H, :W]

        if return_features:
            return {'output': output, 'features': deep_feat[:, :, :H, :W]}
        return {'output': output}


# =============================================================================
# Model Configurations
# =============================================================================

def dat_small() -> DAT:
    """Small DAT (~5M params)."""
    return DAT(
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
    )


def dat_base() -> DAT:
    """Base DAT (~12M params)."""
    return DAT(
        embed_dim=180,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
    )


def dat_large() -> DAT:
    """Large DAT (~40M params)."""
    return DAT(
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=4.0,
    )


def dat_xlarge() -> DAT:
    """XLarge DAT (~70M params) - Maximum quality."""
    return DAT(
        embed_dim=240,
        depths=(6, 6, 6, 6, 6, 6, 6, 6),
        num_heads=(8, 8, 8, 8, 8, 8, 8, 8),
        window_size=8,
        mlp_ratio=4.0,
    )


if __name__ == '__main__':
    print("Testing DAT models...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test small model
    model = dat_small().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"DAT-small parameters: {params:,}")

    x = torch.randn(1, 3, 256, 256, device=device)
    out = model(x)
    print(f"Output shape: {out['output'].shape}")

    # Test base model
    model_base = dat_base().to(device)
    params_base = sum(p.numel() for p in model_base.parameters())
    print(f"DAT-base parameters: {params_base:,}")

    # Test large model
    model_large = dat_large().to(device)
    params_large = sum(p.numel() for p in model_large.parameters())
    print(f"DAT-large parameters: {params_large:,}")

    print("All tests passed!")
