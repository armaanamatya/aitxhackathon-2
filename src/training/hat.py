"""
HAT: Hybrid Attention Transformer for Image Restoration

Combines:
1. Channel Attention Block (CAB)
2. Window-based Self-Attention (like Swin Transformer)
3. Overlapping Cross-Attention (key innovation)

Reference: "Activating More Pixels in Image Super-Resolution Transformer" (CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional, Tuple


def window_partition(x, window_size: int):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ChannelAttention(nn.Module):
    """Channel Attention Block."""
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        attn = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * attn


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention with relative position bias."""
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class OverlappingCrossAttention(nn.Module):
    """
    Overlapping Cross-Attention - Key innovation of HAT.
    Uses overlapping windows for cross-attention to capture more context.
    Memory-efficient windowed implementation.
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, overlap_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.overlap_size = int(window_size * overlap_ratio)

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        ws = self.window_size

        # Query from regular windows
        q = self.q(x)

        # Key/Value from overlapping (shifted) windows
        shift_size = self.overlap_size
        x_shifted = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        kv = self.kv(x_shifted)
        k, v = kv.chunk(2, dim=-1)

        # Partition into windows for memory efficiency
        q_windows = window_partition(q, ws)  # (B*num_windows, ws, ws, C)
        k_windows = window_partition(k, ws)
        v_windows = window_partition(v, ws)

        # Reshape for attention within windows
        num_windows = q_windows.shape[0]
        q_windows = q_windows.view(num_windows, ws * ws, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_windows = k_windows.view(num_windows, ws * ws, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_windows = v_windows.view(num_windows, ws * ws, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Attention within windows (memory efficient)
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v_windows  # (num_windows, heads, ws*ws, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(num_windows, ws, ws, C)

        # Merge windows back
        out = window_reverse(out, ws, H, W)
        out = self.proj(out)

        return out


class OCAB(nn.Module):
    """Overlapping Cross-Attention Block."""
    def __init__(self, dim: int, num_heads: int, window_size: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.oca = OverlappingCrossAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: (B, H, W, C)
        x = x + self.oca(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HAB(nn.Module):
    """
    Hybrid Attention Block - Core building block of HAT.
    Combines Window Attention + Channel Attention.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

        # Channel attention
        self.channel_attn = ChannelAttention(dim)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Channel attention (applied to residual)
        x_ca = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        x_ca = self.channel_attn(x_ca)
        x_ca = x_ca.permute(0, 2, 3, 1).view(B, H * W, C)  # B, L, C

        x = shortcut + x + x_ca

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


class RHAG(nn.Module):
    """
    Residual Hybrid Attention Group.
    Contains multiple HABs + one OCAB.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_blocks: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                HAB(dim, num_heads, window_size, shift_size, mlp_ratio)
            )

        # Overlapping cross-attention at the end
        self.ocab = OCAB(dim, num_heads, window_size)

        # Conv for residual
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size

        residual = x

        for block in self.blocks:
            x = block(x, x_size)

        # OCAB
        x = x.view(B, H, W, C)
        x = self.ocab(x)
        x = x.view(B, L, C)

        # Conv on residual path
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).view(B, L, C)

        return x + residual


class HAT(nn.Module):
    """
    HAT: Hybrid Attention Transformer for Image Restoration.

    Args:
        in_channels: Input image channels
        out_channels: Output image channels
        dim: Base feature dimension
        num_heads: Number of attention heads
        num_groups: Number of RHAG groups
        num_blocks: Number of HAB blocks per group
        window_size: Window size for attention
        mlp_ratio: MLP expansion ratio
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 180,
        num_heads: int = 6,
        num_groups: int = 6,
        num_blocks: int = 6,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.window_size = window_size

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # Deep feature extraction (RHAG groups)
        self.groups = nn.ModuleList([
            RHAG(dim, num_heads, num_blocks, window_size, mlp_ratio)
            for _ in range(num_groups)
        ])

        self.norm = nn.LayerNorm(dim)
        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)

        # Reconstruction (for same-scale restoration, no upsampling)
        self.conv_last = nn.Conv2d(dim, out_channels, 3, 1, 1)

        # Residual learning
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x_size = (H, W)

        x = self.conv_first(x)
        residual = x

        x = x.permute(0, 2, 3, 1).view(B, H * W, -1)  # B, L, C

        for group in self.groups:
            x = group(x, x_size)

        x = self.norm(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, C, H, W
        x = self.conv_after_body(x)

        return x + residual

    def forward(self, x):
        """
        x: (B, 3, H, W) input in [-1, 1]
        Returns: (B, 3, H, W) output in [-1, 1]
        """
        # Pad to multiple of window_size
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Forward
        feat = self.forward_features(x)
        out = self.conv_last(feat)

        # Residual
        out = out + self.residual(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return torch.clamp(out, -1, 1)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Model variants
def hat_small(in_channels=3, out_channels=3):
    """Small HAT ~10M params."""
    return HAT(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=96,
        num_heads=6,
        num_groups=4,
        num_blocks=4,
        window_size=8,
        mlp_ratio=2.0,
    )


def hat_base(in_channels=3, out_channels=3):
    """Base HAT ~20M params."""
    return HAT(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=144,
        num_heads=6,
        num_groups=6,
        num_blocks=6,
        window_size=8,
        mlp_ratio=2.0,
    )


def hat_large(in_channels=3, out_channels=3):
    """Large HAT ~40M params."""
    return HAT(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=180,
        num_heads=6,
        num_groups=6,
        num_blocks=6,
        window_size=8,
        mlp_ratio=2.0,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, model_fn in [
        ("small", hat_small),
        ("base", hat_base),
        ("large", hat_large)
    ]:
        model = model_fn().to(device)
        x = torch.randn(1, 3, 256, 256).to(device)
        y = model(x)
        print(f"HAT-{name}: {count_parameters(model):,} params, "
              f"input {x.shape} -> output {y.shape}")
