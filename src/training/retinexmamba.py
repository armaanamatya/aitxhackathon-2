"""
RetinexMamba: State Space Model for HDR/Low-Light Image Enhancement

Based on Retinex theory (image = illumination × reflectance) combined with
Mamba (Selective State Space Models) for efficient sequence modeling.

Key innovations:
1. Retinex-based decomposition for illumination-aware processing
2. Mamba blocks for linear-complexity global context modeling
3. Illumination-guided attention for adaptive enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange

# Try to import mamba-ssm, fall back to pure PyTorch implementation
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Using native Mamba SSM")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Mamba SSM not available, using PyTorch implementation")


class PurePyTorchMamba(nn.Module):
    """
    Pure PyTorch implementation of Mamba block.
    Slower than native CUDA but fully compatible.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize A (state matrix) - use negative values for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # Causal conv
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)

        # SSM
        y = self.ssm(x)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Memory-efficient SSM using chunked processing."""
        B, L, D = x.shape

        # Get SSM parameters
        x_dbl = self.x_proj(x)  # (B, L, d_state*2 + 1)
        delta, B_param, C_param = x_dbl.split([1, self.d_state, self.d_state], dim=-1)

        # Discretize
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Memory-efficient scan with chunking
        chunk_size = min(64, L)  # Process in chunks to save memory
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            chunk_ys = []

            for i in range(chunk_start, chunk_end):
                # Compute discretized A for this step
                deltaA_i = torch.exp(delta[:, i:i+1, :].unsqueeze(-1) * A)  # (B, 1, d_inner, d_state)
                deltaA_i = deltaA_i.squeeze(1)  # (B, d_inner, d_state)

                # Compute input contribution
                deltaB_x_i = delta[:, i, :].unsqueeze(-1) * B_param[:, i, :].unsqueeze(1) * x[:, i, :].unsqueeze(-1)

                # Update state
                h = deltaA_i * h + deltaB_x_i

                # Compute output
                y_i = (h * C_param[:, i, :].unsqueeze(1)).sum(-1)  # (B, d_inner)
                chunk_ys.append(y_i)

            ys.extend(chunk_ys)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Add D (skip connection)
        y = y + x * self.D

        return y


def get_mamba_block(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
    """Get Mamba block - native if available, else pure PyTorch."""
    if MAMBA_AVAILABLE:
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    else:
        return PurePyTorchMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class SpatialMambaBlock(nn.Module):
    """
    Mamba block for spatial feature processing.
    Processes image features as sequences for global context.
    """
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.mamba_h = get_mamba_block(dim, d_state=d_state)  # Horizontal scan
        self.mamba_v = get_mamba_block(dim, d_state=d_state)  # Vertical scan
        self.proj = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)

        # Horizontal scan
        x_h = rearrange(x, 'b c h w -> (b h) w c')  # (B*H, W, C)
        x_h = self.mamba_h(x_h)
        x_h = rearrange(x_h, '(b h) w c -> b c h w', b=B, h=H)

        # Vertical scan
        x_v = rearrange(x, 'b c h w -> (b w) h c')  # (B*W, H, C)
        x_v = self.mamba_v(x_v)
        x_v = rearrange(x_v, '(b w) h c -> b c h w', b=B, w=W)

        # Combine
        x = torch.cat([x_h, x_v], dim=1)
        x = self.proj(x)

        return x + residual


class RetinexDecomposition(nn.Module):
    """
    Retinex-based decomposition module.
    Decomposes input into illumination and reflectance components.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 32):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # Illumination estimation (single channel)
        self.illum_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Sigmoid()  # Illumination in [0, 1]
        )

        # Reflectance estimation (3 channels)
        self.reflect_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1),
            nn.Sigmoid()  # Reflectance in [0, 1]
        )

    def forward(self, x):
        """
        x: (B, 3, H, W) in range [-1, 1]
        Returns: illumination (B, 1, H, W), reflectance (B, 3, H, W)
        """
        # Convert to [0, 1] for Retinex
        x_01 = (x + 1) / 2

        feat = self.encoder(x_01)
        illum = self.illum_head(feat) + 0.01  # Avoid division by zero
        reflect = self.reflect_head(feat)

        return illum, reflect


class IlluminationGuidedAttention(nn.Module):
    """
    Attention module guided by illumination map.
    Focuses enhancement on poorly-lit regions.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim + 1, dim, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, illum):
        """
        x: (B, C, H, W) features
        illum: (B, 1, H, W) illumination map
        """
        # Concatenate illumination info
        x_illum = torch.cat([x, illum], dim=1)
        x_illum = self.conv(x_illum)

        # Generate attention based on illumination
        # Lower illumination = higher attention (enhance dark regions more)
        gate = self.gate(x_illum)

        return x * gate + x


class RetinexMambaBlock(nn.Module):
    """
    Core RetinexMamba block combining Retinex guidance with Mamba.
    """
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.mamba = SpatialMambaBlock(dim, d_state)
        self.norm2 = LayerNorm2d(dim)
        self.illum_attn = IlluminationGuidedAttention(dim)
        self.channel_attn = ChannelAttention(dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x, illum):
        # Mamba for global context
        x = self.mamba(self.norm1(x)) + x

        # Illumination-guided attention
        x = self.illum_attn(x, illum)

        # Channel attention
        x = self.channel_attn(x)

        # FFN
        x = self.ffn(self.norm2(x)) + x

        return x


class RetinexMamba(nn.Module):
    """
    RetinexMamba: Full model for HDR/Low-Light Image Enhancement.

    Architecture:
    1. Retinex decomposition to get illumination and reflectance
    2. Multi-scale feature extraction
    3. RetinexMamba blocks with illumination guidance
    4. Progressive reconstruction

    Args:
        in_channels: Input image channels (3 for RGB)
        out_channels: Output image channels (3 for RGB)
        dim: Base feature dimension
        num_blocks: Number of RetinexMamba blocks per stage
        d_state: State dimension for Mamba
        stages: Number of encoder-decoder stages
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list = [2, 4, 4, 6],
        d_state: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Retinex decomposition
        self.retinex = RetinexDecomposition(in_channels, hidden_dim=dim)

        # Initial feature extraction
        self.input_proj = nn.Conv2d(in_channels, dim, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        current_dim = dim
        for i, num_block in enumerate(num_blocks[:-1]):
            # Blocks at this scale
            blocks = nn.ModuleList([
                RetinexMambaBlock(current_dim, d_state) for _ in range(num_block)
            ])
            self.encoder_blocks.append(blocks)

            # Downsample
            self.downsample.append(
                nn.Conv2d(current_dim, current_dim * 2, 4, stride=2, padding=1)
            )
            current_dim *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            RetinexMambaBlock(current_dim, d_state) for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i, num_block in enumerate(reversed(num_blocks[:-1])):
            # Upsample
            self.upsample.append(
                nn.ConvTranspose2d(current_dim, current_dim // 2, 4, stride=2, padding=1)
            )
            current_dim //= 2

            # Skip connection fusion
            # Blocks at this scale
            blocks = nn.ModuleList([
                RetinexMambaBlock(current_dim, d_state) for _ in range(num_block)
            ])
            self.decoder_blocks.append(blocks)

        # Output projection (residual learning)
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 3, padding=1),
            nn.Tanh()
        )

        # Illumination downsampling for multi-scale
        self.illum_down = nn.ModuleList([
            nn.AvgPool2d(2**i, 2**i) for i in range(len(num_blocks))
        ])

    def forward(self, x):
        """
        x: (B, 3, H, W) input image in [-1, 1]
        Returns: (B, 3, H, W) enhanced image in [-1, 1]
        """
        # Retinex decomposition for illumination guidance
        illum, reflect = self.retinex(x)

        # Initial projection
        feat = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for i, (blocks, down) in enumerate(zip(self.encoder_blocks, self.downsample)):
            illum_scale = self.illum_down[i](illum)
            for block in blocks:
                feat = block(feat, illum_scale)
            skips.append(feat)
            feat = down(feat)

        # Bottleneck
        illum_bottom = self.illum_down[-1](illum)
        for block in self.bottleneck:
            feat = block(feat, illum_bottom)

        # Decoder with skip connections
        for i, (up, blocks) in enumerate(zip(self.upsample, self.decoder_blocks)):
            feat = up(feat)
            feat = feat + skips[-(i+1)]  # Skip connection
            illum_scale = self.illum_down[-(i+2)](illum)
            for block in blocks:
                feat = block(feat, illum_scale)

        # Output with residual learning
        out = self.output_proj(feat)
        out = out + x  # Residual connection
        out = torch.clamp(out, -1, 1)

        return out

    def get_illumination(self, x):
        """Get illumination map for visualization."""
        illum, _ = self.retinex(x)
        return illum


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Model variants
def retinexmamba_small(in_channels=3, out_channels=3):
    """Small variant ~5M params - good for limited data."""
    return RetinexMamba(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=32,
        num_blocks=[2, 2, 4, 4],
        d_state=8,
    )


def retinexmamba_base(in_channels=3, out_channels=3):
    """Base variant ~12M params - balanced."""
    return RetinexMamba(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=48,
        num_blocks=[2, 4, 4, 6],
        d_state=16,
    )


def retinexmamba_large(in_channels=3, out_channels=3):
    """Large variant ~25M params - maximum quality."""
    return RetinexMamba(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=64,
        num_blocks=[2, 4, 6, 8],
        d_state=16,
    )


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, model_fn in [
        ("small", retinexmamba_small),
        ("base", retinexmamba_base),
        ("large", retinexmamba_large)
    ]:
        model = model_fn().to(device)
        x = torch.randn(1, 3, 256, 256).to(device)
        y = model(x)
        print(f"RetinexMamba-{name}: {count_parameters(model):,} params, "
              f"input {x.shape} -> output {y.shape}")
