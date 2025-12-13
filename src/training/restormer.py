"""
Restormer - Efficient Transformer for High-Resolution Image Restoration

Paper: "Restormer: Efficient Transformer for High-Resolution Image Restoration" (CVPR 2022)

Key features:
- Multi-Dconv Head Transposed Attention (MDTA)
- Gated-Dconv Feed-Forward Network (GDFN)
- Progressive learning strategy
- State-of-the-art on denoising, deblurring, deraining

Adapted for real estate HDR enhancement / color grading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D inputs (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (MDTA).

    Key innovation of Restormer - applies self-attention across channels
    instead of spatial locations, making it efficient for high-res images.
    """

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention: attend across channels, not spatial
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN).

    Uses gating mechanism with depthwise convolutions for
    efficient feature processing.
    """

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """Restormer transformer block with MDTA and GDFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """Pixel unshuffle downsampling."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Pixel shuffle upsampling."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """
    Restormer for Image Enhancement.

    U-shaped architecture with transformer blocks at each level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: List[int] = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: List[int] = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            out_channels: Output channels (3 for RGB)
            dim: Base channel dimension
            num_blocks: Number of transformer blocks at each level
            num_refinement_blocks: Number of refinement blocks
            heads: Number of attention heads at each level
            ffn_expansion_factor: Expansion factor for FFN
            bias: Use bias in convolutions
        """
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # Encoder
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
              for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
              for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
              for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4)

        # Bottleneck
        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias)
              for _ in range(num_blocks[3])]
        )

        # Decoder
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
              for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
              for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
              for _ in range(num_blocks[0])]
        )

        # Refinement
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
              for _ in range(num_refinement_blocks)]
        )

        # Output
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        inp = x

        # Patch embedding
        x = self.patch_embed(x)

        # Encoder
        enc1 = self.encoder_level1(x)
        x = self.down1_2(enc1)

        enc2 = self.encoder_level2(x)
        x = self.down2_3(enc2)

        enc3 = self.encoder_level3(x)
        x = self.down3_4(enc3)

        # Bottleneck
        x = self.latent(x)

        # Decoder
        x = self.up4_3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.reduce_chan_level3(x)
        x = self.decoder_level3(x)

        x = self.up3_2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x)

        x = self.up2_1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.reduce_chan_level1(x)
        x = self.decoder_level1(x)

        # Refinement
        x = self.refinement(x)

        # Output with residual
        x = self.output(x) + inp

        return x


def create_restormer(variant: str = "base") -> Restormer:
    """
    Create Restormer model with predefined configurations.

    Args:
        variant: "tiny", "small", "base", or "large"
    """
    configs = {
        "tiny": {
            "dim": 24,
            "num_blocks": [2, 3, 3, 4],
            "num_refinement_blocks": 2,
            "heads": [1, 2, 4, 8],
        },
        "small": {
            "dim": 32,
            "num_blocks": [3, 4, 4, 6],
            "num_refinement_blocks": 3,
            "heads": [1, 2, 4, 8],
        },
        "base": {
            "dim": 48,
            "num_blocks": [4, 6, 6, 8],
            "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8],
        },
        "large": {
            "dim": 64,
            "num_blocks": [6, 8, 8, 12],
            "num_refinement_blocks": 6,
            "heads": [1, 2, 4, 8],
        },
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}")

    return Restormer(**configs[variant])


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for variant in ["tiny", "small", "base", "large"]:
        model = create_restormer(variant).to(device)
        params = count_parameters(model)

        x = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            y = model(x)

        print(f"Restormer-{variant}: {params/1e6:.2f}M params, "
              f"input {x.shape} -> output {y.shape}")
