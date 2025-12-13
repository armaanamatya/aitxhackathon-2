"""
SwinRestormer: Hybrid Pretrained Swin Encoder + Restormer Decoder
==================================================================

Combines:
- Pretrained Swin Transformer encoder (ImageNet-22K, 14M images)
- Restormer-style decoder with MDTA and GDFN
- Progressive unfreezing for optimal fine-tuning

This approach leverages pretrained features to avoid overfitting on small
datasets (~550 images) while maintaining high-quality restoration capability.

Architecture:
- Encoder: Swin Transformer (frozen initially, then fine-tuned)
- Decoder: Restormer-style transformer blocks
- Skip connections: Multi-scale feature fusion

Training Strategy:
- Stage 1: Freeze encoder, train decoder only (fast convergence)
- Stage 2: Unfreeze encoder with lower LR (fine-tuning)
- Stage 3: Full fine-tuning with very low LR
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


# =============================================================================
# Restormer-style Decoder Components
# =============================================================================

class LayerNorm2d(nn.Module):
    """Layer normalization for 2D inputs."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention."""

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Restormer transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PixelShuffleUpsample(nn.Module):
    """Upsample with pixel shuffle."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


# =============================================================================
# Swin Encoder Wrapper
# =============================================================================

class SwinEncoder(nn.Module):
    """
    Wrapper for pretrained Swin Transformer from timm.
    Extracts multi-scale features for the decoder.
    """

    def __init__(
        self,
        model_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for SwinEncoder")

        # Load pretrained Swin
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Get feature dimensions
        # Swin-Base: [128, 256, 512, 1024] at different stages
        self.feature_dims = self.swin.feature_info.channels()

        # Freeze if requested
        self.frozen = freeze
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze encoder weights."""
        for param in self.swin.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze(self, layers: Optional[List[int]] = None):
        """
        Unfreeze encoder weights.

        Args:
            layers: List of layer indices to unfreeze (0-3). None = all layers.
        """
        if layers is None:
            for param in self.swin.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific stages
            for name, param in self.swin.named_parameters():
                for layer_idx in layers:
                    if f'layers.{layer_idx}' in name or f'stages.{layer_idx}' in name:
                        param.requires_grad = True
        self.frozen = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Returns:
            List of features at 4 scales: [1/4, 1/8, 1/16, 1/32] of input resolution
        """
        features = self.swin(x)
        return features


# =============================================================================
# Restormer-style Decoder
# =============================================================================

class RestormerDecoder(nn.Module):
    """
    Restormer-style decoder with skip connections.
    Takes multi-scale encoder features and produces output image.
    """

    def __init__(
        self,
        encoder_dims: List[int] = [128, 256, 512, 1024],
        decoder_dim: int = 64,
        num_blocks: List[int] = [2, 2, 4, 4],
        num_heads: List[int] = [1, 2, 4, 8],
        out_channels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim

        # Channel reduction for encoder features
        self.reduce_channels = nn.ModuleList([
            nn.Conv2d(enc_dim, decoder_dim * (2 ** i), 1)
            for i, enc_dim in enumerate(encoder_dims)
        ])

        # Decoder levels (bottom-up)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        dims = [decoder_dim * (2 ** i) for i in range(len(encoder_dims))]

        for i in range(len(encoder_dims) - 1, 0, -1):
            # Transformer blocks at this level
            blocks = nn.Sequential(*[
                TransformerBlock(dims[i], num_heads[i], dropout=dropout)
                for _ in range(num_blocks[i])
            ])
            self.decoder_blocks.append(blocks)

            # Upsample
            self.upsample_blocks.append(
                PixelShuffleUpsample(dims[i], dims[i-1])
            )

            # Fusion with skip connection
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(dims[i-1] * 2, dims[i-1], 1),
                    LayerNorm2d(dims[i-1]),
                    nn.GELU(),
                )
            )

        # Final refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(decoder_dim, num_heads[0], dropout=dropout)
            for _ in range(num_blocks[0])
        ])

        # Output projection
        self.output = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_dim, out_channels, 3, padding=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale features to output image.

        Args:
            features: List of encoder features [f1, f2, f3, f4] at increasing depth

        Returns:
            Output image
        """
        # Reduce channel dimensions
        features = [reduce(f) for reduce, f in zip(self.reduce_channels, features)]

        # Start from deepest feature
        x = features[-1]

        # Decode bottom-up
        for i, (decoder, upsample, fusion) in enumerate(zip(
            self.decoder_blocks, self.upsample_blocks, self.fusion_blocks
        )):
            x = decoder(x)
            x = upsample(x)

            # Get corresponding skip feature
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx]

            # Resize if needed
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            # Fuse with skip connection
            x = fusion(torch.cat([x, skip], dim=1))

        # Final refinement
        x = self.refinement(x)

        # Output
        return self.output(x)


# =============================================================================
# SwinRestormer: Full Model
# =============================================================================

class SwinRestormer(nn.Module):
    """
    Hybrid model combining pretrained Swin encoder with Restormer decoder.

    Training Stages:
    1. Freeze encoder, train decoder (epochs 1-50)
    2. Unfreeze last 2 encoder layers, lower LR (epochs 51-100)
    3. Full fine-tuning with very low LR (epochs 101+)
    """

    def __init__(
        self,
        encoder_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        freeze_encoder: bool = True,
        decoder_dim: int = 64,
        num_blocks: List[int] = [2, 2, 4, 4],
        num_heads: List[int] = [1, 2, 4, 8],
        in_channels: int = 3,
        out_channels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels

        # Input projection (handle different input sizes)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.GELU(),
        ) if in_channels != 3 else nn.Identity()

        # Pretrained Swin encoder
        self.encoder = SwinEncoder(
            model_name=encoder_name,
            pretrained=pretrained,
            freeze=freeze_encoder,
        )

        # Restormer-style decoder
        self.decoder = RestormerDecoder(
            encoder_dims=self.encoder.feature_dims,
            decoder_dim=decoder_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            out_channels=out_channels,
            dropout=dropout,
        )

        # Global residual connection
        self.global_residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            Dictionary with 'output' key containing the enhanced image
        """
        # Store original size
        orig_size = x.shape[-2:]

        # Resize to multiple of 32 for Swin (patch size 4, window 7)
        h, w = x.shape[-2:]
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32

        if (new_h, new_w) != (h, w):
            x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            x_resized = x

        # Input projection
        x_proj = self.input_proj(x_resized)

        # Encode
        features = self.encoder(x_proj)

        # Decode
        output = self.decoder(features)

        # Upsample output to match input resolution
        if output.shape[-2:] != (new_h, new_w):
            output = F.interpolate(output, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Global residual
        output = output + self.global_residual(x_resized)

        # Resize back to original if needed
        if (new_h, new_w) != orig_size:
            output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=False)

        return {'output': output}

    def get_encoder_params(self) -> List[torch.nn.Parameter]:
        """Get encoder parameters for optimizer groups."""
        return list(self.encoder.parameters())

    def get_decoder_params(self) -> List[torch.nn.Parameter]:
        """Get decoder parameters for optimizer groups."""
        params = list(self.decoder.parameters())
        params += list(self.input_proj.parameters())
        params += list(self.global_residual.parameters())
        return params

    def freeze_encoder(self):
        """Freeze encoder for stage 1 training."""
        self.encoder.freeze()

    def unfreeze_encoder(self, layers: Optional[List[int]] = None):
        """Unfreeze encoder layers for stage 2/3 training."""
        self.encoder.unfreeze(layers)

    def set_training_stage(self, stage: int):
        """
        Set training stage for progressive unfreezing.

        Stage 1: Freeze encoder, train decoder only
        Stage 2: Unfreeze last 2 encoder layers
        Stage 3: Unfreeze all encoder layers
        """
        if stage == 1:
            self.freeze_encoder()
            print("Stage 1: Encoder frozen, training decoder only")
        elif stage == 2:
            self.unfreeze_encoder(layers=[2, 3])  # Last 2 stages
            print("Stage 2: Unfroze encoder layers 2, 3")
        elif stage == 3:
            self.unfreeze_encoder(layers=None)  # All layers
            print("Stage 3: All encoder layers unfrozen")


# =============================================================================
# Model Configurations
# =============================================================================

def swin_restormer_tiny(pretrained: bool = True, freeze_encoder: bool = True) -> SwinRestormer:
    """Tiny model with Swin-Tiny encoder (~12M params)."""
    return SwinRestormer(
        encoder_name='swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        decoder_dim=48,
        num_blocks=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        dropout=0.1,
    )


def swin_restormer_small(pretrained: bool = True, freeze_encoder: bool = True) -> SwinRestormer:
    """Small model with Swin-Small encoder (~30M params)."""
    return SwinRestormer(
        encoder_name='swin_small_patch4_window7_224',
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        decoder_dim=64,
        num_blocks=[2, 2, 4, 4],
        num_heads=[1, 2, 4, 8],
        dropout=0.1,
    )


def swin_restormer_base(pretrained: bool = True, freeze_encoder: bool = True) -> SwinRestormer:
    """Base model with Swin-Base encoder (~50M params)."""
    return SwinRestormer(
        encoder_name='swin_base_patch4_window7_224',
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        decoder_dim=64,
        num_blocks=[2, 2, 4, 4],
        num_heads=[1, 2, 4, 8],
        dropout=0.1,
    )


if __name__ == '__main__':
    print("Testing SwinRestormer models...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test tiny model
    model = swin_restormer_tiny(pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (encoder frozen): {trainable_params:,}")

    # Test forward pass
    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out['output'].shape}")

    # Test unfreezing
    model.set_training_stage(2)
    trainable_params_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (stage 2): {trainable_params_stage2:,}")

    model.set_training_stage(3)
    trainable_params_stage3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (stage 3): {trainable_params_stage3:,}")

    print("All tests passed!")
