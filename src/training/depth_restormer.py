"""
Depth-Aware Restormer
=====================

Modified Restormer that takes RGBD (4-channel) input.
Uses depth information to help distinguish windows from other bright surfaces.

Key insight: Windows have "infinite" depth (far), while bright interior surfaces
have finite depth (close). This helps the model understand WHAT to fix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .restormer import (
    LayerNorm2d,
    MDTA,
    GDFN,
    TransformerBlock,
    TransformerStage,
    Downsample,
    Upsample
)


class DepthAwareRestormer(nn.Module):
    """
    Restormer with RGBD input for depth-aware image enhancement.

    Architecture:
    - Takes 4-channel input (RGB + Depth)
    - Processes through U-Net transformer architecture
    - Outputs 3-channel RGB with residual connection to RGB input

    The depth channel provides:
    - Window detection (high depth = far = windows/exterior)
    - Scene understanding (depth discontinuities at window edges)
    - Better spatial awareness for selective enhancement
    """

    def __init__(
        self,
        in_channels: int = 4,  # RGBD
        out_channels: int = 3,  # RGB output
        dim: int = 48,
        num_blocks: List[int] = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: List[int] = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        use_checkpointing: bool = False,
        depth_fusion: str = "early",  # "early", "multi_scale", or "attention"
    ):
        """
        Args:
            in_channels: Input channels (4 for RGBD)
            out_channels: Output channels (3 for RGB)
            dim: Base channel dimension
            num_blocks: Number of transformer blocks at each level
            num_refinement_blocks: Number of refinement blocks
            heads: Number of attention heads at each level
            ffn_expansion_factor: Expansion factor for FFN
            bias: Use bias in convolutions
            use_checkpointing: Enable gradient checkpointing
            depth_fusion: How to fuse depth information
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth_fusion = depth_fusion

        # Patch embedding - takes RGBD (4 channels)
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # Additional depth processing for multi-scale fusion
        if depth_fusion == "multi_scale":
            self.depth_encoders = nn.ModuleList([
                nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=bias),
                nn.Conv2d(1, dim * 2, kernel_size=3, padding=1, bias=bias),
                nn.Conv2d(1, dim * 4, kernel_size=3, padding=1, bias=bias),
                nn.Conv2d(1, dim * 8, kernel_size=3, padding=1, bias=bias),
            ])

        # Encoder
        self.encoder_level1 = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
             for _ in range(num_blocks[0])],
            use_checkpoint=use_checkpointing
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = TransformerStage(
            [TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
             for _ in range(num_blocks[1])],
            use_checkpoint=use_checkpointing
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = TransformerStage(
            [TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
             for _ in range(num_blocks[2])],
            use_checkpoint=use_checkpointing
        )
        self.down3_4 = Downsample(dim * 4)

        # Bottleneck
        self.latent = TransformerStage(
            [TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias)
             for _ in range(num_blocks[3])],
            use_checkpoint=use_checkpointing
        )

        # Decoder
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = TransformerStage(
            [TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
             for _ in range(num_blocks[2])],
            use_checkpoint=use_checkpointing
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = TransformerStage(
            [TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
             for _ in range(num_blocks[1])],
            use_checkpoint=use_checkpointing
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
             for _ in range(num_blocks[0])],
            use_checkpoint=use_checkpointing
        )

        # Refinement
        self.refinement = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
             for _ in range(num_refinement_blocks)],
            use_checkpoint=use_checkpointing
        )

        # Output - produces RGB (3 channels)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W] RGB or [B, 4, H, W] RGBD
            depth: Optional separate depth tensor [B, 1, H, W]

        Returns:
            Output RGB tensor [B, 3, H, W]
        """
        # Handle input formats
        if x.shape[1] == 4:
            # RGBD input - split
            rgb_input = x[:, :3]
            depth_input = x[:, 3:4]
        elif x.shape[1] == 3 and depth is not None:
            # Separate RGB and depth
            rgb_input = x
            depth_input = depth
        else:
            # RGB only - use zeros for depth
            rgb_input = x
            depth_input = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        # Concatenate for patch embedding
        rgbd = torch.cat([rgb_input, depth_input], dim=1)

        # Patch embedding
        x = self.patch_embed(rgbd)

        # Multi-scale depth fusion (optional)
        depth_feats = None
        if self.depth_fusion == "multi_scale":
            depth_feats = []
            d = depth_input
            for i, enc in enumerate(self.depth_encoders):
                depth_feats.append(enc(d))
                if i < len(self.depth_encoders) - 1:
                    d = F.avg_pool2d(d, 2)

        # Encoder
        enc1 = self.encoder_level1(x)
        if depth_feats:
            enc1 = enc1 + depth_feats[0]
        x = self.down1_2(enc1)

        enc2 = self.encoder_level2(x)
        if depth_feats:
            enc2 = enc2 + F.interpolate(depth_feats[1], size=enc2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.down2_3(enc2)

        enc3 = self.encoder_level3(x)
        if depth_feats:
            enc3 = enc3 + F.interpolate(depth_feats[2], size=enc3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.down3_4(enc3)

        # Bottleneck
        x = self.latent(x)
        if depth_feats:
            x = x + F.interpolate(depth_feats[3], size=x.shape[-2:], mode='bilinear', align_corners=False)

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

        # Output with residual connection to RGB input
        x = self.output(x) + rgb_input

        return x


def create_depth_restormer(
    variant: str = "small",
    depth_fusion: str = "early",
    use_checkpointing: bool = True
) -> DepthAwareRestormer:
    """
    Create depth-aware Restormer with predefined configurations.

    Args:
        variant: "tiny", "small", "base", or "large"
        depth_fusion: "early", "multi_scale", or "attention"
        use_checkpointing: Enable gradient checkpointing for memory efficiency
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

    config = configs[variant]

    return DepthAwareRestormer(
        in_channels=4,
        out_channels=3,
        depth_fusion=depth_fusion,
        use_checkpointing=use_checkpointing,
        **config
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Depth-Aware Restormer...")

    for variant in ["tiny", "small"]:
        for fusion in ["early", "multi_scale"]:
            model = create_depth_restormer(variant, fusion).to(device)
            params = count_parameters(model)

            # Test with RGBD input
            rgbd = torch.randn(1, 4, 256, 256).to(device)
            with torch.no_grad():
                y = model(rgbd)

            print(f"Depth-Restormer-{variant} ({fusion}): {params/1e6:.2f}M params, "
                  f"RGBD {rgbd.shape} -> RGB {y.shape}")
