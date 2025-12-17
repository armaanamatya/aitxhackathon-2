"""
Multi-Feature Restormer
=======================

Enhanced Restormer that takes RGB + multiple auxiliary features:
- Depth: Scene geometry (windows = far)
- Edge: Structural information for detail preservation
- Saturation: Color richness (windows = washed out)
- Local contrast: Identifies regions needing enhancement

Total input: RGB(3) + Depth(1) + Edge(1) + Saturation(1) = 6 channels
Or configurable subset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

from .restormer import (
    LayerNorm2d,
    TransformerBlock,
    TransformerStage,
    Downsample,
    Upsample
)


class FeatureExtractor(nn.Module):
    """
    Extracts multiple auxiliary features from RGB images.

    All features computed on-the-fly (no external models needed).
    Memory efficient.
    """

    def __init__(self):
        super().__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute luminance from RGB."""
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    def compute_saturation(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute saturation.
        Low saturation = washed out (typical of blown-out windows)
        """
        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        return saturation

    def compute_edges(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute edge magnitude using Sobel operator.
        Edges help preserve structural details.
        """
        # Use luminance for edge detection
        lum = self.compute_luminance(rgb)

        # Apply Sobel filters
        grad_x = F.conv2d(lum, self.sobel_x, padding=1)
        grad_y = F.conv2d(lum, self.sobel_y, padding=1)

        # Edge magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Normalize to [0, 1]
        edges = edges / (edges.max() + 1e-8)

        return edges

    def compute_local_contrast(self, rgb: torch.Tensor, window_size: int = 15) -> torch.Tensor:
        """
        Compute local contrast.
        High contrast regions are visually important.
        Low contrast (like windows) may need enhancement.
        """
        lum = self.compute_luminance(rgb)

        # Local mean
        padding = window_size // 2
        local_mean = F.avg_pool2d(
            F.pad(lum, (padding,)*4, mode='reflect'),
            window_size, stride=1
        )

        # Local standard deviation (contrast measure)
        local_var = F.avg_pool2d(
            F.pad((lum - local_mean)**2, (padding,)*4, mode='reflect'),
            window_size, stride=1
        )
        local_contrast = torch.sqrt(local_var + 1e-8)

        # Normalize per image
        B = local_contrast.shape[0]
        flat = local_contrast.view(B, -1)
        c_max = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        local_contrast = local_contrast / (c_max + 1e-8)

        return local_contrast

    def compute_depth_proxy(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Simple depth proxy based on brightness and saturation.

        Heuristic for real estate:
        - Bright + low saturation → windows → far
        - Dark → interior → close
        """
        lum = self.compute_luminance(rgb)
        sat = self.compute_saturation(rgb)

        # Depth proxy: bright * (1 - sat)
        # High for windows (bright, washed out)
        # Low for interior (darker, more colorful)
        depth_proxy = lum * (1 - sat)

        # Smooth
        kernel_size = 15
        depth_smooth = F.avg_pool2d(
            F.pad(depth_proxy, (kernel_size//2,)*4, mode='reflect'),
            kernel_size, stride=1
        )

        # Normalize per image
        B = depth_smooth.shape[0]
        flat = depth_smooth.view(B, -1)
        d_min = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        d_max = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        depth_smooth = (depth_smooth - d_min) / (d_max - d_min + 1e-8)

        return depth_smooth

    def forward(
        self,
        rgb: torch.Tensor,
        features: List[str] = ['depth', 'edge', 'saturation', 'contrast']
    ) -> Dict[str, torch.Tensor]:
        """
        Extract multiple features from RGB.

        Args:
            rgb: [B, 3, H, W] in range [0, 1]
            features: List of features to extract

        Returns:
            Dict mapping feature names to tensors [B, 1, H, W]
        """
        result = {}

        if 'depth' in features:
            result['depth'] = self.compute_depth_proxy(rgb)

        if 'edge' in features:
            result['edge'] = self.compute_edges(rgb)

        if 'saturation' in features:
            result['saturation'] = self.compute_saturation(rgb)

        if 'contrast' in features:
            result['contrast'] = self.compute_local_contrast(rgb)

        if 'luminance' in features:
            result['luminance'] = self.compute_luminance(rgb)

        return result


class MultiFeatureRestormer(nn.Module):
    """
    Restormer with multi-feature input for robust window recovery.

    Configurable input features beyond RGB.
    """

    def __init__(
        self,
        features: List[str] = ['depth', 'edge', 'saturation'],
        dim: int = 48,
        num_blocks: List[int] = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: List[int] = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        use_checkpointing: bool = True,
    ):
        """
        Args:
            features: List of auxiliary features to use
                     Options: 'depth', 'edge', 'saturation', 'contrast', 'luminance'
            dim: Base channel dimension
            num_blocks: Number of transformer blocks at each level
            num_refinement_blocks: Number of refinement blocks
            heads: Number of attention heads at each level
            ffn_expansion_factor: Expansion factor for FFN
            bias: Use bias in convolutions
            use_checkpointing: Enable gradient checkpointing
        """
        super().__init__()

        self.features = features
        self.feature_extractor = FeatureExtractor()

        # Input channels: RGB (3) + number of features
        in_channels = 3 + len(features)
        out_channels = 3  # RGB output

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

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

        # Output
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input RGB tensor [B, 3, H, W] in range [0, 1]

        Returns:
            Output RGB tensor [B, 3, H, W]
        """
        rgb_input = x

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(x, self.features)

        # Concatenate RGB with features
        feature_list = [x]
        for feat_name in self.features:
            if feat_name in features:
                feature_list.append(features[feat_name])

        x = torch.cat(feature_list, dim=1)

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

        # Output with residual connection to RGB input
        x = self.output(x) + rgb_input

        return x


def create_multi_feature_restormer(
    variant: str = "small",
    features: List[str] = ['depth', 'edge', 'saturation'],
    use_checkpointing: bool = True
) -> MultiFeatureRestormer:
    """
    Create multi-feature Restormer with predefined configurations.

    Args:
        variant: "tiny", "small", "base", or "large"
        features: List of auxiliary features to use
        use_checkpointing: Enable gradient checkpointing
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

    return MultiFeatureRestormer(
        features=features,
        use_checkpointing=use_checkpointing,
        **config
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Multi-Feature Restormer...")

    # Test with different feature combinations
    feature_sets = [
        ['depth'],
        ['depth', 'edge'],
        ['depth', 'edge', 'saturation'],
        ['depth', 'edge', 'saturation', 'contrast'],
    ]

    for features in feature_sets:
        model = create_multi_feature_restormer('small', features).to(device)
        params = sum(p.numel() for p in model.parameters())

        # Test with RGB input (features extracted internally)
        rgb = torch.rand(1, 3, 256, 256).to(device)
        with torch.no_grad():
            y = model(rgb)

        print(f"Features {features}: {params/1e6:.2f}M params, "
              f"RGB {rgb.shape} -> RGB {y.shape}")
