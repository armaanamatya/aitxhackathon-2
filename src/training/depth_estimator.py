"""
Depth Estimation Module
=======================

Uses pretrained depth estimation models to generate depth maps.
Supports:
- MiDaS (torch hub) - Primary option, reliable
- Simple luminance-based fallback

Depth is normalized to [0, 1] and can be used as 4th channel input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MiDaSDepthEstimator(nn.Module):
    """
    MiDaS - Robust monocular depth estimation.

    Paper: "Towards Robust Monocular Depth Estimation"
    Uses torch hub implementation (no tensorflow/transformers dependencies).
    """

    def __init__(self, model_size: str = "small", device: str = "cuda"):
        """
        Args:
            model_size: "small", "base" (DPT-Hybrid), or "large" (DPT-Large)
            device: Device to run model on
        """
        super().__init__()
        self.device = device
        self.model_size = model_size
        self.model = None
        self.transform = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization to avoid loading model until needed."""
        if self._initialized:
            return

        try:
            # MiDaS model variants
            model_types = {
                "small": "MiDaS_small",
                "base": "DPT_Hybrid",
                "large": "DPT_Large",
            }

            model_type = model_types.get(self.model_size, "MiDaS_small")

            print(f"Loading MiDaS ({model_type})...")

            # Load model from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

            if model_type == "MiDaS_small":
                self.transform = midas_transforms.small_transform
                self.input_size = 256
            elif model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
                self.input_size = 384
            else:
                self.transform = midas_transforms.dpt_transform
                self.input_size = 384

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            self._initialized = True
            print(f"MiDaS loaded successfully")

        except Exception as e:
            print(f"Failed to load MiDaS: {e}")
            print("Using luminance-based depth estimation fallback...")
            self._use_simple_fallback = True
            self._initialized = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from RGB images.

        Args:
            x: Input tensor [B, 3, H, W] in range [0, 1]

        Returns:
            Depth tensor [B, 1, H, W] normalized to [0, 1]
            Higher values = farther (windows will have high depth)
        """
        self._lazy_init()

        B, C, H, W = x.shape

        # Simple luminance-based fallback if model failed to load
        if hasattr(self, '_use_simple_fallback') and self._use_simple_fallback:
            return self._simple_depth_estimate(x)

        try:
            return self._midas_forward(x)
        except Exception as e:
            print(f"MiDaS forward error: {e}, using fallback")
            return self._simple_depth_estimate(x)

    def _midas_forward(self, x: torch.Tensor) -> torch.Tensor:
        """MiDaS forward pass - batched."""
        B, C, H, W = x.shape

        # Convert to numpy for MiDaS transform
        x_np = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')

        depths = []
        for i in range(B):
            # Apply MiDaS transform
            img_input = self.transform(x_np[i])

            # Handle different transform outputs
            if isinstance(img_input, dict):
                img_input = img_input['pixel_values']
            if isinstance(img_input, list):
                img_input = img_input[0]
            if not isinstance(img_input, torch.Tensor):
                img_input = torch.from_numpy(img_input)

            img_input = img_input.to(self.device)
            if img_input.dim() == 3:
                img_input = img_input.unsqueeze(0)

            # Forward pass
            depth = self.model(img_input)

            # Resize to original size
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )
            depths.append(depth)

        depth = torch.cat(depths, dim=0)
        return self._normalize_depth(depth)

    def _simple_depth_estimate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple depth estimation fallback based on brightness and saturation.

        Heuristic for real estate images:
        - Bright, low-saturation regions (windows) → far (high depth)
        - Dark regions → close (low depth)
        - This is a rough but useful approximation
        """
        # Luminance
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Saturation (windows are often washed out / low saturation)
        max_rgb = x.max(dim=1, keepdim=True)[0]
        min_rgb = x.min(dim=1, keepdim=True)[0]
        sat = (max_rgb - min_rgb) / (max_rgb + 1e-8)

        # Windows are bright and low saturation
        # depth = brightness * (1 - saturation)
        window_indicator = lum * (1 - sat)

        # Smooth to remove noise
        kernel_size = 15
        depth_smooth = F.avg_pool2d(
            F.pad(window_indicator, (kernel_size//2,)*4, mode='reflect'),
            kernel_size, stride=1
        )

        return self._normalize_depth(depth_smooth)

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize depth to [0, 1] per image."""
        B = depth.shape[0]
        depth_flat = depth.view(B, -1)

        d_min = depth_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        d_max = depth_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        return depth


# Alias for backward compatibility
DepthAnythingEstimator = MiDaSDepthEstimator


class EfficientDepthEstimator(nn.Module):
    """
    Memory-efficient depth estimation that processes in batches.

    For training, we can cache depth maps or compute them efficiently.
    """

    def __init__(self, model_size: str = "small", cache_dir: Optional[str] = None):
        super().__init__()
        self.model_size = model_size
        self.cache_dir = cache_dir
        self.depth_model = None

    def _lazy_init(self, device):
        if self.depth_model is None:
            self.depth_model = DepthAnythingEstimator(self.model_size, device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate depth with memory efficiency."""
        self._lazy_init(x.device)
        return self.depth_model(x)


class DepthAwareFeatureFusion(nn.Module):
    """
    Fuses RGB features with depth information.

    Can be used to inject depth at multiple scales in the network.
    """

    def __init__(self, rgb_dim: int, depth_dim: int = 1, fusion_type: str = "concat"):
        """
        Args:
            rgb_dim: Dimension of RGB features
            depth_dim: Dimension of depth features (1 for raw depth)
            fusion_type: "concat", "add", or "attention"
        """
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            # Project depth to match RGB dim, then concat
            self.depth_proj = nn.Conv2d(depth_dim, rgb_dim // 4, kernel_size=1)
            self.fuse = nn.Conv2d(rgb_dim + rgb_dim // 4, rgb_dim, kernel_size=1)

        elif fusion_type == "add":
            self.depth_proj = nn.Conv2d(depth_dim, rgb_dim, kernel_size=1)

        elif fusion_type == "attention":
            # Use depth as attention weights
            self.depth_proj = nn.Sequential(
                nn.Conv2d(depth_dim, rgb_dim, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, rgb_feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB features with depth.

        Args:
            rgb_feat: [B, C, H, W] RGB features
            depth: [B, 1, H, W] depth map
        """
        # Resize depth to match feature size
        if depth.shape[-2:] != rgb_feat.shape[-2:]:
            depth = F.interpolate(depth, size=rgb_feat.shape[-2:], mode='bilinear', align_corners=False)

        if self.fusion_type == "concat":
            depth_feat = self.depth_proj(depth)
            fused = torch.cat([rgb_feat, depth_feat], dim=1)
            return self.fuse(fused)

        elif self.fusion_type == "add":
            depth_feat = self.depth_proj(depth)
            return rgb_feat + depth_feat

        elif self.fusion_type == "attention":
            depth_attn = self.depth_proj(depth)
            return rgb_feat * depth_attn

        return rgb_feat


def create_depth_estimator(
    model_type: str = "midas",
    model_size: str = "small",
    device: str = "cuda"
) -> nn.Module:
    """
    Factory function for depth estimators.

    Args:
        model_type: "midas", "depth_anything", or "simple"
        model_size: "small", "base", or "large"
        device: Target device
    """
    if model_type in ["midas", "depth_anything"]:
        return MiDaSDepthEstimator(model_size, device)
    elif model_type == "efficient":
        return EfficientDepthEstimator(model_size)
    else:
        # Simple luminance-based (creates MiDaS which has fallback)
        return MiDaSDepthEstimator("small", device)


if __name__ == "__main__":
    # Test depth estimation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing depth estimator...")
    estimator = create_depth_estimator("depth_anything", "small", str(device))

    # Test input
    x = torch.rand(2, 3, 256, 256).to(device)

    depth = estimator(x)
    print(f"Input: {x.shape}")
    print(f"Depth: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
