"""
Bright Region Color Loss for HDR Real Estate Enhancement

Problem: Windows showing sky, plants, outdoor views have vibrant colors in GT
but models tend to produce desaturated/averaged colors.

Solution: Multi-component loss targeting COLOR ACCURACY in bright regions:
1. Hue Loss - Preserve exact hue in bright regions
2. Chroma Loss - Preserve colorfulness/saturation
3. Color Histogram Loss - Match color distribution
4. Per-Channel Loss - RGB channel-wise accuracy in bright regions

Author: Top 0.0001% MLE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BrightRegionHueLoss(nn.Module):
    """
    Hue preservation loss for bright regions.

    Hue is the "color type" (red, green, blue, etc.)
    Critical for: blue sky, green plants, colorful objects in windows
    """

    def __init__(self, brightness_threshold: float = 0.4, hue_weight: float = 2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.hue_weight = hue_weight

    def rgb_to_hue(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert RGB to Hue. Returns (hue, value) where hue is in [0, 1].
        """
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        max_c, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        diff = max_c - min_c + 1e-7

        # Hue calculation
        hue = torch.zeros_like(max_c)

        # When max is R
        mask_r = (max_idx == 0).float()
        hue_r = ((g - b) / diff) % 6
        hue = hue + mask_r * hue_r

        # When max is G
        mask_g = (max_idx == 1).float()
        hue_g = ((b - r) / diff) + 2
        hue = hue + mask_g * hue_g

        # When max is B
        mask_b = (max_idx == 2).float()
        hue_b = ((r - g) / diff) + 4
        hue = hue + mask_b * hue_b

        hue = hue / 6.0  # Normalize to [0, 1]
        hue = torch.clamp(hue, 0, 1)

        return hue, max_c

    def circular_hue_distance(self, hue1: torch.Tensor, hue2: torch.Tensor) -> torch.Tensor:
        """
        Compute circular distance between hues (hue wraps around at 1.0).
        """
        diff = torch.abs(hue1 - hue2)
        return torch.min(diff, 1.0 - diff)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hue, pred_val = self.rgb_to_hue(pred)
        target_hue, target_val = self.rgb_to_hue(target)

        # Bright region mask from target
        bright_mask = (target_val > self.brightness_threshold).float()
        bright_mask = F.avg_pool2d(bright_mask, kernel_size=5, stride=1, padding=2)

        # Also require that target has some saturation (is colorful)
        target_sat = (target.max(dim=1, keepdim=True)[0] - target.min(dim=1, keepdim=True)[0])
        colorful_mask = (target_sat > 0.1).float()  # Has some color

        combined_mask = bright_mask * colorful_mask

        # Circular hue distance in bright colorful regions
        hue_diff = self.circular_hue_distance(pred_hue, target_hue)
        bright_hue_loss = (hue_diff * combined_mask).sum() / (combined_mask.sum() + 1e-7)

        return self.hue_weight * bright_hue_loss


class BrightRegionChromaLoss(nn.Module):
    """
    Chroma (colorfulness) loss for bright regions.

    Chroma = how "colorful" a pixel is (vs gray).
    Critical for: vibrant sky blue, lush green plants, saturated colors
    """

    def __init__(self, brightness_threshold: float = 0.4, chroma_weight: float = 2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.chroma_weight = chroma_weight

    def rgb_to_lab_chroma(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximate chroma from RGB (simplified, avoids full Lab conversion).
        Chroma ≈ sqrt(a² + b²) in Lab space.

        Simplified: Chroma ≈ max(R,G,B) - min(R,G,B) (saturation)
        """
        max_c = rgb.max(dim=1, keepdim=True)[0]
        min_c = rgb.min(dim=1, keepdim=True)[0]

        chroma = max_c - min_c
        lightness = (max_c + min_c) / 2

        return chroma, lightness

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_chroma, pred_light = self.rgb_to_lab_chroma(pred)
        target_chroma, target_light = self.rgb_to_lab_chroma(target)

        # Bright region mask
        bright_mask = (target_light > self.brightness_threshold).float()
        bright_mask = F.avg_pool2d(bright_mask, kernel_size=5, stride=1, padding=2)

        # Chroma difference in bright regions
        chroma_diff = torch.abs(pred_chroma - target_chroma)
        bright_chroma_loss = (chroma_diff * bright_mask).sum() / (bright_mask.sum() + 1e-7)

        # Also penalize if prediction is LESS colorful than target in bright regions
        # This asymmetric loss pushes model to maintain/increase saturation
        under_saturation = F.relu(target_chroma - pred_chroma)
        under_sat_loss = (under_saturation * bright_mask).sum() / (bright_mask.sum() + 1e-7)

        return self.chroma_weight * (bright_chroma_loss + 0.5 * under_sat_loss)


class BrightRegionRGBLoss(nn.Module):
    """
    Per-channel RGB loss in bright regions.

    Ensures each color channel (R, G, B) is accurately reproduced in bright areas.
    Critical for: Getting exact blue in sky, exact green in plants
    """

    def __init__(self, brightness_threshold: float = 0.4, rgb_weight: float = 1.5):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.rgb_weight = rgb_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Brightness from target
        target_brightness = target.mean(dim=1, keepdim=True)
        bright_mask = (target_brightness > self.brightness_threshold).float()
        bright_mask = F.avg_pool2d(bright_mask, kernel_size=7, stride=1, padding=3)

        # Per-channel L1 in bright regions
        channel_diff = torch.abs(pred - target)  # [B, 3, H, W]

        # Expand mask to all channels
        bright_mask_3ch = bright_mask.expand_as(channel_diff)

        bright_rgb_loss = (channel_diff * bright_mask_3ch).sum() / (bright_mask_3ch.sum() + 1e-7)

        return self.rgb_weight * bright_rgb_loss


class ColorHistogramLoss(nn.Module):
    """
    Color histogram matching loss for bright regions.

    Matches the distribution of colors in bright regions, not just per-pixel.
    Critical for: Overall color balance in windows
    """

    def __init__(self, brightness_threshold: float = 0.4, num_bins: int = 16):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.num_bins = num_bins

    def compute_soft_histogram(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute soft histogram of values in x, weighted by mask.
        """
        B, C, H, W = x.shape

        # Bin centers
        bin_centers = torch.linspace(0, 1, self.num_bins, device=x.device)
        bin_width = 1.0 / self.num_bins

        # Reshape for broadcasting
        x_flat = x.view(B, C, -1, 1)  # [B, C, HW, 1]
        mask_flat = mask.view(B, 1, -1, 1)  # [B, 1, HW, 1]
        bin_centers = bin_centers.view(1, 1, 1, -1)  # [1, 1, 1, num_bins]

        # Soft assignment (gaussian kernel)
        weights = torch.exp(-((x_flat - bin_centers) ** 2) / (2 * (bin_width ** 2)))
        weights = weights * mask_flat  # Apply mask

        # Histogram is sum of weights per bin
        hist = weights.sum(dim=2)  # [B, C, num_bins]

        # Normalize
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-7)

        return hist

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Bright region mask
        target_brightness = target.mean(dim=1, keepdim=True)
        bright_mask = (target_brightness > self.brightness_threshold).float()

        # Compute histograms
        pred_hist = self.compute_soft_histogram(pred, bright_mask)
        target_hist = self.compute_soft_histogram(target, bright_mask)

        # L1 distance between histograms
        hist_loss = F.l1_loss(pred_hist, target_hist)

        return hist_loss


class BrightColorLoss(nn.Module):
    """
    Complete Bright Region Color Loss.

    Combines multiple color-focused losses for bright region accuracy:
    - Hue: Exact color type (blue, green, etc.)
    - Chroma: Colorfulness/saturation
    - RGB: Per-channel accuracy
    - Histogram: Color distribution matching

    Usage:
        loss = BrightColorLoss()
        total, components = loss(pred, target)
    """

    def __init__(
        self,
        brightness_threshold: float = 0.4,
        hue_weight: float = 0.3,
        chroma_weight: float = 0.4,
        rgb_weight: float = 0.5,
        histogram_weight: float = 0.2,
    ):
        super().__init__()
        self.hue_weight = hue_weight
        self.chroma_weight = chroma_weight
        self.rgb_weight = rgb_weight
        self.histogram_weight = histogram_weight

        self.hue_loss = BrightRegionHueLoss(brightness_threshold, hue_weight=1.0)
        self.chroma_loss = BrightRegionChromaLoss(brightness_threshold, chroma_weight=1.0)
        self.rgb_loss = BrightRegionRGBLoss(brightness_threshold, rgb_weight=1.0)
        self.hist_loss = ColorHistogramLoss(brightness_threshold, num_bins=16)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        hue = self.hue_loss(pred, target)
        chroma = self.chroma_loss(pred, target)
        rgb = self.rgb_loss(pred, target)
        hist = self.hist_loss(pred, target)

        total = (
            self.hue_weight * hue +
            self.chroma_weight * chroma +
            self.rgb_weight * rgb +
            self.histogram_weight * hist
        )

        components = {
            'hue': hue.item(),
            'chroma': chroma.item(),
            'rgb': rgb.item(),
            'histogram': hist.item(),
        }

        return total, components


class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced Combined Loss: L1 + Window + BrightColor

    This is the recommended loss for HDR real estate enhancement.

    Components:
    - L1 (1.0): Primary pixel accuracy
    - Window (0.3): Extra weight on bright regions
    - BrightColor (0.5): Color accuracy in bright regions (Hue + Chroma + RGB + Histogram)
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        window_weight: float = 0.3,
        bright_color_weight: float = 0.5,
        brightness_threshold: float = 0.4,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.window_weight = window_weight
        self.bright_color_weight = bright_color_weight

        self.l1_loss = nn.L1Loss()
        self.window_loss = WindowAwareLoss(brightness_threshold)
        self.bright_color_loss = BrightColorLoss(brightness_threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        l1 = self.l1_loss(pred, target)
        window = self.window_loss(pred, target)
        bright_color, color_components = self.bright_color_loss(pred, target)

        total = (
            self.l1_weight * l1 +
            self.window_weight * window +
            self.bright_color_weight * bright_color
        )

        components = {
            'l1': l1.item(),
            'window': window.item(),
            'bright_color': bright_color.item(),
            **{f'color_{k}': v for k, v in color_components.items()},
            'total': total.item(),
        }

        return total, components


class WindowAwareLoss(nn.Module):
    """Window-aware loss (from original training script)."""

    def __init__(self, brightness_threshold: float = 0.7, window_weight: float = 2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.window_weight = window_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        luminance = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
        window_mask = (luminance > self.brightness_threshold).float()
        window_mask = F.avg_pool2d(window_mask, kernel_size=15, stride=1, padding=7)
        window_mask = torch.clamp(window_mask * 2, 0, 1)
        weight_mask = 1.0 + (self.window_weight - 1.0) * window_mask
        pixel_loss = torch.abs(pred - target)
        weighted_loss = (pixel_loss * weight_mask).mean()
        return weighted_loss


# Test
if __name__ == "__main__":
    print("Testing Bright Color Loss components...")

    # Create test tensors
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)

    # Test individual losses
    print("\n1. BrightRegionHueLoss:")
    hue_loss = BrightRegionHueLoss()
    print(f"   Loss: {hue_loss(pred, target):.4f}")

    print("\n2. BrightRegionChromaLoss:")
    chroma_loss = BrightRegionChromaLoss()
    print(f"   Loss: {chroma_loss(pred, target):.4f}")

    print("\n3. BrightRegionRGBLoss:")
    rgb_loss = BrightRegionRGBLoss()
    print(f"   Loss: {rgb_loss(pred, target):.4f}")

    print("\n4. ColorHistogramLoss:")
    hist_loss = ColorHistogramLoss()
    print(f"   Loss: {hist_loss(pred, target):.4f}")

    print("\n5. Combined BrightColorLoss:")
    bright_loss = BrightColorLoss()
    total, components = bright_loss(pred, target)
    print(f"   Total: {total:.4f}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")

    print("\n6. EnhancedCombinedLoss (Full):")
    full_loss = EnhancedCombinedLoss()
    total, components = full_loss(pred, target)
    print(f"   Total: {total:.4f}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")

    print("\nAll tests passed!")
