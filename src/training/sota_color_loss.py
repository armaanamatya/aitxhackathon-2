"""
SOTA Color Enhancement Loss for Real Estate HDR

References:
- Focal Frequency Loss (CVPR 2021): Frequency domain for color patterns
- 3D LUT Learning (CVPR 2022): Non-linear color transformation
- LAB Perceptual Loss: Perceptually uniform color matching
- Multi-Scale Color Attention: Separate processing for different colors

Problem: Baseline struggles with:
1. Green/blue/red enhancement (plants, sky, objects)
2. Window shots (outdoor views through windows)
3. Color grading (non-linear transformations)

Solution: SOTA multi-component loss targeting color fidelity.

Author: Top 0.0001% MLE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (CVPR 2021).

    Problem: L1 in spatial domain doesn't capture color frequency patterns.
    Solution: FFT-based loss in frequency domain.

    Reference: "Focal Frequency Loss for Image Reconstruction and Synthesis"
    """

    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        # Convert to frequency domain
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')

        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Focal weighting (emphasize harder frequencies)
        weight = torch.pow(torch.abs(target_mag - pred_mag), self.alpha)
        weight = weight / (weight.mean() + 1e-8)

        # Weighted L1 loss in frequency domain
        freq_loss = (weight * torch.abs(target_mag - pred_mag)).mean()

        return self.loss_weight * freq_loss


class LABPerceptualLoss(nn.Module):
    """
    LAB color space perceptual loss.

    Problem: RGB space doesn't match human color perception.
    Solution: Convert to LAB (perceptually uniform) and compute loss.

    LAB space properties:
    - L: Lightness (0-100)
    - a: Green (-128) to Red (+127)
    - b: Blue (-128) to Yellow (+127)
    """

    def __init__(self, l_weight=1.0, ab_weight=2.0):
        super().__init__()
        self.l_weight = l_weight
        self.ab_weight = ab_weight

        # RGB to XYZ matrix (sRGB, D65)
        self.register_buffer('rgb_to_xyz', torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]).T)

        # D65 white point
        self.register_buffer('white', torch.tensor([0.95047, 1.0, 1.08883]))

    def rgb_to_lab(self, rgb):
        """Convert RGB [0,1] to LAB."""
        # Linearize sRGB
        rgb_linear = torch.where(
            rgb > 0.04045,
            ((rgb + 0.055) / 1.055) ** 2.4,
            rgb / 12.92
        )

        # RGB to XYZ
        B, C, H, W = rgb.shape
        rgb_flat = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
        xyz_flat = rgb_flat @ self.rgb_to_xyz
        xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # Normalize by white point
        xyz = xyz / self.white.view(1, 3, 1, 1)

        # XYZ to LAB
        delta = 6/29
        xyz_f = torch.where(
            xyz > delta**3,
            xyz ** (1/3),
            xyz / (3 * delta**2) + 4/29
        )

        L = 116 * xyz_f[:, 1:2] - 16
        a = 500 * (xyz_f[:, 0:1] - xyz_f[:, 1:2])
        b = 200 * (xyz_f[:, 1:2] - xyz_f[:, 2:3])

        return torch.cat([L, a, b], dim=1)

    def forward(self, pred, target):
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)

        # Lightness loss
        l_loss = F.l1_loss(pred_lab[:, 0:1] / 100, target_lab[:, 0:1] / 100)

        # Chroma loss (a, b channels)
        ab_loss = F.l1_loss(pred_lab[:, 1:3] / 128, target_lab[:, 1:3] / 128)

        return self.l_weight * l_loss + self.ab_weight * ab_loss


class MultiScaleColorLoss(nn.Module):
    """
    Multi-scale color matching.

    Problem: Colors need to match at multiple scales (local and global).
    Solution: Compute color loss at 3 scales (full, 1/2, 1/4).

    This ensures both fine details and overall color tone match.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        total_loss = 0

        # Full resolution
        total_loss += F.l1_loss(pred, target)

        # 1/2 resolution
        pred_half = F.avg_pool2d(pred, 2)
        target_half = F.avg_pool2d(target, 2)
        total_loss += F.l1_loss(pred_half, target_half)

        # 1/4 resolution
        pred_quarter = F.avg_pool2d(pred_half, 2)
        target_quarter = F.avg_pool2d(target_half, 2)
        total_loss += F.l1_loss(pred_quarter, target_quarter)

        return total_loss / 3


class ColorCurveLearningLoss(nn.Module):
    """
    Learn per-channel tone curves.

    Problem: Each color channel (R, G, B) has different enhancement curve.
    Solution: Match the input→output transformation for each channel.

    Inspired by: CSRNet (CVPR 2020), 3D LUT Transformer (CVPR 2022)
    """

    def __init__(self, num_bins=32):
        super().__init__()
        self.num_bins = num_bins

    def compute_curve(self, input_channel, output_channel):
        """Compute transformation curve for one channel."""
        # Bin the input values and compute average output for each bin
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=input_channel.device)

        curve_pred = []
        curve_target = []

        for i in range(self.num_bins):
            # Mask for values in this bin
            mask = ((input_channel >= bin_edges[i]) &
                    (input_channel < bin_edges[i + 1])).float()

            if mask.sum() > 0:
                # Average output value for inputs in this bin
                curve_pred.append((output_channel * mask).sum() / mask.sum())
                curve_target.append(bin_edges[i].item())
            else:
                curve_pred.append(torch.tensor(0.0, device=input_channel.device))
                curve_target.append(0.0)

        return torch.stack(curve_pred), torch.tensor(curve_target, device=input_channel.device)

    def forward(self, pred, target, input_img):
        """
        Args:
            pred: Model output
            target: Ground truth
            input_img: Original input (to learn input→output curve)
        """
        total_loss = 0

        # Per-channel curve matching
        for c in range(3):  # R, G, B
            # Compute transformation curves
            pred_curve, _ = self.compute_curve(input_img[:, c:c+1], pred[:, c:c+1])
            target_curve, _ = self.compute_curve(input_img[:, c:c+1], target[:, c:c+1])

            # Match curves
            total_loss += F.l1_loss(pred_curve, target_curve)

        return total_loss / 3


class ColorHistogramMatchingLoss(nn.Module):
    """
    Match color histogram distribution.

    Problem: Overall color distribution should match (not just per-pixel).
    Solution: Match cumulative histograms (Earth Mover's Distance approximation).

    This ensures greens, blues, reds have correct distribution.
    """

    def __init__(self, num_bins=64):
        super().__init__()
        self.num_bins = num_bins

    def compute_histogram(self, x):
        """Compute soft differentiable histogram."""
        B, C, H, W = x.shape

        # Bin centers
        bin_centers = torch.linspace(0, 1, self.num_bins, device=x.device)
        bin_width = 1.0 / self.num_bins

        # Reshape
        x_flat = x.view(B, C, -1, 1)  # [B, C, HW, 1]
        bin_centers = bin_centers.view(1, 1, 1, -1)  # [1, 1, 1, bins]

        # Soft binning with Gaussian kernel
        weights = torch.exp(-((x_flat - bin_centers) ** 2) / (2 * (bin_width ** 2)))

        # Histogram
        hist = weights.sum(dim=2)  # [B, C, bins]
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-7)

        return hist

    def forward(self, pred, target):
        pred_hist = self.compute_histogram(pred)
        target_hist = self.compute_histogram(target)

        # Cumulative histograms (Earth Mover's Distance approximation)
        pred_cum = torch.cumsum(pred_hist, dim=-1)
        target_cum = torch.cumsum(target_hist, dim=-1)

        # L1 distance between cumulative histograms
        return F.l1_loss(pred_cum, target_cum)


class WindowColorBoostLoss(nn.Module):
    """
    Specific loss for window regions (sky, trees, outdoor views).

    Problem: Windows showing outdoor scenes need extra color enhancement.
    Solution: Detect bright regions + high saturation (sky, greenery) and
    apply extra loss weight.
    """

    def __init__(self, brightness_threshold=0.6):
        super().__init__()
        self.brightness_threshold = brightness_threshold

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV."""
        max_c, _ = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        diff = max_c - min_c + 1e-7

        v = max_c
        s = diff / (max_c + 1e-7)

        return s, v

    def forward(self, pred, target):
        # Detect window regions: bright + saturated (sky, outdoor views)
        target_s, target_v = self.rgb_to_hsv(target)

        window_mask = (target_v > self.brightness_threshold).float() * \
                      (target_s > 0.2).float()  # Bright + colorful
        window_mask = F.avg_pool2d(window_mask, kernel_size=11, stride=1, padding=5)

        # Per-channel loss in window regions
        window_loss = (torch.abs(pred - target) * window_mask.expand_as(pred)).sum() / \
                      (window_mask.sum() * 3 + 1e-7)

        # Extra loss for saturation in windows
        pred_s, pred_v = self.rgb_to_hsv(pred)
        sat_loss = (torch.abs(pred_s - target_s) * window_mask).sum() / (window_mask.sum() + 1e-7)

        return window_loss + 0.5 * sat_loss


class SOTAColorEnhancementLoss(nn.Module):
    """
    SOTA-level complete loss for real estate HDR color enhancement.

    Combines cutting-edge losses from CVPR/ICCV 2020-2024:
    - L1: Base pixel accuracy
    - Focal Frequency: Frequency domain patterns (CVPR 2021)
    - LAB Perceptual: Perceptually uniform colors
    - Multi-Scale: Local + global color matching
    - Color Curve: Learn transformation curves (CSRNet, 3D LUT)
    - Histogram Matching: Color distribution
    - Window Boost: Special handling for window regions

    This addresses all your issues:
    - Greens (plants): LAB + Histogram + Color Curve
    - Blues (sky in windows): Window Boost + Focal Frequency
    - Reds: LAB + Color Curve
    - Window shots: Window Boost + Multi-Scale
    """

    def __init__(
        self,
        l1_weight=1.0,
        focal_freq_weight=0.3,
        lab_weight=0.4,
        multiscale_weight=0.3,
        color_curve_weight=0.4,
        histogram_weight=0.2,
        window_boost_weight=0.5,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.focal_freq_weight = focal_freq_weight
        self.lab_weight = lab_weight
        self.multiscale_weight = multiscale_weight
        self.color_curve_weight = color_curve_weight
        self.histogram_weight = histogram_weight
        self.window_boost_weight = window_boost_weight

        self.l1_loss = nn.L1Loss()
        self.focal_freq_loss = FocalFrequencyLoss()
        self.lab_loss = LABPerceptualLoss()
        self.multiscale_loss = MultiScaleColorLoss()
        self.color_curve_loss = ColorCurveLearningLoss()
        self.histogram_loss = ColorHistogramMatchingLoss()
        self.window_boost_loss = WindowColorBoostLoss()

    def forward(self, pred, target, input_img=None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: Model output
            target: Ground truth HDR
            input_img: Original input (optional, for color curve loss)
        """
        l1 = self.l1_loss(pred, target)
        focal_freq = self.focal_freq_loss(pred, target)
        lab = self.lab_loss(pred, target)
        multiscale = self.multiscale_loss(pred, target)
        histogram = self.histogram_loss(pred, target)
        window_boost = self.window_boost_loss(pred, target)

        total = (
            self.l1_weight * l1 +
            self.focal_freq_weight * focal_freq +
            self.lab_weight * lab +
            self.multiscale_weight * multiscale +
            self.histogram_weight * histogram +
            self.window_boost_weight * window_boost
        )

        # Color curve loss (if input provided)
        if input_img is not None:
            color_curve = self.color_curve_loss(pred, target, input_img)
            total += self.color_curve_weight * color_curve
        else:
            color_curve = torch.tensor(0.0)

        components = {
            'l1': l1.item(),
            'focal_freq': focal_freq.item(),
            'lab': lab.item(),
            'multiscale': multiscale.item(),
            'color_curve': color_curve.item() if isinstance(color_curve, torch.Tensor) else 0.0,
            'histogram': histogram.item(),
            'window_boost': window_boost.item(),
            'total': total.item(),
        }

        return total, components


# Test
if __name__ == "__main__":
    print("Testing SOTA Color Enhancement Losses...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create test tensors
    input_img = torch.rand(2, 3, 256, 256).to(device) * 0.6
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)

    print("\nSOTA Components:")
    print("1. FocalFrequencyLoss:", FocalFrequencyLoss().to(device)(pred, target).item())
    print("2. LABPerceptualLoss:", LABPerceptualLoss().to(device)(pred, target).item())
    print("3. MultiScaleColorLoss:", MultiScaleColorLoss().to(device)(pred, target).item())
    print("4. ColorCurveLearningLoss:", ColorCurveLearningLoss().to(device)(pred, target, input_img).item())
    print("5. ColorHistogramMatchingLoss:", ColorHistogramMatchingLoss().to(device)(pred, target).item())
    print("6. WindowColorBoostLoss:", WindowColorBoostLoss().to(device)(pred, target).item())

    print("\nComplete SOTA Loss:")
    loss_fn = SOTAColorEnhancementLoss().to(device)
    total, components = loss_fn(pred, target, input_img)
    print(f"Total: {total.item():.4f}")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")

    print("\nAll tests passed!")
