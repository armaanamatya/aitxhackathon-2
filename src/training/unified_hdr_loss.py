#!/usr/bin/env python3
"""
Unified HDR Loss for Real Estate Image Enhancement

Top 0.0001% ML Engineering Solution combining:
1. Charbonnier Loss - Robust L1 alternative, handles outliers better
2. Window/Highlight Aware Loss - Extra supervision on bright regions
3. Color Loss (HSV + LAB) - Preserve color vibrancy and saturation
4. Perceptual Loss (VGG) - Texture and color relationships
5. Edge/Gradient Loss - Preserve sharp edges
6. FFT Frequency Loss - HDR frequency content preservation
7. Color Histogram Loss - Match color distribution

Design principles:
- All components are differentiable
- Balanced weighting to prevent any single loss dominating
- Adaptive window detection (no manual annotation needed)
- Memory efficient (gradient checkpointing compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple
import math


# =============================================================================
# BASE LOSS COMPONENTS
# =============================================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1) - more robust to outliers than L1/L2.
    L = sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class GradientLoss(nn.Module):
    """
    Edge-preserving gradient loss using Sobel operators.
    Ensures sharp edges are preserved in the output.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cast kernels to input dtype/device for mixed precision compatibility
        sobel_x = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sobel_y = self.sobel_y.to(dtype=pred.dtype, device=pred.device)

        # Compute gradients
        pred_gx = F.conv2d(pred, sobel_x, padding=1, groups=3)
        pred_gy = F.conv2d(pred, sobel_y, padding=1, groups=3)
        target_gx = F.conv2d(target, sobel_x, padding=1, groups=3)
        target_gy = F.conv2d(target, sobel_y, padding=1, groups=3)

        # L1 loss on gradients
        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)

        return loss_x + loss_y


class FFTLoss(nn.Module):
    """
    Frequency domain loss using FFT.
    Critical for HDR - preserves high-frequency details and prevents blur.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT on each channel
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # Loss on magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return F.l1_loss(pred_mag, target_mag)


# =============================================================================
# WINDOW/HIGHLIGHT AWARE COMPONENTS
# =============================================================================

class WindowRegionDetector(nn.Module):
    """
    Automatically detects window/highlight regions without manual annotation.
    Uses multiple cues: brightness, local contrast, saturation.
    """
    def __init__(self, brightness_threshold: float = 0.7,
                 contrast_kernel: int = 15):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.contrast_kernel = contrast_kernel

        # Averaging kernel for local contrast
        kernel_size = contrast_kernel
        avg_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        self.register_buffer('avg_kernel', avg_kernel)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Returns attention map [0,1] highlighting window/bright regions"""
        B, C, H, W = img.shape

        # Cast kernel to input dtype/device for mixed precision compatibility
        avg_kernel = self.avg_kernel.to(dtype=img.dtype, device=img.device)

        # 1. Brightness mask (high luminance regions)
        luminance = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        brightness_mask = torch.sigmoid((luminance - self.brightness_threshold) * 10)

        # 2. Local contrast (high contrast edges around windows)
        pad = self.contrast_kernel // 2
        local_mean = F.conv2d(luminance, avg_kernel, padding=pad)
        local_var = F.conv2d((luminance - local_mean) ** 2, avg_kernel, padding=pad)
        contrast_mask = torch.sigmoid(local_var * 50 - 0.5)

        # 3. Saturation mask (windows often have low saturation due to overexposure)
        max_rgb, _ = img.max(dim=1, keepdim=True)
        min_rgb, _ = img.min(dim=1, keepdim=True)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        low_sat_mask = torch.sigmoid((0.3 - saturation) * 10)

        # 4. Combine: regions that are bright OR (high contrast AND low saturation)
        combined = brightness_mask + 0.5 * contrast_mask * low_sat_mask
        combined = torch.clamp(combined, 0, 1)

        # Smooth the mask
        combined = F.avg_pool2d(combined, kernel_size=5, stride=1, padding=2)

        return combined


class WindowAwareLoss(nn.Module):
    """
    Applies higher weight to window/highlight regions.
    """
    def __init__(self, window_weight: float = 3.0, base_loss: str = 'charbonnier'):
        super().__init__()
        self.window_weight = window_weight
        self.detector = WindowRegionDetector()

        if base_loss == 'charbonnier':
            self.base_loss = CharbonnierLoss()
        else:
            self.base_loss = nn.L1Loss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Detect window regions in target (ground truth)
        window_mask = self.detector(target)

        # Compute per-pixel loss
        if isinstance(self.base_loss, CharbonnierLoss):
            diff = pred - target
            pixel_loss = torch.sqrt(diff * diff + 1e-12)
        else:
            pixel_loss = torch.abs(pred - target)

        # Weight: base weight + extra weight for windows
        weight = 1.0 + (self.window_weight - 1.0) * window_mask

        # Weighted average
        weighted_loss = pixel_loss * weight
        return weighted_loss.mean()


# =============================================================================
# COLOR PRESERVATION COMPONENTS
# =============================================================================

class HSVColorLoss(nn.Module):
    """
    Loss in HSV space - explicit saturation and value supervision.
    Critical for preventing dull/washed-out colors.
    """
    def __init__(self, saturation_weight: float = 2.0, value_weight: float = 1.0):
        super().__init__()
        self.sat_weight = saturation_weight
        self.val_weight = value_weight

    def rgb_to_hsv(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert RGB [0,1] to H, S, V channels"""
        max_rgb, argmax = rgb.max(dim=1, keepdim=True)
        min_rgb, _ = rgb.min(dim=1, keepdim=True)
        diff = max_rgb - min_rgb

        # Value
        v = max_rgb

        # Saturation
        s = torch.where(max_rgb > 0, diff / (max_rgb + 1e-8), torch.zeros_like(max_rgb))

        # Hue (simplified)
        h = torch.zeros_like(s)

        return h, s, v

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, pred_s, pred_v = self.rgb_to_hsv(pred)
        _, target_s, target_v = self.rgb_to_hsv(target)

        sat_loss = F.l1_loss(pred_s, target_s)
        val_loss = F.l1_loss(pred_v, target_v)

        return self.sat_weight * sat_loss + self.val_weight * val_loss


class LABColorLoss(nn.Module):
    """
    Loss in LAB color space - perceptually uniform color matching.
    Separates luminance from color for better optimization.
    """
    def __init__(self, l_weight: float = 1.0, ab_weight: float = 2.0):
        super().__init__()
        self.l_weight = l_weight
        self.ab_weight = ab_weight

    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Differentiable RGB to LAB conversion"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # Linearize sRGB
        r = torch.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
        g = torch.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
        b = torch.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)

        # RGB to XYZ
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # Normalize by D65
        x, y, z = x / 0.95047, y / 1.0, z / 1.08883

        # XYZ to LAB
        eps = 0.008856
        kappa = 903.3

        fx = torch.where(x > eps, x ** (1/3), (kappa * x + 16) / 116)
        fy = torch.where(y > eps, y ** (1/3), (kappa * y + 16) / 116)
        fz = torch.where(z > eps, z ** (1/3), (kappa * z + 16) / 116)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_ch = 200 * (fy - fz)

        # Normalize to [0, 1] range for consistent loss scaling
        # L: 0-100 -> 0-1, a/b: -128 to 127 -> 0-1
        L = L / 100.0
        a = (a + 128.0) / 255.0
        b_ch = (b_ch + 128.0) / 255.0

        return torch.cat([L, a, b_ch], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)

        l_loss = F.l1_loss(pred_lab[:, 0:1], target_lab[:, 0:1])
        ab_loss = F.l1_loss(pred_lab[:, 1:3], target_lab[:, 1:3])

        return self.l_weight * l_loss + self.ab_weight * ab_loss


class ColorHistogramLoss(nn.Module):
    """
    Soft histogram matching loss - ensures similar color distribution.
    """
    def __init__(self, bins: int = 64):
        super().__init__()
        self.bins = bins

    def soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)

        # Bin centers
        bin_width = 1.0 / self.bins
        centers = torch.linspace(bin_width/2, 1 - bin_width/2, self.bins, device=x.device)
        centers = centers.view(1, 1, 1, self.bins)

        # Soft assignment
        sigma = bin_width
        x_expanded = x_flat.unsqueeze(-1)
        weights = torch.exp(-((x_expanded - centers) ** 2) / (2 * sigma ** 2))

        hist = weights.sum(dim=2)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

        return hist

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hist = self.soft_histogram(pred)
        target_hist = self.soft_histogram(target)

        # EMD approximation via CDF
        pred_cdf = torch.cumsum(pred_hist, dim=-1)
        target_cdf = torch.cumsum(target_hist, dim=-1)

        return F.l1_loss(pred_cdf, target_cdf)


# =============================================================================
# PERCEPTUAL LOSS (VGG)
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    Captures high-level texture and color relationships.
    """
    def __init__(self, layers: list = None, weights: list = None):
        super().__init__()

        # Default: conv1_2, conv2_2, conv3_4, conv4_4
        self.layers = layers or [3, 8, 17, 26]
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Freeze
        for param in vgg.parameters():
            param.requires_grad = False

        # Build feature extractors
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in sorted(self.layers):
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer+1]))
            prev = layer + 1

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x_pred, x_target = pred, target

        for block, weight in zip(self.blocks, self.weights):
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss += weight * F.l1_loss(x_pred, x_target)

        return loss


# =============================================================================
# UNIFIED HDR LOSS
# =============================================================================

class UnifiedHDRLoss(nn.Module):
    """
    Complete unified loss for HDR real estate image enhancement.

    Combines all components with carefully tuned weights:
    - Charbonnier: Base reconstruction loss
    - Window-aware: Extra attention to bright regions
    - HSV Color: Saturation preservation
    - LAB Color: Perceptually uniform color matching
    - Histogram: Color distribution matching
    - Gradient: Edge preservation
    - FFT: Frequency content preservation
    - Perceptual: High-level texture/color (VGG)
    """
    def __init__(self,
                 # Base loss
                 charbonnier_weight: float = 1.0,
                 # Window/highlight
                 window_weight: float = 0.5,
                 window_multiplier: float = 3.0,
                 # Color
                 hsv_weight: float = 0.3,
                 lab_weight: float = 0.3,
                 histogram_weight: float = 0.2,
                 # Edge/frequency
                 gradient_weight: float = 0.1,
                 fft_weight: float = 0.1,
                 # Perceptual
                 perceptual_weight: float = 0.1,
                 # Device
                 device: str = 'cuda'):
        super().__init__()

        self.weights = {
            'charbonnier': charbonnier_weight,
            'window': window_weight,
            'hsv': hsv_weight,
            'lab': lab_weight,
            'histogram': histogram_weight,
            'gradient': gradient_weight,
            'fft': fft_weight,
            'perceptual': perceptual_weight,
        }

        # Initialize components
        self.charbonnier = CharbonnierLoss()
        self.window_aware = WindowAwareLoss(window_weight=window_multiplier)
        self.hsv_color = HSVColorLoss(saturation_weight=2.0, value_weight=1.0)
        self.lab_color = LABColorLoss(l_weight=1.0, ab_weight=2.0)
        self.histogram = ColorHistogramLoss(bins=64)
        self.gradient = GradientLoss()
        self.fft = FFTLoss()

        # VGG perceptual (only if weight > 0)
        if perceptual_weight > 0:
            self.perceptual = VGGPerceptualLoss().to(device)
        else:
            self.perceptual = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Returns dict with individual losses and total.
        """
        losses = {}

        # 1. Base Charbonnier loss
        losses['charbonnier'] = self.charbonnier(pred, target) * self.weights['charbonnier']

        # 2. Window-aware loss
        losses['window'] = self.window_aware(pred, target) * self.weights['window']

        # 3. HSV color loss
        losses['hsv'] = self.hsv_color(pred, target) * self.weights['hsv']

        # 4. LAB color loss
        losses['lab'] = self.lab_color(pred, target) * self.weights['lab']

        # 5. Histogram loss
        losses['histogram'] = self.histogram(pred, target) * self.weights['histogram']

        # 6. Gradient/edge loss
        losses['gradient'] = self.gradient(pred, target) * self.weights['gradient']

        # 7. FFT frequency loss
        losses['fft'] = self.fft(pred, target) * self.weights['fft']

        # 8. Perceptual loss
        if self.perceptual is not None:
            losses['perceptual'] = self.perceptual(pred, target) * self.weights['perceptual']

        # Total
        losses['total'] = sum(losses.values())

        return losses


def get_unified_hdr_loss(config: str = 'optimal', device: str = 'cuda') -> UnifiedHDRLoss:
    """
    Get pre-configured unified HDR loss.

    Configs:
    - 'optimal': Best balance for real estate HDR (recommended)
    - 'color_focus': Extra emphasis on color preservation
    - 'window_focus': Extra emphasis on window/highlight regions
    - 'fast': No VGG perceptual (faster, less memory)
    """
    configs = {
        'optimal': {
            'charbonnier_weight': 1.0,
            'window_weight': 0.5,
            'window_multiplier': 3.0,
            'hsv_weight': 0.3,
            'lab_weight': 0.3,
            'histogram_weight': 0.2,
            'gradient_weight': 0.1,
            'fft_weight': 0.1,
            'perceptual_weight': 0.1,
        },
        'color_focus': {
            'charbonnier_weight': 0.8,
            'window_weight': 0.3,
            'window_multiplier': 2.0,
            'hsv_weight': 0.5,
            'lab_weight': 0.5,
            'histogram_weight': 0.4,
            'gradient_weight': 0.1,
            'fft_weight': 0.05,
            'perceptual_weight': 0.2,
        },
        'window_focus': {
            'charbonnier_weight': 0.8,
            'window_weight': 0.8,
            'window_multiplier': 5.0,
            'hsv_weight': 0.2,
            'lab_weight': 0.2,
            'histogram_weight': 0.1,
            'gradient_weight': 0.15,
            'fft_weight': 0.15,
            'perceptual_weight': 0.1,
        },
        'fast': {
            'charbonnier_weight': 1.0,
            'window_weight': 0.5,
            'window_multiplier': 3.0,
            'hsv_weight': 0.3,
            'lab_weight': 0.3,
            'histogram_weight': 0.2,
            'gradient_weight': 0.1,
            'fft_weight': 0.1,
            'perceptual_weight': 0.0,  # No VGG
        },
    }

    return UnifiedHDRLoss(**configs[config], device=device)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("UNIFIED HDR LOSS - Component Test")
    print("=" * 70)

    device = 'cpu'

    # Create dummy tensors
    pred = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)

    # Test unified loss (fast config for CPU)
    print("\nTesting Unified HDR Loss (fast config)...")
    loss_fn = get_unified_hdr_loss('fast', device='cpu')
    losses = loss_fn(pred, target)

    print("\nLoss components:")
    for name, value in losses.items():
        print(f"  {name:15s}: {value.item():.4f}")

    print("\n✓ All components working!")
    print("=" * 70)
