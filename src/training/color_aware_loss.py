#!/usr/bin/env python3
"""
Color-Aware Loss Functions for HDR Real Estate Enhancement

Addresses the color desaturation problem in neural network outputs by adding
explicit color supervision during training.

Key components:
1. Perceptual Loss (VGG) - preserves texture and color relationships
2. Color Histogram Loss - matches color distribution
3. HSV Saturation Loss - explicit saturation supervision
4. LAB Color Loss - perceptually uniform color matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Dict


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    Better at preserving colors and textures than pixel losses.
    """
    def __init__(self, layers: List[int] = [3, 8, 17, 26], weights: List[float] = None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.layers = layers
        self.weights = weights or [1.0] * len(layers)

        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False

        # Split VGG into blocks
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in sorted(layers):
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer+1]))
            prev = layer + 1

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet stats
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x_pred, x_target = pred, target

        for block, weight in zip(self.blocks, self.weights):
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss += weight * F.l1_loss(x_pred, x_target)

        return loss


class ColorHistogramLoss(nn.Module):
    """
    Loss based on color histogram matching.
    Encourages output to have similar color distribution as target.
    """
    def __init__(self, bins: int = 64):
        super().__init__()
        self.bins = bins

    def soft_histogram(self, x: torch.Tensor, bins: int, min_val: float = 0.0,
                       max_val: float = 1.0) -> torch.Tensor:
        """Differentiable soft histogram using kernel density estimation"""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        # Bin centers
        bin_width = (max_val - min_val) / bins
        centers = torch.linspace(min_val + bin_width/2, max_val - bin_width/2, bins)
        centers = centers.to(x.device).view(1, 1, 1, bins)  # [1, 1, 1, bins]

        # Soft assignment using Gaussian kernel
        sigma = bin_width
        x_expanded = x_flat.unsqueeze(-1)  # [B, C, H*W, 1]
        weights = torch.exp(-((x_expanded - centers) ** 2) / (2 * sigma ** 2))

        # Normalize to get histogram
        hist = weights.sum(dim=2)  # [B, C, bins]
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

        return hist

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hist = self.soft_histogram(pred, self.bins)
        target_hist = self.soft_histogram(target, self.bins)

        # Earth Mover's Distance approximation (cumulative histogram difference)
        pred_cdf = torch.cumsum(pred_hist, dim=-1)
        target_cdf = torch.cumsum(target_hist, dim=-1)

        return F.l1_loss(pred_cdf, target_cdf)


class HSVSaturationLoss(nn.Module):
    """
    Explicit supervision on saturation in HSV space.
    Prevents the model from producing desaturated outputs.
    """
    def __init__(self):
        super().__init__()

    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB [0,1] to HSV [0,1]"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        max_rgb, _ = rgb.max(dim=1, keepdim=True)
        min_rgb, _ = rgb.min(dim=1, keepdim=True)
        diff = max_rgb - min_rgb

        # Saturation
        s = torch.where(max_rgb > 0, diff / (max_rgb + 1e-8), torch.zeros_like(max_rgb))

        # Value
        v = max_rgb

        # Hue (simplified - we mainly care about S and V)
        h = torch.zeros_like(s)

        return torch.cat([h, s, v], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hsv = self.rgb_to_hsv(pred)
        target_hsv = self.rgb_to_hsv(target)

        # Saturation loss (most important for color vibrancy)
        sat_loss = F.l1_loss(pred_hsv[:, 1:2], target_hsv[:, 1:2])

        # Value loss (brightness)
        val_loss = F.l1_loss(pred_hsv[:, 2:3], target_hsv[:, 2:3])

        return sat_loss + 0.5 * val_loss


class LABColorLoss(nn.Module):
    """
    Loss in LAB color space (perceptually uniform).
    Separately weights L (luminance) and a*b* (color) channels.
    """
    def __init__(self, l_weight: float = 1.0, ab_weight: float = 2.0):
        super().__init__()
        self.l_weight = l_weight
        self.ab_weight = ab_weight

    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Approximate RGB to LAB conversion (differentiable)"""
        # RGB to XYZ (D65 illuminant)
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # Linearize (approximate gamma correction)
        r = torch.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
        g = torch.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
        b = torch.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)

        # RGB to XYZ matrix (sRGB D65)
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # Normalize by D65 white point
        x = x / 0.95047
        y = y / 1.0
        z = z / 1.08883

        # XYZ to LAB
        epsilon = 0.008856
        kappa = 903.3

        fx = torch.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
        fy = torch.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
        fz = torch.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_ch = 200 * (fy - fz)

        return torch.cat([L, a, b_ch], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)

        # Luminance loss
        l_loss = F.l1_loss(pred_lab[:, 0:1], target_lab[:, 0:1])

        # Color loss (a* and b* channels)
        ab_loss = F.l1_loss(pred_lab[:, 1:3], target_lab[:, 1:3])

        return self.l_weight * l_loss + self.ab_weight * ab_loss


class ColorAwareLoss(nn.Module):
    """
    Combined color-aware loss for HDR real estate enhancement.

    Addresses undersaturation by combining:
    - L1 loss for pixel accuracy
    - Perceptual loss for texture/color relationships
    - Color histogram loss for distribution matching
    - HSV saturation loss for explicit color supervision
    - LAB loss for perceptually uniform color matching
    """
    def __init__(self,
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 histogram_weight: float = 0.5,
                 saturation_weight: float = 0.3,
                 lab_weight: float = 0.2,
                 device: str = 'cuda'):
        super().__init__()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.histogram_weight = histogram_weight
        self.saturation_weight = saturation_weight
        self.lab_weight = lab_weight

        # Initialize components
        self.perceptual = VGGPerceptualLoss().to(device) if perceptual_weight > 0 else None
        self.histogram = ColorHistogramLoss() if histogram_weight > 0 else None
        self.saturation = HSVSaturationLoss() if saturation_weight > 0 else None
        self.lab = LABColorLoss() if lab_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}

        # L1 loss
        losses['l1'] = F.l1_loss(pred, target) * self.l1_weight

        # Perceptual loss
        if self.perceptual is not None:
            losses['perceptual'] = self.perceptual(pred, target) * self.perceptual_weight

        # Histogram loss
        if self.histogram is not None:
            losses['histogram'] = self.histogram(pred, target) * self.histogram_weight

        # Saturation loss
        if self.saturation is not None:
            losses['saturation'] = self.saturation(pred, target) * self.saturation_weight

        # LAB loss
        if self.lab is not None:
            losses['lab'] = self.lab(pred, target) * self.lab_weight

        # Total
        losses['total'] = sum(losses.values())

        return losses


def get_color_aware_loss(config: str = 'balanced', device: str = 'cuda') -> ColorAwareLoss:
    """
    Get pre-configured color-aware loss.

    Configs:
    - 'balanced': Good default for most cases
    - 'color_focus': Stronger color supervision (for very dull outputs)
    - 'perceptual': Heavy perceptual loss (for texture preservation)
    - 'fast': No VGG (faster training, slightly less quality)
    """
    configs = {
        'balanced': {
            'l1_weight': 1.0,
            'perceptual_weight': 0.1,
            'histogram_weight': 0.5,
            'saturation_weight': 0.3,
            'lab_weight': 0.2,
        },
        'color_focus': {
            'l1_weight': 0.5,
            'perceptual_weight': 0.1,
            'histogram_weight': 1.0,
            'saturation_weight': 0.5,
            'lab_weight': 0.5,
        },
        'perceptual': {
            'l1_weight': 0.5,
            'perceptual_weight': 0.5,
            'histogram_weight': 0.3,
            'saturation_weight': 0.2,
            'lab_weight': 0.2,
        },
        'fast': {
            'l1_weight': 1.0,
            'perceptual_weight': 0.0,
            'histogram_weight': 0.5,
            'saturation_weight': 0.5,
            'lab_weight': 0.3,
        },
    }

    return ColorAwareLoss(**configs[config], device=device)


if __name__ == '__main__':
    # Test the loss functions
    print("Testing Color-Aware Loss Components")
    print("=" * 50)

    device = 'cpu'  # Use CPU for testing

    # Create dummy tensors
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    # Test individual components
    print("\n1. HSV Saturation Loss:")
    sat_loss = HSVSaturationLoss()
    print(f"   Loss: {sat_loss(pred, target).item():.4f}")

    print("\n2. LAB Color Loss:")
    lab_loss = LABColorLoss()
    print(f"   Loss: {lab_loss(pred, target).item():.4f}")

    print("\n3. Color Histogram Loss:")
    hist_loss = ColorHistogramLoss()
    print(f"   Loss: {hist_loss(pred, target).item():.4f}")

    print("\n4. Combined Color-Aware Loss (fast config):")
    combined = get_color_aware_loss('fast', device='cpu')
    losses = combined(pred, target)
    for k, v in losses.items():
        print(f"   {k}: {v.item():.4f}")

    print("\n✓ All loss functions working!")
