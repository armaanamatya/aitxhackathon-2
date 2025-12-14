"""
Comprehensive Color + Perceptual + SSIM Loss for HDR Enhancement

Problem: Model doesn't match colors accurately across the ENTIRE image.

Solution: Multi-component loss targeting global color accuracy:
1. L1 - Pixel accuracy
2. Perceptual (VGG) - Feature-level similarity
3. SSIM - Structural similarity
4. Color Loss - Hue, Saturation, Value matching globally
5. Lab Color Loss - Perceptually uniform color space

Author: Top 0.0001% MLE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import torchvision.models as models


# =============================================================================
# Perceptual Loss (VGG-based)
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.

    Compares feature representations instead of raw pixels.
    Better captures texture, color relationships, and semantic content.
    """

    def __init__(self, layers: list = None, weights: list = None):
        super().__init__()

        # Default layers: early (color/texture) + mid (patterns) + late (semantic)
        if layers is None:
            layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]

        self.layers = layers
        self.weights = weights

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features

        # Freeze VGG weights
        for param in vgg.parameters():
            param.requires_grad = False

        # Split into layer groups
        self.slices = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer+1]))
            prev = layer + 1

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.normalize(pred)
        target = self.normalize(target)

        loss = 0.0
        pred_feat = pred
        target_feat = target

        for i, slice in enumerate(self.slices):
            pred_feat = slice(pred_feat)
            target_feat = slice(target_feat)
            loss += self.weights[i] * F.l1_loss(pred_feat, target_feat)

        return loss / len(self.slices)


# =============================================================================
# SSIM Loss
# =============================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.

    Measures structural similarity based on:
    - Luminance
    - Contrast
    - Structure
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Create Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # 2D window
        window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.size(1)
        window = self.window.repeat(channels, 1, 1, 1)

        padding = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
        mu_target = F.conv2d(target, window, padding=padding, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=padding, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=padding, groups=channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target

        # Clamp to avoid negative values
        sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0)
        sigma_target_sq = torch.clamp(sigma_target_sq, min=0)

        ssim = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
               ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1 - ssim.mean()


# =============================================================================
# Global Color Loss (HSV-based)
# =============================================================================

class GlobalColorLoss(nn.Module):
    """
    Global color matching loss in HSV space.

    Separately matches:
    - Hue (color type): red, green, blue, etc.
    - Saturation (colorfulness): gray vs vibrant
    - Value (brightness): dark vs light
    """

    def __init__(self, hue_weight: float = 1.0, sat_weight: float = 1.0, val_weight: float = 0.5):
        super().__init__()
        self.hue_weight = hue_weight
        self.sat_weight = sat_weight
        self.val_weight = val_weight

    def rgb_to_hsv(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert RGB to HSV."""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        max_c, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        diff = max_c - min_c + 1e-7

        # Value
        v = max_c

        # Saturation
        s = diff / (max_c + 1e-7)

        # Hue
        hue = torch.zeros_like(max_c)

        mask_r = (max_idx == 0).float()
        mask_g = (max_idx == 1).float()
        mask_b = (max_idx == 2).float()

        hue = hue + mask_r * (((g - b) / diff) % 6)
        hue = hue + mask_g * (((b - r) / diff) + 2)
        hue = hue + mask_b * (((r - g) / diff) + 4)
        hue = hue / 6.0

        return hue, s, v

    def circular_distance(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Circular distance for hue (wraps at 1.0)."""
        diff = torch.abs(h1 - h2)
        return torch.min(diff, 1.0 - diff)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        pred_h, pred_s, pred_v = self.rgb_to_hsv(pred)
        target_h, target_s, target_v = self.rgb_to_hsv(target)

        # Only compute hue loss where there's actual color (saturation > threshold)
        color_mask = (target_s > 0.1).float()

        # Hue loss (circular)
        hue_diff = self.circular_distance(pred_h, target_h)
        hue_loss = (hue_diff * color_mask).sum() / (color_mask.sum() + 1e-7)

        # Saturation loss
        sat_loss = F.l1_loss(pred_s, target_s)

        # Value loss (less important, L1 handles this)
        val_loss = F.l1_loss(pred_v, target_v)

        total = self.hue_weight * hue_loss + self.sat_weight * sat_loss + self.val_weight * val_loss

        components = {
            'hue': hue_loss.item(),
            'saturation': sat_loss.item(),
            'value': val_loss.item(),
        }

        return total, components


# =============================================================================
# Lab Color Loss (Perceptually Uniform)
# =============================================================================

class LabColorLoss(nn.Module):
    """
    Color loss in CIE Lab space.

    Lab is perceptually uniform - equal distances = equal perceived differences.
    Better for matching colors as humans see them.
    """

    def __init__(self):
        super().__init__()

        # RGB to XYZ matrix (sRGB, D65)
        self.register_buffer('rgb_to_xyz', torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]).T)

        # D65 white point
        self.register_buffer('white', torch.tensor([0.95047, 1.0, 1.08883]))

    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB [0,1] to Lab."""
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

        # XYZ to Lab
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)

        # Normalize Lab to similar scale as other losses
        # L: 0-100, a/b: -128 to 127 -> normalize to ~0-1 range
        pred_lab_norm = pred_lab.clone()
        target_lab_norm = target_lab.clone()
        pred_lab_norm[:, 0:1] = pred_lab[:, 0:1] / 100.0  # L
        pred_lab_norm[:, 1:3] = pred_lab[:, 1:3] / 128.0  # a, b
        target_lab_norm[:, 0:1] = target_lab[:, 0:1] / 100.0
        target_lab_norm[:, 1:3] = target_lab[:, 1:3] / 128.0

        # L1 loss on normalized Lab
        lab_loss = F.l1_loss(pred_lab_norm, target_lab_norm)

        return lab_loss


# =============================================================================
# Complete Loss
# =============================================================================

class ColorPerceptualLoss(nn.Module):
    """
    Complete loss for accurate color reproduction.

    Components:
    - L1 (1.0): Pixel accuracy
    - Perceptual (0.1): VGG feature similarity
    - SSIM (0.1): Structural similarity
    - Color (0.3): HSV color matching (Hue + Saturation + Value)
    - Lab (0.2): Perceptually uniform color matching
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.1,
        color_weight: float = 0.3,
        lab_weight: float = 0.2,
        use_perceptual: bool = True,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.lab_weight = lab_weight
        self.use_perceptual = use_perceptual

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.color_loss = GlobalColorLoss()
        self.lab_loss = LabColorLoss()

        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # L1
        l1 = self.l1_loss(pred, target)

        # SSIM
        ssim = self.ssim_loss(pred, target)

        # Color (HSV)
        color, color_components = self.color_loss(pred, target)

        # Lab
        lab = self.lab_loss(pred, target)

        # Perceptual
        if self.use_perceptual:
            perceptual = self.perceptual_loss(pred, target)
        else:
            perceptual = torch.tensor(0.0, device=pred.device)

        # Total
        total = (
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.ssim_weight * ssim +
            self.color_weight * color +
            self.lab_weight * lab
        )

        components = {
            'l1': l1.item(),
            'perceptual': perceptual.item() if self.use_perceptual else 0.0,
            'ssim': ssim.item(),
            'color': color.item(),
            'lab': lab.item(),
            'color_hue': color_components['hue'],
            'color_sat': color_components['saturation'],
            'total': total.item(),
        }

        return total, components


# Test
if __name__ == "__main__":
    print("Testing Color + Perceptual + SSIM Loss...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)

    # Test without perceptual (faster)
    print("\n1. Testing without VGG (CPU-friendly):")
    loss_fn = ColorPerceptualLoss(use_perceptual=False).to(device)
    total, components = loss_fn(pred, target)
    print(f"   Total: {total.item():.4f}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")

    # Test with perceptual
    print("\n2. Testing with VGG Perceptual Loss:")
    loss_fn = ColorPerceptualLoss(use_perceptual=True).to(device)
    total, components = loss_fn(pred, target)
    print(f"   Total: {total.item():.4f}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")

    print("\nAll tests passed!")
