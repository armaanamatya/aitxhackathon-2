#!/usr/bin/env python3
"""
Window-Aware Loss for Real Estate HDR Enhancement
==================================================
Top 0.0001% MLE implementation for handling window/high-dynamic-range regions.

The Problem:
- Windows show bright outdoor scenes vs dark interiors
- Standard L1 loss treats all pixels equally
- Model under-optimizes for difficult window regions

The Solution:
- Detect window/HDR regions automatically (no manual annotation)
- Apply higher loss weight to these regions
- Use multi-scale gradient loss for edge preservation
- Combine with perceptual loss for realistic details

Detection Methods:
1. Brightness-based: High luminance regions (windows letting in light)
2. Contrast-based: High local contrast (window edges)
3. Saturation-based: Low saturation in overexposed areas
4. Combined adaptive weighting

Author: Top MLE
Date: 2025-12-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class WindowRegionDetector(nn.Module):
    """
    Automatic window/HDR region detection without manual annotation.

    Uses multiple cues:
    1. High brightness (overexposed windows)
    2. High local contrast (window edges/frames)
    3. Brightness difference between input and target (areas needing most correction)
    """

    def __init__(self,
                 brightness_threshold: float = 0.7,
                 contrast_kernel_size: int = 15,
                 blend_sigma: float = 21.0,
                 device: str = 'cuda'):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.contrast_kernel_size = contrast_kernel_size
        self.blend_sigma = blend_sigma

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Gaussian kernel for smoothing masks
        kernel_size = int(blend_sigma * 2) | 1  # Ensure odd
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * blend_sigma**2))
        gaussian_2d = gaussian_1d.view(-1, 1) @ gaussian_1d.view(1, -1)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        self.register_buffer('gaussian_kernel', gaussian_2d.view(1, 1, kernel_size, kernel_size))
        self.gaussian_padding = kernel_size // 2

    def rgb_to_luminance(self, img: torch.Tensor) -> torch.Tensor:
        """Convert RGB to luminance (perceived brightness)"""
        # ITU-R BT.709 coefficients
        return 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]

    def compute_local_contrast(self, luminance: torch.Tensor) -> torch.Tensor:
        """Compute local contrast using standard deviation in local patches"""
        # Local mean
        kernel_size = self.contrast_kernel_size
        padding = kernel_size // 2

        # Use average pooling for local mean
        local_mean = F.avg_pool2d(luminance, kernel_size, stride=1, padding=padding)

        # Local variance = E[X^2] - E[X]^2
        local_sq_mean = F.avg_pool2d(luminance ** 2, kernel_size, stride=1, padding=padding)
        local_var = torch.clamp(local_sq_mean - local_mean ** 2, min=1e-6)
        local_std = torch.sqrt(local_var)

        return local_std

    def compute_edges(self, luminance: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters"""
        # Apply Sobel filters
        grad_x = F.conv2d(luminance, self.sobel_x, padding=1)
        grad_y = F.conv2d(luminance, self.sobel_y, padding=1)

        # Edge magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return edges

    def smooth_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to create soft mask boundaries"""
        return F.conv2d(mask, self.gaussian_kernel, padding=self.gaussian_padding)

    def forward(self,
                input_img: torch.Tensor,
                target_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Detect window/HDR regions and return attention mask.

        Args:
            input_img: Input image tensor [B, 3, H, W] in [0, 1]
            target_img: Target image tensor [B, 3, H, W] in [0, 1] (optional)

        Returns:
            mask: Attention mask [B, 1, H, W] with higher values for window regions
        """
        # Get luminance
        lum_input = self.rgb_to_luminance(input_img)

        # 1. Brightness-based detection (overexposed regions)
        bright_mask = torch.sigmoid((lum_input - self.brightness_threshold) * 10)

        # 2. High contrast regions (window edges)
        contrast = self.compute_local_contrast(lum_input)
        contrast_norm = contrast / (contrast.max() + 1e-6)
        contrast_mask = torch.sigmoid((contrast_norm - 0.1) * 10)

        # 3. Edge regions (window frames)
        edges = self.compute_edges(lum_input)
        edges_norm = edges / (edges.max() + 1e-6)
        edge_mask = torch.sigmoid((edges_norm - 0.1) * 10)

        # 4. Correction-needed regions (difference between input and target)
        if target_img is not None:
            lum_target = self.rgb_to_luminance(target_img)
            correction_needed = torch.abs(lum_input - lum_target)
            correction_mask = correction_needed / (correction_needed.max() + 1e-6)
        else:
            correction_mask = torch.zeros_like(bright_mask)

        # Combine masks with learned-like weighting
        # High brightness OR (high contrast AND edges) OR needs correction
        combined = torch.max(
            bright_mask * 0.4 + contrast_mask * 0.3 + edge_mask * 0.2 + correction_mask * 0.3,
            bright_mask  # Ensure very bright areas are always weighted
        )

        # Smooth the mask for soft boundaries
        smooth_combined = self.smooth_mask(combined)

        # Normalize to [0, 1] range
        smooth_combined = smooth_combined / (smooth_combined.max() + 1e-6)

        # Scale to desired weight range [1, max_weight]
        # Base weight is 1, window regions get higher weight
        return smooth_combined


class CharbonnierLoss(nn.Module):
    """Charbonnier loss - smooth L1, more robust to outliers"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff ** 2 + self.eps)


class GradientLoss(nn.Module):
    """
    Multi-scale gradient loss for edge preservation.
    Critical for window frames and edges.
    """
    def __init__(self):
        super().__init__()
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute gradients for each channel
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)

        # L1 loss on gradients
        loss_x = torch.abs(pred_grad_x - target_grad_x).mean()
        loss_y = torch.abs(pred_grad_y - target_grad_y).mean()

        return loss_x + loss_y


class FFTLoss(nn.Module):
    """
    Frequency domain loss - helps with fine details and texture.
    Particularly useful for window reflections and outdoor details.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT of prediction and target
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # L1 loss on magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return torch.mean(torch.abs(pred_mag - target_mag))


class WindowAwareLoss(nn.Module):
    """
    Complete window-aware loss function for real estate HDR enhancement.

    Components:
    1. Base Charbonnier loss (robust L1)
    2. Window-weighted loss (higher weight for window regions)
    3. Gradient loss (edge preservation)
    4. FFT loss (frequency details)
    5. Optional: Perceptual loss (VGG features)

    The key insight: Window regions need 2-5x more attention than other regions
    because they have the highest dynamic range and are visually prominent.
    """

    def __init__(self,
                 base_weight: float = 1.0,
                 window_weight: float = 3.0,
                 gradient_weight: float = 0.1,
                 fft_weight: float = 0.05,
                 perceptual_weight: float = 0.0,
                 brightness_threshold: float = 0.7,
                 device: str = 'cuda'):
        super().__init__()

        self.base_weight = base_weight
        self.window_weight = window_weight
        self.gradient_weight = gradient_weight
        self.fft_weight = fft_weight
        self.perceptual_weight = perceptual_weight

        # Loss components
        self.charbonnier = CharbonnierLoss()
        self.gradient_loss = GradientLoss()
        self.fft_loss = FFTLoss()

        # Window detector
        self.window_detector = WindowRegionDetector(
            brightness_threshold=brightness_threshold,
            device=device
        )

        # Optional perceptual loss
        if perceptual_weight > 0:
            self._init_perceptual_loss(device)
        else:
            self.vgg = None

    def _init_perceptual_loss(self, device: str):
        """Initialize VGG for perceptual loss"""
        try:
            from torchvision import models
            vgg = models.vgg19(pretrained=True).features[:16].to(device).eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
        except Exception as e:
            print(f"Warning: Could not load VGG for perceptual loss: {e}")
            self.vgg = None

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features"""
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)

        return F.l1_loss(pred_features, target_features)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                input_img: Optional[torch.Tensor] = None,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute window-aware loss.

        Args:
            pred: Model prediction [B, 3, H, W]
            target: Ground truth [B, 3, H, W]
            input_img: Original input image (for window detection) [B, 3, H, W]
            return_components: If True, return dict with individual loss components

        Returns:
            Total loss (or dict if return_components=True)
        """
        # Use input_img for window detection if provided, otherwise use target
        detection_img = input_img if input_img is not None else target

        # Detect window regions
        window_mask = self.window_detector(detection_img, target)

        # Compute weight map: base_weight for normal regions, higher for windows
        weight_map = self.base_weight + window_mask * (self.window_weight - self.base_weight)

        # 1. Weighted Charbonnier loss
        pixel_loss = self.charbonnier(pred, target)
        weighted_loss = (pixel_loss * weight_map).mean()

        # 2. Gradient loss (edge preservation)
        grad_loss = self.gradient_loss(pred, target)

        # 3. FFT loss (frequency details)
        freq_loss = self.fft_loss(pred, target)

        # 4. Perceptual loss (optional)
        if self.perceptual_weight > 0 and self.vgg is not None:
            perc_loss = self.perceptual_loss(pred, target)
        else:
            perc_loss = torch.tensor(0.0, device=pred.device)

        # Combine losses
        total_loss = (
            weighted_loss +
            self.gradient_weight * grad_loss +
            self.fft_weight * freq_loss +
            self.perceptual_weight * perc_loss
        )

        if return_components:
            return {
                'total': total_loss,
                'weighted_pixel': weighted_loss,
                'gradient': grad_loss,
                'fft': freq_loss,
                'perceptual': perc_loss,
                'window_mask_mean': window_mask.mean()
            }

        return total_loss


class AdaptiveWindowLoss(nn.Module):
    """
    Adaptive version that learns optimal window weighting during training.
    Uses a small learnable network to predict per-pixel weights.
    """

    def __init__(self,
                 base_channels: int = 16,
                 gradient_weight: float = 0.1,
                 fft_weight: float = 0.05):
        super().__init__()

        self.gradient_weight = gradient_weight
        self.fft_weight = fft_weight

        # Learnable weight predictor
        self.weight_net = nn.Sequential(
            nn.Conv2d(6, base_channels, 3, padding=1),  # Input: concat(input, target)
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
            nn.Softplus()  # Ensure positive weights
        )

        self.charbonnier = CharbonnierLoss()
        self.gradient_loss = GradientLoss()
        self.fft_loss = FFTLoss()

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                input_img: torch.Tensor) -> torch.Tensor:
        """Compute adaptive window-aware loss"""
        # Predict per-pixel weights
        weight_input = torch.cat([input_img, target], dim=1)
        weights = self.weight_net(weight_input) + 1.0  # Minimum weight of 1.0

        # Weighted pixel loss
        pixel_loss = self.charbonnier(pred, target)
        weighted_loss = (pixel_loss * weights).mean()

        # Additional losses
        grad_loss = self.gradient_loss(pred, target)
        freq_loss = self.fft_loss(pred, target)

        total_loss = (
            weighted_loss +
            self.gradient_weight * grad_loss +
            self.fft_weight * freq_loss
        )

        return total_loss


def get_window_aware_loss(config: str = 'default', device: str = 'cuda') -> nn.Module:
    """
    Factory function to get pre-configured window-aware loss.

    Args:
        config: Configuration preset
            - 'default': Balanced configuration
            - 'aggressive': Higher window weighting for severe cases
            - 'light': Lower window weighting
            - 'adaptive': Learnable weighting

    Returns:
        Configured loss module
    """
    configs = {
        'default': {
            'base_weight': 1.0,
            'window_weight': 3.0,
            'gradient_weight': 0.1,
            'fft_weight': 0.05,
            'perceptual_weight': 0.0,
            'brightness_threshold': 0.7,
        },
        'aggressive': {
            'base_weight': 1.0,
            'window_weight': 5.0,
            'gradient_weight': 0.15,
            'fft_weight': 0.1,
            'perceptual_weight': 0.1,
            'brightness_threshold': 0.6,
        },
        'light': {
            'base_weight': 1.0,
            'window_weight': 2.0,
            'gradient_weight': 0.05,
            'fft_weight': 0.02,
            'perceptual_weight': 0.0,
            'brightness_threshold': 0.8,
        },
    }

    if config == 'adaptive':
        return AdaptiveWindowLoss()

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    return WindowAwareLoss(**configs[config], device=device)


# Quick test
if __name__ == '__main__':
    print("Testing WindowAwareLoss...")

    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    h, w = 256, 256

    input_img = torch.rand(batch_size, 3, h, w, device=device)
    target = torch.rand(batch_size, 3, h, w, device=device)
    pred = torch.rand(batch_size, 3, h, w, device=device)

    # Test default loss
    loss_fn = get_window_aware_loss('default', device=device)
    loss = loss_fn(pred, target, input_img, return_components=True)

    print(f"Total loss: {loss['total']:.4f}")
    print(f"Weighted pixel loss: {loss['weighted_pixel']:.4f}")
    print(f"Gradient loss: {loss['gradient']:.4f}")
    print(f"FFT loss: {loss['fft']:.4f}")
    print(f"Window mask mean: {loss['window_mask_mean']:.4f}")

    print("\nWindowAwareLoss test passed!")
