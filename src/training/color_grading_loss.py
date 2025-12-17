"""
Color Grading Loss for Real Estate HDR

Problem: Specific colors (green plants, blue sky, red objects) are
DISPROPORTIONATELY brightened/saturated in ground truth.

This is color grading, not just exposure. Each color channel is enhanced
differently to create the "real estate photography" look.

Author: Top 0.0001% MLE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class ColorChannelGradingLoss(nn.Module):
    """
    Per-color-channel grading loss.

    Problem: Greens (plants), blues (sky), reds (decor) are enhanced
    INDEPENDENTLY with different boost factors.

    Solution: Match each color channel with extra weight on that channel's
    dominant regions.
    """

    def __init__(self, green_weight=1.5, blue_weight=1.3, red_weight=1.2):
        super().__init__()
        self.green_weight = green_weight
        self.blue_weight = blue_weight
        self.red_weight = red_weight

    def forward(self, pred, target):
        # Find green-dominant regions (plants, greenery)
        # Green > Red AND Green > Blue
        green_dominant = ((target[:, 1:2] > target[:, 0:1]) &
                          (target[:, 1:2] > target[:, 2:3])).float()
        green_dominant = F.avg_pool2d(green_dominant, kernel_size=5, stride=1, padding=2)

        # Blue-dominant regions (sky, water, blue objects)
        blue_dominant = ((target[:, 2:3] > target[:, 0:1]) &
                         (target[:, 2:3] > target[:, 1:2])).float()
        blue_dominant = F.avg_pool2d(blue_dominant, kernel_size=5, stride=1, padding=2)

        # Red-dominant regions (furniture, decor, warm colors)
        red_dominant = ((target[:, 0:1] > target[:, 1:2]) &
                        (target[:, 0:1] > target[:, 2:3])).float()
        red_dominant = F.avg_pool2d(red_dominant, kernel_size=5, stride=1, padding=2)

        # Per-channel losses with extra weight
        green_loss = F.l1_loss(
            pred[:, 1:2] * (1 + green_dominant * (self.green_weight - 1)),
            target[:, 1:2] * (1 + green_dominant * (self.green_weight - 1))
        )

        blue_loss = F.l1_loss(
            pred[:, 2:3] * (1 + blue_dominant * (self.blue_weight - 1)),
            target[:, 2:3] * (1 + blue_dominant * (self.blue_weight - 1))
        )

        red_loss = F.l1_loss(
            pred[:, 0:1] * (1 + red_dominant * (self.red_weight - 1)),
            target[:, 0:1] * (1 + red_dominant * (self.red_weight - 1))
        )

        return (green_loss + blue_loss + red_loss) / 3


class ColorBoostRatioLoss(nn.Module):
    """
    Match the color boost ratio between input and output.

    Problem: If input has dim green (0.2) and output has bright green (0.6),
    the model should learn this 3x boost factor.

    This captures the color grading transformation curve.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, input_img):
        """
        Args:
            pred: Model output
            target: Ground truth (color graded)
            input_img: Original input
        """
        # Compute boost ratios for each channel
        # Boost = output / (input + epsilon)

        # Target boost ratio (GT / input)
        target_boost_r = target[:, 0:1] / (input_img[:, 0:1] + 0.01)
        target_boost_g = target[:, 1:2] / (input_img[:, 1:2] + 0.01)
        target_boost_b = target[:, 2:3] / (input_img[:, 2:3] + 0.01)

        # Predicted boost ratio (pred / input)
        pred_boost_r = pred[:, 0:1] / (input_img[:, 0:1] + 0.01)
        pred_boost_g = pred[:, 1:2] / (input_img[:, 1:2] + 0.01)
        pred_boost_b = pred[:, 2:3] / (input_img[:, 2:3] + 0.01)

        # Match boost ratios (log space for multiplicative factors)
        boost_loss_r = F.l1_loss(torch.log(pred_boost_r + 0.01), torch.log(target_boost_r + 0.01))
        boost_loss_g = F.l1_loss(torch.log(pred_boost_g + 0.01), torch.log(target_boost_g + 0.01))
        boost_loss_b = F.l1_loss(torch.log(pred_boost_b + 0.01), torch.log(target_boost_b + 0.01))

        return (boost_loss_r + boost_loss_g + boost_loss_b) / 3


class SelectiveColorEnhancementLoss(nn.Module):
    """
    Selective color enhancement for specific hue ranges.

    Problem: Certain hues (green plants ~120°, blue sky ~210°) are
    specifically enhanced in real estate photos.

    Solution: Extra loss weight for specific hue ranges.
    """

    def __init__(self):
        super().__init__()

        # Target hue ranges (in [0, 1] normalized)
        # Green: 90-150° → 0.25-0.42
        # Blue: 180-240° → 0.5-0.67
        # Red: 0-30° + 330-360° → 0-0.08 + 0.92-1.0
        self.green_hue_range = (0.25, 0.42)
        self.blue_hue_range = (0.5, 0.67)
        self.red_hue_range = [(0.0, 0.08), (0.92, 1.0)]

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV."""
        max_c, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        diff = max_c - min_c + 1e-7

        v = max_c
        s = diff / (max_c + 1e-7)

        # Hue
        hue = torch.zeros_like(max_c)
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        mask_r = (max_idx == 0).float()
        mask_g = (max_idx == 1).float()
        mask_b = (max_idx == 2).float()

        hue = hue + mask_r * (((g - b) / diff) % 6)
        hue = hue + mask_g * (((b - r) / diff) + 2)
        hue = hue + mask_b * (((r - g) / diff) + 4)
        hue = hue / 6.0

        return hue, s, v

    def hue_in_range(self, hue, range_tuple):
        """Check if hue is in range."""
        return ((hue >= range_tuple[0]) & (hue <= range_tuple[1])).float()

    def forward(self, pred, target):
        pred_h, pred_s, pred_v = self.rgb_to_hsv(pred)
        target_h, target_s, target_v = self.rgb_to_hsv(target)

        # Green regions
        green_mask = self.hue_in_range(target_h, self.green_hue_range)
        green_mask = green_mask * (target_s > 0.1).float()  # Must have some saturation

        # Blue regions
        blue_mask = self.hue_in_range(target_h, self.blue_hue_range)
        blue_mask = blue_mask * (target_s > 0.1).float()

        # Red regions (two ranges)
        red_mask = self.hue_in_range(target_h, self.red_hue_range[0]) + \
                   self.hue_in_range(target_h, self.red_hue_range[1])
        red_mask = torch.clamp(red_mask, 0, 1) * (target_s > 0.1).float()

        # Saturation and Value losses for each color
        green_loss = 0
        if green_mask.sum() > 0:
            green_s_loss = (torch.abs(pred_s - target_s) * green_mask).sum() / green_mask.sum()
            green_v_loss = (torch.abs(pred_v - target_v) * green_mask).sum() / green_mask.sum()
            green_loss = green_s_loss + green_v_loss

        blue_loss = 0
        if blue_mask.sum() > 0:
            blue_s_loss = (torch.abs(pred_s - target_s) * blue_mask).sum() / blue_mask.sum()
            blue_v_loss = (torch.abs(pred_v - target_v) * blue_mask).sum() / blue_mask.sum()
            blue_loss = blue_s_loss + blue_v_loss

        red_loss = 0
        if red_mask.sum() > 0:
            red_s_loss = (torch.abs(pred_s - target_s) * red_mask).sum() / red_mask.sum()
            red_v_loss = (torch.abs(pred_v - target_v) * red_mask).sum() / red_mask.sum()
            red_loss = red_s_loss + red_v_loss

        return (green_loss + blue_loss + red_loss) / 3


class ColorGradingLoss(nn.Module):
    """
    Complete Color Grading Loss for Real Estate HDR.

    Components:
    - L1: Base pixel accuracy
    - Window: Bright regions (from baseline)
    - ColorChannelGrading: Per-channel enhancement (green, blue, red)
    - ColorBoostRatio: Match input→output boost factor
    - SelectiveColorEnhancement: Hue-specific enhancement
    """

    def __init__(
        self,
        l1_weight=1.0,
        window_weight=0.3,
        color_channel_weight=0.5,
        color_boost_weight=0.3,
        selective_color_weight=0.4,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.window_weight = window_weight
        self.color_channel_weight = color_channel_weight
        self.color_boost_weight = color_boost_weight
        self.selective_color_weight = selective_color_weight

        self.l1_loss = nn.L1Loss()
        self.window_loss = WindowAwareLoss()
        self.color_channel_loss = ColorChannelGradingLoss()
        self.color_boost_loss = ColorBoostRatioLoss()
        self.selective_color_loss = SelectiveColorEnhancementLoss()

    def forward(self, pred, target, input_img) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: Model output
            target: Ground truth (color graded)
            input_img: Original input (REQUIRED for boost ratio)
        """
        l1 = self.l1_loss(pred, target)
        window = self.window_loss(pred, target)
        color_channel = self.color_channel_loss(pred, target)
        color_boost = self.color_boost_loss(pred, target, input_img)
        selective_color = self.selective_color_loss(pred, target)

        total = (
            self.l1_weight * l1 +
            self.window_weight * window +
            self.color_channel_weight * color_channel +
            self.color_boost_weight * color_boost +
            self.selective_color_weight * selective_color
        )

        components = {
            'l1': l1.item(),
            'window': window.item(),
            'color_channel': color_channel.item(),
            'color_boost': color_boost.item(),
            'selective_color': selective_color.item(),
            'total': total.item(),
        }

        return total, components


class WindowAwareLoss(nn.Module):
    """Window-aware loss (from baseline)."""

    def __init__(self, brightness_threshold=0.7, window_weight=2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.window_weight = window_weight

    def forward(self, pred, target):
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
    print("Testing Color Grading Losses...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Simulate input with dim colors and output with enhanced colors
    input_img = torch.rand(2, 3, 256, 256).to(device) * 0.5
    target = input_img * 1.5  # Boosted colors
    pred = input_img * 1.3  # Partial boost

    print("\n1. ColorChannelGradingLoss:")
    loss = ColorChannelGradingLoss().to(device)
    print(f"   Loss: {loss(pred, target).item():.4f}")

    print("\n2. ColorBoostRatioLoss:")
    loss = ColorBoostRatioLoss().to(device)
    print(f"   Loss: {loss(pred, target, input_img).item():.4f}")

    print("\n3. SelectiveColorEnhancementLoss:")
    loss = SelectiveColorEnhancementLoss().to(device)
    print(f"   Loss: {loss(pred, target).item():.4f}")

    print("\n4. Complete ColorGradingLoss:")
    loss_fn = ColorGradingLoss().to(device)
    total, components = loss_fn(pred, target, input_img)
    print(f"   Total: {total.item():.4f}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")

    print("\nAll tests passed!")
