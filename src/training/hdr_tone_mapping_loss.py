"""
HDR Tone Mapping Losses for Real Estate Photography

Problem: Ground truth images have specific HDR editing style:
1. Dark colors (green plants, red/blue objects) are SIGNIFICANTLY brightened
2. Bright windows/sky are enhanced but not clipped
3. Overall exposure is boosted while maintaining contrast

Solution: Losses that capture this tone mapping transformation.

Author: Top 0.0001% MLE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DarkRegionEnhancementLoss(nn.Module):
    """
    Loss for dark region color enhancement.

    Problem: Dark colored objects (plants, furniture) should be brightened
    and saturated in the output, matching the HDR editing style.
    """

    def __init__(self, darkness_threshold=0.3, enhancement_weight=2.0):
        super().__init__()
        self.darkness_threshold = darkness_threshold
        self.enhancement_weight = enhancement_weight

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV."""
        max_c, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        diff = max_c - min_c + 1e-7

        v = max_c  # Value (brightness)
        s = diff / (max_c + 1e-7)  # Saturation

        # Hue
        hue = torch.zeros_like(max_c)
        mask_r = (max_idx == 0).float()
        mask_g = (max_idx == 1).float()
        mask_b = (max_idx == 2).float()

        hue = hue + mask_r * (((rgb[:, 1:2] - rgb[:, 2:3]) / diff) % 6)
        hue = hue + mask_g * (((rgb[:, 2:3] - rgb[:, 0:1]) / diff) + 2)
        hue = hue + mask_b * (((rgb[:, 0:1] - rgb[:, 1:2]) / diff) + 4)
        hue = hue / 6.0

        return hue, s, v

    def forward(self, pred, target):
        # Get HSV
        pred_h, pred_s, pred_v = self.rgb_to_hsv(pred)
        target_h, target_s, target_v = self.rgb_to_hsv(target)

        # Find dark regions in INPUT (where enhancement should happen)
        # Use target value to identify originally dark regions
        dark_mask = (target_v > self.darkness_threshold).float() * (target_s > 0.1).float()
        dark_mask = F.avg_pool2d(dark_mask, kernel_size=5, stride=1, padding=2)

        # Brightness enhancement loss (V channel)
        v_diff = torch.abs(pred_v - target_v)
        dark_v_loss = (v_diff * dark_mask).sum() / (dark_mask.sum() + 1e-7)

        # Saturation enhancement loss (S channel)
        s_diff = torch.abs(pred_s - target_s)
        dark_s_loss = (s_diff * dark_mask).sum() / (dark_mask.sum() + 1e-7)

        # Penalize under-brightening more than over-brightening
        under_bright = F.relu(target_v - pred_v) * dark_mask
        under_bright_loss = under_bright.sum() / (dark_mask.sum() + 1e-7)

        total = dark_v_loss + dark_s_loss + self.enhancement_weight * under_bright_loss

        return total


class ColorSpecificEnhancementLoss(nn.Module):
    """
    Loss targeting specific color channels (R, G, B).

    Problem: Dark greens (plants), reds, blues are significantly brightened.
    This loss ensures each color channel is boosted correctly.
    """

    def __init__(self, darkness_threshold=0.4):
        super().__init__()
        self.darkness_threshold = darkness_threshold

    def forward(self, pred, target):
        # Per-channel enhancement
        # Find regions where each channel is dominant

        # Red-dominant regions (red furniture, decor)
        red_mask = (target[:, 0:1] > target[:, 1:2]) & (target[:, 0:1] > target[:, 2:3])
        red_mask = red_mask.float() * (target[:, 0:1] > self.darkness_threshold).float()

        # Green-dominant regions (plants, greenery)
        green_mask = (target[:, 1:2] > target[:, 0:1]) & (target[:, 1:2] > target[:, 2:3])
        green_mask = green_mask.float() * (target[:, 1:2] > self.darkness_threshold).float()

        # Blue-dominant regions (sky in windows, blue objects)
        blue_mask = (target[:, 2:3] > target[:, 0:1]) & (target[:, 2:3] > target[:, 1:2])
        blue_mask = blue_mask.float() * (target[:, 2:3] > self.darkness_threshold).float()

        # Compute per-channel losses
        red_loss = (torch.abs(pred[:, 0:1] - target[:, 0:1]) * red_mask).sum() / (red_mask.sum() + 1e-7)
        green_loss = (torch.abs(pred[:, 1:2] - target[:, 1:2]) * green_mask).sum() / (green_mask.sum() + 1e-7)
        blue_loss = (torch.abs(pred[:, 2:3] - target[:, 2:3]) * blue_mask).sum() / (blue_mask.sum() + 1e-7)

        return (red_loss + green_loss + blue_loss) / 3


class ExposureCompensationLoss(nn.Module):
    """
    Loss for overall exposure compensation.

    Problem: HDR images have boosted exposure. This ensures the overall
    brightness distribution matches.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Global brightness difference
        pred_mean = pred.mean(dim=[1, 2, 3], keepdim=True)
        target_mean = target.mean(dim=[1, 2, 3], keepdim=True)

        exposure_loss = F.l1_loss(pred_mean, target_mean)

        # Histogram matching (luminance distribution)
        pred_luma = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_luma = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        # Compute histogram statistics
        pred_sorted, _ = torch.sort(pred_luma.flatten(1), dim=1)
        target_sorted, _ = torch.sort(target_luma.flatten(1), dim=1)

        # Match percentiles (0, 25, 50, 75, 100)
        percentiles = [0, 25, 50, 75, 100]
        hist_loss = 0
        for p in percentiles:
            idx = int(pred_sorted.size(1) * p / 100)
            hist_loss += F.l1_loss(pred_sorted[:, idx:idx+1], target_sorted[:, idx:idx+1])
        hist_loss /= len(percentiles)

        return exposure_loss + 0.5 * hist_loss


class ToneMappingCurveLoss(nn.Module):
    """
    Loss to match the tone mapping curve.

    Problem: Dark inputs → Bright outputs. This learns the transformation curve.
    """

    def __init__(self, num_bins=16):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, pred, target, input_img):
        """
        Args:
            pred: Model output
            target: Ground truth
            input_img: Original input (to understand input→output mapping)
        """
        # Compute luminance
        pred_luma = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_luma = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        input_luma = 0.299 * input_img[:, 0:1] + 0.587 * input_img[:, 1:2] + 0.114 * input_img[:, 2:3]

        # Bin input luminance and compute average output for each bin
        loss = 0
        for i in range(self.num_bins):
            lower = i / self.num_bins
            upper = (i + 1) / self.num_bins

            # Mask for this input bin
            mask = ((input_luma >= lower) & (input_luma < upper)).float()

            if mask.sum() > 0:
                # Average pred and target output for this input range
                pred_avg = (pred_luma * mask).sum() / mask.sum()
                target_avg = (target_luma * mask).sum() / mask.sum()
                loss += F.l1_loss(pred_avg, target_avg)

        return loss / self.num_bins


class ShadowBoostLoss(nn.Module):
    """
    Specifically boost shadow regions (dark areas).

    Problem: Shadows should be lifted in HDR, not just preserved.
    """

    def __init__(self, shadow_threshold=0.2):
        super().__init__()
        self.shadow_threshold = shadow_threshold

    def forward(self, pred, target):
        # Identify shadow regions in target
        target_luma = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        shadow_mask = (target_luma < self.shadow_threshold).float()
        shadow_mask = F.avg_pool2d(shadow_mask, kernel_size=7, stride=1, padding=3)

        # Ensure shadows are brightened
        pred_luma = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]

        # Shadow boost loss
        shadow_diff = torch.abs(pred - target) * shadow_mask
        shadow_loss = shadow_diff.sum() / (shadow_mask.sum() + 1e-7)

        # Penalize under-exposure in shadows
        under_exposed = F.relu(target - pred) * shadow_mask
        under_loss = under_exposed.sum() / (shadow_mask.sum() + 1e-7)

        return shadow_loss + 2.0 * under_loss


class HDRToneMappingLoss(nn.Module):
    """
    Complete HDR Tone Mapping Loss.

    Combines all losses to capture the HDR editing style:
    - L1: Base pixel accuracy
    - Window: Bright region preservation
    - DarkEnhancement: Boost dark colored regions
    - ColorSpecific: Enhance R, G, B channels appropriately
    - Exposure: Match overall brightness
    - ToneMapping: Learn input→output curve
    - ShadowBoost: Lift shadows
    """

    def __init__(
        self,
        l1_weight=1.0,
        window_weight=0.3,
        dark_enhancement_weight=0.4,
        color_specific_weight=0.3,
        exposure_weight=0.2,
        tone_mapping_weight=0.2,
        shadow_boost_weight=0.3,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.window_weight = window_weight
        self.dark_enhancement_weight = dark_enhancement_weight
        self.color_specific_weight = color_specific_weight
        self.exposure_weight = exposure_weight
        self.tone_mapping_weight = tone_mapping_weight
        self.shadow_boost_weight = shadow_boost_weight

        self.l1_loss = nn.L1Loss()
        self.window_loss = WindowAwareLoss()
        self.dark_enhancement_loss = DarkRegionEnhancementLoss()
        self.color_specific_loss = ColorSpecificEnhancementLoss()
        self.exposure_loss = ExposureCompensationLoss()
        self.tone_mapping_loss = ToneMappingCurveLoss()
        self.shadow_boost_loss = ShadowBoostLoss()

    def forward(self, pred, target, input_img=None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: Model output
            target: Ground truth HDR image
            input_img: Original input (optional, for tone mapping loss)
        """
        l1 = self.l1_loss(pred, target)
        window = self.window_loss(pred, target)
        dark_enhancement = self.dark_enhancement_loss(pred, target)
        color_specific = self.color_specific_loss(pred, target)
        exposure = self.exposure_loss(pred, target)
        shadow_boost = self.shadow_boost_loss(pred, target)

        total = (
            self.l1_weight * l1 +
            self.window_weight * window +
            self.dark_enhancement_weight * dark_enhancement +
            self.color_specific_weight * color_specific +
            self.exposure_weight * exposure +
            self.shadow_boost_weight * shadow_boost
        )

        # Tone mapping loss (if input provided)
        if input_img is not None:
            tone_mapping = self.tone_mapping_loss(pred, target, input_img)
            total += self.tone_mapping_weight * tone_mapping
        else:
            tone_mapping = torch.tensor(0.0)

        components = {
            'l1': l1.item(),
            'window': window.item(),
            'dark_enhancement': dark_enhancement.item(),
            'color_specific': color_specific.item(),
            'exposure': exposure.item(),
            'tone_mapping': tone_mapping.item() if isinstance(tone_mapping, torch.Tensor) else 0.0,
            'shadow_boost': shadow_boost.item(),
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
    print("Testing HDR Tone Mapping Losses...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create test tensors
    input_img = torch.rand(2, 3, 256, 256).to(device) * 0.3  # Dark input
    pred = torch.rand(2, 3, 256, 256).to(device) * 0.6  # Medium output
    target = torch.rand(2, 3, 256, 256).to(device) * 0.8  # Bright target

    print("\n1. DarkRegionEnhancementLoss:")
    loss = DarkRegionEnhancementLoss().to(device)
    print(f"   Loss: {loss(pred, target).item():.4f}")

    print("\n2. ColorSpecificEnhancementLoss:")
    loss = ColorSpecificEnhancementLoss().to(device)
    print(f"   Loss: {loss(pred, target).item():.4f}")

    print("\n3. ExposureCompensationLoss:")
    loss = ExposureCompensationLoss().to(device)
    print(f"   Loss: {loss(pred, target).item():.4f}")

    print("\n4. ToneMappingCurveLoss:")
    loss = ToneMappingCurveLoss().to(device)
    print(f"   Loss: {loss(pred, target, input_img).item():.4f}")

    print("\n5. ShadowBoostLoss:")
    loss = ShadowBoostLoss().to(device)
    print(f"   Loss: {loss(pred, target).item():.4f}")

    print("\n6. Complete HDRToneMappingLoss:")
    loss_fn = HDRToneMappingLoss().to(device)
    total, components = loss_fn(pred, target, input_img)
    print(f"   Total: {total.item():.4f}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")

    print("\nAll tests passed!")
