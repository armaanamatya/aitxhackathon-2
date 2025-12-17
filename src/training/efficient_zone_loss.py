"""
Efficient Zone-Based Loss for Window Recovery
=============================================

Memory-efficient version that uses adaptive luminance thresholds
(based on image statistics) instead of learned attention.

Key improvement over fixed-threshold approach:
- Adapts thresholds based on image histogram
- Uses percentiles instead of absolute values
- Zone-specific treatment for different exposure levels
- Color direction loss to ensure correct transformation direction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AdaptiveZoneDetector(nn.Module):
    """
    Detects exposure zones using adaptive thresholds based on image statistics.

    More memory efficient than learned attention, but still adapts to each image.
    """
    def __init__(self):
        super().__init__()

        # Smoothing for soft zone boundaries
        self.smooth = nn.AvgPool2d(7, stride=1, padding=3)

    def forward(self, source: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect exposure zones adaptively.

        Args:
            source: [B, 3, H, W] in [0, 1]

        Returns:
            zone_dict: dict with soft masks for each zone
        """
        # Normalize to [0, 1]
        if source.min() < 0:
            source = (source + 1) / 2

        # Compute luminance
        lum = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]

        B = lum.shape[0]

        # Compute per-image statistics for adaptive thresholds
        lum_flat = lum.view(B, -1)

        # Use percentiles for adaptive thresholds
        p10 = torch.quantile(lum_flat, 0.10, dim=1, keepdim=True).view(B, 1, 1, 1)
        p25 = torch.quantile(lum_flat, 0.25, dim=1, keepdim=True).view(B, 1, 1, 1)
        p50 = torch.quantile(lum_flat, 0.50, dim=1, keepdim=True).view(B, 1, 1, 1)
        p75 = torch.quantile(lum_flat, 0.75, dim=1, keepdim=True).view(B, 1, 1, 1)
        p90 = torch.quantile(lum_flat, 0.90, dim=1, keepdim=True).view(B, 1, 1, 1)

        # Also compute saturation for window detection
        max_rgb = source.max(dim=1, keepdim=True)[0]
        min_rgb = source.min(dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)

        # Create soft zone masks with adaptive thresholds
        # Use sigmoid with steep slope for soft boundaries

        # Deep shadow: below 10th percentile
        deep_shadow = torch.sigmoid(20 * (p10 - lum))

        # Shadow: between 10th and 25th percentile
        shadow = torch.sigmoid(15 * (lum - p10)) * torch.sigmoid(15 * (p25 - lum))

        # Midtone: between 25th and 75th percentile
        midtone = torch.sigmoid(10 * (lum - p25)) * torch.sigmoid(10 * (p75 - lum))

        # Highlight: between 75th and 90th percentile
        highlight = torch.sigmoid(15 * (lum - p75)) * torch.sigmoid(15 * (p90 - lum))

        # Blown-out: above 90th percentile AND low saturation (washed out windows)
        bright_mask = torch.sigmoid(20 * (lum - p90))
        low_sat_mask = torch.sigmoid(20 * (0.15 - saturation))
        blown_out = bright_mask * (0.5 + 0.5 * low_sat_mask)  # Even more weight if low saturation

        # Smooth all masks
        zone_dict = {
            'deep_shadow': self.smooth(deep_shadow),
            'shadow': self.smooth(shadow),
            'midtone': self.smooth(midtone),
            'highlight': self.smooth(highlight),
            'blown_out': self.smooth(blown_out),
        }

        return zone_dict


class ColorDirectionLoss(nn.Module):
    """
    Ensures the model transforms colors in the correct direction.

    Key insight: Bad models make moderately bright areas BRIGHTER.
    This loss ensures transformation goes TOWARD target, not away.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        zone_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute direction alignment loss.
        """
        eps = 1e-8

        # Direction vectors
        source_to_target = target - source  # Correct direction
        source_to_pred = pred - source      # Model's direction

        # Normalize directions
        st_norm = F.normalize(source_to_target, dim=1, eps=eps)
        sp_norm = F.normalize(source_to_pred, dim=1, eps=eps)

        # Cosine similarity (1 = aligned, -1 = opposite)
        alignment = (st_norm * sp_norm).sum(dim=1, keepdim=True)

        # Loss: penalize misalignment, especially negative alignment
        # If alignment < 0, model is going OPPOSITE direction (very bad)
        direction_loss = torch.relu(1 - alignment) * zone_mask

        # Also match magnitude of change
        st_mag = torch.norm(source_to_target, dim=1, keepdim=True)
        sp_mag = torch.norm(source_to_pred, dim=1, keepdim=True)

        # Relative magnitude error
        mag_loss = torch.abs(sp_mag - st_mag) / (st_mag + eps) * zone_mask

        return (direction_loss.sum() + 0.5 * mag_loss.sum()) / (zone_mask.sum() + eps)


class EfficientZoneLoss(nn.Module):
    """
    Memory-efficient zone-based loss for window recovery.

    Key features:
    1. Adaptive zone detection (percentile-based, not fixed)
    2. Zone-specific weighting (blown-out gets highest weight)
    3. Color direction loss (ensures correct transformation direction)
    4. Saturation matching in bright regions
    """
    def __init__(
        self,
        weight_blown_out: float = 6.0,
        weight_highlight: float = 4.0,
        weight_midtone: float = 1.0,
        weight_shadow: float = 2.0,
        weight_deep_shadow: float = 1.5,
        lambda_direction: float = 2.0,
        lambda_saturation: float = 1.5,
    ):
        super().__init__()

        self.zone_weights = {
            'blown_out': weight_blown_out,
            'highlight': weight_highlight,
            'midtone': weight_midtone,
            'shadow': weight_shadow,
            'deep_shadow': weight_deep_shadow,
        }

        self.lambda_direction = lambda_direction
        self.lambda_saturation = lambda_saturation

        self.zone_detector = AdaptiveZoneDetector()
        self.direction_loss = ColorDirectionLoss()

    def compute_saturation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        zone_mask: torch.Tensor
    ) -> torch.Tensor:
        """Match saturation in zone."""
        def get_sat(x):
            if x.min() < 0:
                x = (x + 1) / 2
            return (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] + 1e-8)

        pred_sat = get_sat(pred)
        target_sat = get_sat(target)

        return (torch.abs(pred_sat - target_sat) * zone_mask).sum() / (zone_mask.sum() + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        return_components: bool = False
    ):
        """
        Compute efficient zone loss.
        """
        zone_dict = self.zone_detector(source)

        components = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        for zone_name, zone_mask in zone_dict.items():
            weight = self.zone_weights[zone_name]

            # L1 loss in zone
            zone_l1 = (torch.abs(pred - target) * zone_mask).sum() / (zone_mask.sum() + 1e-8)
            components[f'{zone_name}_l1'] = zone_l1
            total_loss = total_loss + weight * zone_l1

            # For blown-out and highlight zones, add direction and saturation loss
            if zone_name in ['blown_out', 'highlight']:
                dir_loss = self.direction_loss(pred, target, source, zone_mask)
                components[f'{zone_name}_direction'] = dir_loss
                total_loss = total_loss + weight * self.lambda_direction * dir_loss

                sat_loss = self.compute_saturation_loss(pred, target, zone_mask)
                components[f'{zone_name}_saturation'] = sat_loss
                total_loss = total_loss + weight * self.lambda_saturation * sat_loss

        if return_components:
            return total_loss, components, zone_dict
        return total_loss


class EfficientWindowRecoveryLoss(nn.Module):
    """
    Complete loss for window recovery with efficient implementation.

    Combines:
    1. Efficient zone loss (adaptive thresholds)
    2. Global L1 baseline
    3. VGG perceptual loss
    4. Gradient preservation
    """
    def __init__(
        self,
        lambda_zone: float = 1.5,
        lambda_global_l1: float = 0.5,
        lambda_vgg: float = 0.1,
        lambda_gradient: float = 0.15,
        blown_out_weight: float = 6.0,
        highlight_weight: float = 4.0,
    ):
        super().__init__()

        self.lambda_zone = lambda_zone
        self.lambda_global_l1 = lambda_global_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_gradient = lambda_gradient

        self.zone_loss = EfficientZoneLoss(
            weight_blown_out=blown_out_weight,
            weight_highlight=highlight_weight,
        )

        self.vgg = None

    def _init_vgg(self, device):
        if self.vgg is None:
            import torchvision.models as models
            vgg = models.vgg16(weights='IMAGENET1K_V1').features[:23]
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg.to(device).eval()

    def _compute_vgg(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        pred_n = (pred - mean) / std
        target_n = (target - mean) / std

        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

    def _compute_gradient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def grad(x):
            return x[:,:,:,1:] - x[:,:,:,:-1], x[:,:,1:,:] - x[:,:,:-1,:]

        px, py = grad(pred)
        tx, ty = grad(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        return_components: bool = False
    ):
        self._init_vgg(pred.device)

        components = {}

        # Global L1
        components['global_l1'] = F.l1_loss(pred, target)

        # Zone loss
        zone_loss, zone_components, zone_dict = self.zone_loss(pred, target, source, return_components=True)
        components['zone_loss'] = zone_loss
        components.update(zone_components)

        # VGG
        if self.lambda_vgg > 0:
            components['vgg'] = self._compute_vgg(pred, target)
        else:
            components['vgg'] = torch.tensor(0.0, device=pred.device)

        # Gradient
        components['gradient'] = self._compute_gradient(pred, target)

        # Total
        total = (
            self.lambda_global_l1 * components['global_l1'] +
            self.lambda_zone * components['zone_loss'] +
            self.lambda_vgg * components['vgg'] +
            self.lambda_gradient * components['gradient']
        )

        if return_components:
            return total, components, zone_dict
        return total


def create_efficient_window_loss(preset: str = 'aggressive') -> EfficientWindowRecoveryLoss:
    """
    Factory function.

    Presets:
        - 'conservative': Stable training
        - 'balanced': Good default
        - 'aggressive': Maximum window focus (recommended)
    """
    configs = {
        'conservative': {
            'lambda_zone': 1.0,
            'lambda_global_l1': 0.7,
            'lambda_vgg': 0.1,
            'lambda_gradient': 0.15,
            'blown_out_weight': 4.0,
            'highlight_weight': 3.0,
        },
        'balanced': {
            'lambda_zone': 1.2,
            'lambda_global_l1': 0.5,
            'lambda_vgg': 0.1,
            'lambda_gradient': 0.15,
            'blown_out_weight': 5.0,
            'highlight_weight': 3.5,
        },
        'aggressive': {
            'lambda_zone': 1.5,
            'lambda_global_l1': 0.3,
            'lambda_vgg': 0.15,
            'lambda_gradient': 0.15,
            'blown_out_weight': 6.0,
            'highlight_weight': 4.0,
        },
    }

    config = configs.get(preset, configs['aggressive'])
    return EfficientWindowRecoveryLoss(**config)
