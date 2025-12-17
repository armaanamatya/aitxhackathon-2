"""
Delta-Aware Loss for Robust Window Recovery
============================================

Key insight: Not all windows need fixing!
- Some windows are already good in source (source ≈ target)
- Some windows are bad and need correction (source ≠ target)

This loss:
1. Computes delta = |source - target| to find what NEEDS changing
2. High delta regions → aggressive correction
3. Low delta regions → preserve (don't overcorrect)
4. Adapts per-image, per-region

This ensures:
- Super bright blown-out windows get fixed
- Already-good windows are preserved
- Model learns WHEN to correct, not just HOW
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DeltaAwareZoneDetector(nn.Module):
    """
    Detects zones based on BOTH luminance AND how much correction is needed.

    Unlike fixed zone detection, this adapts based on source-target difference.
    """
    def __init__(self):
        super().__init__()
        self.smooth = nn.AvgPool2d(7, stride=1, padding=3)

    def compute_luminance(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() < 0:
            x = (x + 1) / 2
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def compute_saturation(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() < 0:
            x = (x + 1) / 2
        max_rgb = x.max(dim=1, keepdim=True)[0]
        min_rgb = x.min(dim=1, keepdim=True)[0]
        return (max_rgb - min_rgb) / (max_rgb + 1e-8)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Detect zones with delta-awareness.

        Returns:
            zones with 'needs_fix' weighting based on source-target difference
        """
        # Normalize
        if source.min() < 0:
            source = (source + 1) / 2
            target = (target + 1) / 2

        src_lum = self.compute_luminance(source)
        tgt_lum = self.compute_luminance(target)
        src_sat = self.compute_saturation(source)
        tgt_sat = self.compute_saturation(target)

        B = source.shape[0]

        # Compute per-image adaptive thresholds
        src_flat = src_lum.view(B, -1)
        p75 = torch.quantile(src_flat, 0.75, dim=1, keepdim=True).view(B, 1, 1, 1)
        p90 = torch.quantile(src_flat, 0.90, dim=1, keepdim=True).view(B, 1, 1, 1)

        # === DELTA COMPUTATION ===
        # How much does this region NEED to change?
        color_delta = torch.abs(source - target).mean(dim=1, keepdim=True)
        lum_delta = torch.abs(src_lum - tgt_lum)
        sat_delta = torch.abs(src_sat - tgt_sat)

        # Combined delta - higher = more correction needed
        total_delta = color_delta + 0.5 * lum_delta + 0.5 * sat_delta

        # Normalize delta to [0, 1] per image
        delta_flat = total_delta.view(B, -1)
        delta_max = delta_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1) + 1e-8
        normalized_delta = total_delta / delta_max

        # === ZONE DETECTION ===
        # Bright regions (potential windows)
        bright_mask = torch.sigmoid(15 * (src_lum - p75))
        very_bright_mask = torch.sigmoid(20 * (src_lum - p90))

        # Low saturation (washed out)
        low_sat = torch.sigmoid(20 * (0.2 - src_sat))

        # === DELTA-WEIGHTED ZONES ===
        # needs_heavy_fix: bright + high delta (bad windows needing correction)
        needs_heavy_fix = very_bright_mask * torch.sigmoid(5 * (normalized_delta - 0.3))

        # needs_light_fix: bright + medium delta
        needs_light_fix = bright_mask * torch.sigmoid(5 * (normalized_delta - 0.15)) * (1 - needs_heavy_fix)

        # preserve: bright but low delta (already good windows)
        preserve = bright_mask * torch.sigmoid(5 * (0.15 - normalized_delta))

        # normal: everything else
        normal = 1 - bright_mask

        # Smooth all masks
        zones = {
            'needs_heavy_fix': self.smooth(needs_heavy_fix),
            'needs_light_fix': self.smooth(needs_light_fix),
            'preserve': self.smooth(preserve),
            'normal': self.smooth(normal),
            'delta_map': normalized_delta,  # For visualization
        }

        return zones


class DeltaAwareLoss(nn.Module):
    """
    Loss that adapts based on how much correction is needed.

    - High delta regions: aggressive correction, direction loss
    - Low delta regions: preservation loss (don't change what's good)
    """
    def __init__(
        self,
        weight_heavy_fix: float = 6.0,
        weight_light_fix: float = 3.0,
        weight_preserve: float = 2.0,
        weight_normal: float = 1.0,
        lambda_direction: float = 2.0,
        lambda_preserve: float = 1.5,
    ):
        super().__init__()

        self.weights = {
            'needs_heavy_fix': weight_heavy_fix,
            'needs_light_fix': weight_light_fix,
            'preserve': weight_preserve,
            'normal': weight_normal,
        }

        self.lambda_direction = lambda_direction
        self.lambda_preserve = lambda_preserve

        self.detector = DeltaAwareZoneDetector()

    def direction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Ensure pred moves toward target, not away."""
        eps = 1e-8

        s2t = target - source  # Correct direction
        s2p = pred - source    # Model's direction

        # Cosine alignment
        s2t_norm = F.normalize(s2t, dim=1, eps=eps)
        s2p_norm = F.normalize(s2p, dim=1, eps=eps)
        alignment = (s2t_norm * s2p_norm).sum(dim=1, keepdim=True)

        # Penalize misalignment
        dir_loss = torch.relu(1 - alignment) * mask

        # Magnitude matching
        s2t_mag = torch.norm(s2t, dim=1, keepdim=True)
        s2p_mag = torch.norm(s2p, dim=1, keepdim=True)
        mag_loss = torch.abs(s2p_mag - s2t_mag) / (s2t_mag + eps) * mask

        return (dir_loss.sum() + 0.5 * mag_loss.sum()) / (mask.sum() + eps)

    def preservation_loss(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        For already-good regions, pred should stay close to source.
        Don't overcorrect what's already good!
        """
        # Penalize deviation from source in preserve regions
        deviation = torch.abs(pred - source) * mask
        return deviation.sum() / (mask.sum() + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        return_components: bool = False
    ):
        zones = self.detector(source, target)

        components = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        eps = 1e-8

        for zone_name in ['needs_heavy_fix', 'needs_light_fix', 'preserve', 'normal']:
            mask = zones[zone_name]
            weight = self.weights[zone_name]

            # Basic L1 loss in zone
            zone_l1 = (torch.abs(pred - target) * mask).sum() / (mask.sum() + eps)
            components[f'{zone_name}_l1'] = zone_l1
            total_loss = total_loss + weight * zone_l1

            # Direction loss for regions needing fix
            if zone_name in ['needs_heavy_fix', 'needs_light_fix']:
                dir_loss = self.direction_loss(pred, target, source, mask)
                components[f'{zone_name}_direction'] = dir_loss

                dir_weight = self.lambda_direction if zone_name == 'needs_heavy_fix' else self.lambda_direction * 0.5
                total_loss = total_loss + weight * dir_weight * dir_loss

            # Preservation loss for good regions
            if zone_name == 'preserve':
                pres_loss = self.preservation_loss(pred, source, mask)
                components['preserve_deviation'] = pres_loss
                total_loss = total_loss + self.lambda_preserve * pres_loss

        if return_components:
            return total_loss, components, zones
        return total_loss


class RobustWindowRecoveryLoss(nn.Module):
    """
    Complete robust loss combining:
    1. Delta-aware zone detection
    2. Direction loss for bad windows
    3. Preservation loss for good windows
    4. Perceptual loss
    5. Gradient preservation
    """
    def __init__(
        self,
        lambda_delta: float = 1.5,
        lambda_global_l1: float = 0.3,
        lambda_vgg: float = 0.1,
        lambda_gradient: float = 0.15,
        lambda_ssim: float = 0.1,
    ):
        super().__init__()

        self.lambda_delta = lambda_delta
        self.lambda_global_l1 = lambda_global_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim

        self.delta_loss = DeltaAwareLoss()
        self.vgg = None

    def _init_vgg(self, device):
        if self.vgg is None:
            import torchvision.models as models
            vgg = models.vgg16(weights='IMAGENET1K_V1').features[:16]  # Lighter VGG
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg.to(device).eval()

    def _vgg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        pred_n = (pred - mean) / std
        target_n = (target - mean) / std

        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def grad(x):
            return x[:,:,:,1:] - x[:,:,:,:-1], x[:,:,1:,:] - x[:,:,:-1,:]

        px, py = grad(pred)
        tx, ty = grad(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2

        mu_p = F.avg_pool2d(pred, 11, stride=1, padding=5)
        mu_t = F.avg_pool2d(target, 11, stride=1, padding=5)

        sigma_p = F.avg_pool2d(pred**2, 11, stride=1, padding=5) - mu_p**2
        sigma_t = F.avg_pool2d(target**2, 11, stride=1, padding=5) - mu_t**2
        sigma_pt = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_p * mu_t

        ssim = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2))

        return 1 - ssim.mean()

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

        # Delta-aware loss
        delta_loss, delta_components, zones = self.delta_loss(
            pred, target, source, return_components=True
        )
        components['delta_loss'] = delta_loss
        components.update(delta_components)

        # VGG
        components['vgg'] = self._vgg_loss(pred, target) if self.lambda_vgg > 0 else torch.tensor(0.0, device=pred.device)

        # Gradient
        components['gradient'] = self._gradient_loss(pred, target)

        # SSIM
        components['ssim'] = self._ssim_loss(pred, target)

        # Total
        total = (
            self.lambda_global_l1 * components['global_l1'] +
            self.lambda_delta * components['delta_loss'] +
            self.lambda_vgg * components['vgg'] +
            self.lambda_gradient * components['gradient'] +
            self.lambda_ssim * components['ssim']
        )

        if return_components:
            return total, components, zones
        return total


def create_robust_window_loss(preset: str = 'robust') -> RobustWindowRecoveryLoss:
    """
    Factory function.

    Presets:
        - 'conservative': Less aggressive
        - 'balanced': Good default
        - 'robust': Maximum adaptivity (recommended)
    """
    configs = {
        'conservative': {
            'lambda_delta': 1.0,
            'lambda_global_l1': 0.5,
            'lambda_vgg': 0.1,
            'lambda_gradient': 0.15,
            'lambda_ssim': 0.1,
        },
        'balanced': {
            'lambda_delta': 1.2,
            'lambda_global_l1': 0.4,
            'lambda_vgg': 0.1,
            'lambda_gradient': 0.15,
            'lambda_ssim': 0.1,
        },
        'robust': {
            'lambda_delta': 1.5,
            'lambda_global_l1': 0.3,
            'lambda_vgg': 0.15,
            'lambda_gradient': 0.15,
            'lambda_ssim': 0.15,
        },
    }

    config = configs.get(preset, configs['robust'])
    return RobustWindowRecoveryLoss(**config)
