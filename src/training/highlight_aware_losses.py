"""
Highlight-Aware Losses for Real Estate HDR Enhancement
=======================================================

Handles ALL bright/overexposed regions - not just windows.
Targets: windows, outdoor views, bright objects (green plants, red items, etc.)

Key insight: Source images have overexposed regions that are washed out.
Ground truth recovers colors and details in these regions.
The model needs to learn AGGRESSIVE color transformation in highlights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class HighlightDetector(nn.Module):
    """
    Detect ALL overexposed/bright regions that need color recovery.

    Unlike WindowDetector which targets low-saturation bright areas,
    this detects any bright region regardless of saturation.
    """
    def __init__(
        self,
        brightness_threshold: float = 0.55,  # Lower to catch more
        use_multi_scale: bool = True,
        smooth_kernel: int = 11,
    ):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.use_multi_scale = use_multi_scale

        if smooth_kernel > 0:
            self.smooth = nn.AvgPool2d(smooth_kernel, stride=1, padding=smooth_kernel//2)
        else:
            self.smooth = None

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Detect highlight regions in source image.

        Args:
            source: [B, 3, H, W] in [-1, 1] or [0, 1]

        Returns:
            mask: [B, 1, H, W] soft mask, higher = more overexposed
        """
        # Normalize to [0, 1]
        if source.min() < 0:
            source = (source + 1) / 2

        # Compute brightness
        brightness = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]

        # Primary mask: bright regions
        highlight_mask = torch.sigmoid(15 * (brightness - self.brightness_threshold))

        if self.use_multi_scale:
            # Also detect at multiple scales to catch larger regions
            scales = [2, 4]
            for scale in scales:
                brightness_down = F.avg_pool2d(brightness, scale, scale)
                mask_down = torch.sigmoid(15 * (brightness_down - self.brightness_threshold))
                mask_up = F.interpolate(mask_down, size=brightness.shape[2:], mode='bilinear', align_corners=False)
                highlight_mask = torch.max(highlight_mask, mask_up)

        # Smooth for soft edges
        if self.smooth is not None:
            highlight_mask = self.smooth(highlight_mask)

        return highlight_mask


class HighlightColorLoss(nn.Module):
    """
    Loss that encourages bold color transformations in highlight regions.

    The key insight: in highlight regions, the output should match TARGET colors,
    NOT source colors. We penalize if output stays too close to source in highlights.
    """
    def __init__(
        self,
        brightness_threshold: float = 0.55,
        match_weight: float = 2.0,    # Weight for matching target
        diverge_weight: float = 0.5,  # Weight for diverging from source
    ):
        super().__init__()
        self.detector = HighlightDetector(brightness_threshold=brightness_threshold)
        self.match_weight = match_weight
        self.diverge_weight = diverge_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute highlight-focused color loss.

        Returns:
            loss: combined loss value
            components: dict with individual loss components
        """
        highlight_mask = self.detector(source)

        # Loss 1: Match target in highlights (weighted L1)
        match_loss = (torch.abs(pred - target) * highlight_mask).sum() / (highlight_mask.sum() + 1e-8)

        # Loss 2: Color diversity in highlights
        # Compute color variance in highlight regions
        pred_in_highlights = pred * highlight_mask
        target_in_highlights = target * highlight_mask

        # We want pred to have similar color variance as target in highlights
        pred_var = pred_in_highlights.var(dim=[2, 3], keepdim=True)
        target_var = target_in_highlights.var(dim=[2, 3], keepdim=True)
        variance_loss = F.l1_loss(pred_var, target_var)

        # Loss 3: Direction alignment
        # The change from source->pred should align with source->target
        source_to_target = target - source
        source_to_pred = pred - source

        # Normalize
        eps = 1e-8
        st_norm = source_to_target / (torch.norm(source_to_target, dim=1, keepdim=True) + eps)
        sp_norm = source_to_pred / (torch.norm(source_to_pred, dim=1, keepdim=True) + eps)

        # Cosine similarity (want it high, so loss is 1 - similarity)
        alignment = (st_norm * sp_norm).sum(dim=1, keepdim=True)
        direction_loss = ((1 - alignment) * highlight_mask).sum() / (highlight_mask.sum() + 1e-8)

        # Loss 4: Magnitude matching
        # Pred should change by similar magnitude as target changes from source
        st_mag = torch.norm(source_to_target, dim=1, keepdim=True)
        sp_mag = torch.norm(source_to_pred, dim=1, keepdim=True)
        magnitude_loss = (torch.abs(st_mag - sp_mag) * highlight_mask).sum() / (highlight_mask.sum() + 1e-8)

        # Combine
        total_loss = (
            self.match_weight * match_loss +
            0.3 * variance_loss +
            0.5 * direction_loss +
            0.3 * magnitude_loss
        )

        components = {
            'highlight_match': match_loss,
            'highlight_variance': variance_loss,
            'highlight_direction': direction_loss,
            'highlight_magnitude': magnitude_loss,
        }

        return total_loss, components


class HighlightSaturationLoss(nn.Module):
    """
    Encourages proper saturation in highlight regions.

    Ground truth often has more saturated colors in areas that are
    washed out in the source.
    """
    def __init__(self, brightness_threshold: float = 0.55):
        super().__init__()
        self.detector = HighlightDetector(brightness_threshold=brightness_threshold)

    def rgb_to_saturation(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute saturation from RGB."""
        # Normalize to [0, 1]
        if rgb.min() < 0:
            rgb = (rgb + 1) / 2

        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]

        # Saturation = (max - min) / max
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        return saturation

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """
        Match saturation of target in highlight regions.
        """
        highlight_mask = self.detector(source)

        pred_sat = self.rgb_to_saturation(pred)
        target_sat = self.rgb_to_saturation(target)

        # L1 loss on saturation in highlight regions
        sat_loss = (torch.abs(pred_sat - target_sat) * highlight_mask).sum() / (highlight_mask.sum() + 1e-8)

        return sat_loss


class HighlightHueLoss(nn.Module):
    """
    Encourages correct hue in highlight regions.

    Important for recovering correct colors (green plants, blue sky, etc.)
    that are washed out in source.
    """
    def __init__(self, brightness_threshold: float = 0.55):
        super().__init__()
        self.detector = HighlightDetector(brightness_threshold=brightness_threshold)

    def rgb_to_hue(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute hue from RGB using a differentiable approximation.
        """
        # Normalize to [0, 1]
        if rgb.min() < 0:
            rgb = (rgb + 1) / 2

        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]
        diff = max_rgb - min_rgb + 1e-8

        # Hue calculation (simplified, returns in [0, 1])
        # Using atan2-style computation for differentiability
        hue = torch.atan2(
            1.732051 * (g - b),  # sqrt(3) * (g - b)
            2 * r - g - b + 1e-8
        ) / (2 * 3.14159)  # Normalize to [0, 1]

        hue = (hue + 1) / 2  # Shift to [0, 1]

        return hue

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """
        Match hue of target in highlight regions.
        """
        highlight_mask = self.detector(source)

        pred_hue = self.rgb_to_hue(pred)
        target_hue = self.rgb_to_hue(target)

        # Circular distance for hue
        hue_diff = torch.abs(pred_hue - target_hue)
        hue_diff = torch.min(hue_diff, 1 - hue_diff)  # Circular

        hue_loss = (hue_diff * highlight_mask).sum() / (highlight_mask.sum() + 1e-8)

        return hue_loss


class ComprehensiveHighlightLoss(nn.Module):
    """
    Complete highlight-aware loss combining all components.

    For real estate HDR: learns to transform ALL bright regions
    (windows, outdoor views, bright objects) to match ground truth.
    """
    def __init__(
        self,
        # Standard losses
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.05,
        # Highlight-specific
        lambda_highlight_color: float = 2.0,
        lambda_highlight_saturation: float = 1.0,
        lambda_highlight_hue: float = 0.5,
        lambda_highlight_weighted_l1: float = 3.0,  # Extra L1 weight on highlights
        # Edge preservation
        lambda_gradient: float = 0.15,
        lambda_ssim: float = 0.1,
        # Detection
        brightness_threshold: float = 0.55,
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_highlight_color = lambda_highlight_color
        self.lambda_highlight_saturation = lambda_highlight_saturation
        self.lambda_highlight_hue = lambda_highlight_hue
        self.lambda_highlight_weighted_l1 = lambda_highlight_weighted_l1
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim

        # Highlight losses
        self.highlight_detector = HighlightDetector(brightness_threshold=brightness_threshold)
        self.highlight_color = HighlightColorLoss(brightness_threshold=brightness_threshold)
        self.highlight_saturation = HighlightSaturationLoss(brightness_threshold=brightness_threshold)
        self.highlight_hue = HighlightHueLoss(brightness_threshold=brightness_threshold)

        # Standard losses - lazy init
        self.vgg = None
        self.lpips = None

    def _init_losses(self, device):
        if self.vgg is None:
            import torchvision.models as models
            vgg = models.vgg16(weights='IMAGENET1K_V1').features[:23]
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg.to(device).eval()

        if self.lpips is None and self.lambda_lpips > 0:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net='alex').to(device).eval()
            except ImportError:
                self.lambda_lpips = 0

    def _compute_vgg(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        pred_n = ((pred + 1) / 2 - mean) / std
        target_n = ((target + 1) / 2 - mean) / std

        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

    def _compute_gradient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def grad(x):
            return x[:,:,:,1:] - x[:,:,:,:-1], x[:,:,1:,:] - x[:,:,:-1,:]

        px, py = grad(pred)
        tx, ty = grad(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2

        mu_p = F.avg_pool2d(pred, 3, 1, 1)
        mu_t = F.avg_pool2d(target, 3, 1, 1)

        sig_p = F.avg_pool2d(pred**2, 3, 1, 1) - mu_p**2
        sig_t = F.avg_pool2d(target**2, 3, 1, 1) - mu_t**2
        sig_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t

        ssim = ((2*mu_p*mu_t + C1) * (2*sig_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
        return 1 - ssim.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        return_components: bool = False
    ):
        self._init_losses(pred.device)

        components = {}

        # Standard L1
        components['l1'] = F.l1_loss(pred, target)

        # Highlight-weighted L1
        highlight_mask = self.highlight_detector(source)
        weight_map = 1.0 + (self.lambda_highlight_weighted_l1 - 1.0) * highlight_mask
        components['highlight_l1'] = (torch.abs(pred - target) * weight_map).mean()

        # Highlight color loss
        color_loss, color_components = self.highlight_color(pred, target, source)
        components['highlight_color'] = color_loss
        components.update(color_components)

        # Highlight saturation
        components['highlight_sat'] = self.highlight_saturation(pred, target, source)

        # Highlight hue
        components['highlight_hue'] = self.highlight_hue(pred, target, source)

        # VGG
        components['vgg'] = self._compute_vgg(pred, target) if self.lambda_vgg > 0 else torch.tensor(0.0, device=pred.device)

        # LPIPS
        if self.lambda_lpips > 0 and self.lpips is not None:
            components['lpips'] = self.lpips(pred, target).mean()
        else:
            components['lpips'] = torch.tensor(0.0, device=pred.device)

        # Gradient
        components['gradient'] = self._compute_gradient(pred, target)

        # SSIM
        components['ssim'] = self._compute_ssim(pred, target)

        # Total
        total = (
            self.lambda_l1 * components['l1'] +
            components['highlight_l1'] +  # Already weighted
            self.lambda_highlight_color * components['highlight_color'] +
            self.lambda_highlight_saturation * components['highlight_sat'] +
            self.lambda_highlight_hue * components['highlight_hue'] +
            self.lambda_vgg * components['vgg'] +
            self.lambda_lpips * components['lpips'] +
            self.lambda_gradient * components['gradient'] +
            self.lambda_ssim * components['ssim']
        )

        if return_components:
            return total, components, highlight_mask
        return total


def create_highlight_aware_loss(preset: str = 'aggressive') -> ComprehensiveHighlightLoss:
    """
    Factory function for highlight-aware loss.

    Presets:
        - 'conservative': Milder highlight focus
        - 'default': Balanced
        - 'aggressive': Strong focus on highlight transformation (RECOMMENDED)
    """
    configs = {
        'conservative': {
            'lambda_l1': 1.0,
            'lambda_highlight_color': 1.0,
            'lambda_highlight_saturation': 0.5,
            'lambda_highlight_hue': 0.3,
            'lambda_highlight_weighted_l1': 2.0,
            'brightness_threshold': 0.65,
        },
        'default': {
            'lambda_l1': 1.0,
            'lambda_highlight_color': 1.5,
            'lambda_highlight_saturation': 0.8,
            'lambda_highlight_hue': 0.4,
            'lambda_highlight_weighted_l1': 2.5,
            'brightness_threshold': 0.58,
        },
        'aggressive': {
            'lambda_l1': 0.8,  # Reduce base L1 to allow more transformation
            'lambda_highlight_color': 2.5,
            'lambda_highlight_saturation': 1.2,
            'lambda_highlight_hue': 0.6,
            'lambda_highlight_weighted_l1': 4.0,
            'brightness_threshold': 0.50,  # Catch more highlights
        },
    }

    config = configs.get(preset, configs['aggressive'])

    return ComprehensiveHighlightLoss(
        lambda_l1=config['lambda_l1'],
        lambda_vgg=0.1,
        lambda_lpips=0.05,
        lambda_highlight_color=config['lambda_highlight_color'],
        lambda_highlight_saturation=config['lambda_highlight_saturation'],
        lambda_highlight_hue=config['lambda_highlight_hue'],
        lambda_highlight_weighted_l1=config['lambda_highlight_weighted_l1'],
        lambda_gradient=0.15,
        lambda_ssim=0.1,
        brightness_threshold=config['brightness_threshold'],
    )
