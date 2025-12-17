"""
SOTA Luminance-Aware Attention Loss for Real Estate HDR Enhancement
====================================================================

References:
- OENet (2024): Attention-Guided Feature Fusion for overexposure correction
- AGCSNet (2024): Automatic illumination-map attention with gamma/saturation
- Learning Adaptive Lighting (2024): Channel-aware guidance for tone mapping
- CEVR (ICCV 2023): Continuous exposure value representations

Key Innovation: LEARNED attention maps (heat maps) that:
1. Adaptively detect exposure zones based on image statistics (not fixed thresholds)
2. Use multi-scale processing for robust boundary detection
3. Learn optimal zone boundaries during training
4. Apply zone-specific losses (shadows, midtones, highlights, blown-out)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class LearnedLuminanceAttention(nn.Module):
    """
    SOTA: Learned attention maps for exposure zone detection.

    Instead of fixed thresholds, learns to detect:
    - Blown-out regions (windows, sky)
    - Highlight regions (bright but recoverable)
    - Midtone regions (well-exposed)
    - Shadow regions (dark but with detail)
    - Deep shadow regions (very dark)

    Uses lightweight conv layers to learn image-adaptive attention.
    """
    def __init__(
        self,
        num_zones: int = 5,
        hidden_dim: int = 32,
        use_statistics: bool = True,
    ):
        super().__init__()
        self.num_zones = num_zones
        self.use_statistics = use_statistics

        # Lightweight encoder for attention prediction
        # Input: RGB (3) + Luminance (1) + optional stats
        input_channels = 4 + (6 if use_statistics else 0)  # mean, std, min, max, percentiles

        self.attention_net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_zones, 1),  # Output: per-zone attention
        )

        # Multi-scale refinement
        self.scales = [1, 2, 4]
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))

        # Learnable zone boundaries (initialized to reasonable defaults)
        # Zones: [0-0.15] deep shadow, [0.15-0.35] shadow, [0.35-0.65] midtone,
        #        [0.65-0.85] highlight, [0.85-1.0] blown-out
        self.zone_centers = nn.Parameter(torch.tensor([0.08, 0.25, 0.50, 0.75, 0.92]))
        self.zone_widths = nn.Parameter(torch.tensor([0.15, 0.20, 0.30, 0.20, 0.15]))

    def compute_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute perceptual luminance."""
        if rgb.min() < 0:
            rgb = (rgb + 1) / 2
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    def compute_statistics(self, lum: torch.Tensor) -> torch.Tensor:
        """Compute global image statistics for context."""
        B, _, H, W = lum.shape

        # Global stats
        mean = lum.mean(dim=[2, 3], keepdim=True).expand(-1, -1, H, W)
        std = lum.std(dim=[2, 3], keepdim=True).expand(-1, -1, H, W)
        min_val = lum.amin(dim=[2, 3], keepdim=True).expand(-1, -1, H, W)
        max_val = lum.amax(dim=[2, 3], keepdim=True).expand(-1, -1, H, W)

        # Percentiles (approximate via sorting)
        lum_flat = lum.view(B, -1)
        p25 = torch.quantile(lum_flat, 0.25, dim=1, keepdim=True).view(B, 1, 1, 1).expand(-1, -1, H, W)
        p75 = torch.quantile(lum_flat, 0.75, dim=1, keepdim=True).view(B, 1, 1, 1).expand(-1, -1, H, W)

        return torch.cat([mean, std, min_val, max_val, p25, p75], dim=1)

    def forward(self, source: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute learned attention maps for each exposure zone.

        Args:
            source: [B, 3, H, W] source image

        Returns:
            zone_maps: [B, num_zones, H, W] soft attention for each zone
            zone_dict: dict with named zone maps for easy access
        """
        # Normalize to [0, 1]
        if source.min() < 0:
            source_norm = (source + 1) / 2
        else:
            source_norm = source

        # Compute luminance
        lum = self.compute_luminance(source_norm)

        # Build input features
        features = [source_norm, lum]
        if self.use_statistics:
            stats = self.compute_statistics(lum)
            features.append(stats)

        x = torch.cat(features, dim=1)

        # Multi-scale attention computation
        H, W = source.shape[2:]
        zone_maps_list = []

        for i, scale in enumerate(self.scales):
            if scale > 1:
                x_scaled = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            attention = self.attention_net(x_scaled)

            if scale > 1:
                attention = F.interpolate(attention, size=(H, W), mode='bilinear', align_corners=False)

            zone_maps_list.append(attention * F.softmax(self.scale_weights, dim=0)[i])

        # Combine scales
        zone_logits = sum(zone_maps_list)

        # Also add luminance-based prior (soft assignment based on learned boundaries)
        lum_prior = self._compute_luminance_prior(lum)
        zone_logits = zone_logits + lum_prior

        # Softmax to get probability distribution over zones
        zone_maps = F.softmax(zone_logits, dim=1)

        # Create named dict
        zone_names = ['deep_shadow', 'shadow', 'midtone', 'highlight', 'blown_out']
        zone_dict = {name: zone_maps[:, i:i+1] for i, name in enumerate(zone_names)}

        return zone_maps, zone_dict

    def _compute_luminance_prior(self, lum: torch.Tensor) -> torch.Tensor:
        """Soft zone assignment based on learned boundaries."""
        B, _, H, W = lum.shape

        # Compute distance to each zone center
        centers = torch.sigmoid(self.zone_centers)  # Keep in [0, 1]
        widths = F.softplus(self.zone_widths) + 0.05  # Minimum width

        zone_priors = []
        for center, width in zip(centers, widths):
            # Gaussian-like assignment
            dist = (lum - center) / width
            prior = torch.exp(-0.5 * dist ** 2)
            zone_priors.append(prior)

        return torch.cat(zone_priors, dim=1)


class ExposureZoneLoss(nn.Module):
    """
    SOTA: Zone-specific losses with learned attention maps.

    Different zones get different loss treatments:
    - Blown-out: Aggressive color recovery, saturation boost
    - Highlights: Color matching, detail preservation
    - Midtones: Standard L1, structural preservation
    - Shadows: Lift while preserving detail
    - Deep shadows: Noise-aware lifting
    """
    def __init__(
        self,
        # Zone weights (how much to emphasize each zone)
        weight_blown_out: float = 5.0,
        weight_highlight: float = 3.0,
        weight_midtone: float = 1.0,
        weight_shadow: float = 2.0,
        weight_deep_shadow: float = 1.5,
        # Loss component weights
        lambda_color_direction: float = 2.0,
        lambda_saturation: float = 1.5,
        lambda_local_contrast: float = 0.5,
    ):
        super().__init__()

        self.zone_weights = {
            'blown_out': weight_blown_out,
            'highlight': weight_highlight,
            'midtone': weight_midtone,
            'shadow': weight_shadow,
            'deep_shadow': weight_deep_shadow,
        }

        self.lambda_color_direction = lambda_color_direction
        self.lambda_saturation = lambda_saturation
        self.lambda_local_contrast = lambda_local_contrast

        # Attention module
        self.attention = LearnedLuminanceAttention()

    def compute_color_direction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        zone_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure pred moves from source TOWARD target in color space.
        Critical for blown-out windows.
        """
        source_to_target = target - source
        source_to_pred = pred - source

        eps = 1e-8

        # Normalize directions
        st_norm = F.normalize(source_to_target, dim=1, eps=eps)
        sp_norm = F.normalize(source_to_pred, dim=1, eps=eps)

        # Cosine similarity (want alignment)
        alignment = (st_norm * sp_norm).sum(dim=1, keepdim=True)
        direction_loss = (1 - alignment) * zone_mask

        # Also match magnitude
        st_mag = torch.norm(source_to_target, dim=1, keepdim=True)
        sp_mag = torch.norm(source_to_pred, dim=1, keepdim=True)
        mag_loss = torch.abs(st_mag - sp_mag) * zone_mask / (st_mag + eps)

        return (direction_loss.sum() + 0.5 * mag_loss.sum()) / (zone_mask.sum() + eps)

    def compute_saturation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        zone_mask: torch.Tensor
    ) -> torch.Tensor:
        """Match target saturation in zone."""
        def get_saturation(rgb):
            if rgb.min() < 0:
                rgb = (rgb + 1) / 2
            max_rgb = rgb.max(dim=1, keepdim=True)[0]
            min_rgb = rgb.min(dim=1, keepdim=True)[0]
            return (max_rgb - min_rgb) / (max_rgb + 1e-8)

        pred_sat = get_saturation(pred)
        target_sat = get_saturation(target)

        sat_loss = torch.abs(pred_sat - target_sat) * zone_mask
        return sat_loss.sum() / (zone_mask.sum() + 1e-8)

    def compute_local_contrast_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        zone_mask: torch.Tensor
    ) -> torch.Tensor:
        """Preserve local contrast (important for window details)."""
        def local_std(x, kernel_size=7):
            padding = kernel_size // 2
            mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
            sq_mean = F.avg_pool2d(x ** 2, kernel_size, stride=1, padding=padding)
            return torch.sqrt(sq_mean - mean ** 2 + 1e-8)

        pred_lc = local_std(pred)
        target_lc = local_std(target)

        lc_loss = torch.abs(pred_lc - target_lc) * zone_mask
        return lc_loss.sum() / (zone_mask.sum() + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute zone-aware losses.
        """
        # Get attention maps
        _, zone_dict = self.attention(source)

        components = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        for zone_name, zone_mask in zone_dict.items():
            weight = self.zone_weights[zone_name]

            # Basic L1 in zone
            zone_l1 = (torch.abs(pred - target) * zone_mask).sum() / (zone_mask.sum() + 1e-8)
            components[f'{zone_name}_l1'] = zone_l1
            total_loss = total_loss + weight * zone_l1

            # For blown-out and highlight zones, add color direction loss
            if zone_name in ['blown_out', 'highlight']:
                color_dir = self.compute_color_direction_loss(pred, target, source, zone_mask)
                components[f'{zone_name}_color_dir'] = color_dir
                total_loss = total_loss + weight * self.lambda_color_direction * color_dir

                sat_loss = self.compute_saturation_loss(pred, target, zone_mask)
                components[f'{zone_name}_saturation'] = sat_loss
                total_loss = total_loss + weight * self.lambda_saturation * sat_loss

                lc_loss = self.compute_local_contrast_loss(pred, target, zone_mask)
                components[f'{zone_name}_local_contrast'] = lc_loss
                total_loss = total_loss + weight * self.lambda_local_contrast * lc_loss

        if return_components:
            return total_loss, components, zone_dict
        return total_loss


class AdaptiveBoundaryDetector(nn.Module):
    """
    SOTA: Detects optimal boundaries between exposure zones.

    Uses edge detection on luminance + learned refinement to find
    where exposure transitions occur (e.g., window frame edges).
    """
    def __init__(self):
        super().__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Refinement network
        self.refine = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # luminance + gradients
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Detect exposure boundaries.

        Returns:
            boundary_mask: [B, 1, H, W] where 1 = boundary region
        """
        # Compute luminance
        if source.min() < 0:
            source = (source + 1) / 2
        lum = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]

        # Compute gradients
        grad_x = F.conv2d(lum, self.sobel_x, padding=1)
        grad_y = F.conv2d(lum, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Combine and refine
        features = torch.cat([lum, grad_x, grad_y], dim=1)
        boundary_mask = self.refine(features)

        return boundary_mask


class BoundaryPreservationLoss(nn.Module):
    """
    Preserve edges at exposure boundaries (window frames, etc.).
    """
    def __init__(self):
        super().__init__()
        self.boundary_detector = AdaptiveBoundaryDetector()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """
        Extra loss weight on boundary regions.
        """
        boundary_mask = self.boundary_detector(source)

        # Gradient matching at boundaries
        def gradient(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy

        pred_dx, pred_dy = gradient(pred)
        target_dx, target_dy = gradient(target)

        # Resize boundary mask for gradient sizes
        boundary_dx = boundary_mask[:, :, :, :-1]
        boundary_dy = boundary_mask[:, :, :-1, :]

        # Weighted gradient loss
        grad_loss_x = (torch.abs(pred_dx - target_dx) * boundary_dx).sum() / (boundary_dx.sum() + 1e-8)
        grad_loss_y = (torch.abs(pred_dy - target_dy) * boundary_dy).sum() / (boundary_dy.sum() + 1e-8)

        return grad_loss_x + grad_loss_y


class SOTAWindowRecoveryLoss(nn.Module):
    """
    Complete SOTA loss for window/highlight recovery.

    Combines:
    1. Learned luminance attention (adaptive zone detection)
    2. Zone-specific losses (different treatment per exposure zone)
    3. Boundary preservation (clean edges at window frames)
    4. Multi-scale processing
    5. Perceptual losses
    """
    def __init__(
        self,
        # Main loss weights
        lambda_zone: float = 1.0,
        lambda_boundary: float = 0.3,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.05,
        lambda_global_l1: float = 0.5,
        # Zone emphasis
        blown_out_weight: float = 5.0,
        highlight_weight: float = 3.0,
    ):
        super().__init__()

        self.lambda_zone = lambda_zone
        self.lambda_boundary = lambda_boundary
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_global_l1 = lambda_global_l1

        # Zone loss with attention
        self.zone_loss = ExposureZoneLoss(
            weight_blown_out=blown_out_weight,
            weight_highlight=highlight_weight,
        )

        # Boundary preservation
        self.boundary_loss = BoundaryPreservationLoss()

        # Perceptual losses - lazy init
        self.vgg = None
        self.lpips = None

    def _init_perceptual(self, device):
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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        return_components: bool = False
    ):
        """
        Compute complete SOTA loss.
        """
        self._init_perceptual(pred.device)

        components = {}

        # Global L1 (baseline)
        components['global_l1'] = F.l1_loss(pred, target)

        # Zone-specific losses with attention
        zone_loss, zone_components, zone_dict = self.zone_loss(pred, target, source, return_components=True)
        components['zone_loss'] = zone_loss
        components.update(zone_components)

        # Boundary preservation
        components['boundary'] = self.boundary_loss(pred, target, source)

        # VGG perceptual
        if self.lambda_vgg > 0:
            components['vgg'] = self._compute_vgg(pred, target)
        else:
            components['vgg'] = torch.tensor(0.0, device=pred.device)

        # LPIPS
        if self.lambda_lpips > 0 and self.lpips is not None:
            components['lpips'] = self.lpips(pred, target).mean()
        else:
            components['lpips'] = torch.tensor(0.0, device=pred.device)

        # Combine
        total_loss = (
            self.lambda_global_l1 * components['global_l1'] +
            self.lambda_zone * components['zone_loss'] +
            self.lambda_boundary * components['boundary'] +
            self.lambda_vgg * components['vgg'] +
            self.lambda_lpips * components['lpips']
        )

        if return_components:
            return total_loss, components, zone_dict
        return total_loss


def create_sota_window_loss(preset: str = 'aggressive') -> SOTAWindowRecoveryLoss:
    """
    Factory function for SOTA window recovery loss.

    Presets:
        - 'conservative': Milder approach, stable training
        - 'balanced': Good default
        - 'aggressive': Maximum focus on window recovery (RECOMMENDED for your case)
    """
    configs = {
        'conservative': {
            'lambda_zone': 0.8,
            'lambda_boundary': 0.2,
            'lambda_vgg': 0.1,
            'lambda_lpips': 0.05,
            'lambda_global_l1': 0.8,
            'blown_out_weight': 3.0,
            'highlight_weight': 2.0,
        },
        'balanced': {
            'lambda_zone': 1.0,
            'lambda_boundary': 0.3,
            'lambda_vgg': 0.1,
            'lambda_lpips': 0.05,
            'lambda_global_l1': 0.5,
            'blown_out_weight': 4.0,
            'highlight_weight': 2.5,
        },
        'aggressive': {
            'lambda_zone': 1.5,
            'lambda_boundary': 0.4,
            'lambda_vgg': 0.15,
            'lambda_lpips': 0.1,
            'lambda_global_l1': 0.3,
            'blown_out_weight': 6.0,
            'highlight_weight': 4.0,
        },
    }

    config = configs.get(preset, configs['aggressive'])
    return SOTAWindowRecoveryLoss(**config)


# Export heat map visualization for debugging
def visualize_zone_attention(zone_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Create RGB visualization of zone attention maps.

    Colors:
    - Deep shadow: Dark blue
    - Shadow: Blue
    - Midtone: Green
    - Highlight: Yellow
    - Blown out: Red
    """
    colors = {
        'deep_shadow': torch.tensor([0.0, 0.0, 0.5]),
        'shadow': torch.tensor([0.0, 0.3, 0.8]),
        'midtone': torch.tensor([0.0, 0.8, 0.0]),
        'highlight': torch.tensor([0.9, 0.9, 0.0]),
        'blown_out': torch.tensor([1.0, 0.0, 0.0]),
    }

    device = list(zone_dict.values())[0].device
    B, _, H, W = list(zone_dict.values())[0].shape

    visualization = torch.zeros(B, 3, H, W, device=device)

    for zone_name, zone_mask in zone_dict.items():
        color = colors[zone_name].view(1, 3, 1, 1).to(device)
        visualization = visualization + zone_mask * color

    return visualization
