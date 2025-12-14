"""
Window-Aware Losses for Real Estate HDR Enhancement
===================================================

Problem: Models learn to keep windows looking like source (washed out) instead of
transforming them to match target (with color/detail recovery).

Solution: Region-aware losses that:
1. Detect window regions automatically from source brightness
2. Apply HIGHER loss weights on window regions
3. Encourage bold color transformations in windows
4. Use adversarial loss to push for realistic window appearance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class WindowDetector(nn.Module):
    """
    Automatically detect window/highlight regions from source image.

    Windows in source images are typically:
    - Overexposed (bright)
    - Low saturation (washed out)
    - Low local contrast
    """
    def __init__(
        self,
        brightness_threshold: float = 0.65,
        saturation_threshold: float = 0.15,
        use_saturation: bool = True,
        smooth_kernel_size: int = 15,
    ):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold
        self.use_saturation = use_saturation

        # Smoothing kernel for soft masks
        if smooth_kernel_size > 0:
            self.smooth = nn.AvgPool2d(
                smooth_kernel_size,
                stride=1,
                padding=smooth_kernel_size // 2
            )
        else:
            self.smooth = None

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Detect window regions in source image.

        Args:
            source: [B, 3, H, W] source image in [-1, 1] or [0, 1] range

        Returns:
            window_mask: [B, 1, H, W] soft mask, higher values = more likely window
        """
        # Normalize to [0, 1] if needed
        if source.min() < 0:
            source = (source + 1) / 2

        # Compute brightness (luminance)
        brightness = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]

        # Brightness mask (overexposed regions)
        bright_mask = torch.sigmoid(20 * (brightness - self.brightness_threshold))

        if self.use_saturation:
            # Low saturation indicates washed out regions
            # Saturation = max(R,G,B) - min(R,G,B)
            max_rgb = source.max(dim=1, keepdim=True)[0]
            min_rgb = source.min(dim=1, keepdim=True)[0]
            saturation = max_rgb - min_rgb

            # Low saturation mask
            low_sat_mask = torch.sigmoid(20 * (self.saturation_threshold - saturation))

            # Combine: bright AND low saturation
            window_mask = bright_mask * low_sat_mask
        else:
            window_mask = bright_mask

        # Smooth the mask for soft transitions
        if self.smooth is not None:
            window_mask = self.smooth(window_mask)

        return window_mask


class WindowWeightedL1Loss(nn.Module):
    """
    L1 loss with higher weight on detected window regions.
    """
    def __init__(
        self,
        window_weight: float = 5.0,
        base_weight: float = 1.0,
        brightness_threshold: float = 0.65,
    ):
        super().__init__()
        self.window_weight = window_weight
        self.base_weight = base_weight
        self.detector = WindowDetector(brightness_threshold=brightness_threshold)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: predicted image
            target: ground truth
            source: source image (for window detection)

        Returns:
            loss: weighted L1 loss
            window_mask: detected window regions (for visualization)
        """
        window_mask = self.detector(source)

        # Create weight map: higher in windows
        weight_map = self.base_weight + (self.window_weight - self.base_weight) * window_mask

        # Weighted L1
        l1_error = torch.abs(pred - target)
        weighted_loss = (l1_error * weight_map).mean()

        return weighted_loss, window_mask


class WindowColorTransformLoss(nn.Module):
    """
    Encourages bold color transformations in window regions.

    If source->target has color shift in windows, output should too.
    Penalizes outputs that stay too close to source in window regions.
    """
    def __init__(
        self,
        brightness_threshold: float = 0.65,
        transform_weight: float = 2.0,
    ):
        super().__init__()
        self.detector = WindowDetector(brightness_threshold=brightness_threshold)
        self.transform_weight = transform_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize if output windows are too similar to source windows.
        Reward if output windows are similar to target windows.
        """
        window_mask = self.detector(source)

        # Color distance from source to prediction in window regions
        pred_source_dist = torch.abs(pred - source) * window_mask

        # Color distance from target to prediction in window regions
        pred_target_dist = torch.abs(pred - target) * window_mask

        # We want: pred close to target, pred different from source
        # Loss = pred_target_dist - alpha * pred_source_dist
        # But we need to be careful not to encourage arbitrary changes

        # Alternative: encourage pred to move FROM source TOWARDS target
        # i.e., the direction of change should match target's direction

        source_to_target = target - source  # Direction we want
        source_to_pred = pred - source      # Direction model moved

        # In window regions, these should align
        # Use cosine similarity per pixel
        eps = 1e-8

        # Flatten spatial dims for cosine computation
        B, C, H, W = source_to_target.shape

        # Normalize and compute alignment
        st_norm = F.normalize(source_to_target.view(B, C, -1), dim=1, eps=eps)
        sp_norm = F.normalize(source_to_pred.view(B, C, -1), dim=1, eps=eps)

        # Cosine similarity: 1 if aligned, -1 if opposite
        alignment = (st_norm * sp_norm).sum(dim=1)  # [B, H*W]
        alignment = alignment.view(B, 1, H, W)

        # Also consider magnitude: pred should move similar amount as target
        st_mag = torch.norm(source_to_target, dim=1, keepdim=True)
        sp_mag = torch.norm(source_to_pred, dim=1, keepdim=True)

        # Magnitude ratio (want it close to 1)
        mag_ratio = sp_mag / (st_mag + eps)
        mag_loss = torch.abs(mag_ratio - 1.0)

        # Combined: penalize misalignment and magnitude mismatch in windows
        window_transform_loss = window_mask * ((1 - alignment) + 0.5 * mag_loss)

        return self.transform_weight * window_transform_loss.mean()


class WindowPerceptualLoss(nn.Module):
    """
    Perceptual loss computed specifically on window regions.
    Uses VGG features from window crops.
    """
    def __init__(self, vgg_model: nn.Module = None):
        super().__init__()
        self.detector = WindowDetector()

        # Use provided VGG or create one
        if vgg_model is not None:
            self.vgg = vgg_model
        else:
            # Import here to avoid circular deps
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:23]  # Up to relu4_3
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss with window region weighting.
        """
        window_mask = self.detector(source)

        # Normalize for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        # Convert from [-1,1] to [0,1] then normalize
        if pred.min() < 0:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
        else:
            pred_norm = pred
            target_norm = target

        pred_norm = (pred_norm - mean) / std
        target_norm = (target_norm - mean) / std

        # Get features
        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)

        # Downsample window mask to match feature size
        window_mask_down = F.interpolate(
            window_mask,
            size=pred_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Weighted perceptual loss
        feat_diff = torch.abs(pred_feat - target_feat)
        weighted_loss = (feat_diff * window_mask_down).sum() / (window_mask_down.sum() + 1e-8)

        return weighted_loss


class AdversarialWindowLoss(nn.Module):
    """
    Adversarial loss focused on window regions.

    Discriminator learns to distinguish:
    - Real: target window crops
    - Fake: predicted window crops

    This pushes the generator to produce realistic window appearances.
    """
    def __init__(self, discriminator: nn.Module = None):
        super().__init__()
        self.detector = WindowDetector()

        if discriminator is None:
            self.discriminator = self._build_patch_discriminator()
        else:
            self.discriminator = discriminator

    def _build_patch_discriminator(self) -> nn.Module:
        """Simple PatchGAN discriminator."""
        return nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # Output
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def extract_window_patches(
        self,
        images: torch.Tensor,
        window_mask: torch.Tensor,
        patch_size: int = 64,
        num_patches: int = 4
    ) -> torch.Tensor:
        """Extract patches from high-window-probability regions."""
        B, C, H, W = images.shape
        patches = []

        for b in range(B):
            mask = window_mask[b, 0]

            # Find top window locations
            mask_flat = mask.view(-1)
            _, top_indices = torch.topk(mask_flat, min(100, mask_flat.numel()))

            for _ in range(num_patches):
                if len(top_indices) == 0:
                    # Random patch if no windows
                    y = torch.randint(0, H - patch_size, (1,)).item()
                    x = torch.randint(0, W - patch_size, (1,)).item()
                else:
                    # Random from top window locations
                    idx = top_indices[torch.randint(0, len(top_indices), (1,)).item()]
                    y = (idx // W).item()
                    x = (idx % W).item()

                    # Clamp to valid range
                    y = min(max(0, y - patch_size // 2), H - patch_size)
                    x = min(max(0, x - patch_size // 2), W - patch_size)

                patch = images[b:b+1, :, y:y+patch_size, x:x+patch_size]
                patches.append(patch)

        return torch.cat(patches, dim=0) if patches else images[:, :, :patch_size, :patch_size]

    def discriminator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """Loss for training discriminator."""
        window_mask = self.detector(source)

        # Extract window patches
        real_patches = self.extract_window_patches(target, window_mask)
        fake_patches = self.extract_window_patches(pred.detach(), window_mask)

        # Real should be 1, fake should be 0
        real_pred = self.discriminator(real_patches)
        fake_pred = self.discriminator(fake_patches)

        real_loss = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )

        return (real_loss + fake_loss) / 2

    def generator_loss(
        self,
        pred: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """Loss for training generator (fool discriminator)."""
        window_mask = self.detector(source)
        fake_patches = self.extract_window_patches(pred, window_mask)
        fake_pred = self.discriminator(fake_patches)

        # Generator wants discriminator to think fake is real
        return F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )


class ComprehensiveWindowAwareLoss(nn.Module):
    """
    Complete loss function combining all window-aware components.
    """
    def __init__(
        self,
        # Standard loss weights
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.1,
        # Window-specific weights
        lambda_window_l1: float = 3.0,       # Extra weight on windows
        lambda_window_color: float = 1.0,    # Color transform loss
        lambda_window_perceptual: float = 0.2,
        lambda_adversarial: float = 0.1,
        # HDR losses (from hdr_losses.py)
        lambda_gradient: float = 0.15,
        lambda_ssim: float = 0.1,
        # Detection threshold
        brightness_threshold: float = 0.65,
        # Whether to use GAN
        use_adversarial: bool = False,
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_window_l1 = lambda_window_l1
        self.lambda_window_color = lambda_window_color
        self.lambda_window_perceptual = lambda_window_perceptual
        self.lambda_adversarial = lambda_adversarial
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim
        self.use_adversarial = use_adversarial

        # Window detector
        self.window_detector = WindowDetector(brightness_threshold=brightness_threshold)

        # Window-aware losses
        self.window_l1 = WindowWeightedL1Loss(
            window_weight=5.0,
            brightness_threshold=brightness_threshold
        )
        self.window_color = WindowColorTransformLoss(
            brightness_threshold=brightness_threshold
        )

        # Standard losses - will be initialized on first forward
        self.vgg = None
        self.lpips = None

        # Adversarial (optional)
        if use_adversarial:
            self.adversarial = AdversarialWindowLoss()
        else:
            self.adversarial = None

    def _init_perceptual_losses(self, device):
        """Initialize VGG and LPIPS on first use."""
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
                print("Warning: lpips not available, skipping LPIPS loss")
                self.lambda_lpips = 0

    def _compute_vgg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute VGG perceptual loss."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        # Convert from [-1,1] to normalized
        pred_norm = ((pred + 1) / 2 - mean) / std
        target_norm = ((target + 1) / 2 - mean) / std

        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)

        return F.l1_loss(pred_feat, target_feat)

    def _compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Edge preservation loss."""
        def gradient(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy

        pred_dx, pred_dy = gradient(pred)
        target_dx, target_dy = gradient(target)

        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    def _compute_ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SSIM loss."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, 3, 1, 1)
        mu_target = F.avg_pool2d(target, 3, 1, 1)

        sigma_pred = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target

        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

        return 1 - ssim.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute comprehensive window-aware loss.

        Args:
            pred: model output [B, 3, H, W]
            target: ground truth [B, 3, H, W]
            source: input image [B, 3, H, W]
            return_components: if True, return dict of individual losses
        """
        self._init_perceptual_losses(pred.device)

        losses = {}

        # Standard L1
        losses['l1'] = F.l1_loss(pred, target)

        # Window-weighted L1 (extra weight on windows)
        window_l1, window_mask = self.window_l1(pred, target, source)
        losses['window_l1'] = window_l1

        # Window color transform loss
        losses['window_color'] = self.window_color(pred, target, source)

        # VGG perceptual
        if self.lambda_vgg > 0:
            losses['vgg'] = self._compute_vgg_loss(pred, target)
        else:
            losses['vgg'] = torch.tensor(0.0, device=pred.device)

        # LPIPS
        if self.lambda_lpips > 0 and self.lpips is not None:
            losses['lpips'] = self.lpips(pred, target).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=pred.device)

        # Gradient/edge loss
        losses['gradient'] = self._compute_gradient_loss(pred, target)

        # SSIM
        losses['ssim'] = self._compute_ssim_loss(pred, target)

        # Adversarial (optional)
        if self.use_adversarial and self.adversarial is not None:
            losses['adversarial'] = self.adversarial.generator_loss(pred, source)
        else:
            losses['adversarial'] = torch.tensor(0.0, device=pred.device)

        # Combine
        total_loss = (
            self.lambda_l1 * losses['l1'] +
            self.lambda_window_l1 * losses['window_l1'] +
            self.lambda_window_color * losses['window_color'] +
            self.lambda_vgg * losses['vgg'] +
            self.lambda_lpips * losses['lpips'] +
            self.lambda_gradient * losses['gradient'] +
            self.lambda_ssim * losses['ssim'] +
            self.lambda_adversarial * losses['adversarial']
        )

        if return_components:
            return total_loss, losses, window_mask
        return total_loss


def create_window_aware_loss(
    preset: str = 'default',
    use_adversarial: bool = False
) -> ComprehensiveWindowAwareLoss:
    """
    Create window-aware loss with preset configurations.

    Presets:
        - 'default': Balanced configuration
        - 'aggressive_window': Strong focus on window transformation
        - 'conservative': Milder window focus, more stable training
    """
    configs = {
        'default': {
            'lambda_l1': 1.0,
            'lambda_vgg': 0.1,
            'lambda_lpips': 0.05,
            'lambda_window_l1': 3.0,
            'lambda_window_color': 1.0,
            'lambda_window_perceptual': 0.2,
            'lambda_gradient': 0.15,
            'lambda_ssim': 0.1,
            'brightness_threshold': 0.65,
        },
        'aggressive_window': {
            'lambda_l1': 0.5,           # Reduced base L1
            'lambda_vgg': 0.1,
            'lambda_lpips': 0.05,
            'lambda_window_l1': 5.0,     # Higher window weight
            'lambda_window_color': 2.0,  # Strong color transform
            'lambda_window_perceptual': 0.3,
            'lambda_gradient': 0.15,
            'lambda_ssim': 0.1,
            'brightness_threshold': 0.60,  # Detect more regions as windows
        },
        'conservative': {
            'lambda_l1': 1.0,
            'lambda_vgg': 0.1,
            'lambda_lpips': 0.1,
            'lambda_window_l1': 2.0,
            'lambda_window_color': 0.5,
            'lambda_window_perceptual': 0.1,
            'lambda_gradient': 0.15,
            'lambda_ssim': 0.1,
            'brightness_threshold': 0.70,
        },
    }

    config = configs.get(preset, configs['default'])
    config['use_adversarial'] = use_adversarial

    return ComprehensiveWindowAwareLoss(**config)
