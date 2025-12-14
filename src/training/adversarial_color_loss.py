"""
Adversarial + Style Loss for Real Estate HDR Enhancement

Problem: Real estate HDR images have a specific "style" with:
- Boosted saturation in windows/sky
- Vibrant outdoor colors
- Commercial appeal aesthetic

Solution: Add adversarial loss to force outputs to match the TARGET DISTRIBUTION,
not just minimize pixel-wise error.

Components:
1. PatchGAN Discriminator - Forces local regions to look realistic
2. Style Loss (Gram Matrix) - Matches color/texture distributions
3. Color Histogram Loss - Exact color distribution matching
4. Adversarial Loss - Discriminator feedback

Reference Papers:
- pix2pix (CVPR 2017): PatchGAN for image-to-image
- ESRGAN (ECCV 2018): Perceptual + adversarial for super-resolution
- Real-ESRGAN (ICCV 2021): Real-world image restoration

Author: Top 0.0001% MLE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import torchvision.models as models


# =============================================================================
# PatchGAN Discriminator
# =============================================================================

class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Instead of classifying the whole image as real/fake,
    it classifies each NxN patch. This provides:
    - More training signal (every patch is a training example)
    - Better texture/style learning
    - Works well for image-to-image translation

    Default: 70x70 receptive field
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()

        layers = []

        # First layer (no normalization)
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Second to last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer - output 1 channel (real/fake score per patch)
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================================================================
# Style Loss (Gram Matrix)
# =============================================================================

class StyleLoss(nn.Module):
    """
    Style loss using Gram matrix matching.

    Gram matrix captures the correlation between features,
    which represents the "style" (textures, colors, patterns).

    This is key for matching the HDR editing style, not just pixels.
    """

    def __init__(self, layers: list = None):
        super().__init__()

        if layers is None:
            layers = [3, 8, 15, 22]  # VGG relu layers

        self.layers = layers

        # Load VGG
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        for param in vgg.parameters():
            param.requires_grad = False

        self.slices = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer+1]))
            prev = layer + 1

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style representation."""
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.normalize(pred)
        target = self.normalize(target)

        loss = 0.0
        pred_feat = pred
        target_feat = target

        for slice in self.slices:
            pred_feat = slice(pred_feat)
            target_feat = slice(target_feat)

            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)

            loss += F.mse_loss(pred_gram, target_gram)

        return loss / len(self.slices)


# =============================================================================
# Color Histogram Loss
# =============================================================================

class ColorHistogramLoss(nn.Module):
    """
    Global color histogram matching.

    Forces the COLOR DISTRIBUTION of the output to match the target,
    not just per-pixel colors. This is crucial for style matching.
    """

    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins

    def soft_histogram(self, x: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Compute differentiable soft histogram."""
        B, C, H, W = x.shape

        # Bin centers
        bin_centers = torch.linspace(0, 1, num_bins, device=x.device)
        bin_width = 1.0 / num_bins

        # Reshape for broadcasting
        x_flat = x.view(B, C, -1, 1)  # [B, C, HW, 1]
        bin_centers = bin_centers.view(1, 1, 1, -1)  # [1, 1, 1, num_bins]

        # Soft assignment with RBF kernel
        weights = torch.exp(-((x_flat - bin_centers) ** 2) / (2 * (bin_width ** 2) + 1e-7))

        # Histogram
        hist = weights.sum(dim=2)  # [B, C, num_bins]
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-7)

        return hist

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hist = self.soft_histogram(pred, self.num_bins)
        target_hist = self.soft_histogram(target, self.num_bins)

        # Earth Mover's Distance approximation (L1 on cumulative histograms)
        pred_cum = torch.cumsum(pred_hist, dim=-1)
        target_cum = torch.cumsum(target_hist, dim=-1)

        emd = F.l1_loss(pred_cum, target_cum)

        return emd


# =============================================================================
# Perceptual Loss (for completeness)
# =============================================================================

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        for param in vgg.parameters():
            param.requires_grad = False

        # Use features up to relu3_3 (layer 15)
        self.features = nn.Sequential(*list(vgg.children())[:16])

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        pred_feat = self.features(pred)
        target_feat = self.features(target)

        return F.l1_loss(pred_feat, target_feat)


# =============================================================================
# Complete Adversarial Style Loss
# =============================================================================

class AdversarialStyleLoss(nn.Module):
    """
    Complete loss for learning HDR style with adversarial training.

    Components:
    - L1 (1.0): Pixel accuracy
    - Perceptual (0.1): Feature similarity
    - Style (0.1): Gram matrix matching (style distribution)
    - Histogram (0.1): Color distribution matching
    - Adversarial (0.01): Discriminator feedback

    The adversarial loss is small (0.01) but crucial for pushing
    outputs toward the target DISTRIBUTION, not just average.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        style_weight: float = 0.1,
        histogram_weight: float = 0.1,
        adversarial_weight: float = 0.01,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.histogram_weight = histogram_weight
        self.adversarial_weight = adversarial_weight

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.histogram_loss = ColorHistogramLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_pred: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute generator loss.

        Args:
            pred: Generator output
            target: Ground truth
            disc_pred: Discriminator output on pred (for adversarial loss)
        """
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        style = self.style_loss(pred, target)
        histogram = self.histogram_loss(pred, target)

        total = (
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.style_weight * style +
            self.histogram_weight * histogram
        )

        # Adversarial loss (generator wants discriminator to output 1 for pred)
        adv_loss = torch.tensor(0.0, device=pred.device)
        if disc_pred is not None:
            adv_loss = F.binary_cross_entropy_with_logits(
                disc_pred,
                torch.ones_like(disc_pred)
            )
            total = total + self.adversarial_weight * adv_loss

        components = {
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'style': style.item(),
            'histogram': histogram.item(),
            'adversarial': adv_loss.item(),
            'total': total.item(),
        }

        return total, components


def discriminator_loss(
    disc_real: torch.Tensor,
    disc_fake: torch.Tensor
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute discriminator loss.

    Real images should be classified as 1, fake as 0.
    """
    real_loss = F.binary_cross_entropy_with_logits(
        disc_real,
        torch.ones_like(disc_real)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        disc_fake,
        torch.zeros_like(disc_fake)
    )

    total = (real_loss + fake_loss) / 2

    components = {
        'real': real_loss.item(),
        'fake': fake_loss.item(),
        'total': total.item(),
    }

    return total, components


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Adversarial Style Loss...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test discriminator
    print("\n1. PatchDiscriminator:")
    disc = PatchDiscriminator().to(device)
    x = torch.rand(2, 3, 256, 256).to(device)
    out = disc(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")

    # Test losses
    print("\n2. Individual losses:")
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)

    style_loss = StyleLoss().to(device)
    print(f"   Style Loss: {style_loss(pred, target).item():.4f}")

    hist_loss = ColorHistogramLoss().to(device)
    print(f"   Histogram Loss: {hist_loss(pred, target).item():.4f}")

    # Test complete loss
    print("\n3. AdversarialStyleLoss:")
    loss_fn = AdversarialStyleLoss().to(device)
    disc_pred = disc(pred)
    total, components = loss_fn(pred, target, disc_pred)
    print(f"   Total: {total.item():.4f}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")

    print("\nAll tests passed!")
