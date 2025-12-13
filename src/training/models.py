"""
Neural Network Models for Real Estate HDR Enhancement

Implements:
1. U-Net Generator with residual connections (for image-to-image translation)
2. PatchGAN Discriminator (for adversarial training)
3. Multi-Scale Discriminator (for better high-frequency details)
4. Perceptual loss network (VGG-based)
5. LPIPS loss (learned perceptual metric)
6. LAB Color Loss (perceptually uniform color space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Install with: pip install lpips")


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownBlock(nn.Module):
    """Downsampling block for encoder."""

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block for decoder with skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for image-to-image translation.

    Features:
    - Encoder-decoder architecture with skip connections
    - Residual blocks in bottleneck
    - Instance normalization for stable training
    - Outputs residual (learns the "edit" rather than full image)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 64,
        num_residual_blocks: int = 9,
        learn_residual: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            base_features: Base number of features (doubled at each level)
            num_residual_blocks: Number of residual blocks in bottleneck
            learn_residual: If True, output is added to input (learns the edit)
        """
        super().__init__()
        self.learn_residual = learn_residual

        # Encoder
        self.enc1 = DownBlock(in_channels, base_features, normalize=False)      # 256
        self.enc2 = DownBlock(base_features, base_features * 2)                  # 128
        self.enc3 = DownBlock(base_features * 2, base_features * 4)              # 64
        self.enc4 = DownBlock(base_features * 4, base_features * 8)              # 32
        self.enc5 = DownBlock(base_features * 8, base_features * 8)              # 16
        self.enc6 = DownBlock(base_features * 8, base_features * 8)              # 8
        self.enc7 = DownBlock(base_features * 8, base_features * 8)              # 4
        self.enc8 = DownBlock(base_features * 8, base_features * 8, normalize=False)  # 2

        # Residual blocks at bottleneck
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_features * 8) for _ in range(num_residual_blocks)]
        )

        # Decoder with skip connections
        self.dec1 = UpBlock(base_features * 8, base_features * 8, dropout=0.5)   # 4
        self.dec2 = UpBlock(base_features * 16, base_features * 8, dropout=0.5)  # 8
        self.dec3 = UpBlock(base_features * 16, base_features * 8, dropout=0.5)  # 16
        self.dec4 = UpBlock(base_features * 16, base_features * 8)               # 32
        self.dec5 = UpBlock(base_features * 16, base_features * 4)               # 64
        self.dec6 = UpBlock(base_features * 8, base_features * 2)                # 128
        self.dec7 = UpBlock(base_features * 4, base_features)                    # 256

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_features * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual learning
        input_img = x

        # Encoder path with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Bottleneck with residual blocks
        bottleneck = self.residual_blocks(e8)

        # Decoder path
        d1 = self.dec1(bottleneck, e7)
        d2 = self.dec2(d1, e6)
        d3 = self.dec3(d2, e5)
        d4 = self.dec4(d3, e4)
        d5 = self.dec5(d4, e3)
        d6 = self.dec6(d5, e2)
        d7 = self.dec7(d6, e1)

        output = self.final(d7)

        # Residual learning: output the edit, not the full image
        if self.learn_residual:
            output = output + input_img
            output = torch.clamp(output, -1, 1)

        return output


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Classifies whether 70x70 overlapping patches are real or fake.
    This encourages high-frequency detail preservation.
    """

    def __init__(
        self,
        in_channels: int = 6,  # Concatenated source + target/generated
        base_features: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(
                    base_features * nf_mult_prev,
                    base_features * nf_mult,
                    4, stride=2, padding=1, bias=False
                ),
                nn.InstanceNorm2d(base_features * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        layers += [
            nn.Conv2d(
                base_features * nf_mult_prev,
                base_features * nf_mult,
                4, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm2d(base_features * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_features * nf_mult, 1, 4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: Source image [B, 3, H, W]
            target: Target image (real or generated) [B, 3, H, W]

        Returns:
            Patch predictions [B, 1, H', W']
        """
        x = torch.cat([source, target], dim=1)
        return self.model(x)


class SpectralNormPatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization.

    Spectral normalization constrains the Lipschitz constant of the discriminator,
    preventing it from becoming too powerful and causing training instability.
    This is a proven technique used in BigGAN, StyleGAN, and other stable GANs.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_features: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        # First layer with spectral norm (no instance norm)
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, base_features, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        base_features * nf_mult_prev,
                        base_features * nf_mult,
                        4, stride=2, padding=1, bias=False
                    )
                ),
                # No InstanceNorm with SpectralNorm - they can conflict
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(
                    base_features * nf_mult_prev,
                    base_features * nf_mult,
                    4, stride=1, padding=1, bias=False
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(base_features * nf_mult, 1, 4, stride=1, padding=1)
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([source, target], dim=1)
        return self.model(x)


class SpectralNormMultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator with Spectral Normalization.

    Combines multi-scale discrimination with spectral normalization
    for stable training and good high-frequency detail capture.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_features: int = 64,
        num_scales: int = 2,
    ):
        super().__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            SpectralNormPatchDiscriminator(in_channels, base_features)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(2, stride=2, padding=0, count_include_pad=False)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> List[torch.Tensor]:
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(source, target))
            if i < self.num_scales - 1:
                source = self.downsample(source)
                target = self.downsample(target)
        return outputs


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.

    Compares high-level features to encourage perceptually similar outputs.
    """

    def __init__(
        self,
        feature_layers: List[int] = [3, 8, 15, 22],
        use_input_norm: bool = True,
    ):
        """
        Args:
            feature_layers: VGG layer indices to extract features from
            use_input_norm: Whether to normalize inputs to VGG expected range
        """
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()

        self.slices = nn.ModuleList()
        prev_layer = 0
        for layer in feature_layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_layer:layer]))
            prev_layer = layer

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        self.use_input_norm = use_input_norm
        if use_input_norm:
            # ImageNet normalization (input should be in [0, 1])
            self.register_buffer(
                'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] to ImageNet normalized."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss between generated and target images.

        Args:
            generated: Generated image [B, 3, H, W] in [-1, 1]
            target: Target image [B, 3, H, W] in [-1, 1]

        Returns:
            Perceptual loss (scalar)
        """
        if self.use_input_norm:
            generated = self._normalize(generated)
            target = self._normalize(target)

        loss = 0.0
        gen_features = generated
        tar_features = target

        for slice_module in self.slices:
            gen_features = slice_module(gen_features)
            tar_features = slice_module(tar_features)
            loss += F.l1_loss(gen_features, tar_features)

        return loss


class HDREnhancementModel(nn.Module):
    """
    Complete HDR Enhancement Model combining generator and losses.

    This is the main model class that wraps the generator and provides
    easy-to-use training and inference methods.
    """

    def __init__(
        self,
        image_size: int = 512,
        base_features: int = 64,
        num_residual_blocks: int = 9,
        learn_residual: bool = True,
    ):
        super().__init__()

        self.generator = UNetGenerator(
            in_channels=3,
            out_channels=3,
            base_features=base_features,
            num_residual_blocks=num_residual_blocks,
            learn_residual=learn_residual,
        )

        self.image_size = image_size

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Generate enhanced image from source."""
        return self.generator(source)

    @torch.no_grad()
    def enhance(self, source: torch.Tensor) -> torch.Tensor:
        """
        Enhance images (inference mode).

        Args:
            source: Source images [B, 3, H, W] in [-1, 1]

        Returns:
            Enhanced images [B, 3, H, W] in [-1, 1]
        """
        self.eval()
        return self.generator(source)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator for better high-frequency details.

    Runs multiple discriminators at different scales to capture
    both coarse and fine details in the images.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_features: int = 64,
        num_scales: int = 2,
    ):
        """
        Args:
            in_channels: Input channels (source + target concatenated)
            base_features: Base number of features
            num_scales: Number of scales (discriminators)
        """
        super().__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, base_features)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(2, stride=2, padding=0, count_include_pad=False)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Args:
            source: Source image [B, 3, H, W]
            target: Target image [B, 3, H, W]

        Returns:
            List of discriminator outputs at each scale
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(source, target))
            if i < self.num_scales - 1:
                source = self.downsample(source)
                target = self.downsample(target)
        return outputs


class LPIPSLoss(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) Loss.

    Better than VGG perceptual loss for capturing perceptual similarity.
    Uses a pretrained network (AlexNet) to extract features.
    """

    def __init__(self, net: str = 'alex'):
        """
        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
        """
        super().__init__()

        if not LPIPS_AVAILABLE:
            raise ImportError("lpips not installed. Install with: pip install lpips")

        self.loss_fn = lpips.LPIPS(net=net, verbose=False)

        # Freeze LPIPS network
        for param in self.loss_fn.parameters():
            param.requires_grad = False

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS loss between generated and target images.

        Args:
            generated: Generated image [B, 3, H, W] in [-1, 1]
            target: Target image [B, 3, H, W] in [-1, 1]

        Returns:
            LPIPS loss (scalar)
        """
        return self.loss_fn(generated, target).mean()


def rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to LAB color space.

    Args:
        img: RGB image [B, 3, H, W] in [-1, 1]

    Returns:
        LAB image [B, 3, H, W] where:
            L: [0, 100]
            a: [-128, 127]
            b: [-128, 127]
    """
    # Convert from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    img = img.clamp(0, 1)

    # RGB to XYZ matrix (D65 illuminant)
    # sRGB gamma correction
    mask = img > 0.04045
    img_linear = torch.where(
        mask,
        ((img + 0.055) / 1.055) ** 2.4,
        img / 12.92
    )

    # RGB to XYZ
    r, g, b = img_linear[:, 0:1], img_linear[:, 1:2], img_linear[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize by D65 white point
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3

    fx = torch.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
    fy = torch.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
    fz = torch.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ch = 200 * (fy - fz)

    return torch.cat([L, a, b_ch], dim=1)


class LABColorLoss(nn.Module):
    """
    LAB Color Space Loss.

    Penalizes color differences in LAB space which is perceptually uniform.
    This helps the model learn correct colors more effectively.
    """

    def __init__(self, l_weight: float = 0.5, ab_weight: float = 1.0):
        """
        Args:
            l_weight: Weight for luminance (L) channel loss
            ab_weight: Weight for color (a, b) channels loss
        """
        super().__init__()
        self.l_weight = l_weight
        self.ab_weight = ab_weight

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LAB color loss.

        Args:
            generated: Generated image [B, 3, H, W] in [-1, 1]
            target: Target image [B, 3, H, W] in [-1, 1]

        Returns:
            LAB color loss (scalar)
        """
        gen_lab = rgb_to_lab(generated)
        tar_lab = rgb_to_lab(target)

        # L channel (luminance): 0-100
        l_loss = F.l1_loss(gen_lab[:, 0:1] / 100, tar_lab[:, 0:1] / 100)

        # a, b channels (color): -128 to 127
        ab_loss = F.l1_loss(gen_lab[:, 1:3] / 128, tar_lab[:, 1:3] / 128)

        return self.l_weight * l_loss + self.ab_weight * ab_loss


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity Index) Loss.

    Measures structural similarity between images.
    Returns 1 - SSIM so that lower is better (for use as a loss).
    """

    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        gauss = torch.exp(
            -torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2)
        )
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images."""
        channel = img1.size(1)
        window = self.window

        if img1.is_cuda:
            window = window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            generated: Generated image [B, 3, H, W] in [-1, 1]
            target: Target image [B, 3, H, W] in [-1, 1]

        Returns:
            SSIM loss (1 - SSIM, so lower is better)
        """
        # Convert to [0, 1] for SSIM computation
        gen = (generated + 1) / 2
        tar = (target + 1) / 2
        return 1 - self._ssim(gen, tar)


class FFTLoss(nn.Module):
    """
    FFT (Frequency Domain) Loss.

    Computes L1 loss in the frequency domain to encourage
    sharp textures and reduce artifacts.
    """

    def __init__(self):
        super().__init__()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT loss between generated and target images.

        Args:
            generated: Generated image [B, 3, H, W] in [-1, 1]
            target: Target image [B, 3, H, W] in [-1, 1]

        Returns:
            FFT loss (scalar)
        """
        # Compute 2D FFT
        gen_fft = torch.fft.fft2(generated, norm='ortho')
        tar_fft = torch.fft.fft2(target, norm='ortho')

        # Compute magnitude spectrum
        gen_mag = torch.abs(gen_fft)
        tar_mag = torch.abs(tar_fft)

        # L1 loss on magnitude spectrum
        return F.l1_loss(gen_mag, tar_mag)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (robust L1 variant).

    L = sqrt((x - y)^2 + epsilon^2)

    More robust to outliers than L1, smoother than L1 at zero.
    Commonly used in image restoration tasks.
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Charbonnier loss.

        Args:
            generated: Generated image [B, 3, H, W]
            target: Target image [B, 3, H, W]

        Returns:
            Charbonnier loss (scalar)
        """
        diff = generated - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class ColorHistogramLoss(nn.Module):
    """
    Color Histogram Matching Loss.

    Encourages the output to have a similar color distribution to the target.
    """

    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins

    def compute_histogram(self, img: torch.Tensor) -> torch.Tensor:
        """Compute soft histogram for each channel."""
        # Convert to [0, 1]
        img = (img + 1) / 2

        batch_size, channels, h, w = img.shape
        img = img.view(batch_size, channels, -1)  # [B, C, H*W]

        # Soft histogram using soft binning
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=img.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute distance to each bin center
        img_expanded = img.unsqueeze(-1)  # [B, C, H*W, 1]
        centers_expanded = bin_centers.view(1, 1, 1, -1)  # [1, 1, 1, num_bins]

        # Soft assignment using Gaussian kernel
        sigma = 1.0 / self.num_bins
        weights = torch.exp(-((img_expanded - centers_expanded) ** 2) / (2 * sigma ** 2))

        # Normalize to get histogram
        hist = weights.sum(dim=2)  # [B, C, num_bins]
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

        return hist

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute histogram matching loss."""
        gen_hist = self.compute_histogram(generated)
        tar_hist = self.compute_histogram(target)

        # L1 loss between histograms
        return F.l1_loss(gen_hist, tar_hist)


class EdgeAwareLoss(nn.Module):
    """
    Edge-Aware Loss for preserving sharp edges.

    Critical for real estate photos where furniture edges, walls,
    and window frames need to remain crisp.
    """

    def __init__(self):
        super().__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def _compute_edges(self, img: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters."""
        edges_x = F.conv2d(img, self.sobel_x, padding=1, groups=3)
        edges_y = F.conv2d(img, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge-aware loss.

        Args:
            generated: Generated image [B, 3, H, W] in [-1, 1]
            target: Target image [B, 3, H, W] in [-1, 1]

        Returns:
            Edge-aware loss (scalar)
        """
        # Convert to [0, 1]
        gen = (generated + 1) / 2
        tar = (target + 1) / 2

        # Compute edges
        gen_edges = self._compute_edges(gen)
        tar_edges = self._compute_edges(tar)

        # Edge similarity loss - ensure generated edges match target edges
        edge_loss = F.l1_loss(gen_edges, tar_edges)

        # Edge-weighted pixel loss - penalize more at edges
        edge_weight = 1 + tar_edges.mean(dim=1, keepdim=True)
        weighted_pixel_loss = (torch.abs(gen - tar) * edge_weight).mean()

        return 0.5 * edge_loss + 0.5 * weighted_pixel_loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test generator
    gen = UNetGenerator(learn_residual=True).to(device)
    print(f"Generator parameters: {count_parameters(gen):,}")

    x = torch.randn(1, 3, 512, 512).to(device)
    y = gen(x)
    print(f"Generator input: {x.shape} -> output: {y.shape}")

    # Test discriminator
    disc = PatchDiscriminator().to(device)
    print(f"Discriminator parameters: {count_parameters(disc):,}")

    pred = disc(x, y)
    print(f"Discriminator output: {pred.shape}")

    # Test perceptual loss
    vgg_loss = VGGPerceptualLoss().to(device)
    loss = vgg_loss(y, x)
    print(f"Perceptual loss: {loss.item():.4f}")
