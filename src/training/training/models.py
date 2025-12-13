"""
Neural Network Models for Real Estate HDR Enhancement

Implements:
1. U-Net Generator with residual connections (for image-to-image translation)
2. PatchGAN Discriminator (for adversarial training)
3. Perceptual loss network (VGG-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional


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
