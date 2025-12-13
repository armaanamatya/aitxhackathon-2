"""
ControlNet for Real Estate HDR Photo Enhancement
=================================================

ControlNet architecture adapted for image-to-image translation.
Uses the source (unedited) image as conditioning to guide the
generation of the professionally edited output.

Key Features:
- Zero-convolution initialization for stable training
- Multi-scale conditioning injection
- Compatible with pretrained U-Net weights
- Efficient for fine-tuning on small datasets

Reference:
    "Adding Conditional Control to Text-to-Image Diffusion Models"
    https://arxiv.org/abs/2302.05543
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import copy


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    This is the key initialization for ControlNet stability.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ZeroConv2d(nn.Module):
    """
    Convolution layer initialized to zero.
    Ensures ControlNet starts with no influence and gradually learns.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=True
        )
        # Zero initialization
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ControlNetConditioningEncoder(nn.Module):
    """
    Encoder for the conditioning image (source/unedited photo).
    Extracts multi-scale features to inject into the main network.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Progressive downsampling with increasing channels
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.blocks = nn.ModuleList([
            # 512 -> 256
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
            ),
            # 256 -> 128
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
            ),
            # 128 -> 64
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
            ),
            # 64 -> 32
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 8),
                nn.SiLU(),
                nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
                nn.GroupNorm(8, base_channels * 8),
                nn.SiLU(),
            ),
        ])

        # Zero convolutions for each scale
        self.zero_convs = nn.ModuleList([
            ZeroConv2d(base_channels, base_channels),
            ZeroConv2d(base_channels * 2, base_channels * 2),
            ZeroConv2d(base_channels * 4, base_channels * 4),
            ZeroConv2d(base_channels * 8, base_channels * 8),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale conditioning features.

        Args:
            x: Conditioning image [B, 3, H, W]

        Returns:
            List of features at each scale, processed through zero convs
        """
        h = self.conv_in(x)
        outputs = []

        for block, zero_conv in zip(self.blocks, self.zero_convs):
            h = block(h)
            outputs.append(zero_conv(h))

        return outputs


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or channels

        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.skip = nn.Identity() if channels == out_channels else nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class DownBlock(nn.Module):
    """Downsampling block for encoder."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, control: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features
            control: Optional ControlNet conditioning to add

        Returns:
            (downsampled output, skip connection)
        """
        for res_block in self.res_blocks:
            x = res_block(x)

        if control is not None:
            x = x + control

        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for decoder."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels + out_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        for res_block in self.res_blocks:
            x = res_block(x)

        return x


class ControlNetHDR(nn.Module):
    """
    ControlNet for HDR Photo Enhancement.

    Uses the source image as conditioning to guide the generation
    of professionally edited output. The architecture follows the
    ControlNet paradigm with zero-initialized control injection.

    Args:
        in_channels: Input image channels (3 for RGB)
        out_channels: Output image channels (3 for RGB)
        base_channels: Base channel count (doubled at each level)
        num_res_blocks: Number of residual blocks per level
        learn_residual: If True, learns the edit (output + input)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        learn_residual: bool = True,
    ):
        super().__init__()
        self.learn_residual = learn_residual

        # ControlNet conditioning encoder
        self.control_encoder = ControlNetConditioningEncoder(in_channels, base_channels)

        # Main U-Net encoder
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        ch = base_channels
        self.down1 = DownBlock(ch, ch, num_res_blocks)          # 512 -> 256
        self.down2 = DownBlock(ch, ch * 2, num_res_blocks)      # 256 -> 128
        self.down3 = DownBlock(ch * 2, ch * 4, num_res_blocks)  # 128 -> 64
        self.down4 = DownBlock(ch * 4, ch * 8, num_res_blocks)  # 64 -> 32

        # Bottleneck
        self.mid = nn.Sequential(
            ResidualBlock(ch * 8),
            ResidualBlock(ch * 8),
            ResidualBlock(ch * 8),
        )

        # Decoder
        self.up4 = UpBlock(ch * 8, ch * 4, num_res_blocks)      # 32 -> 64
        self.up3 = UpBlock(ch * 4, ch * 2, num_res_blocks)      # 64 -> 128
        self.up2 = UpBlock(ch * 2, ch, num_res_blocks)          # 128 -> 256
        self.up1 = UpBlock(ch, ch, num_res_blocks)              # 256 -> 512

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ControlNet conditioning.

        Args:
            x: Source image [B, 3, H, W] in [-1, 1]

        Returns:
            Enhanced image [B, 3, H, W] in [-1, 1]
        """
        # Store input for residual learning
        input_img = x

        # Extract multi-scale control features
        control_features = self.control_encoder(x)

        # Encoder
        h = self.conv_in(x)
        h, s1 = self.down1(h, control_features[0])  # Inject control at each scale
        h, s2 = self.down2(h, control_features[1])
        h, s3 = self.down3(h, control_features[2])
        h, s4 = self.down4(h, control_features[3])

        # Bottleneck
        h = self.mid(h)

        # Decoder with skip connections
        h = self.up4(h, s4)
        h = self.up3(h, s3)
        h = self.up2(h, s2)
        h = self.up1(h, s1)

        # Output
        output = self.conv_out(h)

        # Residual learning
        if self.learn_residual:
            output = output + input_img
            output = torch.clamp(output, -1, 1)

        return output


class ControlNetLite(nn.Module):
    """
    Lightweight ControlNet variant for faster inference.

    Uses fewer channels and simpler architecture while
    maintaining the key ControlNet principles.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        learn_residual: bool = True,
    ):
        super().__init__()
        self.learn_residual = learn_residual

        # Simplified control encoder
        self.control_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
        )

        # Zero convolution for control injection
        self.control_zero = ZeroConv2d(base_channels * 4, base_channels * 4)

        # Main network (lightweight U-Net style)
        ch = base_channels

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1),
            nn.SiLU(),
        )

        # Bottleneck (with control injection)
        self.mid = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1),
            nn.SiLU(),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch, 4, stride=2, padding=1),  # ch*4 due to skip
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1),  # ch*2 due to skip
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_img = x

        # Extract control features
        control = self.control_encoder(x)
        control = self.control_zero(control)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck with control injection
        h = self.mid(e3 + control)

        # Decoder with skip connections
        h = self.dec3(h)
        h = torch.cat([h, e2], dim=1)
        h = self.dec2(h)
        h = torch.cat([h, e1], dim=1)
        output = self.dec1(h)

        # Residual learning
        if self.learn_residual:
            output = output + input_img
            output = torch.clamp(output, -1, 1)

        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing ControlNet models...")
    print(f"Device: {device}")
    print("")

    x = torch.randn(1, 3, 512, 512).to(device)

    # Test ControlNetHDR
    print("ControlNetHDR:")
    model = ControlNetHDR().to(device)
    print(f"  Parameters: {count_parameters(model):,}")
    y = model(x)
    print(f"  Input: {x.shape} -> Output: {y.shape}")

    # Test ControlNetLite
    print("\nControlNetLite:")
    model_lite = ControlNetLite().to(device)
    print(f"  Parameters: {count_parameters(model_lite):,}")
    y_lite = model_lite(x)
    print(f"  Input: {x.shape} -> Output: {y_lite.shape}")

    print("\nAll tests passed!")
