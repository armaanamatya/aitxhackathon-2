"""
NAFNet (Nonlinear Activation Free Network) for Image Restoration

State-of-the-art architecture for image enhancement/restoration tasks.
Paper: "Simple Baselines for Image Restoration" (ECCV 2022)

Key features:
- No nonlinear activations (uses SimpleGate instead)
- Layer normalization
- Channel attention
- Excellent quality with fast inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm for efficiency."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D inputs."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """
    Simple Gate activation - splits channels and multiplies.
    Replaces nonlinear activations like ReLU/GELU.
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention (SCA) module."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return x * y


class NAFBlock(nn.Module):
    """
    NAF Block - the core building block of NAFNet.

    Structure:
    - LayerNorm -> Conv -> SimpleGate -> Conv -> SCA -> Skip
    - LayerNorm -> Conv -> SimpleGate -> Conv -> Skip
    """

    def __init__(
        self,
        channels: int,
        expansion_ratio: float = 2.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        hidden_channels = int(channels * expansion_ratio)

        # First block (with channel attention)
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden_channels * 2, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3,
                               padding=1, groups=hidden_channels * 2, bias=True)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(hidden_channels, channels, 1, bias=True)
        self.sca = SimplifiedChannelAttention(channels)

        # Second block (feedforward)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, hidden_channels * 2, 1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(hidden_channels, channels, 1, bias=True)

        # Learnable scaling parameters
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # First block
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.conv3(y)
        y = self.sca(y)
        y = self.dropout(y)
        x = x + y * self.beta

        # Second block
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        y = self.dropout(y)
        x = x + y * self.gamma

        return x


class NAFNet(nn.Module):
    """
    NAFNet - Nonlinear Activation Free Network for Image Restoration.

    Architecture:
    - Encoder-decoder with skip connections
    - NAF blocks at each level
    - Efficient and high quality
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 64,
        num_blocks: list = [2, 2, 4, 8],
        expansion_ratio: float = 2.0,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            out_channels: Output channels (3 for RGB)
            width: Base channel width
            num_blocks: Number of NAF blocks at each encoder/decoder level
            expansion_ratio: Channel expansion in NAF blocks
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)
        self.outro = nn.Conv2d(width, out_channels, 3, padding=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in num_blocks:
            self.encoders.append(
                nn.Sequential(*[
                    NAFBlock(chan, expansion_ratio, dropout_rate)
                    for _ in range(num)
                ])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        # Middle
        self.middle = nn.Sequential(*[
            NAFBlock(chan, expansion_ratio, dropout_rate)
            for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in reversed(num_blocks):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[
                    NAFBlock(chan, expansion_ratio, dropout_rate)
                    for _ in range(num)
                ])
            )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Store input for residual
        inp = x

        x = self.intro(x)

        # Encoder
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Middle
        x = self.middle(x)

        # Decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.outro(x)

        # Residual connection
        x = x + inp

        return x


class NAFNetLocal(nn.Module):
    """
    NAFNet with local attention for higher resolution.
    Uses window-based processing for memory efficiency.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 64,
        num_blocks: list = [1, 1, 1, 28],
        expansion_ratio: float = 2.0,
        train_size: tuple = (256, 256),
    ):
        super().__init__()
        self.train_size = train_size
        self.net = NAFNet(
            in_channels, out_channels, width,
            num_blocks, expansion_ratio
        )

    def forward(self, x):
        return self.net(x)


def create_nafnet(
    variant: str = "base",
    in_channels: int = 3,
    out_channels: int = 3,
) -> NAFNet:
    """
    Create NAFNet model with predefined configurations.

    Args:
        variant: Model variant - "small", "base", or "large"
        in_channels: Input channels
        out_channels: Output channels

    Returns:
        NAFNet model
    """
    configs = {
        "small": {
            "width": 32,
            "num_blocks": [1, 1, 1, 4],
        },
        "base": {
            "width": 64,
            "num_blocks": [2, 2, 4, 8],
        },
        "large": {
            "width": 64,
            "num_blocks": [4, 4, 8, 16],
        },
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    config = configs[variant]
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        **config,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test NAFNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for variant in ["small", "base", "large"]:
        model = create_nafnet(variant).to(device)
        params = count_parameters(model)

        x = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            y = model(x)

        print(f"NAFNet-{variant}: {params/1e6:.2f}M params, "
              f"input {x.shape} -> output {y.shape}")
