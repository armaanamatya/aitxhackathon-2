"""
Context-Aware Implicit Neural Retouching Network

Inspired by INRetouch (WACV 2026) but designed for multi-image learning.
Combines:
- Context-aware processing with spatial awareness
- Implicit Neural Representation for color transformation
- Multi-scale feature extraction
- Global and local editing capabilities

Optimized for maximum quality on paired before/after image datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# =============================================================================
# Positional Encoding
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    2D sinusoidal positional encoding for spatial awareness.
    Encodes (x, y) coordinates into high-dimensional features.
    """

    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Output dim = 2 (x,y) * 2 (sin,cos) * num_frequencies + 2 (original coords)
        self.output_dim = 2 + 4 * num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2, H, W) normalized coordinates in [-1, 1]
        Returns:
            (B, output_dim, H, W) positional encoding
        """
        encodings = [coords]

        for i in range(self.num_frequencies):
            freq = 2.0 ** i * math.pi
            encodings.append(torch.sin(freq * coords))
            encodings.append(torch.cos(freq * coords))

        return torch.cat(encodings, dim=1)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding using 1x1 convolutions.
    More flexible than fixed sinusoidal encoding.
    """

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, output_dim, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.encoder(coords)


def create_coordinate_grid(batch_size: int, height: int, width: int,
                           device: torch.device) -> torch.Tensor:
    """Create normalized coordinate grid."""
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([x_grid, y_grid], dim=0)  # (2, H, W)
    return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, 2, H, W)


# =============================================================================
# Context Encoder
# =============================================================================

class ContextBlock(nn.Module):
    """
    Context-aware block using depthwise separable convolutions.
    Captures local texture, edges, and semantic context efficiently.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        # 1x1 -> 3x3 depthwise -> 1x1 (from INRetouch)
        self.pw1 = nn.Conv2d(in_channels, out_channels, 1)
        self.dw = nn.Conv2d(out_channels, out_channels, kernel_size,
                           padding=kernel_size//2, groups=out_channels)
        self.pw2 = nn.Conv2d(out_channels, out_channels, 1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.act(x)
        x = self.dw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class MultiScaleContextEncoder(nn.Module):
    """
    Multi-scale context encoder for extracting rich semantic features.
    Uses a U-Net style architecture for capturing both local and global context.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_scales: int = 4, context_dim: int = 128):
        super().__init__()

        self.num_scales = num_scales

        # Encoder path
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = in_channels
        for i in range(num_scales):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(nn.Sequential(
                ContextBlock(ch, out_ch),
                ContextBlock(out_ch, out_ch),
            ))
            if i < num_scales - 1:
                self.downsamples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(ch, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Decoder path for multi-scale features
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(num_scales - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.decoders.append(nn.Sequential(
                ContextBlock(in_ch + out_ch, out_ch),
                ContextBlock(out_ch, out_ch),
            ))

        # Final projection to context features
        self.final_proj = nn.Conv2d(base_channels, context_dim, 1)
        self.context_dim = context_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            spatial_context: (B, context_dim, H, W) - spatial context features
            global_context: (B, context_dim) - global image context
        """
        # Encoder
        features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            features.append(x)
            if i < self.num_scales - 1:
                x = self.downsamples[i](x)

        # Global context
        global_feat = self.global_pool(x).flatten(1)
        global_context = self.global_fc(global_feat)

        # Decoder
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            skip = features[self.num_scales - 2 - i]
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        spatial_context = self.final_proj(x)

        return spatial_context, global_context


# =============================================================================
# Implicit Neural Representation
# =============================================================================

class INRBlock(nn.Module):
    """
    INR block using 1x1 convolutions for efficient image-space processing.
    Processes entire image as spatial tensor rather than individual pixels.
    """

    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()

        self.use_residual = use_residual and (in_channels == out_channels)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.use_residual:
            out = out + x
        return out


class ContextModulatedINR(nn.Module):
    """
    Context-modulated Implicit Neural Representation.
    Uses FiLM (Feature-wise Linear Modulation) to inject context into INR.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 context_dim: int = 128, num_layers: int = 6):
        super().__init__()

        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 1)

        # INR layers with FiLM modulation
        self.inr_layers = nn.ModuleList()
        self.film_generators = nn.ModuleList()

        for i in range(num_layers):
            self.inr_layers.append(INRBlock(hidden_dim, hidden_dim))
            # FiLM: generates scale (gamma) and shift (beta)
            self.film_generators.append(nn.Sequential(
                nn.Conv2d(context_dim, hidden_dim * 2, 1),
            ))

        self.output_proj = nn.Conv2d(hidden_dim, 3, 1)

    def forward(self, x: torch.Tensor, spatial_context: torch.Tensor,
                global_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, H, W) - input features (pos encoding + RGB)
            spatial_context: (B, context_dim, H, W) - spatial context
            global_context: (B, context_dim) - global context
        """
        # Combine spatial and global context
        B, C, H, W = spatial_context.shape
        global_expanded = global_context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        context = spatial_context + global_expanded

        # Process through INR with FiLM modulation
        h = self.input_proj(x)

        for inr_layer, film_gen in zip(self.inr_layers, self.film_generators):
            # Generate FiLM parameters
            film_params = film_gen(context)
            gamma, beta = film_params.chunk(2, dim=1)

            # Apply INR layer
            h = inr_layer(h)

            # FiLM modulation
            h = gamma * h + beta

        return self.output_proj(h)


# =============================================================================
# Color Transform Modules
# =============================================================================

class GlobalColorTransform(nn.Module):
    """
    Learns global color transformation parameters.
    Outputs 3x4 affine color matrix for global color grading.
    """

    def __init__(self, context_dim: int = 128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 12),  # 3x4 affine matrix
        )

        # Initialize to identity transform
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data = torch.tensor([
            1, 0, 0, 0,  # R
            0, 1, 0, 0,  # G
            0, 0, 1, 0,  # B
        ], dtype=torch.float32)

    def forward(self, image: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        """
        Apply learned global color transform.
        """
        B, C, H, W = image.shape

        # Get affine matrix
        params = self.fc(global_context)
        matrix = params.view(B, 3, 4)

        # Apply affine transform: output = matrix @ [R, G, B, 1]
        image_flat = image.view(B, 3, -1)  # (B, 3, H*W)
        ones = torch.ones(B, 1, H * W, device=image.device)
        image_homo = torch.cat([image_flat, ones], dim=1)  # (B, 4, H*W)

        output_flat = torch.bmm(matrix, image_homo)  # (B, 3, H*W)
        output = output_flat.view(B, 3, H, W)

        return output


class Local3DLUT(nn.Module):
    """
    Learnable 3D LUT with spatial adaptation.
    Combines efficiency of LUT with spatial awareness.
    """

    def __init__(self, lut_dim: int = 33, context_dim: int = 128):
        super().__init__()

        self.lut_dim = lut_dim

        # Base 3D LUT (identity initialized)
        self.base_lut = nn.Parameter(self._create_identity_lut(lut_dim))

        # Spatial LUT offset predictor
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(context_dim, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 3, 1),  # RGB offset
            nn.Tanh(),
        )

        self.offset_scale = nn.Parameter(torch.tensor(0.1))

    def _create_identity_lut(self, dim: int) -> torch.Tensor:
        """Create identity 3D LUT."""
        coords = torch.linspace(0, 1, dim)
        r, g, b = torch.meshgrid(coords, coords, coords, indexing='ij')
        lut = torch.stack([r, g, b], dim=-1)  # (D, D, D, 3)
        return lut

    def forward(self, image: torch.Tensor, spatial_context: torch.Tensor) -> torch.Tensor:
        """Apply 3D LUT with spatial adaptation."""
        B, C, H, W = image.shape

        # Get spatial offset
        offset = self.offset_predictor(spatial_context) * self.offset_scale

        # Apply base LUT via trilinear interpolation
        # Normalize image to [0, 1] for LUT lookup
        image_norm = (image + 1) / 2  # [-1,1] -> [0,1]
        image_norm = image_norm.clamp(0, 1)

        # Reshape for grid_sample
        # image: (B, 3, H, W) -> (B, H, W, 3) for grid_sample
        image_grid = image_norm.permute(0, 2, 3, 1)  # (B, H, W, 3)

        # Scale to [-1, 1] for grid_sample
        image_grid = image_grid * 2 - 1

        # Expand LUT for batch
        lut = self.base_lut.unsqueeze(0).expand(B, -1, -1, -1, -1)
        lut = lut.permute(0, 4, 1, 2, 3)  # (B, 3, D, D, D)

        # 3D grid sample
        image_grid = image_grid.unsqueeze(1)  # (B, 1, H, W, 3)
        output = F.grid_sample(lut, image_grid, mode='bilinear',
                              padding_mode='border', align_corners=True)
        output = output.squeeze(2)  # (B, 3, H, W)

        # Back to [-1, 1]
        output = output * 2 - 1

        # Add spatial offset
        output = output + offset

        return output


# =============================================================================
# Main Model
# =============================================================================

class ContextAwareRetouchNet(nn.Module):
    """
    Context-Aware Retouching Network.

    Combines multiple retouching strategies:
    1. Context-modulated INR for complex local edits
    2. Global color transform for overall color grading
    3. Optional 3D LUT for efficient color mapping

    Uses global residual connection for intensity control.
    """

    def __init__(
        self,
        base_channels: int = 64,
        context_dim: int = 128,
        inr_hidden_dim: int = 256,
        inr_num_layers: int = 6,
        num_scales: int = 4,
        pos_encoding_freqs: int = 10,
        use_3dlut: bool = True,
        use_global_transform: bool = True,
    ):
        super().__init__()

        self.use_3dlut = use_3dlut
        self.use_global_transform = use_global_transform

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(pos_encoding_freqs)
        pos_dim = self.pos_encoder.output_dim

        # Context encoder
        self.context_encoder = MultiScaleContextEncoder(
            in_channels=3,
            base_channels=base_channels,
            num_scales=num_scales,
            context_dim=context_dim,
        )

        # INR input: RGB (3) + positional encoding
        inr_input_dim = 3 + pos_dim

        # Context-modulated INR
        self.inr = ContextModulatedINR(
            input_dim=inr_input_dim,
            hidden_dim=inr_hidden_dim,
            context_dim=context_dim,
            num_layers=inr_num_layers,
        )

        # Optional components
        if use_global_transform:
            self.global_transform = GlobalColorTransform(context_dim)

        if use_3dlut:
            self.local_lut = Local3DLUT(lut_dim=33, context_dim=context_dim)

        # Fusion weights (learnable)
        num_branches = 1 + int(use_global_transform) + int(use_3dlut)
        self.fusion_weights = nn.Parameter(torch.ones(num_branches) / num_branches)

        # Global residual scale (intensity control)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image in [-1, 1]
            alpha: intensity control for residual connection

        Returns:
            (B, 3, H, W) retouched image in [-1, 1]
        """
        B, C, H, W = x.shape

        # Create coordinate grid
        coords = create_coordinate_grid(B, H, W, x.device)

        # Positional encoding
        pos_encoding = self.pos_encoder(coords)

        # Get context features
        spatial_context, global_context = self.context_encoder(x)

        # INR input
        inr_input = torch.cat([x, pos_encoding], dim=1)

        # Get INR output
        inr_output = self.inr(inr_input, spatial_context, global_context)

        # Collect all branch outputs
        outputs = [inr_output]

        if self.use_global_transform:
            global_output = self.global_transform(x, global_context)
            outputs.append(global_output)

        if self.use_3dlut:
            lut_output = self.local_lut(x, spatial_context)
            outputs.append(lut_output)

        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, outputs))

        # Global residual with intensity control
        output = x + alpha * self.residual_scale * (fused - x)

        # Clamp to valid range
        output = output.clamp(-1, 1)

        return output


# =============================================================================
# Model Variants
# =============================================================================

def inretouch_small(pretrained: bool = False) -> ContextAwareRetouchNet:
    """Small variant (~2M params) - fast training/inference."""
    return ContextAwareRetouchNet(
        base_channels=32,
        context_dim=64,
        inr_hidden_dim=128,
        inr_num_layers=4,
        num_scales=3,
        pos_encoding_freqs=6,
        use_3dlut=False,
        use_global_transform=True,
    )


def inretouch_base(pretrained: bool = False) -> ContextAwareRetouchNet:
    """Base variant (~8M params) - balanced quality/speed."""
    return ContextAwareRetouchNet(
        base_channels=64,
        context_dim=128,
        inr_hidden_dim=256,
        inr_num_layers=6,
        num_scales=4,
        pos_encoding_freqs=10,
        use_3dlut=True,
        use_global_transform=True,
    )


def inretouch_large(pretrained: bool = False) -> ContextAwareRetouchNet:
    """Large variant (~20M params) - maximum quality."""
    return ContextAwareRetouchNet(
        base_channels=96,
        context_dim=192,
        inr_hidden_dim=384,
        inr_num_layers=8,
        num_scales=5,
        pos_encoding_freqs=12,
        use_3dlut=True,
        use_global_transform=True,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test all variants
    for name, model_fn in [("small", inretouch_small),
                           ("base", inretouch_base),
                           ("large", inretouch_large)]:
        model = model_fn()
        params = count_parameters(model)
        print(f"INRetouch-{name}: {params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        print(f"  Input: {x.shape} -> Output: {y.shape}")
        print()
