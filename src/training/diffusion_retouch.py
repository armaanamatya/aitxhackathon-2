"""
DiffusionRetouch: Conditional Diffusion Model for Image-to-Image Retouching
============================================================================

A high-quality diffusion model optimized for deterministic image-to-image translation.
Designed to match _src.jpg -> _tar.jpg with maximum accuracy.

Key Features:
- Conditional DDPM with source image as strong conditioning
- Multi-scale U-Net with attention at multiple resolutions
- Deterministic DDIM sampling (eta=0) for exact matching
- Classifier-free guidance for controllable generation
- Additional pixel-level losses for accurate reconstruction
- Progressive refinement from source features

Architecture:
- Encoder: Extract multi-scale features from source image
- U-Net Denoiser: Predict noise conditioned on source + timestep
- Time Embedding: Sinusoidal + MLP projection
- Cross-Attention: Attend to source features at each scale
- Self-Attention: Model long-range dependencies

Training:
- Noise prediction loss (MSE)
- Optional reconstruction losses (L1, LPIPS, SSIM)
- Classifier-free guidance dropout
- EMA for stable outputs

Inference:
- DDIM sampling with configurable steps (50-100)
- Deterministic mode (eta=0) or stochastic (eta>0)
- Optional self-ensemble for maximum quality
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# =============================================================================
# Time Embeddings
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion models."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeEmbedding(nn.Module):
    """Time embedding with MLP projection."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal(t)
        return self.mlp(emb)


# =============================================================================
# Attention Layers
# =============================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention with optional flash attention."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout),
        )
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (heads d) h w -> b heads (h w) d', heads=self.heads)
        k = rearrange(k, 'b (heads d) h w -> b heads (h w) d', heads=self.heads)
        v = rearrange(v, 'b (heads d) h w -> b heads (h w) d', heads=self.heads)

        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=h, w=w)

        return self.to_out(out)


class CrossAttention(nn.Module):
    """Cross-attention to attend to source image features."""

    def __init__(self, dim: int, context_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.norm = nn.GroupNorm(32, dim)
        self.norm_context = nn.GroupNorm(32, context_dim)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(context_dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout),
        )
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        _, cc, ch, cw = context.shape

        x = self.norm(x)
        context = self.norm_context(context)

        # Resize context if needed
        if ch != h or cw != w:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=False)

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (heads d) h w -> b heads (h w) d', heads=self.heads)
        k = rearrange(k, 'b (heads d) h w -> b heads (h w) d', heads=self.heads)
        v = rearrange(v, 'b (heads d) h w -> b heads (h w) d', heads=self.heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=h, w=w)

        return self.to_out(out)


# =============================================================================
# ResNet Blocks
# =============================================================================

class ResnetBlock(nn.Module):
    """Residual block with time embedding modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        groups: int = 32,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time modulation (scale and shift)
        time_out = self.time_mlp(time_emb)
        time_out = rearrange(time_out, 'b c -> b c 1 1')
        scale, shift = time_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# =============================================================================
# Downsampling and Upsampling
# =============================================================================

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# =============================================================================
# Source Image Encoder
# =============================================================================

class SourceEncoder(nn.Module):
    """Multi-scale encoder for source image conditioning."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_mults]
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        prev_ch = base_channels
        for i, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(self._make_res_block(prev_ch, ch))
                prev_ch = ch
            self.encoder_blocks.append(blocks)

            if i < len(channels) - 1:
                self.downsamplers.append(Downsample(ch))
            else:
                self.downsamplers.append(nn.Identity())

        self.out_channels = channels

    def _make_res_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns multi-scale features for cross-attention."""
        features = []

        h = self.init_conv(x)

        for blocks, down in zip(self.encoder_blocks, self.downsamplers):
            for block in blocks:
                # Simplified res block forward
                residual = block[-1](h) if isinstance(block[-1], nn.Conv2d) else h
                for layer in block[:-1]:
                    h = layer(h)
                h = h + residual
            features.append(h)
            h = down(h)

        return features


# =============================================================================
# U-Net Denoiser
# =============================================================================

class UNetDenoiser(nn.Module):
    """
    U-Net denoiser with source image conditioning.

    Architecture:
    - Concatenation conditioning (source + noisy target)
    - Cross-attention to source features at each scale
    - Self-attention at low resolution
    - Time embedding modulation
    """

    def __init__(
        self,
        in_channels: int = 6,  # source + noisy target
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        context_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        time_dim = base_channels * 4
        self.time_emb = TimeEmbedding(base_channels, time_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Channel schedule
        channels = [base_channels * m for m in channel_mults]
        num_levels = len(channels)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_cross_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        prev_ch = base_channels
        current_res = 256  # Assume input resolution

        for level, ch in enumerate(channels):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            cross_attns = nn.ModuleList()

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(prev_ch, ch, time_dim, dropout))

                # Self-attention at certain resolutions
                if current_res in attention_resolutions:
                    attns.append(SelfAttention(ch, num_heads))
                else:
                    attns.append(nn.Identity())

                # Cross-attention to source features
                if use_cross_attention and context_channels:
                    ctx_ch = context_channels[min(level, len(context_channels) - 1)]
                    cross_attns.append(CrossAttention(ch, ctx_ch, num_heads))
                else:
                    cross_attns.append(nn.Identity())

                prev_ch = ch

            self.encoder_blocks.append(blocks)
            self.encoder_attns.append(attns)
            self.encoder_cross_attns.append(cross_attns)

            if level < num_levels - 1:
                self.downsamplers.append(Downsample(ch))
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle block
        self.mid_block1 = ResnetBlock(channels[-1], channels[-1], time_dim, dropout)
        self.mid_attn = SelfAttention(channels[-1], num_heads)
        self.mid_block2 = ResnetBlock(channels[-1], channels[-1], time_dim, dropout)

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_cross_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        channels_rev = list(reversed(channels))

        for level, ch in enumerate(channels_rev):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            cross_attns = nn.ModuleList()

            skip_ch = channels_rev[level]

            for i in range(num_res_blocks + 1):
                in_ch = prev_ch + skip_ch if i == 0 else ch
                blocks.append(ResnetBlock(in_ch, ch, time_dim, dropout))

                if current_res in attention_resolutions:
                    attns.append(SelfAttention(ch, num_heads))
                else:
                    attns.append(nn.Identity())

                if use_cross_attention and context_channels:
                    ctx_idx = num_levels - 1 - level
                    ctx_ch = context_channels[min(ctx_idx, len(context_channels) - 1)]
                    cross_attns.append(CrossAttention(ch, ctx_ch, num_heads))
                else:
                    cross_attns.append(nn.Identity())

                prev_ch = ch

            self.decoder_blocks.append(blocks)
            self.decoder_attns.append(attns)
            self.decoder_cross_attns.append(cross_attns)

            if level < num_levels - 1:
                self.upsamplers.append(Upsample(ch))
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # Output
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Concatenation of source and noisy target [B, 6, H, W]
            t: Timesteps [B]
            source_features: Multi-scale source encoder features
        """
        # Time embedding
        t_emb = self.time_emb(t)

        # Initial conv
        h = self.init_conv(x)

        # Encoder path with skip connections
        skips = []

        for level, (blocks, attns, cross_attns, down) in enumerate(zip(
            self.encoder_blocks, self.encoder_attns, self.encoder_cross_attns, self.downsamplers
        )):
            for block, attn, cross_attn in zip(blocks, attns, cross_attns):
                h = block(h, t_emb)
                h = h + attn(h)
                if source_features is not None:
                    feat_idx = min(level, len(source_features) - 1)
                    h = h + cross_attn(h, source_features[feat_idx])
                skips.append(h)
            h = down(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = h + self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder path
        for level, (blocks, attns, cross_attns, up) in enumerate(zip(
            self.decoder_blocks, self.decoder_attns, self.decoder_cross_attns, self.upsamplers
        )):
            for i, (block, attn, cross_attn) in enumerate(zip(blocks, attns, cross_attns)):
                if skips:
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = h + attn(h)
                if source_features is not None:
                    feat_idx = len(source_features) - 1 - min(level, len(source_features) - 1)
                    h = h + cross_attn(h, source_features[feat_idx])
            h = up(h)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h


# =============================================================================
# Noise Scheduler (DDPM/DDIM)
# =============================================================================

class NoiseScheduler:
    """
    Noise scheduler for DDPM/DDIM.
    Supports linear and cosine schedules.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = 'cosine',
    ):
        self.num_timesteps = num_timesteps

        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            # Cosine schedule from Improved DDPM
            steps = num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)

    def to(self, device: torch.device) -> 'NoiseScheduler':
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance = self.posterior_log_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to x_0 at timestep t: q(x_t | x_0)."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

    def posterior_mean(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute posterior mean q(x_{t-1} | x_t, x_0)."""
        coef1 = self.posterior_mean_coef1[t][:, None, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None, None]
        return coef1 * x_0 + coef2 * x_t


# =============================================================================
# Diffusion Retouch Model
# =============================================================================

class DiffusionRetouch(nn.Module):
    """
    Complete diffusion model for image retouching.

    Combines:
    - Source encoder for conditioning
    - U-Net denoiser
    - Noise scheduler
    - DDIM sampling
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        num_heads: int = 8,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        noise_schedule: str = 'cosine',
        cfg_dropout: float = 0.1,  # Classifier-free guidance dropout
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.cfg_dropout = cfg_dropout

        # Source encoder
        self.source_encoder = SourceEncoder(
            in_channels=in_channels,
            base_channels=base_channels // 2,
            channel_mults=channel_mults,
            num_res_blocks=1,
        )

        # U-Net denoiser
        self.denoiser = UNetDenoiser(
            in_channels=in_channels * 2,  # source + noisy
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            dropout=dropout,
            use_cross_attention=True,
            context_channels=self.source_encoder.out_channels,
        )

        # Noise scheduler
        self.scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            schedule=noise_schedule,
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            source: Source image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            t: Optional timesteps, randomly sampled if None

        Returns:
            Dictionary with noise prediction and ground truth noise
        """
        B = source.shape[0]
        device = source.device

        # Move scheduler to device
        self.scheduler.to(device)

        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(target)

        # Add noise to target
        noisy_target = self.scheduler.add_noise(target, noise, t)

        # Classifier-free guidance: randomly drop source conditioning
        if self.training and self.cfg_dropout > 0:
            mask = torch.rand(B, device=device) < self.cfg_dropout
            source_cond = source.clone()
            source_cond[mask] = 0  # Drop conditioning
        else:
            source_cond = source

        # Get source features
        source_features = self.source_encoder(source_cond)

        # Predict noise
        x_input = torch.cat([source_cond, noisy_target], dim=1)
        noise_pred = self.denoiser(x_input, t, source_features)

        # Predict x_0 for auxiliary losses
        x0_pred = self.scheduler.predict_x0_from_noise(noisy_target, t, noise_pred)

        return {
            'noise_pred': noise_pred,
            'noise': noise,
            'x0_pred': x0_pred,
            'x0': target,
        }

    @torch.no_grad()
    def sample(
        self,
        source: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,  # DDIM: 0 = deterministic, 1 = DDPM
        guidance_scale: float = 1.0,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        DDIM sampling for inference.

        Args:
            source: Source image [B, 3, H, W]
            num_steps: Number of sampling steps
            eta: DDIM eta (0 = deterministic)
            guidance_scale: Classifier-free guidance scale
            return_intermediates: Whether to return intermediate steps

        Returns:
            Generated image [B, 3, H, W]
        """
        B, C, H, W = source.shape
        device = source.device

        # Move scheduler to device
        self.scheduler.to(device)

        # Get source features
        source_features = self.source_encoder(source)

        # Unconditional source features for CFG
        if guidance_scale > 1.0:
            source_uncond = torch.zeros_like(source)
            source_features_uncond = self.source_encoder(source_uncond)

        # Start from pure noise
        x = torch.randn(B, C, H, W, device=device)

        # DDIM sampling schedule
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        intermediates = []

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            x_input = torch.cat([source, x], dim=1)
            noise_pred = self.denoiser(x_input, t_batch, source_features)

            # Classifier-free guidance
            if guidance_scale > 1.0:
                x_input_uncond = torch.cat([source_uncond, x], dim=1)
                noise_pred_uncond = self.denoiser(x_input_uncond, t_batch, source_features_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # Predict x_0
            x_0_pred = self.scheduler.predict_x0_from_noise(x, t_batch, noise_pred)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            if i < len(timesteps) - 1:
                # Get prev timestep
                t_prev = timesteps[i + 1]

                # DDIM update
                alpha_t = self.scheduler.alphas_cumprod[t]
                alpha_prev = self.scheduler.alphas_cumprod[t_prev]

                sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)

                pred_dir = torch.sqrt(1 - alpha_prev - sigma_t ** 2) * noise_pred
                x = torch.sqrt(alpha_prev) * x_0_pred + pred_dir

                if eta > 0:
                    x = x + sigma_t * torch.randn_like(x)
            else:
                x = x_0_pred

            if return_intermediates:
                intermediates.append(x_0_pred)

        if return_intermediates:
            return x, intermediates
        return x


# =============================================================================
# Model Configurations
# =============================================================================

@dataclass
class DiffusionConfig:
    """Configuration for diffusion model."""
    base_channels: int = 128
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    num_heads: int = 8
    dropout: float = 0.1
    num_timesteps: int = 1000
    noise_schedule: str = 'cosine'
    cfg_dropout: float = 0.1


def diffusion_small() -> DiffusionRetouch:
    """Small diffusion model (~15M params)."""
    return DiffusionRetouch(
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        num_heads=4,
        dropout=0.0,
        num_timesteps=1000,
        noise_schedule='cosine',
    )


def diffusion_base() -> DiffusionRetouch:
    """Base diffusion model (~50M params)."""
    return DiffusionRetouch(
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        num_heads=8,
        dropout=0.1,
        num_timesteps=1000,
        noise_schedule='cosine',
    )


def diffusion_large() -> DiffusionRetouch:
    """Large diffusion model (~100M+ params) for maximum quality."""
    return DiffusionRetouch(
        base_channels=192,
        channel_mults=(1, 2, 3, 4, 8),
        num_res_blocks=3,
        attention_resolutions=(64, 32, 16, 8),
        num_heads=8,
        dropout=0.1,
        num_timesteps=1000,
        noise_schedule='cosine',
        cfg_dropout=0.1,
    )


# =============================================================================
# Self-Ensemble Inference
# =============================================================================

@torch.no_grad()
def self_ensemble_sample(
    model: DiffusionRetouch,
    source: torch.Tensor,
    num_steps: int = 50,
    num_augments: int = 8,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Self-ensemble sampling for maximum quality.
    Average predictions from geometric augmentations.
    """
    predictions = []

    augments = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.flip(x, dims=[-1, -2]),
        lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
        lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
        lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
        lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1]),
    ]

    inverse_augments = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.flip(x, dims=[-1, -2]),
        lambda x: torch.rot90(x, k=-1, dims=[-2, -1]),
        lambda x: torch.rot90(x, k=-2, dims=[-2, -1]),
        lambda x: torch.rot90(x, k=-3, dims=[-2, -1]),
        lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=-1, dims=[-2, -1]),
    ]

    for i in range(min(num_augments, len(augments))):
        aug_input = augments[i](source)
        output = model.sample(aug_input, num_steps=num_steps, guidance_scale=guidance_scale)
        predictions.append(inverse_augments[i](output))

    return torch.stack(predictions, dim=0).mean(dim=0)


if __name__ == '__main__':
    # Test model instantiation
    print("Testing DiffusionRetouch models...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test small model
    model = diffusion_small().to(device)
    print(f"Small model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    source = torch.randn(1, 3, 256, 256, device=device)
    target = torch.randn(1, 3, 256, 256, device=device)

    outputs = model(source, target)
    print(f"Noise pred shape: {outputs['noise_pred'].shape}")
    print(f"X0 pred shape: {outputs['x0_pred'].shape}")

    # Test sampling
    with torch.no_grad():
        sample = model.sample(source, num_steps=10)
        print(f"Sample shape: {sample.shape}")

    print("All tests passed!")
