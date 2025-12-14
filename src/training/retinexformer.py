"""
Retinexformer for Real Estate HDR Enhancement

Based on "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement"
ICCV 2023 + ECCV 2024 Enhanced Version

Key advantages for HDR/window preservation:
1. Explicit illumination estimation - learns different treatment for bright/dark regions
2. Illumination-guided attention - naturally handles windows vs interior
3. Physics-based (Retinex theory) - interpretable decomposition

Architecture:
- Illumination Estimator: Estimates illumination map and features
- IG-MSA: Illumination-Guided Multi-head Self-Attention
- Denoiser: U-Net with IGAB blocks, uses illumination as guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from typing import Optional, Tuple, List


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Illumination_Estimator(nn.Module):
    """
    Estimates illumination map from input image.

    This is KEY for window preservation:
    - Learns to identify bright regions (windows) vs dark regions (interior)
    - Outputs illumination features used to guide attention
    - Outputs illumination map for initial brightness adjustment
    """
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2,
            bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img: [b, 3, h, w]
        # Compute mean channel as additional input
        mean_c = img.mean(dim=1).unsqueeze(1)  # [b, 1, h, w]
        input = torch.cat([img, mean_c], dim=1)  # [b, 4, h, w]

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)  # Illumination features
        illu_map = self.conv2(illu_fea)  # Illumination map

        return illu_fea, illu_map


class IG_MSA(nn.Module):
    """
    Illumination-Guided Multi-head Self-Attention.

    Uses illumination features to modulate attention values.
    This allows different attention for bright vs dark regions.
    """
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b, h, w, c]
        illu_fea_trans: [b, h, w, c] - illumination features
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2))
        )

        # KEY: Modulate values by illumination
        v = v * illu_attn

        # Transposed attention
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        out = out_c + out_p
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """Illumination-Guided Attention Block."""
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b, c, h, w]
        illu_fea: [b, c, h, w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    """
    U-Net style denoiser with Illumination-Guided Attention Blocks.

    Uses illumination features at each scale to guide restoration.
    """
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim,
                     heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim,
            num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
                     dim_head=dim, heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB_layer, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB_layer(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlock(fea, illu_fea)

        # Output with residual
        out = self.mapping(fea) + x
        return out


class RetinexFormer(nn.Module):
    """
    Retinexformer: One-stage Retinex-based Transformer

    Process:
    1. Estimate illumination map and features
    2. Apply illumination adjustment: input_img = img * illu_map + img
    3. Denoise/enhance with illumination-guided attention
    4. Output residual learning

    For real estate HDR:
    - Illumination estimator learns window (bright) vs interior (dark)
    - Guided attention treats them differently
    - Preserves window details while lifting shadows
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 40,
        stage: int = 1,
        num_blocks: List[int] = [1, 2, 2]
    ):
        super(RetinexFormer, self).__init__()
        self.stage = stage
        self.n_feat = n_feat
        self.num_blocks = num_blocks

        # Build stages
        modules_body = []
        for _ in range(stage):
            modules_body.append(
                RetinexFormer_Single_Stage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_feat=n_feat,
                    level=2,
                    num_blocks=num_blocks
                )
            )
        self.body = nn.Sequential(*modules_body)
        self.padder_size = 2 ** len(num_blocks)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.check_image_size(x)
        out = self.body(x)
        return out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)
        return x


class RetinexFormer_Single_Stage(nn.Module):
    """Single stage of Retinexformer."""
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
            in_dim=in_channels, out_dim=out_channels,
            dim=n_feat, level=level, num_blocks=num_blocks
        )

    def forward(self, img):
        # Estimate illumination
        illu_fea, illu_map = self.estimator(img)

        # Apply illumination adjustment
        input_img = img * illu_map + img

        # Denoise/enhance with illumination guidance
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


# Model configurations
def create_retinexformer(size: str = 'base') -> RetinexFormer:
    """
    Create Retinexformer model.

    Sizes:
    - tiny: ~1.6M params, n_feat=24, blocks=[1,1,1]
    - small: ~5.5M params, n_feat=32, blocks=[1,2,2]
    - base: ~15.5M params, n_feat=40, blocks=[1,2,2] (default)
    - large: ~25M params, n_feat=48, blocks=[2,4,4]
    """
    configs = {
        'tiny': {'n_feat': 24, 'num_blocks': [1, 1, 1], 'stage': 1},
        'small': {'n_feat': 32, 'num_blocks': [1, 2, 2], 'stage': 1},
        'base': {'n_feat': 40, 'num_blocks': [1, 2, 2], 'stage': 1},
        'large': {'n_feat': 48, 'num_blocks': [2, 4, 4], 'stage': 1},
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return RetinexFormer(**configs[size])


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    print("Testing Retinexformer configurations:")
    for size in ['tiny', 'small', 'base', 'large']:
        model = create_retinexformer(size)
        params = count_parameters(model)

        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y = model(x)

        print(f"  {size}: {params/1e6:.2f}M params, input {x.shape} -> output {y.shape}")

    print("\nAll tests passed!")
