"""
Training script for Context-Aware Retouching Network (INRetouch-style).

Comprehensive training pipeline with:
- Multiple quality-focused loss functions
- EMA for stable outputs
- Multi-scale training
- Extensive augmentation
- Gradient accumulation for effective larger batch sizes
- Mixed precision training
- Comprehensive logging

Optimized for maximum image quality on paired datasets.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.inretouch import (
    ContextAwareRetouchNet,
    inretouch_small, inretouch_base, inretouch_large,
    count_parameters
)

# Import loss functions from models.py
from src.training.models import (
    CharbonnierLoss,
    SSIMLoss,
    FFTLoss,
    VGGPerceptualLoss,
    LABColorLoss,
    ColorHistogramLoss,
)

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")


# =============================================================================
# Additional Loss Functions for Maximum Quality
# =============================================================================

class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss."""

    def __init__(self, net: str = 'alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net, verbose=False)
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips(pred, target).mean()


class GradientLoss(nn.Module):
    """
    Gradient/Edge loss for preserving edge sharpness.
    Computes L1 loss on Sobel gradients.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute gradients
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_gx = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_gy = F.conv2d(target, self.sobel_y, padding=1, groups=3)

        # L1 loss on gradients
        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)

        return loss_x + loss_y


class ColorConsistencyLoss(nn.Module):
    """
    Color consistency loss ensuring global color statistics match.
    Computes loss on mean and std of each channel.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        pred_mean = pred.mean(dim=[2, 3])
        pred_std = pred.std(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        target_std = target.std(dim=[2, 3])

        loss_mean = F.l1_loss(pred_mean, target_mean)
        loss_std = F.l1_loss(pred_std, target_std)

        return loss_mean + loss_std


# =============================================================================
# Discriminator for Adversarial Training (Maximum Quality)
# =============================================================================

class SpectralNorm(nn.Module):
    """Spectral normalization wrapper."""
    def __init__(self, module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)

    def forward(self, x):
        return self.module(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.
    Uses spectral normalization for stable training.
    """

    def __init__(self, in_channels: int = 6, base_channels: int = 64, num_layers: int = 4):
        super().__init__()

        layers = []

        # First layer (no norm)
        layers.append(SpectralNorm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers
        ch = base_channels
        for i in range(1, num_layers):
            out_ch = min(ch * 2, 512)
            layers.append(SpectralNorm(nn.Conv2d(ch, out_ch, 4, 2, 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = out_ch

        # Output layer
        layers.append(SpectralNorm(nn.Conv2d(ch, 1, 4, 1, 1)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Generated or real image
            condition: Source image (condition)
        """
        return self.model(torch.cat([x, condition], dim=1))


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better quality at different frequencies.
    """

    def __init__(self, in_channels: int = 6, num_discriminators: int = 3):
        super().__init__()

        self.discriminators = nn.ModuleList()
        for _ in range(num_discriminators):
            self.discriminators.append(PatchDiscriminator(in_channels))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x, condition))
            x = self.downsample(x)
            condition = self.downsample(condition)
        return outputs


class GANLoss(nn.Module):
    """
    GAN loss with support for different GAN types.
    """

    def __init__(self, gan_type: str = 'lsgan'):
        super().__init__()
        self.gan_type = gan_type

        if gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'hinge':
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_type == 'hinge':
            if target_is_real:
                return torch.mean(F.relu(1 - pred))
            else:
                return torch.mean(F.relu(1 + pred))
        else:
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.loss(pred, target)


# =============================================================================
# Self-Ensemble for Maximum Quality at Inference
# =============================================================================

def self_ensemble_inference(model: nn.Module, x: torch.Tensor, num_augments: int = 8) -> torch.Tensor:
    """
    Self-ensemble inference with geometric augmentations.
    Averages predictions from multiple augmented versions.

    Args:
        model: The trained model
        x: Input image (B, C, H, W)
        num_augments: Number of augmentations (1, 2, 4, or 8)

    Returns:
        Averaged prediction
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original
        predictions.append(model(x))

        if num_augments >= 2:
            # Horizontal flip
            x_flip = torch.flip(x, dims=[3])
            pred_flip = model(x_flip)
            predictions.append(torch.flip(pred_flip, dims=[3]))

        if num_augments >= 4:
            # Vertical flip
            x_vflip = torch.flip(x, dims=[2])
            pred_vflip = model(x_vflip)
            predictions.append(torch.flip(pred_vflip, dims=[2]))

            # Both flips
            x_both = torch.flip(x, dims=[2, 3])
            pred_both = model(x_both)
            predictions.append(torch.flip(pred_both, dims=[2, 3]))

        if num_augments >= 8:
            # 90 degree rotations
            for k in [1, 2, 3]:
                x_rot = torch.rot90(x, k, dims=[2, 3])
                pred_rot = model(x_rot)
                predictions.append(torch.rot90(pred_rot, -k, dims=[2, 3]))

    # Average predictions
    return torch.stack(predictions[:num_augments]).mean(dim=0)


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for capturing both fine details and global structure.
    """

    def __init__(self, base_loss: nn.Module, scales: List[float] = [1.0, 0.5, 0.25]):
        super().__init__()
        self.base_loss = base_loss
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for scale in self.scales:
            if scale < 1.0:
                size = (int(pred.shape[2] * scale), int(pred.shape[3] * scale))
                pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target

            total_loss += self.base_loss(pred_scaled, target_scaled)

        return total_loss / len(self.scales)


# =============================================================================
# Dataset with Advanced Augmentation
# =============================================================================

class RetouchingDataset(Dataset):
    """
    Dataset for paired before/after retouching images.
    Includes comprehensive augmentation for robust training.
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        image_size: int = 512,
        is_train: bool = True,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.is_train = is_train
        self.augment = augment and is_train

        # Load samples
        self.samples = []
        jsonl_file = self.data_root / jsonl_path
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

        # Base transforms
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        """Load image and convert to RGB."""
        full_path = self.data_root / path
        return Image.open(full_path).convert('RGB')

    def _apply_augmentation(
        self, source: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply synchronized augmentation to source and target."""

        # Random crop (larger than final size for more variation)
        crop_size = int(self.image_size * random.uniform(1.0, 1.5))
        i, j, h, w = transforms.RandomCrop.get_params(source, (crop_size, crop_size))
        source = TF.crop(source, i, j, h, w)
        target = TF.crop(target, i, j, h, w)

        # Resize to target size
        source = TF.resize(source, (self.image_size, self.image_size), interpolation=Image.LANCZOS)
        target = TF.resize(target, (self.image_size, self.image_size), interpolation=Image.LANCZOS)

        # Random horizontal flip
        if random.random() > 0.5:
            source = TF.hflip(source)
            target = TF.hflip(target)

        # Random vertical flip (for real estate, less common but still useful)
        if random.random() > 0.8:
            source = TF.vflip(source)
            target = TF.vflip(target)

        # Random rotation (small angles)
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            source = TF.rotate(source, angle, interpolation=Image.BILINEAR)
            target = TF.rotate(target, angle, interpolation=Image.BILINEAR)

        return source, target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        # Load images (support both 'src'/'tar' and 'source'/'target' keys)
        src_key = 'src' if 'src' in item else 'source'
        tar_key = 'tar' if 'tar' in item else 'target'
        source = self._load_image(item[src_key])
        target = self._load_image(item[tar_key])

        if self.augment:
            source, target = self._apply_augmentation(source, target)
        else:
            # Center crop and resize for validation
            min_size = min(source.size)
            source = TF.center_crop(source, min_size)
            target = TF.center_crop(target, min_size)
            source = TF.resize(source, (self.image_size, self.image_size), interpolation=Image.LANCZOS)
            target = TF.resize(target, (self.image_size, self.image_size), interpolation=Image.LANCZOS)

        # To tensor and normalize to [-1, 1]
        source = self.to_tensor(source) * 2 - 1
        target = self.to_tensor(target) * 2 - 1

        return {
            'source': source,
            'target': target,
            'name': item.get(src_key, str(idx)),
        }


def get_dataloaders(
    data_root: str,
    jsonl_path: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 8,
    train_ratio: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    # Load all samples
    samples = []
    with open(Path(data_root) / jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Split
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Write temporary JSONL files
    train_jsonl = Path(data_root) / 'train_split.jsonl'
    val_jsonl = Path(data_root) / 'val_split.jsonl'

    with open(train_jsonl, 'w') as f:
        for s in train_samples:
            f.write(json.dumps(s) + '\n')

    with open(val_jsonl, 'w') as f:
        for s in val_samples:
            f.write(json.dumps(s) + '\n')

    # Create datasets
    train_dataset = RetouchingDataset(
        data_root=data_root,
        jsonl_path='train_split.jsonl',
        image_size=image_size,
        is_train=True,
        augment=True,
    )

    val_dataset = RetouchingDataset(
        data_root=data_root,
        jsonl_path='val_split.jsonl',
        image_size=image_size,
        is_train=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Trainer
# =============================================================================

class INRetouchTrainer:
    """
    Comprehensive trainer for Context-Aware Retouching Network.

    Optimized for MAXIMUM QUALITY matching of _src.jpg -> _tar.jpg.
    Features:
    - Multi-scale discriminator for GAN training (sharper outputs)
    - Heavy perceptual losses (LPIPS, VGG, LAB)
    - Color matching losses (histogram, consistency)
    - EMA for stable outputs
    - Self-ensemble inference for best quality
    - Progressive training support
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        model_size: str = 'large',  # Use large for max quality
        batch_size: int = 4,
        num_epochs: int = 300,  # More epochs for convergence
        image_size: int = 512,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        # Loss weights OPTIMIZED for exact _tar.jpg matching
        lambda_l1: float = 10.0,  # Strong pixel loss
        lambda_ssim: float = 1.0,  # Structural similarity
        lambda_perceptual: float = 1.0,  # VGG perceptual
        lambda_lpips: float = 1.0,  # Perceptual quality (CRITICAL)
        lambda_lab: float = 1.0,  # Color accuracy (CRITICAL)
        lambda_hist: float = 0.5,  # Color distribution
        lambda_gradient: float = 0.5,  # Edge preservation
        lambda_color_consistency: float = 0.5,  # Global color stats
        # GAN training for sharpness
        use_gan: bool = True,
        lambda_gan: float = 0.1,  # Adversarial loss weight
        gan_type: str = 'lsgan',  # lsgan, vanilla, hinge
        # Training options
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.9999,  # Higher decay for stability
        grad_accum_steps: int = 2,  # Effective batch 8
        num_workers: int = 8,
        save_interval: int = 10,
        sample_interval: int = 5,
        resume_from: Optional[str] = None,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.save_interval = save_interval
        self.sample_interval = sample_interval

        # Loss weights
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_lpips = lambda_lpips
        self.lambda_lab = lambda_lab
        self.lambda_hist = lambda_hist
        self.lambda_gradient = lambda_gradient
        self.lambda_color_consistency = lambda_color_consistency
        self.lambda_gan = lambda_gan
        self.use_gan = use_gan
        self.gan_type = gan_type

        # Output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.sample_dir = self.output_dir / 'samples'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Data
        print("Loading datasets...")
        self.train_loader, self.val_loader = get_dataloaders(
            data_root=data_root,
            jsonl_path=jsonl_path,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
        print(f"Loaded {len(self.train_loader.dataset)} train samples")
        print(f"Loaded {len(self.val_loader.dataset)} val samples")

        # Model
        print(f"Initializing INRetouch-{model_size}...")
        if model_size == 'small':
            self.model = inretouch_small().to(self.device)
        elif model_size == 'base':
            self.model = inretouch_base().to(self.device)
        elif model_size == 'large':
            self.model = inretouch_large().to(self.device)
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        print(f"Model parameters: {count_parameters(self.model):,}")

        # EMA
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            print(f"Using EMA with decay {ema_decay}")
        else:
            self.ema = None

        # Loss functions
        self.criterion_l1 = CharbonnierLoss().to(self.device)
        self.criterion_ssim = SSIMLoss().to(self.device)
        self.criterion_perceptual = VGGPerceptualLoss().to(self.device)
        self.criterion_gradient = GradientLoss().to(self.device)
        self.criterion_color = ColorConsistencyLoss().to(self.device)

        if LPIPS_AVAILABLE and lambda_lpips > 0:
            self.criterion_lpips = LPIPSLoss(net='alex').to(self.device)
            print("Using LPIPS loss")
        else:
            self.criterion_lpips = None

        if lambda_lab > 0:
            self.criterion_lab = LABColorLoss().to(self.device)
            print("Using LAB color loss")
        else:
            self.criterion_lab = None

        if lambda_hist > 0:
            self.criterion_hist = ColorHistogramLoss().to(self.device)
            print("Using Color Histogram loss")
        else:
            self.criterion_hist = None

        # Optimizer for generator
        self.optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # GAN components
        if use_gan:
            print("Using GAN training with multi-scale discriminator")
            self.discriminator = MultiScaleDiscriminator(in_channels=6).to(self.device)
            self.criterion_gan = GANLoss(gan_type=gan_type).to(self.device)

            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=lr * 0.5,  # Slightly lower LR for discriminator
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
            print(f"Discriminator parameters: {count_parameters(self.discriminator):,}")
        else:
            self.discriminator = None
            self.criterion_gan = None
            self.optimizer_d = None

        # Scheduler (cosine annealing)
        self.scheduler_g = CosineAnnealingLR(
            self.optimizer_g,
            T_max=num_epochs,
            eta_min=lr * 0.01,
        )

        if use_gan:
            self.scheduler_d = CosineAnnealingLR(
                self.optimizer_d,
                T_max=num_epochs,
                eta_min=lr * 0.005,
            )
        else:
            self.scheduler_d = None

        # AMP scaler
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)

    def compute_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with all components."""

        losses = {}

        # L1 / Charbonnier
        loss_l1 = self.criterion_l1(output, target)
        losses['l1'] = loss_l1.item()

        # SSIM
        loss_ssim = self.criterion_ssim(output, target)
        losses['ssim'] = loss_ssim.item()

        # Perceptual
        loss_perceptual = self.criterion_perceptual(output, target)
        losses['perceptual'] = loss_perceptual.item()

        # Gradient
        loss_gradient = self.criterion_gradient(output, target)
        losses['gradient'] = loss_gradient.item()

        # Color consistency
        loss_color = self.criterion_color(output, target)
        losses['color'] = loss_color.item()

        # Total (base)
        total_loss = (
            self.lambda_l1 * loss_l1 +
            self.lambda_ssim * loss_ssim +
            self.lambda_perceptual * loss_perceptual +
            self.lambda_gradient * loss_gradient +
            self.lambda_color_consistency * loss_color
        )

        # Optional losses
        if self.criterion_lpips is not None:
            loss_lpips = self.criterion_lpips(output, target)
            losses['lpips'] = loss_lpips.item()
            total_loss += self.lambda_lpips * loss_lpips

        if self.criterion_lab is not None:
            loss_lab = self.criterion_lab(output, target)
            losses['lab'] = loss_lab.item()
            total_loss += self.lambda_lab * loss_lab

        if self.criterion_hist is not None:
            loss_hist = self.criterion_hist(output, target)
            losses['hist'] = loss_hist.item()
            total_loss += self.lambda_hist * loss_hist

        losses['total'] = total_loss.item()

        return total_loss, losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with optional GAN training."""

        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        epoch_losses = {k: 0.0 for k in ['total', 'l1', 'ssim', 'perceptual',
                                          'lpips', 'lab', 'hist', 'gradient', 'color',
                                          'g_gan', 'd_real', 'd_fake']}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        self.optimizer_g.zero_grad()
        if self.optimizer_d is not None:
            self.optimizer_d.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            # =====================
            # Train Discriminator
            # =====================
            if self.discriminator is not None and (batch_idx + 1) % self.grad_accum_steps == 0:
                with autocast(enabled=self.use_amp):
                    with torch.no_grad():
                        fake = self.model(source)

                    # Real
                    d_real_outputs = self.discriminator(target, source)
                    d_loss_real = sum(self.criterion_gan(out, True) for out in d_real_outputs)
                    d_loss_real = d_loss_real / len(d_real_outputs)

                    # Fake
                    d_fake_outputs = self.discriminator(fake.detach(), source)
                    d_loss_fake = sum(self.criterion_gan(out, False) for out in d_fake_outputs)
                    d_loss_fake = d_loss_fake / len(d_fake_outputs)

                    d_loss = (d_loss_real + d_loss_fake) * 0.5

                if self.use_amp:
                    self.scaler.scale(d_loss).backward()
                    self.scaler.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.scaler.step(self.optimizer_d)
                else:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.optimizer_d.step()

                self.optimizer_d.zero_grad()
                epoch_losses['d_real'] += d_loss_real.item()
                epoch_losses['d_fake'] += d_loss_fake.item()

            # =====================
            # Train Generator
            # =====================
            with autocast(enabled=self.use_amp):
                output = self.model(source)
                loss_g, losses = self.compute_loss(output, target)

                # GAN loss for generator
                if self.discriminator is not None:
                    g_fake_outputs = self.discriminator(output, source)
                    g_loss_gan = sum(self.criterion_gan(out, True) for out in g_fake_outputs)
                    g_loss_gan = g_loss_gan / len(g_fake_outputs)
                    loss_g = loss_g + self.lambda_gan * g_loss_gan
                    losses['g_gan'] = g_loss_gan.item()
                    epoch_losses['g_gan'] += g_loss_gan.item()

                loss_g = loss_g / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss_g).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer_g)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer_g)
                    self.scaler.update()
                    self.optimizer_g.zero_grad()

                    if self.ema is not None:
                        self.ema.update()
            else:
                loss_g.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer_g.step()
                    self.optimizer_g.zero_grad()

                    if self.ema is not None:
                        self.ema.update()

            # Accumulate losses
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v

            self.global_step += 1

            pbar.set_postfix({
                'Loss': f"{losses['total']:.4f}",
                'LPIPS': f"{losses.get('lpips', 0):.4f}",
                'GAN': f"{losses.get('g_gan', 0):.4f}",
            })

        # Average losses
        num_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""

        # Use EMA weights if available
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        val_losses = {k: 0.0 for k in ['total', 'l1', 'ssim', 'perceptual',
                                        'lpips', 'lab', 'hist', 'gradient', 'color']}
        val_l1 = 0.0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(source)
            _, losses = self.compute_loss(output, target)

            for k, v in losses.items():
                if k in val_losses:
                    val_losses[k] += v

            # Raw L1 for comparison
            val_l1 += F.l1_loss(output, target).item()

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        # Average
        num_batches = len(self.val_loader)
        for k in val_losses:
            val_losses[k] /= num_batches
        val_l1 /= num_batches

        val_losses['raw_l1'] = val_l1

        return val_losses

    @torch.no_grad()
    def save_samples(self, epoch: int):
        """Save sample outputs."""

        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()

        # Get a batch from validation
        batch = next(iter(self.val_loader))
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)

        output = self.model(source)

        # Convert to images
        source_np = ((source[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        target_np = ((target[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        output_np = ((output[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        # Create comparison image
        comparison = np.concatenate([source_np, output_np, target_np], axis=1)
        Image.fromarray(comparison).save(self.sample_dir / f'epoch_{epoch:04d}.jpg', quality=95)

        if self.ema is not None:
            self.ema.restore()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""

        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        if self.ema is not None:
            state['ema_shadow'] = self.ema.shadow

        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest
        torch.save(state, self.checkpoint_dir / 'latest.pt')

        # Save periodic
        if (epoch + 1) % self.save_interval == 0:
            torch.save(state, self.checkpoint_dir / f'epoch_{epoch+1:04d}.pt')

        # Save best
        if is_best:
            torch.save(state, self.checkpoint_dir / 'best.pt')

            # Also save just the model weights for easy loading
            if self.ema is not None:
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model_ema.pt')
                self.ema.restore()
            else:
                torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model.pt')

    def load_checkpoint(self, path: str):
        """Load checkpoint."""

        print(f"Loading checkpoint from {path}")
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer_g.load_state_dict(state['optimizer_g_state_dict'])
        self.scheduler_g.load_state_dict(state['scheduler_g_state_dict'])
        self.start_epoch = state['epoch'] + 1
        self.global_step = state['global_step']
        self.best_val_loss = state['best_val_loss']

        if self.ema is not None and 'ema_shadow' in state:
            self.ema.shadow = state['ema_shadow']

        if self.scaler is not None and 'scaler_state_dict' in state:
            self.scaler.load_state_dict(state['scaler_state_dict'])

        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop."""

        print(f"\nStarting INRetouch training from epoch {self.start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.batch_size * self.grad_accum_steps}")
        print(f"Using AMP: {self.use_amp}")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate()

            # Update scheduler
            self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

            # Check if best
            is_best = val_losses['raw_l1'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['raw_l1']

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Save samples
            if (epoch + 1) % self.sample_interval == 0:
                self.save_samples(epoch + 1)

            # Logging
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"L1: {train_losses['l1']:.4f}, "
                  f"SSIM: {train_losses['ssim']:.4f}, "
                  f"LPIPS: {train_losses.get('lpips', 0):.4f}")
            print(f"  Val - L1: {val_losses['raw_l1']:.4f} {'(best)' if is_best else ''}")
            print(f"  LR: {self.scheduler_g.get_last_lr()[0]:.2e}")

        print("\nTraining complete!")
        print(f"Best validation L1: {self.best_val_loss:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train INRetouch model')

    # Data
    parser.add_argument('--data_root', type=str, default='.',
                        help='Root directory for data')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl',
                        help='Path to JSONL file with image pairs')
    parser.add_argument('--output_dir', type=str, default='outputs_inretouch',
                        help='Output directory')

    # Model
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='Model size variant')

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Loss weights (optimized for max quality)
    parser.add_argument('--lambda_l1', type=float, default=10.0)
    parser.add_argument('--lambda_ssim', type=float, default=1.0)
    parser.add_argument('--lambda_perceptual', type=float, default=1.0)
    parser.add_argument('--lambda_lpips', type=float, default=1.0)
    parser.add_argument('--lambda_lab', type=float, default=1.0)
    parser.add_argument('--lambda_hist', type=float, default=0.5)
    parser.add_argument('--lambda_gradient', type=float, default=0.5)
    parser.add_argument('--lambda_color', type=float, default=0.5)

    # GAN training
    parser.add_argument('--use_gan', action='store_true', default=True)
    parser.add_argument('--no_gan', action='store_false', dest='use_gan')
    parser.add_argument('--lambda_gan', type=float, default=0.1)
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        choices=['lsgan', 'vanilla', 'hinge'])

    # Training options
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()

    trainer = INRetouchTrainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_perceptual=args.lambda_perceptual,
        lambda_lpips=args.lambda_lpips,
        lambda_lab=args.lambda_lab,
        lambda_hist=args.lambda_hist,
        lambda_gradient=args.lambda_gradient,
        lambda_color_consistency=args.lambda_color,
        use_gan=args.use_gan,
        lambda_gan=args.lambda_gan,
        gan_type=args.gan_type,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        resume_from=args.resume_from,
    )

    trainer.train()


if __name__ == '__main__':
    main()
