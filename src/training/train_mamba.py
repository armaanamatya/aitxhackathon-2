"""
MambaDiffusion Training Script
==============================

Comprehensive training script for MambaDiffusion model.
Combines MambaIR (ECCV 2024 / CVPR 2025) with optional diffusion refinement.

Features:
- Multi-loss training: Charbonnier, SSIM, LPIPS, LAB, Histogram
- Mixed precision training (AMP)
- EMA for stable outputs
- Gradient accumulation for larger effective batch size
- Learning rate scheduling with warmup
- Comprehensive logging and checkpointing
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Import model
from mamba_diffusion import (
    MambaDiffusion, mamba_small, mamba_base, mamba_large,
    mamba_diffusion_base, mamba_diffusion_large, self_ensemble_inference
)


# =============================================================================
# Loss Functions
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant) - More robust than L1."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """Structural Similarity Loss."""

    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel

        # Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

        self.register_buffer('window', window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        window = self.window

        if channel != self.channel:
            window = self.window[0:1].expand(channel, 1, self.window_size, self.window_size)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""

    def __init__(self):
        super().__init__()

        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features

        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
            vgg[18:27], # relu4_4
        ])

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.weights = [1.0, 1.0, 1.0, 1.0]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet range
        pred = (pred * 0.5 + 0.5 - self.mean) / self.std
        target = (target * 0.5 + 0.5 - self.mean) / self.std

        loss = 0.0
        x, y = pred, target

        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            loss += weight * F.l1_loss(x, y)

        return loss


class LABColorLoss(nn.Module):
    """LAB color space loss for better color accuracy."""

    def __init__(self):
        super().__init__()

    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space."""
        # RGB to XYZ
        rgb = rgb * 0.5 + 0.5  # [-1, 1] -> [0, 1]

        mask = rgb > 0.04045
        rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # sRGB to XYZ matrix
        x = rgb[:, 0] * 0.4124564 + rgb[:, 1] * 0.3575761 + rgb[:, 2] * 0.1804375
        y = rgb[:, 0] * 0.2126729 + rgb[:, 1] * 0.7151522 + rgb[:, 2] * 0.0721750
        z = rgb[:, 0] * 0.0193339 + rgb[:, 1] * 0.1191920 + rgb[:, 2] * 0.9503041

        # Normalize to D65
        x = x / 0.95047
        z = z / 1.08883

        # XYZ to LAB
        def f(t):
            delta = 6.0 / 29.0
            return torch.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)

        fx, fy, fz = f(x), f(y), f(z)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return torch.stack([L, a, b], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        return F.l1_loss(pred_lab, target_lab)


class ColorHistogramLoss(nn.Module):
    """Color histogram matching loss."""

    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins

    def compute_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable histogram."""
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x = x.clamp(0, 1 - 1e-6)

        B, C, H, W = x.shape
        x = x.view(B, C, -1)

        bins = torch.linspace(0, 1, self.num_bins + 1, device=x.device)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        sigma = 1.0 / self.num_bins

        x = x.unsqueeze(-1)
        bin_centers = bin_centers.view(1, 1, 1, -1)

        weights = torch.exp(-0.5 * ((x - bin_centers) / sigma) ** 2)
        hist = weights.sum(dim=2)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

        return hist

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hist = self.compute_histogram(pred)
        target_hist = self.compute_histogram(target)
        return F.l1_loss(pred_hist, target_hist)


class GradientLoss(nn.Module):
    """Gradient-based edge preservation loss."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1, groups=3)

        target_gx = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_gy = F.conv2d(target, self.sobel_y, padding=1, groups=3)

        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


# =============================================================================
# Dataset
# =============================================================================

class RetouchDataset(Dataset):
    """Dataset for image retouching."""

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        image_size: int = 256,
        is_training: bool = True,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.is_training = is_training
        self.augment = augment and is_training

        # Load samples
        self.samples = []
        jsonl_file = self.data_root / jsonl_path
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

        # Split
        split_idx = int(len(self.samples) * 0.9)
        if is_training:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        print(f"Loaded {len(self.samples)} {'train' if is_training else 'val'} samples")

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image."""
        img_path = self.data_root / path
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img * 2 - 1  # [0, 1] -> [-1, 1]
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        # Support both 'src'/'tar' and 'source'/'target' keys
        src_key = 'src' if 'src' in item else 'source'
        tar_key = 'tar' if 'tar' in item else 'target'

        source = self._load_image(item[src_key])
        target = self._load_image(item[tar_key])

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                source = torch.flip(source, dims=[-1])
                target = torch.flip(target, dims=[-1])
            if random.random() > 0.5:
                source = torch.flip(source, dims=[-2])
                target = torch.flip(target, dims=[-2])
            if random.random() > 0.5:
                k = random.randint(1, 3)
                source = torch.rot90(source, k, dims=[-2, -1])
                target = torch.rot90(target, k, dims=[-2, -1])

        return {
            'source': source,
            'target': target,
        }


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
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
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Trainer
# =============================================================================

class MambaTrainer:
    """Trainer for MambaDiffusion."""

    def __init__(
        self,
        model: MambaDiffusion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 200,
        lambda_char: float = 10.0,
        lambda_ssim: float = 1.0,
        lambda_perceptual: float = 1.0,
        lambda_lpips: float = 1.0,
        lambda_lab: float = 1.0,
        lambda_hist: float = 0.5,
        lambda_gradient: float = 0.5,
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        grad_accum_steps: int = 1,
        save_interval: int = 10,
        sample_interval: int = 5,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.save_interval = save_interval
        self.sample_interval = sample_interval

        # Loss weights
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_lpips = lambda_lpips
        self.lambda_lab = lambda_lab
        self.lambda_hist = lambda_hist
        self.lambda_gradient = lambda_gradient

        # Losses
        self.char_loss = CharbonnierLoss().to(self.device)
        self.ssim_loss = SSIMLoss().to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.lab_loss = LABColorLoss().to(self.device)
        self.hist_loss = ColorHistogramLoss().to(self.device)
        self.gradient_loss = GradientLoss().to(self.device)

        # LPIPS
        try:
            import lpips
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_loss.eval()
            print("Using LPIPS loss")
        except:
            self.lpips_loss = None
            print("LPIPS not available")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=lr * 0.01,
        )

        # AMP
        self.scaler = GradScaler() if use_amp else None

        # EMA
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            print(f"Using EMA with decay {ema_decay}")

        # Best val loss
        self.best_val_loss = float('inf')
        self.start_epoch = 0

    def compute_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        # Charbonnier
        losses['char'] = self.char_loss(pred, target)

        # SSIM
        losses['ssim'] = self.ssim_loss(pred, target)

        # Perceptual
        losses['perceptual'] = self.perceptual_loss(pred, target)

        # LPIPS
        if self.lpips_loss is not None:
            losses['lpips'] = self.lpips_loss(pred, target).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=self.device)

        # LAB
        losses['lab'] = self.lab_loss(pred, target)

        # Histogram
        losses['hist'] = self.hist_loss(pred, target)

        # Gradient
        losses['gradient'] = self.gradient_loss(pred, target)

        # Total
        losses['total'] = (
            self.lambda_char * losses['char'] +
            self.lambda_ssim * losses['ssim'] +
            self.lambda_perceptual * losses['perceptual'] +
            self.lambda_lpips * losses['lpips'] +
            self.lambda_lab * losses['lab'] +
            self.lambda_hist * losses['hist'] +
            self.lambda_gradient * losses['gradient']
        )

        return losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
        running_losses = {
            'total': 0.0, 'char': 0.0, 'ssim': 0.0, 'lpips': 0.0,
            'lab': 0.0, 'hist': 0.0, 'gradient': 0.0,
        }
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            with autocast(enabled=self.use_amp):
                # Forward
                if self.model.use_diffusion:
                    outputs = self.model(source, target)
                    # Use predicted x0 for losses
                    pred = outputs.get('x0_pred', outputs.get('output'))
                    # Add noise prediction loss
                    noise_loss = F.mse_loss(outputs['noise_pred'], outputs['noise'])
                    losses = self.compute_losses(pred, target)
                    losses['total'] = losses['total'] + noise_loss
                else:
                    outputs = self.model(source)
                    pred = outputs['output']
                    losses = self.compute_losses(pred, target)

                loss = losses['total'] / self.grad_accum_steps

            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            # Update running losses
            for key in running_losses:
                if key in losses:
                    running_losses[key] += losses[key].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{running_losses['total'] / num_batches:.4f}",
                'Char': f"{running_losses['char'] / num_batches:.4f}",
            })

        # Average
        for key in running_losses:
            running_losses[key] /= num_batches

        return running_losses

    @torch.no_grad()
    def validate(self) -> float:
        """Validation pass."""
        self.model.eval()

        if self.use_ema:
            self.ema.apply_shadow()

        total_l1 = 0.0
        num_samples = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            if self.model.use_diffusion:
                pred = self.model.sample(source, num_steps=50)
            else:
                outputs = self.model(source)
                pred = outputs['output']

            # L1 in [0, 1] range
            pred_01 = pred * 0.5 + 0.5
            target_01 = target * 0.5 + 0.5

            l1 = F.l1_loss(pred_01, target_01).item()
            total_l1 += l1 * source.size(0)
            num_samples += source.size(0)

        if self.use_ema:
            self.ema.restore()

        return total_l1 / num_samples

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        if self.use_ema:
            state['ema_shadow'] = self.ema.shadow

        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest
        torch.save(state, self.output_dir / 'checkpoint_latest.pt')

        # Save periodic
        if (epoch + 1) % self.save_interval == 0:
            torch.save(state, self.output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

        # Save best
        if is_best:
            torch.save(state, self.output_dir / 'checkpoint_best.pt')

    def save_samples(self, epoch: int):
        """Save sample outputs."""
        self.model.eval()

        if self.use_ema:
            self.ema.apply_shadow()

        samples_dir = self.output_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            batch = next(iter(self.val_loader))
            source = batch['source'][:4].to(self.device)
            target = batch['target'][:4].to(self.device)

            if self.model.use_diffusion:
                pred = self.model.sample(source, num_steps=50)
            else:
                outputs = self.model(source)
                pred = outputs['output']

            # Convert to images
            for i in range(min(4, source.size(0))):
                src_img = (source[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
                tgt_img = (target[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
                prd_img = (pred[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255

                src_img = np.clip(src_img, 0, 255).astype(np.uint8)
                tgt_img = np.clip(tgt_img, 0, 255).astype(np.uint8)
                prd_img = np.clip(prd_img, 0, 255).astype(np.uint8)

                # Concatenate
                combined = np.concatenate([src_img, prd_img, tgt_img], axis=1)
                Image.fromarray(combined).save(
                    samples_dir / f'epoch_{epoch + 1}_sample_{i}.png'
                )

        if self.use_ema:
            self.ema.restore()

    def train(self):
        """Main training loop."""
        print(f"\nStarting MambaDiffusion training from epoch {self.start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Image size: {self.train_loader.dataset.image_size}")
        print(f"Diffusion mode: {self.model.use_diffusion}")
        print(f"Using AMP: {self.use_amp}")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_l1 = self.validate()

            # Check best
            is_best = val_l1 < self.best_val_loss
            if is_best:
                self.best_val_loss = val_l1

            # Update scheduler
            self.scheduler.step()

            # Print
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Char: {train_losses['char']:.4f}, SSIM: {train_losses['ssim']:.4f}, "
                  f"LPIPS: {train_losses['lpips']:.4f}, LAB: {train_losses['lab']:.4f}")
            print(f"  Val - L1: {val_l1:.4f} {'(best)' if is_best else ''}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # Save
            self.save_checkpoint(epoch, is_best)

            # Save samples
            if (epoch + 1) % self.sample_interval == 0:
                self.save_samples(epoch)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDiffusion Training')

    # Data
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_mamba')
    parser.add_argument('--image_size', type=int, default=256)

    # Model
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large', 'diff_base', 'diff_large'])

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Loss weights
    parser.add_argument('--lambda_char', type=float, default=10.0)
    parser.add_argument('--lambda_ssim', type=float, default=1.0)
    parser.add_argument('--lambda_perceptual', type=float, default=1.0)
    parser.add_argument('--lambda_lpips', type=float, default=1.0)
    parser.add_argument('--lambda_lab', type=float, default=1.0)
    parser.add_argument('--lambda_hist', type=float, default=0.5)
    parser.add_argument('--lambda_gradient', type=float, default=0.5)

    # Optimization
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    # Checkpointing
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Dataset
    print("Loading datasets...")
    train_dataset = RetouchDataset(
        args.data_root, args.jsonl_path, args.image_size, is_training=True
    )
    val_dataset = RetouchDataset(
        args.data_root, args.jsonl_path, args.image_size, is_training=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    print(f"Initializing MambaDiffusion-{args.model_size}...")
    if args.model_size == 'small':
        model = mamba_small()
    elif args.model_size == 'base':
        model = mamba_base()
    elif args.model_size == 'large':
        model = mamba_large()
    elif args.model_size == 'diff_base':
        model = mamba_diffusion_base()
    elif args.model_size == 'diff_large':
        model = mamba_diffusion_large()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Trainer
    trainer = MambaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        lambda_char=args.lambda_char,
        lambda_ssim=args.lambda_ssim,
        lambda_perceptual=args.lambda_perceptual,
        lambda_lpips=args.lambda_lpips,
        lambda_lab=args.lambda_lab,
        lambda_hist=args.lambda_hist,
        lambda_gradient=args.lambda_gradient,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        grad_accum_steps=args.grad_accum_steps,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
    )

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint['best_val_loss']
        if 'ema_shadow' in checkpoint and trainer.use_ema:
            trainer.ema.shadow = checkpoint['ema_shadow']
        print(f"Resumed from epoch {trainer.start_epoch}")

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
