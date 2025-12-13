#!/usr/bin/env python3
"""
Optimized Restormer Training Script
====================================
Enhanced training with:
- Charbonnier loss (more robust than L1)
- SSIM loss (structural similarity)
- Color Histogram loss
- EMA (Exponential Moving Average)
- Gradient loss (edge preservation)
- Strong data augmentation
- Gradient clipping
- Cosine annealing with warm restarts
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from restormer import create_restormer

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available")


# =============================================================================
# Loss Functions (Same as MambaDiffusion)
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss - More robust than L1 for image restoration."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        return loss.mean()


class SSIMLoss(nn.Module):
    """Structural Similarity Loss."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3
        self.window = self._create_window(window_size, self.channel, sigma)

    def _create_window(self, window_size: int, channel: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class ColorHistogramLoss(nn.Module):
    """Color histogram matching loss."""
    def __init__(self, bins: int = 64):
        super().__init__()
        self.bins = bins

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        loss = 0.0

        for c in range(C):
            pred_hist = torch.histc(pred[:, c], bins=self.bins, min=0, max=1)
            target_hist = torch.histc(target[:, c], bins=self.bins, min=0, max=1)
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            target_hist = target_hist / (target_hist.sum() + 1e-8)
            loss += F.l1_loss(pred_hist, target_hist)

        return loss / C


class LABColorLoss(nn.Module):
    """LAB color space loss for better color accuracy."""
    def __init__(self):
        super().__init__()

    def rgb_to_lab(self, img: torch.Tensor) -> torch.Tensor:
        # Simplified RGB to LAB conversion
        img = torch.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)

        M = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=img.device, dtype=img.dtype)

        B, C, H, W = img.shape
        img_flat = img.view(B, C, -1)
        xyz = torch.matmul(M.unsqueeze(0), img_flat)

        xyz[:, 0] /= 0.95047
        xyz[:, 1] /= 1.0
        xyz[:, 2] /= 1.08883

        epsilon = 0.008856
        kappa = 903.3
        f = torch.where(xyz > epsilon, xyz ** (1/3), (kappa * xyz + 16) / 116)

        L = 116 * f[:, 1:2] - 16
        a = 500 * (f[:, 0:1] - f[:, 1:2])
        b = 200 * (f[:, 1:2] - f[:, 2:3])

        return torch.cat([L, a, b], dim=1).view(B, 3, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self.rgb_to_lab(pred.clamp(0, 1))
        target_lab = self.rgb_to_lab(target.clamp(0, 1))
        return F.l1_loss(pred_lab, target_lab)


class GradientLoss(nn.Module):
    """Gradient-based edge preservation loss."""
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sobel_x.device != pred.device:
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)

        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        pred_gx = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_gx = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_gy = F.conv2d(target_gray, self.sobel_y, padding=1)

        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

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

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# =============================================================================
# Dataset with Strong Augmentation
# =============================================================================

class HDRDataset(Dataset):
    """Dataset with strong augmentation for better generalization."""

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        image_size: int = 128,
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
                self.samples.append(json.loads(line.strip()))

        # Split
        split_idx = int(len(self.samples) * 0.9)
        if is_training:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        print(f"Loaded {len(self.samples)} {'train' if is_training else 'val'} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        source_path = self.data_root / item['src']
        target_path = self.data_root / item['tar']

        source = Image.open(source_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        # Resize
        source = source.resize((self.image_size, self.image_size), Image.LANCZOS)
        target = target.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensor
        source = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 255.0
        target = torch.from_numpy(np.array(target)).permute(2, 0, 1).float() / 255.0

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                source = torch.flip(source, [2])
                target = torch.flip(target, [2])

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                source = torch.flip(source, [1])
                target = torch.flip(target, [1])

            # Random 90-degree rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                source = torch.rot90(source, k, [1, 2])
                target = torch.rot90(target, k, [1, 2])

            # Color jitter (only on source - simulates different camera settings)
            if torch.rand(1).item() > 0.5:
                # Brightness
                brightness = 0.9 + torch.rand(1).item() * 0.2  # 0.9 to 1.1
                source = (source * brightness).clamp(0, 1)

            if torch.rand(1).item() > 0.5:
                # Contrast
                contrast = 0.9 + torch.rand(1).item() * 0.2
                mean = source.mean()
                source = ((source - mean) * contrast + mean).clamp(0, 1)

        return {'source': source, 'target': target}


# =============================================================================
# Trainer
# =============================================================================

class OptimizedTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        lr: float = 2e-4,
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        grad_clip: float = 1.0,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip

        # Optimizer with cosine annealing
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scaler = GradScaler() if use_amp else None

        # EMA
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None

        # Losses (same weights as MambaDiffusion)
        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.lab_loss = LABColorLoss()
        self.histogram_loss = ColorHistogramLoss()
        self.gradient_loss = GradientLoss()

        if LPIPS_AVAILABLE:
            self.lpips_loss = lpips.LPIPS(net='alex').to(device)
            self.lpips_loss.eval()
        else:
            self.lpips_loss = None

        # Loss weights (tuned for balance)
        self.w_char = 1.0
        self.w_ssim = 0.5
        self.w_lpips = 0.3
        self.w_lab = 0.1
        self.w_hist = 0.05
        self.w_grad = 0.1

        # Tracking
        self.best_val_loss = float('inf')
        self.epoch = 0

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        # Charbonnier (main reconstruction loss)
        losses['char'] = self.charbonnier(pred, target)

        # SSIM
        losses['ssim'] = self.ssim_loss(pred, target)

        # LPIPS
        if self.lpips_loss is not None:
            with torch.no_grad():
                losses['lpips'] = self.lpips_loss(pred * 2 - 1, target * 2 - 1).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=pred.device)

        # LAB color
        losses['lab'] = self.lab_loss(pred, target)

        # Histogram
        losses['hist'] = self.histogram_loss(pred, target)

        # Gradient
        losses['grad'] = self.gradient_loss(pred, target)

        # Total
        total = (self.w_char * losses['char'] +
                 self.w_ssim * losses['ssim'] +
                 self.w_lpips * losses['lpips'] +
                 self.w_lab * losses['lab'] +
                 self.w_hist * losses['hist'] +
                 self.w_grad * losses['grad'])

        return total, {k: v.item() for k, v in losses.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_losses = {k: 0.0 for k in ['total', 'char', 'ssim', 'lpips', 'lab', 'hist', 'grad']}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(source)
                    loss, loss_dict = self.compute_loss(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(source)
                loss, loss_dict = self.compute_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update()

            # Track losses
            total_losses['total'] += loss.item()
            for k, v in loss_dict.items():
                total_losses[k] += v

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average
        n = len(self.train_loader)
        return {k: v / n for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> float:
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        total_l1 = 0.0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(source)
            total_l1 += F.l1_loss(output, target).item()

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        return total_l1 / len(self.val_loader)

    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        torch.save(checkpoint, self.output_dir / filename)

    def train(self, num_epochs: int, patience: int = 30):
        print(f"\nStarting optimized training for {num_epochs} epochs")
        print(f"Loss weights: Char={self.w_char}, SSIM={self.w_ssim}, LPIPS={self.w_lpips}, "
              f"LAB={self.w_lab}, Hist={self.w_hist}, Grad={self.w_grad}")
        print(f"Using EMA: {self.use_ema}")
        print(f"Gradient clipping: {self.grad_clip}")
        print("-" * 50)

        no_improve = 0

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch

            # Cosine annealing LR
            lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Check improvement
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.save_checkpoint('checkpoint_best.pt')
                no_improve = 0
            else:
                no_improve += 1

            # Print
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}, Char: {train_losses['char']:.4f}, "
                  f"SSIM: {train_losses['ssim']:.4f}, LPIPS: {train_losses['lpips']:.4f}")
            print(f"  Val - L1: {val_loss:.4f} {'(best)' if improved else ''}")
            print(f"  LR: {lr:.2e}")

            # Early stopping
            if no_improve >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break

            # Periodic save
            if epoch % 20 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Final save
        self.save_checkpoint('checkpoint_final.pt')
        print(f"\nTraining complete. Best Val L1: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Optimized Restormer Training')
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_optimized')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_ema', action='store_true')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Data
    train_dataset = HDRDataset(args.data_root, args.jsonl_path, args.image_size, is_training=True)
    val_dataset = HDRDataset(args.data_root, args.jsonl_path, args.image_size, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    print(f"\nInitializing Restormer-{args.model_size}...")
    model = create_restormer(args.model_size)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        lr=args.lr,
        use_amp=not args.no_amp,
        use_ema=not args.no_ema,
        device=device,
    )

    # Train
    trainer.train(args.num_epochs)


if __name__ == '__main__':
    main()
