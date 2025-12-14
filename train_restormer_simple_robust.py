#!/usr/bin/env python3
"""
SIMPLIFIED ROBUST RESTORMER TRAINING
=====================================

Fixes for stuck validation loss:
1. Simplified loss (3 components instead of 8)
2. Proper loss normalization
3. More aggressive learning rate schedule
4. Added gradient monitoring
5. Removed numerical instability sources

Core losses only:
- L1 Loss (base reconstruction)
- Window-Aware L1 (3x weight on bright regions)
- Perceptual VGG (optional, can disable)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer


# =============================================================================
# SIMPLIFIED ROBUST LOSS
# =============================================================================

class WindowDetector(nn.Module):
    """Detect bright/window regions without overcomplication"""
    def __init__(self, threshold: float = 0.75):
        super().__init__()
        self.threshold = threshold

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Returns binary-ish mask [0,1] for bright regions"""
        # Luminance
        lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        # Smooth sigmoid
        mask = torch.sigmoid((lum - self.threshold) * 20)
        return mask


class SimplifiedWindowLoss(nn.Module):
    """Simple weighted L1 with higher weight on windows"""
    def __init__(self, window_weight: float = 3.0):
        super().__init__()
        self.window_weight = window_weight
        self.detector = WindowDetector(threshold=0.75)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Detect windows in target
        window_mask = self.detector(target)

        # Per-pixel L1
        pixel_loss = torch.abs(pred - target)

        # Weighted: 1.0 for non-windows, window_weight for windows
        weight = 1.0 + (self.window_weight - 1.0) * window_mask

        return (pixel_loss * weight).mean()


class SimplifiedLoss(nn.Module):
    """
    Simplified loss with only essential components.
    NO histogram, NO FFT, NO LAB - these caused numerical issues.
    """
    def __init__(self,
                 use_perceptual: bool = False,
                 window_weight: float = 3.0):
        super().__init__()

        self.use_perceptual = use_perceptual

        # Core losses
        self.l1_loss = nn.L1Loss()
        self.window_loss = SimplifiedWindowLoss(window_weight=window_weight)

        # Optional perceptual (VGG) - can disable for faster training
        if use_perceptual:
            from torchvision import models
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:17]
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.vgg = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        losses = {}

        # 1. Base L1 loss
        losses['l1'] = self.l1_loss(pred, target)

        # 2. Window-aware L1
        losses['window'] = self.window_loss(pred, target)

        # 3. Optional perceptual
        if self.vgg is not None:
            pred_norm = (pred - self.vgg_mean) / self.vgg_std
            target_norm = (target - self.vgg_mean) / self.vgg_std
            pred_feat = self.vgg(pred_norm)
            target_feat = self.vgg(target_norm)
            losses['perceptual'] = F.l1_loss(pred_feat, target_feat) * 0.1

        # Total: balanced weights
        losses['total'] = losses['l1'] + losses['window']
        if 'perceptual' in losses:
            losses['total'] += losses['perceptual']

        return losses


# =============================================================================
# DATASET
# =============================================================================

class HDRDataset(Dataset):
    def __init__(self, jsonl_path: str, resolution: int = 512, augment: bool = True):
        self.resolution = resolution
        self.augment = augment
        self.pairs = []

        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load
        src = cv2.imread(pair['src'])
        tar = cv2.imread(pair['tar'])

        if src is None or tar is None:
            raise RuntimeError(f"Failed to load: {pair}")

        # BGR to RGB
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        # Resize
        src = cv2.resize(src, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        tar = cv2.resize(tar, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # Augment
        if self.augment and np.random.random() > 0.5:
            src = np.fliplr(src).copy()
            tar = np.fliplr(tar).copy()

        # To tensor
        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar


# =============================================================================
# TRAINER
# =============================================================================

class RobustTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Output dir
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / 'training.log'

        self._log("=" * 70)
        self._log("SIMPLIFIED ROBUST RESTORMER TRAINING")
        self._log("=" * 70)
        self._log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        self._log(f"Resolution: {args.resolution}")
        self._log(f"Batch size: {args.batch_size}")
        self._log(f"Learning rate: {args.lr}")
        self._log(f"Use perceptual: {args.use_perceptual}")

    def _log(self, msg: str):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

    def setup_data(self):
        self._log("\n📂 Loading data...")

        train_dataset = HDRDataset(self.args.train_jsonl, self.args.resolution, augment=True)
        val_dataset = HDRDataset(self.args.val_jsonl, self.args.resolution, augment=False)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size,
            shuffle=True, num_workers=self.args.num_workers,
            pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=self.args.num_workers,
            pin_memory=True
        )

        self._log(f"   Train: {len(train_dataset)} samples, {len(self.train_loader)} batches")
        self._log(f"   Val: {len(val_dataset)} samples, {len(self.val_loader)} batches")

    def setup_model(self):
        self._log("\n🏗️  Creating Restormer...")

        # Clear GPU cache from previous runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log("   Cleared CUDA cache")

        self.model = Restormer(
            in_channels=3,
            out_channels=3,
            dim=self.args.dim,
            num_blocks=self.args.num_blocks,
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            use_checkpointing=True  # Enable gradient checkpointing to save memory
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        self._log(f"   Parameters: {num_params:,}")

        # Loss
        self.criterion = SimplifiedLoss(
            use_perceptual=self.args.use_perceptual,
            window_weight=3.0
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        # Cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs - self.args.warmup_epochs,
            eta_min=1e-6
        )

        # Mixed precision
        self.scaler = GradScaler() if self.args.mixed_precision else None

        self._log(f"   Loss: Simplified (L1 + Window)")
        if self.args.use_perceptual:
            self._log(f"   + Perceptual (VGG)")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_losses = {'total': 0.0, 'l1': 0.0, 'window': 0.0}
        if self.args.use_perceptual:
            epoch_losses['perceptual'] = 0.0

        num_batches = 0
        grad_norms = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        for src, tar in pbar:
            src, tar = src.to(self.device), tar.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    pred = self.model(src)
                    losses = self.criterion(pred, tar)

                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(src)
                losses = self.criterion(pred, tar)
                losses['total'].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            grad_norms.append(grad_norm.item())

            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})

        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        epoch_losses['grad_norm'] = np.mean(grad_norms)
        return epoch_losses

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        epoch_losses = {'total': 0.0, 'l1': 0.0, 'window': 0.0}
        if self.args.use_perceptual:
            epoch_losses['perceptual'] = 0.0

        num_batches = 0

        for src, tar in tqdm(self.val_loader, desc="Validating"):
            src, tar = src.to(self.device), tar.to(self.device)

            pred = self.model(src)
            pred = torch.clamp(pred, 0, 1)
            losses = self.criterion(pred, tar)

            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            num_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }

        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')

    def train(self):
        self._log("\n🚀 Starting training...")

        for epoch in range(self.args.epochs):
            # Warmup
            if epoch < self.args.warmup_epochs:
                lr = self.args.lr * (epoch + 1) / self.args.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate()

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                best_str = " (best)"
            else:
                self.patience_counter += 1
                best_str = ""

            self._log(
                f"Epoch {epoch+1:3d}/{self.args.epochs}: "
                f"Train={train_losses['total']:.4f}, Val={val_losses['total']:.4f}, "
                f"L1={val_losses['l1']:.4f}, Win={val_losses['window']:.4f}, "
                f"Grad={train_losses['grad_norm']:.4f}, LR={lr:.2e}{best_str}"
            )

            # Save
            self.save_checkpoint(epoch + 1, val_losses['total'], is_best)

            # Early stopping
            if self.patience_counter >= self.args.patience:
                self._log(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break

        self._log("\n✅ Training complete!")
        self._log(f"Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--train_jsonl', type=str, default='data_splits/fold_1/train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='data_splits/fold_1/val.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_simple')

    # Model
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[4, 6, 6, 8])

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)

    # Loss
    parser.add_argument('--use_perceptual', action='store_true', help='Use VGG perceptual loss')

    # System
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    trainer = RobustTrainer(args)
    trainer.setup_data()
    trainer.setup_model()
    trainer.train()


if __name__ == '__main__':
    main()
