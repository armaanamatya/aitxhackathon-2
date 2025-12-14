#!/usr/bin/env python3
"""
Restormer 512 Training with Combined Loss (L1 + Window + Color)

Data Split Strategy:
- TEST: First 10 images (held out, NEVER seen during training)
- TRAIN: 90% of remaining 567 images (511 samples)
- VAL: 10% of remaining 567 images (56 samples)

Loss Components:
- L1 Loss (1.0): Primary metric, directly optimizes MAE
- Window Loss (0.5): Extra weight on bright window regions
- Color Loss (0.3): HSV saturation preservation

Author: Top 0.0001% MLE
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
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from torchvision import transforms as T

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import create_restormer


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    """HDR Real Estate Dataset with proper split handling"""

    def __init__(self, jsonl_path, resolution=512, augment=False):
        self.resolution = resolution
        self.augment = augment

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize
        input_img = TF.resize(input_img, (self.resolution, self.resolution),
                              interpolation=T.InterpolationMode.BILINEAR)
        target_img = TF.resize(target_img, (self.resolution, self.resolution),
                               interpolation=T.InterpolationMode.BILINEAR)

        # Light augmentation (horizontal flip only - preserves color)
        if self.augment and torch.rand(1) > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        # To tensor [0, 1]
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor


# =============================================================================
# Loss Functions
# =============================================================================

class WindowAwareLoss(nn.Module):
    """
    Window-aware loss that applies extra weight to bright regions.
    """

    def __init__(self, brightness_threshold=0.7, window_weight=2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.window_weight = window_weight

    def forward(self, pred, target):
        # Compute luminance
        luminance = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]

        # Create window mask (bright regions)
        window_mask = (luminance > self.brightness_threshold).float()

        # Smooth the mask
        window_mask = F.avg_pool2d(window_mask, kernel_size=15, stride=1, padding=7)
        window_mask = torch.clamp(window_mask * 2, 0, 1)

        # Weight mask: 1.0 for normal regions, window_weight for window regions
        weight_mask = 1.0 + (self.window_weight - 1.0) * window_mask

        # Weighted L1 loss
        pixel_loss = torch.abs(pred - target)
        weighted_loss = (pixel_loss * weight_mask).mean()

        return weighted_loss


class BrightRegionSaturationLoss(nn.Module):
    """
    Saturation loss focused on BRIGHT regions only.

    Problem: In windows/bright areas, colors like green plants, blue sky
    aren't captured correctly when source is dim but target is bright.

    Solution: Apply saturation loss specifically to bright regions in the target,
    forcing the model to learn vibrant colors where they should appear.
    """

    def __init__(self, brightness_threshold=0.5, saturation_weight=2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.saturation_weight = saturation_weight

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV. Input shape: [B, 3, H, W]"""
        max_rgb, _ = torch.max(rgb, dim=1, keepdim=True)
        min_rgb, _ = torch.min(rgb, dim=1, keepdim=True)
        diff = max_rgb - min_rgb + 1e-7

        # Value (brightness)
        v = max_rgb

        # Saturation
        s = diff / (max_rgb + 1e-7)

        return s, v

    def forward(self, pred, target):
        # Get saturation and value for both
        pred_sat, pred_val = self.rgb_to_hsv(pred)
        target_sat, target_val = self.rgb_to_hsv(target)

        # Create bright region mask from TARGET (where colors should be vibrant)
        bright_mask = (target_val > self.brightness_threshold).float()

        # Smooth the mask
        bright_mask = F.avg_pool2d(bright_mask, kernel_size=7, stride=1, padding=3)

        # Compute saturation loss only in bright regions
        # This forces the model to match saturation where target is bright
        sat_diff = torch.abs(pred_sat - target_sat)
        bright_sat_loss = (sat_diff * bright_mask).sum() / (bright_mask.sum() + 1e-7)

        # Also add a small global saturation loss
        global_sat_loss = F.l1_loss(pred_sat, target_sat)

        return self.saturation_weight * bright_sat_loss + 0.5 * global_sat_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: L1 + Window + Bright-Region Saturation

    Weights optimized for:
    - Primary metric (L1): weight 1.0
    - Window regions: weight 0.5 (secondary)
    - Bright-region saturation: weight 0.3 (color preservation in bright areas)
    """

    def __init__(self, l1_weight=1.0, window_weight=0.5, saturation_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.window_weight = window_weight
        self.saturation_weight = saturation_weight

        self.l1_loss = nn.L1Loss()
        self.window_loss = WindowAwareLoss()
        self.saturation_loss = BrightRegionSaturationLoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        window = self.window_loss(pred, target)
        saturation = self.saturation_loss(pred, target)

        total = self.l1_weight * l1 + self.window_weight * window + self.saturation_weight * saturation

        return total, {
            'l1': l1.item(),
            'window': window.item(),
            'saturation': saturation.item(),
            'total': total.item()
        }


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    loss_components = {'l1': 0, 'window': 0, 'saturation': 0}

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, components = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in components.items():
            if k != 'total':
                loss_components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    loss_components = {'l1': 0, 'window': 0, 'saturation': 0}

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, components = criterion(outputs, targets)

        total_loss += loss.item()
        for k, v in components.items():
            if k != 'total':
                loss_components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


def main():
    parser = argparse.ArgumentParser(description='Restormer 512 Combined Loss Training')

    # Data paths (using proper split)
    parser.add_argument('--train_jsonl', type=str, default='data_splits/proper_split/train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='data_splits/proper_split/val.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_512_combined')

    # Model
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--model_size', type=str, default='base')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--window_weight', type=float, default=0.5)
    parser.add_argument('--color_weight', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("RESTORMER 512 - COMBINED LOSS (L1 + Window + Color)")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Resolution: {args.resolution}")
    print(f"Loss weights: L1={args.l1_weight}, Window={args.window_weight}, Color={args.color_weight}")
    print()

    # Data
    print("Loading data (proper split - test set held out)...")
    train_dataset = HDRDataset(args.train_jsonl, args.resolution, augment=True)
    val_dataset = HDRDataset(args.val_jsonl, args.resolution, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print()

    # Model
    print("Creating Restormer model...")
    model = create_restormer(args.model_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # Loss
    criterion = CombinedLoss(args.l1_weight, args.window_weight, args.color_weight)
    print(f"Loss: L1({args.l1_weight}) + Window({args.window_weight}) + Color({args.color_weight})")
    print()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_loss = float('inf')
    best_val_l1 = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_l1': [], 'lr': []}

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.grad_clip
        )

        # Validate
        val_loss, val_components = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_components['l1'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - start_time

        # Check for best (using val L1 as primary metric)
        is_best = val_components['l1'] < best_val_l1

        if is_best:
            best_val_loss = val_loss
            best_val_l1 = val_components['l1']
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_l1': best_val_l1,
                'args': vars(args)
            }, output_dir / 'checkpoint_best.pt')

        else:
            patience_counter += 1

        # Print progress
        best_marker = "🌟 BEST" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"L1: {val_components['l1']:.4f} | Win: {val_components['window']:.4f} | "
              f"Sat: {val_components['saturation']:.4f} | LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s {best_marker}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_l1': val_components['l1'],
            'args': vars(args)
        }, output_dir / 'checkpoint_latest.pt')

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 80)
    print(f"Training complete!")
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
