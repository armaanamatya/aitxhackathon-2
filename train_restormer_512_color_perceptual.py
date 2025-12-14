#!/usr/bin/env python3
"""
Restormer 512 Training with Color + Perceptual + SSIM Loss

Focus: Accurate color reproduction across ALL bright colors in the image

Loss Components:
- L1 (1.0): Pixel accuracy
- Perceptual (0.1): VGG feature similarity (textures, colors, patterns)
- SSIM (0.1): Structural similarity
- Color (0.3): HSV color matching (Hue + Saturation + Value)
- Lab (0.2): Perceptually uniform color space matching

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
from training.color_perceptual_loss import ColorPerceptualLoss


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    """HDR Real Estate Dataset"""

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

        # Light augmentation
        if self.augment and torch.rand(1) > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    loss_components = {}

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
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    loss_components = {}

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, components = criterion(outputs, targets)

        total_loss += loss.item()
        for k, v in components.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


def main():
    parser = argparse.ArgumentParser(description='Restormer 512 with Color + Perceptual + SSIM Loss')

    # Data paths
    parser.add_argument('--train_jsonl', type=str, default='data_splits/proper_split/train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='data_splits/proper_split/val.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_512_color_perceptual')

    # Model
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--model_size', type=str, default='base')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--perceptual_weight', type=float, default=0.1)
    parser.add_argument('--ssim_weight', type=float, default=0.1)
    parser.add_argument('--color_weight', type=float, default=0.3)
    parser.add_argument('--lab_weight', type=float, default=0.2)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
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
    print("RESTORMER 512 - COLOR + PERCEPTUAL + SSIM LOSS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Resolution: {args.resolution}")
    print()
    print("LOSS COMPONENTS (targeting ALL bright colors):")
    print(f"  L1: {args.l1_weight} (pixel accuracy)")
    print(f"  Perceptual: {args.perceptual_weight} (VGG features)")
    print(f"  SSIM: {args.ssim_weight} (structural similarity)")
    print(f"  Color: {args.color_weight} (HSV: Hue + Saturation)")
    print(f"  Lab: {args.lab_weight} (perceptually uniform color)")
    print()

    # Data
    print("Loading data...")
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
    criterion = ColorPerceptualLoss(
        l1_weight=args.l1_weight,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        color_weight=args.color_weight,
        lab_weight=args.lab_weight,
        use_perceptual=True,
    ).to(device)
    print("Loss: L1 + Perceptual(VGG) + SSIM + Color(HSV) + Lab")
    print()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
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

        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.grad_clip
        )

        val_loss, val_components = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_components.get('l1', 0))
        history['lr'].append(current_lr)

        epoch_time = time.time() - start_time

        val_l1 = val_components.get('l1', val_loss)
        is_best = val_l1 < best_val_l1

        if is_best:
            best_val_loss = val_loss
            best_val_l1 = val_l1
            patience_counter = 0

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
        best_marker = " BEST" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"L1: {val_components.get('l1', 0):.4f} | "
              f"Perc: {val_components.get('perceptual', 0):.4f} | "
              f"SSIM: {val_components.get('ssim', 0):.4f} | "
              f"Color: {val_components.get('color', 0):.4f} | "
              f"Lab: {val_components.get('lab', 0):.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s{best_marker}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_l1': val_l1,
            'args': vars(args)
        }, output_dir / 'checkpoint_latest.pt')

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 80)
    print(f"Training complete!")
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
