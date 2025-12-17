#!/usr/bin/env python3
"""
Train Restormer with SOTA Color Enhancement Loss

Uses the complete SOTA color loss for optimal color matching:
- Focal Frequency Loss (CVPR 2021)
- LAB Perceptual Loss
- Multi-Scale Color Loss
- Color Curve Learning
- Color Histogram Matching
- Window Color Boost

Author: Top 0.0001% MLE
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import create_restormer
from training.sota_color_loss import SOTAColorEnhancementLoss


class HDRDataset(Dataset):
    """Dataset for HDR image enhancement with proper data split."""

    def __init__(self, jsonl_path, resolution=512, augment=False):
        self.resolution = resolution
        self.augment = augment

        # Load samples from JSONL
        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize to target resolution
        input_img = TF.resize(input_img, (self.resolution, self.resolution))
        target_img = TF.resize(target_img, (self.resolution, self.resolution))

        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)

            # Random vertical flip
            if torch.rand(1) > 0.5:
                input_img = TF.vflip(input_img)
                target_img = TF.vflip(target_img)

        # Convert to tensors
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor


def compute_metrics(pred, target):
    """Compute L1, PSNR, SSIM for validation."""
    with torch.no_grad():
        l1 = F.l1_loss(pred, target).item()

        mse = F.mse_loss(pred, target)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()

        # Simple SSIM
        window_size = 11
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred ** 2
        sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred * mu_target

        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return l1, psnr, ssim.mean().item()


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    loss_components_sum = {
        'l1': 0, 'focal_freq': 0, 'lab': 0, 'multiscale': 0,
        'color_curve': 0, 'histogram': 0, 'window_boost': 0
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (input_img, target_img) in enumerate(pbar):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            output = model(input_img)
            output = torch.clamp(output, 0, 1)

            # SOTA color loss (requires input for color curve learning)
            loss, components = criterion(output, target_img, input_img)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss += loss.item()
        for k, v in components.items():
            if k != 'total' and k in loss_components_sum:
                loss_components_sum[k] += v

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l1': f"{components['l1']:.4f}",
            'lab': f"{components['lab']:.4f}",
        })

    # Average metrics
    n = len(dataloader)
    avg_loss = total_loss / n
    avg_components = {k: v / n for k, v in loss_components_sum.items()}

    return avg_loss, avg_components


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()

    total_loss = 0
    metrics = {'l1': [], 'psnr': [], 'ssim': []}

    with torch.no_grad():
        for input_img, target_img in tqdm(dataloader, desc="Validating"):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            output = model(input_img)
            output = torch.clamp(output, 0, 1)

            # Compute loss
            loss, _ = criterion(output, target_img, input_img)
            total_loss += loss.item()

            # Compute metrics
            l1, psnr, ssim = compute_metrics(output, target_img)
            metrics['l1'].append(l1)
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)

    avg_loss = total_loss / len(dataloader)
    avg_l1 = np.mean(metrics['l1'])
    avg_psnr = np.mean(metrics['psnr'])
    avg_ssim = np.mean(metrics['ssim'])

    return avg_loss, avg_l1, avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_512_sota_color')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    print("="*80)
    print("TRAINING: Restormer with SOTA Color Enhancement Loss")
    print("="*80)
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print()

    # Data loaders
    print("Loading datasets...")
    train_dataset = HDRDataset(
        'data_splits/proper_split/train.jsonl',
        resolution=args.resolution,
        augment=True
    )
    val_dataset = HDRDataset(
        'data_splits/proper_split/val.jsonl',
        resolution=args.resolution,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print()

    # Model
    print("Creating model...")
    model = create_restormer('base').to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Loss function - SOTA Color Enhancement Loss
    print("Creating SOTA Color Enhancement Loss...")
    criterion = SOTAColorEnhancementLoss(
        l1_weight=1.0,
        focal_freq_weight=0.3,
        lab_weight=0.4,
        multiscale_weight=0.3,
        color_curve_weight=0.4,
        histogram_weight=0.2,
        window_boost_weight=0.5,
    ).to(device)
    print("Loss components:")
    print("  - L1 Loss (1.0)")
    print("  - Focal Frequency Loss (0.3) - CVPR 2021")
    print("  - LAB Perceptual Loss (0.4)")
    print("  - Multi-Scale Color Loss (0.3)")
    print("  - Color Curve Learning (0.4)")
    print("  - Histogram Matching (0.2)")
    print("  - Window Color Boost (0.5)")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Load checkpoint if specified
    start_epoch = 0
    best_val_l1 = float('inf')

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_l1 = ckpt.get('best_val_l1', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best val L1: {best_val_l1:.4f}")
        print()

    # Training loop
    print("="*80)
    print("TRAINING START")
    print("="*80)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch+1
        )

        print(f"Train Loss: {train_loss:.4f}")
        print("  Components:")
        for k, v in train_components.items():
            print(f"    {k}: {v:.4f}")

        # Validate
        val_loss, val_l1, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device
        )

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val L1: {val_l1:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_l1': val_l1,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'best_val_l1': best_val_l1,
            'train_components': train_components,
        }

        # Save last checkpoint
        torch.save(checkpoint, output_dir / 'checkpoint_last.pt')

        # Save best checkpoint
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            checkpoint['best_val_l1'] = best_val_l1
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"✓ New best model! Val L1: {val_l1:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Checkpoints saved to: {output_dir}/")
    print()


if __name__ == '__main__':
    main()
