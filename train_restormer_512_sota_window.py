#!/usr/bin/env python3
"""
SOTA Window Recovery Training Script
=====================================

Uses learned luminance attention maps (heat maps) for:
1. Adaptive exposure zone detection (not fixed thresholds)
2. Zone-specific losses (different treatment per zone)
3. Boundary preservation at window frames
4. Multi-scale processing

This is the OPTIMAL ROBUST SOLUTION for window brightness issues.

References:
- OENet (2024): Attention-Guided Feature Fusion
- AGCSNet (2024): Automatic illumination-map attention
- Learning Adaptive Lighting (2024): Channel-aware guidance
- CEVR (ICCV 2023): Continuous exposure value representations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

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
from training.luminance_attention_loss import (
    SOTAWindowRecoveryLoss,
    create_sota_window_loss,
    visualize_zone_attention
)


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

            # Random vertical flip (less common for real estate)
            if torch.rand(1) > 0.85:
                input_img = TF.vflip(input_img)
                target_img = TF.vflip(target_img)

            # Random rotation (90 degree increments)
            if torch.rand(1) > 0.7:
                angle = int(torch.randint(1, 4, (1,)).item()) * 90
                input_img = TF.rotate(input_img, angle)
                target_img = TF.rotate(target_img, angle)

        # Convert to tensors [0, 1]
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
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, 11, stride=1, padding=5)
        mu_target = F.avg_pool2d(target, 11, stride=1, padding=5)

        sigma_pred_sq = F.avg_pool2d(pred ** 2, 11, stride=1, padding=5) - mu_pred ** 2
        sigma_target_sq = F.avg_pool2d(target ** 2, 11, stride=1, padding=5) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_pred * mu_target

        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return l1, psnr, ssim.mean().item()


def compute_zone_metrics(pred, target, source, criterion):
    """Compute metrics per exposure zone."""
    with torch.no_grad():
        _, zone_dict = criterion.zone_loss.attention(source)

        zone_l1 = {}
        for zone_name, zone_mask in zone_dict.items():
            mask_sum = zone_mask.sum() + 1e-8
            zone_error = (torch.abs(pred - target) * zone_mask).sum() / mask_sum
            zone_l1[zone_name] = zone_error.item()

        return zone_l1


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, log_interval=50):
    """Train for one epoch with zone-aware logging."""
    model.train()

    total_loss = 0
    zone_losses = {
        'blown_out': 0, 'highlight': 0, 'midtone': 0, 'shadow': 0, 'deep_shadow': 0
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

            # SOTA window recovery loss
            loss, components, zone_dict = criterion(
                output, target_img, input_img, return_components=True
            )

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss += loss.item()

        # Track zone-specific losses
        for zone_name in zone_losses.keys():
            key = f'{zone_name}_l1'
            if key in components:
                zone_losses[zone_name] += components[key].item() if isinstance(components[key], torch.Tensor) else components[key]

        # Update progress bar
        blown_out_l1 = components.get('blown_out_l1', 0)
        if isinstance(blown_out_l1, torch.Tensor):
            blown_out_l1 = blown_out_l1.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'blown_out': f"{blown_out_l1:.4f}",
            'boundary': f"{components['boundary'].item():.4f}",
        })

    # Average metrics
    n = len(dataloader)
    avg_loss = total_loss / n
    avg_zone_losses = {k: v / n for k, v in zone_losses.items()}

    return avg_loss, avg_zone_losses


def validate(model, dataloader, criterion, device):
    """Validate the model with zone-specific metrics."""
    model.eval()

    total_loss = 0
    metrics = {'l1': [], 'psnr': [], 'ssim': []}
    zone_metrics = {
        'blown_out': [], 'highlight': [], 'midtone': [], 'shadow': [], 'deep_shadow': []
    }

    with torch.no_grad():
        for input_img, target_img in tqdm(dataloader, desc="Validating"):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            output = model(input_img)
            output = torch.clamp(output, 0, 1)

            # Compute loss
            loss, components, _ = criterion(output, target_img, input_img, return_components=True)
            total_loss += loss.item()

            # Compute global metrics
            l1, psnr, ssim = compute_metrics(output, target_img)
            metrics['l1'].append(l1)
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)

            # Compute zone metrics
            zone_l1 = compute_zone_metrics(output, target_img, input_img, criterion)
            for zone_name, zone_error in zone_l1.items():
                zone_metrics[zone_name].append(zone_error)

    avg_loss = total_loss / len(dataloader)
    avg_l1 = np.mean(metrics['l1'])
    avg_psnr = np.mean(metrics['psnr'])
    avg_ssim = np.mean(metrics['ssim'])
    avg_zone_metrics = {k: np.mean(v) for k, v in zone_metrics.items()}

    return avg_loss, avg_l1, avg_psnr, avg_ssim, avg_zone_metrics


def save_zone_visualization(model, criterion, dataloader, device, output_dir, epoch):
    """Save visualization of zone attention maps."""
    model.eval()

    with torch.no_grad():
        # Get first batch
        input_img, target_img = next(iter(dataloader))
        input_img = input_img.to(device)[:4]  # First 4 samples
        target_img = target_img.to(device)[:4]

        output = model(input_img)
        output = torch.clamp(output, 0, 1)

        # Get zone attention maps
        _, zone_dict = criterion.zone_loss.attention(input_img)

        # Visualize
        zone_viz = visualize_zone_attention(zone_dict)

        # Save images
        from torchvision.utils import save_image

        save_dir = Path(output_dir) / 'visualizations'
        save_dir.mkdir(exist_ok=True)

        save_image(input_img, save_dir / f'epoch_{epoch}_input.png', nrow=2)
        save_image(target_img, save_dir / f'epoch_{epoch}_target.png', nrow=2)
        save_image(output, save_dir / f'epoch_{epoch}_output.png', nrow=2)
        save_image(zone_viz, save_dir / f'epoch_{epoch}_zones.png', nrow=2)

        # Save individual zone maps
        for zone_name, zone_mask in zone_dict.items():
            save_image(zone_mask, save_dir / f'epoch_{epoch}_zone_{zone_name}.png', nrow=2)


def main():
    parser = argparse.ArgumentParser(description='SOTA Window Recovery Training')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_512_sota_window')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None, help='Load pretrained weights')
    parser.add_argument('--preset', type=str, default='aggressive',
                        choices=['conservative', 'balanced', 'aggressive'],
                        help='Loss preset (aggressive recommended for window issues)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print("SOTA WINDOW RECOVERY TRAINING")
    print("=" * 80)
    print(f"Started: {timestamp}")
    print()
    print("Configuration:")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Loss preset: {args.preset}")
    print(f"  Output dir: {args.output_dir}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['timestamp'] = timestamp
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
        pin_memory=True,
        drop_last=True
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
    print("Creating Restormer model...")
    model = create_restormer('base').to(device)

    # Load pretrained weights if specified
    if args.pretrained:
        print(f"Loading pretrained weights from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("Pretrained weights loaded!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # SOTA Loss function with learned attention
    print("Creating SOTA Window Recovery Loss...")
    print(f"  Using preset: {args.preset}")
    criterion = create_sota_window_loss(preset=args.preset)

    # Move attention network to device and set as trainable
    criterion.zone_loss.attention.to(device)
    criterion.boundary_loss.to(device)

    print()
    print("Loss Components:")
    print("  - Learned Luminance Attention (adaptive zone detection)")
    print("  - Zone-specific losses:")
    print(f"    - Blown-out weight: {criterion.zone_loss.zone_weights['blown_out']}")
    print(f"    - Highlight weight: {criterion.zone_loss.zone_weights['highlight']}")
    print(f"    - Midtone weight: {criterion.zone_loss.zone_weights['midtone']}")
    print(f"    - Shadow weight: {criterion.zone_loss.zone_weights['shadow']}")
    print("  - Color direction loss (ensures correct transformation)")
    print("  - Saturation matching")
    print("  - Local contrast preservation")
    print("  - Boundary preservation")
    print("  - VGG perceptual loss")
    print()

    # Optimizer - include attention network parameters
    all_params = list(model.parameters()) + list(criterion.zone_loss.attention.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler()

    # Load checkpoint if specified
    start_epoch = 0
    best_val_l1 = float('inf')
    best_blown_out_l1 = float('inf')

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_l1 = ckpt.get('best_val_l1', float('inf'))
        best_blown_out_l1 = ckpt.get('best_blown_out_l1', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
        print(f"  Best val L1: {best_val_l1:.4f}")
        print(f"  Best blown-out L1: {best_blown_out_l1:.4f}")
        print()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_l1': [],
        'val_psnr': [],
        'val_ssim': [],
        'zone_metrics': []
    }

    # Training loop
    print("=" * 80)
    print("TRAINING START")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_zone_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch+1
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print("  Zone losses:")
        for zone_name, zone_loss in train_zone_losses.items():
            print(f"    {zone_name}: {zone_loss:.4f}")

        # Validate
        val_loss, val_l1, val_psnr, val_ssim, val_zone_metrics = validate(
            model, val_loader, criterion, device
        )

        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"Val Metrics: L1={val_l1:.4f}, PSNR={val_psnr:.2f}dB, SSIM={val_ssim:.4f}")
        print("  Zone L1 metrics:")
        for zone_name, zone_l1 in val_zone_metrics.items():
            indicator = " <-- TARGET" if zone_name in ['blown_out', 'highlight'] else ""
            print(f"    {zone_name}: {zone_l1:.4f}{indicator}")

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nLearning rate: {current_lr:.6f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_l1)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['zone_metrics'].append(val_zone_metrics)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'attention_state_dict': criterion.zone_loss.attention.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_l1': val_l1,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'val_zone_metrics': val_zone_metrics,
            'best_val_l1': best_val_l1,
            'best_blown_out_l1': best_blown_out_l1,
            'config': config,
        }

        # Save last checkpoint
        torch.save(checkpoint, output_dir / 'checkpoint_last.pt')

        # Save best overall checkpoint
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            checkpoint['best_val_l1'] = best_val_l1
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"  New best overall model! Val L1: {val_l1:.4f}")

        # Save best blown-out checkpoint (primary target)
        blown_out_l1 = val_zone_metrics['blown_out']
        if blown_out_l1 < best_blown_out_l1:
            best_blown_out_l1 = blown_out_l1
            checkpoint['best_blown_out_l1'] = best_blown_out_l1
            torch.save(checkpoint, output_dir / 'checkpoint_best_window.pt')
            print(f"  New best WINDOW model! Blown-out L1: {blown_out_l1:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # Save visualizations periodically
        if (epoch + 1) % 5 == 0:
            try:
                save_zone_visualization(model, criterion, val_loader, device, output_dir, epoch+1)
            except Exception as e:
                print(f"  Warning: Could not save visualization: {e}")

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Best Blown-out L1: {best_blown_out_l1:.4f}")
    print(f"Checkpoints saved to: {output_dir}/")
    print()
    print("Key files:")
    print(f"  - checkpoint_best.pt: Best overall model")
    print(f"  - checkpoint_best_window.pt: Best for window regions")
    print(f"  - visualizations/: Zone attention heat maps")
    print()


if __name__ == '__main__':
    main()
