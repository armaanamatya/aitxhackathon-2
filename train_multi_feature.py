#!/usr/bin/env python3
"""
Multi-Feature Window Recovery Training
======================================

Uses multiple auxiliary features for robust window recovery:
- Depth: Scene geometry (windows = far)
- Edge: Structural details for preservation
- Saturation: Color richness (windows = washed out)
- Contrast: Local contrast for enhancement

All features computed on-the-fly - no external models needed.
Memory efficient and robust.
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
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.multi_feature_restormer import create_multi_feature_restormer
from training.delta_aware_loss import create_robust_window_loss


class HDRDataset(Dataset):
    def __init__(self, jsonl_path, resolution=512, augment=False):
        self.resolution = resolution
        self.augment = augment

        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        input_img = TF.resize(input_img, (self.resolution, self.resolution))
        target_img = TF.resize(target_img, (self.resolution, self.resolution))

        if self.augment:
            # Horizontal flip
            if torch.rand(1) > 0.5:
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)

            # Random brightness (helps generalization)
            if torch.rand(1) > 0.7:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                input_img = TF.adjust_brightness(input_img, factor)

        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor


def compute_metrics(pred, target):
    with torch.no_grad():
        l1 = F.l1_loss(pred, target).item()
        mse = F.mse_loss(pred, target)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
        return l1, psnr


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, accum_steps=1):
    model.train()

    total_loss = 0
    zone_losses = {'needs_heavy_fix': 0, 'needs_light_fix': 0, 'preserve': 0}

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (input_img, target_img) in enumerate(pbar):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        with autocast('cuda'):
            # Model extracts features internally from RGB
            output = model(input_img)
            output = torch.clamp(output, 0, 1)

            loss, components, zones = criterion(output, target_img, input_img, return_components=True)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

        # Track zone losses
        for zone in zone_losses.keys():
            key = f'{zone}_l1'
            if key in components:
                val = components[key]
                zone_losses[zone] += val.item() if isinstance(val, torch.Tensor) else val

        # Progress bar
        heavy = components.get('needs_heavy_fix_l1', torch.tensor(0.0))
        preserve = components.get('preserve_l1', torch.tensor(0.0))
        if isinstance(heavy, torch.Tensor):
            heavy = heavy.item()
        if isinstance(preserve, torch.Tensor):
            preserve = preserve.item()

        pbar.set_postfix({
            'loss': f"{loss.item() * accum_steps:.4f}",
            'fix': f"{heavy:.3f}",
            'preserve': f"{preserve:.3f}",
        })

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in zone_losses.items()}


def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    metrics = {'l1': [], 'psnr': []}
    zone_metrics = {'needs_heavy_fix': [], 'needs_light_fix': [], 'preserve': []}

    with torch.no_grad():
        for input_img, target_img in tqdm(dataloader, desc="Val"):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            output = model(input_img)
            output = torch.clamp(output, 0, 1)

            loss, components, _ = criterion(output, target_img, input_img, return_components=True)
            total_loss += loss.item()

            l1, psnr = compute_metrics(output, target_img)
            metrics['l1'].append(l1)
            metrics['psnr'].append(psnr)

            for zone in zone_metrics.keys():
                key = f'{zone}_l1'
                if key in components:
                    val = components[key]
                    zone_metrics[zone].append(val.item() if isinstance(val, torch.Tensor) else val)

    return (
        total_loss / len(dataloader),
        np.mean(metrics['l1']),
        np.mean(metrics['psnr']),
        {k: np.mean(v) if v else 0 for k, v in zone_metrics.items()}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accum_steps', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--features', type=str, nargs='+',
                        default=['depth', 'edge', 'saturation'],
                        choices=['depth', 'edge', 'saturation', 'contrast', 'luminance'])
    parser.add_argument('--output_dir', type=str, default='outputs_multi_feature')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--preset', type=str, default='robust')
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-FEATURE WINDOW RECOVERY TRAINING")
    print("=" * 70)
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Model: {args.model_size}")
    print(f"Features: RGB + {args.features}")
    print(f"Total input channels: {3 + len(args.features)}")
    print(f"Batch: {args.batch_size} x {args.accum_steps} = {args.batch_size * args.accum_steps} effective")
    print(f"Epochs: {args.epochs}")
    print(f"Preset: {args.preset}")
    print()
    print("Key Features:")
    print("  - Multi-feature input for robust scene understanding")
    if 'depth' in args.features:
        print("  - Depth: Windows (far) vs interior (close)")
    if 'edge' in args.features:
        print("  - Edge: Structural detail preservation")
    if 'saturation' in args.features:
        print("  - Saturation: Identifies washed-out windows")
    if 'contrast' in args.features:
        print("  - Contrast: Local enhancement guidance")
    print("  - Delta-aware loss for robust training")
    print("  - No external models needed - all features computed on-the-fly")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Data
    train_dataset = HDRDataset('data_splits/proper_split/train.jsonl', args.resolution, augment=True)
    val_dataset = HDRDataset('data_splits/proper_split/val.jsonl', args.resolution)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples")
    print()

    # Model - Multi-feature Restormer
    model = create_multi_feature_restormer(
        args.model_size,
        args.features,
        use_checkpointing=True
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print()

    # Loss - Delta-aware for robust training
    criterion = create_robust_window_loss(args.preset)
    print("Loss: Robust Delta-Aware Loss")
    print(f"  Heavy fix weight: {criterion.delta_loss.weights['needs_heavy_fix']}")
    print(f"  Light fix weight: {criterion.delta_loss.weights['needs_light_fix']}")
    print(f"  Preserve weight: {criterion.delta_loss.weights['preserve']}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate schedule
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (args.epochs - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    # Resume
    start_epoch = 0
    best_val_l1 = float('inf')
    best_heavy_fix = float('inf')

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_l1 = ckpt.get('best_val_l1', float('inf'))
        best_heavy_fix = ckpt.get('best_heavy_fix', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # History
    history = {'train_loss': [], 'val_loss': [], 'val_l1': [], 'zone_metrics': []}

    # Training
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        train_loss, train_zones = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch+1, args.accum_steps
        )

        val_loss, val_l1, val_psnr, val_zones = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val: L1={val_l1:.4f}, PSNR={val_psnr:.2f}dB")
        print(f"Zone L1: heavy_fix={val_zones['needs_heavy_fix']:.4f}, "
              f"light_fix={val_zones['needs_light_fix']:.4f}, preserve={val_zones['preserve']:.4f}")
        print(f"LR: {lr:.6f}")

        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_l1)
        history['zone_metrics'].append(val_zones)

        # Save
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_l1': val_l1,
            'val_psnr': val_psnr,
            'zone_metrics': val_zones,
            'best_val_l1': best_val_l1,
            'best_heavy_fix': best_heavy_fix,
            'config': config,
        }

        torch.save(ckpt, output_dir / 'checkpoint_last.pt')

        # Best overall
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            ckpt['best_val_l1'] = best_val_l1
            torch.save(ckpt, output_dir / 'checkpoint_best.pt')
            print(f"  -> New best overall: {val_l1:.4f}")

        # Best heavy fix (primary metric for window recovery)
        if val_zones['needs_heavy_fix'] < best_heavy_fix:
            best_heavy_fix = val_zones['needs_heavy_fix']
            ckpt['best_heavy_fix'] = best_heavy_fix
            torch.save(ckpt, output_dir / 'checkpoint_best_window.pt')
            print(f"  -> New best window fix: {val_zones['needs_heavy_fix']:.4f}")

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Best Window Fix: {best_heavy_fix:.4f}")
    print(f"Output: {output_dir}/")


if __name__ == '__main__':
    main()
