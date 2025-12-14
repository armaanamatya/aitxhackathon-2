#!/usr/bin/env python3
"""
Simple DarkIR Training Script - Robust and Stable
Based on working train_restormer_cleaned.py
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm

# Add DarkIR to path
sys.path.insert(0, str(Path(__file__).parent / 'DarkIR'))
from archs.DarkIR import DarkIR


class RealEstateDataset(Dataset):
    """Real estate image dataset"""

    def __init__(self, jsonl_path: str, base_dir: str, resolution: int):
        self.base_dir = base_dir
        self.resolution = resolution

        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        src_path = os.path.join(self.base_dir, pair['src'])
        tar_path = os.path.join(self.base_dir, pair['tar'])

        src = cv2.imread(src_path)
        tar = cv2.imread(tar_path)

        if src is None:
            raise ValueError(f"Could not load: {src_path}")
        if tar is None:
            raise ValueError(f"Could not load: {tar_path}")

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        # Resize
        src = cv2.resize(src, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        tar = cv2.resize(tar, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # To tensor [0, 1]
        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    count = 0

    for src, tar in tqdm(dataloader, desc="Training", leave=False):
        src, tar = src.to(device), tar.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                out = model(src)
                loss = criterion(out, tar)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(src)
            loss = criterion(out, tar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count


def validate(model, dataloader, criterion, device):
    """Validate"""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for src, tar in dataloader:
            src, tar = src.to(device), tar.to(device)
            out = model(src)
            loss = criterion(out, tar)
            total_loss += loss.item()
            count += 1

    return total_loss / count


def main():
    parser = argparse.ArgumentParser(description="Train DarkIR")

    # Data
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--val_jsonl', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--resolution', type=int, default=512)

    # Model
    parser.add_argument('--width', type=int, default=32, help='32 for DarkIR-m, 64 for DarkIR-l')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=15)

    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_every', type=int, default=10)

    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 70)
    print("DARKIR TRAINING")
    print("=" * 70)
    print(f"Resolution: {args.resolution}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Output: {args.output_dir}")

    # Create datasets
    train_dataset = RealEstateDataset(args.train_jsonl, args.base_dir, args.resolution)
    val_dataset = RealEstateDataset(args.val_jsonl, args.base_dir, args.resolution)

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Create model
    print(f"\nCreating DarkIR model (width={args.width})...")
    model = DarkIR(
        img_channel=3,
        width=args.width,
        middle_blk_num_enc=2,
        middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3],
        dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def warmup_cosine_scheduler(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_scheduler)

    # Loss - simple L1
    criterion = nn.L1Loss()

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None

    # Training loop
    print(f"\nStarting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, scaler)
        val_loss = validate(model, val_loader, criterion, args.device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"LR={current_lr:.2e} {'(best)' if is_best else ''}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

        # Save checkpoint
        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
