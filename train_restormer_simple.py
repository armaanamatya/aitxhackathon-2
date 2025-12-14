#!/usr/bin/env python3
"""
Simple Restormer Training with 3-Fold Cross-Validation
======================================================
Optimized single-model training from scratch.

Usage:
    python3 train_restormer_simple.py \
        --data_splits_dir data_splits \
        --resolution 512 \
        --batch_size 8 \
        --epochs 150 \
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torchvision.models as models

# Add model path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer


# ============================================================================
# DATASET
# ============================================================================

class RealEstateDataset(Dataset):
    """Real estate HDR dataset with augmentation."""

    def __init__(self, jsonl_path: str, base_dir: str = '.', resolution: int = 384, augment: bool = True):
        self.base_dir = Path(base_dir)
        self.resolution = resolution
        self.augment = augment

        # Load pairs from JSONL
        self.pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.pairs.append(json.loads(line))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load images
        src_path = self.base_dir / pair['src']
        tar_path = self.base_dir / pair['tar']

        src = cv2.imread(str(src_path))
        tar = cv2.imread(str(tar_path))

        if src is None or tar is None:
            raise ValueError(f"Failed to load images: {src_path}, {tar_path}")

        # Convert BGR to RGB
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        # Resize
        src = cv2.resize(src, (self.resolution, self.resolution))
        tar = cv2.resize(tar, (self.resolution, self.resolution))

        # Random augmentation
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                src = cv2.flip(src, 1)
                tar = cv2.flip(tar, 1)

            # Vertical flip
            if np.random.rand() > 0.5:
                src = cv2.flip(src, 0)
                tar = cv2.flip(tar, 0)

            # Rotation (90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                src = np.rot90(src, k).copy()
                tar = np.rot90(tar, k).copy()

        # Normalize to [0, 1]
        src = src.astype(np.float32) / 255.0
        tar = tar.astype(np.float32) / 255.0

        # To tensor [C, H, W]
        src = torch.from_numpy(src).permute(2, 0, 1)
        tar = torch.from_numpy(tar).permute(2, 0, 1)

        return src, tar


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        for layer in self.feature_layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        return loss / len(self.feature_layers)


def ssim_loss(pred, target, window_size=11):
    """SSIM loss."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

    sigma_x = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size//2) - mu_x**2
    sigma_y = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size//2) - mu_y**2
    sigma_xy = F.avg_pool2d(pred*target, window_size, stride=1, padding=window_size//2) - mu_x*mu_y

    ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim.mean()


def compute_psnr(pred, target):
    """Compute PSNR."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ============================================================================
# TRAINING
# ============================================================================

def train_fold(fold_num: int, train_jsonl: str, val_jsonl: str, args):
    """Train a single fold."""
    print(f"{'='*80}")
    print(f"FOLD {fold_num + 1}/{args.n_folds}")
    print(f"{'='*80}\n")

    device = torch.device(args.device)

    # Create datasets
    train_dataset = RealEstateDataset(train_jsonl, resolution=args.resolution, augment=True)
    val_dataset = RealEstateDataset(val_jsonl, resolution=args.resolution, augment=False)

    print(f"📊 Dataset:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples\n")

    # Model
    model = Restormer(
        in_channels=3,
        out_channels=3,
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=args.num_refinement_blocks,
        heads=args.heads,
        ffn_expansion_factor=args.ffn_expansion_factor,
        bias=args.bias,
        use_checkpointing=args.use_checkpointing
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🤖 Model: Restormer")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Trainable ratio: {100.0 * trainable_params / total_params:.1f}%\n")

    # Loss functions
    vgg_loss = VGGPerceptualLoss().to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Scheduler
    total_steps = len(train_dataset) // args.batch_size * args.epochs
    warmup_steps = len(train_dataset) // args.batch_size * args.warmup_epochs
    pct_start = min(warmup_steps / total_steps, 0.3) if total_steps > 0 else 0.1

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=pct_start, anneal_strategy='cos'
    )

    # Mixed precision
    scaler = GradScaler() if args.mixed_precision else None

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    print(f"🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.early_stopping_patience}\n")

    # Output directory
    output_dir = Path(args.output_dir) / f'fold_{fold_num + 1}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_psnr = 0.0
    patience_counter = 0
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_psnrs': [],
        'best_epoch': 0,
        'best_val_psnr': 0.0
    }

    for epoch in range(args.epochs):
        print(f"{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc="Training", leave=False)

        for src, tar in train_pbar:
            src, tar = src.to(device), tar.to(device)

            optimizer.zero_grad()

            if args.mixed_precision:
                with autocast():
                    pred = model(src)
                    loss_l1 = F.l1_loss(pred, tar)
                    loss_vgg = vgg_loss(pred, tar)
                    loss_ssim = ssim_loss(pred, tar)
                    loss = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_ssim * loss_ssim

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(src)
                loss_l1 = F.l1_loss(pred, tar)
                loss_vgg = vgg_loss(pred, tar)
                loss_ssim = ssim_loss(pred, tar)
                loss = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_ssim * loss_ssim

                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        history['train_losses'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for src, tar in val_pbar:
                src, tar = src.to(device), tar.to(device)

                if args.mixed_precision:
                    with autocast():
                        pred = model(src)
                        loss_l1 = F.l1_loss(pred, tar)
                        loss_vgg = vgg_loss(pred, tar)
                        loss_ssim = ssim_loss(pred, tar)
                        loss = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_ssim * loss_ssim
                else:
                    pred = model(src)
                    loss_l1 = F.l1_loss(pred, tar)
                    loss_vgg = vgg_loss(pred, tar)
                    loss_ssim = ssim_loss(pred, tar)
                    loss = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_ssim * loss_ssim

                val_loss += loss.item()
                val_psnr += compute_psnr(pred, tar).item()

                val_pbar.set_postfix({'loss': loss.item(), 'psnr': compute_psnr(pred, tar).item()})

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        history['val_losses'].append(val_loss)
        history['val_psnrs'].append(val_psnr)

        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val PSNR: {val_psnr:.2f} dB")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            history['best_val_psnr'] = best_psnr

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_psnr': val_psnr,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"  ✅ New best PSNR! Saved checkpoint")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_psnr': val_psnr,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')
            print(f"  💾 Saved periodic checkpoint")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
            print(f"  Best PSNR: {best_psnr:.2f} dB at epoch {history['best_epoch']}")
            break

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Fold {fold_num + 1} complete!")
    print(f"  Best Val PSNR: {best_psnr:.2f} dB")
    print(f"{'='*80}")

    return {
        'fold': fold_num + 1,
        'best_val_psnr': best_psnr,
        'best_epoch': history['best_epoch']
    }


def main():
    parser = argparse.ArgumentParser(description="Simple Restormer 3-Fold CV Training")

    # Data
    parser.add_argument('--data_splits_dir', type=str, required=True, help='Data splits directory')
    parser.add_argument('--resolution', type=int, default=384, help='Input resolution')

    # Model
    parser.add_argument('--dim', type=int, default=48, help='Base dimension')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[4, 6, 6, 8])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--ffn_expansion_factor', type=float, default=2.66)
    parser.add_argument('--bias', action='store_true', help='Use bias in convolutions')
    parser.add_argument('--use_checkpointing', action='store_true', help='Use gradient checkpointing')

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=15)

    # Loss
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_vgg', type=float, default=0.2)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)

    # CV
    parser.add_argument('--n_folds', type=int, default=3)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_simple')
    parser.add_argument('--save_every', type=int, default=10)

    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mixed_precision', action='store_true')

    args = parser.parse_args()

    print("=" * 80)
    print("SIMPLE RESTORMER 3-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"  Resolution: {args.resolution}px")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping: {args.early_stopping_patience} epochs")
    print(f"  Loss weights: L1={args.lambda_l1}, VGG={args.lambda_vgg}, SSIM={args.lambda_ssim}")
    print(f"  CV folds: {args.n_folds}")

    # Train all folds
    fold_results = []
    for fold in range(args.n_folds):
        train_jsonl = Path(args.data_splits_dir) / f'fold_{fold + 1}' / 'train.jsonl'
        val_jsonl = Path(args.data_splits_dir) / f'fold_{fold + 1}' / 'val.jsonl'

        result = train_fold(fold, str(train_jsonl), str(val_jsonl), args)
        fold_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    avg_psnr = np.mean([r['best_val_psnr'] for r in fold_results])
    std_psnr = np.std([r['best_val_psnr'] for r in fold_results])

    print(f"\n📊 Results:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_val_psnr']:.2f} dB (epoch {r['best_epoch']})")

    print(f"\n📈 Overall:")
    print(f"  Mean PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  Min PSNR: {min(r['best_val_psnr'] for r in fold_results):.2f} dB")
    print(f"  Max PSNR: {max(r['best_val_psnr'] for r in fold_results):.2f} dB")

    print(f"\n✅ Training complete!")
    print(f"   Output: {args.output_dir}")


if __name__ == '__main__':
    main()
