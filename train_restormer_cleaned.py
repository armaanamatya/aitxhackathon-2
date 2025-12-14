#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
Train Restormer with cleaned dataset and configurable preprocessing.
MLE-grade training script with experiment tracking.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer
from preprocessing import PreprocessingPipeline, PreprocessConfig, get_preset_config


# ============================================================================
# DATASET
# ============================================================================

class RealEstateDataset(Dataset):
    """Real estate image dataset with preprocessing"""

    def __init__(self, jsonl_path: str, base_dir: str, resolution: int,
                 preprocessing: PreprocessingPipeline = None):
        self.base_dir = base_dir
        self.resolution = resolution
        self.preprocessing = preprocessing

        # Load pairs
        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load images
        src_path = os.path.join(self.base_dir, pair['src'])
        tar_path = os.path.join(self.base_dir, pair['tar'])

        src = cv2.imread(src_path)
        tar = cv2.imread(tar_path)

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        # Resize
        src = cv2.resize(src, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        tar = cv2.resize(tar, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # Apply preprocessing
        if self.preprocessing:
            src, tar = self.preprocessing(src, tar)

        # To tensor [0, 1]
        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    count = 0

    for src, tar in dataloader:
        src, tar = src.to(device), tar.to(device)

        optimizer.zero_grad()

        if scaler:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                out = model(src)
                loss = criterion(out, tar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(src)
            loss = criterion(out, tar)
            loss.backward()
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Restormer with cleaned dataset")

    # Data
    parser.add_argument('--train_jsonl', type=str, default='train_cleaned.jsonl',
                        help='Training data JSONL (cleaned)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Training resolution')

    # Preprocessing
    parser.add_argument('--preprocess', type=str, default='light_aug',
                        choices=['none', 'light_aug', 'standard_aug', 'normalize_exposure',
                                'histogram_match', 'lab_colorspace', 'quality_enhance', 'aggressive'],
                        help='Preprocessing preset')
    parser.add_argument('--custom_preprocess', type=str, default=None,
                        help='Path to custom preprocessing config JSON')

    # Model
    parser.add_argument('--dim', type=int, default=48, help='Model dimension')
    parser.add_argument('--num_blocks', type=int, nargs=4, default=[4, 6, 6, 8],
                        help='Number of blocks per stage')
    parser.add_argument('--ffn_expansion', type=float, default=2.0,
                        help='Expansion factor inside the feed-forward network')
    parser.add_argument('--disable_checkpoint', action='store_true',
                        help='Disable gradient checkpointing (enabled by default)')

    # Training
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled, recommended: 10-15)')

    # Output
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')

    # System
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 70)
    print(f"TRAINING RESTORMER WITH CLEANED DATASET")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Train data: {args.train_jsonl}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Preprocessing: {args.preprocess}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {args.output_dir}")
    print(f"  FFN expansion: {args.ffn_expansion}")
    print(f"  Gradient checkpointing: {'disabled' if args.disable_checkpoint else 'enabled'}")

    # Setup preprocessing
    if args.custom_preprocess:
        with open(args.custom_preprocess) as f:
            config_dict = json.load(f)
        preprocess_config = PreprocessConfig(**config_dict)
    else:
        preprocess_config = get_preset_config(args.preprocess)

    train_pipeline = PreprocessingPipeline(preprocess_config)

    # Save preprocessing config
    with open(output_dir / 'preprocessing_config.json', 'w') as f:
        json.dump(train_pipeline.get_config(), f, indent=2)

    print(f"\n📋 Preprocessing pipeline:")
    for t in train_pipeline.transforms:
        print(f"   - {t.get_config()['name']}")

    # Load dataset
    print(f"\n📂 Loading dataset...")
    full_dataset = RealEstateDataset(
        args.train_jsonl,
        base_dir=os.path.dirname(args.train_jsonl),
        resolution=args.resolution,
        preprocessing=train_pipeline
    )

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    print(f"\n🏗️  Creating model...")
    model = Restormer(
        in_channels=3,
        out_channels=3,
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=args.ffn_expansion,
        bias=False,
        use_checkpointing=not args.disable_checkpoint
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing with warmup
    def warmup_cosine_scheduler(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_scheduler)

    # Loss
    criterion = nn.L1Loss()

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Training loop
    print(f"\n🚀 Starting training...\n")
    if args.early_stopping_patience > 0:
        print(f"⏱️  Early stopping enabled (patience: {args.early_stopping_patience} epochs)\n")

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, scaler)

        # Validate
        val_loss = validate(model, val_loader, criterion, args.device)

        # Scheduler step
        scheduler.step()

        # Log
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

        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
            print(f"   Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            print(f"   No improvement for {epochs_without_improvement} epochs")
            break

        # Save checkpoint
        if epoch % args.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }

            if is_best:
                torch.save(checkpoint, output_dir / 'checkpoint_best.pt')

            if epoch % args.save_every == 0:
                torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final
    torch.save(checkpoint, output_dir / 'checkpoint_final.pt')

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
