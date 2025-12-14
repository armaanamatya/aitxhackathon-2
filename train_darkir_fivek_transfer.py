#!/usr/bin/env python3
"""
DarkIR Training with FiveK Transfer Learning
=============================================
Two-phase training:
1. Pretrain on FiveK (learn general tone mapping)
2. Fine-tune on Real Estate (domain adaptation)
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

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'DarkIR'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))

from archs.DarkIR import DarkIR
from window_aware_loss import get_window_aware_loss


class FiveKDataset(Dataset):
    """FiveK dataset for pretraining"""

    def __init__(self, data_dir: str, resolution: int = 512, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.augment = augment

        # Load pairs
        pairs_file = self.data_dir / 'pairs.jsonl'
        if pairs_file.exists():
            self.pairs = []
            with open(pairs_file) as f:
                for line in f:
                    if line.strip():
                        self.pairs.append(json.loads(line.strip()))
        else:
            raise ValueError(f"pairs.jsonl not found in {data_dir}")

        print(f"FiveK: {len(self.pairs)} pairs loaded")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        input_path = self.data_dir / pair['input']
        target_path = self.data_dir / pair['target']

        input_img = cv2.imread(str(input_path))
        target_img = cv2.imread(str(target_path))

        if input_img is None or target_img is None:
            # Return random tensor if image fails to load
            return torch.rand(3, self.resolution, self.resolution), \
                   torch.rand(3, self.resolution, self.resolution)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Random crop
        h, w = input_img.shape[:2]
        if h >= self.resolution and w >= self.resolution:
            if self.augment:
                top = np.random.randint(0, h - self.resolution + 1)
                left = np.random.randint(0, w - self.resolution + 1)
            else:
                top = (h - self.resolution) // 2
                left = (w - self.resolution) // 2

            input_img = input_img[top:top+self.resolution, left:left+self.resolution]
            target_img = target_img[top:top+self.resolution, left:left+self.resolution]
        else:
            input_img = cv2.resize(input_img, (self.resolution, self.resolution))
            target_img = cv2.resize(target_img, (self.resolution, self.resolution))

        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()

        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor


class RealEstateDataset(Dataset):
    """Real estate dataset for fine-tuning"""

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

        if src is None or tar is None:
            return torch.rand(3, self.resolution, self.resolution), \
                   torch.rand(3, self.resolution, self.resolution)

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        src = cv2.resize(src, (self.resolution, self.resolution))
        tar = cv2.resize(tar, (self.resolution, self.resolution))

        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar


def train_epoch(model, dataloader, optimizer, criterion, device, use_window_loss=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    count = 0

    for src, tar in tqdm(dataloader, desc="Training", leave=False):
        src, tar = src.to(device), tar.to(device)

        optimizer.zero_grad()
        out = model(src)

        if use_window_loss:
            loss = criterion(out, tar, input_img=src)
        else:
            loss = criterion(out, tar)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count


def validate(model, dataloader, criterion, device, use_window_loss=False):
    """Validate"""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for src, tar in dataloader:
            src, tar = src.to(device), tar.to(device)
            out = model(src)

            if use_window_loss:
                loss = criterion(out, tar, input_img=src)
            else:
                loss = criterion(out, tar)

            total_loss += loss.item()
            count += 1

    return total_loss / count


def main():
    parser = argparse.ArgumentParser(description="DarkIR with FiveK Transfer Learning")

    # FiveK data
    parser.add_argument('--fivek_dir', type=str, required=True,
                        help='Path to FiveK dataset')

    # Real estate data
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--val_jsonl', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='.')

    # Common settings
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--width', type=int, default=32)

    # Phase 1: FiveK pretraining
    parser.add_argument('--fivek_epochs', type=int, default=30)
    parser.add_argument('--fivek_batch_size', type=int, default=16)
    parser.add_argument('--fivek_lr', type=float, default=2e-4)

    # Phase 2: Fine-tuning
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_batch_size', type=int, default=8)
    parser.add_argument('--finetune_lr', type=float, default=2e-5)  # 10x lower!
    parser.add_argument('--early_stopping_patience', type=int, default=15)

    # Loss
    parser.add_argument('--use_window_loss', action='store_true',
                        help='Use window-aware loss for fine-tuning')

    # Output
    parser.add_argument('--output_dir', type=str, required=True)

    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create model
    print("=" * 70)
    print("DARKIR WITH FIVEK TRANSFER LEARNING")
    print("=" * 70)

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
    print(f"Model parameters: {num_params:,}")

    # ==================== PHASE 1: FIVEK PRETRAINING ====================
    print("\n" + "=" * 70)
    print("PHASE 1: PRETRAINING ON FIVEK")
    print("=" * 70)

    fivek_dataset = FiveKDataset(args.fivek_dir, args.resolution, augment=True)

    # Split FiveK into train/val
    val_size = int(len(fivek_dataset) * 0.1)
    train_size = len(fivek_dataset) - val_size
    fivek_train, fivek_val = torch.utils.data.random_split(
        fivek_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"FiveK Train: {len(fivek_train)}")
    print(f"FiveK Val: {len(fivek_val)}")

    fivek_train_loader = DataLoader(
        fivek_train, batch_size=args.fivek_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    fivek_val_loader = DataLoader(
        fivek_val, batch_size=args.fivek_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer for Phase 1
    optimizer = optim.AdamW(model.parameters(), lr=args.fivek_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fivek_epochs)
    criterion_l1 = nn.L1Loss()

    best_fivek_val = float('inf')

    for epoch in range(1, args.fivek_epochs + 1):
        train_loss = train_epoch(model, fivek_train_loader, optimizer, criterion_l1, args.device)
        val_loss = validate(model, fivek_val_loader, criterion_l1, args.device)
        scheduler.step()

        is_best = val_loss < best_fivek_val
        if is_best:
            best_fivek_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'checkpoint_fivek_best.pt')

        print(f"[FiveK] Epoch {epoch:3d}/{args.fivek_epochs}: "
              f"Train={train_loss:.4f}, Val={val_loss:.4f} "
              f"{'(best)' if is_best else ''}")

    # Save pretrained checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'fivek_best_val': best_fivek_val,
    }, output_dir / 'checkpoint_fivek_pretrained.pt')

    print(f"\nPhase 1 complete! Best FiveK val loss: {best_fivek_val:.4f}")

    # ==================== PHASE 2: REAL ESTATE FINE-TUNING ====================
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNING ON REAL ESTATE")
    print("=" * 70)

    re_train = RealEstateDataset(args.train_jsonl, args.base_dir, args.resolution)
    re_val = RealEstateDataset(args.val_jsonl, args.base_dir, args.resolution)

    print(f"Real Estate Train: {len(re_train)}")
    print(f"Real Estate Val: {len(re_val)}")

    re_train_loader = DataLoader(
        re_train, batch_size=args.finetune_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    re_val_loader = DataLoader(
        re_val, batch_size=args.finetune_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer for Phase 2 (LOWER LR!)
    optimizer = optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    # Loss for fine-tuning
    if args.use_window_loss:
        criterion = get_window_aware_loss('default', device=args.device)
        use_window = True
        print("Using Window-Aware Loss for fine-tuning")
    else:
        criterion = nn.L1Loss()
        use_window = False

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = train_epoch(model, re_train_loader, optimizer, criterion,
                                  args.device, use_window_loss=use_window)
        val_loss = validate(model, re_val_loader, criterion,
                            args.device, use_window_loss=use_window)
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'checkpoint_best.pt')
        else:
            epochs_without_improvement += 1

        print(f"[RE] Epoch {epoch:3d}/{args.finetune_epochs}: "
              f"Train={train_loss:.4f}, Val={val_loss:.4f} "
              f"{'(best)' if is_best else ''}")

        # Early stopping
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final history
    history = {
        'fivek_best_val': best_fivek_val,
        'finetune_best_val': best_val_loss,
        'finetune_best_epoch': best_epoch,
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRANSFER LEARNING COMPLETE!")
    print("=" * 70)
    print(f"FiveK pretrain best val: {best_fivek_val:.4f}")
    print(f"Fine-tune best val: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
