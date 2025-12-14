#!/usr/bin/env python3
"""
Restormer DDP Training - Multi-GPU with Distributed Data Parallel

Usage:
    torchrun --nproc_per_node=2 train_restormer_ddp.py --resolution 3296 --batch_size 4

For 2x H200 (282GB total):
    - 7MP (3296x2192) without checkpointing
    - batch_size=4 (2 per GPU)
    - torch.compile enabled
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from torchvision import transforms as T

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import Restormer


# =============================================================================
# DDP Setup
# =============================================================================

def setup_ddp():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return dist.get_rank() == 0


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    """HDR Real Estate Dataset with proper aspect ratio"""

    def __init__(self, jsonl_path, resolution=3296, height=None, augment=False):
        self.resolution = resolution
        if height is None:
            self.height = ((resolution * 2201 // 3297) // 16) * 16
        else:
            self.height = height
        self.augment = augment

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        input_img = TF.resize(input_img, (self.height, self.resolution),
                              interpolation=T.InterpolationMode.BILINEAR)
        target_img = TF.resize(target_img, (self.height, self.resolution),
                               interpolation=T.InterpolationMode.BILINEAR)

        if self.augment and torch.rand(1) > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        return TF.to_tensor(input_img), TF.to_tensor(target_img)


# =============================================================================
# Loss Functions
# =============================================================================

class WindowAwareLoss(nn.Module):
    def __init__(self, brightness_threshold=0.7, window_weight=2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.window_weight = window_weight

    def forward(self, pred, target):
        luminance = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
        window_mask = (luminance > self.brightness_threshold).float()
        window_mask = F.avg_pool2d(window_mask, kernel_size=15, stride=1, padding=7)
        window_mask = torch.clamp(window_mask * 2, 0, 1)
        weight_mask = 1.0 + (self.window_weight - 1.0) * window_mask
        pixel_loss = torch.abs(pred - target)
        return (pixel_loss * weight_mask).mean()


class BrightRegionSaturationLoss(nn.Module):
    def __init__(self, brightness_threshold=0.5, saturation_weight=2.0):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.saturation_weight = saturation_weight

    def rgb_to_hsv(self, rgb):
        max_rgb, _ = torch.max(rgb, dim=1, keepdim=True)
        min_rgb, _ = torch.min(rgb, dim=1, keepdim=True)
        diff = max_rgb - min_rgb + 1e-7
        v = max_rgb
        s = diff / (max_rgb + 1e-7)
        return s, v

    def forward(self, pred, target):
        pred_sat, _ = self.rgb_to_hsv(pred)
        target_sat, target_val = self.rgb_to_hsv(target)
        bright_mask = (target_val > self.brightness_threshold).float()
        bright_mask = F.avg_pool2d(bright_mask, kernel_size=7, stride=1, padding=3)
        sat_diff = torch.abs(pred_sat - target_sat)
        bright_sat_loss = (sat_diff * bright_mask).sum() / (bright_mask.sum() + 1e-7)
        global_sat_loss = F.l1_loss(pred_sat, target_sat)
        return self.saturation_weight * bright_sat_loss + 0.5 * global_sat_loss


class CombinedLoss(nn.Module):
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
        return total, {'l1': l1.item(), 'window': window.item(), 'saturation': saturation.item()}


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs, is_main):
    model.train()
    total_loss = 0
    loss_components = {'l1': 0, 'window': 0, 'saturation': 0}
    n_batches = len(dataloader)

    if is_main:
        pbar = tqdm(enumerate(dataloader), total=n_batches,
                    desc=f"Epoch {epoch+1}/{total_epochs}", ncols=100)
    else:
        pbar = enumerate(dataloader)

    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, components = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v

        if is_main and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'L1': f"{components['l1']:.4f}"})

    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    loss_components = {'l1': 0, 'window': 0, 'saturation': 0}

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast('cuda'):
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, components = criterion(outputs, targets)

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


def main():
    parser = argparse.ArgumentParser(description='Restormer DDP Training')
    parser.add_argument('--train_jsonl', type=str, default='data_splits/proper_split/train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='data_splits/proper_split/val.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_ddp')
    parser.add_argument('--resolution', type=int, default=3296)
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--batch_size', type=int, default=2, help='Total batch size across all GPUs')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_checkpointing', action='store_true')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile()')
    args = parser.parse_args()

    # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    is_main = is_main_process()

    if is_main:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        print("=" * 80)
        print("RESTORMER DDP TRAINING - MULTI-GPU")
        print("=" * 80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPUs: {world_size}")
        print(f"Resolution: {args.resolution} x {((args.resolution * 2201 // 3297) // 16) * 16}")
        print(f"Total batch size: {args.batch_size} ({args.batch_size // world_size} per GPU)")
        print()

    # Dataset with DistributedSampler
    train_dataset = HDRDataset(args.train_jsonl, args.resolution, augment=True)
    val_dataset = HDRDataset(args.val_jsonl, args.resolution, augment=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    batch_size_per_gpu = args.batch_size // world_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu,
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu,
                            sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    if is_main:
        print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches/GPU")
        print(f"Val: {len(val_dataset)} samples")
        print()

    # Model
    configs = {
        "tiny": {"dim": 24, "num_blocks": [2, 3, 3, 4], "num_refinement_blocks": 2, "heads": [1, 2, 4, 8]},
        "small": {"dim": 32, "num_blocks": [3, 4, 4, 6], "num_refinement_blocks": 3, "heads": [1, 2, 4, 8]},
        "base": {"dim": 48, "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4, "heads": [1, 2, 4, 8]},
        "large": {"dim": 64, "num_blocks": [6, 8, 8, 12], "num_refinement_blocks": 6, "heads": [1, 2, 4, 8]},
    }
    model = Restormer(**configs[args.model_size], use_checkpointing=args.use_checkpointing).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
        if args.use_checkpointing:
            print("Gradient checkpointing: ENABLED")

    # Compile model (works on H100/H200)
    if args.compile:
        if is_main:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        if is_main:
            print("Model compiled: ENABLED")

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    if is_main:
        print()

    # Loss, optimizer, scheduler
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    # Training loop
    if is_main:
        print("=" * 80)
        print("TRAINING")
        print("=" * 80)
        print(f"Starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total epochs: {args.epochs}, Batches per epoch: {len(train_loader)}")
        print("-" * 80)

    best_val_l1 = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        start_time = time.time()

        train_loss, train_comp = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, args.epochs, is_main
        )

        val_loss, val_comp = validate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - start_time

        # Sync best loss across processes
        val_l1_tensor = torch.tensor([val_comp['l1']], device=device)
        dist.all_reduce(val_l1_tensor, op=dist.ReduceOp.AVG)
        val_l1 = val_l1_tensor.item()

        is_best = val_l1 < best_val_l1

        # Update best_val_l1 on all ranks for proper early stopping
        if is_best:
            best_val_l1 = val_l1
            patience_counter = 0
        else:
            patience_counter += 1

        if is_main:
            if is_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'val_l1': best_val_l1,
                    'args': vars(args)
                }, Path(args.output_dir) / 'checkpoint_best.pt')

            marker = "* BEST" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train: {train_loss:.4f} | Val L1: {val_l1:.4f} | "
                  f"Time: {epoch_time:.1f}s {marker}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'val_l1': val_l1,
                'args': vars(args)
            }, Path(args.output_dir) / 'checkpoint_latest.pt')

        # Early stopping - all ranks must break together
        if patience_counter >= args.patience:
            if is_main:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if is_main:
        print()
        print("=" * 80)
        print(f"Training complete! Best Val L1: {best_val_l1:.4f}")
        print("=" * 80)

    cleanup_ddp()


if __name__ == '__main__':
    main()
