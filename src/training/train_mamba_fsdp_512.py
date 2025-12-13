#!/usr/bin/env python3
"""
MambaDiffusion FSDP Training - 512x512
=======================================

Multi-node training with PyTorch FSDP (Fully Sharded Data Parallel).
Shards model parameters across GPUs for 3x effective memory.

Features:
- PyTorch FSDP with full sharding
- Model parameters sharded across all GPUs (3x memory)
- CPU offloading for optimizer states
- Mixed precision (bfloat16/float16)
- Activation checkpointing
"""

import argparse
import functools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mamba_diffusion import mamba_diffusion_large


# =============================================================================
# Loss Functions
# =============================================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel

        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        window = self.window

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class HistogramLoss(nn.Module):
    def __init__(self, bins: int = 64):
        super().__init__()
        self.bins = bins

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for c in range(3):
            pred_hist = torch.histc(pred[:, c], bins=self.bins, min=0, max=1)
            target_hist = torch.histc(target[:, c], bins=self.bins, min=0, max=1)
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            target_hist = target_hist / (target_hist.sum() + 1e-8)
            loss += F.l1_loss(pred_hist, target_hist)
        return loss / 3


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    def __init__(self, data_root: str, jsonl_path: str, split: str = 'train',
                 crop_size: int = 512, augment: bool = True):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.augment = augment and (split == 'train')

        all_samples = []
        jsonl_file = self.data_root / jsonl_path
        with open(jsonl_file, 'r') as f:
            for line in f:
                all_samples.append(json.loads(line.strip()))

        n_val = max(1, int(len(all_samples) * 0.1))
        self.samples = all_samples[-n_val:] if split == 'val' else all_samples[:-n_val]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        source = Image.open(self.data_root / item['src']).convert('RGB')
        target = Image.open(self.data_root / item['tar']).convert('RGB')

        # Ensure minimum size
        min_size = self.crop_size + 32
        if source.width < min_size or source.height < min_size:
            scale = max(min_size / source.width, min_size / source.height)
            new_w, new_h = int(source.width * scale), int(source.height * scale)
            source = source.resize((new_w, new_h), Image.LANCZOS)
            target = target.resize((new_w, new_h), Image.LANCZOS)

        # Random crop
        w, h = source.size
        if w > self.crop_size and h > self.crop_size:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            source = source.crop((x, y, x + self.crop_size, y + self.crop_size))
            target = target.crop((x, y, x + self.crop_size, y + self.crop_size))
        else:
            source = source.resize((self.crop_size, self.crop_size), Image.LANCZOS)
            target = target.resize((self.crop_size, self.crop_size), Image.LANCZOS)

        source = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 255.0
        target = torch.from_numpy(np.array(target)).permute(2, 0, 1).float() / 255.0

        if self.augment:
            if random.random() > 0.5:
                source = torch.flip(source, [2])
                target = torch.flip(target, [2])
            if random.random() > 0.5:
                source = torch.flip(source, [1])
                target = torch.flip(target, [1])

        return {'source': source, 'target': target}


# =============================================================================
# Training
# =============================================================================

def setup_distributed():
    """Initialize distributed training."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group('nccl')

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def get_fsdp_config():
    """Get FSDP configuration for memory-efficient training."""

    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # Auto wrap policy - wrap modules larger than 100M parameters
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1e7,  # 10M params
    )

    return {
        'sharding_strategy': ShardingStrategy.FULL_SHARD,  # Shard everything
        'cpu_offload': CPUOffload(offload_params=True),  # Offload to CPU
        'mixed_precision': mp_policy,
        'auto_wrap_policy': auto_wrap_policy,
        'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
        'device_id': torch.cuda.current_device(),
    }


def save_fsdp_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save FSDP checkpoint (full state dict)."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()

        if dist.get_rank() == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'val_loss': val_loss,
            }, path)


def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"\n{'='*60}")
        print("MambaDiffusion FSDP Training - 512x512")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Sharding: FULL_SHARD (model split across all GPUs)")
        print(f"Effective VRAM: {world_size * 80}GB")
        print(f"Resolution: {args.crop_size}x{args.crop_size}")
        print(f"Batch per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation}")
        print(f"Effective batch: {args.batch_size * world_size * args.gradient_accumulation}")
        print(f"{'='*60}\n")

    # Create model
    model = mamba_diffusion_large()

    # Wrap with FSDP
    fsdp_config = get_fsdp_config()
    model = FSDP(model, **fsdp_config)

    if rank == 0:
        print(f"Model wrapped with FSDP (FULL_SHARD + CPU offload)")

    # Datasets
    train_dataset = HDRDataset(args.data_root, args.jsonl_path, 'train', args.crop_size, True)
    val_dataset = HDRDataset(args.data_root, args.jsonl_path, 'val', args.crop_size, False)

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Loss functions (keep on each GPU)
    char_loss = CharbonnierLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    hist_loss = HistogramLoss().to(device)

    loss_weights = {'charbonnier': 1.0, 'ssim': 0.5, 'histogram': 0.1}

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=(rank != 0))

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            # Forward pass (FSDP handles mixed precision internally)
            output = model(source)

            losses = {
                'charbonnier': char_loss(output, target),
                'ssim': ssim_loss(output, target),
                'histogram': hist_loss(output, target),
            }
            total_loss = sum(loss_weights[k] * losses[k] for k in losses)
            total_loss = total_loss / args.gradient_accumulation

            total_loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * args.gradient_accumulation
            num_batches += 1

            if rank == 0:
                pbar.set_postfix({'loss': f"{total_loss.item() * args.gradient_accumulation:.4f}"})

        scheduler.step()
        epoch_loss /= max(num_batches, 1)

        # Validation (all ranks participate but only rank 0 reports)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                source = batch['source'].to(device)
                target = batch['target'].to(device)
                output = model(source)
                val_loss += F.l1_loss(output, target).item()
        val_loss /= len(val_loader)

        # Sync val_loss across ranks
        val_loss_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            print(f"  Train Loss: {epoch_loss:.4f}")
            print(f"  Val L1: {val_loss:.4f}" + (" (best)" if val_loss < best_val_loss else ""))
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_fsdp_checkpoint(model, optimizer, epoch, val_loss, output_dir / 'checkpoint_best.pt')

        if (epoch + 1) % 20 == 0:
            save_fsdp_checkpoint(model, optimizer, epoch, val_loss, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

        dist.barrier()

    dist.destroy_process_group()

    if rank == 0:
        print(f"\nTraining complete! Best Val L1: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_mamba_512')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
