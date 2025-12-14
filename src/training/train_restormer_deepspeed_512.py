#!/usr/bin/env python3
"""
Restormer DeepSpeed Training - 512x512
=======================================

Multi-GPU training with DeepSpeed ZeRO Stage 2 for memory efficiency.
Enables 512x512 resolution training across 3 GPUs.

Features:
- DeepSpeed ZeRO-2 with CPU offloading
- Activation checkpointing
- Mixed precision (FP16)
- Multi-loss training: Charbonnier, SSIM, LPIPS
- EMA for stable outputs
- Gradient accumulation
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image
from tqdm import tqdm

import deepspeed
from deepspeed import DeepSpeedEngine

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from restormer import create_restormer


# =============================================================================
# Loss Functions
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss - More robust than L1."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """Structural Similarity Loss."""
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


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features

        self.blocks = nn.ModuleList([
            vgg[:4],
            vgg[4:9],
            vgg[9:18],
            vgg[18:27],
        ])

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights = [1.0, 1.0, 1.0, 1.0]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x, y = pred, target

        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            loss += weight * F.l1_loss(x, y)

        return loss


class HistogramLoss(nn.Module):
    """Histogram matching loss."""
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
    """Dataset for HDR image pairs with 512x512 crops."""

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        split: str = 'train',
        crop_size: int = 512,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.augment = augment and (split == 'train')
        self.samples = []

        # Load samples
        all_samples = []
        jsonl_file = self.data_root / jsonl_path
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                all_samples.append(item)

        # Split data (90/10)
        n_total = len(all_samples)
        n_val = max(1, int(n_total * 0.1))

        if split == 'val':
            self.samples = all_samples[-n_val:]
        else:
            self.samples = all_samples[:-n_val]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        source_path = self.data_root / item['src']
        target_path = self.data_root / item['tar']

        source = Image.open(source_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        # Resize to ensure minimum size for 512 crop
        min_size = self.crop_size + 32
        if source.width < min_size or source.height < min_size:
            scale = max(min_size / source.width, min_size / source.height)
            new_w = int(source.width * scale)
            new_h = int(source.height * scale)
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

        # Convert to tensor
        source = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 255.0
        target = torch.from_numpy(np.array(target)).permute(2, 0, 1).float() / 255.0

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                source = torch.flip(source, [2])
                target = torch.flip(target, [2])

            # Random vertical flip
            if random.random() > 0.5:
                source = torch.flip(source, [1])
                target = torch.flip(target, [1])

            # Random 90 degree rotation
            if random.random() > 0.5:
                k = random.randint(1, 3)
                source = torch.rot90(source, k, [1, 2])
                target = torch.rot90(target, k, [1, 2])

        return {'source': source, 'target': target}


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Training
# =============================================================================

def train(args):
    # Initialize distributed
    deepspeed.init_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        print(f"\n{'='*60}")
        print("Restormer DeepSpeed Training - 512x512")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Resolution: {args.crop_size}x{args.crop_size}")
        print(f"Batch per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation}")
        print(f"Effective batch: {args.batch_size * world_size * args.gradient_accumulation}")
        print(f"{'='*60}\n")

    # Create model
    model = create_restormer('base')

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Create datasets
    train_dataset = HDRDataset(
        args.data_root, args.jsonl_path, split='train',
        crop_size=args.crop_size, augment=True
    )
    val_dataset = HDRDataset(
        args.data_root, args.jsonl_path, split='val',
        crop_size=args.crop_size, augment=False
    )

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    # DeepSpeed config
    ds_config = {
        "train_batch_size": args.batch_size * world_size * args.gradient_accumulation,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": 1e-6,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 500,
                "total_num_steps": args.num_epochs * len(train_dataset) // (args.batch_size * world_size * args.gradient_accumulation)
            }
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4
        },
        "wall_clock_breakdown": False
    }

    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=ds_config,
    )

    # Validation dataloader
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Loss functions
    device = model_engine.local_rank
    char_loss = CharbonnierLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    hist_loss = HistogramLoss().to(device)

    # Loss weights
    loss_weights = {
        'charbonnier': 1.0,
        'ssim': 0.5,
        'perceptual': 0.1,
        'histogram': 0.1,
    }

    # EMA
    ema = EMA(model_engine.module, decay=0.999)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.num_epochs):
        model_engine.train()

        epoch_losses = {k: 0.0 for k in loss_weights.keys()}
        epoch_losses['total'] = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
                    disable=(rank != 0))

        for batch in pbar:
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            # Forward pass
            output = model_engine(source)

            # Compute losses
            losses = {}
            losses['charbonnier'] = char_loss(output, target)
            losses['ssim'] = ssim_loss(output, target)
            losses['perceptual'] = perceptual_loss(output, target)
            losses['histogram'] = hist_loss(output, target)

            # Total weighted loss
            total_loss = sum(loss_weights[k] * losses[k] for k in losses)

            # Backward pass
            model_engine.backward(total_loss)
            model_engine.step()

            # Update EMA
            if global_step % 10 == 0:
                ema.update(model_engine.module)

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            epoch_losses['total'] += total_loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'char': f"{losses['charbonnier'].item():.4f}",
                })

        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        # Validation
        if rank == 0:
            model_engine.eval()
            ema.apply_shadow(model_engine.module)

            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    source = batch['source'].to(device)
                    target = batch['target'].to(device)

                    output = model_engine(source)
                    val_loss += F.l1_loss(output, target).item()

            val_loss /= len(val_loader)

            ema.restore(model_engine.module)

            # Log
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            print(f"  Train - Total: {epoch_losses['total']:.4f}, "
                  f"Char: {epoch_losses['charbonnier']:.4f}, "
                  f"SSIM: {epoch_losses['ssim']:.4f}, "
                  f"Perc: {epoch_losses['perceptual']:.4f}")
            print(f"  Val - L1: {val_loss:.4f}" + (" (best)" if val_loss < best_val_loss else ""))

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ema.apply_shadow(model_engine.module)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_engine.module.state_dict(),
                    'val_loss': val_loss,
                }, output_dir / 'checkpoint_best.pt')
                ema.restore(model_engine.module)

            # Save periodic checkpoint
            if (epoch + 1) % 20 == 0:
                model_engine.save_checkpoint(output_dir, tag=f'epoch_{epoch+1}')

    if rank == 0:
        print(f"\nTraining complete!")
        print(f"Best validation L1: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Restormer DeepSpeed Training')
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_restormer_512')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
