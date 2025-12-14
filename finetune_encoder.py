#!/usr/bin/env python3
"""
Encoder Finetuning Script for Restormer

Loads best weights from initial training and finetunes ONLY the encoder.
Decoder weights are frozen to preserve learned reconstruction capabilities.

Usage:
    python finetune_encoder.py \
        --checkpoint outputs_restormer_combined/checkpoint_best.pt \
        --train_jsonl data_splits/proper_split/train.jsonl \
        --val_jsonl data_splits/proper_split/val.jsonl \
        --resolution 3197 \
        --output_dir outputs_encoder_finetuned
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
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from torchvision import transforms as T

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import create_restormer


# =============================================================================
# Dataset (same as main training)
# =============================================================================

class HDRDataset(Dataset):
    """HDR Real Estate Dataset"""

    def __init__(self, jsonl_path, resolution=512, augment=False):
        self.resolution = resolution
        self.augment = augment

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize to target resolution
        input_img = TF.resize(input_img, (self.resolution, int(self.resolution * 2201 / 3197)),
                              interpolation=T.InterpolationMode.BILINEAR)
        target_img = TF.resize(target_img, (self.resolution, int(self.resolution * 2201 / 3197)),
                               interpolation=T.InterpolationMode.BILINEAR)

        # Light augmentation
        if self.augment and torch.rand(1) > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor


# =============================================================================
# Loss Functions (same as main training)
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
        weighted_loss = (pixel_loss * weight_mask).mean()
        return weighted_loss


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
        pred_sat, pred_val = self.rgb_to_hsv(pred)
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
        return total, {
            'l1': l1.item(),
            'window': window.item(),
            'saturation': saturation.item(),
            'total': total.item()
        }


# =============================================================================
# Encoder Freezing Utility
# =============================================================================

def freeze_decoder(model):
    """
    Freeze decoder layers, keep encoder trainable.

    Encoder layers (TRAINABLE):
        - patch_embed
        - encoder_level1, down1_2
        - encoder_level2, down2_3
        - encoder_level3, down3_4
        - latent (bottleneck)

    Decoder layers (FROZEN):
        - up4_3, reduce_chan_level3, decoder_level3
        - up3_2, reduce_chan_level2, decoder_level2
        - up2_1, reduce_chan_level1, decoder_level1
        - refinement
        - output
    """
    decoder_layers = [
        'up4_3', 'reduce_chan_level3', 'decoder_level3',
        'up3_2', 'reduce_chan_level2', 'decoder_level2',
        'up2_1', 'reduce_chan_level1', 'decoder_level1',
        'refinement', 'output'
    ]

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        # Check if this parameter belongs to decoder
        is_decoder = any(layer in name for layer in decoder_layers)

        if is_decoder:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()

    print(f"Encoder (trainable): {trainable_count:,} parameters")
    print(f"Decoder (frozen): {frozen_count:,} parameters")

    return model


def freeze_encoder(model):
    """
    Freeze encoder layers, keep decoder trainable.
    Opposite of freeze_decoder - use this if you want to finetune decoder instead.
    """
    encoder_layers = [
        'patch_embed',
        'encoder_level1', 'down1_2',
        'encoder_level2', 'down2_3',
        'encoder_level3', 'down3_4',
        'latent'
    ]

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        is_encoder = any(layer in name for layer in encoder_layers)

        if is_encoder:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()

    print(f"Encoder (frozen): {frozen_count:,} parameters")
    print(f"Decoder (trainable): {trainable_count:,} parameters")

    return model


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    loss_components = {'l1': 0, 'window': 0, 'saturation': 0}

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, components = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in components.items():
            if k != 'total':
                loss_components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    loss_components = {'l1': 0, 'window': 0, 'saturation': 0}

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, components = criterion(outputs, targets)

        total_loss += loss.item()
        for k, v in components.items():
            if k != 'total':
                loss_components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


def main():
    parser = argparse.ArgumentParser(description='Restormer Encoder Finetuning')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best checkpoint from initial training')

    # Data paths
    parser.add_argument('--train_jsonl', type=str, default='data_splits/proper_split/train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='data_splits/proper_split/val.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_encoder_finetuned')

    # Model
    parser.add_argument('--resolution', type=int, default=3197,
                        help='Training resolution (width). Height auto-calculated as 3197:2201 ratio')
    parser.add_argument('--model_size', type=str, default='base')

    # Finetuning mode
    parser.add_argument('--finetune_mode', type=str, default='encoder',
                        choices=['encoder', 'decoder'],
                        help='Which part to finetune (encoder=freeze decoder, decoder=freeze encoder)')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--window_weight', type=float, default=0.5)
    parser.add_argument('--color_weight', type=float, default=0.3)

    # Training (lower LR for finetuning)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5, help='Lower LR for finetuning')
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("RESTORMER ENCODER FINETUNING")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Resolution: {args.resolution} x {int(args.resolution * 2201 / 3197)}")
    print(f"Finetune mode: {args.finetune_mode}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Data
    print("Loading data...")
    train_dataset = HDRDataset(args.train_jsonl, args.resolution, augment=True)
    val_dataset = HDRDataset(args.val_jsonl, args.resolution, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print()

    # Model
    print("Loading model from checkpoint...")
    model = create_restormer(args.model_size).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded weights from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Previous val L1: {checkpoint.get('val_l1', 'unknown')}")
    print()

    # Freeze appropriate layers
    print(f"Freezing {'decoder' if args.finetune_mode == 'encoder' else 'encoder'}...")
    if args.finetune_mode == 'encoder':
        model = freeze_decoder(model)
    else:
        model = freeze_encoder(model)
    print()

    # Loss
    criterion = CombinedLoss(args.l1_weight, args.window_weight, args.color_weight)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer tracking {len(trainable_params)} parameter groups")

    # Scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Training loop
    print("=" * 80)
    print("FINETUNING")
    print("=" * 80)

    best_val_loss = float('inf')
    best_val_l1 = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_l1': [], 'lr': []}

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.grad_clip
        )

        val_loss, val_components = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_components['l1'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - start_time

        is_best = val_components['l1'] < best_val_l1

        if is_best:
            best_val_loss = val_loss
            best_val_l1 = val_components['l1']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_l1': best_val_l1,
                'finetune_mode': args.finetune_mode,
                'args': vars(args)
            }, output_dir / 'checkpoint_best.pt')
        else:
            patience_counter += 1

        best_marker = "* BEST" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"L1: {val_components['l1']:.4f} | LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s {best_marker}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_l1': val_components['l1'],
            'finetune_mode': args.finetune_mode,
            'args': vars(args)
        }, output_dir / 'checkpoint_latest.pt')

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 80)
    print(f"Finetuning complete!")
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
