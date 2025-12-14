#!/usr/bin/env python3
"""
Restormer 512 with Adversarial + Style Training

PhD-Level Approach for Learning HDR Style:
- Adversarial loss forces outputs to match TARGET DISTRIBUTION
- Style loss (Gram matrix) captures color/texture patterns
- Histogram loss ensures exact color distribution matching

This addresses the core problem: L1 regression produces "average" colors,
while adversarial training pushes toward the vibrant HDR style.

Reference: ESRGAN, Real-ESRGAN, pix2pix

Author: Top 0.0001% MLE
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from torchvision import transforms as T

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import create_restormer
from training.adversarial_color_loss import (
    PatchDiscriminator,
    AdversarialStyleLoss,
    discriminator_loss
)


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    def __init__(self, jsonl_path, resolution=512, augment=False):
        self.resolution = resolution
        self.augment = augment

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

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

        if self.augment and torch.rand(1) > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        return TF.to_tensor(input_img), TF.to_tensor(target_img)


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    generator, discriminator,
    dataloader, g_criterion,
    g_optimizer, d_optimizer,
    g_scaler, d_scaler,
    device, grad_clip=1.0,
    train_disc_every=1
):
    generator.train()
    discriminator.train()

    total_g_loss = 0
    total_d_loss = 0
    g_components = {}
    d_components = {}

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # =====================================================================
        # Train Discriminator
        # =====================================================================
        if batch_idx % train_disc_every == 0:
            d_optimizer.zero_grad()

            with autocast():
                # Generate fake images
                with torch.no_grad():
                    fake = generator(inputs)
                    fake = torch.clamp(fake, 0, 1)

                # Discriminator on real and fake
                disc_real = discriminator(targets)
                disc_fake = discriminator(fake.detach())

                d_loss, d_comp = discriminator_loss(disc_real, disc_fake)

            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()

            total_d_loss += d_loss.item()
            for k, v in d_comp.items():
                if k not in d_components:
                    d_components[k] = 0
                d_components[k] += v

        # =====================================================================
        # Train Generator
        # =====================================================================
        g_optimizer.zero_grad()

        with autocast():
            fake = generator(inputs)
            fake = torch.clamp(fake, 0, 1)

            # Discriminator output on fake (for adversarial loss)
            disc_pred = discriminator(fake)

            g_loss, g_comp = g_criterion(fake, targets, disc_pred)

        g_scaler.scale(g_loss).backward()
        g_scaler.unscale_(g_optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
        g_scaler.step(g_optimizer)
        g_scaler.update()

        total_g_loss += g_loss.item()
        for k, v in g_comp.items():
            if k not in g_components:
                g_components[k] = 0
            g_components[k] += v

    n_batches = len(dataloader)
    return (
        total_g_loss / n_batches,
        total_d_loss / (n_batches // train_disc_every + 1),
        {k: v / n_batches for k, v in g_components.items()},
        {k: v / (n_batches // train_disc_every + 1) for k, v in d_components.items()}
    )


@torch.no_grad()
def validate(generator, dataloader, criterion, device):
    generator.eval()
    total_loss = 0
    components = {}

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = generator(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss, comp = criterion(outputs, targets, None)  # No adversarial in val

        total_loss += loss.item()
        for k, v in comp.items():
            if k not in components:
                components[k] = 0
            components[k] += v

    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in components.items()}


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--train_jsonl', default='data_splits/proper_split/train.jsonl')
    parser.add_argument('--val_jsonl', default='data_splits/proper_split/val.jsonl')
    parser.add_argument('--output_dir', default='outputs_restormer_512_adversarial')

    # Model
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--model_size', default='base')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--perceptual_weight', type=float, default=0.1)
    parser.add_argument('--style_weight', type=float, default=0.1)
    parser.add_argument('--histogram_weight', type=float, default=0.1)
    parser.add_argument('--adversarial_weight', type=float, default=0.01)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_disc_every', type=int, default=1)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("RESTORMER 512 - ADVERSARIAL + STYLE TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Resolution: {args.resolution}")
    print()
    print("LOSS COMPONENTS:")
    print(f"  L1: {args.l1_weight}")
    print(f"  Perceptual: {args.perceptual_weight}")
    print(f"  Style (Gram): {args.style_weight}")
    print(f"  Histogram: {args.histogram_weight}")
    print(f"  Adversarial: {args.adversarial_weight}")
    print()

    # Data
    train_dataset = HDRDataset(args.train_jsonl, args.resolution, augment=True)
    val_dataset = HDRDataset(args.val_jsonl, args.resolution, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print()

    # Models
    print("Creating models...")
    generator = create_restormer(args.model_size).to(device)
    discriminator = PatchDiscriminator(in_channels=3, ndf=64).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator params: {g_params:,}")
    print(f"Discriminator params: {d_params:,}")
    print()

    # Loss
    g_criterion = AdversarialStyleLoss(
        l1_weight=args.l1_weight,
        perceptual_weight=args.perceptual_weight,
        style_weight=args.style_weight,
        histogram_weight=args.histogram_weight,
        adversarial_weight=args.adversarial_weight,
    ).to(device)

    # Optimizers
    g_optimizer = optim.AdamW(generator.parameters(), lr=args.g_lr, weight_decay=args.weight_decay)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    # Schedulers
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda)
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda)

    g_scaler = GradScaler()
    d_scaler = GradScaler()

    # Training
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_l1 = float('inf')
    patience_counter = 0
    history = {'train_g_loss': [], 'train_d_loss': [], 'val_loss': [], 'val_l1': []}

    for epoch in range(args.epochs):
        start_time = time.time()

        g_loss, d_loss, g_comp, d_comp = train_epoch(
            generator, discriminator,
            train_loader, g_criterion,
            g_optimizer, d_optimizer,
            g_scaler, d_scaler,
            device, args.grad_clip, args.train_disc_every
        )

        val_loss, val_comp = validate(generator, val_loader, g_criterion, device)

        g_scheduler.step()
        d_scheduler.step()

        history['train_g_loss'].append(g_loss)
        history['train_d_loss'].append(d_loss)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_comp.get('l1', 0))

        epoch_time = time.time() - start_time

        val_l1 = val_comp.get('l1', val_loss)
        is_best = val_l1 < best_val_l1

        if is_best:
            best_val_l1 = val_l1
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_l1': best_val_l1,
                'args': vars(args)
            }, output_dir / 'checkpoint_best.pt')
        else:
            patience_counter += 1

        best_marker = " BEST" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"G: {g_loss:.4f} D: {d_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"L1: {val_comp.get('l1', 0):.4f} | "
              f"Style: {val_comp.get('style', 0):.4f} | "
              f"Hist: {val_comp.get('histogram', 0):.4f} | "
              f"{epoch_time:.1f}s{best_marker}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'val_l1': val_l1,
            'args': vars(args)
        }, output_dir / 'checkpoint_latest.pt')

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 80)
    print(f"Training complete! Best Val L1: {best_val_l1:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
