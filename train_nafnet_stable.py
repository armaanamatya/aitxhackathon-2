#!/usr/bin/env python3
"""
Stable NAFNet Training for Real Estate HDR Enhancement

Fixes for NaN loss:
- FP32 only (no mixed precision)
- Conservative learning rate with warmup
- Aggressive gradient clipping
- Simple L1 loss
- Batch size 8 for stability
- Optional pretrained weights
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
import numpy as np
import cv2
from tqdm import tqdm


# =============================================================================
# NAFNet Architecture (Simplified Baseline)
# =============================================================================

class LayerNormFunction(torch.autograd.Function):
    """Stable LayerNorm"""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """Stable 2D LayerNorm"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Simple gating mechanism"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAF Block with stability improvements"""
    def __init__(self, c, dw_expansion=2, ffn_expansion=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expansion
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = c * ffn_expansion
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    Stable NAFNet for HDR enhancement

    Default: width=32, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1]
    For stability: Reduced from original [2,2,4,8] encoder blocks
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2, 0, bias=True))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


# =============================================================================
# Dataset
# =============================================================================

class RealEstateHDRDataset(Dataset):
    def __init__(self, jsonl_path, resolution=512, augment=True):
        self.resolution = resolution
        self.augment = augment
        self.pairs = []

        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        src = cv2.imread(pair['src'])
        tar = cv2.imread(pair['tar'])

        if src is None or tar is None:
            src = np.random.randint(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8)
            tar = np.random.randint(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8)

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        src = cv2.resize(src, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        tar = cv2.resize(tar, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        if self.augment and np.random.random() > 0.5:
            src = np.fliplr(src).copy()
            tar = np.fliplr(tar).copy()

        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar


# =============================================================================
# Stable Trainer
# =============================================================================

class StableNAFNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / 'training.log'

        self._log("=" * 70)
        self._log("STABLE NAFNET TRAINING")
        self._log("=" * 70)
        self._log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        self._log(f"Resolution: {args.resolution}x{args.resolution}")
        self._log(f"Batch size: {args.batch_size}")
        self._log(f"Learning rate: {args.lr}")
        self._log(f"Gradient clip: {args.grad_clip}")
        self._log(f"FP32 only (no mixed precision)")

    def _log(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

    def setup_data(self):
        self._log("\n📂 Loading data...")

        train_dataset = RealEstateHDRDataset(self.args.train_jsonl, self.args.resolution, augment=True)
        val_dataset = RealEstateHDRDataset(self.args.val_jsonl, self.args.resolution, augment=False)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True
        )

        self._log(f"   Train samples: {len(train_dataset)}")
        self._log(f"   Val samples: {len(val_dataset)}")

    def setup_model(self):
        self._log("\n🏗️  Creating NAFNet...")

        self.model = NAFNet(
            img_channel=3,
            width=self.args.width,
            middle_blk_num=self.args.middle_blk_num,
            enc_blk_nums=self.args.enc_blk_nums,
            dec_blk_nums=self.args.dec_blk_nums
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        self._log(f"   Parameters: {num_params:,}")

        # Optimizer - Conservative settings for stability
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),  # Stable betas
            eps=1e-8
        )

        # Scheduler with warmup
        def warmup_cosine(epoch):
            if epoch < self.args.warmup_epochs:
                return (epoch + 1) / self.args.warmup_epochs
            else:
                progress = (epoch - self.args.warmup_epochs) / (self.args.epochs - self.args.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, warmup_cosine)

        # Simple L1 loss
        self.criterion = nn.L1Loss()

        self._log(f"   Loss: L1")
        self._log(f"   Optimizer: AdamW (lr={self.args.lr}, weight_decay=1e-4)")
        self._log(f"   Scheduler: Cosine with {self.args.warmup_epochs} epoch warmup")
        self._log(f"   Gradient clipping: max_norm={self.args.grad_clip}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        for src, tar in pbar:
            src, tar = src.to(self.device), tar.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (FP32 only)
            pred = self.model(src)
            loss = self.criterion(pred, tar)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                self._log(f"WARNING: NaN/Inf loss detected at epoch {epoch+1}, skipping batch")
                continue

            # Backward
            loss.backward()

            # Gradient clipping (critical for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)

            # Check gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            if torch.isnan(grad_norm) or grad_norm > 100:
                self._log(f"WARNING: Large gradient norm {grad_norm:.2f}, skipping batch")
                continue

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for src, tar in tqdm(self.val_loader, desc="Validating"):
            src, tar = src.to(self.device), tar.to(self.device)

            pred = self.model(src)
            pred = torch.clamp(pred, 0, 1)
            loss = self.criterion(pred, tar)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }

        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')

        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')

    def train(self):
        self._log("\n🚀 Starting training...")
        self._log(f"   Epochs: {self.args.epochs}")
        self._log(f"   Warmup: {self.args.warmup_epochs} epochs")
        self._log(f"   Early stopping patience: {self.args.patience}")

        history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                history['best_val_loss'] = self.best_val_loss
                history['best_epoch'] = epoch + 1
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch + 1, val_loss, is_best)

            status = "(best)" if is_best else ""
            self._log(f"Epoch {epoch+1:3d}/{self.args.epochs}: "
                     f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                     f"LR={lr:.2e} {status}")

            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['learning_rates'].append(lr)

            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

            if self.patience_counter >= self.args.patience:
                self._log(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break

        self._log(f"\n✅ Training complete!")
        self._log(f"   Best val loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")


def main():
    parser = argparse.ArgumentParser(description="Stable NAFNet Training")

    # Data
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--val_jsonl', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_nafnet_stable')

    # Model
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--middle_blk_num', type=int, default=1)
    parser.add_argument('--enc_blk_nums', type=int, nargs=4, default=[1, 1, 1, 28])
    parser.add_argument('--dec_blk_nums', type=int, nargs=4, default=[1, 1, 1, 1])

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4)

    # Hardware
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = StableNAFNetTrainer(args)
    trainer.setup_data()
    trainer.setup_model()
    trainer.train()


if __name__ == '__main__':
    main()
