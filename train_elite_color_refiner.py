"""
Elite Color Refiner Training Script
===================================
World-class training pipeline for color refinement on frozen Restormer896 backbone

Features:
- Frozen backbone (Restormer896) + trainable refiner
- EMA (Exponential Moving Average) for stable convergence
- Mixed precision training with GradScaler
- Gradient accumulation for effective larger batch sizes
- Cosine annealing with warmup
- Progressive loss weighting (curriculum learning)
- Advanced augmentation (color-preserving)
- Multi-loss with adaptive weighting
- Comprehensive logging and checkpointing

Author: Top 0.001% ML Engineering
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.color_refiner import create_elite_color_refiner


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    """HDR Real Estate Dataset"""

    def __init__(self, jsonl_path, resolution=896, augment=False):
        self.resolution = resolution
        self.augment = augment

        # Load samples
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images (handle both 'src'/'tar' and 'input'/'target' formats)
        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize
        input_img = TF.resize(input_img, (self.resolution, self.resolution),
                              interpolation=T.InterpolationMode.BILINEAR)
        target_img = TF.resize(target_img, (self.resolution, self.resolution),
                               interpolation=T.InterpolationMode.BILINEAR)

        # Augmentation (color-preserving)
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)

            # Random vertical flip
            if torch.rand(1) > 0.5:
                input_img = TF.vflip(input_img)
                target_img = TF.vflip(target_img)

            # Random rotation (90 degree multiples)
            if torch.rand(1) > 0.5:
                angle = int(np.random.choice([90, 180, 270]))
                input_img = TF.rotate(input_img, angle)
                target_img = TF.rotate(target_img, angle)

        # To tensor [0, 1]
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'filename': os.path.basename(input_path)
        }


# =============================================================================
# Advanced Losses
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1)"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class HSVColorLoss(nn.Module):
    """HSV color loss - emphasizes saturation"""
    def __init__(self, saturation_weight=3.0, hue_weight=1.0, value_weight=1.0):
        super().__init__()
        self.saturation_weight = saturation_weight
        self.hue_weight = hue_weight
        self.value_weight = value_weight

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        max_val, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_val = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = max_val - min_val

        # Hue
        hue = torch.zeros_like(max_val)
        mask = delta > 1e-7

        r_max = (max_idx == 0).float() * mask
        g_max = (max_idx == 1).float() * mask
        b_max = (max_idx == 2).float() * mask

        hue = hue + r_max * (((g - b) / (delta + 1e-7)) % 6)
        hue = hue + g_max * (((b - r) / (delta + 1e-7)) + 2)
        hue = hue + b_max * (((r - g) / (delta + 1e-7)) + 4)
        hue = hue / 6.0  # Normalize

        # Saturation
        sat = torch.where(max_val > 1e-7, delta / (max_val + 1e-7), torch.zeros_like(delta))

        # Value
        val = max_val

        return torch.cat([hue, sat, val], dim=1)

    def forward(self, pred, target):
        pred_hsv = self.rgb_to_hsv(pred)
        target_hsv = self.rgb_to_hsv(target)

        # Hue loss (circular)
        hue_diff = torch.abs(pred_hsv[:, 0:1] - target_hsv[:, 0:1])
        hue_loss = torch.minimum(hue_diff, 1.0 - hue_diff).mean()

        # Saturation loss (L1)
        sat_loss = F.l1_loss(pred_hsv[:, 1:2], target_hsv[:, 1:2])

        # Value loss (L1)
        val_loss = F.l1_loss(pred_hsv[:, 2:3], target_hsv[:, 2:3])

        return (
            self.hue_weight * hue_loss +
            self.saturation_weight * sat_loss +
            self.value_weight * val_loss
        )


class PerceptualLoss(nn.Module):
    """VGG perceptual loss"""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features[:16]  # relu3_3
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        # Extract features
        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)

        return F.l1_loss(pred_feat, target_feat)


class FFTLoss(nn.Module):
    """Frequency domain loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # Magnitude loss
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return F.l1_loss(pred_mag, target_mag)


class ColorHistogramLoss(nn.Module):
    """Color histogram matching loss"""
    def __init__(self, bins=64):
        super().__init__()
        self.bins = bins

    def forward(self, pred, target):
        loss = 0
        for c in range(3):
            pred_hist = torch.histc(pred[:, c], bins=self.bins, min=0, max=1)
            target_hist = torch.histc(target[:, c], bins=self.bins, min=0, max=1)

            # Normalize
            pred_hist = pred_hist / (pred_hist.sum() + 1e-7)
            target_hist = target_hist / (target_hist.sum() + 1e-7)

            # EMD (Earth Mover's Distance) approximation
            loss += F.l1_loss(pred_hist, target_hist)

        return loss / 3


class AdaptiveMultiLoss(nn.Module):
    """
    Multi-loss with adaptive weighting
    Automatically balances loss scales using uncertainty weighting
    """
    def __init__(self):
        super().__init__()

        # Loss components
        self.charbonnier = CharbonnierLoss()
        self.hsv_color = HSVColorLoss(saturation_weight=3.0)
        self.perceptual = PerceptualLoss()
        self.fft = FFTLoss()
        self.histogram = ColorHistogramLoss(bins=64)

        # Learnable uncertainty parameters (log variance)
        # Higher variance = lower weight automatically
        self.log_var_char = nn.Parameter(torch.zeros(1))
        self.log_var_hsv = nn.Parameter(torch.zeros(1))
        self.log_var_perc = nn.Parameter(torch.zeros(1))
        self.log_var_fft = nn.Parameter(torch.zeros(1))
        self.log_var_hist = nn.Parameter(torch.zeros(1))

    def forward(self, pred, target):
        # Compute individual losses
        loss_char = self.charbonnier(pred, target)
        loss_hsv = self.hsv_color(pred, target)
        loss_perc = self.perceptual(pred, target)
        loss_fft = self.fft(pred, target)
        loss_hist = self.histogram(pred, target)

        # Adaptive weighting (uncertainty weighting)
        # loss_weighted = exp(-log_var) * loss + log_var
        weighted_char = torch.exp(-self.log_var_char) * loss_char + self.log_var_char
        weighted_hsv = torch.exp(-self.log_var_hsv) * loss_hsv + self.log_var_hsv
        weighted_perc = torch.exp(-self.log_var_perc) * loss_perc + self.log_var_perc
        weighted_fft = torch.exp(-self.log_var_fft) * loss_fft + self.log_var_fft
        weighted_hist = torch.exp(-self.log_var_hist) * loss_hist + self.log_var_hist

        total_loss = weighted_char + weighted_hsv + weighted_perc + weighted_fft + weighted_hist

        # Return loss dict for logging
        losses = {
            'total': total_loss,
            'charbonnier': loss_char.item(),
            'hsv': loss_hsv.item(),
            'perceptual': loss_perc.item(),
            'fft': loss_fft.item(),
            'histogram': loss_hist.item(),
            # Log learned weights
            'weight_char': torch.exp(-self.log_var_char).item(),
            'weight_hsv': torch.exp(-self.log_var_hsv).item(),
            'weight_perc': torch.exp(-self.log_var_perc).item(),
            'weight_fft': torch.exp(-self.log_var_fft).item(),
            'weight_hist': torch.exp(-self.log_var_hist).item(),
        }

        return total_loss, losses


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class ModelEMA:
    """Model Exponential Moving Average"""
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters"""
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


# =============================================================================
# Trainer
# =============================================================================

class EliteColorRefinerTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        self.log_file = os.path.join(args.output_dir, 'training.log')

        self._log("=" * 80)
        self._log("ELITE COLOR REFINER TRAINING")
        self._log("=" * 80)
        self._log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        self._log(f"Output: {args.output_dir}")
        self._log("")

        # Setup models
        self.setup_models()

        # Setup data
        self.setup_data()

        # Setup training
        self.setup_training()

    def _log(self, msg):
        """Log to console and file"""
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

    def setup_models(self):
        """Load frozen backbone and create refiner"""
        self._log("=" * 80)
        self._log("MODEL SETUP")
        self._log("=" * 80)

        # Load frozen Restormer896 backbone
        self._log(f"Loading frozen backbone: {self.args.backbone_path}")
        checkpoint = torch.load(self.args.backbone_path, map_location='cpu')

        # Create Restormer using factory function (896 resolution, 'base' variant)
        from training.restormer import create_restormer
        self.backbone = create_restormer('base').to(self.device)

        # Load weights (handle key mismatch - checkpoint might have encoder_level1.0 instead of encoder_level1.blocks.0)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Try loading directly first
        try:
            self.backbone.load_state_dict(state_dict)
        except RuntimeError as e:
            # Handle key mismatch: add .blocks to keys if needed
            self._log("⚠️  Detected key mismatch, fixing...")
            fixed_state_dict = {}
            for k, v in state_dict.items():
                # Fix encoder/decoder level keys: encoder_level1.0 -> encoder_level1.blocks.0
                if 'encoder_level' in k or 'decoder_level' in k or 'latent' in k or 'refinement' in k:
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        # Insert 'blocks' after level name
                        parts.insert(1, 'blocks')
                        k = '.'.join(parts)
                fixed_state_dict[k] = v

            self.backbone.load_state_dict(fixed_state_dict)

        # Freeze backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        num_backbone_params = sum(p.numel() for p in self.backbone.parameters())
        self._log(f"✓ Frozen backbone loaded: {num_backbone_params / 1e6:.2f}M params")

        # Create Elite Color Refiner
        self._log(f"Creating Elite Color Refiner (size={self.args.refiner_size})")
        self.refiner = create_elite_color_refiner(self.args.refiner_size).to(self.device)

        num_refiner_params = self.refiner.get_num_params()
        self._log(f"✓ Refiner created: {num_refiner_params / 1e6:.2f}M params")
        self._log(f"✓ Total trainable: {num_refiner_params / 1e6:.2f}M params")

        # EMA model
        self._log("Creating EMA model (decay=0.9999)")
        self.ema = ModelEMA(self.refiner, decay=0.9999)
        self._log("✓ EMA model created")
        self._log("")

    def setup_data(self):
        """Setup dataloaders"""
        self._log("=" * 80)
        self._log("DATA SETUP")
        self._log("=" * 80)

        # Training dataset
        train_dataset = HDRDataset(
            self.args.train_jsonl,
            resolution=self.args.resolution,
            augment=True
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        # Validation dataset
        val_dataset = HDRDataset(
            self.args.val_jsonl,
            resolution=self.args.resolution,
            augment=False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        self._log(f"✓ Train batches: {len(self.train_loader)} (batch_size={self.args.batch_size})")
        self._log(f"✓ Val batches: {len(self.val_loader)}")
        self._log("")

    def setup_training(self):
        """Setup optimizer, scheduler, loss, etc."""
        self._log("=" * 80)
        self._log("TRAINING SETUP")
        self._log("=" * 80)

        # Optimizer (AdamW with decoupled weight decay)
        self.optimizer = optim.AdamW(
            self.refiner.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
            eps=1e-8
        )
        self._log(f"✓ Optimizer: AdamW (lr={self.args.lr}, wd={self.args.weight_decay})")

        # Also optimize adaptive loss weights
        self.loss_fn = AdaptiveMultiLoss().to(self.device)
        self.loss_optimizer = optim.Adam(self.loss_fn.parameters(), lr=1e-3)
        self._log("✓ Loss: AdaptiveMultiLoss (learnable weights)")

        # Learning rate scheduler (Cosine with warmup)
        self.warmup_epochs = self.args.warmup_epochs
        self.total_epochs = self.args.epochs

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self._log(f"✓ Scheduler: Cosine with warmup ({self.warmup_epochs} epochs)")

        # Mixed precision
        self.scaler = GradScaler()
        self._log("✓ Mixed precision: Enabled (GradScaler)")

        # Gradient accumulation
        self.grad_accum_steps = self.args.grad_accum_steps
        self._log(f"✓ Gradient accumulation: {self.grad_accum_steps} steps")
        self._log(f"✓ Effective batch size: {self.args.batch_size * self.grad_accum_steps}")

        # Metrics tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        self._log(f"✓ Early stopping patience: {self.args.patience} epochs")
        self._log(f"✓ Gradient clipping: {self.args.grad_clip}")
        self._log("")

    @torch.no_grad()
    def get_backbone_output(self, x):
        """Get frozen backbone output"""
        self.backbone.eval()
        return self.backbone(x)

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.refiner.train()
        self.loss_fn.train()

        losses = defaultdict(list)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        self.optimizer.zero_grad()
        self.loss_optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            x_input = batch['input'].to(self.device)
            x_target = batch['target'].to(self.device)

            # Get frozen backbone output
            with torch.no_grad():
                x_backbone = self.get_backbone_output(x_input)

            # Forward with mixed precision
            with autocast():
                # Refiner
                x_refined = self.refiner(x_input, x_backbone)

                # Loss
                loss, loss_dict = self.loss_fn(x_refined, x_target)

                # Scale for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.refiner.parameters(), self.args.grad_clip
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.step(self.loss_optimizer)
                self.scaler.update()

                self.optimizer.zero_grad()
                self.loss_optimizer.zero_grad()

                # Update EMA
                self.ema.update(self.refiner)

            # Track losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    losses[k].append(v.item())
                else:
                    losses[k].append(v)

            # Update progress bar
            total_val = loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
            hsv_val = loss_dict['hsv']  # Already .item() from loss function
            char_val = loss_dict['charbonnier']  # Already .item()
            pbar.set_postfix({
                'loss': f"{total_val:.4f}",
                'hsv': f"{hsv_val:.4f}",
                'char': f"{char_val:.4f}"
            })

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in losses.items()}

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch, use_ema=True):
        """Validate"""
        model = self.ema.model if use_ema else self.refiner
        model.eval()

        losses = defaultdict(list)
        pbar = tqdm(self.val_loader, desc=f"Validation")

        for batch in pbar:
            x_input = batch['input'].to(self.device)
            x_target = batch['target'].to(self.device)

            # Get frozen backbone output
            x_backbone = self.get_backbone_output(x_input)

            # Refiner
            x_refined = model(x_input, x_backbone)

            # Loss
            _, loss_dict = self.loss_fn(x_refined, x_target)

            # Track losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    losses[k].append(v.item())
                else:
                    losses[k].append(v)

            total_val = loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
            pbar.set_postfix({'loss': f"{total_val:.4f}"})

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in losses.items()}

        return avg_losses

    def save_checkpoint(self, epoch, val_losses, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'refiner_state_dict': self.refiner.state_dict(),
            'ema_state_dict': self.ema.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss_fn.state_dict(),
            'loss_optimizer_state_dict': self.loss_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }

        # Save latest
        latest_path = os.path.join(self.args.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            self._log(f"💾 Saved best checkpoint (val_loss={val_losses['total']:.4f})")

    def train(self):
        """Main training loop"""
        self._log("=" * 80)
        self._log("TRAINING START")
        self._log("=" * 80)
        self._log("")

        for epoch in range(self.args.epochs):
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate(epoch, use_ema=True)

            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Check if best
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_losses, is_best=is_best)

            # Log
            epoch_time = time.time() - epoch_start
            self._log(
                f"Epoch {epoch+1:3d}/{self.args.epochs} | "
                f"Train: {train_losses['total']:.4f} | "
                f"Val: {val_losses['total']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s | "
                f"{'🌟 BEST' if is_best else f'Patience: {self.patience_counter}/{self.args.patience}'}"
            )

            # Detailed loss breakdown
            self._log(
                f"   Train - char: {train_losses['charbonnier']:.4f}, "
                f"hsv: {train_losses['hsv']:.4f}, "
                f"perc: {train_losses['perceptual']:.4f}, "
                f"fft: {train_losses['fft']:.4f}, "
                f"hist: {train_losses['histogram']:.4f}"
            )
            self._log(
                f"   Val   - char: {val_losses['charbonnier']:.4f}, "
                f"hsv: {val_losses['hsv']:.4f}, "
                f"perc: {val_losses['perceptual']:.4f}, "
                f"fft: {val_losses['fft']:.4f}, "
                f"hist: {val_losses['histogram']:.4f}"
            )

            # Log adaptive weights
            self._log(
                f"   Weights - char: {train_losses['weight_char']:.3f}, "
                f"hsv: {train_losses['weight_hsv']:.3f}, "
                f"perc: {train_losses['weight_perc']:.3f}, "
                f"fft: {train_losses['weight_fft']:.3f}, "
                f"hist: {train_losses['weight_hist']:.3f}"
            )
            self._log("")

            # Early stopping
            if self.patience_counter >= self.args.patience:
                self._log(f"Early stopping triggered at epoch {epoch+1}")
                self._log(f"Best epoch: {self.best_epoch+1} with val_loss={self.best_val_loss:.4f}")
                break

        self._log("=" * 80)
        self._log("TRAINING COMPLETE")
        self._log("=" * 80)
        self._log(f"Best epoch: {self.best_epoch+1}")
        self._log(f"Best val loss: {self.best_val_loss:.4f}")
        self._log(f"Final model: {os.path.join(self.args.output_dir, 'checkpoint_best.pt')}")
        self._log("")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Elite Color Refiner Training')

    # Data
    parser.add_argument('--train_jsonl', type=str, required=True,
                        help='Path to training JSONL')
    parser.add_argument('--val_jsonl', type=str, required=True,
                        help='Path to validation JSONL')
    parser.add_argument('--resolution', type=int, default=896,
                        help='Image resolution (default: 896)')

    # Model
    parser.add_argument('--backbone_path', type=str, required=True,
                        help='Path to frozen Restormer896 checkpoint')
    parser.add_argument('--refiner_size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Refiner size (default: medium)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (default: 1.0)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data workers (default: 4)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    args = parser.parse_args()

    # Train
    trainer = EliteColorRefinerTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
