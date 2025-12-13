"""
SwinRestormer Fine-Tuning Training Script
==========================================

Optimal fine-tuning strategy for small datasets (~550 images).

Anti-Overfitting Techniques:
1. Progressive unfreezing (3 stages)
2. Layer-wise learning rate decay
3. Strong data augmentation (flips, rotations, color jitter, mixup)
4. Early stopping with patience
5. EMA for stable outputs
6. Dropout in decoder
7. Weight decay regularization
8. Gradient clipping
9. Warmup + cosine annealing LR schedule

Training Stages:
- Stage 1 (epochs 1-30): Freeze encoder, train decoder only, LR=2e-4
- Stage 2 (epochs 31-60): Unfreeze last 2 encoder layers, LR=5e-5
- Stage 3 (epochs 61+): Full fine-tuning, LR=1e-5
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from swin_restormer import (
    SwinRestormer, swin_restormer_tiny, swin_restormer_small, swin_restormer_base
)


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
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(3, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=3)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=3)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=3) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=3) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27]])
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred * 0.5 + 0.5 - self.mean) / self.std
        target = (target * 0.5 + 0.5 - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            pred, target = block(pred), block(target)
            loss += F.l1_loss(pred, target)
        return loss


class LABColorLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def rgb_to_lab(rgb):
            rgb = rgb * 0.5 + 0.5
            mask = rgb > 0.04045
            rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
            x = rgb[:, 0] * 0.4124564 + rgb[:, 1] * 0.3575761 + rgb[:, 2] * 0.1804375
            y = rgb[:, 0] * 0.2126729 + rgb[:, 1] * 0.7151522 + rgb[:, 2] * 0.0721750
            z = rgb[:, 0] * 0.0193339 + rgb[:, 1] * 0.1191920 + rgb[:, 2] * 0.9503041
            x, z = x / 0.95047, z / 1.08883
            def f(t):
                delta = 6.0 / 29.0
                return torch.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4/29)
            fx, fy, fz = f(x), f(y), f(z)
            return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)
        return F.l1_loss(rgb_to_lab(pred), rgb_to_lab(target))


# =============================================================================
# Dataset with Strong Augmentation
# =============================================================================

class RetouchDatasetAugmented(Dataset):
    """Dataset with strong augmentation for preventing overfitting."""

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        image_size: int = 256,
        is_training: bool = True,
        use_strong_augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.is_training = is_training
        self.use_strong_augment = use_strong_augment and is_training

        self.samples = []
        with open(self.data_root / jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        split_idx = int(len(self.samples) * 0.9)
        self.samples = self.samples[:split_idx] if is_training else self.samples[split_idx:]
        print(f"Loaded {len(self.samples)} {'train' if is_training else 'val'} samples")

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(self.data_root / path).convert('RGB')

    def _apply_augmentation(self, source: Image.Image, target: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply identical geometric augmentation to both images."""

        # Random horizontal flip
        if random.random() > 0.5:
            source = source.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            source = source.transpose(Image.FLIP_TOP_BOTTOM)
            target = target.transpose(Image.FLIP_TOP_BOTTOM)

        # Random rotation (0, 90, 180, 270)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            source = source.rotate(angle, expand=False)
            target = target.rotate(angle, expand=False)

        # Random crop (with some margin)
        if random.random() > 0.5:
            w, h = source.size
            crop_size = int(min(w, h) * random.uniform(0.8, 1.0))
            left = random.randint(0, w - crop_size)
            top = random.randint(0, h - crop_size)
            source = source.crop((left, top, left + crop_size, top + crop_size))
            target = target.crop((left, top, left + crop_size, top + crop_size))

        return source, target

    def _apply_color_augmentation(self, img: Image.Image) -> Image.Image:
        """Apply color augmentation to source only (simulates different input conditions)."""
        if random.random() > 0.7:
            # Brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() > 0.7:
            # Contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() > 0.8:
            # Saturation
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        src_key = 'src' if 'src' in item else 'source'
        tar_key = 'tar' if 'tar' in item else 'target'

        source = self._load_image(item[src_key])
        target = self._load_image(item[tar_key])

        # Apply augmentation
        if self.use_strong_augment:
            source, target = self._apply_augmentation(source, target)
            source = self._apply_color_augmentation(source)

        # Resize
        source = source.resize((self.image_size, self.image_size), Image.LANCZOS)
        target = target.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensor
        source = torch.from_numpy(np.array(source).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        target = torch.from_numpy(np.array(target).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1

        return {'source': source, 'target': target}


# =============================================================================
# Mixup Augmentation
# =============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation for regularization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam


# =============================================================================
# EMA
# =============================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def update_shadow_params(self):
        """Update shadow dict when trainable params change (after unfreezing)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()


# =============================================================================
# Trainer with Progressive Unfreezing
# =============================================================================

class SwinRestormerTrainer:
    def __init__(
        self,
        model: SwinRestormer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        # Stage-wise learning rates
        lr_stage1: float = 2e-4,
        lr_stage2: float = 5e-5,
        lr_stage3: float = 1e-5,
        # Stage epochs
        stage1_epochs: int = 30,
        stage2_epochs: int = 30,
        stage3_epochs: int = 40,
        # Regularization
        weight_decay: float = 0.05,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        # Loss weights
        lambda_char: float = 10.0,
        lambda_ssim: float = 1.0,
        lambda_perceptual: float = 0.5,
        lambda_lab: float = 0.5,
        # Training settings
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        patience: int = 15,
        save_interval: int = 5,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage configuration
        self.lr_stages = [lr_stage1, lr_stage2, lr_stage3]
        self.stage_epochs = [stage1_epochs, stage2_epochs, stage3_epochs]
        self.total_epochs = sum(self.stage_epochs)
        self.current_stage = 1

        self.weight_decay = weight_decay
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_amp = use_amp
        self.save_interval = save_interval

        # Loss functions
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_lab = lambda_lab

        self.char_loss = CharbonnierLoss().to(self.device)
        self.ssim_loss = SSIMLoss().to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.lab_loss = LABColorLoss().to(self.device)

        # LPIPS
        try:
            import lpips
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_loss.eval()
            self.lambda_lpips = 0.5
            print("Using LPIPS loss")
        except:
            self.lpips_loss = None
            self.lambda_lpips = 0.0

        # Initialize optimizer for stage 1
        self._setup_stage(1)

        # AMP
        self.scaler = GradScaler() if use_amp else None

        # EMA
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            print(f"Using EMA with decay {ema_decay}")

        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 0

    def _setup_stage(self, stage: int):
        """Setup optimizer and scheduler for a training stage."""
        self.current_stage = stage
        self.model.set_training_stage(stage)

        lr = self.lr_stages[stage - 1]
        stage_epochs = self.stage_epochs[stage - 1]

        # Get appropriate parameters
        if stage == 1:
            # Only decoder params
            params = self.model.get_decoder_params()
        else:
            # Different LR for encoder and decoder
            encoder_params = self.model.get_encoder_params()
            decoder_params = self.model.get_decoder_params()
            params = [
                {'params': encoder_params, 'lr': lr * 0.1},  # Lower LR for encoder
                {'params': decoder_params, 'lr': lr},
            ]

        self.optimizer = AdamW(
            params if stage == 1 else params,
            lr=lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=stage_epochs,
            T_mult=1,
            eta_min=lr * 0.01,
        )

        print(f"\nStage {stage}: LR={lr}, Epochs={stage_epochs}")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

    def compute_losses(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        losses['char'] = self.char_loss(pred, target)
        losses['ssim'] = self.ssim_loss(pred, target)
        losses['perceptual'] = self.perceptual_loss(pred, target)
        losses['lab'] = self.lab_loss(pred, target)

        if self.lpips_loss is not None:
            losses['lpips'] = self.lpips_loss(pred, target).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=self.device)

        losses['total'] = (
            self.lambda_char * losses['char'] +
            self.lambda_ssim * losses['ssim'] +
            self.lambda_perceptual * losses['perceptual'] +
            self.lambda_lab * losses['lab'] +
            self.lambda_lpips * losses['lpips']
        )
        return losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.total_epochs} (Stage {self.current_stage})")

        running_losses = {'total': 0.0, 'char': 0.0, 'ssim': 0.0}
        num_batches = 0

        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            # Mixup augmentation
            if self.use_mixup and random.random() > 0.5:
                source, target, _ = mixup_data(source, target, self.mixup_alpha)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output = self.model(source)['output']
                losses = self.compute_losses(output, target)

            if self.use_amp:
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if self.use_ema:
                self.ema.update()

            for key in running_losses:
                if key in losses:
                    running_losses[key] += losses[key].item()
            num_batches += 1

            pbar.set_postfix({
                'Loss': f"{running_losses['total'] / num_batches:.4f}",
                'Stage': self.current_stage,
            })

        self.scheduler.step()

        return {k: v / num_batches for k, v in running_losses.items()}

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        if self.use_ema:
            self.ema.apply_shadow()

        total_l1 = 0.0
        num_samples = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            output = self.model(source)['output']

            pred_01 = output * 0.5 + 0.5
            target_01 = target * 0.5 + 0.5
            total_l1 += F.l1_loss(pred_01, target_01).item() * source.size(0)
            num_samples += source.size(0)

        if self.use_ema:
            self.ema.restore()

        return total_l1 / num_samples

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if self.use_ema:
            state['ema_shadow'] = self.ema.shadow
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(state, self.output_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(state, self.output_dir / 'checkpoint_best.pt')
        if (epoch + 1) % self.save_interval == 0:
            torch.save(state, self.output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    def train(self):
        print(f"\n{'='*60}")
        print("SwinRestormer Fine-Tuning with Progressive Unfreezing")
        print(f"{'='*60}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Stages: {self.stage_epochs}")
        print(f"{'='*60}\n")

        epoch = self.start_epoch
        stage_start_epochs = [0, self.stage_epochs[0], self.stage_epochs[0] + self.stage_epochs[1]]

        while epoch < self.total_epochs:
            # Check if we need to switch stages
            if epoch == stage_start_epochs[1] and self.current_stage == 1:
                self._setup_stage(2)
                if self.use_ema:
                    self.ema.update_shadow_params()
            elif epoch == stage_start_epochs[2] and self.current_stage == 2:
                self._setup_stage(3)
                if self.use_ema:
                    self.ema.update_shadow_params()

            train_losses = self.train_epoch(epoch)
            val_l1 = self.validate()

            is_best = val_l1 < self.best_val_loss
            if is_best:
                self.best_val_loss = val_l1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            print(f"\nEpoch {epoch + 1}/{self.total_epochs} (Stage {self.current_stage})")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val L1: {val_l1:.4f} {'(best)' if is_best else ''}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")

            self.save_checkpoint(epoch, is_best)

            # Early stopping (only in stage 3)
            if self.current_stage == 3 and self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            epoch += 1

        print(f"\nTraining complete! Best Val L1: {self.best_val_loss:.4f}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='SwinRestormer Fine-Tuning')

    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs_swin_restormer')
    parser.add_argument('--image_size', type=int, default=256)

    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'base'])

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--stage1_epochs', type=int, default=30)
    parser.add_argument('--stage2_epochs', type=int, default=30)
    parser.add_argument('--stage3_epochs', type=int, default=40)

    parser.add_argument('--lr_stage1', type=float, default=2e-4)
    parser.add_argument('--lr_stage2', type=float, default=5e-5)
    parser.add_argument('--lr_stage3', type=float, default=1e-5)

    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)

    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("Loading datasets with strong augmentation...")
    train_dataset = RetouchDatasetAugmented(
        args.data_root, args.jsonl_path, args.image_size,
        is_training=True, use_strong_augment=True
    )
    val_dataset = RetouchDatasetAugmented(
        args.data_root, args.jsonl_path, args.image_size,
        is_training=False, use_strong_augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"\nInitializing SwinRestormer-{args.model_size} with pretrained encoder...")
    if args.model_size == 'tiny':
        model = swin_restormer_tiny(pretrained=True, freeze_encoder=True)
    elif args.model_size == 'small':
        model = swin_restormer_small(pretrained=True, freeze_encoder=True)
    else:
        model = swin_restormer_base(pretrained=True, freeze_encoder=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    trainer = SwinRestormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        lr_stage3=args.lr_stage3,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        weight_decay=args.weight_decay,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        patience=args.patience,
    )

    trainer.train()


if __name__ == '__main__':
    main()
