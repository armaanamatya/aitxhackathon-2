"""
Training Script for ControlNet HDR Enhancement
===============================================

Trains ControlNet for real estate photo enhancement.
Uses the source image as conditioning to guide generation.

Usage:
    python src/training/train_controlnet.py \
        --data_root ./data \
        --jsonl_path train.jsonl \
        --output_dir outputs/controlnet \
        --epochs 100
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import get_dataloaders
from src.training.controlnet import ControlNetHDR, ControlNetLite, count_parameters
from src.training.models import (
    VGGPerceptualLoss,
    LPIPSLoss,
    LABColorLoss,
    ColorHistogramLoss,
    SSIMLoss,
    EdgeAwareLoss,
    LPIPS_AVAILABLE,
)


class ControlNetTrainer:
    """
    Trainer for ControlNet HDR Enhancement.

    Unlike GAN training, this uses direct supervision without adversarial loss,
    making training more stable and faster.
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        model_type: str = "full",  # "full" or "lite"
        image_size: int = 512,
        batch_size: int = 4,
        num_epochs: int = 100,
        lr: float = 2e-4,
        # Loss weights
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 10.0,
        lambda_lpips: float = 5.0,
        lambda_ssim: float = 5.0,
        lambda_edge: float = 2.0,
        lambda_lab: float = 10.0,
        lambda_hist: float = 1.0,
        # Training settings
        use_amp: bool = True,
        num_workers: int = 4,
        save_interval: int = 10,
        sample_interval: int = 5,
        resume_checkpoint: str = None,
        grad_clip: float = 1.0,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Settings
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_lpips = lambda_lpips
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_lab = lambda_lab
        self.lambda_hist = lambda_hist
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_interval = save_interval
        self.sample_interval = sample_interval
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

        # Output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        print("Loading datasets...")
        self.train_loader, self.val_loader = get_dataloaders(
            data_root=data_root,
            jsonl_path=jsonl_path,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            train_ratio=0.9,
        )

        # Model
        print(f"Initializing ControlNet ({model_type})...")
        if model_type == "lite":
            self.model = ControlNetLite(
                in_channels=3,
                out_channels=3,
                base_channels=32,
                learn_residual=True,
            ).to(self.device)
        else:
            self.model = ControlNetHDR(
                in_channels=3,
                out_channels=3,
                base_channels=64,
                num_res_blocks=2,
                learn_residual=True,
            ).to(self.device)

        print(f"Model parameters: {count_parameters(self.model):,}")

        # Loss functions
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLoss().to(self.device)

        if LPIPS_AVAILABLE and lambda_lpips > 0:
            self.criterion_lpips = LPIPSLoss(net='alex').to(self.device)
            print("Using LPIPS loss")
        else:
            self.criterion_lpips = None

        if lambda_ssim > 0:
            self.criterion_ssim = SSIMLoss().to(self.device)
            print("Using SSIM loss")
        else:
            self.criterion_ssim = None

        if lambda_edge > 0:
            self.criterion_edge = EdgeAwareLoss().to(self.device)
            print("Using Edge-Aware loss")
        else:
            self.criterion_edge = None

        if lambda_lab > 0:
            self.criterion_lab = LABColorLoss().to(self.device)
            print("Using LAB color loss")
        else:
            self.criterion_lab = None

        if lambda_hist > 0:
            self.criterion_hist = ColorHistogramLoss().to(self.device)
            print("Using Color Histogram loss")
        else:
            self.criterion_hist = None

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(10, (num_epochs - warmup_epochs) // 3),
            T_mult=2,
            eta_min=min_lr,
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.base_lr = lr

        # Resume from checkpoint if provided
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    def _get_warmup_lr(self, epoch: int) -> float:
        """Calculate learning rate with warmup."""
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        return self.base_lr

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """Compute all losses and return total + components."""
        losses = {}

        # L1 loss
        losses['l1'] = self.criterion_l1(pred, target)

        # Perceptual loss
        losses['perceptual'] = self.criterion_perceptual(pred, target)

        # LPIPS loss
        losses['lpips'] = torch.tensor(0.0, device=self.device)
        if self.criterion_lpips is not None:
            losses['lpips'] = self.criterion_lpips(pred, target)

        # SSIM loss
        losses['ssim'] = torch.tensor(0.0, device=self.device)
        if self.criterion_ssim is not None:
            losses['ssim'] = self.criterion_ssim(pred, target)

        # Edge loss
        losses['edge'] = torch.tensor(0.0, device=self.device)
        if self.criterion_edge is not None:
            losses['edge'] = self.criterion_edge(pred, target)

        # LAB loss
        losses['lab'] = torch.tensor(0.0, device=self.device)
        if self.criterion_lab is not None:
            losses['lab'] = self.criterion_lab(pred, target)

        # Histogram loss
        losses['hist'] = torch.tensor(0.0, device=self.device)
        if self.criterion_hist is not None:
            losses['hist'] = self.criterion_hist(pred, target)

        # Total loss
        total = (
            self.lambda_l1 * losses['l1'] +
            self.lambda_perceptual * losses['perceptual'] +
            self.lambda_lpips * losses['lpips'] +
            self.lambda_ssim * losses['ssim'] +
            self.lambda_edge * losses['edge'] +
            self.lambda_lab * losses['lab'] +
            self.lambda_hist * losses['hist']
        )

        return total, losses

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'l1': 0.0,
            'perceptual': 0.0,
            'lpips': 0.0,
            'ssim': 0.0,
            'edge': 0.0,
            'lab': 0.0,
            'hist': 0.0,
        }

        # Warmup learning rate
        if epoch < self.warmup_epochs:
            warmup_lr = self._get_warmup_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Forward pass
                pred = self.model(source)

                # Compute losses
                total_loss, losses = self.compute_loss(pred, target)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Update metrics
            epoch_losses['total'] += total_loss.item()
            for key in losses:
                epoch_losses[key] += losses[key].item()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'L1': f"{losses['l1'].item():.3f}",
                'SSIM': f"{losses['ssim'].item():.3f}",
            })

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average L1 loss."""
        self.model.eval()

        total_l1_loss = 0.0
        total_ssim_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            pred = self.model(source)
            l1_loss = self.criterion_l1(pred, target)

            total_l1_loss += l1_loss.item()
            if self.criterion_ssim is not None:
                total_ssim_loss += self.criterion_ssim(pred, target).item()
            num_batches += 1

        return total_l1_loss / num_batches

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample images for visualization."""
        self.model.eval()

        batch = next(iter(self.val_loader))
        source = batch['source'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)

        pred = self.model(source)

        def tensor_to_image(tensor):
            tensor = (tensor + 1) / 2
            tensor = tensor.clamp(0, 1)
            tensor = tensor.cpu().numpy().transpose(1, 2, 0)
            tensor = (tensor * 255).astype(np.uint8)
            return Image.fromarray(tensor)

        for i in range(min(num_samples, source.size(0))):
            src_img = tensor_to_image(source[i])
            tar_img = tensor_to_image(target[i])
            pred_img = tensor_to_image(pred[i])

            width = src_img.width
            height = src_img.height
            comparison = Image.new('RGB', (width * 3, height))
            comparison.paste(src_img, (0, 0))
            comparison.paste(pred_img, (width, 0))
            comparison.paste(tar_img, (width * 2, 0))

            comparison.save(self.sample_dir / f"epoch_{epoch:04d}_sample_{i}.jpg")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        if (epoch + 1) % self.save_interval == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            torch.save(self.model.state_dict(), self.checkpoint_dir / "controlnet_best.pt")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop."""
        print(f"\nStarting ControlNet training from epoch {self.start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        print(f"Using AMP: {self.use_amp}")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update learning rate (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Logging
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}")
            print(f"  Losses - L1: {train_losses['l1']:.4f}, VGG: {train_losses['perceptual']:.4f}, "
                  f"SSIM: {train_losses['ssim']:.4f}, Edge: {train_losses['edge']:.4f}")
            print(f"  Val - L1: {val_loss:.4f} {'(best)' if is_best else ''}")
            warmup_status = "(warmup)" if epoch < self.warmup_epochs else ""
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e} {warmup_status}")

            # Save samples
            if (epoch + 1) % self.sample_interval == 0:
                self.save_samples(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best validation L1 loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet for HDR Enhancement")

    # Data arguments
    parser.add_argument("--data_root", type=str, default=".",
                        help="Root directory containing images folder")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl",
                        help="Path to train.jsonl file")
    parser.add_argument("--output_dir", type=str, default="outputs/controlnet",
                        help="Output directory for checkpoints and samples")

    # Model arguments
    parser.add_argument("--model_type", type=str, default="full",
                        choices=["full", "lite"],
                        help="Model variant: 'full' (ControlNetHDR) or 'lite' (ControlNetLite)")

    # Training arguments
    parser.add_argument("--image_size", type=int, default=512,
                        help="Training image size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")

    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=100.0,
                        help="L1 loss weight")
    parser.add_argument("--lambda_perceptual", type=float, default=10.0,
                        help="VGG perceptual loss weight")
    parser.add_argument("--lambda_lpips", type=float, default=5.0,
                        help="LPIPS loss weight")
    parser.add_argument("--lambda_ssim", type=float, default=5.0,
                        help="SSIM loss weight")
    parser.add_argument("--lambda_edge", type=float, default=2.0,
                        help="Edge-aware loss weight")
    parser.add_argument("--lambda_lab", type=float, default=10.0,
                        help="LAB color loss weight")
    parser.add_argument("--lambda_hist", type=float, default=1.0,
                        help="Color histogram loss weight")

    # Training enhancements
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")

    # Other arguments
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data loader workers")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="Save samples every N epochs")

    args = parser.parse_args()

    # Create trainer
    trainer = ControlNetTrainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        lambda_perceptual=args.lambda_perceptual,
        lambda_lpips=args.lambda_lpips,
        lambda_ssim=args.lambda_ssim,
        lambda_edge=args.lambda_edge,
        lambda_lab=args.lambda_lab,
        lambda_hist=args.lambda_hist,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        resume_checkpoint=args.resume,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
