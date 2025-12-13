"""
Training Script for Restormer HDR Enhancement Model

Trains Restormer (transformer-based) with:
- L1 + Perceptual + LPIPS + LAB losses
- No adversarial training (simpler, more stable)
- Progressive learning strategy
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
from src.training.restormer import Restormer, create_restormer, count_parameters
from src.training.models import (
    VGGPerceptualLoss,
    LPIPSLoss,
    LABColorLoss,
    ColorHistogramLoss,
    LPIPS_AVAILABLE,
)


class RestormerTrainer:
    """
    Trainer class for Restormer HDR Enhancement.

    No adversarial training - uses pure regression losses.
    More stable and often produces better results for image restoration.
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        model_size: str = "base",
        image_size: int = 512,
        batch_size: int = 4,
        num_epochs: int = 200,
        lr: float = 3e-4,
        lambda_l1: float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_lpips: float = 0.1,
        lambda_lab: float = 0.1,
        use_amp: bool = True,
        num_workers: int = 4,
        save_interval: int = 10,
        sample_interval: int = 5,
        resume_checkpoint: str = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! This training script requires a CUDA-capable GPU.")
        self.device = torch.device("cuda")
        print(f"Using device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Settings
        self.model_size = model_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_lpips = lambda_lpips
        self.lambda_lab = lambda_lab
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_interval = save_interval
        self.sample_interval = sample_interval

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
        print(f"Initializing Restormer-{model_size}...")
        self.model = create_restormer(model_size).to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")

        # Loss functions
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLoss().to(self.device)

        # Optional losses
        if LPIPS_AVAILABLE and lambda_lpips > 0:
            self.criterion_lpips = LPIPSLoss(net='alex').to(self.device)
            print("Using LPIPS loss")
        else:
            self.criterion_lpips = None

        if lambda_lab > 0:
            self.criterion_lab = LABColorLoss().to(self.device)
            print("Using LAB color loss")
        else:
            self.criterion_lab = None

        # Optimizer - AdamW with cosine annealing
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.02,
        )

        # Learning rate scheduler - Cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'l1': 0.0,
            'perceptual': 0.0,
            'lpips': 0.0,
            'lab': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Forward pass
                output = self.model(source)

                # L1 loss
                loss_l1 = self.criterion_l1(output, target)

                # Perceptual loss
                loss_perceptual = self.criterion_perceptual(output, target)

                # LPIPS loss
                loss_lpips = torch.tensor(0.0, device=self.device)
                if self.criterion_lpips is not None:
                    loss_lpips = self.criterion_lpips(output, target)

                # LAB color loss
                loss_lab = torch.tensor(0.0, device=self.device)
                if self.criterion_lab is not None:
                    loss_lab = self.criterion_lab(output, target)

                # Total loss
                loss = (
                    self.lambda_l1 * loss_l1 +
                    self.lambda_perceptual * loss_perceptual +
                    self.lambda_lpips * loss_lpips +
                    self.lambda_lab * loss_lab
                )

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
                self.optimizer.step()

            # Update metrics
            epoch_losses['total'] += loss.item()
            epoch_losses['l1'] += loss_l1.item()
            epoch_losses['perceptual'] += loss_perceptual.item()
            epoch_losses['lpips'] += loss_lpips.item()
            epoch_losses['lab'] += loss_lab.item()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'L1': f"{loss_l1.item():.4f}",
                'LPIPS': f"{loss_lpips.item():.4f}",
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
        num_batches = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(source)
            loss = self.criterion_l1(output, target)

            total_l1_loss += loss.item()
            num_batches += 1

        return total_l1_loss / num_batches

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample images for visualization."""
        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        source = batch['source'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)

        output = self.model(source)

        actual_samples = min(num_samples, source.size(0))

        def tensor_to_image(tensor):
            tensor = (tensor + 1) / 2
            tensor = tensor.clamp(0, 1)
            tensor = tensor.cpu().numpy().transpose(1, 2, 0)
            tensor = (tensor * 255).astype(np.uint8)
            return Image.fromarray(tensor)

        for i in range(actual_samples):
            src_img = tensor_to_image(source[i])
            tar_img = tensor_to_image(target[i])
            out_img = tensor_to_image(output[i])

            width = src_img.width
            height = src_img.height
            comparison = Image.new('RGB', (width * 3, height))
            comparison.paste(src_img, (0, 0))
            comparison.paste(out_img, (width, 0))
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
            'model_size': self.model_size,
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        if (epoch + 1) % self.save_interval == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            torch.save(
                self.model.state_dict(),
                self.checkpoint_dir / "best_generator.pt"
            )

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
        print(f"\nStarting Restormer training from epoch {self.start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        print(f"Model: Restormer-{self.model_size}")
        print(f"Using AMP: {self.use_amp}")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Logging
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"L1: {train_losses['l1']:.4f}, "
                  f"Perceptual: {train_losses['perceptual']:.4f}, "
                  f"LPIPS: {train_losses['lpips']:.4f}, "
                  f"LAB: {train_losses['lab']:.4f}")
            print(f"  Val - L1: {val_loss:.4f} {'(best)' if is_best else ''}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # Save samples
            if (epoch + 1) % self.sample_interval == 0:
                self.save_samples(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best validation L1 loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Restormer HDR Enhancement Model")

    # Data arguments
    parser.add_argument("--data_root", type=str, default=".",
                        help="Root directory containing images folder")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl",
                        help="Path to train.jsonl file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and samples")

    # Model arguments
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["tiny", "small", "base", "large"],
                        help="Restormer model size")

    # Training arguments
    parser.add_argument("--image_size", type=int, default=512,
                        help="Training image size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")

    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=1.0,
                        help="L1 loss weight")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1,
                        help="VGG perceptual loss weight")
    parser.add_argument("--lambda_lpips", type=float, default=0.1,
                        help="LPIPS perceptual loss weight")
    parser.add_argument("--lambda_lab", type=float, default=0.1,
                        help="LAB color loss weight")

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
    trainer = RestormerTrainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        lambda_perceptual=args.lambda_perceptual,
        lambda_lpips=args.lambda_lpips,
        lambda_lab=args.lambda_lab,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        resume_checkpoint=args.resume,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
