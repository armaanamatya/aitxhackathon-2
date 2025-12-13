"""
Training Script for NAFNet-based Real Estate HDR Enhancement

NAFNet achieves state-of-the-art results for image restoration.
Simpler training than GANs - just L1 + Perceptual loss.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import get_dataloaders
from src.training.nafnet import create_nafnet, count_parameters
from src.training.models import VGGPerceptualLoss


class PSNRLoss(nn.Module):
    """PSNR-oriented loss (Charbonnier loss)."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class NAFNetTrainer:
    """
    Trainer for NAFNet image enhancement model.

    Simpler than GAN training - uses only reconstruction losses.
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        model_variant: str = "base",
        image_size: int = 512,
        batch_size: int = 8,
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        lambda_perceptual: float = 0.1,
        use_amp: bool = True,
        num_workers: int = 4,
        save_interval: int = 10,
        sample_interval: int = 5,
        resume_checkpoint: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Settings
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_perceptual = lambda_perceptual
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
        print(f"Initializing NAFNet-{model_variant}...")
        self.model = create_nafnet(
            variant=model_variant,
            in_channels=3,
            out_channels=3,
        ).to(self.device)

        print(f"Model parameters: {count_parameters(self.model):,}")

        # Loss functions
        self.criterion_l1 = nn.L1Loss()
        self.criterion_psnr = PSNRLoss()
        self.criterion_perceptual = VGGPerceptualLoss().to(self.device)

        # Optimizer - AdamW with cosine annealing
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.9),
            weight_decay=0.01,
        )

        # Learning rate scheduler - Cosine annealing with warmup
        warmup_epochs = 5
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-7,
        )

        # Warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * len(self.train_loader),
        )
        self.warmup_epochs = warmup_epochs

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_psnr = 0.0

        # Resume from checkpoint if provided
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'l1': 0.0,
            'perceptual': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                # Forward pass
                output = self.model(source)

                # Losses
                loss_l1 = self.criterion_l1(output, target)
                loss_perceptual = self.criterion_perceptual(output, target)

                # Total loss
                loss = loss_l1 + self.lambda_perceptual * loss_perceptual

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

            # Warmup scheduler step
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()

            # Update metrics
            epoch_losses['total'] += loss.item()
            epoch_losses['l1'] += loss_l1.item()
            epoch_losses['perceptual'] += loss_perceptual.item()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'L1': f"{loss_l1.item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        # Step cosine scheduler after warmup
        if epoch >= self.warmup_epochs:
            self.scheduler.step()

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> tuple:
        """Run validation and return metrics."""
        self.model.eval()

        total_l1_loss = 0.0
        total_psnr = 0.0
        num_batches = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(source)

            # L1 loss
            l1_loss = self.criterion_l1(output, target)
            total_l1_loss += l1_loss.item()

            # PSNR (convert from [-1,1] to [0,1] range)
            output_01 = (output + 1) / 2
            target_01 = (target + 1) / 2
            mse = torch.mean((output_01 - target_01) ** 2)
            psnr = 10 * torch.log10(1.0 / mse)
            total_psnr += psnr.item()

            num_batches += 1

        avg_l1 = total_l1_loss / num_batches
        avg_psnr = total_psnr / num_batches

        return avg_l1, avg_psnr

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample images for visualization."""
        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)

        output = self.model(source)

        # Adjust num_samples to actual batch size
        actual_samples = min(num_samples, source.size(0))

        def tensor_to_image(tensor):
            """Convert [-1, 1] tensor to PIL Image."""
            tensor = (tensor + 1) / 2
            tensor = tensor.clamp(0, 1)
            tensor = tensor.cpu().numpy().transpose(1, 2, 0)
            tensor = (tensor * 255).astype(np.uint8)
            return Image.fromarray(tensor)

        for i in range(actual_samples):
            src_img = tensor_to_image(source[i])
            out_img = tensor_to_image(output[i])
            tar_img = tensor_to_image(target[i])

            # Create side-by-side comparison: Source | Output | Target
            width = src_img.width
            height = src_img.height
            comparison = Image.new('RGB', (width * 3, height))
            comparison.paste(src_img, (0, 0))
            comparison.paste(out_img, (width, 0))
            comparison.paste(tar_img, (width * 2, 0))

            comparison.save(self.sample_dir / f"epoch_{epoch:04d}_sample_{i}.jpg", quality=95)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_psnr': self.best_psnr,
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        # Save epoch checkpoint
        if (epoch + 1) % self.save_interval == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            torch.save(
                self.model.state_dict(),
                self.checkpoint_dir / "best_model.pt"
            )
            print(f"  Saved best model (PSNR: {self.best_psnr:.2f} dB)")

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
        self.best_psnr = checkpoint.get('best_psnr', 0.0)

        print(f"Resumed from epoch {self.start_epoch}, best PSNR: {self.best_psnr:.2f} dB")

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("NAFNet Training for Real Estate HDR Enhancement")
        print(f"{'='*60}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        print(f"Using AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_l1, val_psnr = self.validate()

            # Check if best model
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_val_loss = val_l1
                self.best_psnr = val_psnr

            # Logging
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train - Loss: {train_losses['total']:.4f}, "
                  f"L1: {train_losses['l1']:.4f}, "
                  f"Perceptual: {train_losses['perceptual']:.4f}")
            print(f"  Val   - L1: {val_l1:.4f}, PSNR: {val_psnr:.2f} dB {'(best)' if is_best else ''}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save samples
            if (epoch + 1) % self.sample_interval == 0:
                self.save_samples(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation PSNR: {self.best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train NAFNet for Real Estate HDR Enhancement")

    # Data arguments
    parser.add_argument("--data_root", type=str, default=".",
                        help="Root directory containing images folder")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl",
                        help="Path to train.jsonl file")
    parser.add_argument("--output_dir", type=str, default="outputs_nafnet",
                        help="Output directory for checkpoints and samples")

    # Model arguments
    parser.add_argument("--model_variant", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="NAFNet variant")

    # Training arguments
    parser.add_argument("--image_size", type=int, default=512,
                        help="Training image size")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1,
                        help="Perceptual loss weight")

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
    trainer = NAFNetTrainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        model_variant=args.model_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lambda_perceptual=args.lambda_perceptual,
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
