"""
Training Script for Real Estate HDR Enhancement Model

Trains a Pix2Pix-style GAN with:
- U-Net Generator with residual learning
- PatchGAN Discriminator
- L1 + Perceptual + Adversarial losses
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
from src.training.models import (
    UNetGenerator,
    PatchDiscriminator,
    VGGPerceptualLoss,
    count_parameters,
)


class Trainer:
    """
    Trainer class for Real Estate HDR Enhancement.

    Handles:
    - GAN training with generator and discriminator
    - Mixed precision training (AMP)
    - Checkpointing and resuming
    - Logging and visualization
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        image_size: int = 512,
        batch_size: int = 4,
        num_epochs: int = 200,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 10.0,
        lambda_adv: float = 1.0,
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
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv
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

        # Models
        print("Initializing models...")
        self.generator = UNetGenerator(
            in_channels=3,
            out_channels=3,
            base_features=64,
            num_residual_blocks=9,
            learn_residual=True,
        ).to(self.device)

        self.discriminator = PatchDiscriminator(
            in_channels=6,
            base_features=64,
        ).to(self.device)

        print(f"Generator parameters: {count_parameters(self.generator):,}")
        print(f"Discriminator parameters: {count_parameters(self.discriminator):,}")

        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLoss().to(self.device)

        # Optimizers
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=lr_g,
            betas=(0.5, 0.999),
            weight_decay=0.01,
        )
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(0.5, 0.999),
            weight_decay=0.01,
        )

        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=num_epochs, eta_min=1e-6
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=num_epochs, eta_min=1e-6
        )

        # Mixed precision scalers
        self.scaler_g = GradScaler() if self.use_amp else None
        self.scaler_d = GradScaler() if self.use_amp else None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint if provided
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()

        epoch_losses = {
            'g_total': 0.0,
            'g_adv': 0.0,
            'g_l1': 0.0,
            'g_perceptual': 0.0,
            'd_total': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            batch_size = source.size(0)

            # ==================
            # Train Discriminator
            # ==================
            self.optimizer_d.zero_grad()

            with autocast(enabled=self.use_amp):
                # Generate fake images
                fake = self.generator(source)

                # Real loss
                pred_real = self.discriminator(source, target)

                # Create labels matching discriminator output shape
                real_label = torch.ones_like(pred_real)
                fake_label = torch.zeros_like(pred_real)

                loss_d_real = self.criterion_gan(pred_real, real_label)

                # Fake loss
                pred_fake = self.discriminator(source, fake.detach())
                loss_d_fake = self.criterion_gan(pred_fake, fake_label)

                # Total discriminator loss
                loss_d = (loss_d_real + loss_d_fake) * 0.5

            if self.use_amp:
                self.scaler_d.scale(loss_d).backward()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                loss_d.backward()
                self.optimizer_d.step()

            # ==================
            # Train Generator
            # ==================
            self.optimizer_g.zero_grad()

            with autocast(enabled=self.use_amp):
                # Generate fake images (need to regenerate for generator update)
                fake = self.generator(source)

                # Adversarial loss
                pred_fake = self.discriminator(source, fake)
                # Generator wants discriminator to think fake is real
                real_label_g = torch.ones_like(pred_fake)
                loss_g_adv = self.criterion_gan(pred_fake, real_label_g)

                # L1 loss
                loss_g_l1 = self.criterion_l1(fake, target)

                # Perceptual loss
                loss_g_perceptual = self.criterion_perceptual(fake, target)

                # Total generator loss
                loss_g = (
                    self.lambda_adv * loss_g_adv +
                    self.lambda_l1 * loss_g_l1 +
                    self.lambda_perceptual * loss_g_perceptual
                )

            if self.use_amp:
                self.scaler_g.scale(loss_g).backward()
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                loss_g.backward()
                self.optimizer_g.step()

            # Update metrics
            epoch_losses['g_total'] += loss_g.item()
            epoch_losses['g_adv'] += loss_g_adv.item()
            epoch_losses['g_l1'] += loss_g_l1.item()
            epoch_losses['g_perceptual'] += loss_g_perceptual.item()
            epoch_losses['d_total'] += loss_d.item()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'G': f"{loss_g.item():.4f}",
                'D': f"{loss_d.item():.4f}",
                'L1': f"{loss_g_l1.item():.4f}",
            })

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average L1 loss."""
        self.generator.eval()

        total_l1_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            fake = self.generator(source)
            loss = self.criterion_l1(fake, target)

            total_l1_loss += loss.item()
            num_batches += 1

        return total_l1_loss / num_batches

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample images for visualization."""
        self.generator.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        source = batch['source'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)

        fake = self.generator(source)

        # Adjust num_samples to actual batch size
        actual_samples = min(num_samples, source.size(0))

        # Convert tensors to images
        def tensor_to_image(tensor):
            """Convert [-1, 1] tensor to PIL Image."""
            tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
            tensor = tensor.clamp(0, 1)
            tensor = tensor.cpu().numpy().transpose(1, 2, 0)
            tensor = (tensor * 255).astype(np.uint8)
            return Image.fromarray(tensor)

        # Create comparison grid
        for i in range(actual_samples):
            src_img = tensor_to_image(source[i])
            tar_img = tensor_to_image(target[i])
            fake_img = tensor_to_image(fake[i])

            # Create side-by-side comparison
            width = src_img.width
            height = src_img.height
            comparison = Image.new('RGB', (width * 3, height))
            comparison.paste(src_img, (0, 0))
            comparison.paste(fake_img, (width, 0))
            comparison.paste(tar_img, (width * 2, 0))

            comparison.save(self.sample_dir / f"epoch_{epoch:04d}_sample_{i}.jpg")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        if self.use_amp:
            checkpoint['scaler_g_state_dict'] = self.scaler_g.state_dict()
            checkpoint['scaler_d_state_dict'] = self.scaler_d.state_dict()

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        # Save epoch checkpoint
        if (epoch + 1) % self.save_interval == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            # Also save just the generator for easy inference
            torch.save(
                self.generator.state_dict(),
                self.checkpoint_dir / "best_generator.pt"
            )

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        if self.use_amp and 'scaler_g_state_dict' in checkpoint:
            self.scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
            self.scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training from epoch {self.start_epoch}")
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

            # Update learning rates
            self.scheduler_g.step()
            self.scheduler_d.step()

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Logging
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train - G: {train_losses['g_total']:.4f}, "
                  f"D: {train_losses['d_total']:.4f}, "
                  f"L1: {train_losses['g_l1']:.4f}, "
                  f"Perceptual: {train_losses['g_perceptual']:.4f}")
            print(f"  Val - L1: {val_loss:.4f} {'(best)' if is_best else ''}")
            print(f"  LR - G: {self.scheduler_g.get_last_lr()[0]:.2e}, "
                  f"D: {self.scheduler_d.get_last_lr()[0]:.2e}")

            # Save samples
            if (epoch + 1) % self.sample_interval == 0:
                self.save_samples(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best validation L1 loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Real Estate HDR Enhancement Model")

    # Data arguments
    parser.add_argument("--data_root", type=str, default=".",
                        help="Root directory containing images folder")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl",
                        help="Path to train.jsonl file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and samples")

    # Training arguments
    parser.add_argument("--image_size", type=int, default=512,
                        help="Training image size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--lr_g", type=float, default=2e-4,
                        help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4,
                        help="Discriminator learning rate")

    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=100.0,
                        help="L1 loss weight")
    parser.add_argument("--lambda_perceptual", type=float, default=10.0,
                        help="Perceptual loss weight")
    parser.add_argument("--lambda_adv", type=float, default=1.0,
                        help="Adversarial loss weight")

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
    trainer = Trainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_l1=args.lambda_l1,
        lambda_perceptual=args.lambda_perceptual,
        lambda_adv=args.lambda_adv,
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
