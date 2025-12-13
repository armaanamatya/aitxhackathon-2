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
    MultiScaleDiscriminator,
    SpectralNormMultiScaleDiscriminator,
    VGGPerceptualLoss,
    LPIPSLoss,
    LABColorLoss,
    ColorHistogramLoss,
    count_parameters,
    LPIPS_AVAILABLE,
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
        lambda_lpips: float = 5.0,
        lambda_lab: float = 10.0,
        lambda_hist: float = 1.0,
        lambda_adv: float = 1.0,
        use_multiscale_disc: bool = True,
        num_disc_scales: int = 2,
        use_spectral_norm: bool = False,
        use_amp: bool = True,
        num_workers: int = 4,
        save_interval: int = 10,
        sample_interval: int = 5,
        resume_checkpoint: str = None,
        # GAN stabilization parameters
        label_smoothing: float = 0.1,
        instance_noise: float = 0.1,
        instance_noise_decay: float = 0.9995,
        grad_clip_g: float = 1.0,
        grad_clip_d: float = 1.0,
        d_update_freq: int = 1,
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
        self.lambda_lab = lambda_lab
        self.lambda_hist = lambda_hist
        self.lambda_adv = lambda_adv
        self.use_multiscale_disc = use_multiscale_disc
        self.num_disc_scales = num_disc_scales
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_interval = save_interval
        self.sample_interval = sample_interval

        # GAN stabilization settings
        self.label_smoothing = label_smoothing
        self.instance_noise = instance_noise
        self.instance_noise_decay = instance_noise_decay
        self.current_noise = instance_noise
        self.grad_clip_g = grad_clip_g
        self.grad_clip_d = grad_clip_d
        self.d_update_freq = d_update_freq

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

        # Use multi-scale or single-scale discriminator
        if use_spectral_norm:
            # Spectral Normalization for stable GAN training
            self.discriminator = SpectralNormMultiScaleDiscriminator(
                in_channels=6,
                base_features=64,
                num_scales=num_disc_scales,
            ).to(self.device)
            print(f"Using Spectral Norm Multi-Scale Discriminator with {num_disc_scales} scales")
        elif use_multiscale_disc:
            self.discriminator = MultiScaleDiscriminator(
                in_channels=6,
                base_features=64,
                num_scales=num_disc_scales,
            ).to(self.device)
            print(f"Using Multi-Scale Discriminator with {num_disc_scales} scales")
        else:
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

        # Additional losses for color quality
        if LPIPS_AVAILABLE and lambda_lpips > 0:
            self.criterion_lpips = LPIPSLoss(net='alex').to(self.device)
            print("Using LPIPS loss")
        else:
            self.criterion_lpips = None
            if lambda_lpips > 0:
                print("Warning: LPIPS not available, skipping")

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

    def _add_instance_noise(self, x):
        """Add instance noise to help stabilize GAN training."""
        if self.current_noise > 0:
            noise = torch.randn_like(x) * self.current_noise
            return x + noise
        return x

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()

        epoch_losses = {
            'g_total': 0.0,
            'g_adv': 0.0,
            'g_l1': 0.0,
            'g_perceptual': 0.0,
            'g_lpips': 0.0,
            'g_lab': 0.0,
            'g_hist': 0.0,
            'd_total': 0.0,
        }

        # Label smoothing values
        real_label_val = 1.0 - self.label_smoothing  # 0.9
        fake_label_val = self.label_smoothing  # 0.1

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            batch_size = source.size(0)

            # ==================
            # Train Discriminator (with frequency control)
            # ==================
            train_d_this_step = (self.global_step % self.d_update_freq == 0)

            if train_d_this_step:
                self.optimizer_d.zero_grad()

                with autocast(enabled=self.use_amp):
                    # Generate fake images
                    fake = self.generator(source)

                    # Add instance noise to inputs for regularization
                    target_noisy = self._add_instance_noise(target)
                    fake_noisy = self._add_instance_noise(fake.detach())

                    # Handle multi-scale or single-scale discriminator
                    if self.use_multiscale_disc:
                        # Multi-scale discriminator returns list of outputs
                        pred_real_list = self.discriminator(source, target_noisy)
                        pred_fake_list = self.discriminator(source, fake_noisy)

                        loss_d_real = 0
                        loss_d_fake = 0
                        for pred_real, pred_fake in zip(pred_real_list, pred_fake_list):
                            # Label smoothing: real=0.9, fake=0.1
                            real_label = torch.full_like(pred_real, real_label_val)
                            fake_label = torch.full_like(pred_fake, fake_label_val)
                            loss_d_real += self.criterion_gan(pred_real, real_label)
                            loss_d_fake += self.criterion_gan(pred_fake, fake_label)

                        loss_d_real /= len(pred_real_list)
                        loss_d_fake /= len(pred_fake_list)
                    else:
                        # Single-scale discriminator
                        pred_real = self.discriminator(source, target_noisy)
                        # Label smoothing
                        real_label = torch.full_like(pred_real, real_label_val)
                        fake_label = torch.full_like(pred_real, fake_label_val)

                        loss_d_real = self.criterion_gan(pred_real, real_label)
                        pred_fake = self.discriminator(source, fake_noisy)
                        loss_d_fake = self.criterion_gan(pred_fake, fake_label)

                    # Total discriminator loss
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

                if self.use_amp:
                    self.scaler_d.scale(loss_d).backward()
                    self.scaler_d.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_d)
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_d)
                    self.optimizer_d.step()
            else:
                # Still compute D loss for logging but don't update
                with torch.no_grad():
                    fake = self.generator(source)
                    if self.use_multiscale_disc:
                        pred_real_list = self.discriminator(source, target)
                        pred_fake_list = self.discriminator(source, fake)
                        loss_d = 0
                        for pred_real, pred_fake in zip(pred_real_list, pred_fake_list):
                            real_label = torch.full_like(pred_real, real_label_val)
                            fake_label = torch.full_like(pred_fake, fake_label_val)
                            loss_d += self.criterion_gan(pred_real, real_label)
                            loss_d += self.criterion_gan(pred_fake, fake_label)
                        loss_d = loss_d / (2 * len(pred_real_list))
                    else:
                        pred_real = self.discriminator(source, target)
                        pred_fake = self.discriminator(source, fake)
                        real_label = torch.full_like(pred_real, real_label_val)
                        fake_label = torch.full_like(pred_fake, fake_label_val)
                        loss_d = (self.criterion_gan(pred_real, real_label) +
                                  self.criterion_gan(pred_fake, fake_label)) * 0.5

            # ==================
            # Train Generator
            # ==================
            self.optimizer_g.zero_grad()

            with autocast(enabled=self.use_amp):
                # Generate fake images (need to regenerate for generator update)
                fake = self.generator(source)

                # Adversarial loss (multi-scale or single-scale)
                if self.use_multiscale_disc:
                    pred_fake_list = self.discriminator(source, fake)
                    loss_g_adv = 0
                    for pred_fake in pred_fake_list:
                        real_label_g = torch.ones_like(pred_fake)
                        loss_g_adv += self.criterion_gan(pred_fake, real_label_g)
                    loss_g_adv /= len(pred_fake_list)
                else:
                    pred_fake = self.discriminator(source, fake)
                    real_label_g = torch.ones_like(pred_fake)
                    loss_g_adv = self.criterion_gan(pred_fake, real_label_g)

                # L1 loss
                loss_g_l1 = self.criterion_l1(fake, target)

                # Perceptual loss (VGG)
                loss_g_perceptual = self.criterion_perceptual(fake, target)

                # LPIPS loss (if available)
                loss_g_lpips = torch.tensor(0.0, device=self.device)
                if self.criterion_lpips is not None:
                    loss_g_lpips = self.criterion_lpips(fake, target)

                # LAB color loss
                loss_g_lab = torch.tensor(0.0, device=self.device)
                if self.criterion_lab is not None:
                    loss_g_lab = self.criterion_lab(fake, target)

                # Histogram loss
                loss_g_hist = torch.tensor(0.0, device=self.device)
                if self.criterion_hist is not None:
                    loss_g_hist = self.criterion_hist(fake, target)

                # Total generator loss
                loss_g = (
                    self.lambda_adv * loss_g_adv +
                    self.lambda_l1 * loss_g_l1 +
                    self.lambda_perceptual * loss_g_perceptual +
                    self.lambda_lpips * loss_g_lpips +
                    self.lambda_lab * loss_g_lab +
                    self.lambda_hist * loss_g_hist
                )

            if self.use_amp:
                self.scaler_g.scale(loss_g).backward()
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_g)
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_g)
                self.optimizer_g.step()

            # Decay instance noise
            self.current_noise *= self.instance_noise_decay

            # Update metrics
            epoch_losses['g_total'] += loss_g.item()
            epoch_losses['g_adv'] += loss_g_adv.item()
            epoch_losses['g_l1'] += loss_g_l1.item()
            epoch_losses['g_perceptual'] += loss_g_perceptual.item()
            epoch_losses['g_lpips'] += loss_g_lpips.item()
            epoch_losses['g_lab'] += loss_g_lab.item()
            epoch_losses['g_hist'] += loss_g_hist.item()
            epoch_losses['d_total'] += loss_d.item()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'G': f"{loss_g.item():.3f}",
                'D': f"{loss_d.item():.3f}",
                'L1': f"{loss_g_l1.item():.3f}",
                'LPIPS': f"{loss_g_lpips.item():.3f}",
                'LAB': f"{loss_g_lab.item():.3f}",
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
            print(f"  Train - G: {train_losses['g_total']:.4f}, D: {train_losses['d_total']:.4f}")
            print(f"  Losses - L1: {train_losses['g_l1']:.4f}, VGG: {train_losses['g_perceptual']:.4f}, "
                  f"LPIPS: {train_losses['g_lpips']:.4f}, LAB: {train_losses['g_lab']:.4f}, Hist: {train_losses['g_hist']:.4f}")
            print(f"  Val - L1: {val_loss:.4f} {'(best)' if is_best else ''}")
            print(f"  LR - G: {self.scheduler_g.get_last_lr()[0]:.2e}, D: {self.scheduler_d.get_last_lr()[0]:.2e}")

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
                        help="VGG perceptual loss weight")
    parser.add_argument("--lambda_lpips", type=float, default=5.0,
                        help="LPIPS perceptual loss weight")
    parser.add_argument("--lambda_lab", type=float, default=10.0,
                        help="LAB color loss weight")
    parser.add_argument("--lambda_hist", type=float, default=1.0,
                        help="Color histogram loss weight")
    parser.add_argument("--lambda_adv", type=float, default=1.0,
                        help="Adversarial loss weight")

    # Discriminator options
    parser.add_argument("--no_multiscale_disc", action="store_true",
                        help="Disable multi-scale discriminator")
    parser.add_argument("--num_disc_scales", type=int, default=2,
                        help="Number of discriminator scales")
    parser.add_argument("--use_spectral_norm", action="store_true",
                        help="Use Spectral Normalization in discriminator (improves stability)")

    # GAN stabilization arguments
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for discriminator (real=1-x, fake=x)")
    parser.add_argument("--instance_noise", type=float, default=0.1,
                        help="Initial instance noise std for D inputs")
    parser.add_argument("--instance_noise_decay", type=float, default=0.9995,
                        help="Instance noise decay per step")
    parser.add_argument("--grad_clip_g", type=float, default=1.0,
                        help="Gradient clipping for generator")
    parser.add_argument("--grad_clip_d", type=float, default=1.0,
                        help="Gradient clipping for discriminator")
    parser.add_argument("--d_update_freq", type=int, default=1,
                        help="Update discriminator every N steps")

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
        lambda_lpips=args.lambda_lpips,
        lambda_lab=args.lambda_lab,
        lambda_hist=args.lambda_hist,
        lambda_adv=args.lambda_adv,
        use_multiscale_disc=not args.no_multiscale_disc,
        num_disc_scales=args.num_disc_scales,
        use_spectral_norm=args.use_spectral_norm,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        resume_checkpoint=args.resume,
        # GAN stabilization
        label_smoothing=args.label_smoothing,
        instance_noise=args.instance_noise,
        instance_noise_decay=args.instance_noise_decay,
        grad_clip_g=args.grad_clip_g,
        grad_clip_d=args.grad_clip_d,
        d_update_freq=args.d_update_freq,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
