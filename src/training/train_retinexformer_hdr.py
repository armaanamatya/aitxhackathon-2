"""
Retinexformer Training with HDR Losses for Real Estate Photo Enhancement

Why Retinexformer for HDR/Window Preservation:
1. Physics-based - Retinex theory separates illumination from reflectance
2. Illumination-guided attention - naturally handles bright/dark regions differently
3. Explicit illumination estimation - learns to identify windows vs interior
4. ICCV 2023 + ECCV 2024 - proven architecture with strong results

Architecture advantage for windows:
- Illumination Estimator learns: "this is a window (bright), this is interior (dark)"
- IG-MSA (attention) uses this to process them differently
- Result: Windows preserved while shadows lifted
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

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
from src.training.retinexformer import RetinexFormer, create_retinexformer, count_parameters
from src.training.hdr_losses import HDRLoss, compute_psnr, compute_ssim
from src.training.models import VGGPerceptualLoss, LPIPSLoss, LPIPS_AVAILABLE


class RetinexformerHDRTrainer:
    """
    Trainer for Retinexformer with HDR-specific losses.

    Key features:
    - HDR losses for window/highlight preservation
    - PSNR/SSIM tracking with early stopping
    - Warmup + cosine annealing LR
    - Gradient accumulation for effective larger batches
    - Automatic mixed precision training
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        model_size: str = "base",
        image_size: int = 512,
        batch_size: int = 2,
        gradient_accumulation: int = 4,
        num_epochs: int = 100,
        lr: float = 2e-4,
        warmup_epochs: int = 5,
        early_stopping_patience: int = 20,
        # Standard losses
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.05,
        # HDR-specific losses
        lambda_gradient: float = 0.15,
        lambda_highlight: float = 0.25,  # Higher for Retinexformer
        lambda_laplacian: float = 0.1,
        lambda_ssim: float = 0.1,
        highlight_threshold: float = 0.3,
        # Training settings
        use_amp: bool = True,
        num_workers: int = 8,
        save_every: int = 10,
        sample_every: int = 5,
        resume_checkpoint: str = None,
        seed: int = 42,
    ):
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Settings
        self.model_size = model_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.effective_batch_size = batch_size * gradient_accumulation
        self.num_epochs = num_epochs
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_every = save_every
        self.sample_every = sample_every

        # Loss weights
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_gradient = lambda_gradient
        self.lambda_highlight = lambda_highlight
        self.lambda_laplacian = lambda_laplacian
        self.lambda_ssim = lambda_ssim

        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)

        # Save config
        self.config = {
            "model": "Retinexformer",
            "model_size": model_size,
            "image_size": image_size,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "warmup_epochs": warmup_epochs,
            "lambda_l1": lambda_l1,
            "lambda_vgg": lambda_vgg,
            "lambda_lpips": lambda_lpips,
            "lambda_gradient": lambda_gradient,
            "lambda_highlight": lambda_highlight,
            "lambda_laplacian": lambda_laplacian,
            "lambda_ssim": lambda_ssim,
            "highlight_threshold": highlight_threshold,
        }
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

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
        print(f"  Train: {len(self.train_loader.dataset)} samples")
        print(f"  Val: {len(self.val_loader.dataset)} samples")

        # Model
        print(f"\nInitializing Retinexformer-{model_size}...")
        self.model = create_retinexformer(model_size).to(self.device)
        num_params = count_parameters(self.model)
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        # HDR Loss
        print("\nInitializing losses...")
        self.hdr_loss = HDRLoss(
            lambda_l1=lambda_l1,
            lambda_gradient=lambda_gradient,
            lambda_highlight=lambda_highlight,
            lambda_laplacian=lambda_laplacian,
            lambda_local_contrast=0.05,
            lambda_ssim=lambda_ssim,
            highlight_threshold=highlight_threshold,
        ).to(self.device)
        print("  HDR losses: gradient, highlight, laplacian, ssim ✓")

        # VGG Perceptual Loss
        self.vgg_loss = VGGPerceptualLoss().to(self.device)
        print("  VGG perceptual loss ✓")

        # LPIPS Loss (optional)
        self.lpips_loss = None
        if LPIPS_AVAILABLE and lambda_lpips > 0:
            self.lpips_loss = LPIPSLoss(net='alex').to(self.device)
            print("  LPIPS loss ✓")

        # Optimizer - AdamW with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # LR Scheduler with warmup + cosine decay
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
                return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_psnr = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = []

        # Resume
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0.0, 'l1': 0.0, 'gradient': 0.0, 'highlight': 0.0,
            'laplacian': 0.0, 'ssim': 0.0, 'vgg': 0.0, 'lpips': 0.0,
        }
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            with autocast(enabled=self.use_amp):
                # Forward pass
                output = self.model(source)

                # HDR losses
                hdr_total, hdr_components = self.hdr_loss(
                    output, target, return_components=True
                )

                # VGG perceptual loss
                vgg = self.vgg_loss(output, target)

                # LPIPS loss
                lpips = torch.tensor(0.0, device=self.device)
                if self.lpips_loss is not None:
                    lpips = self.lpips_loss(output, target)

                # Total loss
                loss = hdr_total + self.lambda_vgg * vgg + self.lambda_lpips * lpips
                loss = loss / self.gradient_accumulation

            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient step
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            epoch_losses['total'] += loss.item() * self.gradient_accumulation
            epoch_losses['l1'] += hdr_components['l1'].item()
            epoch_losses['gradient'] += hdr_components['gradient'].item()
            epoch_losses['highlight'] += hdr_components['highlight'].item()
            epoch_losses['laplacian'] += hdr_components['laplacian'].item()
            epoch_losses['ssim'] += hdr_components['ssim'].item()
            epoch_losses['vgg'] += vgg.item()
            epoch_losses['lpips'] += lpips.item()
            num_batches += 1

            # Progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item() * self.gradient_accumulation:.4f}",
                'HL': f"{hdr_components['highlight'].item():.4f}",
                'Grad': f"{hdr_components['gradient'].item():.4f}",
            })

        # Average
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(source)

            loss = self.hdr_loss(output, target)
            total_loss += loss.item()

            total_psnr += compute_psnr(output, target)
            total_ssim += compute_ssim(output, target)
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches,
        }

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save visual samples."""
        self.model.eval()

        batch = next(iter(self.val_loader))
        source = batch['source'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)
        output = self.model(source)

        def to_pil(t):
            t = (t + 1) / 2
            t = t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            return Image.fromarray((t * 255).astype(np.uint8))

        for i in range(min(num_samples, source.size(0))):
            src = to_pil(source[i])
            out = to_pil(output[i])
            tar = to_pil(target[i])

            w, h = src.size
            comp = Image.new('RGB', (w * 3, h))
            comp.paste(src, (0, 0))
            comp.paste(out, (w, 0))
            comp.paste(tar, (w * 2, 0))
            comp.save(self.output_dir / "samples" / f"epoch_{epoch:04d}_{i}.jpg")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_psnr': self.best_val_psnr,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")

        if (epoch + 1) % self.save_every == 0:
            torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])

        self.start_epoch = ckpt['epoch'] + 1
        self.global_step = ckpt['global_step']
        self.best_val_psnr = ckpt.get('best_val_psnr', 0.0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = ckpt.get('history', [])

        print(f"  Resumed from epoch {self.start_epoch}, best PSNR: {self.best_val_psnr:.2f} dB")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("Retinexformer HDR Training")
        print("=" * 70)
        print(f"  Model: Retinexformer-{self.model_size}")
        print(f"  Resolution: {self.image_size}x{self.image_size}")
        print(f"  Batch: {self.batch_size} x {self.gradient_accumulation} = {self.effective_batch_size}")
        print(f"  Epochs: {self.num_epochs} (early stop: {self.early_stopping_patience})")
        print(f"  LR: {self.lr} (warmup: {self.warmup_epochs})")
        print(f"  HDR losses: highlight={self.lambda_highlight}, gradient={self.lambda_gradient}")
        print(f"  Output: {self.output_dir}")
        print("=" * 70 + "\n")

        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Check improvement
            is_best = val_metrics['psnr'] > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_metrics['psnr']
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Log
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train: loss={train_losses['total']:.4f}, "
                  f"L1={train_losses['l1']:.4f}, "
                  f"HL={train_losses['highlight']:.4f}, "
                  f"Grad={train_losses['gradient']:.4f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"PSNR={val_metrics['psnr']:.2f}dB, "
                  f"SSIM={val_metrics['ssim']:.4f} "
                  f"{'🏆 BEST' if is_best else ''}")
            print(f"  LR: {current_lr:.2e}")

            # History
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_losses['total'],
                'val_loss': val_metrics['loss'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
                'lr': current_lr,
            })

            # Save samples
            if (epoch + 1) % self.sample_every == 0:
                self.save_samples(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⏹️  Early stopping after {self.epochs_without_improvement} epochs without improvement")
                break

        # Save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"  Best PSNR: {self.best_val_psnr:.2f} dB")
        print(f"  Best Loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train Retinexformer with HDR Losses")

    # Data
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs_retinexformer_hdr")

    # Model
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["tiny", "small", "base", "large"])

    # Training
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=20)

    # Losses
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_vgg", type=float, default=0.1)
    parser.add_argument("--lambda_lpips", type=float, default=0.05)
    parser.add_argument("--lambda_gradient", type=float, default=0.15)
    parser.add_argument("--lambda_highlight", type=float, default=0.25)
    parser.add_argument("--lambda_laplacian", type=float, default=0.1)
    parser.add_argument("--lambda_ssim", type=float, default=0.1)
    parser.add_argument("--highlight_threshold", type=float, default=0.3)

    # Other
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    trainer = RetinexformerHDRTrainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_epochs=args.num_epochs,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_vgg,
        lambda_lpips=args.lambda_lpips,
        lambda_gradient=args.lambda_gradient,
        lambda_highlight=args.lambda_highlight,
        lambda_laplacian=args.lambda_laplacian,
        lambda_ssim=args.lambda_ssim,
        highlight_threshold=args.highlight_threshold,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        save_every=args.save_every,
        sample_every=args.sample_every,
        resume_checkpoint=args.resume,
        seed=args.seed,
    )

    trainer.train()


if __name__ == "__main__":
    main()
