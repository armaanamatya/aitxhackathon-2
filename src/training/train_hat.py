"""
Training Script for HAT (Hybrid Attention Transformer) Image Enhancement

Features:
- HAT architecture with Window Attention + Channel Attention + Overlapping Cross-Attention
- Multiple loss functions for color accuracy
- EMA for stable outputs
- Mixed precision training
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import get_dataloaders
from src.training.hat import hat_small, hat_base, hat_large, count_parameters
from src.training.models import (
    VGGPerceptualLoss,
    LPIPSLoss,
    LABColorLoss,
    ColorHistogramLoss,
    SSIMLoss,
    FFTLoss,
    CharbonnierLoss,
    LPIPS_AVAILABLE,
)


class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class HATTrainer:
    """Trainer for HAT Image Enhancement."""

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        model_size: str = "base",
        image_size: int = 512,
        batch_size: int = 4,
        num_epochs: int = 200,
        lr: float = 2e-4,
        # Loss weights
        lambda_charbonnier: float = 1.0,
        lambda_ssim: float = 0.1,
        lambda_fft: float = 0.05,
        lambda_perceptual: float = 0.1,
        lambda_lpips: float = 0.1,
        lambda_lab: float = 0.1,
        lambda_hist: float = 0.05,
        # EMA
        ema_decay: float = 0.999,
        use_ema: bool = True,
        # Training
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

        self.model_size = model_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Loss weights
        self.lambda_charbonnier = lambda_charbonnier
        self.lambda_ssim = lambda_ssim
        self.lambda_fft = lambda_fft
        self.lambda_perceptual = lambda_perceptual
        self.lambda_lpips = lambda_lpips
        self.lambda_lab = lambda_lab
        self.lambda_hist = lambda_hist

        self.use_ema = use_ema
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_interval = save_interval
        self.sample_interval = sample_interval

        # Output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Data
        print("Loading datasets...")
        self.train_loader, self.val_loader = get_dataloaders(
            data_root=data_root,
            jsonl_path=jsonl_path,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            train_ratio=0.9,
        )
        print(f"Loaded {len(self.train_loader.dataset)} train samples")
        print(f"Loaded {len(self.val_loader.dataset)} val samples")

        # Model
        print(f"Initializing HAT-{model_size}...")
        if model_size == "small":
            self.model = hat_small().to(self.device)
        elif model_size == "base":
            self.model = hat_base().to(self.device)
        elif model_size == "large":
            self.model = hat_large().to(self.device)
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        print(f"Model parameters: {count_parameters(self.model):,}")

        # EMA
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            print(f"Using EMA with decay {ema_decay}")
        else:
            self.ema = None

        # Loss functions
        self.criterion_charbonnier = CharbonnierLoss().to(self.device)
        self.criterion_ssim = SSIMLoss().to(self.device)
        self.criterion_fft = FFTLoss().to(self.device)
        self.criterion_perceptual = VGGPerceptualLoss().to(self.device)

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
            weight_decay=0.02,
        )

        # Scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )

        # AMP
        self.scaler = GradScaler() if self.use_amp else None

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()

        epoch_losses = {
            'total': 0.0, 'charbonnier': 0.0, 'ssim': 0.0, 'fft': 0.0,
            'perceptual': 0.0, 'lpips': 0.0, 'lab': 0.0, 'hist': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output = self.model(source)

                loss_charbonnier = self.criterion_charbonnier(output, target)
                loss_ssim = self.criterion_ssim(output, target)
                loss_fft = self.criterion_fft(output, target)
                loss_perceptual = self.criterion_perceptual(output, target)

                loss_lpips = torch.tensor(0.0, device=self.device)
                if self.criterion_lpips is not None:
                    loss_lpips = self.criterion_lpips(output, target)

                loss_lab = torch.tensor(0.0, device=self.device)
                if self.criterion_lab is not None:
                    loss_lab = self.criterion_lab(output, target)

                loss_hist = torch.tensor(0.0, device=self.device)
                if self.criterion_hist is not None:
                    loss_hist = self.criterion_hist(output, target)

                loss = (
                    self.lambda_charbonnier * loss_charbonnier +
                    self.lambda_ssim * loss_ssim +
                    self.lambda_fft * loss_fft +
                    self.lambda_perceptual * loss_perceptual +
                    self.lambda_lpips * loss_lpips +
                    self.lambda_lab * loss_lab +
                    self.lambda_hist * loss_hist
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update()

            epoch_losses['total'] += loss.item()
            epoch_losses['charbonnier'] += loss_charbonnier.item()
            epoch_losses['ssim'] += loss_ssim.item()
            epoch_losses['fft'] += loss_fft.item()
            epoch_losses['perceptual'] += loss_perceptual.item()
            epoch_losses['lpips'] += loss_lpips.item()
            epoch_losses['lab'] += loss_lab.item()
            epoch_losses['hist'] += loss_hist.item()

            self.global_step += 1

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LPIPS': f"{loss_lpips.item():.4f}",
                'LAB': f"{loss_lab.item():.4f}",
            })

        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self, use_ema: bool = True) -> float:
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        total_l1_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            output = self.model(source)
            loss = F.l1_loss(output, target)
            total_l1_loss += loss.item()
            num_batches += 1

        if use_ema and self.ema is not None:
            self.ema.restore()

        return total_l1_loss / num_batches

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()

        batch = next(iter(self.val_loader))
        source = batch['source'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)
        output = self.model(source)

        if self.ema is not None:
            self.ema.restore()

        actual_samples = min(num_samples, source.size(0))

        def tensor_to_image(tensor):
            tensor = (tensor + 1) / 2
            tensor = tensor.clamp(0, 1)
            tensor = tensor.cpu().numpy().transpose(1, 2, 0)
            tensor = (tensor * 255).astype(np.uint8)
            return Image.fromarray(tensor)

        for i in range(actual_samples):
            src_img = tensor_to_image(source[i])
            out_img = tensor_to_image(output[i])
            tar_img = tensor_to_image(target[i])

            width = src_img.width
            height = src_img.height
            comparison = Image.new('RGB', (width * 3, height))
            comparison.paste(src_img, (0, 0))
            comparison.paste(out_img, (width, 0))
            comparison.paste(tar_img, (width * 2, 0))

            comparison.save(self.sample_dir / f"epoch_{epoch:04d}_sample_{i}.jpg")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
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

        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow

        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        if (epoch + 1) % self.save_interval == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            if self.ema is not None:
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model_ema.pt")
                self.ema.restore()
            else:
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']

        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        print(f"\nStarting HAT training from epoch {self.start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        print(f"Model: HAT-{self.model_size}")
        print(f"Using AMP: {self.use_amp}")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.num_epochs):
            train_losses = self.train_epoch(epoch)
            val_loss = self.validate(use_ema=True)
            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Char: {train_losses['charbonnier']:.4f}, "
                  f"SSIM: {train_losses['ssim']:.4f}, "
                  f"LPIPS: {train_losses['lpips']:.4f}, "
                  f"LAB: {train_losses['lab']:.4f}")
            print(f"  Val - L1: {val_loss:.4f} {'(best)' if is_best else ''}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            if (epoch + 1) % self.sample_interval == 0:
                self.save_samples(epoch)

            self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best validation L1 loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train HAT for Image Enhancement")

    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs_hat")

    parser.add_argument("--model_size", type=str, default="base",
                        choices=["small", "base", "large"])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--lambda_charbonnier", type=float, default=1.0)
    parser.add_argument("--lambda_ssim", type=float, default=0.1)
    parser.add_argument("--lambda_fft", type=float, default=0.05)
    parser.add_argument("--lambda_perceptual", type=float, default=0.1)
    parser.add_argument("--lambda_lpips", type=float, default=0.1)
    parser.add_argument("--lambda_lab", type=float, default=0.1)
    parser.add_argument("--lambda_hist", type=float, default=0.05)

    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--no_ema", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=5)

    args = parser.parse_args()

    trainer = HATTrainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_charbonnier=args.lambda_charbonnier,
        lambda_ssim=args.lambda_ssim,
        lambda_fft=args.lambda_fft,
        lambda_perceptual=args.lambda_perceptual,
        lambda_lpips=args.lambda_lpips,
        lambda_lab=args.lambda_lab,
        lambda_hist=args.lambda_hist,
        ema_decay=args.ema_decay,
        use_ema=not args.no_ema,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        resume_checkpoint=args.resume,
    )

    trainer.train()


if __name__ == "__main__":
    main()
