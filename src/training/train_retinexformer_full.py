"""
Retinexformer-Large Full Training Script with Optimal HDR Losses

Data Split (NO LEAKAGE):
- Test: 10 samples (HELD OUT - never seen during training)
- Train: 511 samples (90%)
- Val: 56 samples (10%)

Losses (HDR-Optimized for Window Preservation):
- L1: Pixel accuracy
- VGG: Perceptual quality
- LPIPS: Learned perceptual similarity
- Gradient: Edge preservation (prevents cracks)
- Highlight: Window/bright region preservation
- Laplacian: Multi-scale edge preservation
- SSIM: Structural similarity
- Local Contrast: Preserves local structure

Architecture: Retinexformer-Large (3.7M params)
- Illumination Estimator: Learns bright vs dark regions
- IG-MSA: Illumination-guided attention
- Physics-based: Retinex theory separation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.retinexformer import RetinexFormer, create_retinexformer, count_parameters
from src.training.hdr_losses import HDRLoss, compute_psnr, compute_ssim
from src.training.models import VGGPerceptualLoss, LPIPSLoss, LPIPS_AVAILABLE


# =============================================================================
# Dataset with Proper Splits (No Leakage)
# =============================================================================

class HDRDatasetFromSplit(Dataset):
    """
    Dataset that loads from pre-split JSONL files.
    Ensures no data leakage between train/val/test.
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        image_size: int = 512,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment

        # Load pairs from JSONL
        self.pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

        # Transforms
        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.LANCZOS,
            antialias=True
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        print(f"  Loaded {len(self.pairs)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.pairs)

    def _augment(self, src: Image.Image, tar: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Synchronized augmentation."""
        # Horizontal flip (50%)
        if np.random.random() > 0.5:
            src = transforms.functional.hflip(src)
            tar = transforms.functional.hflip(tar)

        # Vertical flip (10%)
        if np.random.random() > 0.9:
            src = transforms.functional.vflip(src)
            tar = transforms.functional.vflip(tar)

        # Small rotation (30%, ±5°)
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            src = transforms.functional.rotate(src, angle)
            tar = transforms.functional.rotate(tar, angle)

        # Color jitter (20%, very subtle for HDR)
        if np.random.random() > 0.8:
            jitter = transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05
            )
            # Only apply to source (simulate different capture conditions)
            src = jitter(src)

        return src, tar

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        src_path = self.data_root / pair['src']
        tar_path = self.data_root / pair['tar']

        src_img = Image.open(src_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Resize
        src_img = self.resize(src_img)
        tar_img = self.resize(tar_img)

        # Augment (training only)
        if self.augment:
            src_img, tar_img = self._augment(src_img, tar_img)

        # To tensor and normalize
        src = self.normalize(self.to_tensor(src_img))
        tar = self.normalize(self.to_tensor(tar_img))

        return {
            'source': src,
            'target': tar,
            'source_path': str(src_path),
            'target_path': str(tar_path),
        }


# =============================================================================
# Complete HDR Loss with All Components
# =============================================================================

class ComprehensiveHDRLoss(nn.Module):
    """
    Complete loss for HDR enhancement with window preservation.

    Components:
    - L1: Pixel accuracy (weight: 1.0)
    - VGG: Perceptual (weight: 0.1)
    - LPIPS: Learned perceptual (weight: 0.05)
    - Gradient: Edge preservation (weight: 0.15)
    - Highlight: Window preservation (weight: 0.25)
    - Laplacian: Multi-scale edges (weight: 0.1)
    - SSIM: Structure (weight: 0.1)
    - Local Contrast: Local structure (weight: 0.05)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.05,
        lambda_gradient: float = 0.15,
        lambda_highlight: float = 0.25,
        lambda_laplacian: float = 0.1,
        lambda_ssim: float = 0.1,
        lambda_local_contrast: float = 0.05,
        highlight_threshold: float = 0.3,
        device: str = 'cuda',
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_gradient = lambda_gradient
        self.lambda_highlight = lambda_highlight
        self.lambda_laplacian = lambda_laplacian
        self.lambda_ssim = lambda_ssim
        self.lambda_local_contrast = lambda_local_contrast
        self.highlight_threshold = highlight_threshold

        # HDR losses
        self.hdr_loss = HDRLoss(
            lambda_l1=lambda_l1,
            lambda_gradient=lambda_gradient,
            lambda_highlight=lambda_highlight,
            lambda_laplacian=lambda_laplacian,
            lambda_local_contrast=lambda_local_contrast,
            lambda_ssim=lambda_ssim,
            highlight_threshold=highlight_threshold,
        ).to(device)

        # VGG perceptual
        self.vgg_loss = VGGPerceptualLoss().to(device)

        # LPIPS
        self.lpips_loss = None
        if LPIPS_AVAILABLE and lambda_lpips > 0:
            self.lpips_loss = LPIPSLoss(net='alex').to(device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:

        # HDR losses
        hdr_total, hdr_components = self.hdr_loss(pred, target, return_components=True)

        # VGG
        vgg = self.vgg_loss(pred, target)

        # LPIPS
        lpips = torch.tensor(0.0, device=pred.device)
        if self.lpips_loss is not None:
            lpips = self.lpips_loss(pred, target)

        # Total
        total = hdr_total + self.lambda_vgg * vgg + self.lambda_lpips * lpips

        if return_components:
            components = {
                **hdr_components,
                'vgg': vgg,
                'lpips': lpips,
            }
            return total, components

        return total


# =============================================================================
# Trainer
# =============================================================================

class RetinexformerFullTrainer:
    """
    Complete trainer for Retinexformer-Large with optimal HDR losses.

    Features:
    - No data leakage (10 test held out)
    - 90:10 train:val split
    - All HDR losses for window preservation
    - PSNR/SSIM tracking
    - Early stopping on val PSNR
    - Warmup + cosine LR schedule
    - Gradient accumulation
    - Mixed precision training
    """

    def __init__(
        self,
        data_root: str,
        splits_dir: str,
        output_dir: str,
        fold: int = 1,
        model_size: str = "large",
        image_size: int = 512,
        batch_size: int = 2,
        gradient_accumulation: int = 4,
        num_epochs: int = 100,
        lr: float = 2e-4,
        warmup_epochs: int = 5,
        early_stopping_patience: int = 25,
        # Loss weights
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.05,
        lambda_gradient: float = 0.15,
        lambda_highlight: float = 0.25,
        lambda_laplacian: float = 0.1,
        lambda_ssim: float = 0.1,
        highlight_threshold: float = 0.3,
        # Other
        use_amp: bool = True,
        num_workers: int = 8,
        save_every: int = 10,
        sample_every: int = 5,
        resume: str = None,
        seed: int = 42,
    ):
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n{'='*70}")
        print(f"Retinexformer-{model_size.upper()} Full Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")

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
        self.fold = fold

        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)

        # Save config
        self.config = {
            "model": "Retinexformer",
            "model_size": model_size,
            "fold": fold,
            "image_size": image_size,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "warmup_epochs": warmup_epochs,
            "early_stopping_patience": early_stopping_patience,
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

        # Data loaders (from pre-split files)
        print(f"\nLoading data splits from {splits_dir}/fold_{fold}/")
        train_jsonl = Path(splits_dir) / f"fold_{fold}" / "train.jsonl"
        val_jsonl = Path(splits_dir) / f"fold_{fold}" / "val.jsonl"
        test_jsonl = Path(splits_dir) / "test.jsonl"

        print("  Train set:")
        train_dataset = HDRDatasetFromSplit(
            data_root=data_root, jsonl_path=train_jsonl,
            image_size=image_size, augment=True
        )
        print("  Val set:")
        val_dataset = HDRDatasetFromSplit(
            data_root=data_root, jsonl_path=val_jsonl,
            image_size=image_size, augment=False
        )
        print("  Test set (HELD OUT):")
        self.test_dataset = HDRDatasetFromSplit(
            data_root=data_root, jsonl_path=test_jsonl,
            image_size=image_size, augment=False
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            num_workers=2, pin_memory=True
        )

        # Model
        print(f"\nInitializing Retinexformer-{model_size}...")
        self.model = create_retinexformer(model_size).to(self.device)
        num_params = count_parameters(self.model)
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        # Loss
        print("\nInitializing losses...")
        self.criterion = ComprehensiveHDRLoss(
            lambda_l1=lambda_l1,
            lambda_vgg=lambda_vgg,
            lambda_lpips=lambda_lpips,
            lambda_gradient=lambda_gradient,
            lambda_highlight=lambda_highlight,
            lambda_laplacian=lambda_laplacian,
            lambda_ssim=lambda_ssim,
            highlight_threshold=highlight_threshold,
            device=self.device,
        )
        print(f"  L1: {lambda_l1}, VGG: {lambda_vgg}, LPIPS: {lambda_lpips}")
        print(f"  Gradient: {lambda_gradient}, Highlight: {lambda_highlight}")
        print(f"  Laplacian: {lambda_laplacian}, SSIM: {lambda_ssim}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # LR scheduler with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
                return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_psnr = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = []

        # Resume
        if resume:
            self.load_checkpoint(resume)

        print(f"\n{'='*70}")
        print("Configuration Summary:")
        print(f"  Model: Retinexformer-{model_size} ({num_params/1e6:.2f}M params)")
        print(f"  Resolution: {image_size}x{image_size}")
        print(f"  Batch: {batch_size} x {gradient_accumulation} = {self.effective_batch_size}")
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(self.test_dataset)}")
        print(f"  LR: {lr} (warmup: {warmup_epochs} epochs)")
        print(f"  Early stopping: {early_stopping_patience} epochs")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*70}\n")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0, 'l1': 0, 'gradient': 0, 'highlight': 0,
            'laplacian': 0, 'ssim': 0, 'vgg': 0, 'lpips': 0,
        }
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(source)
                loss, components = self.criterion(output, target, return_components=True)
                loss = loss / self.gradient_accumulation

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

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

            # Accumulate
            epoch_losses['total'] += loss.item() * self.gradient_accumulation
            for k, v in components.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            num_batches += 1

            pbar.set_postfix({
                'Loss': f"{loss.item() * self.gradient_accumulation:.4f}",
                'HL': f"{components['highlight'].item():.4f}",
            })

        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on val set."""
        self.model.eval()

        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        n = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(source)
            loss = self.criterion(output, target)

            total_loss += loss.item()
            total_psnr += compute_psnr(output, target)
            total_ssim += compute_ssim(output, target)
            n += 1

        return {
            'loss': total_loss / n,
            'psnr': total_psnr / n,
            'ssim': total_ssim / n,
        }

    @torch.no_grad()
    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on held-out test set."""
        self.model.eval()

        total_psnr = 0
        total_ssim = 0
        n = 0

        for batch in self.test_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(source)

            total_psnr += compute_psnr(output, target)
            total_ssim += compute_ssim(output, target)
            n += 1

        return {
            'psnr': total_psnr / n,
            'ssim': total_ssim / n,
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
        ckpt = {
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
            ckpt['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(ckpt, self.output_dir / "checkpoint_latest.pt")

        if (epoch + 1) % self.save_every == 0:
            torch.save(ckpt, self.output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        if is_best:
            torch.save(ckpt, self.output_dir / "checkpoint_best.pt")

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
        self.best_val_psnr = ckpt.get('best_val_psnr', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = ckpt.get('history', [])

        print(f"  Resumed from epoch {self.start_epoch}, best PSNR: {self.best_val_psnr:.2f}")

    def train(self):
        """Main training loop."""
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Scheduler step
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

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
                  f"HL={train_losses['highlight']:.4f}, "
                  f"Grad={train_losses['gradient']:.4f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"PSNR={val_metrics['psnr']:.2f}dB, "
                  f"SSIM={val_metrics['ssim']:.4f} "
                  f"{'🏆 BEST' if is_best else ''}")
            print(f"  LR: {lr:.2e}")

            # History
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_losses['total'],
                'val_loss': val_metrics['loss'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
                'lr': lr,
            })

            # Samples
            if (epoch + 1) % self.sample_every == 0:
                self.save_samples(epoch)

            # Checkpoint
            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⏹️  Early stopping after {self.epochs_without_improvement} epochs")
                break

        # Final test evaluation
        print("\n" + "="*70)
        print("Evaluating on HELD-OUT TEST SET (10 samples)")
        print("="*70)

        # Load best model for test
        best_ckpt = self.output_dir / "checkpoint_best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded best model from epoch {ckpt['epoch']+1}")

        test_metrics = self.evaluate_test()

        print(f"\n📊 TEST RESULTS (NO DATA LEAKAGE):")
        print(f"   PSNR: {test_metrics['psnr']:.2f} dB")
        print(f"   SSIM: {test_metrics['ssim']:.4f}")

        # Save results
        results = {
            'best_val_psnr': self.best_val_psnr,
            'best_val_loss': self.best_val_loss,
            'test_psnr': test_metrics['psnr'],
            'test_ssim': test_metrics['ssim'],
            'history': self.history,
        }
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n✅ Training Complete!")
        print(f"   Best Val PSNR: {self.best_val_psnr:.2f} dB")
        print(f"   Test PSNR: {test_metrics['psnr']:.2f} dB")
        print(f"   Outputs: {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Retinexformer Full Training")

    # Data
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--splits_dir", type=str, default="data_splits")
    parser.add_argument("--output_dir", type=str, default="outputs_retinexformer_full")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3])

    # Model
    parser.add_argument("--model_size", type=str, default="large",
                        choices=["tiny", "small", "base", "large"])

    # Training
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=25)

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

    trainer = RetinexformerFullTrainer(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        fold=args.fold,
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
        resume=args.resume,
        seed=args.seed,
    )

    trainer.train()


if __name__ == "__main__":
    main()
