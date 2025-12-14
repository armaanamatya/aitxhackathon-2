#!/usr/bin/env python3
"""
Window-Aware Training for Real Estate HDR Enhancement
======================================================

Trains models with special focus on window transformation.
Uses region-aware losses to force the model to transform windows
instead of keeping them similar to source.

Key Features:
- Automatic window detection from source images
- Higher loss weight on window regions
- Color transformation loss for bold window changes
- Optional adversarial loss for realistic windows
- Compatible with Retinexformer, Restormer, NAFNet
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.window_aware_losses import (
    ComprehensiveWindowAwareLoss,
    create_window_aware_loss,
    WindowDetector
)
from src.training.highlight_aware_losses import (
    ComprehensiveHighlightLoss,
    create_highlight_aware_loss,
    HighlightDetector
)


# =============================================================================
# Dataset
# =============================================================================

class HDRDatasetWithSource(Dataset):
    """
    Dataset that returns source, target, AND source for window detection.
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

        # Load samples from JSONL
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

        # Transforms
        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.LANCZOS
        )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image and convert to tensor."""
        img = Image.open(self.data_root / path).convert('RGB')
        img = self.resize(img)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        source_path = sample.get('source') or sample.get('src') or sample.get('input')
        target_path = sample.get('target') or sample.get('tar') or sample.get('gt')

        source = self._load_image(source_path)
        target = self._load_image(target_path)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                source = torch.flip(source, dims=[2])
                target = torch.flip(target, dims=[2])

            # Random vertical flip (less common for real estate)
            if random.random() > 0.8:
                source = torch.flip(source, dims=[1])
                target = torch.flip(target, dims=[1])

        # Normalize to [-1, 1]
        source = source * 2 - 1
        target = target * 2 - 1

        return {
            'source': source,
            'target': target,
            'path': str(source_path)
        }


# =============================================================================
# Training Loop
# =============================================================================

class WindowAwareTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        args,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model = self.model.to(self.device)

        # Create loss function based on preset
        # 'aggressive' uses highlight-aware (based on error analysis: 46x error density)
        if args.loss_preset == 'aggressive':
            print("Using HIGHLIGHT-AWARE loss (based on error analysis)")
            self.criterion = create_highlight_aware_loss(preset='aggressive').to(self.device)
            self.highlight_detector = HighlightDetector().to(self.device)
        else:
            print(f"Using WINDOW-AWARE loss (preset: {args.loss_preset})")
            self.criterion = create_window_aware_loss(
                preset=args.loss_preset,
                use_adversarial=args.use_adversarial
            ).to(self.device)
            self.highlight_detector = WindowDetector().to(self.device)

        # Alias for compatibility
        self.window_detector = self.highlight_detector

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

        # LR Scheduler: warmup + cosine
        warmup_steps = len(train_loader) * args.warmup_epochs
        total_steps = len(train_loader) * args.num_epochs

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=args.lr * 0.01
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        # Mixed precision
        self.scaler = GradScaler() if args.use_amp else None

        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)

        # Tracking
        self.best_psnr = 0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_window_psnr': [],  # PSNR specifically on window regions
            'lr': []
        }

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR in dB."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        # Images in [-1, 1], so max value is 2
        return 10 * torch.log10(4 / mse).item()

    def compute_window_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> float:
        """Compute PSNR specifically on window regions."""
        window_mask = self.window_detector(source)

        if window_mask.sum() < 100:
            return float('nan')

        # Mask and compute
        pred_windows = pred * window_mask
        target_windows = target * window_mask

        mse = (pred_windows - target_windows).pow(2).sum() / (window_mask.sum() * 3 + 1e-8)
        if mse < 1e-10:
            return 100.0
        return 10 * torch.log10(4 / mse).item()

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SSIM."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, 3, 1, 1)
        mu_target = F.avg_pool2d(target, 3, 1, 1)

        sigma_pred = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target

        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

        return ssim.mean().item()

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {}

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    pred = self.model(source)
                    loss, components, _ = self.criterion(
                        pred, target, source, return_components=True
                    )

                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
            else:
                pred = self.model(source)
                loss, components, _ = self.criterion(
                    pred, target, source, return_components=True
                )
                loss.backward()

                if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

            total_loss += loss.item()

            # Track components
            for k, v in components.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v.item()

            # Update progress bar (handle both window and highlight loss components)
            highlight_key = 'highlight_l1' if 'highlight_l1' in components else 'window_l1'
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hl_l1': f'{components[highlight_key].item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_window_psnr = 0
        window_count = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            pred = self.model(source)
            pred = torch.clamp(pred, -1, 1)

            loss = self.criterion(pred, target, source)
            total_loss += loss.item()

            # Metrics
            total_psnr += self.compute_psnr(pred, target)
            total_ssim += self.compute_ssim(pred, target)

            # Window-specific PSNR
            window_psnr = self.compute_window_psnr(pred, target, source)
            if not np.isnan(window_psnr):
                total_window_psnr += window_psnr
                window_count += 1

        n = len(self.val_loader)
        return {
            'loss': total_loss / n,
            'psnr': total_psnr / n,
            'ssim': total_ssim / n,
            'window_psnr': total_window_psnr / max(window_count, 1)
        }

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample images for visual inspection."""
        self.model.eval()

        # Get a batch from validation
        batch = next(iter(self.val_loader))
        source = batch['source'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)

        pred = self.model(source)
        pred = torch.clamp(pred, -1, 1)

        # Get window masks
        window_mask = self.window_detector(source)

        # Convert to images
        def to_img(t):
            t = (t + 1) / 2  # [-1,1] -> [0,1]
            t = t.clamp(0, 1)
            return (t.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        sources = to_img(source)
        targets = to_img(target)
        preds = to_img(pred)
        masks = (window_mask.squeeze(1).cpu().numpy() * 255).astype(np.uint8)

        # Create comparison images
        for i in range(num_samples):
            # Create 4-panel comparison: source | pred | target | window_mask
            h, w = sources[i].shape[:2]

            # Convert mask to RGB (colorize)
            mask_rgb = np.stack([masks[i], np.zeros_like(masks[i]), np.zeros_like(masks[i])], axis=-1)

            comparison = np.concatenate([
                sources[i],
                preds[i],
                targets[i],
                mask_rgb
            ], axis=1)

            Image.fromarray(comparison).save(
                self.output_dir / 'samples' / f'epoch_{epoch:03d}_sample_{i}.jpg'
            )

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'history': self.history,
            'args': vars(self.args)
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')

        # Save periodic
        if epoch % self.args.save_every == 0:
            torch.save(
                checkpoint,
                self.output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch:03d}.pt'
            )

    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Window-Aware Training")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Loss preset: {self.args.loss_preset}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.args.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(self.scheduler.get_last_lr()[0])

            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['val_window_psnr'].append(val_metrics['window_psnr'])

            # Check for improvement
            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Print metrics
            print(f"\nEpoch {epoch}/{self.args.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB {'(BEST)' if is_best else ''}")
            print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Window PSNR: {val_metrics['window_psnr']:.2f} dB  <- Key metric!")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Save samples
            if epoch % self.args.sample_every == 0:
                self.save_samples(epoch)

            # Save history
            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

            # Early stopping
            if self.epochs_without_improvement >= self.args.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Final test evaluation
        print(f"\n{'='*60}")
        print("Final Test Evaluation")
        print(f"{'='*60}")
        self.evaluate_test()

    @torch.no_grad()
    def evaluate_test(self):
        """Evaluate on held-out test set."""
        # Load best model
        checkpoint = torch.load(self.output_dir / 'checkpoint_best.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        total_psnr = 0
        total_ssim = 0
        total_window_psnr = 0
        window_count = 0
        results = []

        for batch in tqdm(self.test_loader, desc='Testing'):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            paths = batch['path']

            pred = self.model(source)
            pred = torch.clamp(pred, -1, 1)

            for i in range(source.shape[0]):
                psnr = self.compute_psnr(pred[i:i+1], target[i:i+1])
                ssim = self.compute_ssim(pred[i:i+1], target[i:i+1])
                window_psnr = self.compute_window_psnr(
                    pred[i:i+1], target[i:i+1], source[i:i+1]
                )

                results.append({
                    'path': paths[i],
                    'psnr': psnr,
                    'ssim': ssim,
                    'window_psnr': window_psnr if not np.isnan(window_psnr) else None
                })

                total_psnr += psnr
                total_ssim += ssim
                if not np.isnan(window_psnr):
                    total_window_psnr += window_psnr
                    window_count += 1

        n = len(self.test_loader.dataset)
        test_results = {
            'avg_psnr': total_psnr / n,
            'avg_ssim': total_ssim / n,
            'avg_window_psnr': total_window_psnr / max(window_count, 1),
            'num_samples': n,
            'per_sample': results
        }

        print(f"\nTest Results ({n} samples):")
        print(f"  Avg PSNR: {test_results['avg_psnr']:.2f} dB")
        print(f"  Avg SSIM: {test_results['avg_ssim']:.4f}")
        print(f"  Avg Window PSNR: {test_results['avg_window_psnr']:.2f} dB")

        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        return test_results


# =============================================================================
# Model Factory
# =============================================================================

def create_model(model_type: str, model_size: str = 'base'):
    """Create model based on type."""
    if model_type == 'retinexformer':
        from src.training.retinexformer import create_retinexformer
        return create_retinexformer(model_size)

    elif model_type == 'restormer':
        from src.training.restormer import create_restormer
        return create_restormer(model_size)

    elif model_type == 'nafnet':
        # If you have NAFNet implemented
        from src.models.nafnet import NAFNet
        configs = {
            'tiny': {'width': 16, 'enc_blk_nums': [1, 1, 1, 8]},
            'small': {'width': 32, 'enc_blk_nums': [1, 1, 1, 14]},
            'base': {'width': 32, 'enc_blk_nums': [2, 2, 4, 22]},
            'large': {'width': 64, 'enc_blk_nums': [2, 2, 4, 28]},
        }
        config = configs.get(model_size, configs['base'])
        return NAFNet(**config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Window-Aware Training')

    # Data
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--split_dir', type=str, default='data_splits')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='outputs_window_aware')

    # Model
    parser.add_argument('--model_type', type=str, default='retinexformer',
                       choices=['retinexformer', 'restormer', 'nafnet'])
    parser.add_argument('--model_size', type=str, default='large')
    parser.add_argument('--image_size', type=int, default=512)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=20)

    # Loss
    parser.add_argument('--loss_preset', type=str, default='aggressive',
                       choices=['default', 'aggressive_window', 'conservative', 'aggressive'])
    parser.add_argument('--use_adversarial', action='store_true')

    # Other
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load datasets
    split_dir = Path(args.split_dir)

    train_dataset = HDRDatasetWithSource(
        data_root=args.data_root,
        jsonl_path=split_dir / f'fold_{args.fold}' / 'train.jsonl',
        image_size=args.image_size,
        augment=True
    )

    val_dataset = HDRDatasetWithSource(
        data_root=args.data_root,
        jsonl_path=split_dir / f'fold_{args.fold}' / 'val.jsonl',
        image_size=args.image_size,
        augment=False
    )

    test_dataset = HDRDatasetWithSource(
        data_root=args.data_root,
        jsonl_path=split_dir / 'test.jsonl',
        image_size=args.image_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    model = create_model(args.model_type, args.model_size)

    # Create trainer and run
    trainer = WindowAwareTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args
    )

    trainer.train()


if __name__ == '__main__':
    main()
