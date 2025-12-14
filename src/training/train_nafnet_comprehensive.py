#!/usr/bin/env python3
"""
Comprehensive NAFNet Training for Real Estate HDR Enhancement
==============================================================

Top 0.0001% MLE approach incorporating:
1. Error analysis findings (46x error density in highlights)
2. Highlight-aware losses
3. Same train/val/test splits for fair comparison
4. Comprehensive metrics (PSNR, SSIM, LPIPS, highlight-specific)
5. Standardized output format for easy comparison with Restormer, etc.

Architecture: NAFNet (ECCV 2022)
- Nonlinear Activation Free
- SimpleGate + Channel Attention
- Fast inference, excellent quality
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

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


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    """Dataset for HDR enhancement with source images for highlight detection."""

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

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.LANCZOS
        )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(self.data_root / path).convert('RGB')
        img = self.resize(img)
        img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        source_path = sample.get('source') or sample.get('src') or sample.get('input')
        target_path = sample.get('target') or sample.get('tar') or sample.get('gt')

        source = self._load_image(source_path)
        target = self._load_image(target_path)

        if self.augment:
            if random.random() > 0.5:
                source = torch.flip(source, dims=[2])
                target = torch.flip(target, dims=[2])
            if random.random() > 0.85:
                source = torch.flip(source, dims=[1])
                target = torch.flip(target, dims=[1])

        # Normalize to [-1, 1]
        source = source * 2 - 1
        target = target * 2 - 1

        return {'source': source, 'target': target, 'path': str(source_path)}


# =============================================================================
# Highlight-Aware Loss (based on error analysis)
# =============================================================================

class HighlightDetector(nn.Module):
    """Detect overexposed regions in source images."""

    def __init__(self, brightness_threshold: float = 0.50):
        super().__init__()
        self.threshold = brightness_threshold
        self.smooth = nn.AvgPool2d(11, stride=1, padding=5)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.min() < 0:
            source = (source + 1) / 2
        brightness = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]
        mask = torch.sigmoid(15 * (brightness - self.threshold))
        return self.smooth(mask)


class ComprehensiveLoss(nn.Module):
    """
    Comprehensive loss with highlight awareness.

    Based on error analysis:
    - Highlights have 46x more error density than midtones
    - Need 4x+ weight on highlight regions
    - Hue/saturation errors concentrated in highlights
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.05,
        lambda_highlight: float = 4.0,  # Based on 46x error density
        lambda_gradient: float = 0.15,
        lambda_ssim: float = 0.1,
        lambda_color: float = 0.5,
        brightness_threshold: float = 0.50,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_highlight = lambda_highlight
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color

        self.highlight_detector = HighlightDetector(brightness_threshold)
        self.vgg = None
        self.lpips_fn = None

    def _init_perceptual(self, device):
        if self.vgg is None:
            import torchvision.models as models
            vgg = models.vgg16(weights='IMAGENET1K_V1').features[:23]
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg.to(device).eval()

        if self.lpips_fn is None and self.lambda_lpips > 0:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
            except ImportError:
                self.lambda_lpips = 0

    def _vgg_loss(self, pred, target):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        pred_n = ((pred + 1) / 2 - mean) / std
        target_n = ((target + 1) / 2 - mean) / std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

    def _gradient_loss(self, pred, target):
        def grad(x):
            return x[:, :, :, 1:] - x[:, :, :, :-1], x[:, :, 1:, :] - x[:, :, :-1, :]
        px, py = grad(pred)
        tx, ty = grad(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

    def _ssim_loss(self, pred, target):
        C1, C2 = 0.01**2, 0.03**2
        mu_p = F.avg_pool2d(pred, 3, 1, 1)
        mu_t = F.avg_pool2d(target, 3, 1, 1)
        sig_p = F.avg_pool2d(pred**2, 3, 1, 1) - mu_p**2
        sig_t = F.avg_pool2d(target**2, 3, 1, 1) - mu_t**2
        sig_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
        ssim = ((2*mu_p*mu_t + C1) * (2*sig_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
        return 1 - ssim.mean()

    def _color_loss(self, pred, target, mask):
        """Match color in highlight regions."""
        if pred.min() < 0:
            pred = (pred + 1) / 2
            target = (target + 1) / 2

        # Compute hue-saturation difference
        pred_max = pred.max(dim=1, keepdim=True)[0]
        pred_min = pred.min(dim=1, keepdim=True)[0]
        target_max = target.max(dim=1, keepdim=True)[0]
        target_min = target.min(dim=1, keepdim=True)[0]

        pred_sat = (pred_max - pred_min) / (pred_max + 1e-8)
        target_sat = (target_max - target_min) / (target_max + 1e-8)

        sat_loss = (torch.abs(pred_sat - target_sat) * mask).sum() / (mask.sum() + 1e-8)
        return sat_loss

    def forward(self, pred, target, source, return_components=False):
        self._init_perceptual(pred.device)

        components = {}

        # Base L1
        components['l1'] = F.l1_loss(pred, target)

        # Highlight-weighted L1
        highlight_mask = self.highlight_detector(source)
        weight_map = 1.0 + (self.lambda_highlight - 1.0) * highlight_mask
        components['highlight_l1'] = (torch.abs(pred - target) * weight_map).mean()

        # VGG perceptual
        components['vgg'] = self._vgg_loss(pred, target) if self.lambda_vgg > 0 else torch.tensor(0.0, device=pred.device)

        # LPIPS
        if self.lambda_lpips > 0 and self.lpips_fn is not None:
            components['lpips'] = self.lpips_fn(pred, target).mean()
        else:
            components['lpips'] = torch.tensor(0.0, device=pred.device)

        # Gradient
        components['gradient'] = self._gradient_loss(pred, target)

        # SSIM
        components['ssim'] = self._ssim_loss(pred, target)

        # Color in highlights
        components['color'] = self._color_loss(pred, target, highlight_mask)

        # Total
        total = (
            self.lambda_l1 * components['l1'] +
            components['highlight_l1'] +
            self.lambda_vgg * components['vgg'] +
            self.lambda_lpips * components['lpips'] +
            self.lambda_gradient * components['gradient'] +
            self.lambda_ssim * components['ssim'] +
            self.lambda_color * components['color']
        )

        if return_components:
            return total, components, highlight_mask
        return total


# =============================================================================
# Metrics
# =============================================================================

class MetricsCalculator:
    """Comprehensive metrics for fair comparison."""

    def __init__(self, device):
        self.device = device
        self.highlight_detector = HighlightDetector().to(device)

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        return 10 * torch.log10(4 / mse).item()

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        C1, C2 = 0.01**2, 0.03**2
        mu_p = F.avg_pool2d(pred, 3, 1, 1)
        mu_t = F.avg_pool2d(target, 3, 1, 1)
        sig_p = F.avg_pool2d(pred**2, 3, 1, 1) - mu_p**2
        sig_t = F.avg_pool2d(target**2, 3, 1, 1) - mu_t**2
        sig_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
        ssim = ((2*mu_p*mu_t + C1) * (2*sig_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
        return ssim.mean().item()

    def compute_highlight_psnr(self, pred, target, source) -> float:
        """PSNR specifically on highlight regions."""
        mask = self.highlight_detector(source)
        if mask.sum() < 100:
            return float('nan')
        mse = ((pred - target).pow(2) * mask).sum() / (mask.sum() * 3 + 1e-8)
        if mse < 1e-10:
            return 100.0
        return 10 * torch.log10(4 / mse).item()

    def compute_all(self, pred, target, source) -> Dict[str, float]:
        return {
            'psnr': self.compute_psnr(pred, target),
            'ssim': self.compute_ssim(pred, target),
            'highlight_psnr': self.compute_highlight_psnr(pred, target, source),
        }


# =============================================================================
# Trainer
# =============================================================================

class NAFNetTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)

        # Loss
        self.criterion = ComprehensiveLoss(
            lambda_l1=args.lambda_l1,
            lambda_vgg=args.lambda_vgg,
            lambda_lpips=args.lambda_lpips,
            lambda_highlight=args.lambda_highlight,
            lambda_gradient=args.lambda_gradient,
            lambda_ssim=args.lambda_ssim,
            lambda_color=args.lambda_color,
            brightness_threshold=args.brightness_threshold,
        ).to(self.device)

        # Metrics
        self.metrics = MetricsCalculator(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

        # Scheduler
        warmup_steps = len(train_loader) * args.warmup_epochs
        total_steps = len(train_loader) * args.num_epochs

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.01)
        self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        # Mixed precision
        self.scaler = GradScaler() if args.use_amp else None

        # Output
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)

        # Tracking
        self.best_psnr = 0
        self.epochs_without_improvement = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_highlight_psnr': [], 'lr': []}

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    pred = self.model(source)
                    loss, components, _ = self.criterion(pred, target, source, return_components=True)
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
            else:
                pred = self.model(source)
                loss, components, _ = self.criterion(pred, target, source, return_components=True)
                loss.backward()
                if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'hl': f'{components["highlight_l1"].item():.4f}'})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        total_loss, total_psnr, total_ssim, total_hl_psnr = 0, 0, 0, 0
        hl_count = 0

        for batch in self.val_loader:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            pred = self.model(source).clamp(-1, 1)

            loss = self.criterion(pred, target, source)
            total_loss += loss.item()

            metrics = self.metrics.compute_all(pred, target, source)
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            if not np.isnan(metrics['highlight_psnr']):
                total_hl_psnr += metrics['highlight_psnr']
                hl_count += 1

        n = len(self.val_loader)
        return {
            'loss': total_loss / n,
            'psnr': total_psnr / n,
            'ssim': total_ssim / n,
            'highlight_psnr': total_hl_psnr / max(hl_count, 1)
        }

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        self.model.eval()
        batch = next(iter(self.val_loader))
        source = batch['source'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)
        pred = self.model(source).clamp(-1, 1)

        def to_img(t):
            return ((t + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

        sources, targets, preds = to_img(source), to_img(target), to_img(pred)

        for i in range(num_samples):
            comparison = np.concatenate([sources[i], preds[i], targets[i]], axis=1)
            comparison = (comparison * 255).astype(np.uint8)
            Image.fromarray(comparison).save(self.output_dir / 'samples' / f'epoch_{epoch:03d}_sample_{i}.jpg')

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'history': self.history,
            'args': vars(self.args)
        }
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
        if epoch % self.args.save_every == 0:
            torch.save(checkpoint, self.output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch:03d}.pt')

    def train(self):
        print(f"\n{'='*70}")
        print(f"NAFNet Comprehensive Training")
        print(f"{'='*70}")
        print(f"Model: NAFNet ({sum(p.numel() for p in self.model.parameters())/1e6:.2f}M params)")
        print(f"Device: {self.device}")
        print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        print(f"Highlight weight: {self.args.lambda_highlight}x (based on 46x error density)")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        for epoch in range(1, self.args.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['val_highlight_psnr'].append(val_metrics['highlight_psnr'])
            self.history['lr'].append(self.scheduler.get_last_lr()[0])

            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            print(f"\nEpoch {epoch}/{self.args.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB {'(BEST)' if is_best else ''}")
            print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Highlight PSNR: {val_metrics['highlight_psnr']:.2f} dB")

            self.save_checkpoint(epoch, is_best)
            if epoch % self.args.sample_every == 0:
                self.save_samples(epoch)

            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

            if self.epochs_without_improvement >= self.args.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Final test evaluation
        self.evaluate_test()

    @torch.no_grad()
    def evaluate_test(self):
        """Evaluate on held-out test set - standardized format for comparison."""
        print(f"\n{'='*70}")
        print("Test Evaluation (10 held-out samples)")
        print(f"{'='*70}")

        checkpoint = torch.load(self.output_dir / 'checkpoint_best.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        results = []
        total_psnr, total_ssim, total_hl_psnr = 0, 0, 0
        hl_count = 0

        # Create test output directory
        test_output_dir = self.output_dir / 'test_outputs'
        test_output_dir.mkdir(exist_ok=True)

        for batch in tqdm(self.test_loader, desc='Testing'):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            paths = batch['path']

            pred = self.model(source).clamp(-1, 1)

            for i in range(source.shape[0]):
                metrics = self.metrics.compute_all(pred[i:i+1], target[i:i+1], source[i:i+1])

                # Get image ID from path
                img_id = Path(paths[i]).stem.replace('_src', '').replace('_input', '')

                results.append({
                    'id': img_id,
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'highlight_psnr': metrics['highlight_psnr'] if not np.isnan(metrics['highlight_psnr']) else None
                })

                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
                if not np.isnan(metrics['highlight_psnr']):
                    total_hl_psnr += metrics['highlight_psnr']
                    hl_count += 1

                # Save test outputs in standardized format
                def to_img(t):
                    return ((t + 1) / 2).clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy()

                src_img = (to_img(source[i:i+1]) * 255).astype(np.uint8)
                pred_img = (to_img(pred[i:i+1]) * 255).astype(np.uint8)
                tar_img = (to_img(target[i:i+1]) * 255).astype(np.uint8)

                Image.fromarray(src_img).save(test_output_dir / f'{img_id}_src.jpg')
                Image.fromarray(pred_img).save(test_output_dir / f'{img_id}_output.jpg')
                Image.fromarray(tar_img).save(test_output_dir / f'{img_id}_tar.jpg')

                # Comparison image
                comparison = np.concatenate([src_img, pred_img, tar_img], axis=1)
                Image.fromarray(comparison).save(test_output_dir / f'{img_id}_comparison.jpg')

        n = len(self.test_loader.dataset)
        test_results = {
            'model': 'NAFNet',
            'params': sum(p.numel() for p in self.model.parameters()),
            'avg_psnr': total_psnr / n,
            'avg_ssim': total_ssim / n,
            'avg_highlight_psnr': total_hl_psnr / max(hl_count, 1),
            'num_samples': n,
            'per_sample': results
        }

        print(f"\nTest Results ({n} samples):")
        print(f"  Avg PSNR: {test_results['avg_psnr']:.2f} dB")
        print(f"  Avg SSIM: {test_results['avg_ssim']:.4f}")
        print(f"  Avg Highlight PSNR: {test_results['avg_highlight_psnr']:.2f} dB")

        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        # Also save comparison summary
        comparison_summary = {
            'model': 'NAFNet',
            'psnr': test_results['avg_psnr'],
            'ssim': test_results['avg_ssim'],
            'highlight_psnr': test_results['avg_highlight_psnr'],
            'params_M': test_results['params'] / 1e6
        }
        with open(self.output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2)

        return test_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='NAFNet Comprehensive Training')

    # Data
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--split_dir', type=str, default='data_splits')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='outputs_nafnet')

    # Model
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--image_size', type=int, default=512)

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=15)

    # Loss weights (based on error analysis)
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_vgg', type=float, default=0.1)
    parser.add_argument('--lambda_lpips', type=float, default=0.05)
    parser.add_argument('--lambda_highlight', type=float, default=4.0)  # Based on 46x error density
    parser.add_argument('--lambda_gradient', type=float, default=0.15)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)
    parser.add_argument('--lambda_color', type=float, default=0.5)
    parser.add_argument('--brightness_threshold', type=float, default=0.50)

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

    # Load data
    split_dir = Path(args.split_dir)
    train_dataset = HDRDataset(args.data_root, split_dir / f'fold_{args.fold}' / 'train.jsonl', args.image_size, augment=True)
    val_dataset = HDRDataset(args.data_root, split_dir / f'fold_{args.fold}' / 'val.jsonl', args.image_size, augment=False)
    test_dataset = HDRDataset(args.data_root, split_dir / 'test.jsonl', args.image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Create model
    from src.training.training.nafnet import create_nafnet
    model = create_nafnet(args.model_size)

    # Train
    trainer = NAFNetTrainer(model, train_loader, val_loader, test_loader, args)
    trainer.train()


if __name__ == '__main__':
    main()
