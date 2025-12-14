#!/usr/bin/env python3
"""
Comprehensive Retinexformer Training for Real Estate HDR Enhancement
=====================================================================

Top 0.0001% MLE solution optimized for:
1. Window/highlight color recovery (46x error density finding)
2. Edge preservation (gradient loss)
3. Color accuracy in bright regions (hue/saturation matching)
4. Physics-based Retinex decomposition

Why Retinexformer is ideal for this task:
- Illumination-guided attention (IG-MSA) processes windows differently
- Retinex theory separates illumination from reflectance
- Naturally handles HDR/exposure issues
- Lightweight (3.7M params) - good for 577 samples

Based on error analysis:
- Highlights have 46.5x more error density than midtones
- Hue error in highlights is 1.8-2.7x higher than average
- Edge regions have higher error than flat regions
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Dataset
# =============================================================================

class HDRDataset(Dataset):
    def __init__(self, data_root: str, jsonl_path: str, image_size: int = 512, augment: bool = True):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(self.data_root / path).convert('RGB')
        img = self.resize(img)
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        source_path = sample.get('source') or sample.get('src') or sample.get('input')
        target_path = sample.get('target') or sample.get('tar') or sample.get('gt')
        source, target = self._load_image(source_path), self._load_image(target_path)
        if self.augment:
            if random.random() > 0.5:
                source, target = torch.flip(source, [2]), torch.flip(target, [2])
            if random.random() > 0.85:
                source, target = torch.flip(source, [1]), torch.flip(target, [1])
        return {'source': source * 2 - 1, 'target': target * 2 - 1, 'path': str(source_path)}


# =============================================================================
# Optimal Loss for Windows/Highlights/Edges
# =============================================================================

class HighlightDetector(nn.Module):
    """Detect overexposed/bright regions that need color recovery."""
    def __init__(self, brightness_threshold: float = 0.50, saturation_threshold: float = 0.15):
        super().__init__()
        self.bright_thresh = brightness_threshold
        self.sat_thresh = saturation_threshold
        self.smooth = nn.AvgPool2d(11, stride=1, padding=5)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.min() < 0:
            source = (source + 1) / 2
        brightness = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]
        max_rgb, min_rgb = source.max(dim=1, keepdim=True)[0], source.min(dim=1, keepdim=True)[0]
        saturation = max_rgb - min_rgb
        bright_mask = torch.sigmoid(15 * (brightness - self.bright_thresh))
        low_sat_mask = torch.sigmoid(15 * (self.sat_thresh - saturation))
        highlight_mask = bright_mask * low_sat_mask * 0.7 + bright_mask * 0.3
        return self.smooth(highlight_mask)


class EdgeDetector(nn.Module):
    """Detect edge regions using Sobel-like operators."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() < 0:
            x = (x + 1) / 2
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return torch.sigmoid(10 * (magnitude - 0.1))


class OptimalHDRLoss(nn.Module):
    """
    Optimal loss for real estate HDR - windows, highlights, edges, colors.

    Based on error analysis findings:
    - 46.5x error density in highlights -> 4x+ highlight weight
    - 1.8-2.7x hue error in highlights -> explicit hue loss
    - Edge regions have higher error -> edge-aware loss
    """
    def __init__(
        self,
        # Base losses
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.1,
        lambda_lpips: float = 0.05,
        # Highlight losses (KEY based on error analysis)
        lambda_highlight_l1: float = 4.0,
        lambda_highlight_color: float = 1.5,
        lambda_highlight_hue: float = 0.8,
        lambda_highlight_saturation: float = 0.6,
        # Edge losses
        lambda_gradient: float = 0.2,
        lambda_edge_aware: float = 0.3,
        # Structural
        lambda_ssim: float = 0.15,
        lambda_ms_ssim: float = 0.1,
        # Thresholds
        brightness_threshold: float = 0.50,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_highlight_l1 = lambda_highlight_l1
        self.lambda_highlight_color = lambda_highlight_color
        self.lambda_highlight_hue = lambda_highlight_hue
        self.lambda_highlight_saturation = lambda_highlight_saturation
        self.lambda_gradient = lambda_gradient
        self.lambda_edge_aware = lambda_edge_aware
        self.lambda_ssim = lambda_ssim
        self.lambda_ms_ssim = lambda_ms_ssim

        self.highlight_detector = HighlightDetector(brightness_threshold)
        self.edge_detector = EdgeDetector()
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
            except:
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

    def _edge_aware_loss(self, pred, target, edge_mask):
        """Higher weight on edges to prevent artifacts."""
        error = torch.abs(pred - target)
        weighted_error = error * (1.0 + 2.0 * edge_mask)
        return weighted_error.mean()

    def _ssim_loss(self, pred, target):
        C1, C2 = 0.01**2, 0.03**2
        mu_p = F.avg_pool2d(pred, 3, 1, 1)
        mu_t = F.avg_pool2d(target, 3, 1, 1)
        sig_p = F.avg_pool2d(pred**2, 3, 1, 1) - mu_p**2
        sig_t = F.avg_pool2d(target**2, 3, 1, 1) - mu_t**2
        sig_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
        ssim = ((2*mu_p*mu_t + C1) * (2*sig_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
        return 1 - ssim.mean()

    def _hue_loss(self, pred, target, mask):
        """Match hue in highlight regions - critical for window colors."""
        if pred.min() < 0:
            pred, target = (pred + 1) / 2, (target + 1) / 2
        eps = 1e-8
        pred_hue = torch.atan2(1.732 * (pred[:, 1:2] - pred[:, 2:3]), 2 * pred[:, 0:1] - pred[:, 1:2] - pred[:, 2:3] + eps)
        target_hue = torch.atan2(1.732 * (target[:, 1:2] - target[:, 2:3]), 2 * target[:, 0:1] - target[:, 1:2] - target[:, 2:3] + eps)
        hue_diff = torch.abs(pred_hue - target_hue)
        hue_diff = torch.min(hue_diff, 2 * 3.14159 - hue_diff)
        return (hue_diff * mask).sum() / (mask.sum() + eps)

    def _saturation_loss(self, pred, target, mask):
        """Match saturation in highlights - washed out areas need color recovery."""
        if pred.min() < 0:
            pred, target = (pred + 1) / 2, (target + 1) / 2
        pred_sat = (pred.max(dim=1, keepdim=True)[0] - pred.min(dim=1, keepdim=True)[0]) / (pred.max(dim=1, keepdim=True)[0] + 1e-8)
        target_sat = (target.max(dim=1, keepdim=True)[0] - target.min(dim=1, keepdim=True)[0]) / (target.max(dim=1, keepdim=True)[0] + 1e-8)
        return (torch.abs(pred_sat - target_sat) * mask).sum() / (mask.sum() + 1e-8)

    def _highlight_color_loss(self, pred, target, mask):
        """Match overall color distribution in highlights."""
        if pred.min() < 0:
            pred, target = (pred + 1) / 2, (target + 1) / 2
        pred_masked = pred * mask
        target_masked = target * mask
        color_diff = torch.abs(pred_masked - target_masked)
        return color_diff.sum() / (mask.sum() * 3 + 1e-8)

    def forward(self, pred, target, source, return_components=False):
        self._init_perceptual(pred.device)

        highlight_mask = self.highlight_detector(source)
        edge_mask = self.edge_detector(target)

        components = {}

        # Base L1
        components['l1'] = F.l1_loss(pred, target)

        # Highlight-weighted L1 (KEY - 4x weight based on 46x error density)
        weight_map = 1.0 + (self.lambda_highlight_l1 - 1.0) * highlight_mask
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

        # Edge-aware
        components['edge_aware'] = self._edge_aware_loss(pred, target, edge_mask)

        # SSIM
        components['ssim'] = self._ssim_loss(pred, target)

        # Highlight hue (critical for window colors)
        components['highlight_hue'] = self._hue_loss(pred, target, highlight_mask)

        # Highlight saturation
        components['highlight_sat'] = self._saturation_loss(pred, target, highlight_mask)

        # Highlight color
        components['highlight_color'] = self._highlight_color_loss(pred, target, highlight_mask)

        # Total
        total = (
            self.lambda_l1 * components['l1'] +
            components['highlight_l1'] +
            self.lambda_vgg * components['vgg'] +
            self.lambda_lpips * components['lpips'] +
            self.lambda_gradient * components['gradient'] +
            self.lambda_edge_aware * components['edge_aware'] +
            self.lambda_ssim * components['ssim'] +
            self.lambda_highlight_hue * components['highlight_hue'] +
            self.lambda_highlight_saturation * components['highlight_sat'] +
            self.lambda_highlight_color * components['highlight_color']
        )

        if return_components:
            return total, components, highlight_mask
        return total


# =============================================================================
# Metrics
# =============================================================================

class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.highlight_detector = HighlightDetector().to(device)

    def compute_psnr(self, pred, target):
        mse = F.mse_loss(pred, target)
        return 100.0 if mse == 0 else 10 * torch.log10(4 / mse).item()

    def compute_ssim(self, pred, target):
        C1, C2 = 0.01**2, 0.03**2
        mu_p, mu_t = F.avg_pool2d(pred, 3, 1, 1), F.avg_pool2d(target, 3, 1, 1)
        sig_p = F.avg_pool2d(pred**2, 3, 1, 1) - mu_p**2
        sig_t = F.avg_pool2d(target**2, 3, 1, 1) - mu_t**2
        sig_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
        ssim = ((2*mu_p*mu_t + C1) * (2*sig_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
        return ssim.mean().item()

    def compute_highlight_psnr(self, pred, target, source):
        mask = self.highlight_detector(source)
        if mask.sum() < 100:
            return float('nan')
        mse = ((pred - target).pow(2) * mask).sum() / (mask.sum() * 3 + 1e-8)
        return 100.0 if mse < 1e-10 else 10 * torch.log10(4 / mse).item()

    def compute_all(self, pred, target, source):
        return {
            'psnr': self.compute_psnr(pred, target),
            'ssim': self.compute_ssim(pred, target),
            'highlight_psnr': self.compute_highlight_psnr(pred, target, source)
        }


# =============================================================================
# Trainer
# =============================================================================

class RetinexformerTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.criterion = OptimalHDRLoss(
            lambda_l1=args.lambda_l1,
            lambda_vgg=args.lambda_vgg,
            lambda_lpips=args.lambda_lpips,
            lambda_highlight_l1=args.lambda_highlight_l1,
            lambda_highlight_color=args.lambda_highlight_color,
            lambda_highlight_hue=args.lambda_highlight_hue,
            lambda_highlight_saturation=args.lambda_highlight_saturation,
            lambda_gradient=args.lambda_gradient,
            lambda_edge_aware=args.lambda_edge_aware,
            lambda_ssim=args.lambda_ssim,
            brightness_threshold=args.brightness_threshold,
        ).to(self.device)

        self.metrics = MetricsCalculator(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        warmup_steps = len(train_loader) * args.warmup_epochs
        total_steps = len(train_loader) * args.num_epochs
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.01)
        self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        self.scaler = GradScaler() if args.use_amp else None
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)

        self.best_psnr = 0
        self.epochs_without_improvement = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_highlight_psnr': [], 'lr': []}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            source, target = batch['source'].to(self.device), batch['target'].to(self.device)
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
    def validate(self):
        self.model.eval()
        total_loss, total_psnr, total_ssim, total_hl_psnr, hl_count = 0, 0, 0, 0, 0
        for batch in self.val_loader:
            source, target = batch['source'].to(self.device), batch['target'].to(self.device)
            pred = self.model(source).clamp(-1, 1)
            total_loss += self.criterion(pred, target, source).item()
            m = self.metrics.compute_all(pred, target, source)
            total_psnr += m['psnr']
            total_ssim += m['ssim']
            if not np.isnan(m['highlight_psnr']):
                total_hl_psnr += m['highlight_psnr']
                hl_count += 1
        n = len(self.val_loader)
        return {'loss': total_loss/n, 'psnr': total_psnr/n, 'ssim': total_ssim/n, 'highlight_psnr': total_hl_psnr/max(hl_count,1)}

    @torch.no_grad()
    def save_samples(self, epoch, num_samples=4):
        self.model.eval()
        batch = next(iter(self.val_loader))
        source, target = batch['source'][:num_samples].to(self.device), batch['target'][:num_samples].to(self.device)
        pred = self.model(source).clamp(-1, 1)
        def to_img(t):
            return ((t+1)/2).clamp(0,1).permute(0,2,3,1).cpu().numpy()
        for i in range(num_samples):
            comp = np.concatenate([to_img(source)[i], to_img(pred)[i], to_img(target)[i]], axis=1)
            Image.fromarray((comp*255).astype(np.uint8)).save(self.output_dir/'samples'/f'epoch_{epoch:03d}_sample_{i}.jpg')

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'best_psnr': self.best_psnr, 'history': self.history}
        torch.save(ckpt, self.output_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(ckpt, self.output_dir / 'checkpoint_best.pt')

    def train(self):
        print(f"\n{'='*70}\nRetinexformer Comprehensive Training\n{'='*70}")
        print(f"Model: Retinexformer ({sum(p.numel() for p in self.model.parameters())/1e6:.2f}M params)")
        print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        print(f"Highlight weight: {self.args.lambda_highlight_l1}x, Hue: {self.args.lambda_highlight_hue}, Sat: {self.args.lambda_highlight_saturation}")
        print(f"Output: {self.output_dir}\n{'='*70}\n")

        for epoch in range(1, self.args.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val['loss'])
            self.history['val_psnr'].append(val['psnr'])
            self.history['val_ssim'].append(val['ssim'])
            self.history['val_highlight_psnr'].append(val['highlight_psnr'])
            self.history['lr'].append(self.scheduler.get_last_lr()[0])

            is_best = val['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val['psnr']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            print(f"\nEpoch {epoch}: Train={train_loss:.4f}, Val={val['loss']:.4f}, PSNR={val['psnr']:.2f}{'*' if is_best else ''}, SSIM={val['ssim']:.4f}, HL_PSNR={val['highlight_psnr']:.2f}")

            self.save_checkpoint(epoch, is_best)
            if epoch % self.args.sample_every == 0:
                self.save_samples(epoch)
            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

            if self.epochs_without_improvement >= self.args.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.evaluate_test()

    @torch.no_grad()
    def evaluate_test(self):
        print(f"\n{'='*70}\nTest Evaluation\n{'='*70}")
        ckpt = torch.load(self.output_dir / 'checkpoint_best.pt')
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        test_out = self.output_dir / 'test_outputs'
        test_out.mkdir(exist_ok=True)

        results, total_psnr, total_ssim, total_hl, hl_count = [], 0, 0, 0, 0
        for batch in tqdm(self.test_loader, desc='Testing'):
            source, target, paths = batch['source'].to(self.device), batch['target'].to(self.device), batch['path']
            pred = self.model(source).clamp(-1, 1)
            for i in range(source.shape[0]):
                m = self.metrics.compute_all(pred[i:i+1], target[i:i+1], source[i:i+1])
                img_id = Path(paths[i]).stem.replace('_src', '').replace('_input', '')
                results.append({'id': img_id, 'psnr': m['psnr'], 'ssim': m['ssim'], 'highlight_psnr': m['highlight_psnr'] if not np.isnan(m['highlight_psnr']) else None})
                total_psnr += m['psnr']
                total_ssim += m['ssim']
                if not np.isnan(m['highlight_psnr']):
                    total_hl += m['highlight_psnr']
                    hl_count += 1
                # Save outputs
                def to_np(t):
                    return ((t+1)/2).clamp(0,1).squeeze().permute(1,2,0).cpu().numpy()
                for suffix, tensor in [('_src', source[i:i+1]), ('_output', pred[i:i+1]), ('_tar', target[i:i+1])]:
                    Image.fromarray((to_np(tensor)*255).astype(np.uint8)).save(test_out/f'{img_id}{suffix}.jpg')
                comp = np.concatenate([to_np(source[i:i+1]), to_np(pred[i:i+1]), to_np(target[i:i+1])], axis=1)
                Image.fromarray((comp*255).astype(np.uint8)).save(test_out/f'{img_id}_comparison.jpg')

        n = len(self.test_loader.dataset)
        test_results = {'model': 'Retinexformer', 'params': sum(p.numel() for p in self.model.parameters()), 'avg_psnr': total_psnr/n, 'avg_ssim': total_ssim/n, 'avg_highlight_psnr': total_hl/max(hl_count,1), 'num_samples': n, 'per_sample': results}
        print(f"\nTest: PSNR={test_results['avg_psnr']:.2f}, SSIM={test_results['avg_ssim']:.4f}, HL_PSNR={test_results['avg_highlight_psnr']:.2f}")
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        with open(self.output_dir / 'comparison_summary.json', 'w') as f:
            json.dump({'model': 'Retinexformer', 'psnr': test_results['avg_psnr'], 'ssim': test_results['avg_ssim'], 'highlight_psnr': test_results['avg_highlight_psnr'], 'params_M': test_results['params']/1e6}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.')
    parser.add_argument('--split_dir', default='data_splits')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--output_dir', default='outputs_retinexformer')
    parser.add_argument('--model_size', default='large')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    # Loss weights (optimal based on error analysis)
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_vgg', type=float, default=0.1)
    parser.add_argument('--lambda_lpips', type=float, default=0.05)
    parser.add_argument('--lambda_highlight_l1', type=float, default=4.0)
    parser.add_argument('--lambda_highlight_color', type=float, default=1.5)
    parser.add_argument('--lambda_highlight_hue', type=float, default=0.8)
    parser.add_argument('--lambda_highlight_saturation', type=float, default=0.6)
    parser.add_argument('--lambda_gradient', type=float, default=0.2)
    parser.add_argument('--lambda_edge_aware', type=float, default=0.3)
    parser.add_argument('--lambda_ssim', type=float, default=0.15)
    parser.add_argument('--brightness_threshold', type=float, default=0.50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    split_dir = Path(args.split_dir)
    train_ds = HDRDataset(args.data_root, split_dir/f'fold_{args.fold}'/'train.jsonl', args.image_size, True)
    val_ds = HDRDataset(args.data_root, split_dir/f'fold_{args.fold}'/'val.jsonl', args.image_size, False)
    test_ds = HDRDataset(args.data_root, split_dir/'test.jsonl', args.image_size, False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    from src.training.retinexformer import create_retinexformer
    model = create_retinexformer(args.model_size)

    trainer = RetinexformerTrainer(model, train_loader, val_loader, test_loader, args)
    trainer.train()


if __name__ == '__main__':
    main()
