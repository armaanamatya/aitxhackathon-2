#!/usr/bin/env python3
"""
ControlNet-Restormer Training with 3-Fold Cross-Validation
===========================================================
Combines:
- Restormer architecture (fast, efficient)
- ControlNet training strategy (robust for small datasets)
- Cross-validation for optimal generalization
- Pretrained SIDD/GoPro weights + domain adaptation

Optimized for:
- Small datasets (464 samples)
- Maximum quality on unseen test set
- High-resolution training on B200 GPU (512-1024px)

Usage:
    python3 train_controlnet_restormer_cv.py \
        --data_splits_dir data_splits \
        --resolution 512 \
        --pretrained_path path/to/restormer_sidd.pt \
        --epochs 100 \
        --device cuda
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torchvision.models as models

# Add model path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer


# ============================================================================
# PRETRAINED WEIGHT UTILITIES
# ============================================================================

_STAGE_PATTERN = re.compile(r'^(encoder_level\d+|decoder_level\d+|latent|refinement)\.(\d+)\.(.+)$')


def _unwrap_checkpoint_state(checkpoint: Dict) -> Dict:
    """Return the actual state_dict tensor mapping from various checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict', 'params'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def convert_restormer_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert official Restormer checkpoint keys to match our implementation.

    Handles:
        - patch_embed.proj.* -> patch_embed.*
        - LayerNorm ".body." naming
        - Stage numbering (encoder_level1.0 -> encoder_level1.blocks.0)
        - Distributed checkpoints prefixed with "module."
    """
    converted = {}
    for key, tensor in state_dict.items():
        new_key = key

        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]

        if new_key.startswith('patch_embed.proj.'):
            new_key = new_key.replace('patch_embed.proj.', 'patch_embed.')

        if '.body.' in new_key:
            new_key = new_key.replace('.body.', '.')

        match = _STAGE_PATTERN.match(new_key)
        if match and '.blocks.' not in new_key:
            stage, block_idx, remainder = match.groups()
            new_key = f'{stage}.blocks.{block_idx}.{remainder}'

        converted[new_key] = tensor

    return converted


# ============================================================================
# CONTROLNET-RESTORMER ARCHITECTURE
# ============================================================================

class ZeroConv(nn.Module):
    """Zero-initialized convolution from ControlNet."""
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlNetRestormer(nn.Module):
    """
    Restormer with ControlNet-style training.

    Key innovation: Zero convolutions prevent catastrophic forgetting
    when fine-tuning on small datasets.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 dim: int = 48,
                 num_blocks: List[int] = None,
                 num_refinement_blocks: int = 4,
                 heads: List[int] = None,
                 ffn_expansion_factor: float = 2.66,
                 bias: bool = False,
                 use_checkpointing: bool = False,
                 freeze_base: bool = True,
                 pretrained_path: str = None):
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        # Create base model (will be frozen)
        self.base_model = Restormer(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            use_checkpointing=use_checkpointing
        )

        converted_state_dict = None
        if pretrained_path:
            pretrained_file = Path(pretrained_path)
            if pretrained_file.exists():
                print(f"📥 Loading pretrained weights from {pretrained_file}")
                checkpoint = torch.load(pretrained_file, map_location='cpu')
                raw_state_dict = _unwrap_checkpoint_state(checkpoint)
                converted_state_dict = convert_restormer_state_dict(raw_state_dict)

                load_result = self.base_model.load_state_dict(converted_state_dict, strict=False)
                total_params = len(self.base_model.state_dict())
                loaded_params = total_params - len(load_result.missing_keys)
                print(f"   Loaded tensors: {loaded_params}/{total_params}")

                if load_result.missing_keys:
                    print(f"   Missing keys (random init): {len(load_result.missing_keys)}")
                    if len(load_result.missing_keys) <= 5:
                        for key in load_result.missing_keys:
                            print(f"     - {key}")
                if load_result.unexpected_keys:
                    print(f"   Unexpected keys (ignored): {len(load_result.unexpected_keys)}")
                    if len(load_result.unexpected_keys) <= 5:
                        for key in load_result.unexpected_keys:
                            print(f"     - {key}")
                if not load_result.missing_keys and not load_result.unexpected_keys:
                    print("✅ Pretrained weights loaded successfully")
            else:
                print(f"⚠️  Pretrained path not found: {pretrained_path}. Proceeding without initialization.")

        # Freeze base model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Trainable copy (learns real estate domain)
        self.trainable_model = Restormer(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            use_checkpointing=use_checkpointing
        )

        if converted_state_dict is not None:
            self.trainable_model.load_state_dict(converted_state_dict, strict=False)
            print("   Copied pretrained weights into the trainable branch.")

        # Zero convolution for output blending (ControlNet innovation)
        self.zero_conv_out = ZeroConv(out_channels, out_channels)

        # Store config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        # Base model (frozen pretrained knowledge)
        if self.training:
            with torch.no_grad():
                base_out = self.base_model(x)
        else:
            base_out = self.base_model(x)

        # Trainable model (learns domain-specific features)
        trainable_out = self.trainable_model(x)

        # Blend outputs using zero conv
        # At start: output ≈ base_out (uses pretrained knowledge)
        # After training: output = base_out + learned_adaptation
        adaptation = self.zero_conv_out(trainable_out)

        return base_out + adaptation


# ============================================================================
# DATASET
# ============================================================================

class RealEstateDataset(Dataset):
    """Real estate HDR dataset with augmentation."""

    def __init__(self, jsonl_path: str, base_dir: str = '.', resolution: int = 384, augment: bool = True):
        self.base_dir = Path(base_dir)
        self.resolution = resolution
        self.augment = augment

        # Load pairs from JSONL
        self.pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.pairs.append(json.loads(line))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load images
        src_path = self.base_dir / pair['src']
        tar_path = self.base_dir / pair['tar']

        src = cv2.imread(str(src_path))
        tar = cv2.imread(str(tar_path))

        if src is None or tar is None:
            raise ValueError(f"Failed to load: {src_path} or {tar_path}")

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        h, w = src.shape[:2]

        # Augmentation
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                src = np.fliplr(src).copy()
                tar = np.fliplr(tar).copy()

            # Vertical flip (conservative for real estate)
            if np.random.rand() > 0.8:
                src = np.flipud(src).copy()
                tar = np.flipud(tar).copy()

            # Small rotation (±5 degrees)
            if np.random.rand() > 0.7:
                angle = np.random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                src = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                tar = cv2.warpAffine(tar, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Resize
        src = cv2.resize(src, (self.resolution, self.resolution))
        tar = cv2.resize(tar, (self.resolution, self.resolution))

        # To tensor [0, 1]
        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar, pair['src']


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""

    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        return self.criterion(pred_feat, target_feat)


class SSIMLoss(nn.Module):
    """SSIM loss."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size//2)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size//2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(pred ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size//2) - mu_pred_target

        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return 1 - ssim.mean()


# ============================================================================
# METRICS
# ============================================================================

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR between two images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate SSIM between two images."""
    ssim_loss = SSIMLoss()
    return 1 - ssim_loss(img1, img2).item()


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                criterion_l1: nn.Module,
                criterion_vgg: nn.Module,
                criterion_ssim: nn.Module,
                scaler: GradScaler,
                device: str,
                lambda_l1: float,
                lambda_vgg: float,
                lambda_ssim: float,
                mixed_precision: bool) -> Dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_l1 = 0
    total_vgg = 0
    total_ssim = 0

    pbar = tqdm(dataloader, desc="Training")
    for src, tar, _ in pbar:
        src, tar = src.to(device), tar.to(device)

        optimizer.zero_grad()

        # Forward pass
        if mixed_precision:
            with autocast():
                out = model(src)
                loss_l1 = criterion_l1(out, tar)
                loss_vgg = criterion_vgg(out, tar)
                loss_ssim = criterion_ssim(out, tar)
                loss = lambda_l1 * loss_l1 + lambda_vgg * loss_vgg + lambda_ssim * loss_ssim

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(src)
            loss_l1 = criterion_l1(out, tar)
            loss_vgg = criterion_vgg(out, tar)
            loss_ssim = criterion_ssim(out, tar)
            loss = lambda_l1 * loss_l1 + lambda_vgg * loss_vgg + lambda_ssim * loss_ssim

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_l1 += loss_l1.item()
        total_vgg += loss_vgg.item()
        total_ssim += loss_ssim.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{loss_l1.item():.4f}'
        })

    return {
        'loss': total_loss / len(dataloader),
        'l1': total_l1 / len(dataloader),
        'vgg': total_vgg / len(dataloader),
        'ssim': total_ssim / len(dataloader)
    }


def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion_l1: nn.Module,
            criterion_vgg: nn.Module,
            criterion_ssim: nn.Module,
            device: str,
            lambda_l1: float,
            lambda_vgg: float,
            lambda_ssim: float) -> Dict:
    """Validate and compute metrics."""
    model.eval()

    total_loss = 0
    total_l1 = 0
    total_vgg = 0
    total_ssim = 0
    total_psnr = 0
    total_ssim_metric = 0

    with torch.no_grad():
        for src, tar, _ in tqdm(dataloader, desc="Validating"):
            src, tar = src.to(device), tar.to(device)

            out = model(src)

            loss_l1 = criterion_l1(out, tar)
            loss_vgg = criterion_vgg(out, tar)
            loss_ssim = criterion_ssim(out, tar)
            loss = lambda_l1 * loss_l1 + lambda_vgg * loss_vgg + lambda_ssim * loss_ssim

            total_loss += loss.item()
            total_l1 += loss_l1.item()
            total_vgg += loss_vgg.item()
            total_ssim += loss_ssim.item()

            # Metrics
            psnr = calculate_psnr(out, tar)
            ssim = calculate_ssim(out, tar)
            total_psnr += psnr
            total_ssim_metric += ssim

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'l1': total_l1 / n,
        'vgg': total_vgg / n,
        'ssim': total_ssim / n,
        'psnr': total_psnr / n,
        'ssim_metric': total_ssim_metric / n
    }


def train_fold(fold_num: int,
               train_jsonl: str,
               val_jsonl: str,
               args: argparse.Namespace) -> Dict:
    """Train a single fold."""

    print("=" * 80)
    print(f"FOLD {fold_num + 1}/{args.n_folds}")
    print("=" * 80)

    # Setup
    device = args.device
    output_dir = Path(args.output_dir) / f'fold_{fold_num + 1}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    project_root = Path(train_jsonl).parent.parent.parent
    train_dataset = RealEstateDataset(train_jsonl, base_dir=str(project_root),
                                     resolution=args.resolution, augment=True)
    val_dataset = RealEstateDataset(val_jsonl, base_dir=str(project_root),
                                   resolution=args.resolution, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True if args.num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=True if args.num_workers > 0 else False)

    print(f"\n📊 Dataset:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Model
    model = ControlNetRestormer(
        in_channels=3,
        out_channels=3,
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=args.num_refinement_blocks,
        heads=args.heads,
        ffn_expansion_factor=args.ffn_expansion_factor,
        bias=args.bias,
        use_checkpointing=args.use_checkpointing,
        freeze_base=args.freeze_base,
        pretrained_path=args.pretrained_path
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🤖 Model: ControlNet-Restormer")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.1%}")

    # Loss functions
    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGPerceptualLoss(device=device)
    criterion_ssim = SSIMLoss()

    # Optimizer (only train trainable_model + zero_conv_out)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    # Ensure pct_start is between 0 and 1 (clamp to 0.3 max for safety)
    pct_start = min(warmup_steps / total_steps, 0.3) if total_steps > 0 else 0.1
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos'
    )

    scaler = GradScaler() if args.mixed_precision else None

    # Training loop
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_psnrs': [],
        'val_ssims': [],
        'best_val_psnr': 0,
        'best_epoch': 0
    }

    best_psnr = 0
    epochs_without_improvement = 0

    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_l1, criterion_vgg, criterion_ssim,
            scaler, device, args.lambda_l1, args.lambda_vgg, args.lambda_ssim, args.mixed_precision
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion_l1, criterion_vgg, criterion_ssim,
            device, args.lambda_l1, args.lambda_vgg, args.lambda_ssim
        )

        # Update scheduler
        scheduler.step()

        # Log
        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"  Val SSIM: {val_metrics['ssim_metric']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save history
        history['train_losses'].append(train_metrics['loss'])
        history['val_losses'].append(val_metrics['loss'])
        history['val_psnrs'].append(val_metrics['psnr'])
        history['val_ssims'].append(val_metrics['ssim_metric'])

        # Check improvement
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            history['best_val_psnr'] = best_psnr
            history['best_epoch'] = epoch + 1
            epochs_without_improvement = 0

            # Save best checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim_metric'],
                'val_loss': val_metrics['loss']
            }
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"  ✅ New best PSNR! Saved checkpoint.")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠️  No improvement for {epochs_without_improvement} epochs (best: {best_psnr:.2f} dB)")

        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # Early stopping
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
            print(f"  Best PSNR: {best_psnr:.2f} dB at epoch {history['best_epoch']}")
            break

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Fold {fold_num + 1} complete!")
    print(f"  Best Val PSNR: {best_psnr:.2f} dB")
    print(f"{'='*80}")

    return {
        'fold': fold_num + 1,
        'best_val_psnr': best_psnr,
        'best_epoch': history['best_epoch']
    }


def main():
    parser = argparse.ArgumentParser(description="ControlNet-Restormer 3-Fold CV Training")

    # Data
    parser.add_argument('--data_splits_dir', type=str, required=True, help='Data splits directory')
    parser.add_argument('--resolution', type=int, default=384, help='Input resolution')

    # Model
    parser.add_argument('--dim', type=int, default=48, help='Base dimension')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[4, 6, 6, 8])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--ffn_expansion_factor', type=float, default=2.66)
    parser.add_argument('--bias', action='store_true', help='Use bias in convolutions')
    parser.add_argument('--use_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Pretrained Restormer weights')
    parser.add_argument('--freeze_base', action='store_true', help='Freeze base model (only use with pretrained weights)')

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=15)

    # Loss
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_vgg', type=float, default=0.2)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)

    # CV
    parser.add_argument('--n_folds', type=int, default=3)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_controlnet_restormer_cv')
    parser.add_argument('--save_every', type=int, default=10)

    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mixed_precision', action='store_true')

    args = parser.parse_args()

    print("=" * 80)
    print("CONTROLNET-RESTORMER 3-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"  Resolution: {args.resolution}px")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping: {args.early_stopping_patience} epochs")
    print(f"  Loss weights: L1={args.lambda_l1}, VGG={args.lambda_vgg}, SSIM={args.lambda_ssim}")
    print(f"  CV folds: {args.n_folds}")
    print(f"  Pretrained: {args.pretrained_path or 'None (train from scratch)'}")

    # Auto-determine freeze_base: only freeze if we have pretrained weights
    if args.pretrained_path is None and args.freeze_base:
        print("\n⚠️  WARNING: --freeze_base is set but no pretrained weights provided.")
        print("   Setting freeze_base=False to train from scratch properly.")
        args.freeze_base = False
    elif args.pretrained_path and not args.freeze_base:
        print(f"  Freeze base: False (finetuning all layers)")
    elif args.pretrained_path and args.freeze_base:
        print(f"  Freeze base: True (ControlNet mode)")
    else:
        print(f"  Freeze base: False (training from scratch)")

    # Train all folds
    fold_results = []
    for fold in range(args.n_folds):
        train_jsonl = Path(args.data_splits_dir) / f'fold_{fold + 1}' / 'train.jsonl'
        val_jsonl = Path(args.data_splits_dir) / f'fold_{fold + 1}' / 'val.jsonl'

        result = train_fold(fold, str(train_jsonl), str(val_jsonl), args)
        fold_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    avg_psnr = np.mean([r['best_val_psnr'] for r in fold_results])
    std_psnr = np.std([r['best_val_psnr'] for r in fold_results])

    print(f"\n📊 Results:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_val_psnr']:.2f} dB (epoch {r['best_epoch']})")

    print(f"\n📈 Overall:")
    print(f"  Mean PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  Min PSNR: {min(r['best_val_psnr'] for r in fold_results):.2f} dB")
    print(f"  Max PSNR: {max(r['best_val_psnr'] for r in fold_results):.2f} dB")

    # Save summary
    summary = {
        'mean_psnr': float(avg_psnr),
        'std_psnr': float(std_psnr),
        'fold_results': fold_results,
        'config': vars(args)
    }

    with open(Path(args.output_dir) / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ All folds complete!")
    print(f"📊 Summary saved to: {args.output_dir}/cv_summary.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
