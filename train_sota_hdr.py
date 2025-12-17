#!/usr/bin/env python3
"""
SOTA HDR Training with Zone-Aware Loss
=======================================

Key improvements based on research and ground truth analysis:

1. Zone-Aware Loss: Different weights for windows vs shadows vs midtones
2. Exposure Masking: Reduce gradient contribution from saturated regions
3. Perceptual Loss (LPIPS): Better texture/structure preservation
4. Color Consistency Loss: Preserve color relationships
5. Safe Residual Learning: Prevent black spots with clamped residuals

Ground Truth Analysis Results:
- Windows (220+): Only -7 brightness, with blue color shift
- Midtones (80-180): +75 brightness boost
- Shadows (0-80): +63 brightness boost

The model should NOT aggressively darken windows - just slight reduction + color shift.

References:
- Single Image HDR Reconstruction with Masked Features (SIGGRAPH 2020)
- LPIPS: The Unreasonable Effectiveness of Deep Features (CVPR 2018)
- Intrinsic Single-Image HDR Reconstruction (ECCV 2024)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import models

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from training.restormer import TransformerBlock, TransformerStage, Downsample, Upsample


# ============================================================================
# SOTA LOSS FUNCTIONS
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for texture/structure preservation."""

    def __init__(self, layers=[3, 8, 15, 22], weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.blocks = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev_layer:layer+1]))
            prev_layer = layer + 1

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.weights = weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize for VGG
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x_pred, x_target = pred, target

        for block, weight in zip(self.blocks, self.weights):
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss += weight * F.l1_loss(x_pred, x_target)

        return loss


class ZoneAwareLoss(nn.Module):
    """
    Zone-aware loss that applies different weights to different brightness regions.

    Based on ground truth analysis:
    - Windows (bright): Small reduction needed, don't penalize heavily
    - Midtones: Large boost needed, moderate penalty
    - Shadows: Large boost needed, high penalty for under-enhancement
    """

    def __init__(self, window_weight=0.3, midtone_weight=1.0, shadow_weight=1.5):
        super().__init__()
        self.window_weight = window_weight      # Lower = allow more flexibility for windows
        self.midtone_weight = midtone_weight
        self.shadow_weight = shadow_weight

    def forward(self, pred, target, source):
        """
        Args:
            pred: Model prediction [B, 3, H, W]
            target: Ground truth [B, 3, H, W]
            source: Original input [B, 3, H, W]
        """
        # Convert source to grayscale for zone detection
        source_gray = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]

        # Create soft zone masks
        # Window zone: source brightness > 0.85 (220/255)
        window_mask = torch.sigmoid((source_gray - 0.85) * 20)  # Soft threshold

        # Shadow zone: source brightness < 0.3 (77/255)
        shadow_mask = torch.sigmoid((0.3 - source_gray) * 20)

        # Midtone zone: everything else
        midtone_mask = 1.0 - window_mask - shadow_mask
        midtone_mask = torch.clamp(midtone_mask, 0, 1)

        # Compute per-pixel L1 loss
        pixel_loss = torch.abs(pred - target)

        # Apply zone weights
        weighted_loss = (
            self.window_weight * window_mask * pixel_loss +
            self.midtone_weight * midtone_mask * pixel_loss +
            self.shadow_weight * shadow_mask * pixel_loss
        )

        return weighted_loss.mean()


class ExposureMaskingLoss(nn.Module):
    """
    Reduce gradient contribution from saturated regions.
    Based on Santos et al. "Single image HDR reconstruction using a CNN
    with masked features and perceptual loss" (SIGGRAPH 2020)
    """

    def __init__(self, saturation_threshold=0.95):
        super().__init__()
        self.threshold = saturation_threshold

    def create_saturation_mask(self, img):
        """Create mask that reduces weight for saturated pixels."""
        # Max channel value
        max_val = img.max(dim=1, keepdim=True)[0]

        # Soft saturation mask: 1.0 for normal pixels, lower for saturated
        # This reduces the loss contribution from blown-out areas
        mask = 1.0 - torch.sigmoid((max_val - self.threshold) * 50)

        return mask

    def forward(self, pred, target, source):
        """Masked L1 loss with reduced weight for saturated regions."""
        mask = self.create_saturation_mask(source)

        pixel_loss = torch.abs(pred - target)
        masked_loss = pixel_loss * mask

        # Normalize by mask sum to avoid scale issues
        return masked_loss.sum() / (mask.sum() + 1e-6)


class ColorConsistencyLoss(nn.Module):
    """
    Preserve color relationships and prevent color shifts in non-window regions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Convert to LAB for perceptually uniform color comparison
        # Simplified: use color ratios instead

        # Color ratio consistency
        pred_ratios = pred / (pred.mean(dim=1, keepdim=True) + 1e-6)
        target_ratios = target / (target.mean(dim=1, keepdim=True) + 1e-6)

        ratio_loss = F.l1_loss(pred_ratios, target_ratios)

        # Saturation consistency
        pred_sat = pred.max(dim=1)[0] - pred.min(dim=1)[0]
        target_sat = target.max(dim=1)[0] - target.min(dim=1)[0]

        sat_loss = F.l1_loss(pred_sat, target_sat)

        return ratio_loss + 0.5 * sat_loss


class WindowRecoveryLoss(nn.Module):
    """
    Specific loss for window regions to encourage detail recovery.
    Based on ground truth: windows should have slight blue shift and contrast.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, source):
        # Detect window regions in source
        source_gray = 0.299 * source[:, 0:1] + 0.587 * source[:, 1:2] + 0.114 * source[:, 2:3]
        window_mask = torch.sigmoid((source_gray - 0.85) * 20)

        if window_mask.sum() < 100:
            return torch.tensor(0.0, device=pred.device)

        # In window regions, check for:
        # 1. Gradient/edge preservation (structure should be visible)
        pred_edges = self._sobel_edges(pred)
        target_edges = self._sobel_edges(target)

        edge_loss = F.l1_loss(pred_edges * window_mask, target_edges * window_mask)

        # 2. Color matching in windows
        color_loss = F.l1_loss(pred * window_mask, target * window_mask)

        return edge_loss + color_loss

    def _sobel_edges(self, img):
        """Compute edge magnitude."""
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)

        return torch.sqrt(gx**2 + gy**2 + 1e-8)


class SOTALoss(nn.Module):
    """
    Combined SOTA loss function for HDR real estate enhancement.
    """

    def __init__(self,
                 l1_weight=1.0,
                 perceptual_weight=0.1,
                 zone_weight=0.5,
                 exposure_mask_weight=0.3,
                 color_weight=0.2,
                 window_weight=0.3):
        super().__init__()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.zone_weight = zone_weight
        self.exposure_mask_weight = exposure_mask_weight
        self.color_weight = color_weight
        self.window_weight = window_weight

        self.perceptual_loss = VGGPerceptualLoss()
        self.zone_loss = ZoneAwareLoss(window_weight=0.3, midtone_weight=1.0, shadow_weight=1.5)
        self.exposure_loss = ExposureMaskingLoss()
        self.color_loss = ColorConsistencyLoss()
        self.window_loss = WindowRecoveryLoss()

    def forward(self, pred, target, source):
        losses = {}

        # Basic L1
        losses['l1'] = F.l1_loss(pred, target) * self.l1_weight

        # Perceptual (VGG)
        losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight

        # Zone-aware
        losses['zone'] = self.zone_loss(pred, target, source) * self.zone_weight

        # Exposure masking
        losses['exposure'] = self.exposure_loss(pred, target, source) * self.exposure_mask_weight

        # Color consistency
        losses['color'] = self.color_loss(pred, target) * self.color_weight

        # Window recovery
        losses['window'] = self.window_loss(pred, target, source) * self.window_weight

        # Total
        total = sum(losses.values())
        losses['total'] = total

        return total, losses


# ============================================================================
# DATASET
# ============================================================================

class HDRDataset(Dataset):
    """Dataset for HDR real estate image pairs."""

    def __init__(self, data_dir, resolution=512, augment=True):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.augment = augment

        # Find all source images
        self.pairs = []
        for src_file in self.data_dir.glob('*_src.*'):
            base = src_file.stem.replace('_src', '')
            tar_file = None
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                candidate = self.data_dir / f"{base}_tar{ext}"
                if candidate.exists():
                    tar_file = candidate
                    break

            if tar_file:
                self.pairs.append((src_file, tar_file))

        print(f"Found {len(self.pairs)} image pairs in {data_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, tar_path = self.pairs[idx]

        # Load images
        src = Image.open(src_path).convert('RGB')
        tar = Image.open(tar_path).convert('RGB')

        # Resize to same size
        src = src.resize((self.resolution, self.resolution), Image.LANCZOS)
        tar = tar.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Augmentation
        if self.augment:
            if torch.rand(1) > 0.5:
                src = TF.hflip(src)
                tar = TF.hflip(tar)
            if torch.rand(1) > 0.5:
                src = TF.vflip(src)
                tar = TF.vflip(tar)

        # To tensor
        src = TF.to_tensor(src)
        tar = TF.to_tensor(tar)

        return src, tar


# ============================================================================
# MODEL WITH SAFE RESIDUAL
# ============================================================================

class FeatureExtractor(nn.Module):
    """Extract depth proxy, edge, and saturation features."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, rgb):
        # Luminance
        lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

        # Saturation
        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]
        sat = (max_rgb - min_rgb) / (max_rgb + 1e-8)

        # Depth proxy (bright, desaturated = far/windows)
        depth = lum * (1 - sat)
        depth = F.avg_pool2d(F.pad(depth, (7,)*4, mode='reflect'), 15, stride=1)
        B = depth.shape[0]
        flat = depth.view(B, -1)
        d_min = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        d_max = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        # Edges
        gx = F.conv2d(lum, self.sobel_x, padding=1)
        gy = F.conv2d(lum, self.sobel_y, padding=1)
        edges = torch.sqrt(gx**2 + gy**2 + 1e-8)
        edges = edges / (edges.max() + 1e-8)

        return {'depth': depth, 'edge': edges, 'saturation': sat}


class SafeRestormer(nn.Module):
    """
    Restormer with safe residual learning to prevent black spots.

    Key: Clamp residual to prevent extreme negative values.
    """

    def __init__(self, dim=48, num_blocks=[4,6,6,8], num_refinement_blocks=4,
                 heads=[1,2,4,8], ffn_expansion_factor=2.66, bias=False,
                 max_negative_residual=0.7):
        super().__init__()

        self.max_negative_residual = max_negative_residual
        self.feature_extractor = FeatureExtractor()

        in_channels = 6  # RGB + depth + edge + sat

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1, bias=bias)

        # Encoder
        self.encoder_level1 = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])], True)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = TransformerStage(
            [TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])], True)
        self.down2_3 = Downsample(dim*2)
        self.encoder_level3 = TransformerStage(
            [TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])], True)
        self.down3_4 = Downsample(dim*4)

        # Bottleneck
        self.latent = TransformerStage(
            [TransformerBlock(dim*8, heads[3], ffn_expansion_factor, bias) for _ in range(num_blocks[3])], True)

        # Decoder
        self.up4_3 = Upsample(dim*8)
        self.reduce_chan_level3 = nn.Conv2d(dim*8, dim*4, 1, bias=bias)
        self.decoder_level3 = TransformerStage(
            [TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])], True)

        self.up3_2 = Upsample(dim*4)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.decoder_level2 = TransformerStage(
            [TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])], True)

        self.up2_1 = Upsample(dim*2)
        self.reduce_chan_level1 = nn.Conv2d(dim*2, dim, 1, bias=bias)
        self.decoder_level1 = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])], True)

        # Refinement
        self.refinement = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_refinement_blocks)], True)

        self.output = nn.Conv2d(dim, 3, 3, padding=1, bias=bias)

    def forward(self, rgb):
        # Extract features
        features = self.feature_extractor(rgb)
        x = torch.cat([rgb, features['depth'], features['edge'], features['saturation']], dim=1)

        # Encoder
        x = self.patch_embed(x)
        enc1 = self.encoder_level1(x)
        x = self.down1_2(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down2_3(enc2)
        enc3 = self.encoder_level3(x)
        x = self.down3_4(enc3)

        # Bottleneck
        x = self.latent(x)

        # Decoder
        x = self.up4_3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.reduce_chan_level3(x)
        x = self.decoder_level3(x)

        x = self.up3_2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x)

        x = self.up2_1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.reduce_chan_level1(x)
        x = self.decoder_level1(x)

        # Refinement
        x = self.refinement(x)

        # SAFE RESIDUAL: Clamp to prevent black spots
        residual = self.output(x)

        # Limit how much we can darken (max 70% reduction)
        min_residual = -self.max_negative_residual * rgb
        residual = torch.max(residual, min_residual)

        return torch.clamp(residual + rgb, 0, 1)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    loss_components = {}

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for src, tar in pbar:
        src, tar = src.to(device), tar.to(device)

        optimizer.zero_grad()

        pred = model(src)
        loss, losses = criterion(pred, tar, src)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            loss_components[k] = loss_components.get(k, 0) + v.item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for src, tar in loader:
        src, tar = src.to(device), tar.to(device)
        pred = model(src)
        loss, _ = criterion(pred, tar, src)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description='SOTA HDR Training')
    parser.add_argument('--data_dir', type=str, default='images')
    parser.add_argument('--output_dir', type=str, default='outputs_sota_hdr')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Stop if no improvement for N epochs')
    parser.add_argument('--min_epochs', type=int, default=20,
                        help='Minimum epochs before early stopping kicks in')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SOTA HDR TRAINING")
    print("=" * 70)
    print(f"Data: {args.data_dir}")
    print(f"Resolution: {args.resolution}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print("=" * 70)

    # Dataset
    dataset = HDRDataset(args.data_dir, args.resolution, augment=True)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = SafeRestormer().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Loss and optimizer
    criterion = SOTALoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'loss_components': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, loss_components = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['loss_components'].append(loss_components)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        print(f"  Components: {', '.join(f'{k}={v:.4f}' for k, v in loss_components.items())}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'checkpoint_best.pt')
            print(f"  *** New best model saved ***")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, output_dir / 'checkpoint_last.pt')

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping check
        if epoch >= args.min_epochs and patience_counter >= args.early_stopping_patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING at epoch {epoch}")
            print(f"No improvement for {args.early_stopping_patience} epochs")
            print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            print(f"{'='*70}")
            break

    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Total epochs: {epoch}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
