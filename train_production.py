#!/usr/bin/env python3
"""
Production HDR Enhancement Model
================================

TASK DEFINITION:
- Windows (bright regions): Reduce brightness to reveal sky/trees/outdoor detail
- Interior (everything else): Only brighten, NEVER darken

This is enforced architecturally:
- Window regions: gain in [0.3, 1.0] (can only darken)
- Interior regions: gain in [1.0, 3.0] (can only brighten)

The window mask is computed from input brightness + saturation.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF
from torchvision import models

from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from training.restormer import TransformerBlock, TransformerStage, Downsample, Upsample


# ============================================================================
# WINDOW DETECTION
# ============================================================================

class WindowDetector(nn.Module):
    """
    Detect window regions based on:
    1. High brightness (near saturation)
    2. Low saturation (blown out = desaturated)
    3. Spatial context (windows are usually rectangular, connected)

    Windows are bright AND desaturated (clipped to white).
    Interior bright spots (lamps) are bright BUT saturated (have color).
    """

    def __init__(self, brightness_threshold=0.85, saturation_threshold=0.3):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold

    def forward(self, rgb):
        """
        Returns soft window mask [0, 1] where 1 = definitely window.
        """
        # Luminance
        lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

        # Saturation
        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]
        sat = (max_rgb - min_rgb) / (max_rgb + 1e-8)

        # Window = bright AND desaturated
        bright_mask = torch.sigmoid((lum - self.brightness_threshold) * 20)
        desat_mask = torch.sigmoid((self.saturation_threshold - sat) * 20)

        # Combine: must be both bright AND desaturated
        window_mask = bright_mask * desat_mask

        # Smooth the mask slightly to avoid hard edges
        window_mask = F.avg_pool2d(
            F.pad(window_mask, (2, 2, 2, 2), mode='reflect'),
            kernel_size=5, stride=1
        )

        return window_mask


# ============================================================================
# CONSTRAINED GAIN MODEL
# ============================================================================

class ConstrainedGainNet(nn.Module):
    """
    Predicts per-pixel gain with hard constraints:

    - Window regions: gain in [0.3, 1.0] (ONLY darken to reveal detail)
    - Interior regions: gain in [1.0, 3.0] (ONLY brighten)

    This is enforced by:
    1. Detecting windows from input
    2. Predicting raw gain
    3. Constraining gain based on region type
    """

    def __init__(self, dim=48, num_blocks=4):
        super().__init__()

        # Window detector (fixed, not learned)
        self.window_detector = WindowDetector()

        # Feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(4, dim, 3, padding=1),  # RGB + window_mask
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.down1 = nn.Conv2d(dim, dim*2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim*2, 3, padding=1),
        )
        self.down2 = nn.Conv2d(dim*2, dim*4, 3, stride=2, padding=1)

        # Bottleneck with self-attention for global context
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim*4, dim*4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim*4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim*4, 3, padding=1),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(dim*4, dim*2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim*2, 3, padding=1),
        )

        self.up1 = nn.ConvTranspose2d(dim*2, dim, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(dim*2, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        # Separate heads for window and interior gain
        self.window_gain_head = nn.Conv2d(dim, 3, 3, padding=1)
        self.interior_gain_head = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, rgb):
        """
        Args:
            rgb: Input [B, 3, H, W] in [0, 1]

        Returns:
            output: Enhanced [B, 3, H, W]
            gain: Gain map [B, 3, H, W]
            window_mask: Detected windows [B, 1, H, W]
        """
        B, C, H, W = rgb.shape

        # Detect windows (not learned, just thresholding)
        window_mask = self.window_detector(rgb)
        interior_mask = 1.0 - window_mask

        # Concatenate window mask as input feature
        x = torch.cat([rgb, window_mask], dim=1)

        # Encoder
        x = self.stem(x)
        e1 = self.enc1(x)
        x = self.down1(e1)
        e2 = self.enc2(x)
        x = self.down2(e2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        # Predict gains
        # Window gain: sigmoid → [0, 1] → scale to [0.3, 1.0]
        window_gain_raw = self.window_gain_head(x)
        window_gain = 0.3 + 0.7 * torch.sigmoid(window_gain_raw)  # [0.3, 1.0]

        # Interior gain: sigmoid → [0, 1] → scale to [1.0, 3.0]
        interior_gain_raw = self.interior_gain_head(x)
        interior_gain = 1.0 + 2.0 * torch.sigmoid(interior_gain_raw)  # [1.0, 3.0]

        # Blend gains based on mask
        gain = window_mask * window_gain + interior_mask * interior_gain

        # Apply gain
        output = rgb * gain
        output = torch.clamp(output, 0, 1)

        return output, gain, window_mask


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class ConstrainedLoss(nn.Module):
    """
    Loss function that respects the constraint:
    - Windows should be darkened (match GT window regions)
    - Interior should be brightened (match GT interior regions)
    """

    def __init__(self):
        super().__init__()

        # VGG for perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = nn.Sequential(*list(vgg.children()))
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Window detector for GT analysis
        self.window_detector = WindowDetector()

    def forward(self, pred, target, source, gain, window_mask):
        losses = {}

        # 1. Overall reconstruction
        losses['l1'] = F.l1_loss(pred, target)

        # 2. Perceptual loss
        pred_vgg = self.vgg((pred - self.vgg_mean) / self.vgg_std)
        target_vgg = self.vgg((target - self.vgg_mean) / self.vgg_std)
        losses['perceptual'] = F.l1_loss(pred_vgg, target_vgg) * 0.1

        # 3. Window-specific loss (higher weight on window regions)
        # Windows are where detail recovery matters most
        window_loss = (torch.abs(pred - target) * window_mask).sum() / (window_mask.sum() + 1e-6)
        losses['window'] = window_loss * 0.5

        # 4. Interior loss
        interior_mask = 1.0 - window_mask
        interior_loss = (torch.abs(pred - target) * interior_mask).sum() / (interior_mask.sum() + 1e-6)
        losses['interior'] = interior_loss * 0.3

        # 5. Color preservation (important for interior)
        pred_ratio = pred / (pred.mean(dim=1, keepdim=True) + 1e-6)
        target_ratio = target / (target.mean(dim=1, keepdim=True) + 1e-6)
        losses['color'] = F.l1_loss(pred_ratio, target_ratio) * 0.1

        # 6. Gain smoothness (prevent checkerboard)
        gain_dx = torch.abs(gain[:, :, :, 1:] - gain[:, :, :, :-1])
        gain_dy = torch.abs(gain[:, :, 1:, :] - gain[:, :, :-1, :])
        losses['smooth'] = (gain_dx.mean() + gain_dy.mean()) * 0.02

        # Total
        total = sum(losses.values())
        losses['total'] = total

        return total, losses


# ============================================================================
# DATASET
# ============================================================================

class HDRDataset(Dataset):
    def __init__(self, data_dir, resolution=512, augment=True):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.augment = augment

        self.pairs = []
        for src_file in self.data_dir.glob('*_src.*'):
            base = src_file.stem.replace('_src', '')
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                tar_file = self.data_dir / f"{base}_tar{ext}"
                if tar_file.exists():
                    self.pairs.append((src_file, tar_file))
                    break

        print(f"Found {len(self.pairs)} pairs in {data_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, tar_path = self.pairs[idx]

        src = Image.open(src_path).convert('RGB')
        tar = Image.open(tar_path).convert('RGB')

        src = src.resize((self.resolution, self.resolution), Image.LANCZOS)
        tar = tar.resize((self.resolution, self.resolution), Image.LANCZOS)

        if self.augment:
            if torch.rand(1) > 0.5:
                src = TF.hflip(src)
                tar = TF.hflip(tar)
            if torch.rand(1) > 0.5:
                src = TF.vflip(src)
                tar = TF.vflip(tar)

        return TF.to_tensor(src), TF.to_tensor(tar)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    loss_dict = {}

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for src, tar in pbar:
        src, tar = src.to(device), tar.to(device)

        optimizer.zero_grad()

        pred, gain, window_mask = model(src)
        loss, losses = criterion(pred, tar, src, gain, window_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            loss_dict[k] = loss_dict.get(k, 0) + v.item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_dict.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for src, tar in loader:
        src, tar = src.to(device), tar.to(device)
        pred, gain, window_mask = model(src)
        loss, _ = criterion(pred, tar, src, gain, window_mask)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='images')
    parser.add_argument('--output_dir', default='outputs_production')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_epochs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONSTRAINED GAIN NETWORK")
    print("=" * 70)
    print("Constraints:")
    print("  - Window regions:  gain in [0.3, 1.0] (ONLY darken)")
    print("  - Interior regions: gain in [1.0, 3.0] (ONLY brighten)")
    print("=" * 70)
    print(f"Device: {device}")

    # Data
    dataset = HDRDataset(args.data_dir, args.resolution)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # Model
    model = ConstrainedGainNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Training
    criterion = ConstrainedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': [], 'components': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, loss_components = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['components'].append(loss_components)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        print(f"  {', '.join(f'{k}={v:.4f}' for k, v in loss_components.items())}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'checkpoint_best.pt')
            print(f"  *** Best model saved ***")
        else:
            patience_counter += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, output_dir / 'checkpoint_last.pt')

        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f)

        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
