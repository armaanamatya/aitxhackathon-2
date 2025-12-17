#!/usr/bin/env python3
"""
MiDaS Depth + Multi-Feature Window Recovery - OPTIMAL SOLUTION
==============================================================

The most robust solution combining:
1. TRUE depth from MiDaS (not proxy)
2. Edge detection for structural preservation
3. Saturation for washed-out window detection
4. Delta-aware loss for adaptive fixing/preservation

This is the fully-fledged optimal solution for HDR real estate.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.delta_aware_loss import create_robust_window_loss


# ============================================================================
# MiDaS DEPTH ESTIMATOR
# ============================================================================

class MiDaSDepth(nn.Module):
    """MiDaS depth estimation - loads once, reuses for all images."""

    def __init__(self, model_type="MiDaS_small"):
        super().__init__()
        self.model_type = model_type
        self.model = None
        self.transform = None
        self._initialized = False

    def _init(self, device):
        if self._initialized:
            return

        print(f"Loading MiDaS ({self.model_type})...")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            self.model = self.model.to(device).eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if "DPT" in self.model_type:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            for p in self.model.parameters():
                p.requires_grad = False

            self._initialized = True
            print("MiDaS loaded successfully")
        except Exception as e:
            print(f"MiDaS load failed: {e}, using fallback")
            self._use_fallback = True
            self._initialized = True

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from RGB.

        Args:
            rgb: [B, 3, H, W] in [0, 1]
        Returns:
            depth: [B, 1, H, W] normalized to [0, 1]
        """
        self._init(rgb.device)

        if hasattr(self, '_use_fallback') and self._use_fallback:
            return self._fallback_depth(rgb)

        B, C, H, W = rgb.shape

        # Convert to numpy for MiDaS
        rgb_np = (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        depths = []
        for i in range(B):
            # Apply transform
            img_input = self.transform(rgb_np[i])
            if isinstance(img_input, dict):
                img_input = img_input.get('pixel_values', list(img_input.values())[0])
            if not isinstance(img_input, torch.Tensor):
                img_input = torch.from_numpy(np.array(img_input))

            img_input = img_input.to(rgb.device)
            if img_input.dim() == 3:
                img_input = img_input.unsqueeze(0)

            # Forward
            depth = self.model(img_input)

            # Resize to original
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            depths.append(depth)

        depth = torch.cat(depths, dim=0)

        # Normalize per image
        B = depth.shape[0]
        flat = depth.view(B, -1)
        d_min = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        d_max = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        return depth

    def _fallback_depth(self, rgb):
        """Luminance-based fallback."""
        lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]
        sat = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        depth = lum * (1 - sat)

        # Smooth
        depth = F.avg_pool2d(F.pad(depth, (7,)*4, mode='reflect'), 15, stride=1)

        B = depth.shape[0]
        flat = depth.view(B, -1)
        d_min = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        d_max = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (depth - d_min) / (d_max - d_min + 1e-8)


# ============================================================================
# FEATURE EXTRACTOR (Edge + Saturation)
# ============================================================================

class FeatureExtractor(nn.Module):
    """Extract edge and saturation features."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_edges(self, rgb):
        lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        gx = F.conv2d(lum, self.sobel_x, padding=1)
        gy = F.conv2d(lum, self.sobel_y, padding=1)
        edges = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return edges / (edges.max() + 1e-8)

    def compute_saturation(self, rgb):
        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]
        return (max_rgb - min_rgb) / (max_rgb + 1e-8)

    def forward(self, rgb):
        return {
            'edge': self.compute_edges(rgb),
            'saturation': self.compute_saturation(rgb)
        }


# ============================================================================
# OPTIMAL RESTORMER WITH MIDAS DEPTH
# ============================================================================

from training.restormer import (
    LayerNorm2d, TransformerBlock, TransformerStage, Downsample, Upsample
)

class OptimalRestormer(nn.Module):
    """
    Optimal Restormer for HDR real estate.

    Input: RGB(3) + MiDaS_Depth(1) + Edge(1) + Saturation(1) = 6 channels
    Output: RGB(3)
    """

    def __init__(
        self,
        dim: int = 48,
        num_blocks: list = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: list = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()

        in_channels = 6  # RGB + depth + edge + saturation
        out_channels = 3

        # Feature extractors
        self.midas = MiDaSDepth("MiDaS_small")
        self.feature_extractor = FeatureExtractor()

        # Restormer architecture
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # Encoder
        self.encoder_level1 = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
             for _ in range(num_blocks[0])],
            use_checkpoint=True
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = TransformerStage(
            [TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
             for _ in range(num_blocks[1])],
            use_checkpoint=True
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = TransformerStage(
            [TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
             for _ in range(num_blocks[2])],
            use_checkpoint=True
        )
        self.down3_4 = Downsample(dim * 4)

        # Bottleneck
        self.latent = TransformerStage(
            [TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias)
             for _ in range(num_blocks[3])],
            use_checkpoint=True
        )

        # Decoder
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = TransformerStage(
            [TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
             for _ in range(num_blocks[2])],
            use_checkpoint=True
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = TransformerStage(
            [TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
             for _ in range(num_blocks[1])],
            use_checkpoint=True
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
             for _ in range(num_blocks[0])],
            use_checkpoint=True
        )

        # Refinement
        self.refinement = TransformerStage(
            [TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
             for _ in range(num_refinement_blocks)],
            use_checkpoint=True
        )

        # Output
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            rgb: [B, 3, H, W] RGB input in [0, 1]
        Returns:
            [B, 3, H, W] enhanced RGB
        """
        # Extract features
        with torch.no_grad():
            depth = self.midas(rgb)
            features = self.feature_extractor(rgb)

        # Concatenate: RGB + depth + edge + saturation
        x = torch.cat([rgb, depth, features['edge'], features['saturation']], dim=1)

        # Patch embed
        x = self.patch_embed(x)

        # Encoder
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

        # Output with residual
        x = self.output(x) + rgb

        return x


def create_optimal_restormer(size: str = "base") -> OptimalRestormer:
    """Create optimal restormer."""
    configs = {
        "small": {"dim": 32, "num_blocks": [3, 4, 4, 6], "num_refinement_blocks": 3, "heads": [1, 2, 4, 8]},
        "base": {"dim": 48, "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4, "heads": [1, 2, 4, 8]},
    }
    return OptimalRestormer(**configs.get(size, configs["base"]))


# ============================================================================
# DATASET
# ============================================================================

class HDRDataset(Dataset):
    def __init__(self, jsonl_path, resolution=512, augment=False):
        self.resolution = resolution
        self.augment = augment

        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        input_img = TF.resize(input_img, (self.resolution, self.resolution))
        target_img = TF.resize(target_img, (self.resolution, self.resolution))

        if self.augment:
            if torch.rand(1) > 0.5:
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)
            if torch.rand(1) > 0.7:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                input_img = TF.adjust_brightness(input_img, factor)

        return TF.to_tensor(input_img), TF.to_tensor(target_img)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, accum_steps):
    model.train()
    total_loss = 0
    zone_losses = {'needs_heavy_fix': 0, 'needs_light_fix': 0, 'preserve': 0}

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (input_img, target_img) in enumerate(pbar):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        with autocast('cuda'):
            output = model(input_img)
            output = torch.clamp(output, 0, 1)
            loss, components, _ = criterion(output, target_img, input_img, return_components=True)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

        for zone in zone_losses:
            key = f'{zone}_l1'
            if key in components:
                val = components[key]
                zone_losses[zone] += val.item() if isinstance(val, torch.Tensor) else val

        heavy = components.get('needs_heavy_fix_l1', 0)
        preserve = components.get('preserve_l1', 0)
        if isinstance(heavy, torch.Tensor): heavy = heavy.item()
        if isinstance(preserve, torch.Tensor): preserve = preserve.item()

        pbar.set_postfix({'loss': f"{loss.item()*accum_steps:.4f}", 'fix': f"{heavy:.3f}", 'pres': f"{preserve:.3f}"})

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in zone_losses.items()}


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'l1': [], 'psnr': []}
    zone_metrics = {'needs_heavy_fix': [], 'needs_light_fix': [], 'preserve': []}

    with torch.no_grad():
        for input_img, target_img in tqdm(dataloader, desc="Val"):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            output = model(input_img)
            output = torch.clamp(output, 0, 1)

            loss, components, _ = criterion(output, target_img, input_img, return_components=True)
            total_loss += loss.item()

            l1 = F.l1_loss(output, target_img).item()
            mse = F.mse_loss(output, target_img)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
            metrics['l1'].append(l1)
            metrics['psnr'].append(psnr)

            for zone in zone_metrics:
                key = f'{zone}_l1'
                if key in components:
                    val = components[key]
                    zone_metrics[zone].append(val.item() if isinstance(val, torch.Tensor) else val)

    return (
        total_loss / len(dataloader),
        np.mean(metrics['l1']),
        np.mean(metrics['psnr']),
        {k: np.mean(v) if v else 0 for k, v in zone_metrics.items()}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base'])
    parser.add_argument('--output_dir', type=str, default='outputs_midas_optimal')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--preset', type=str, default='robust')
    args = parser.parse_args()

    print("=" * 70)
    print("OPTIMAL HDR REAL ESTATE - MiDaS DEPTH + MULTI-FEATURE")
    print("=" * 70)
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Model: {args.model_size}")
    print(f"Input: RGB(3) + MiDaS_Depth(1) + Edge(1) + Saturation(1) = 6 channels")
    print(f"Batch: {args.batch_size} x {args.accum_steps} = {args.batch_size * args.accum_steps} effective")
    print(f"Epochs: {args.epochs}")
    print()
    print("OPTIMAL FEATURES:")
    print("  ✓ MiDaS true depth (scene geometry)")
    print("  ✓ Edge detection (structural preservation)")
    print("  ✓ Saturation (washed-out window detection)")
    print("  ✓ Delta-aware loss (fix bad, preserve good)")
    print("  ✓ Gradient checkpointing (memory efficient)")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Data
    train_dataset = HDRDataset('data_splits/proper_split/train.jsonl', args.resolution, augment=True)
    val_dataset = HDRDataset('data_splits/proper_split/val.jsonl', args.resolution)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples")
    print()

    # Model
    model = create_optimal_restormer(args.model_size).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")
    print()

    # Loss
    criterion = create_robust_window_loss(args.preset)
    print("Loss: Robust Delta-Aware")
    print(f"  Heavy fix weight: {criterion.delta_loss.weights['needs_heavy_fix']}")
    print(f"  Preserve weight: {criterion.delta_loss.weights['preserve']}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (args.epochs - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    # Resume
    start_epoch = 0
    best_val_l1 = float('inf')
    best_heavy_fix = float('inf')

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_l1 = ckpt.get('best_val_l1', float('inf'))
        best_heavy_fix = ckpt.get('best_heavy_fix', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    history = {'train_loss': [], 'val_loss': [], 'val_l1': [], 'zone_metrics': []}

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        train_loss, train_zones = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch+1, args.accum_steps
        )

        val_loss, val_l1, val_psnr, val_zones = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val: L1={val_l1:.4f}, PSNR={val_psnr:.2f}dB")
        print(f"Zone L1: fix={val_zones['needs_heavy_fix']:.4f}, preserve={val_zones['preserve']:.4f}")
        print(f"LR: {lr:.6f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_l1)
        history['zone_metrics'].append(val_zones)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_l1': val_l1,
            'val_psnr': val_psnr,
            'zone_metrics': val_zones,
            'best_val_l1': best_val_l1,
            'best_heavy_fix': best_heavy_fix,
            'config': config,
        }

        torch.save(ckpt, output_dir / 'checkpoint_last.pt')

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            ckpt['best_val_l1'] = best_val_l1
            torch.save(ckpt, output_dir / 'checkpoint_best.pt')
            print(f"  ★ New best overall: {val_l1:.4f}")

        if val_zones['needs_heavy_fix'] < best_heavy_fix:
            best_heavy_fix = val_zones['needs_heavy_fix']
            ckpt['best_heavy_fix'] = best_heavy_fix
            torch.save(ckpt, output_dir / 'checkpoint_best_window.pt')
            print(f"  ★ New best window fix: {val_zones['needs_heavy_fix']:.4f}")

        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Best Window Fix: {best_heavy_fix:.4f}")
    print(f"Output: {output_dir}/")


if __name__ == '__main__':
    main()
