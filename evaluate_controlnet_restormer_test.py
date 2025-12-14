#!/usr/bin/env python3
"""
ControlNet-Restormer Test Set Evaluation
=========================================
Evaluates ensemble of 3 CV folds on held-out test set.

Usage:
    python3 evaluate_controlnet_restormer_test.py \
        --model_dir outputs_controlnet_restormer_512_cv \
        --test_jsonl data_splits/test.jsonl \
        --output_dir evaluation_results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add model path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer


# ============================================================================
# ARCHITECTURE (same as training)
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
    """Restormer with ControlNet-style training."""

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
                 freeze_base: bool = True):
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        # Base model (frozen during inference)
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

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Trainable model
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

        # Zero convolution
        self.zero_conv_out = ZeroConv(out_channels, out_channels)

    def forward(self, x):
        base_out = self.base_model(x)
        trainable_out = self.trainable_model(x)
        adaptation = self.zero_conv_out(trainable_out)
        return base_out + adaptation


# ============================================================================
# DATASET
# ============================================================================

class TestDataset(Dataset):
    """Test dataset (no augmentation)."""

    def __init__(self, jsonl_path: str, base_dir: str = '.', resolution: int = 384):
        self.base_dir = Path(base_dir)
        self.resolution = resolution

        self.pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.pairs.append(json.loads(line))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        src_path = self.base_dir / pair['src']
        tar_path = self.base_dir / pair['tar']

        src = cv2.imread(str(src_path))
        tar = cv2.imread(str(tar_path))

        if src is None or tar is None:
            raise ValueError(f"Failed to load: {src_path} or {tar_path}")

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        # Resize
        src = cv2.resize(src, (self.resolution, self.resolution))
        tar = cv2.resize(tar, (self.resolution, self.resolution))

        # To tensor
        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar, pair['src']


# ============================================================================
# METRICS
# ============================================================================

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Calculate SSIM."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.mean().item()


# ============================================================================
# EVALUATION
# ============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """Load ControlNet-Restormer from checkpoint."""
    print(f"  Loading {checkpoint_path}...")

    # Create model
    model = ControlNetRestormer(
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        use_checkpointing=False,
        freeze_base=False  # Load all weights
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_single_model(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    """Evaluate single model."""
    total_psnr = 0
    total_ssim = 0
    total_l1 = 0

    with torch.no_grad():
        for src, tar, _ in tqdm(dataloader, desc="Evaluating"):
            src, tar = src.to(device), tar.to(device)

            out = model(src)

            psnr = calculate_psnr(out, tar)
            ssim = calculate_ssim(out, tar)
            l1 = F.l1_loss(out, tar).item()

            total_psnr += psnr
            total_ssim += ssim
            total_l1 += l1

    n = len(dataloader)
    return {
        'psnr': total_psnr / n,
        'ssim': total_ssim / n,
        'l1': total_l1 / n
    }


def evaluate_ensemble(models: List[nn.Module], dataloader: DataLoader, device: str, save_dir: Path = None) -> dict:
    """Evaluate ensemble of models."""
    total_psnr = 0
    total_ssim = 0
    total_l1 = 0

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for src, tar, names in tqdm(dataloader, desc="Ensemble"):
            src, tar = src.to(device), tar.to(device)

            # Ensemble prediction (average)
            outputs = [model(src) for model in models]
            ensemble_out = torch.stack(outputs).mean(dim=0)

            psnr = calculate_psnr(ensemble_out, tar)
            ssim = calculate_ssim(ensemble_out, tar)
            l1 = F.l1_loss(ensemble_out, tar).item()

            total_psnr += psnr
            total_ssim += ssim
            total_l1 += l1

            # Save visualizations
            if save_dir:
                for i, name in enumerate(names):
                    # Convert to numpy
                    src_np = src[i].permute(1, 2, 0).cpu().numpy()
                    tar_np = tar[i].permute(1, 2, 0).cpu().numpy()
                    out_np = ensemble_out[i].permute(1, 2, 0).cpu().numpy()

                    # To uint8
                    src_np = (src_np * 255).clip(0, 255).astype(np.uint8)
                    tar_np = (tar_np * 255).clip(0, 255).astype(np.uint8)
                    out_np = (out_np * 255).clip(0, 255).astype(np.uint8)

                    # RGB to BGR
                    src_np = cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR)
                    tar_np = cv2.cvtColor(tar_np, cv2.COLOR_RGB2BGR)
                    out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

                    # Save
                    name_stem = Path(name).stem
                    cv2.imwrite(str(save_dir / f'{name_stem}_input.jpg'), src_np)
                    cv2.imwrite(str(save_dir / f'{name_stem}_target.jpg'), tar_np)
                    cv2.imwrite(str(save_dir / f'{name_stem}_output.jpg'), out_np)

                    # Comparison grid
                    grid = np.hstack([src_np, out_np, tar_np])
                    cv2.imwrite(str(save_dir / f'{name_stem}_comparison.jpg'), grid)

    n = len(dataloader)
    return {
        'psnr': total_psnr / n,
        'ssim': total_ssim / n,
        'l1': total_l1 / n
    }


def main():
    parser = argparse.ArgumentParser(description="ControlNet-Restormer Test Evaluation")
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory (CV output)')
    parser.add_argument('--test_jsonl', type=str, default='data_splits/test.jsonl', help='Test JSONL')
    parser.add_argument('--output_dir', type=str, default='evaluation_controlnet_restormer', help='Output directory')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution (must match training)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    args = parser.parse_args()

    print("=" * 80)
    print("CONTROLNET-RESTORMER TEST SET EVALUATION")
    print("=" * 80)

    # Load test dataset
    project_root = Path(args.test_jsonl).parent.parent
    test_dataset = TestDataset(args.test_jsonl, base_dir=str(project_root), resolution=args.resolution)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"\n📊 Test set: {len(test_dataset)} samples")
    print(f"📁 Model dir: {args.model_dir}")

    # Load models from all folds
    model_dir = Path(args.model_dir)
    models = []
    fold_results = []

    for fold in range(1, 4):
        fold_dir = model_dir / f'fold_{fold}'
        checkpoint_path = fold_dir / 'checkpoint_best.pt'

        if not checkpoint_path.exists():
            print(f"⚠️  Warning: {checkpoint_path} not found, skipping fold {fold}")
            continue

        print(f"\n📥 Loading Fold {fold}...")
        model = load_model(str(checkpoint_path), args.device)
        models.append(model)

        # Evaluate single model
        print(f"🧪 Evaluating Fold {fold}...")
        metrics = evaluate_single_model(model, test_loader, args.device)
        fold_results.append({
            'fold': fold,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'l1': metrics['l1']
        })

        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  L1: {metrics['l1']:.4f}")

    # Ensemble evaluation
    print(f"\n{'='*80}")
    print("ENSEMBLE EVALUATION (Average of all folds)")
    print(f"{'='*80}")

    output_dir = Path(args.output_dir)
    ensemble_metrics = evaluate_ensemble(models, test_loader, args.device, save_dir=output_dir / 'visualizations')

    print(f"\n📊 Ensemble Results:")
    print(f"  PSNR: {ensemble_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {ensemble_metrics['ssim']:.4f}")
    print(f"  L1: {ensemble_metrics['l1']:.4f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\n📈 Individual Folds:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['psnr']:.2f} dB (SSIM: {r['ssim']:.4f})")

    avg_fold_psnr = np.mean([r['psnr'] for r in fold_results])
    print(f"\n  Mean (single fold): {avg_fold_psnr:.2f} dB")
    print(f"  Ensemble: {ensemble_metrics['psnr']:.2f} dB")
    print(f"  Ensemble gain: +{ensemble_metrics['psnr'] - avg_fold_psnr:.2f} dB")

    # Save results
    results = {
        'ensemble': ensemble_metrics,
        'folds': fold_results,
        'ensemble_gain': float(ensemble_metrics['psnr'] - avg_fold_psnr)
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_dir}/test_results.json")
    print(f"📸 Visualizations saved to: {output_dir}/visualizations/")
    print("=" * 80)


if __name__ == '__main__':
    main()
