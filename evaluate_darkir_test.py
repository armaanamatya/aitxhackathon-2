#!/usr/bin/env python3
"""
DarkIR Test Set Evaluation - Zero Leakage
==========================================
Evaluate ensemble of 3 CV folds on held-out test set.
Test set was never seen during training/validation.

Author: Top MLE
Date: 2025-12-13
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add DarkIR to path
sys.path.insert(0, str(Path(__file__).parent / 'DarkIR'))
from archs.DarkIR import DarkIR


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate SSIM (simplified)"""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, 11, 1, 5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, 11, 1, 5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


class TestDataset(Dataset):
    """Test dataset"""

    def __init__(self, jsonl_path: str, base_dir: str, resolution: int):
        self.base_dir = base_dir
        self.resolution = resolution

        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        src_path = os.path.join(self.base_dir, pair['src'])
        tar_path = os.path.join(self.base_dir, pair['tar'])

        src = cv2.imread(src_path)
        tar = cv2.imread(tar_path)

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        # Resize
        src = cv2.resize(src, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        tar = cv2.resize(tar, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # To tensor
        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar, pair['src']


def load_fold_model(fold_dir: Path, model_size: str, device: str):
    """Load model from fold checkpoint"""
    width = 32 if model_size == 'm' else 64
    model = DarkIR(
        img_channel=3,
        width=width,
        middle_blk_num_enc=2,
        middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3],
        dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True
    ).to(device)

    checkpoint_path = fold_dir / 'checkpoint_best.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_ensemble(models: List[nn.Module],
                      dataloader: DataLoader,
                      device: str,
                      save_dir: Path = None) -> dict:
    """Evaluate ensemble of models"""

    total_psnr = 0
    total_ssim = 0
    total_l1 = 0
    count = 0

    results = []

    with torch.no_grad():
        for src, tar, names in tqdm(dataloader, desc="Evaluating"):
            src, tar = src.to(device), tar.to(device)

            # Ensemble prediction (average of all folds)
            outputs = []
            for model in models:
                out = model(src)
                outputs.append(out)

            # Average ensemble
            ensemble_out = torch.stack(outputs).mean(dim=0)

            # Metrics
            for i in range(src.shape[0]):
                psnr = calculate_psnr(ensemble_out[i:i+1], tar[i:i+1])
                ssim = calculate_ssim(ensemble_out[i:i+1], tar[i:i+1])
                l1 = F.l1_loss(ensemble_out[i:i+1], tar[i:i+1]).item()

                total_psnr += psnr
                total_ssim += ssim
                total_l1 += l1
                count += 1

                results.append({
                    'image': names[i],
                    'psnr': psnr,
                    'ssim': ssim,
                    'l1': l1
                })

                # Save visual results if requested
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # Convert to numpy
                    src_np = (src[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    tar_np = (tar[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    out_np = (ensemble_out[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    # Concatenate
                    combined = np.hstack([src_np, out_np, tar_np])
                    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

                    # Save
                    name = Path(names[i]).stem
                    cv2.imwrite(str(save_dir / f"{name}_comparison.jpg"), combined)

    summary = {
        'avg_psnr': total_psnr / count,
        'avg_ssim': total_ssim / count,
        'avg_l1': total_l1 / count,
        'per_image': results
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate DarkIR ensemble on test set")
    parser.add_argument('--cv_dir', type=str, required=True,
                        help='Directory with CV fold checkpoints')
    parser.add_argument('--test_jsonl', type=str, default='data_splits/test.jsonl',
                        help='Test set JSONL')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution')
    parser.add_argument('--model_size', type=str, default='m', choices=['m', 'l'],
                        help='Model size')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of folds')
    parser.add_argument('--save_visuals', action='store_true',
                        help='Save visual comparisons')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    args = parser.parse_args()

    print("="*80)
    print("DARKIR TEST SET EVALUATION (ZERO LEAKAGE)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  CV dir: {args.cv_dir}")
    print(f"  Test set: {args.test_jsonl}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Model: DarkIR-{args.model_size}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Output: {args.output_dir}\n")

    # Load models from all folds
    print("📥 Loading models from all folds...")
    models = []
    for fold_num in range(1, args.n_folds + 1):
        fold_dir = Path(args.cv_dir) / f"fold_{fold_num}"
        print(f"   Loading fold {fold_num}... ", end="")
        model = load_fold_model(fold_dir, args.model_size, args.device)
        models.append(model)
        print("✓")

    # Create test dataloader
    # Base dir should be the project root (where images/ folder is)
    project_root = Path(args.test_jsonl).parent.parent
    test_dataset = TestDataset(
        args.test_jsonl,
        base_dir=str(project_root),
        resolution=args.resolution
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    print(f"\n📊 Test set: {len(test_dataset)} samples (NEVER SEEN DURING TRAINING)")

    # Evaluate
    print(f"\n🔍 Evaluating ensemble...")
    save_dir = Path(args.output_dir) / 'visuals' if args.save_visuals else None
    results = evaluate_ensemble(models, test_loader, args.device, save_dir)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("✅ TEST SET EVALUATION COMPLETE")
    print("="*80)
    print(f"\n📊 Results (Ensemble of {args.n_folds} folds):")
    print(f"   Average PSNR: {results['avg_psnr']:.2f} dB")
    print(f"   Average SSIM: {results['avg_ssim']:.4f}")
    print(f"   Average L1:   {results['avg_l1']:.4f}")
    print(f"\n📁 Results saved to: {output_dir}/test_results.json")
    if args.save_visuals:
        print(f"📸 Visual comparisons: {output_dir}/visuals/")
    print("="*80)


if __name__ == '__main__':
    main()
