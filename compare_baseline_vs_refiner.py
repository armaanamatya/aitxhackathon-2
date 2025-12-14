#!/usr/bin/env python3
"""
A/B Comparison: Backbone Only vs Backbone + Elite Refiner

Compares:
1. outputs_full_baseline (Restormer 512, pure L1, no preprocessing)
2. outputs_elite_refiner_512 (Same backbone + Elite Refiner)

Fair comparison - same backbone, only difference is the refiner module.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import create_restormer
from models.color_refiner import create_elite_color_refiner


def pil_to_tensor(img):
    """Convert PIL Image to tensor [C, H, W] in range [0, 1]"""
    np_img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img.transpose(2, 0, 1))


def tensor_to_pil(tensor):
    """Convert tensor [C, H, W] to PIL Image"""
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pad_to_multiple(tensor, multiple=16):
    """Pad tensor to make dimensions divisible by multiple"""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, (h, w)


def unpad(tensor, original_size):
    """Remove padding to restore original size"""
    h, w = original_size
    return tensor[:, :, :h, :w]


class MetricsCalculator:
    """Compute image quality metrics"""

    @staticmethod
    def l1_loss(pred, target):
        """L1 / MAE loss - PRIMARY METRIC"""
        return F.l1_loss(pred, target).item()

    @staticmethod
    def psnr(pred, target, max_val=1.0):
        """Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))).item()

    @staticmethod
    def ssim(pred, target):
        """Structural Similarity Index (simplified)"""
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_std = pred.std()
        target_std = target.std()
        covariance = ((pred - pred_mean) * (target - target_mean)).mean()
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * pred_mean * target_mean + c1) * (2 * covariance + c2)) / \
               ((pred_mean**2 + target_mean**2 + c1) * (pred_std**2 + target_std**2 + c2))
        return ssim.item()


def load_backbone(checkpoint_path, device):
    """Load Restormer backbone"""
    print(f"Loading backbone from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    model = create_restormer('base')

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Fix key prefixes if needed
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"  ✓ Backbone loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model


def load_refiner(backbone_path, refiner_path, device):
    """Load backbone + refiner"""
    print(f"Loading backbone from {backbone_path}...")
    print(f"Loading refiner from {refiner_path}...")

    # Load backbone
    backbone = load_backbone(backbone_path, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Load refiner
    refiner_checkpoint = torch.load(refiner_path, map_location='cpu')

    # Create refiner
    refiner = create_elite_color_refiner(size='medium')

    # Load refiner weights
    if 'refiner_state_dict' in refiner_checkpoint:
        refiner.load_state_dict(refiner_checkpoint['refiner_state_dict'])
    elif 'model_state_dict' in refiner_checkpoint:
        refiner.load_state_dict(refiner_checkpoint['model_state_dict'])
    else:
        refiner.load_state_dict(refiner_checkpoint)

    refiner = refiner.to(device)
    refiner.eval()

    print(f"  ✓ Refiner loaded: {sum(p.numel() for p in refiner.parameters())/1e6:.2f}M params")

    return backbone, refiner


def run_inference(model, input_tensor, device, refiner=None):
    """Run inference with optional refiner"""
    with torch.no_grad():
        # Pad input
        input_padded, orig_size = pad_to_multiple(input_tensor, 16)
        input_padded = input_padded.to(device)

        # Backbone forward
        output = model(input_padded)

        # Refiner forward (if provided)
        if refiner is not None:
            output = refiner(output, input_padded)

        # Unpad
        output = unpad(output, orig_size)
        output = torch.clamp(output, 0, 1)

    return output.cpu()


def create_comparison_figure(input_img, target_img, backbone_output, refiner_output,
                             backbone_metrics, refiner_metrics, save_path, sample_name):
    """Create side-by-side comparison figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Input
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input (Source)', fontsize=14)
    axes[0, 0].axis('off')

    # Target
    axes[0, 1].imshow(target_img)
    axes[0, 1].set_title('Target (Ground Truth)', fontsize=14)
    axes[0, 1].axis('off')

    # Backbone only
    axes[1, 0].imshow(backbone_output)
    axes[1, 0].set_title(f'Backbone Only\nL1={backbone_metrics["l1"]:.4f} | PSNR={backbone_metrics["psnr"]:.2f}dB', fontsize=14)
    axes[1, 0].axis('off')

    # Backbone + Refiner
    axes[1, 1].imshow(refiner_output)
    axes[1, 1].set_title(f'Backbone + Refiner\nL1={refiner_metrics["l1"]:.4f} | PSNR={refiner_metrics["psnr"]:.2f}dB', fontsize=14)
    axes[1, 1].axis('off')

    # Compute delta
    l1_delta = refiner_metrics["l1"] - backbone_metrics["l1"]
    psnr_delta = refiner_metrics["psnr"] - backbone_metrics["psnr"]

    delta_color = 'green' if l1_delta < 0 else 'red'
    delta_text = f'Refiner Impact: L1 {"↓" if l1_delta < 0 else "↑"}{abs(l1_delta):.4f} | PSNR {"↑" if psnr_delta > 0 else "↓"}{abs(psnr_delta):.2f}dB'

    fig.suptitle(f'{sample_name}\n{delta_text}', fontsize=16, color=delta_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Backbone vs Backbone+Refiner')
    parser.add_argument('--backbone_path', type=str,
                        default='outputs_full_baseline/checkpoint_best.pt',
                        help='Path to backbone checkpoint')
    parser.add_argument('--refiner_path', type=str,
                        default='outputs_elite_refiner_512/checkpoint_best.pt',
                        help='Path to refiner checkpoint')
    parser.add_argument('--data_jsonl', type=str,
                        default='data_splits/fold_1/val.jsonl',
                        help='Path to validation data')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Output directory')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Inference resolution')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("A/B COMPARISON: BACKBONE vs BACKBONE + REFINER")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Backbone: {args.backbone_path}")
    print(f"Refiner: {args.refiner_path}")
    print(f"Resolution: {args.resolution}")
    print(f"Samples: {args.num_samples}")
    print()

    # Check if refiner exists
    if not os.path.exists(args.refiner_path):
        print(f"⚠️  Refiner checkpoint not found: {args.refiner_path}")
        print("   Elite Refiner 512 training may still be pending.")
        print("   Run this script again once training completes.")
        return

    # Load models
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    backbone = load_backbone(args.backbone_path, args.device)
    backbone_with_refiner, refiner = load_refiner(args.backbone_path, args.refiner_path, args.device)

    # Load test data
    print()
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    samples = []
    with open(args.data_jsonl, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Limit samples
    samples = samples[:args.num_samples]
    print(f"Testing on {len(samples)} samples")

    # Run inference
    print()
    print("=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)

    metrics_calc = MetricsCalculator()
    all_backbone_metrics = []
    all_refiner_metrics = []

    for i, sample in enumerate(samples):
        # Get paths
        input_path = sample.get('input', sample.get('src'))
        target_path = sample.get('target', sample.get('tar'))
        sample_name = os.path.basename(input_path).replace('_src.jpg', '')

        print(f"\n[{i+1}/{len(samples)}] Processing {sample_name}...")

        # Load images
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize to inference resolution
        input_img_resized = input_img.resize((args.resolution, args.resolution), Image.BILINEAR)
        target_img_resized = target_img.resize((args.resolution, args.resolution), Image.BILINEAR)

        # Convert to tensors
        input_tensor = pil_to_tensor(input_img_resized).unsqueeze(0)
        target_tensor = pil_to_tensor(target_img_resized).unsqueeze(0)

        # Backbone only inference
        backbone_output = run_inference(backbone, input_tensor, args.device)

        # Backbone + Refiner inference
        refiner_output = run_inference(backbone_with_refiner, input_tensor, args.device, refiner)

        # Compute metrics
        backbone_metrics = {
            'l1': metrics_calc.l1_loss(backbone_output, target_tensor),
            'psnr': metrics_calc.psnr(backbone_output, target_tensor),
            'ssim': metrics_calc.ssim(backbone_output, target_tensor)
        }

        refiner_metrics = {
            'l1': metrics_calc.l1_loss(refiner_output, target_tensor),
            'psnr': metrics_calc.psnr(refiner_output, target_tensor),
            'ssim': metrics_calc.ssim(refiner_output, target_tensor)
        }

        all_backbone_metrics.append(backbone_metrics)
        all_refiner_metrics.append(refiner_metrics)

        # Print sample metrics
        l1_delta = refiner_metrics['l1'] - backbone_metrics['l1']
        psnr_delta = refiner_metrics['psnr'] - backbone_metrics['psnr']

        print(f"  Backbone:  L1={backbone_metrics['l1']:.4f} | PSNR={backbone_metrics['psnr']:.2f}dB | SSIM={backbone_metrics['ssim']:.4f}")
        print(f"  +Refiner:  L1={refiner_metrics['l1']:.4f} | PSNR={refiner_metrics['psnr']:.2f}dB | SSIM={refiner_metrics['ssim']:.4f}")
        print(f"  Delta:     L1={'↓' if l1_delta < 0 else '↑'}{abs(l1_delta):.4f} | PSNR={'↑' if psnr_delta > 0 else '↓'}{abs(psnr_delta):.2f}dB")

        # Create comparison figure
        backbone_pil = tensor_to_pil(backbone_output.squeeze(0))
        refiner_pil = tensor_to_pil(refiner_output.squeeze(0))

        fig_path = output_dir / f'{sample_name}_comparison.png'
        create_comparison_figure(
            input_img_resized, target_img_resized,
            backbone_pil, refiner_pil,
            backbone_metrics, refiner_metrics,
            fig_path, sample_name
        )

    # Aggregate results
    print()
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    avg_backbone = {
        'l1': np.mean([m['l1'] for m in all_backbone_metrics]),
        'psnr': np.mean([m['psnr'] for m in all_backbone_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_backbone_metrics])
    }

    avg_refiner = {
        'l1': np.mean([m['l1'] for m in all_refiner_metrics]),
        'psnr': np.mean([m['psnr'] for m in all_refiner_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_refiner_metrics])
    }

    print(f"\nBackbone Only (n={len(samples)}):")
    print(f"  L1:   {avg_backbone['l1']:.4f}")
    print(f"  PSNR: {avg_backbone['psnr']:.2f} dB")
    print(f"  SSIM: {avg_backbone['ssim']:.4f}")

    print(f"\nBackbone + Refiner (n={len(samples)}):")
    print(f"  L1:   {avg_refiner['l1']:.4f}")
    print(f"  PSNR: {avg_refiner['psnr']:.2f} dB")
    print(f"  SSIM: {avg_refiner['ssim']:.4f}")

    l1_improvement = avg_backbone['l1'] - avg_refiner['l1']
    psnr_improvement = avg_refiner['psnr'] - avg_backbone['psnr']
    ssim_improvement = avg_refiner['ssim'] - avg_backbone['ssim']

    print(f"\n{'='*40}")
    print(f"REFINER IMPACT:")
    print(f"{'='*40}")
    print(f"  L1:   {'↓' if l1_improvement > 0 else '↑'}{abs(l1_improvement):.4f} ({'BETTER' if l1_improvement > 0 else 'WORSE'})")
    print(f"  PSNR: {'↑' if psnr_improvement > 0 else '↓'}{abs(psnr_improvement):.2f} dB ({'BETTER' if psnr_improvement > 0 else 'WORSE'})")
    print(f"  SSIM: {'↑' if ssim_improvement > 0 else '↓'}{abs(ssim_improvement):.4f} ({'BETTER' if ssim_improvement > 0 else 'WORSE'})")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'backbone_path': args.backbone_path,
            'refiner_path': args.refiner_path,
            'resolution': args.resolution,
            'num_samples': len(samples)
        },
        'backbone_only': {
            'mean_l1': avg_backbone['l1'],
            'mean_psnr': avg_backbone['psnr'],
            'mean_ssim': avg_backbone['ssim'],
            'per_sample': all_backbone_metrics
        },
        'backbone_plus_refiner': {
            'mean_l1': avg_refiner['l1'],
            'mean_psnr': avg_refiner['psnr'],
            'mean_ssim': avg_refiner['ssim'],
            'per_sample': all_refiner_metrics
        },
        'improvement': {
            'l1_reduction': l1_improvement,
            'psnr_gain': psnr_improvement,
            'ssim_gain': ssim_improvement,
            'refiner_helps': l1_improvement > 0
        }
    }

    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Comparison images saved to: {output_dir}/")

    # Final verdict
    print()
    print("=" * 80)
    if l1_improvement > 0:
        print(f"✅ VERDICT: Elite Refiner IMPROVES L1 by {l1_improvement:.4f}")
    else:
        print(f"❌ VERDICT: Elite Refiner HURTS L1 by {abs(l1_improvement):.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
