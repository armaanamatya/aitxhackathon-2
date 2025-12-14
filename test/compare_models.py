#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script

Compares:
1. Restormer384 (Backbone Only)
2. Restormer384 + Elite Refiner (CNN)
3. Before Post-Processing
4. After Post-Processing

Runs on CPU, evaluates on test set, generates detailed metrics and visualizations.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.restormer import create_restormer
from src.models.color_refiner import create_elite_color_refiner


def pil_to_tensor(img):
    """Convert PIL Image to tensor [C, H, W] in range [0, 1]"""
    np_img = np.array(img).astype(np.float32) / 255.0
    if len(np_img.shape) == 2:  # Grayscale
        np_img = np_img[..., np.newaxis]
    return torch.from_numpy(np_img.transpose(2, 0, 1))  # HWC -> CHW


def tensor_to_pil(tensor):
    """Convert tensor [C, H, W] to PIL Image"""
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pad_to_multiple(tensor, multiple=16):
    """Pad tensor to make dimensions divisible by multiple"""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        # Pad: (left, right, top, bottom)
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    return tensor, (h, w)


def unpad(tensor, original_size):
    """Remove padding to restore original size"""
    h, w = original_size
    return tensor[:, :, :h, :w]


class MetricsCalculator:
    """Compute image quality metrics"""

    @staticmethod
    def psnr(pred, target, max_val=1.0):
        """Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))

    @staticmethod
    def ssim(pred, target, window_size=11):
        """Structural Similarity Index (simplified)"""
        # Use simple correlation as approximation
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

    @staticmethod
    def charbonnier_loss(pred, target, eps=1e-3):
        """Charbonnier loss"""
        diff = pred - target
        loss = torch.sqrt(diff * diff + eps * eps)
        return loss.mean().item()

    @staticmethod
    def rgb_to_hsv(rgb):
        """Convert RGB to HSV (simplified)"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        max_val, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_val = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = max_val - min_val

        # Saturation
        sat = torch.where(max_val > 1e-7, delta / (max_val + 1e-7), torch.zeros_like(delta))

        # Value
        val = max_val

        return sat, val

    @staticmethod
    def hsv_color_loss(pred, target):
        """HSV color loss (saturation + value)"""
        pred_sat, pred_val = MetricsCalculator.rgb_to_hsv(pred)
        target_sat, target_val = MetricsCalculator.rgb_to_hsv(target)

        sat_loss = F.l1_loss(pred_sat, target_sat)
        val_loss = F.l1_loss(pred_val, target_val)

        return (3.0 * sat_loss + val_loss).item()

    @staticmethod
    def saturation_mean(rgb):
        """Average saturation of image"""
        sat, _ = MetricsCalculator.rgb_to_hsv(rgb)
        return sat.mean().item()


class PostProcessor:
    """Apply post-processing to images"""

    @staticmethod
    def saturation_boost(img_tensor, boost=1.2):
        """Boost saturation in HSV space"""
        # Convert to numpy for easier HSV manipulation
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC

        # Simple saturation boost (approximate)
        mean = img_np.mean(axis=2, keepdims=True)
        saturation = img_np - mean
        boosted = mean + saturation * boost
        boosted = np.clip(boosted, 0, 1)

        return torch.from_numpy(boosted.transpose(2, 0, 1)).float()  # HWC -> CHW

    @staticmethod
    def histogram_equalization(img_tensor):
        """Histogram equalization (simplified)"""
        # Per-channel histogram equalization
        img_np = img_tensor.cpu().numpy()

        result = np.zeros_like(img_np)
        for c in range(3):
            channel = img_np[c]
            # Simple contrast stretching
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                result[c] = (channel - min_val) / (max_val - min_val)
            else:
                result[c] = channel

        return torch.from_numpy(result).float()

    @staticmethod
    def combined(img_tensor, sat_boost=1.15, hist_eq=False):
        """Combined post-processing"""
        result = img_tensor

        if sat_boost != 1.0:
            result = PostProcessor.saturation_boost(result, sat_boost)

        if hist_eq:
            result = PostProcessor.histogram_equalization(result)

        return result


def load_model(model_type, checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    print(f"Loading {model_type} from {checkpoint_path}...")

    if not os.path.exists(checkpoint_path):
        print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_type == 'restormer':
        model = create_restormer('base').to(device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

    elif model_type == 'refiner':
        model = create_elite_color_refiner('medium').to(device)
        # Use EMA weights if available
        if 'ema_state_dict' in checkpoint:
            state_dict = checkpoint['ema_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    print(f"  ✓ Loaded successfully")
    return model


def load_test_data(jsonl_path):
    """Load test dataset"""
    print(f"\nLoading test data from {jsonl_path}...")

    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"  ✓ Loaded {len(samples)} samples")
    return samples


def run_inference(samples, backbone, refiner, device='cpu', output_dir='test/comparison'):
    """Run inference on all samples with all model variants"""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    results = defaultdict(list)
    metrics_calc = MetricsCalculator()

    print(f"\nRunning inference on {len(samples)} samples...")

    for idx, sample in enumerate(tqdm(samples, desc="Processing")):
        input_path = sample['src']
        target_path = sample['tar']

        # Load images
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # To tensor
        input_tensor = pil_to_tensor(input_img).unsqueeze(0).to(device)
        target_tensor = pil_to_tensor(target_img).unsqueeze(0).to(device)

        # Pad to multiple of 16 (required for Restormer)
        input_padded, original_size = pad_to_multiple(input_tensor, multiple=16)
        target_padded, _ = pad_to_multiple(target_tensor, multiple=16)

        # === Model 1: Restormer Backbone Only ===
        with torch.no_grad():
            if backbone is not None:
                backbone_output_padded = backbone(input_padded)
                backbone_output = unpad(backbone_output_padded, original_size)
            else:
                backbone_output = input_tensor.clone()

        # === Model 2: Restormer + Elite Refiner ===
        with torch.no_grad():
            if refiner is not None and backbone is not None:
                refiner_output_padded = refiner(input_padded, backbone_output_padded)
                refiner_output = unpad(refiner_output_padded, original_size)
            else:
                refiner_output = backbone_output.clone()

        # === Post-Processing Variants ===
        backbone_postproc = PostProcessor.combined(backbone_output[0], sat_boost=1.15)
        refiner_postproc = PostProcessor.combined(refiner_output[0], sat_boost=1.05)  # Less boost since refiner already enhances

        # === Compute Metrics ===
        variants = {
            'backbone_raw': backbone_output[0],
            'backbone_postproc': backbone_postproc,
            'refiner_raw': refiner_output[0],
            'refiner_postproc': refiner_postproc,
        }

        for variant_name, output in variants.items():
            metrics = {
                'sample_id': idx,
                'filename': os.path.basename(input_path),
                'psnr': metrics_calc.psnr(output.unsqueeze(0), target_tensor).item(),
                'ssim': metrics_calc.ssim(output.unsqueeze(0), target_tensor),
                'charbonnier': metrics_calc.charbonnier_loss(output.unsqueeze(0), target_tensor),
                'hsv_loss': metrics_calc.hsv_color_loss(output.unsqueeze(0), target_tensor),
                'saturation_mean': metrics_calc.saturation_mean(output.unsqueeze(0)),
                'saturation_target': metrics_calc.saturation_mean(target_tensor),
            }
            results[variant_name].append(metrics)

        # Save sample images (first 10 samples)
        if idx < 10:
            save_comparison_image(
                input_tensor[0],
                target_tensor[0],
                variants,
                f"{output_dir}/images/sample_{idx:03d}.png"
            )

    return results


def save_comparison_image(input_img, target_img, variants, save_path):
    """Save side-by-side comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images = {
        'Input': input_img,
        'Target': target_img,
        'Backbone': variants['backbone_raw'],
        'Backbone+Post': variants['backbone_postproc'],
        'Refiner': variants['refiner_raw'],
        'Refiner+Post': variants['refiner_postproc'],
    }

    for ax, (title, img) in zip(axes.flat, images.items()):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_results(results, output_dir):
    """Analyze and visualize results"""

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Compute averages
    summary = {}
    for variant_name, metrics_list in results.items():
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key not in ['sample_id', 'filename']:
                values = [m[key] for m in metrics_list]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        summary[variant_name] = avg_metrics

    # Print table
    print(f"\n{'Model Variant':<25} {'PSNR↑':<10} {'SSIM↑':<10} {'Charb↓':<10} {'HSV↓':<10} {'Sat':<10}")
    print("-" * 80)

    for variant_name, metrics in summary.items():
        print(f"{variant_name:<25} "
              f"{metrics['psnr']:>8.2f}  "
              f"{metrics['ssim']:>8.4f}  "
              f"{metrics['charbonnier']:>8.4f}  "
              f"{metrics['hsv_loss']:>8.4f}  "
              f"{metrics['saturation_mean']:>8.4f}")

    # Save detailed results
    with open(f"{output_dir}/results_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")

        for variant_name, metrics in summary.items():
            f.write(f"\n{variant_name.upper()}\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                if not key.endswith('_std'):
                    std_value = metrics.get(f'{key}_std', 0)
                    f.write(f"  {key:<20}: {value:>10.4f} ± {std_value:.4f}\n")

    # Generate plots
    generate_plots(results, summary, output_dir)

    # Save raw data
    with open(f"{output_dir}/results_raw.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  - results_summary.txt")
    print(f"  - results_raw.json")
    print(f"  - images/ (sample comparisons)")
    print(f"  - plots/ (metric visualizations)")


def generate_plots(results, summary, output_dir):
    """Generate comparison plots"""

    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # 1. Bar chart of average metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    variants = list(summary.keys())
    metrics_to_plot = ['psnr', 'ssim', 'charbonnier', 'saturation_mean']
    metric_labels = ['PSNR (dB) ↑', 'SSIM ↑', 'Charbonnier Loss ↓', 'Saturation']

    for ax, metric, label in zip(axes.flat, metrics_to_plot, metric_labels):
        values = [summary[v][metric] for v in variants]
        colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12']

        bars = ax.bar(range(len(variants)), values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([v.replace('_', '\n') for v in variants], rotation=0, fontsize=9)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'Average {label}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/metrics_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Box plots for distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, metric, label in zip(axes.flat, metrics_to_plot, metric_labels):
        data = []
        labels = []
        for variant in variants:
            values = [m[metric] for m in results[variant]]
            data.append(values)
            labels.append(variant.replace('_', '\n'))

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/metrics_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Saturation comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    variant_labels = ['Backbone\nRaw', 'Backbone\n+Post', 'Refiner\nRaw', 'Refiner\n+Post']
    sat_pred = [summary[v]['saturation_mean'] for v in variants]
    sat_target = summary[variants[0]]['saturation_target']  # Same for all

    x = np.arange(len(variant_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, sat_pred, width, label='Predicted', color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, [sat_target]*len(x), width, label='Target', color='#2ecc71', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Average Saturation', fontsize=12, fontweight='bold')
    ax.set_title('Saturation Comparison: Predicted vs Target', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/saturation_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Plots saved to {output_dir}/plots/")


def main():
    """Main comparison pipeline"""

    print("="*80)
    print("MODEL COMPARISON - Restormer vs Restormer+Refiner")
    print("="*80)

    # Configuration
    device = torch.device('cpu')
    test_jsonl = 'data_splits/fold_1/val.jsonl'  # Using val as test
    backbone_checkpoint = 'outputs_restormer_384/checkpoint_best.pt'
    refiner_checkpoint = 'outputs_elite_refiner_384/checkpoint_best.pt'
    output_dir = 'test/comparison_final'

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Test data: {test_jsonl}")
    print(f"  Backbone checkpoint: {backbone_checkpoint}")
    print(f"  Refiner checkpoint: {refiner_checkpoint}")
    print(f"  Output directory: {output_dir}")

    # Load models
    print(f"\n{'='*80}")
    print("LOADING MODELS")
    print("="*80)

    backbone = load_model('restormer', backbone_checkpoint, device)
    refiner = load_model('refiner', refiner_checkpoint, device)

    if backbone is None:
        print("\n⚠️  Warning: Backbone not loaded, using identity transform")
    if refiner is None:
        print("\n⚠️  Warning: Refiner not loaded, will skip refiner evaluation")

    # Load test data
    samples = load_test_data(test_jsonl)

    # Run inference
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print("="*80)

    results = run_inference(samples, backbone, refiner, device, output_dir)

    # Analyze results
    print(f"\n{'='*80}")
    print("ANALYZING RESULTS")
    print("="*80)

    analyze_results(results, output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"\nTo view results:")
    print(f"  - Summary: cat {output_dir}/results_summary.txt")
    print(f"  - Images: ls {output_dir}/images/")
    print(f"  - Plots: ls {output_dir}/plots/")


if __name__ == '__main__':
    main()
