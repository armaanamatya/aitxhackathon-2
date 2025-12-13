#!/usr/bin/env python3
"""
Comprehensive Comparison: MambaDiffusion vs Restormer-Base
===========================================================

This script provides thorough analysis comparing two HDR models:
- Visual side-by-side comparisons (source, target, restormer output, mamba output)
- Multiple quality metrics (L1, PSNR, SSIM, LPIPS, Delta-E)
- Statistical analysis (mean, std, median, percentiles)
- Inference time benchmarking
- Memory usage analysis
- Per-image breakdown and aggregated results

Handles resolution differences between models by:
1. Running inference at each model's native resolution
2. Resizing outputs to target resolution for fair metric computation
3. Computing metrics at multiple scales
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available")

try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available, using PyTorch implementations")


# =============================================================================
# Metrics
# =============================================================================

class MetricsCalculator:
    """Comprehensive metrics calculation."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.lpips_fn = None
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()

    def compute_l1(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """L1 (Mean Absolute Error)."""
        return F.l1_loss(pred, target).item()

    def compute_mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Squared Error."""
        return F.mse_loss(pred, target).item()

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio (higher is better)."""
        mse = F.mse_loss(pred, target).item()
        if mse < 1e-10:
            return 100.0
        return 10 * np.log10(1.0 / mse)

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Structural Similarity Index (higher is better)."""
        if SKIMAGE_AVAILABLE:
            # Convert to numpy [H, W, C]
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return ssim_func(pred_np, target_np, channel_axis=2, data_range=1.0)
        else:
            # Simple SSIM approximation
            mu_pred = pred.mean()
            mu_target = target.mean()
            sigma_pred = pred.var()
            sigma_target = target.var()
            sigma_both = ((pred - mu_pred) * (target - mu_target)).mean()

            C1, C2 = 0.01**2, 0.03**2
            ssim = ((2*mu_pred*mu_target + C1) * (2*sigma_both + C2)) / \
                   ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))
            return ssim.item()

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """LPIPS perceptual similarity (lower is better)."""
        if self.lpips_fn is None:
            return 0.0
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            pred_scaled = pred * 2 - 1
            target_scaled = target * 2 - 1
            return self.lpips_fn(pred_scaled, target_scaled).item()

    def compute_delta_e(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Delta-E color difference in LAB space.
        Lower is better (0 = identical).
        """
        # Convert RGB to LAB (simplified)
        def rgb_to_lab(img):
            # sRGB to linear RGB
            img = torch.where(img <= 0.04045, img / 12.92,
                            ((img + 0.055) / 1.055) ** 2.4)

            # Linear RGB to XYZ
            M = torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ], device=img.device, dtype=img.dtype)

            # Reshape for matrix multiplication
            B, C, H, W = img.shape
            img_flat = img.view(B, C, -1)  # [B, 3, H*W]
            xyz = torch.matmul(M.unsqueeze(0), img_flat)  # [B, 3, H*W]

            # Normalize by D65 white point
            xyz[:, 0] /= 0.95047
            xyz[:, 1] /= 1.0
            xyz[:, 2] /= 1.08883

            # XYZ to LAB
            epsilon = 0.008856
            kappa = 903.3

            f = torch.where(xyz > epsilon, xyz ** (1/3),
                          (kappa * xyz + 16) / 116)

            L = 116 * f[:, 1:2] - 16
            a = 500 * (f[:, 0:1] - f[:, 1:2])
            b = 200 * (f[:, 1:2] - f[:, 2:3])

            return torch.cat([L, a, b], dim=1)

        lab_pred = rgb_to_lab(pred)
        lab_target = rgb_to_lab(target)

        # Delta-E (CIE76)
        delta_e = torch.sqrt(((lab_pred - lab_target) ** 2).sum(dim=1)).mean()
        return delta_e.item()

    def compute_all(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'L1': self.compute_l1(pred, target),
            'MSE': self.compute_mse(pred, target),
            'PSNR': self.compute_psnr(pred, target),
            'SSIM': self.compute_ssim(pred, target),
            'LPIPS': self.compute_lpips(pred, target),
            'DeltaE': self.compute_delta_e(pred, target),
        }


# =============================================================================
# Dataset
# =============================================================================

class ComparisonDataset(Dataset):
    """Dataset for model comparison."""

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        split: str = 'val',
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.samples = []

        # Load JSONL
        jsonl_file = self.data_root / jsonl_path
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('split', 'train') == split:
                    self.samples.append(item)

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load images at original resolution
        source_path = self.data_root / item['source']
        target_path = self.data_root / item['target']

        source = Image.open(source_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        # Convert to tensor [0, 1]
        source_tensor = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(np.array(target)).permute(2, 0, 1).float() / 255.0

        return {
            'source': source_tensor,
            'target': target_tensor,
            'source_path': str(source_path),
            'target_path': str(target_path),
            'name': Path(item['source']).stem.replace('_src', ''),
        }


# =============================================================================
# Model Loading
# =============================================================================

def load_restormer(checkpoint_path: str, device: str = 'cuda'):
    """Load Restormer model."""
    from restormer import restormer_base

    model = restormer_base()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def load_mamba(checkpoint_path: str, device: str = 'cuda'):
    """Load MambaDiffusion model."""
    from mamba_diffusion import mamba_diffusion_large

    model = mamba_diffusion_large()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model,
    source: torch.Tensor,
    target_size: Tuple[int, int],
    model_input_size: int,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, float, float]:
    """
    Run inference with a model.

    Args:
        model: The model to use
        source: Source image tensor [1, C, H, W]
        target_size: Size to resize output to (H, W)
        model_input_size: Model's native input size
        device: Device to use

    Returns:
        output: Model output resized to target_size
        inference_time: Time for inference in seconds
        memory_mb: Peak GPU memory used in MB
    """
    # Resize input to model's native size
    source_resized = F.interpolate(
        source,
        size=(model_input_size, model_input_size),
        mode='bilinear',
        align_corners=False
    ).to(device)

    # Clear GPU cache and record memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warm-up
    with torch.no_grad():
        _ = model(source_resized)

    torch.cuda.synchronize()

    # Timed inference
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(source_resized)
        if isinstance(output, dict):
            output = output.get('output', output.get('pred', list(output.values())[0]))
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    # Memory usage
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Resize output to target size
    output = F.interpolate(
        output.clamp(0, 1),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )

    return output.cpu(), inference_time, memory_mb


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_image(
    source: np.ndarray,
    target: np.ndarray,
    restormer_out: np.ndarray,
    mamba_out: np.ndarray,
    metrics_restormer: Dict[str, float],
    metrics_mamba: Dict[str, float],
    name: str,
) -> Image.Image:
    """
    Create a side-by-side comparison image with metrics.

    Layout:
    +------------------+------------------+
    |     Source       |      Target      |
    +------------------+------------------+
    | Restormer Output | Mamba Output     |
    | (metrics)        | (metrics)        |
    +------------------+------------------+
    """
    from PIL import ImageDraw, ImageFont

    H, W = source.shape[:2]
    padding = 10
    text_height = 80

    # Create canvas
    canvas_w = W * 2 + padding * 3
    canvas_h = H * 2 + padding * 3 + text_height * 2
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))

    # Convert arrays to PIL
    source_pil = Image.fromarray((source * 255).astype(np.uint8))
    target_pil = Image.fromarray((target * 255).astype(np.uint8))
    restormer_pil = Image.fromarray((restormer_out * 255).astype(np.uint8))
    mamba_pil = Image.fromarray((mamba_out * 255).astype(np.uint8))

    # Paste images
    canvas.paste(source_pil, (padding, padding))
    canvas.paste(target_pil, (W + padding * 2, padding))
    canvas.paste(restormer_pil, (padding, H + padding * 2 + text_height))
    canvas.paste(mamba_pil, (W + padding * 2, H + padding * 2 + text_height))

    # Add labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Labels for top row
    draw.text((padding + W//2 - 30, padding - 20 if padding > 20 else 5),
              "Source", fill=(0, 0, 0), font=font)
    draw.text((W + padding * 2 + W//2 - 30, padding - 20 if padding > 20 else 5),
              "Target (Ground Truth)", fill=(0, 0, 0), font=font)

    # Labels and metrics for bottom row
    y_text = H + padding + 5

    # Restormer metrics
    draw.text((padding, y_text), "Restormer-Base", fill=(0, 0, 200), font=font)
    metrics_text = f"PSNR: {metrics_restormer['PSNR']:.2f} | SSIM: {metrics_restormer['SSIM']:.4f}"
    draw.text((padding, y_text + 20), metrics_text, fill=(0, 0, 0), font=font_small)
    metrics_text2 = f"LPIPS: {metrics_restormer['LPIPS']:.4f} | L1: {metrics_restormer['L1']:.4f}"
    draw.text((padding, y_text + 35), metrics_text2, fill=(0, 0, 0), font=font_small)
    metrics_text3 = f"ΔE: {metrics_restormer['DeltaE']:.2f}"
    draw.text((padding, y_text + 50), metrics_text3, fill=(0, 0, 0), font=font_small)

    # Mamba metrics
    draw.text((W + padding * 2, y_text), "MambaDiffusion", fill=(200, 0, 0), font=font)
    metrics_text = f"PSNR: {metrics_mamba['PSNR']:.2f} | SSIM: {metrics_mamba['SSIM']:.4f}"
    draw.text((W + padding * 2, y_text + 20), metrics_text, fill=(0, 0, 0), font=font_small)
    metrics_text2 = f"LPIPS: {metrics_mamba['LPIPS']:.4f} | L1: {metrics_mamba['L1']:.4f}"
    draw.text((W + padding * 2, y_text + 35), metrics_text2, fill=(0, 0, 0), font=font_small)
    metrics_text3 = f"ΔE: {metrics_mamba['DeltaE']:.2f}"
    draw.text((W + padding * 2, y_text + 50), metrics_text3, fill=(0, 0, 0), font=font_small)

    # Highlight winner
    better_psnr = "Mamba" if metrics_mamba['PSNR'] > metrics_restormer['PSNR'] else "Restormer"
    better_ssim = "Mamba" if metrics_mamba['SSIM'] > metrics_restormer['SSIM'] else "Restormer"
    better_lpips = "Mamba" if metrics_mamba['LPIPS'] < metrics_restormer['LPIPS'] else "Restormer"

    return canvas


def create_difference_map(
    target: np.ndarray,
    pred: np.ndarray,
    amplify: float = 5.0,
) -> np.ndarray:
    """Create an amplified difference map."""
    diff = np.abs(target - pred)
    diff_amplified = np.clip(diff * amplify, 0, 1)
    return diff_amplified


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(
    restormer_ckpt: str,
    mamba_ckpt: str,
    data_root: str,
    jsonl_path: str,
    output_dir: str,
    restormer_size: int = 256,
    mamba_size: int = 128,
    comparison_size: int = 256,
    max_samples: Optional[int] = None,
    device: str = 'cuda',
):
    """Run comprehensive comparison between Restormer and MambaDiffusion."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_path / 'comparisons').mkdir(exist_ok=True)
    (output_path / 'difference_maps').mkdir(exist_ok=True)
    (output_path / 'individual').mkdir(exist_ok=True)

    # Initialize
    print("=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("Restormer-Base vs MambaDiffusion")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Restormer input size: {restormer_size}x{restormer_size}")
    print(f"Mamba input size: {mamba_size}x{mamba_size}")
    print(f"Comparison size: {comparison_size}x{comparison_size}")
    print("-" * 70)

    # Load models
    print("\nLoading models...")
    restormer = load_restormer(restormer_ckpt, device)
    print(f"  Restormer loaded from: {restormer_ckpt}")

    mamba = load_mamba(mamba_ckpt, device)
    print(f"  Mamba loaded from: {mamba_ckpt}")

    # Count parameters
    restormer_params = sum(p.numel() for p in restormer.parameters())
    mamba_params = sum(p.numel() for p in mamba.parameters())
    print(f"\n  Restormer parameters: {restormer_params:,}")
    print(f"  Mamba parameters: {mamba_params:,}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ComparisonDataset(data_root, jsonl_path, split='val', max_samples=max_samples)

    # Initialize metrics
    metrics_calc = MetricsCalculator(device)

    all_results = []
    restormer_times = []
    mamba_times = []
    restormer_memory = []
    mamba_memory = []

    # Run comparison
    print(f"\nRunning comparison on {len(dataset)} images...")
    print("-" * 70)

    for idx in tqdm(range(len(dataset)), desc="Comparing"):
        sample = dataset[idx]
        source = sample['source'].unsqueeze(0)  # [1, C, H, W]
        target = sample['target'].unsqueeze(0)
        name = sample['name']

        orig_h, orig_w = source.shape[2], source.shape[3]

        # Resize target to comparison size for metrics
        target_resized = F.interpolate(
            target,
            size=(comparison_size, comparison_size),
            mode='bilinear',
            align_corners=False
        )

        # Run Restormer
        restormer_out, rest_time, rest_mem = run_inference(
            restormer, source, (comparison_size, comparison_size),
            restormer_size, device
        )
        restormer_times.append(rest_time)
        restormer_memory.append(rest_mem)

        # Run Mamba
        mamba_out, mamba_time, mamba_mem = run_inference(
            mamba, source, (comparison_size, comparison_size),
            mamba_size, device
        )
        mamba_times.append(mamba_time)
        mamba_memory.append(mamba_mem)

        # Compute metrics
        restormer_out_device = restormer_out.to(device)
        mamba_out_device = mamba_out.to(device)
        target_device = target_resized.to(device)

        metrics_rest = metrics_calc.compute_all(restormer_out_device, target_device)
        metrics_mamba = metrics_calc.compute_all(mamba_out_device, target_device)

        # Store results
        result = {
            'name': name,
            'restormer': metrics_rest,
            'mamba': metrics_mamba,
            'restormer_time': rest_time,
            'mamba_time': mamba_time,
        }
        all_results.append(result)

        # Create visualizations
        source_np = F.interpolate(source, size=(comparison_size, comparison_size),
                                  mode='bilinear', align_corners=False)
        source_np = source_np.squeeze(0).permute(1, 2, 0).numpy()
        target_np = target_resized.squeeze(0).permute(1, 2, 0).numpy()
        restormer_np = restormer_out.squeeze(0).permute(1, 2, 0).numpy()
        mamba_np = mamba_out.squeeze(0).permute(1, 2, 0).numpy()

        # Side-by-side comparison
        comparison_img = create_comparison_image(
            source_np, target_np, restormer_np, mamba_np,
            metrics_rest, metrics_mamba, name
        )
        comparison_img.save(output_path / 'comparisons' / f'{name}_comparison.png')

        # Difference maps
        diff_restormer = create_difference_map(target_np, restormer_np)
        diff_mamba = create_difference_map(target_np, mamba_np)

        diff_combined = np.concatenate([diff_restormer, diff_mamba], axis=1)
        diff_img = Image.fromarray((diff_combined * 255).astype(np.uint8))
        diff_img.save(output_path / 'difference_maps' / f'{name}_diff.png')

        # Individual outputs
        Image.fromarray((restormer_np * 255).astype(np.uint8)).save(
            output_path / 'individual' / f'{name}_restormer.png'
        )
        Image.fromarray((mamba_np * 255).astype(np.uint8)).save(
            output_path / 'individual' / f'{name}_mamba.png'
        )

    # ==========================================================================
    # Statistical Analysis
    # ==========================================================================

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Aggregate metrics
    metrics_names = ['L1', 'MSE', 'PSNR', 'SSIM', 'LPIPS', 'DeltaE']

    restormer_metrics = {m: [] for m in metrics_names}
    mamba_metrics = {m: [] for m in metrics_names}

    for r in all_results:
        for m in metrics_names:
            restormer_metrics[m].append(r['restormer'][m])
            mamba_metrics[m].append(r['mamba'][m])

    # Compute statistics
    stats = {}
    for m in metrics_names:
        rest_arr = np.array(restormer_metrics[m])
        mamba_arr = np.array(mamba_metrics[m])

        stats[m] = {
            'restormer': {
                'mean': float(np.mean(rest_arr)),
                'std': float(np.std(rest_arr)),
                'median': float(np.median(rest_arr)),
                'min': float(np.min(rest_arr)),
                'max': float(np.max(rest_arr)),
                'p25': float(np.percentile(rest_arr, 25)),
                'p75': float(np.percentile(rest_arr, 75)),
            },
            'mamba': {
                'mean': float(np.mean(mamba_arr)),
                'std': float(np.std(mamba_arr)),
                'median': float(np.median(mamba_arr)),
                'min': float(np.min(mamba_arr)),
                'max': float(np.max(mamba_arr)),
                'p25': float(np.percentile(mamba_arr, 25)),
                'p75': float(np.percentile(mamba_arr, 75)),
            }
        }

    # Print comparison table
    print("\n" + "-" * 70)
    print("METRIC COMPARISON (Mean ± Std)")
    print("-" * 70)
    print(f"{'Metric':<12} {'Restormer':<25} {'Mamba':<25} {'Winner':<12}")
    print("-" * 70)

    winners = {'restormer': 0, 'mamba': 0}

    for m in metrics_names:
        rest_mean = stats[m]['restormer']['mean']
        rest_std = stats[m]['restormer']['std']
        mamba_mean = stats[m]['mamba']['mean']
        mamba_std = stats[m]['mamba']['std']

        # Determine winner (higher is better for PSNR, SSIM; lower for others)
        if m in ['PSNR', 'SSIM']:
            winner = 'Mamba' if mamba_mean > rest_mean else 'Restormer'
            winners['mamba' if mamba_mean > rest_mean else 'restormer'] += 1
        else:
            winner = 'Mamba' if mamba_mean < rest_mean else 'Restormer'
            winners['mamba' if mamba_mean < rest_mean else 'restormer'] += 1

        print(f"{m:<12} {rest_mean:>8.4f} ± {rest_std:<8.4f}   {mamba_mean:>8.4f} ± {mamba_std:<8.4f}   {winner:<12}")

    # Timing comparison
    print("\n" + "-" * 70)
    print("INFERENCE TIME (seconds)")
    print("-" * 70)
    rest_time_mean = np.mean(restormer_times)
    mamba_time_mean = np.mean(mamba_times)
    print(f"Restormer: {rest_time_mean:.4f}s ± {np.std(restormer_times):.4f}s")
    print(f"Mamba:     {mamba_time_mean:.4f}s ± {np.std(mamba_times):.4f}s")
    print(f"Speedup:   {rest_time_mean/mamba_time_mean:.2f}x {'(Mamba faster)' if mamba_time_mean < rest_time_mean else '(Restormer faster)'}")

    # Memory comparison
    print("\n" + "-" * 70)
    print("GPU MEMORY (MB)")
    print("-" * 70)
    print(f"Restormer: {np.mean(restormer_memory):.1f} MB (peak)")
    print(f"Mamba:     {np.mean(mamba_memory):.1f} MB (peak)")

    # Overall winner
    print("\n" + "=" * 70)
    print("OVERALL WINNER")
    print("=" * 70)
    print(f"Metrics won - Restormer: {winners['restormer']}, Mamba: {winners['mamba']}")
    overall = 'MambaDiffusion' if winners['mamba'] > winners['restormer'] else 'Restormer-Base'
    if winners['mamba'] == winners['restormer']:
        overall = 'TIE'
    print(f"Overall Winner: {overall}")

    # ==========================================================================
    # Save Results
    # ==========================================================================

    # Per-image results CSV
    csv_path = output_path / 'per_image_results.csv'
    with open(csv_path, 'w') as f:
        headers = ['name'] + [f'rest_{m}' for m in metrics_names] + [f'mamba_{m}' for m in metrics_names]
        headers += ['rest_time', 'mamba_time']
        f.write(','.join(headers) + '\n')

        for r in all_results:
            row = [r['name']]
            row += [str(r['restormer'][m]) for m in metrics_names]
            row += [str(r['mamba'][m]) for m in metrics_names]
            row += [str(r['restormer_time']), str(r['mamba_time'])]
            f.write(','.join(row) + '\n')

    # Summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'restormer_ckpt': restormer_ckpt,
            'mamba_ckpt': mamba_ckpt,
            'restormer_size': restormer_size,
            'mamba_size': mamba_size,
            'comparison_size': comparison_size,
            'num_samples': len(dataset),
        },
        'model_info': {
            'restormer_params': restormer_params,
            'mamba_params': mamba_params,
        },
        'metrics_stats': stats,
        'timing': {
            'restormer_mean': float(np.mean(restormer_times)),
            'restormer_std': float(np.std(restormer_times)),
            'mamba_mean': float(np.mean(mamba_times)),
            'mamba_std': float(np.std(mamba_times)),
        },
        'memory_mb': {
            'restormer_mean': float(np.mean(restormer_memory)),
            'mamba_mean': float(np.mean(mamba_memory)),
        },
        'winners': winners,
        'overall_winner': overall,
    }

    with open(output_path / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Detailed log
    log_path = output_path / 'comparison_log.txt'
    with open(log_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON LOG\n")
        f.write("Restormer-Base vs MambaDiffusion\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Restormer checkpoint: {restormer_ckpt}\n")
        f.write(f"Mamba checkpoint: {mamba_ckpt}\n")
        f.write(f"Samples evaluated: {len(dataset)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("MODEL PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Restormer: {restormer_params:,} parameters\n")
        f.write(f"Mamba: {mamba_params:,} parameters\n\n")

        f.write("-" * 70 + "\n")
        f.write("RESOLUTION CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Restormer input: {restormer_size}x{restormer_size}\n")
        f.write(f"Mamba input: {mamba_size}x{mamba_size}\n")
        f.write(f"Comparison (output): {comparison_size}x{comparison_size}\n\n")

        f.write("-" * 70 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("-" * 70 + "\n\n")

        for m in metrics_names:
            f.write(f"{m}:\n")
            f.write(f"  Restormer: {stats[m]['restormer']['mean']:.6f} ± {stats[m]['restormer']['std']:.6f}\n")
            f.write(f"    Range: [{stats[m]['restormer']['min']:.6f}, {stats[m]['restormer']['max']:.6f}]\n")
            f.write(f"    Percentiles: 25th={stats[m]['restormer']['p25']:.6f}, 50th={stats[m]['restormer']['median']:.6f}, 75th={stats[m]['restormer']['p75']:.6f}\n")
            f.write(f"  Mamba: {stats[m]['mamba']['mean']:.6f} ± {stats[m]['mamba']['std']:.6f}\n")
            f.write(f"    Range: [{stats[m]['mamba']['min']:.6f}, {stats[m]['mamba']['max']:.6f}]\n")
            f.write(f"    Percentiles: 25th={stats[m]['mamba']['p25']:.6f}, 50th={stats[m]['mamba']['median']:.6f}, 75th={stats[m]['mamba']['p75']:.6f}\n")
            f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("INFERENCE PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Restormer: {np.mean(restormer_times)*1000:.2f}ms ± {np.std(restormer_times)*1000:.2f}ms\n")
        f.write(f"Mamba: {np.mean(mamba_times)*1000:.2f}ms ± {np.std(mamba_times)*1000:.2f}ms\n")
        f.write(f"Restormer memory: {np.mean(restormer_memory):.1f} MB\n")
        f.write(f"Mamba memory: {np.mean(mamba_memory):.1f} MB\n\n")

        f.write("=" * 70 + "\n")
        f.write(f"OVERALL WINNER: {overall}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {output_path}")
    print(f"  - comparisons/: Side-by-side comparison images")
    print(f"  - difference_maps/: Error visualization maps")
    print(f"  - individual/: Individual model outputs")
    print(f"  - per_image_results.csv: Per-image metrics")
    print(f"  - comparison_summary.json: Statistical summary")
    print(f"  - comparison_log.txt: Detailed analysis log")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare Restormer vs MambaDiffusion')
    parser.add_argument('--restormer_ckpt', type=str, required=True,
                        help='Path to Restormer checkpoint')
    parser.add_argument('--mamba_ckpt', type=str, required=True,
                        help='Path to MambaDiffusion checkpoint')
    parser.add_argument('--data_root', type=str, default='.',
                        help='Data root directory')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl',
                        help='Path to JSONL file')
    parser.add_argument('--output_dir', type=str, default='compare_restormerbase_mamba',
                        help='Output directory')
    parser.add_argument('--restormer_size', type=int, default=256,
                        help='Restormer input size')
    parser.add_argument('--mamba_size', type=int, default=128,
                        help='Mamba input size')
    parser.add_argument('--comparison_size', type=int, default=256,
                        help='Size for metric comparison')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to evaluate')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_comparison(
        restormer_ckpt=args.restormer_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        restormer_size=args.restormer_size,
        mamba_size=args.mamba_size,
        comparison_size=args.comparison_size,
        max_samples=args.max_samples,
        device=device,
    )


if __name__ == '__main__':
    main()
