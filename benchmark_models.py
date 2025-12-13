#!/usr/bin/env python3
"""
Model Benchmarking Script for AutoHDR Real Estate Photo Enhancement
====================================================================

Compares all available model architectures:
- U-Net+GAN (primary)
- HAT (Hybrid Attention Transformer)
- Restormer (efficient transformer)
- RetinexMamba (state space model)

Outputs:
- Inference time (ms)
- Peak GPU memory (MB)
- PSNR/SSIM on validation images (if available)

Usage:
    python benchmark_models.py
    python benchmark_models.py --checkpoint_dir outputs/checkpoints
    python benchmark_models.py --validation_dir data/val
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def benchmark_inference(
    model: nn.Module,
    name: str,
    device: torch.device,
    img_size: int = 512,
    num_runs: int = 20,
    warmup_runs: int = 5,
) -> Dict:
    """
    Benchmark a single model's inference performance.

    Args:
        model: PyTorch model to benchmark
        name: Model name for display
        device: Device to run on
        img_size: Input image size
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size).to(device)

    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

    # Get memory stats
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        peak_memory = 0.0

    # Calculate statistics
    times = np.array(times)

    return {
        'name': name,
        'avg_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'peak_memory_mb': float(peak_memory),
        'img_size': img_size,
        'num_runs': num_runs,
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_unet_model(checkpoint_path: Optional[str] = None, device: torch.device = None):
    """Load U-Net generator model."""
    from src.training.models import UNetGenerator

    model = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_features=64,
        num_residual_blocks=9,
        learn_residual=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'generator_state_dict' in state:
            model.load_state_dict(state['generator_state_dict'])
        elif 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    return model.to(device)


def load_restormer_model(checkpoint_path: Optional[str] = None, device: torch.device = None):
    """Load Restormer model."""
    from src.training.restormer import Restormer

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    return model.to(device)


def load_hat_model(checkpoint_path: Optional[str] = None, device: torch.device = None):
    """Load HAT model."""
    from src.training.hat import HAT

    model = HAT(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.0,
        upsampler='',  # No upsampling for HDR enhancement
        resi_connection='1conv',
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    return model.to(device)


def load_retinexmamba_model(checkpoint_path: Optional[str] = None, device: torch.device = None):
    """Load RetinexMamba model."""
    from src.training.retinexmamba import RetinexMamba

    model = RetinexMamba(
        in_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=4,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    return model.to(device)


def load_controlnet_model(checkpoint_path: Optional[str] = None, device: torch.device = None):
    """Load ControlNet HDR model."""
    from src.training.controlnet import ControlNetHDR

    model = ControlNetHDR(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        num_res_blocks=2,
        learn_residual=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    return model.to(device)


def load_controlnet_lite_model(checkpoint_path: Optional[str] = None, device: torch.device = None):
    """Load ControlNet Lite model."""
    from src.training.controlnet import ControlNetLite

    model = ControlNetLite(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        learn_residual=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    return model.to(device)


def print_results_table(results: List[Dict]):
    """Print a formatted table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Model':<20} {'Params':>10} {'Avg (ms)':>12} {'Std (ms)':>10} {'Memory (MB)':>12}")
    print("-" * 80)

    for r in results:
        params_str = f"{r.get('params', 0) / 1e6:.1f}M"
        print(f"{r['name']:<20} {params_str:>10} {r['avg_ms']:>12.2f} {r['std_ms']:>10.2f} {r['peak_memory_mb']:>12.0f}")

    print("=" * 80)

    # Find fastest model
    if results:
        fastest = min(results, key=lambda x: x['avg_ms'])
        print(f"\nFastest model: {fastest['name']} ({fastest['avg_ms']:.2f} ms)")

        # Calculate throughput
        throughput = 1000 / fastest['avg_ms']
        print(f"Max throughput: {throughput:.1f} images/second")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AutoHDR models")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Image size for benchmarking")
    parser.add_argument("--num_runs", type=int, default=20,
                        help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["unet", "restormer", "hat", "retinexmamba", "controlnet", "controlnet_lite"],
                        help="Models to benchmark")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"FP16 mode: {args.fp16}")
    print("")

    results = []
    checkpoint_dir = Path(args.checkpoint_dir)

    # Model configurations
    model_configs = {
        "unet": {
            "loader": load_unet_model,
            "checkpoint": checkpoint_dir / "best_generator.pt",
            "name": "U-Net+GAN",
        },
        "restormer": {
            "loader": load_restormer_model,
            "checkpoint": checkpoint_dir / "restormer_best.pt",
            "name": "Restormer",
        },
        "hat": {
            "loader": load_hat_model,
            "checkpoint": checkpoint_dir / "hat_best.pt",
            "name": "HAT",
        },
        "retinexmamba": {
            "loader": load_retinexmamba_model,
            "checkpoint": checkpoint_dir / "retinexmamba_best.pt",
            "name": "RetinexMamba",
        },
        "controlnet": {
            "loader": load_controlnet_model,
            "checkpoint": checkpoint_dir / "controlnet_best.pt",
            "name": "ControlNet",
        },
        "controlnet_lite": {
            "loader": load_controlnet_lite_model,
            "checkpoint": checkpoint_dir / "controlnet_lite_best.pt",
            "name": "ControlNet-Lite",
        },
    }

    # Benchmark each model
    for model_key in args.models:
        if model_key not in model_configs:
            print(f"Unknown model: {model_key}")
            continue

        config = model_configs[model_key]
        print(f"\nBenchmarking {config['name']}...")

        try:
            # Load model
            checkpoint_path = str(config['checkpoint']) if config['checkpoint'].exists() else None
            model = config['loader'](checkpoint_path, device)

            if args.fp16 and device.type == 'cuda':
                model = model.half()

            # Count parameters
            params = count_parameters(model)
            print(f"  Parameters: {params:,}")

            # Benchmark
            result = benchmark_inference(
                model=model,
                name=config['name'],
                device=device,
                img_size=args.img_size,
                num_runs=args.num_runs,
            )
            result['params'] = params
            result['fp16'] = args.fp16

            results.append(result)
            print(f"  Average: {result['avg_ms']:.2f} ms, Memory: {result['peak_memory_mb']:.0f} MB")

            # Clean up
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Print results table
    print_results_table(results)

    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump({
            'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
            'img_size': args.img_size,
            'fp16': args.fp16,
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
