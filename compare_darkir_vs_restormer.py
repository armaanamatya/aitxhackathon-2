#!/usr/bin/env python3
"""
Compare DarkIR vs Restormer on held-out test set.
Generates comprehensive metrics and side-by-side visual comparisons.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'DarkIR'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))

from archs.DarkIR import DarkIR
from restormer import Restormer


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def load_darkir(checkpoint_path, device, width=32):
    """Load DarkIR model"""
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint.get('epoch', '?'), checkpoint.get('best_val_loss', checkpoint.get('val_loss', '?'))


def load_restormer(checkpoint_path, device, dim=48, num_blocks=[4,6,6,8]):
    """Load Restormer model"""
    model = Restormer(
        in_channels=3,
        out_channels=3,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.0,
        bias=False,
        use_checkpointing=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint.get('epoch', '?'), checkpoint.get('best_val_loss', checkpoint.get('val_loss', '?'))


def run_inference(model, src_tensor, device):
    """Run inference with timing"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(src_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return out, (t1 - t0) * 1000


def main():
    parser = argparse.ArgumentParser(description="Compare DarkIR vs Restormer")
    parser.add_argument('--darkir_ckpt', default='outputs_darkir_512_cv/fold_1/checkpoint_best.pt')
    parser.add_argument('--restormer_ckpt', default='outputs_full_light_aug/checkpoint_best.pt')
    parser.add_argument('--test_jsonl', default='data_splits/test.jsonl')
    parser.add_argument('--output_dir', default='test/comparison')
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    print("=" * 80)
    print("DARKIR vs RESTORMER COMPARISON")
    print("=" * 80)
    print(f"Device: {device_name}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print()

    # Load models
    print("Loading DarkIR...")
    darkir, darkir_epoch, darkir_val = load_darkir(args.darkir_ckpt, device)
    darkir_params = sum(p.numel() for p in darkir.parameters())
    darkir_size = sum(p.numel() * p.element_size() for p in darkir.parameters()) / 1024 / 1024
    print(f"  Epoch: {darkir_epoch}, Val Loss: {darkir_val}")
    print(f"  Params: {darkir_params:,} ({darkir_size:.2f} MB)")

    print("\nLoading Restormer...")
    restormer, restormer_epoch, restormer_val = load_restormer(args.restormer_ckpt, device)
    restormer_params = sum(p.numel() for p in restormer.parameters())
    restormer_size = sum(p.numel() * p.element_size() for p in restormer.parameters()) / 1024 / 1024
    print(f"  Epoch: {restormer_epoch}, Val Loss: {restormer_val}")
    print(f"  Params: {restormer_params:,} ({restormer_size:.2f} MB)")

    # Warmup
    print("\nWarming up models...")
    dummy = torch.randn(1, 3, args.resolution, args.resolution).to(device)
    for _ in range(3):
        with torch.no_grad():
            _ = darkir(dummy)
            _ = restormer(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Load test set
    test_pairs = [json.loads(l) for l in open(args.test_jsonl) if l.strip()]
    print(f"\nTest images: {len(test_pairs)}")
    print("-" * 80)

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    darkir_results = []
    restormer_results = []

    for pair in tqdm(test_pairs, desc="Comparing"):
        # Load and preprocess
        src = cv2.cvtColor(cv2.imread(pair['src']), cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(cv2.imread(pair['tar']), cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (args.resolution, args.resolution))
        tar = cv2.resize(tar, (args.resolution, args.resolution))

        src_t = torch.from_numpy(src).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

        # DarkIR inference
        darkir_out, darkir_time = run_inference(darkir, src_t, device)
        darkir_np = (torch.clamp(darkir_out, 0, 1)[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)

        # Restormer inference
        restormer_out, restormer_time = run_inference(restormer, src_t, device)
        restormer_np = (torch.clamp(restormer_out, 0, 1)[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)

        name = Path(pair['src']).stem.replace('_src', '')

        # Calculate metrics
        darkir_results.append({
            'image': name,
            'psnr': calculate_psnr(darkir_np, tar),
            'ssim': calculate_ssim(darkir_np, tar),
            'l1': np.mean(np.abs(darkir_np.astype(float) - tar.astype(float))) / 255.0,
            'time_ms': darkir_time
        })

        restormer_results.append({
            'image': name,
            'psnr': calculate_psnr(restormer_np, tar),
            'ssim': calculate_ssim(restormer_np, tar),
            'l1': np.mean(np.abs(restormer_np.astype(float) - tar.astype(float))) / 255.0,
            'time_ms': restormer_time
        })

        # Save comparison image: Input | DarkIR | Restormer | Target
        comparison = np.hstack([src, darkir_np, restormer_np, tar])
        cv2.imwrite(str(output_dir / f"{name}_comparison.jpg"),
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    # Calculate averages
    def avg(results, key):
        return np.mean([r[key] for r in results])

    gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print("\n┌─────────────────┬─────────────┬─────────────┬─────────────┐")
    print("│     Metric      │   DarkIR    │  Restormer  │   Winner    │")
    print("├─────────────────┼─────────────┼─────────────┼─────────────┤")

    darkir_psnr = avg(darkir_results, 'psnr')
    restormer_psnr = avg(restormer_results, 'psnr')
    psnr_winner = "DarkIR" if darkir_psnr > restormer_psnr else "Restormer"
    print(f"│ PSNR (dB)       │ {darkir_psnr:11.2f} │ {restormer_psnr:11.2f} │ {psnr_winner:^11} │")

    darkir_ssim = avg(darkir_results, 'ssim')
    restormer_ssim = avg(restormer_results, 'ssim')
    ssim_winner = "DarkIR" if darkir_ssim > restormer_ssim else "Restormer"
    print(f"│ SSIM            │ {darkir_ssim:11.4f} │ {restormer_ssim:11.4f} │ {ssim_winner:^11} │")

    darkir_l1 = avg(darkir_results, 'l1')
    restormer_l1 = avg(restormer_results, 'l1')
    l1_winner = "DarkIR" if darkir_l1 < restormer_l1 else "Restormer"
    print(f"│ L1 Loss         │ {darkir_l1:11.4f} │ {restormer_l1:11.4f} │ {l1_winner:^11} │")

    darkir_ms = avg(darkir_results, 'time_ms')
    restormer_ms = avg(restormer_results, 'time_ms')
    speed_winner = "DarkIR" if darkir_ms < restormer_ms else "Restormer"
    print(f"│ Inference (ms)  │ {darkir_ms:11.2f} │ {restormer_ms:11.2f} │ {speed_winner:^11} │")

    darkir_fps = 1000 / darkir_ms
    restormer_fps = 1000 / restormer_ms
    fps_winner = "DarkIR" if darkir_fps > restormer_fps else "Restormer"
    print(f"│ FPS             │ {darkir_fps:11.1f} │ {restormer_fps:11.1f} │ {fps_winner:^11} │")

    params_winner = "DarkIR" if darkir_params < restormer_params else "Restormer"
    print(f"│ Params (M)      │ {darkir_params/1e6:11.2f} │ {restormer_params/1e6:11.2f} │ {params_winner:^11} │")

    size_winner = "DarkIR" if darkir_size < restormer_size else "Restormer"
    print(f"│ Size (MB)       │ {darkir_size:11.2f} │ {restormer_size:11.2f} │ {size_winner:^11} │")

    print("└─────────────────┴─────────────┴─────────────┴─────────────┘")

    # Per-image comparison
    print("\n" + "=" * 80)
    print("PER-IMAGE PSNR COMPARISON")
    print("=" * 80)
    for d, r in zip(darkir_results, restormer_results):
        diff = d['psnr'] - r['psnr']
        winner = "DarkIR" if diff > 0 else "Restormer"
        print(f"  {d['image']:20s}: DarkIR={d['psnr']:.2f}dB, Restormer={r['psnr']:.2f}dB → {winner} (+{abs(diff):.2f}dB)")

    # Save JSON results
    summary = {
        'test_images': len(test_pairs),
        'resolution': args.resolution,
        'device': device_name,
        'gpu_peak_mb': gpu_peak,
        'darkir': {
            'checkpoint': args.darkir_ckpt,
            'epoch': darkir_epoch,
            'val_loss': darkir_val if isinstance(darkir_val, float) else None,
            'params': darkir_params,
            'size_mb': darkir_size,
            'avg_psnr': darkir_psnr,
            'avg_ssim': darkir_ssim,
            'avg_l1': darkir_l1,
            'avg_ms': darkir_ms,
            'fps': darkir_fps,
            'per_image': darkir_results
        },
        'restormer': {
            'checkpoint': args.restormer_ckpt,
            'epoch': restormer_epoch,
            'val_loss': restormer_val if isinstance(restormer_val, float) else None,
            'params': restormer_params,
            'size_mb': restormer_size,
            'avg_psnr': restormer_psnr,
            'avg_ssim': restormer_ssim,
            'avg_l1': restormer_l1,
            'avg_ms': restormer_ms,
            'fps': restormer_fps,
            'per_image': restormer_results
        }
    }

    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")
    print("  - comparison_results.json")
    print("  - {name}_comparison.jpg (Input | DarkIR | Restormer | Target)")
    print("=" * 80)


if __name__ == '__main__':
    main()
