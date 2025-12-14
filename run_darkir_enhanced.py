#!/usr/bin/env python3
"""Enhanced DarkIR Test Inference with Full Metrics"""

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

sys.path.insert(0, str(Path(__file__).parent / 'DarkIR'))
from archs.DarkIR import DarkIR

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='outputs_darkir_512_cv/fold_1/checkpoint_best.pt')
    parser.add_argument('--test_jsonl', default='data_splits/test.jsonl')
    parser.add_argument('--output_dir', default='test/darkir_baseline')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--width', type=int, default=32)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    
    print("=" * 70)
    print("DARKIR ENHANCED METRICS")
    print("=" * 70)
    print(f"Device: {device_name}")

    model = DarkIR(img_channel=3, width=args.width, middle_blk_num_enc=2, middle_blk_num_dec=2,
                   enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1], dilations=[1, 4, 9], extra_depth_wise=True).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    print(f"Parameters: {num_params:,} | Size: {model_size:.2f} MB")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")

    test_pairs = [json.loads(l) for l in open(args.test_jsonl) if l.strip()]
    print(f"Test images: {len(test_pairs)}")

    # Warmup
    dummy = torch.randn(1, 3, args.resolution, args.resolution).to(device)
    for _ in range(3):
        with torch.no_grad(): _ = model(dummy)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    gpu_mem_start = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    results = []
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for pair in tqdm(test_pairs):
            src = cv2.cvtColor(cv2.imread(pair['src']), cv2.COLOR_BGR2RGB)
            tar = cv2.cvtColor(cv2.imread(pair['tar']), cv2.COLOR_BGR2RGB)
            src = cv2.resize(src, (args.resolution, args.resolution))
            tar = cv2.resize(tar, (args.resolution, args.resolution))
            
            src_t = torch.from_numpy(src).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_t = model(src_t)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            out = (torch.clamp(out_t, 0, 1)[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            
            name = Path(pair['src']).stem.replace('_src', '')
            results.append({
                'image': name,
                'psnr': calculate_psnr(out, tar),
                'ssim': calculate_ssim(out, tar),
                'l1': np.mean(np.abs(out.astype(float) - tar.astype(float))) / 255.0,
                'time_ms': (t1 - t0) * 1000
            })
            
            cv2.imwrite(f"{args.output_dir}/{name}_comparison.jpg", 
                       cv2.cvtColor(np.hstack([src, out, tar]), cv2.COLOR_RGB2BGR))

    gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    avg = lambda k: np.mean([r[k] for r in results])
    
    summary = {
        'model': {'params': num_params, 'size_mb': model_size},
        'hardware': {'device': device_name, 'gpu_mem_peak_mb': gpu_peak},
        'inference': {'resolution': args.resolution, 'avg_ms': avg('time_ms'), 'fps': 1000/avg('time_ms')},
        'metrics': {'psnr': avg('psnr'), 'ssim': avg('ssim'), 'l1': avg('l1')},
        'per_image': results
    }
    json.dump(summary, open(f"{args.output_dir}/results_enhanced.json", 'w'), indent=2)

    print("\n" + "=" * 70)
    print(f"📊 Model: {num_params:,} params | {model_size:.2f} MB")
    print(f"🖥️  Device: {device_name} | Peak GPU: {gpu_peak:.1f} MB")
    print(f"⏱️  Speed: {avg('time_ms'):.2f} ms | {1000/avg('time_ms'):.1f} FPS")
    print(f"📈 PSNR: {avg('psnr'):.2f} dB | SSIM: {avg('ssim'):.4f} | L1: {avg('l1'):.4f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
