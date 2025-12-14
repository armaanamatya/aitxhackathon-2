#!/usr/bin/env python3
"""
Run DarkIR inference on held-out test set.
Saves: input, output, target, and side-by-side comparison.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add DarkIR to path
sys.path.insert(0, str(Path(__file__).parent / 'DarkIR'))
from archs.DarkIR import DarkIR


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_l1(img1, img2):
    """Calculate L1 (MAE) between two images normalized to [0,1]"""
    return np.mean(np.abs(img1.astype(float) - img2.astype(float))) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Run DarkIR inference on test set")
    parser.add_argument('--checkpoint', type=str,
                        default='outputs_darkir_512_cv/fold_1/checkpoint_best.pt')
    parser.add_argument('--test_jsonl', type=str, default='data_splits/test.jsonl')
    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='test/darkir_baseline')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DARKIR TEST SET INFERENCE")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test set: {args.test_jsonl}")
    print(f"Output: {args.output_dir}")

    # Load model
    print("\nLoading model...")
    model = DarkIR(
        img_channel=3,
        width=args.width,
        middle_blk_num_enc=2,
        middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3],
        dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'unknown'))
    print(f"Loaded checkpoint from epoch {epoch}")
    if isinstance(val_loss, float):
        print(f"Checkpoint val loss: {val_loss:.4f}")

    # Load test pairs
    test_pairs = []
    with open(args.test_jsonl) as f:
        for line in f:
            if line.strip():
                test_pairs.append(json.loads(line.strip()))

    print(f"\nTest images: {len(test_pairs)} (NEVER SEEN DURING TRAINING)")
    print("-" * 70)

    # Run inference
    results = []

    with torch.no_grad():
        for pair in tqdm(test_pairs, desc="Processing"):
            src_path = os.path.join(args.base_dir, pair['src'])
            tar_path = os.path.join(args.base_dir, pair['tar'])

            # Load images
            src_orig = cv2.imread(src_path)
            tar_orig = cv2.imread(tar_path)

            if src_orig is None or tar_orig is None:
                print(f"Warning: Could not load {src_path} or {tar_path}")
                continue

            src_rgb = cv2.cvtColor(src_orig, cv2.COLOR_BGR2RGB)
            tar_rgb = cv2.cvtColor(tar_orig, cv2.COLOR_BGR2RGB)

            # Resize for model
            src_resized = cv2.resize(src_rgb, (args.resolution, args.resolution),
                                      interpolation=cv2.INTER_AREA)
            tar_resized = cv2.resize(tar_rgb, (args.resolution, args.resolution),
                                      interpolation=cv2.INTER_AREA)

            # To tensor
            src_tensor = torch.from_numpy(src_resized).permute(2, 0, 1).float() / 255.0
            src_tensor = src_tensor.unsqueeze(0).to(args.device)

            # Inference
            out_tensor = model(src_tensor)
            out_tensor = torch.clamp(out_tensor, 0, 1)

            # To numpy
            out_np = (out_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Calculate metrics
            psnr = calculate_psnr(out_np, tar_resized)
            l1 = calculate_l1(out_np, tar_resized)

            # Get image name
            img_name = Path(pair['src']).stem.replace('_src', '')

            results.append({
                'image': img_name,
                'psnr': float(psnr),
                'l1': float(l1)
            })

            # Save outputs
            # 1. Input (source)
            cv2.imwrite(str(output_dir / f"{img_name}_input.jpg"),
                       cv2.cvtColor(src_resized, cv2.COLOR_RGB2BGR))

            # 2. Output (prediction)
            cv2.imwrite(str(output_dir / f"{img_name}_output.jpg"),
                       cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))

            # 3. Target (ground truth)
            cv2.imwrite(str(output_dir / f"{img_name}_target.jpg"),
                       cv2.cvtColor(tar_resized, cv2.COLOR_RGB2BGR))

            # 4. Side-by-side comparison (Input | Output | Target)
            comparison = np.hstack([src_resized, out_np, tar_resized])
            cv2.imwrite(str(output_dir / f"{img_name}_comparison.jpg"),
                       cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

            print(f"  {img_name}: PSNR={psnr:.2f}dB, L1={l1:.4f}")

    # Calculate averages
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_l1 = np.mean([r['l1'] for r in results])

    # Save results
    summary = {
        'model': 'DarkIR',
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': epoch,
        'checkpoint_val_loss': val_loss if isinstance(val_loss, float) else None,
        'resolution': args.resolution,
        'test_samples': len(results),
        'avg_psnr': float(avg_psnr),
        'avg_l1': float(avg_l1),
        'per_image': results
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS (NO DATA LEAKAGE)")
    print("=" * 70)
    print(f"Model: DarkIR (width={args.width})")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Test images: {len(results)}")
    print(f"")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average L1:   {avg_l1:.4f}")
    print(f"")
    print(f"Outputs saved to: {output_dir}/")
    print("  - {name}_input.jpg      (source image)")
    print("  - {name}_output.jpg     (model prediction)")
    print("  - {name}_target.jpg     (ground truth)")
    print("  - {name}_comparison.jpg (side-by-side: input|output|target)")
    print("  - results.json          (metrics)")
    print("=" * 70)


if __name__ == '__main__':
    main()
