#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
Test-Time Augmentation (TTA) for Restormer Inference

HONEST ASSESSMENT:
- TTA can improve L1 by 1-3% (marginal)
- Costs 2-8x inference time
- For hackathon with 30% cost metric, may not be worth it
- Test and compare with/without TTA to decide

Augmentation strategies:
- flip_h: Horizontal flip (2x cost, ~1-2% improvement)
- flip_both: H + V flip (4x cost, ~2-3% improvement)
- full: All rotations + flips (8x cost, ~2-4% improvement)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import time

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer


def load_model(checkpoint_path, device='cuda'):
    """Load trained Restormer model"""
    model = Restormer(
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


class TTAPredictor:
    """Test-Time Augmentation Predictor"""

    def __init__(self, model, device='cuda', strategy='flip_h'):
        """
        Args:
            model: Trained model
            device: Device to run on
            strategy: TTA strategy
                - 'none': No TTA (baseline)
                - 'flip_h': Horizontal flip only (2x)
                - 'flip_both': H + V flip (4x)
                - 'full': All 8 orientations (8x)
        """
        self.model = model
        self.device = device
        self.strategy = strategy

        # Define augmentation functions
        self.augmentations = self._get_augmentations()

    def _get_augmentations(self):
        """Get list of (augment_fn, inverse_fn) based on strategy"""
        # Identity (always included)
        identity = (lambda x: x, lambda x: x)

        # Horizontal flip
        flip_h = (
            lambda x: torch.flip(x, dims=[-1]),
            lambda x: torch.flip(x, dims=[-1])
        )

        # Vertical flip
        flip_v = (
            lambda x: torch.flip(x, dims=[-2]),
            lambda x: torch.flip(x, dims=[-2])
        )

        # Both flips
        flip_hv = (
            lambda x: torch.flip(x, dims=[-1, -2]),
            lambda x: torch.flip(x, dims=[-1, -2])
        )

        # 90 degree rotation
        rot90 = (
            lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
            lambda x: torch.rot90(x, k=-1, dims=[-2, -1])
        )

        # 180 degree rotation
        rot180 = (
            lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
            lambda x: torch.rot90(x, k=2, dims=[-2, -1])
        )

        # 270 degree rotation
        rot270 = (
            lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
            lambda x: torch.rot90(x, k=1, dims=[-2, -1])
        )

        # 90 + flip
        rot90_flip = (
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1]),
            lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=-1, dims=[-2, -1])
        )

        if self.strategy == 'none':
            return [identity]
        elif self.strategy == 'flip_h':
            return [identity, flip_h]
        elif self.strategy == 'flip_both':
            return [identity, flip_h, flip_v, flip_hv]
        elif self.strategy == 'full':
            return [identity, flip_h, flip_v, flip_hv, rot90, rot180, rot270, rot90_flip]
        else:
            raise ValueError(f"Unknown TTA strategy: {self.strategy}")

    @torch.no_grad()
    def predict(self, img_tensor):
        """
        Run prediction with TTA.

        Args:
            img_tensor: Input tensor [1, C, H, W]

        Returns:
            output: Averaged prediction [1, C, H, W]
        """
        predictions = []

        for augment_fn, inverse_fn in self.augmentations:
            # Augment input
            augmented = augment_fn(img_tensor)

            # Run model
            pred = self.model(augmented)

            # Inverse augment prediction
            pred_restored = inverse_fn(pred)

            predictions.append(pred_restored)

        # Average all predictions
        output = torch.stack(predictions, dim=0).mean(dim=0)

        return output


def process_image(img_path, model, tta_predictor, resolution, device):
    """Process single image with TTA"""
    # Load image
    img = Image.open(img_path).convert('RGB')
    original_size = img.size

    # Resize
    img_resized = img.resize((resolution, resolution), Image.LANCZOS)

    # To tensor
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Predict with TTA
    output = tta_predictor.predict(img_tensor)

    # To numpy
    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)

    return output_np, original_size


def calculate_l1_loss(pred_path, target_path, resolution):
    """Calculate L1 loss between prediction and target"""
    pred = Image.open(pred_path).convert('RGB').resize((resolution, resolution), Image.LANCZOS)
    target = Image.open(target_path).convert('RGB').resize((resolution, resolution), Image.LANCZOS)

    pred_np = np.array(pred).astype(np.float32) / 255.0
    target_np = np.array(target).astype(np.float32) / 255.0

    l1_loss = np.mean(np.abs(pred_np - target_np))
    return l1_loss


def main():
    parser = argparse.ArgumentParser(description="Test inference with TTA")
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--resolution', type=int, default=896, help='Inference resolution')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--tta', type=str, default='flip_h',
                        choices=['none', 'flip_h', 'flip_both', 'full'],
                        help='TTA strategy (none=baseline)')
    parser.add_argument('--test_jsonl', type=str, default='test.jsonl', help='Test set JSONL')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    print("=" * 70)
    print("TEST INFERENCE WITH TTA")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Resolution: {args.resolution}")
    print(f"  TTA Strategy: {args.tta}")
    print(f"  Output: {args.output_dir}")

    # Load model
    print(f"\n📂 Loading model...")
    model = load_model(args.model, args.device)

    # Create TTA predictor
    tta_predictor = TTAPredictor(model, args.device, args.tta)
    num_augmentations = len(tta_predictor.augmentations)
    print(f"   TTA augmentations: {num_augmentations}x")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test pairs
    base_dir = Path(args.test_jsonl).parent
    if str(base_dir) == '.':
        base_dir = Path('.')

    test_pairs = []
    with open(args.test_jsonl) as f:
        for line in f:
            if line.strip():
                test_pairs.append(json.loads(line.strip()))

    print(f"\n🧪 Processing {len(test_pairs)} test images...")

    results = []
    total_time = 0

    for pair in test_pairs:
        src_path = base_dir / pair['src']
        tar_path = base_dir / pair['tar']

        # Get image number
        img_num = Path(pair['src']).stem.split('_')[0]

        # Time the inference
        start_time = time.time()

        # Process with TTA
        output_np, original_size = process_image(src_path, model, tta_predictor, args.resolution, args.device)

        inference_time = time.time() - start_time
        total_time += inference_time

        # Save output
        output_path = output_dir / f"{img_num}_output.jpg"
        Image.fromarray(output_np).save(output_path, quality=95)

        # Copy source and target for comparison
        Image.open(src_path).save(output_dir / f"{img_num}_src.jpg", quality=95)
        Image.open(tar_path).save(output_dir / f"{img_num}_tar.jpg", quality=95)

        # Calculate L1 loss
        l1_loss = calculate_l1_loss(output_path, tar_path, args.resolution)

        results.append({
            'image_num': img_num,
            'src': str(pair['src']),
            'tar': str(pair['tar']),
            'output': str(output_path),
            'l1_loss': float(l1_loss),
            'inference_time': inference_time
        })

        print(f"   {img_num}: L1={l1_loss:.4f}, Time={inference_time:.2f}s")

    # Calculate averages
    avg_l1 = np.mean([r['l1_loss'] for r in results])
    avg_time = total_time / len(results)

    # Save results
    results_data = {
        'model': args.model,
        'resolution': args.resolution,
        'tta_strategy': args.tta,
        'num_augmentations': num_augmentations,
        'num_samples': len(results),
        'avg_l1_loss': float(avg_l1),
        'avg_inference_time': float(avg_time),
        'total_inference_time': float(total_time),
        'results': results
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"📊 RESULTS")
    print(f"=" * 70)
    print(f"   TTA Strategy: {args.tta} ({num_augmentations}x augmentations)")
    print(f"   Avg L1 Loss: {avg_l1:.4f}")
    print(f"   Avg Inference Time: {avg_time:.2f}s per image")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"\n   Results saved to: {output_dir / 'results.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
