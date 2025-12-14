#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
Enhanced test inference with data-driven pre/post processing.
Based on dataset analysis: gamma correction (pre) + brightness/shadow/saturation adjustment (post)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys
from PIL import Image, ImageEnhance

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer


def apply_gamma(img_np, gamma=0.418):
    """Apply gamma correction (pre-processing)."""
    return np.power(img_np, gamma)


def adjust_brightness(img, factor=1.75):
    """Adjust brightness (post-processing)."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img, factor=1.19):
    """Adjust contrast (post-processing)."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def adjust_saturation(img, factor=0.57):
    """Adjust saturation (post-processing)."""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def lift_shadows(img_np, amount=0.18):
    """Lift shadows selectively (post-processing)."""
    # Convert to HSV to work with luminance
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    hsv = np.array(img_pil.convert('HSV')).astype(np.float32) / 255.0

    # Extract value channel (brightness)
    v = hsv[:,:,2]

    # Create shadow mask (stronger effect on darker pixels)
    shadow_mask = 1.0 - v
    shadow_mask = np.clip(shadow_mask, 0, 1)

    # Apply shadow lift
    v_lifted = v + (amount * shadow_mask)
    v_lifted = np.clip(v_lifted, 0, 1)

    # Put back
    hsv[:,:,2] = v_lifted

    # Convert back to RGB
    hsv_uint8 = (hsv * 255).astype(np.uint8)
    img_lifted = Image.fromarray(hsv_uint8, mode='HSV').convert('RGB')

    return np.array(img_lifted).astype(np.float32) / 255.0


def load_image_with_preprocessing(path, target_size=None, use_gamma=True):
    """Load and preprocess image with gamma correction."""
    img = Image.open(path).convert('RGB')

    if target_size:
        img = img.resize((target_size, target_size), Image.BICUBIC)

    # Convert to tensor [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0

    # Apply gamma correction (pre-processing)
    if use_gamma:
        img_np = apply_gamma(img_np, gamma=0.418)

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def save_image_with_postprocessing(tensor, path, use_postprocessing=True):
    """Save tensor as image with post-processing adjustments."""
    # Convert to numpy and PIL
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np_uint8 = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np_uint8)

    if use_postprocessing:
        # Apply data-driven post-processing
        # 1. Lift shadows (+18.2%)
        img_np_lifted = lift_shadows(img_np, amount=0.182)
        img = Image.fromarray((img_np_lifted * 255).astype(np.uint8))

        # 2. Brightness boost (1.75x)
        img = adjust_brightness(img, factor=1.75)

        # 3. Contrast enhancement (1.19x)
        img = adjust_contrast(img, factor=1.19)

        # 4. Saturation reduction (0.57x)
        img = adjust_saturation(img, factor=0.57)

    # Save
    img.save(path, quality=95)


def run_inference(model_path, test_jsonl, output_subdir, resolution,
                  use_preprocessing=False, use_postprocessing=False, device='cuda'):
    """Run inference with optional pre/post processing."""
    suffix = ""
    if use_preprocessing and use_postprocessing:
        suffix = " (Pre+Post)"
    elif use_preprocessing:
        suffix = " (Pre)"
    elif use_postprocessing:
        suffix = " (Post)"

    print(f"\n{'='*60}")
    print(f"Running inference: {output_subdir}{suffix}")
    print(f"Model: {model_path}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Pre-processing: {'✓ Gamma' if use_preprocessing else '✗'}")
    print(f"Post-processing: {'✓ Brightness/Shadow/Sat' if use_postprocessing else '✗'}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path('test') / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
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

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load test samples
    print(f"\nLoading test samples from {test_jsonl}...")
    test_samples = []
    with open(test_jsonl) as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line.strip()))

    print(f"Found {len(test_samples)} test samples\n")

    # Run inference
    results = []
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_samples, desc="Processing")):
            src_path = Path(sample['src'])
            tar_path = Path(sample['tar'])

            # Load source image (with optional pre-processing)
            src_tensor = load_image_with_preprocessing(src_path, target_size=resolution,
                                                      use_gamma=use_preprocessing)
            src_tensor = src_tensor.to(device)

            # Run model
            out_tensor = model(src_tensor)

            # Generate output filename
            img_num = src_path.stem.split('_')[0]
            output_filename = f"{img_num}_output.jpg"
            output_path = output_dir / output_filename

            # Save output (with optional post-processing)
            save_image_with_postprocessing(out_tensor, output_path,
                                          use_postprocessing=use_postprocessing)

            # Also save source and target for comparison (no processing)
            src_copy_path = output_dir / f"{img_num}_src.jpg"
            tar_copy_path = output_dir / f"{img_num}_tar.jpg"

            # Load unprocessed versions for comparison
            src_img = Image.open(src_path).convert('RGB').resize((resolution, resolution), Image.BICUBIC)
            tar_img = Image.open(tar_path).convert('RGB').resize((resolution, resolution), Image.BICUBIC)

            src_img.save(src_copy_path, quality=95)
            tar_img.save(tar_copy_path, quality=95)

            # Compute metrics (on unprocessed output vs target)
            tar_np = np.array(tar_img).astype(np.float32) / 255.0
            tar_tensor = torch.from_numpy(tar_np).permute(2, 0, 1).unsqueeze(0).to(device)

            l1_loss = F.l1_loss(out_tensor, tar_tensor).item()

            results.append({
                'image_num': img_num,
                'src': str(src_path),
                'tar': str(tar_path),
                'output': str(output_path),
                'l1_loss': l1_loss
            })

    # Save results summary
    summary_path = output_dir / 'results.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'model': str(model_path),
            'resolution': resolution,
            'preprocessing': use_preprocessing,
            'postprocessing': use_postprocessing,
            'num_samples': len(test_samples),
            'avg_l1_loss': float(np.mean([r['l1_loss'] for r in results])),
            'results': results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"Output directory: {output_dir}")
    print(f"Average L1 loss: {np.mean([r['l1_loss'] for r in results]):.4f}")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}\n")

    return results


def main():
    """Main inference function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    test_jsonl = Path('test.jsonl')

    if not test_jsonl.exists():
        print(f"ERROR: {test_jsonl} not found!")
        return

    # Test best model (896) with different processing combinations
    best_checkpoint = Path('outputs_restormer_896_v2/checkpoint_best.pt')

    if not best_checkpoint.exists():
        print(f"ERROR: {best_checkpoint} not found!")
        return

    # Run 4 variants to compare
    print("\n" + "="*60)
    print("TESTING DATA-DRIVEN PRE/POST PROCESSING")
    print("="*60)
    print("\nBased on dataset analysis:")
    print("  Pre:  Gamma correction (γ=0.418)")
    print("  Post: Shadow lift (+18%), Brightness (1.75x), Contrast (1.19x), Saturation (0.57x)")
    print()

    configs = [
        {'subdir': 'restormer_896_baseline', 'pre': False, 'post': False},
        {'subdir': 'restormer_896_pre_only', 'pre': True, 'post': False},
        {'subdir': 'restormer_896_post_only', 'pre': False, 'post': True},
        {'subdir': 'restormer_896_pre_post', 'pre': True, 'post': True},
    ]

    for config in configs:
        try:
            run_inference(
                model_path=best_checkpoint,
                test_jsonl=test_jsonl,
                output_subdir=config['subdir'],
                resolution=896,
                use_preprocessing=config['pre'],
                use_postprocessing=config['post'],
                device=device
            )
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print("\nCompare outputs in:")
    print("  test/restormer_896_baseline/     (no processing)")
    print("  test/restormer_896_pre_only/     (gamma correction)")
    print("  test/restormer_896_post_only/    (brightness/shadow/sat)")
    print("  test/restormer_896_pre_post/     (all processing)")
    print()


if __name__ == '__main__':
    main()
