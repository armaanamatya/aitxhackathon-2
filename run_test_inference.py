#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
Robust test inference script for Restormer models.
Runs both 384x384 and 896x896 models on test.jsonl and saves results to test/ directory.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer


def load_image(path, target_size=None):
    """Load and preprocess image."""
    img = Image.open(path).convert('RGB')

    if target_size:
        # Resize to target size
        img = img.resize((target_size, target_size), Image.BICUBIC)

    # Convert to tensor [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def save_image(tensor, path):
    """Save tensor as image."""
    # Clamp to [0, 1] and convert to numpy
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)

    # Save
    img = Image.fromarray(img_np)
    img.save(path, quality=95)


def run_inference(model_path, test_jsonl, output_subdir, resolution, device='cuda'):
    """
    Run inference on test dataset.

    Args:
        model_path: Path to checkpoint_best.pt
        test_jsonl: Path to test.jsonl
        output_subdir: Subdirectory name under test/ (e.g., 'restormer_384')
        resolution: Target resolution (384 or 896)
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Running inference: {output_subdir}")
    print(f"Model: {model_path}")
    print(f"Resolution: {resolution}x{resolution}")
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
        best_val = checkpoint.get('best_val_loss', None)
        print(f"Loaded checkpoint from epoch {epoch}")
        if best_val is not None:
            print(f"Best val loss: {best_val:.4f}")
        else:
            print("Best val loss: unknown")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (legacy format)")

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

            # Load source image
            src_tensor = load_image(src_path, target_size=resolution)
            src_tensor = src_tensor.to(device)

            # Run model
            out_tensor = model(src_tensor)

            # Generate output filename
            # Extract image number from path (e.g., "1_src.jpg" -> "1")
            img_num = src_path.stem.split('_')[0]
            output_filename = f"{img_num}_output.jpg"
            output_path = output_dir / output_filename

            # Save output
            save_image(out_tensor, output_path)

            # Also save source and target for comparison
            src_copy_path = output_dir / f"{img_num}_src.jpg"
            tar_copy_path = output_dir / f"{img_num}_tar.jpg"

            # Load and save source/target at same resolution
            src_resized = load_image(src_path, target_size=resolution)
            tar_resized = load_image(tar_path, target_size=resolution)

            save_image(src_resized, src_copy_path)
            save_image(tar_resized, tar_copy_path)

            # Compute metrics
            tar_tensor = tar_resized.to(device)
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

    # Run inference on all models
    models = [
        {
            'name': 'Restormer 384x384 (Best: 0.0588)',
            'checkpoint': Path('outputs_restormer_384_v2/checkpoint_best.pt'),
            'output_subdir': 'restormer_384',
            'resolution': 384
        },
        {
            'name': 'Restormer 896x896 (Best: 0.0635)',
            'checkpoint': Path('outputs_restormer_896_v2/checkpoint_best.pt'),
            'output_subdir': 'restormer_896',
            'resolution': 896
        },
        {
            'name': 'Restormer Pretrained 384x384 (Best: 0.0618)',
            'checkpoint': Path('outputs_restormer_pretrained_384_v2/checkpoint_best.pt'),
            'output_subdir': 'restormer_pretrained_384',
            'resolution': 384
        }
    ]

    for model_config in models:
        if not model_config['checkpoint'].exists():
            print(f"WARNING: {model_config['checkpoint']} not found, skipping...")
            continue

        try:
            run_inference(
                model_path=model_config['checkpoint'],
                test_jsonl=test_jsonl,
                output_subdir=model_config['output_subdir'],
                resolution=model_config['resolution'],
                device=device
            )
        except Exception as e:
            print(f"ERROR running {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("ALL INFERENCE COMPLETE!")
    print("="*60)
    print("\nOutput structure:")
    print("test/")
    print("├── restormer_384/")
    print("│   ├── 1_src.jpg       (source image)")
    print("│   ├── 1_tar.jpg       (target image)")
    print("│   ├── 1_output.jpg    (model output)")
    print("│   ├── ...             (other test images)")
    print("│   └── results.json    (metrics)")
    print("└── restormer_896/")
    print("    ├── 1_src.jpg")
    print("    ├── 1_tar.jpg")
    print("    ├── 1_output.jpg")
    print("    ├── ...")
    print("    └── results.json")
    print()


if __name__ == '__main__':
    main()
