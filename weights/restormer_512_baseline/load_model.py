#!/usr/bin/env python3
"""
Load Restormer 512 Baseline Model

Standalone script to load the model for inference.
Can be used by the inference team.

Usage:
    python load_model.py --input image.jpg --output output_hdr.jpg
    python load_model.py --input_dir ./images/ --output_dir ./results/
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

# Add src to path (adjust if needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from training.restormer import create_restormer


def load_model(checkpoint_path, device='cuda'):
    """
    Load the Restormer model from checkpoint.

    Args:
        checkpoint_path: Path to model_checkpoint.pt
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model in eval mode
        checkpoint_info: Dict with training info
    """
    print(f"Loading model from: {checkpoint_path}")

    # Create model
    model = create_restormer('base')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to eval mode
    model.eval()

    # Move to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Extract info
    info = {
        'epoch': checkpoint['epoch'],
        'val_l1': checkpoint.get('val_l1', 'N/A'),
        'device': str(device),
    }

    print(f"✓ Model loaded successfully!")
    print(f"  Epoch: {info['epoch']}")
    print(f"  Val L1: {info['val_l1']}")
    print(f"  Device: {info['device']}")

    return model, info


def process_single_image(model, input_path, output_path, device='cuda'):
    """
    Process a single image.

    Args:
        model: Loaded model
        input_path: Path to input image
        output_path: Path to save output image
        device: Device to run on
    """
    # Load and preprocess
    img = Image.open(input_path).convert('RGB')
    original_size = img.size

    # Resize to 512x512
    img_resized = TF.resize(img, (512, 512))
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.clamp(output, 0, 1)

    # Convert back to PIL
    output_img = TF.to_pil_image(output.squeeze(0).cpu())

    # Resize back to original size (optional)
    # output_img = output_img.resize(original_size, Image.LANCZOS)

    # Save
    output_img.save(output_path)


def process_directory(model, input_dir, output_dir, device='cuda'):
    """
    Process all images in a directory.

    Args:
        model: Loaded model
        input_dir: Directory with input images
        output_dir: Directory to save output images
        device: Device to run on
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Supported extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))

    print(f"\nFound {len(image_files)} images in {input_dir}")

    # Process each image
    for i, img_path in enumerate(image_files, 1):
        output_path = output_dir / f"{img_path.stem}_hdr{img_path.suffix}"

        print(f"[{i}/{len(image_files)}] Processing {img_path.name}...")

        try:
            process_single_image(model, img_path, output_path, device)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print(f"\n✓ Processed {len(image_files)} images")
    print(f"  Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Restormer 512 Baseline Inference')
    parser.add_argument('--checkpoint', type=str, default='model_checkpoint.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--input_dir', type=str, help='Input directory (batch mode)')
    parser.add_argument('--output_dir', type=str, help='Output directory (batch mode)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to run on')
    args = parser.parse_args()

    print("="*80)
    print("RESTORMER 512 BASELINE - INFERENCE")
    print("="*80)

    # Load model
    model, info = load_model(args.checkpoint, args.device)

    # Single image mode
    if args.input and args.output:
        print(f"\nProcessing single image: {args.input}")
        process_single_image(model, args.input, args.output, args.device)
        print(f"✓ Output saved to: {args.output}")

    # Batch directory mode
    elif args.input_dir and args.output_dir:
        process_directory(model, args.input_dir, args.output_dir, args.device)

    else:
        print("\nError: Specify either --input/--output OR --input_dir/--output_dir")
        parser.print_help()
        return

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
