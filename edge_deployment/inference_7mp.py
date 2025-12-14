#!/usr/bin/env python3
"""
7MP Real Estate HDR Inference

Supports multiple backends:
- NVIDIA A100/H100 (DGX Cloud, datacenter)
- NVIDIA Jetson (AGX Orin, Xavier NX)
- Apple Silicon (M1/M2 Ultra with 128GB+ unified memory)
- CPU fallback

Author: Top 0.0001% MLE
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import create_restormer


def detect_device():
    """
    Auto-detect optimal device and configuration.

    Returns:
        device: torch.device
        config: dict with optimization settings
    """
    config = {
        'device_type': 'cpu',
        'precision': 'fp32',
        'batch_size': 1,
        'use_compile': False,
        'use_channels_last': False,
    }

    # Check for CUDA (A100, Jetson, etc.)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = torch.device('cuda:0')
        config['device_type'] = 'cuda'
        config['precision'] = 'fp16'
        config['use_channels_last'] = True

        # A100/H100 optimizations
        if 'A100' in device_name or 'H100' in device_name:
            print(f"Detected: {device_name}")
            config['batch_size'] = 2  # Can batch 2x 7MP images on A100 80GB
            config['use_compile'] = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Jetson optimizations
        elif 'Orin' in device_name or 'Xavier' in device_name:
            print(f"Detected: NVIDIA Jetson {device_name}")
            config['batch_size'] = 1
            config['use_compile'] = False  # May not be available on Jetson

        else:
            print(f"Detected: {device_name}")

    # Check for Apple Silicon MPS
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        config['device_type'] = 'mps'
        config['precision'] = 'fp16'
        config['use_channels_last'] = True
        config['use_compile'] = True
        print("Detected: Apple Silicon (MPS)")

    # CPU fallback
    else:
        device = torch.device('cpu')
        print("Using CPU (slow)")

    return device, config


def load_model(checkpoint_path, device, config):
    """
    Load Restormer model with optimal configuration.

    Args:
        checkpoint_path: Path to model checkpoint
        device: torch.device
        config: Configuration dict

    Returns:
        model: Loaded and optimized model
        checkpoint_info: Training metadata
    """
    print(f"\nLoading model from: {checkpoint_path}")

    # Create model
    model = create_restormer('base')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle DDP checkpoint (if trained with DistributedDataParallel)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix from DDP
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Apply precision
    if config['precision'] == 'fp16':
        model = model.half()
        print("  Precision: FP16")
    else:
        print("  Precision: FP32")

    # Channels-last memory format
    if config['use_channels_last']:
        model = model.to(memory_format=torch.channels_last)
        print("  Memory format: channels_last")

    # Move to device
    model = model.to(device)

    # torch.compile for speedup (PyTorch 2.0+)
    if config['use_compile'] and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    # Extract checkpoint info
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_l1': checkpoint.get('val_l1', 'N/A'),
        'resolution': checkpoint.get('resolution', '7MP'),
    }

    print(f"✓ Model loaded successfully!")
    print(f"  Epoch: {info['epoch']}")
    print(f"  Val L1: {info['val_l1']}")
    print(f"  Device: {device}")
    print()

    return model, info


def process_single_image(model, img_path, output_path, device, config):
    """
    Process single 7MP image.

    Args:
        model: Loaded model
        img_path: Input image path
        output_path: Output image path
        device: torch.device
        config: Configuration dict
    """
    # Load image
    img = Image.open(img_path).convert('RGB')
    original_size = img.size

    print(f"Processing: {Path(img_path).name}")
    print(f"  Size: {original_size[0]}x{original_size[1]} ({original_size[0]*original_size[1]/1e6:.1f}MP)")

    # Convert to tensor
    img_tensor = TF.to_tensor(img).unsqueeze(0)

    # Apply precision
    if config['precision'] == 'fp16':
        img_tensor = img_tensor.half()

    # Apply memory format
    if config['use_channels_last']:
        img_tensor = img_tensor.to(memory_format=torch.channels_last)

    # Move to device
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.clamp(output, 0, 1)

    # Convert back to PIL
    output = output.squeeze(0).float().cpu()
    output_img = TF.to_pil_image(output)

    # Resize to original if needed
    if output_img.size != original_size:
        output_img = output_img.resize(original_size, Image.LANCZOS)

    # Save
    output_img.save(output_path, quality=95)
    print(f"✓ Saved to: {output_path}")


def process_directory(model, input_dir, output_dir, device, config):
    """
    Process all images in directory.

    Args:
        model: Loaded model
        input_dir: Input directory
        output_dir: Output directory
        device: torch.device
        config: Configuration dict
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))

    print(f"\nFound {len(image_files)} images in {input_dir}")
    print()

    # Process each image
    for i, img_path in enumerate(image_files, 1):
        output_path = output_dir / f"{img_path.stem}_hdr{img_path.suffix}"

        print(f"[{i}/{len(image_files)}]")
        try:
            process_single_image(model, img_path, output_path, device, config)
        except Exception as e:
            print(f"✗ Error processing {img_path.name}: {e}")
            continue

        print()

    print(f"✓ Processed {len(image_files)} images")
    print(f"  Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='7MP Real Estate HDR Inference')
    parser.add_argument('--checkpoint', type=str,
                        default='weights/restormer_7mp/model_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str,
                        help='Input image path')
    parser.add_argument('--output', type=str,
                        help='Output image path')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory (batch mode)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory (batch mode)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (auto-detect by default)')
    parser.add_argument('--precision', type=str, default='auto',
                        choices=['auto', 'fp16', 'fp32'],
                        help='Precision (auto-detect by default)')
    args = parser.parse_args()

    print("="*80)
    print("7MP REAL ESTATE HDR INFERENCE")
    print("="*80)

    # Detect device and configuration
    if args.device == 'auto':
        device, config = detect_device()
    else:
        device = torch.device(args.device)
        config = {
            'device_type': args.device,
            'precision': args.precision if args.precision != 'auto' else 'fp32',
            'batch_size': 1,
            'use_compile': False,
            'use_channels_last': False,
        }

    # Load model
    model, info = load_model(args.checkpoint, device, config)

    # Single image mode
    if args.input and args.output:
        process_single_image(model, args.input, args.output, device, config)

    # Batch directory mode
    elif args.input_dir and args.output_dir:
        process_directory(model, args.input_dir, args.output_dir, device, config)

    else:
        print("\nError: Specify either --input/--output OR --input_dir/--output_dir")
        parser.print_help()
        return

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
