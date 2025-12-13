#!/usr/bin/env python3
"""
Inference with Test-Time Augmentation (TTA) for AutoHDR
========================================================

Improves image quality by averaging predictions from:
- Original image
- Horizontally flipped image

This typically improves PSNR/SSIM by 0.1-0.3 dB with 2x inference cost.

Usage:
    python infer_tta.py --input image.jpg --output enhanced.jpg --checkpoint outputs/checkpoints/best_generator.pt
    python infer_tta.py --input_dir data/test --output_dir results --checkpoint outputs/checkpoints/best_generator.pt
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def load_image(path: str, size: Optional[int] = None) -> torch.Tensor:
    """Load and preprocess image."""
    img = Image.open(path).convert('RGB')
    original_size = img.size

    transform_list = []
    if size:
        transform_list.append(transforms.Resize((size, size)))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])

    transform = transforms.Compose(transform_list)
    return transform(img).unsqueeze(0), original_size


def save_image(tensor: torch.Tensor, path: str, original_size: Optional[tuple] = None):
    """Save tensor as image."""
    # Denormalize from [-1, 1] to [0, 1]
    img = tensor.squeeze(0).clamp(-1, 1)
    img = (img + 1) / 2

    # Convert to numpy
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    # Create PIL image
    pil_img = Image.fromarray(img)

    # Resize to original if needed
    if original_size:
        pil_img = pil_img.resize(original_size, Image.LANCZOS)

    pil_img.save(path, quality=95)


def tta_inference(model: nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Test-time augmentation inference.

    Averages predictions from:
    1. Original image
    2. Horizontally flipped image (flipped back)

    Args:
        model: Generator model
        image: Input image tensor [1, 3, H, W]
        device: Device to run on

    Returns:
        Enhanced image tensor [1, 3, H, W]
    """
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        # Original prediction
        out_original = model(image)

        # Horizontal flip prediction
        flipped = torch.flip(image, dims=[3])  # Flip along width
        out_flipped = model(flipped)
        out_flipped = torch.flip(out_flipped, dims=[3])  # Flip back

        # Average predictions
        output = (out_original + out_flipped) / 2

    return output


def load_model(checkpoint_path: str, model_type: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    if model_type == "unet":
        from src.training.models import UNetGenerator
        model = UNetGenerator(
            in_channels=3,
            out_channels=3,
            base_features=64,
            num_residual_blocks=9,
            learn_residual=True,
        )
    elif model_type == "restormer":
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
    elif model_type == "hat":
        from src.training.hat import HAT
        model = HAT(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=96,
            depths=[6, 6, 6, 6],
            num_heads=[6, 6, 6, 6],
            window_size=8,
        )
    elif model_type == "retinexmamba":
        from src.training.retinexmamba import RetinexMamba
        model = RetinexMamba(
            in_channels=3,
            out_channels=3,
            dim=64,
            num_blocks=4,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    if 'generator_state_dict' in state:
        model.load_state_dict(state['generator_state_dict'])
    elif 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    return model.to(device)


def process_single_image(
    model: nn.Module,
    input_path: str,
    output_path: str,
    device: torch.device,
    size: Optional[int] = None,
    use_tta: bool = True,
    fp16: bool = False,
) -> float:
    """
    Process a single image.

    Returns inference time in milliseconds.
    """
    # Load image
    image, original_size = load_image(input_path, size)

    if fp16:
        image = image.half()

    # Inference
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()

    if use_tta:
        output = tta_inference(model, image, device)
    else:
        with torch.no_grad():
            output = model(image.to(device))

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) * 1000

    # Save output
    save_image(output.float(), output_path, original_size if not size else None)

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="AutoHDR Inference with TTA")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--input_dir", type=str, help="Input directory for batch processing")
    parser.add_argument("--output", type=str, help="Output image path")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch processing")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "restormer", "hat", "retinexmamba"],
                        help="Model architecture")
    parser.add_argument("--size", type=int, default=None,
                        help="Process at specific size (default: original size)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision")

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input_dir is required")
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"TTA: {'Enabled' if not args.no_tta else 'Disabled'}")
    print(f"FP16: {args.fp16}")
    print("")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.model, device)
    model.eval()

    if args.fp16 and device.type == 'cuda':
        model = model.half()

    # Process single image
    if args.input:
        print(f"Processing: {args.input}")
        elapsed = process_single_image(
            model=model,
            input_path=args.input,
            output_path=args.output,
            device=device,
            size=args.size,
            use_tta=not args.no_tta,
            fp16=args.fp16,
        )
        print(f"Saved: {args.output}")
        print(f"Inference time: {elapsed:.2f} ms")

    # Process directory
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        print(f"Found {len(images)} images")

        times = []
        for img_path in tqdm(images, desc="Processing"):
            output_path = output_dir / img_path.name
            elapsed = process_single_image(
                model=model,
                input_path=str(img_path),
                output_path=str(output_path),
                device=device,
                size=args.size,
                use_tta=not args.no_tta,
                fp16=args.fp16,
            )
            times.append(elapsed)

        # Statistics
        print(f"\nProcessed {len(images)} images")
        print(f"Average time: {np.mean(times):.2f} ms")
        print(f"Median time: {np.median(times):.2f} ms")
        print(f"Total time: {sum(times) / 1000:.2f} s")


if __name__ == "__main__":
    main()
