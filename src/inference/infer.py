"""
Inference Script for Real Estate HDR Enhancement

Supports:
- Single image inference
- Batch directory processing
- Multiple output resolutions
- PyTorch and TensorRT optimized models
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.models import UNetGenerator


class HDREnhancer:
    """
    Real Estate HDR Enhancement inference class.

    Handles:
    - Loading PyTorch or TensorRT models
    - Single and batch inference
    - Resolution handling (tiles for high-res)
    """

    def __init__(
        self,
        model_path: str,
        image_size: int = 512,
        device: str = "cuda",
        use_tensorrt: bool = False,
        precision: str = "fp16",
    ):
        """
        Args:
            model_path: Path to model weights
            image_size: Model input size
            device: Device to run on
            use_tensorrt: Whether to use TensorRT model
            precision: Precision for inference (fp32, fp16)
        """
        self.image_size = image_size
        self.device = torch.device(device)
        self.precision = precision
        self.use_fp16 = precision == "fp16" and device == "cuda"

        print(f"Loading model from {model_path}...")
        print(f"Device: {device}, Precision: {precision}")

        if use_tensorrt and model_path.endswith('.ts'):
            # Load TensorRT model
            self.model = torch.jit.load(model_path)
        else:
            # Load PyTorch model
            self.model = UNetGenerator(
                in_channels=3,
                out_channels=3,
                base_features=64,
                num_residual_blocks=9,
                learn_residual=True,
            )

            state_dict = torch.load(model_path, map_location=self.device)
            if 'generator_state_dict' in state_dict:
                state_dict = state_dict['generator_state_dict']
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()

        # Compile with torch.compile for additional speedup
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not available: {e}")

        # Warmup
        self._warmup()

    def _warmup(self, num_runs: int = 3):
        """Warmup the model for accurate timing."""
        dummy = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        if self.use_fp16:
            dummy = dummy.half()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy)

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        print("Model warmed up")

    def preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model input.

        Args:
            image: PIL Image

        Returns:
            Tensor [1, 3, H, W] in [-1, 1], original size
        """
        original_size = image.size  # (W, H)

        # Resize to model input size
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensor
        tensor = torch.from_numpy(np.array(image)).float()
        tensor = tensor.permute(2, 0, 1) / 255.0  # [H, W, C] -> [C, H, W], [0, 255] -> [0, 1]
        tensor = tensor * 2 - 1  # [0, 1] -> [-1, 1]
        tensor = tensor.unsqueeze(0).to(self.device)

        if self.use_fp16:
            tensor = tensor.half()

        return tensor, original_size

    def postprocess(
        self,
        tensor: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Convert model output tensor to PIL Image.

        Args:
            tensor: Output tensor [1, 3, H, W] in [-1, 1]
            original_size: If provided, resize to this size (W, H)

        Returns:
            PIL Image
        """
        # Convert to numpy
        tensor = tensor.squeeze(0).float().cpu()
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
        tensor = tensor.clamp(0, 1)
        tensor = tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        tensor = (tensor * 255).astype(np.uint8)

        image = Image.fromarray(tensor)

        # Resize to original size if provided
        if original_size is not None:
            image = image.resize(original_size, Image.LANCZOS)

        return image

    @torch.no_grad()
    def enhance(
        self,
        image: Image.Image,
        preserve_resolution: bool = True,
    ) -> Image.Image:
        """
        Enhance a single image.

        Args:
            image: Input PIL Image
            preserve_resolution: If True, output matches input resolution

        Returns:
            Enhanced PIL Image
        """
        # Preprocess
        tensor, original_size = self.preprocess(image)

        # Run model
        output = self.model(tensor)

        # Postprocess
        if preserve_resolution:
            return self.postprocess(output, original_size)
        else:
            return self.postprocess(output)

    @torch.no_grad()
    def enhance_tiled(
        self,
        image: Image.Image,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> Image.Image:
        """
        Enhance high-resolution image using tiled processing.

        This allows processing images larger than the model's training size
        while maintaining quality.

        Args:
            image: Input PIL Image
            tile_size: Size of each tile
            overlap: Overlap between tiles for seamless blending

        Returns:
            Enhanced PIL Image at original resolution
        """
        original_size = image.size  # (W, H)
        W, H = original_size

        # Pad image if needed
        pad_w = (tile_size - W % tile_size) % tile_size
        pad_h = (tile_size - H % tile_size) % tile_size

        # Convert to tensor for easier manipulation
        img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor * 2 - 1  # [0, 1] -> [-1, 1]

        # Pad
        if pad_w > 0 or pad_h > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        _, H_pad, W_pad = img_tensor.shape

        # Process tiles
        output_tensor = torch.zeros_like(img_tensor)
        weight_tensor = torch.zeros(1, H_pad, W_pad)

        step = tile_size - overlap

        for y in range(0, H_pad - tile_size + 1, step):
            for x in range(0, W_pad - tile_size + 1, step):
                # Extract tile
                tile = img_tensor[:, y:y+tile_size, x:x+tile_size]
                tile = tile.unsqueeze(0).to(self.device)

                if self.use_fp16:
                    tile = tile.half()

                # Process tile
                enhanced_tile = self.model(tile).float().cpu().squeeze(0)

                # Blend with weights (simple averaging for overlap regions)
                output_tensor[:, y:y+tile_size, x:x+tile_size] += enhanced_tile
                weight_tensor[:, y:y+tile_size, x:x+tile_size] += 1

        # Average overlapping regions
        output_tensor = output_tensor / weight_tensor.clamp(min=1)

        # Remove padding
        output_tensor = output_tensor[:, :H, :W]

        # Convert back to image
        output_tensor = (output_tensor + 1) / 2
        output_tensor = output_tensor.clamp(0, 1)
        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_np = (output_np * 255).astype(np.uint8)

        return Image.fromarray(output_np)

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        preserve_resolution: bool = True,
        use_tiled: bool = False,
    ) -> dict:
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: File extensions to process
            preserve_resolution: Maintain original resolution
            use_tiled: Use tiled processing for high-res

        Returns:
            Statistics dict
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"Found {len(image_files)} images to process")

        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'total_time': 0,
        }

        times = []

        for img_path in tqdm(image_files, desc="Processing"):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')

                # Process
                start = time.perf_counter()

                if use_tiled and max(image.size) > self.image_size * 2:
                    enhanced = self.enhance_tiled(image)
                else:
                    enhanced = self.enhance(image, preserve_resolution)

                elapsed = time.perf_counter() - start
                times.append(elapsed)

                # Save
                output_file = output_path / img_path.name
                enhanced.save(output_file, quality=95)

                stats['success'] += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                stats['failed'] += 1

        stats['total_time'] = sum(times)
        stats['avg_time'] = np.mean(times) if times else 0
        stats['throughput'] = len(times) / stats['total_time'] if stats['total_time'] > 0 else 0

        print(f"\nProcessing complete:")
        print(f"  Success: {stats['success']}/{stats['total']}")
        print(f"  Average time: {stats['avg_time']*1000:.2f} ms/image")
        print(f"  Throughput: {stats['throughput']:.2f} images/sec")

        return stats


def main():
    parser = argparse.ArgumentParser(description="HDR Enhancement Inference")

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model weights")

    # Input/output
    parser.add_argument("--input", type=str, required=True,
                        help="Input image or directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image or directory")

    # Model settings
    parser.add_argument("--image_size", type=int, default=512,
                        help="Model input size")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16"],
                        help="Inference precision")
    parser.add_argument("--tensorrt", action="store_true",
                        help="Use TensorRT model")

    # Processing options
    parser.add_argument("--preserve_resolution", action="store_true", default=True,
                        help="Preserve original image resolution")
    parser.add_argument("--tiled", action="store_true",
                        help="Use tiled processing for high-res images")

    args = parser.parse_args()

    # Create enhancer
    enhancer = HDREnhancer(
        model_path=args.model_path,
        image_size=args.image_size,
        precision=args.precision,
        use_tensorrt=args.tensorrt,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single image processing
        print(f"Processing single image: {input_path}")

        image = Image.open(input_path).convert('RGB')

        start = time.perf_counter()
        if args.tiled and max(image.size) > args.image_size * 2:
            enhanced = enhancer.enhance_tiled(image)
        else:
            enhanced = enhancer.enhance(image, args.preserve_resolution)
        elapsed = time.perf_counter() - start

        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path, quality=95)

        print(f"Saved to: {output_path}")
        print(f"Processing time: {elapsed*1000:.2f} ms")

    elif input_path.is_dir():
        # Batch processing
        enhancer.process_directory(
            input_dir=str(input_path),
            output_dir=str(output_path),
            preserve_resolution=args.preserve_resolution,
            use_tiled=args.tiled,
        )
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
