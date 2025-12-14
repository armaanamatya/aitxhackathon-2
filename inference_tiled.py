#!/usr/bin/env python3
"""
Tiled Inference for High-Resolution Images (3301×2199)
=======================================================
Optimized for DGX Spark (128GB unified memory, 273 GB/s bandwidth)

Key features:
- Tile-based processing for memory efficiency
- Overlap with feathered blending (no seams)
- Batch processing for throughput
- FP16 mixed precision
- TensorRT-ready

Author: Top 0.0001% MLE
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.restormer import create_restormer
from models.color_refiner import create_elite_color_refiner


def create_feather_mask(tile_size, overlap, device='cuda'):
    """
    Create a feathered weight mask for seamless tile blending.

    Uses cosine interpolation in overlap regions for smooth transitions.
    """
    mask = torch.ones(tile_size, tile_size, device=device)

    if overlap > 0:
        # Create 1D ramp
        ramp = torch.linspace(0, 1, overlap, device=device)
        ramp = (1 - torch.cos(ramp * np.pi)) / 2  # Cosine feathering

        # Apply to edges
        mask[:overlap, :] *= ramp.view(-1, 1)  # Top
        mask[-overlap:, :] *= ramp.flip(0).view(-1, 1)  # Bottom
        mask[:, :overlap] *= ramp.view(1, -1)  # Left
        mask[:, -overlap:] *= ramp.flip(0).view(1, -1)  # Right

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def extract_tiles(image, tile_size, overlap):
    """
    Extract overlapping tiles from image.

    Args:
        image: [B, C, H, W] tensor
        tile_size: Size of each tile
        overlap: Overlap between tiles

    Returns:
        tiles: List of [B, C, tile_size, tile_size] tensors
        positions: List of (x, y) top-left positions
    """
    _, _, H, W = image.shape
    stride = tile_size - overlap

    tiles = []
    positions = []

    y = 0
    while y < H:
        x = 0
        while x < W:
            # Adjust for edge tiles
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = y_end - tile_size
            x_start = x_end - tile_size

            tile = image[:, :, y_start:y_end, x_start:x_end]
            tiles.append(tile)
            positions.append((x_start, y_start))

            x += stride
            if x >= W - overlap:
                break

        y += stride
        if y >= H - overlap:
            break

    return tiles, positions


def blend_tiles(tiles, positions, output_shape, tile_size, overlap, device='cuda'):
    """
    Blend tiles with feathered weights for seamless output.

    Args:
        tiles: List of processed tiles
        positions: List of (x, y) positions
        output_shape: (H, W) of output
        tile_size: Size of each tile
        overlap: Overlap between tiles

    Returns:
        output: [B, C, H, W] blended result
    """
    B, C = tiles[0].shape[:2]
    H, W = output_shape

    output = torch.zeros(B, C, H, W, device=device)
    weight_sum = torch.zeros(B, 1, H, W, device=device)

    # Create feather mask
    mask = create_feather_mask(tile_size, overlap, device)

    for tile, (x, y) in zip(tiles, positions):
        tile = tile.to(device)

        # Get actual tile dimensions (may be smaller at edges)
        _, _, th, tw = tile.shape

        # Adjust mask for edge tiles
        tile_mask = mask[:, :, :th, :tw]

        output[:, :, y:y+th, x:x+tw] += tile * tile_mask
        weight_sum[:, :, y:y+th, x:x+tw] += tile_mask

    # Normalize by weights
    output = output / (weight_sum + 1e-8)

    return output


class TiledInference:
    """
    High-resolution tiled inference pipeline.

    Optimized for DGX Spark:
    - 768×768 tiles (good context for transformer)
    - 96px overlap (12.5% for smooth blending)
    - Batch of 4 tiles (fits in memory)
    - FP16 mixed precision
    """

    def __init__(
        self,
        backbone_path,
        refiner_path=None,
        tile_size=768,
        overlap=96,
        batch_size=4,
        device='cuda',
        use_fp16=True
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16

        # Load backbone
        print(f"Loading backbone from {backbone_path}...")
        self.backbone = create_restormer('base').to(device)
        checkpoint = torch.load(backbone_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.eval()

        # Load refiner (optional)
        self.refiner = None
        if refiner_path and os.path.exists(refiner_path):
            print(f"Loading refiner from {refiner_path}...")
            self.refiner = create_elite_color_refiner(size='medium').to(device)
            refiner_checkpoint = torch.load(refiner_path, map_location='cpu')

            if 'refiner_state_dict' in refiner_checkpoint:
                self.refiner.load_state_dict(refiner_checkpoint['refiner_state_dict'])
            else:
                self.refiner.load_state_dict(refiner_checkpoint)

            self.refiner.eval()

        print(f"Tile size: {tile_size}, Overlap: {overlap}, Batch: {batch_size}")
        print(f"FP16: {use_fp16}")

    @torch.no_grad()
    def process_batch(self, tiles):
        """Process a batch of tiles."""
        batch = torch.stack(tiles).to(self.device)

        if self.use_fp16:
            with torch.cuda.amp.autocast():
                output = self.backbone(batch.half())
                if self.refiner:
                    output = self.refiner(output, batch.half())
                output = output.float()
        else:
            output = self.backbone(batch)
            if self.refiner:
                output = self.refiner(output, batch)

        output = torch.clamp(output, 0, 1)
        return [output[i] for i in range(output.shape[0])]

    def __call__(self, image):
        """
        Process a high-resolution image.

        Args:
            image: [1, C, H, W] tensor or PIL Image

        Returns:
            output: [1, C, H, W] processed image
        """
        # Convert PIL to tensor if needed
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        image = image.to(self.device)
        _, C, H, W = image.shape

        # Pad to tile-aligned size
        stride = self.tile_size - self.overlap
        pad_H = ceil((H - self.overlap) / stride) * stride + self.overlap
        pad_W = ceil((W - self.overlap) / stride) * stride + self.overlap

        if pad_H > H or pad_W > W:
            image = F.pad(image, (0, pad_W - W, 0, pad_H - H), mode='reflect')

        # Extract tiles
        tiles, positions = extract_tiles(image, self.tile_size, self.overlap)
        print(f"Processing {len(tiles)} tiles...")

        # Process in batches
        processed_tiles = []
        for i in range(0, len(tiles), self.batch_size):
            batch = tiles[i:i + self.batch_size]
            # Squeeze batch dimension for each tile
            batch = [t.squeeze(0) for t in batch]
            processed = self.process_batch(batch)
            processed_tiles.extend(processed)

        # Blend tiles
        output = blend_tiles(
            [t.unsqueeze(0) for t in processed_tiles],
            positions,
            (pad_H, pad_W),
            self.tile_size,
            self.overlap,
            self.device
        )

        # Crop to original size
        output = output[:, :, :H, :W]

        return output


def main():
    parser = argparse.ArgumentParser(description='Tiled High-Resolution Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--backbone', type=str, required=True, help='Backbone checkpoint')
    parser.add_argument('--refiner', type=str, default=None, help='Refiner checkpoint (optional)')
    parser.add_argument('--tile_size', type=int, default=768, help='Tile size')
    parser.add_argument('--overlap', type=int, default=96, help='Overlap between tiles')
    parser.add_argument('--batch_size', type=int, default=4, help='Tiles per batch')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize inference pipeline
    pipeline = TiledInference(
        backbone_path=args.backbone,
        refiner_path=args.refiner,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        use_fp16=args.fp16
    )

    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    print(f"\nProcessing {len(image_files)} images...")

    total_time = 0
    for img_path in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")

        # Load image
        image = Image.open(img_path).convert('RGB')
        print(f"Size: {image.size[0]}×{image.size[1]}")

        # Process
        start_time = time.time()
        output = pipeline(image)
        elapsed = time.time() - start_time
        total_time += elapsed

        print(f"Time: {elapsed:.2f}s")

        # Save output
        output_np = (output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        output_img = Image.fromarray(output_np)

        output_path = output_dir / f"{img_path.stem}_enhanced.png"
        output_img.save(output_path, quality=95)
        print(f"Saved: {output_path}")

    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average: {total_time/len(image_files):.2f}s per image")


if __name__ == '__main__':
    main()
