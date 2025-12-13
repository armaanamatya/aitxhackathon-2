"""
Real Estate HDR Dataset
Loads paired source (unedited) and target (professionally edited) images.
"""

import os
import json
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class RealEstateHDRDataset(Dataset):
    """
    Dataset for paired real estate image enhancement.

    Loads source images (unedited) and target images (professionally edited)
    from the AutoHDR dataset format.
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        image_size: int = 512,
        split: str = "train",
        train_ratio: float = 0.9,
        augment: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_root: Root directory containing images folder
            jsonl_path: Path to train.jsonl file
            image_size: Target size for images (will be resized)
            split: 'train' or 'val'
            train_ratio: Ratio of data to use for training
            augment: Whether to apply data augmentation
            seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == "train")

        # Load image pairs from JSONL
        self.pairs = self._load_pairs(jsonl_path)

        # Split into train/val
        random.seed(seed)
        indices = list(range(len(self.pairs)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        # Define transforms
        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.LANCZOS,
            antialias=True
        )

        self.to_tensor = transforms.ToTensor()

        # Normalize to [-1, 1] for diffusion models
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        print(f"Loaded {len(self.indices)} {split} samples")

    def _load_pairs(self, jsonl_path: str) -> List[Dict[str, str]]:
        """Load image pairs from JSONL file."""
        pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                pair = json.loads(line.strip())
                # Verify both files exist
                src_path = self.data_root / pair['src']
                tar_path = self.data_root / pair['tar']
                if src_path.exists() and tar_path.exists():
                    pairs.append(pair)
        return pairs

    def __len__(self) -> int:
        return len(self.indices)

    def _apply_augmentation(
        self,
        src_img: Image.Image,
        tar_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply synchronized augmentation to both images."""
        # Random horizontal flip
        if random.random() > 0.5:
            src_img = transforms.functional.hflip(src_img)
            tar_img = transforms.functional.hflip(tar_img)

        # Random vertical flip (less common for real estate)
        if random.random() > 0.9:
            src_img = transforms.functional.vflip(src_img)
            tar_img = transforms.functional.vflip(tar_img)

        # Random rotation (small angles only)
        if random.random() > 0.7:
            angle = random.uniform(-5, 5)
            src_img = transforms.functional.rotate(src_img, angle)
            tar_img = transforms.functional.rotate(tar_img, angle)

        return src_img, tar_img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'source': Source image tensor [C, H, W] in [-1, 1]
                - 'target': Target image tensor [C, H, W] in [-1, 1]
                - 'source_path': Path to source image
                - 'target_path': Path to target image
        """
        pair_idx = self.indices[idx]
        pair = self.pairs[pair_idx]

        # Load images
        src_path = self.data_root / pair['src']
        tar_path = self.data_root / pair['tar']

        src_img = Image.open(src_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Resize
        src_img = self.resize(src_img)
        tar_img = self.resize(tar_img)

        # Augmentation (synchronized for both)
        if self.augment:
            src_img, tar_img = self._apply_augmentation(src_img, tar_img)

        # Convert to tensor and normalize
        src_tensor = self.normalize(self.to_tensor(src_img))
        tar_tensor = self.normalize(self.to_tensor(tar_img))

        return {
            'source': src_tensor,
            'target': tar_tensor,
            'source_path': str(src_path),
            'target_path': str(tar_path),
        }


class RealEstateInferenceDataset(Dataset):
    """Dataset for inference - only loads source images."""

    def __init__(
        self,
        image_paths: List[str],
        image_size: int = 512,
    ):
        self.image_paths = image_paths
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.LANCZOS,
                antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Store original size for later upscaling
        original_size = img.size

        img_tensor = self.transform(img)

        return {
            'image': img_tensor,
            'path': img_path,
            'original_size': original_size,
        }


def get_dataloaders(
    data_root: str,
    jsonl_path: str,
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 4,
    train_ratio: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_root: Root directory containing images
        jsonl_path: Path to train.jsonl
        batch_size: Batch size for training
        image_size: Target image size
        num_workers: Number of data loading workers
        train_ratio: Train/val split ratio

    Returns:
        train_loader, val_loader
    """
    train_dataset = RealEstateHDRDataset(
        data_root=data_root,
        jsonl_path=jsonl_path,
        image_size=image_size,
        split="train",
        train_ratio=train_ratio,
        augment=True,
    )

    val_dataset = RealEstateHDRDataset(
        data_root=data_root,
        jsonl_path=jsonl_path,
        image_size=image_size,
        split="val",
        train_ratio=train_ratio,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = RealEstateHDRDataset(
        data_root=".",
        jsonl_path="train.jsonl",
        image_size=512,
        split="train",
    )

    sample = dataset[0]
    print(f"Source shape: {sample['source'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Source range: [{sample['source'].min():.2f}, {sample['source'].max():.2f}]")
