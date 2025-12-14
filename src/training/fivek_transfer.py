#!/usr/bin/env python3
"""
FiveK Transfer Learning for Real Estate HDR Enhancement
========================================================
Top 0.0001% MLE implementation for transfer learning from MIT-Adobe FiveK dataset.

Why FiveK?
- 5000 images with expert retouching (5 experts: A, B, C, D, E)
- Diverse scenes including outdoor, windows, high dynamic range
- Professional color grading and tone mapping
- Teaches model general HDR/tone mapping principles

Transfer Learning Strategy:
1. Phase 1: Pretrain on FiveK (learn general tone mapping)
2. Phase 2: Fine-tune on real estate (domain adaptation)

Key Insights:
- Use Expert C (most consistent, middle-of-road style)
- Lower learning rate for fine-tuning (preserve FiveK knowledge)
- Gradual unfreezing for stable transfer
- Domain adaptation through careful data mixing

Author: Top MLE
Date: 2025-12-13
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm


class FiveKDataset(Dataset):
    """
    MIT-Adobe FiveK Dataset for pretraining.

    Dataset structure expected:
    fivek_dataset/
        input/          # Original images (DNG converted to JPG/PNG)
        expertC/        # Expert C retouched images

    Or using the prepared pairs format:
    fivek_dataset/
        pairs.jsonl     # {"input": "path", "target": "path"}
    """

    def __init__(self,
                 data_dir: str,
                 resolution: int = 512,
                 expert: str = 'C',
                 augment: bool = True,
                 max_samples: Optional[int] = None):
        """
        Args:
            data_dir: Path to FiveK dataset
            resolution: Training resolution
            expert: Which expert to use (A, B, C, D, E) - C is recommended
            augment: Whether to apply augmentation
            max_samples: Maximum number of samples (for faster experiments)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.augment = augment

        # Try to load from pairs.jsonl first
        pairs_file = self.data_dir / 'pairs.jsonl'
        if pairs_file.exists():
            self.pairs = self._load_pairs_jsonl(pairs_file)
        else:
            # Fall back to directory-based loading
            self.pairs = self._load_from_directories(expert)

        if max_samples and len(self.pairs) > max_samples:
            random.seed(42)
            self.pairs = random.sample(self.pairs, max_samples)

        print(f"FiveK dataset: {len(self.pairs)} pairs loaded")

    def _load_pairs_jsonl(self, pairs_file: Path) -> List[Dict]:
        """Load pairs from JSONL file"""
        pairs = []
        with open(pairs_file) as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line.strip()))
        return pairs

    def _load_from_directories(self, expert: str) -> List[Dict]:
        """Load pairs from input/expertX directories"""
        input_dir = self.data_dir / 'input'
        expert_dir = self.data_dir / f'expert{expert}'

        if not input_dir.exists() or not expert_dir.exists():
            raise ValueError(f"Expected directories: {input_dir} and {expert_dir}")

        pairs = []
        for input_file in sorted(input_dir.glob('*.jpg')) + sorted(input_dir.glob('*.png')):
            # Find corresponding expert file
            expert_file = expert_dir / input_file.name
            if expert_file.exists():
                pairs.append({
                    'input': str(input_file),
                    'target': str(expert_file)
                })

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Handle both absolute and relative paths
        if os.path.isabs(pair['input']):
            input_path = pair['input']
            target_path = pair['target']
        else:
            input_path = self.data_dir / pair['input']
            target_path = self.data_dir / pair['target']

        # Load images
        input_img = cv2.imread(str(input_path))
        target_img = cv2.imread(str(target_path))

        if input_img is None:
            raise ValueError(f"Could not load: {input_path}")
        if target_img is None:
            raise ValueError(f"Could not load: {target_path}")

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Random crop (FiveK images are typically larger)
        h, w = input_img.shape[:2]
        if h > self.resolution and w > self.resolution:
            if self.augment:
                top = random.randint(0, h - self.resolution)
                left = random.randint(0, w - self.resolution)
            else:
                top = (h - self.resolution) // 2
                left = (w - self.resolution) // 2

            input_img = input_img[top:top+self.resolution, left:left+self.resolution]
            target_img = target_img[top:top+self.resolution, left:left+self.resolution]
        else:
            # Resize if smaller
            input_img = cv2.resize(input_img, (self.resolution, self.resolution))
            target_img = cv2.resize(target_img, (self.resolution, self.resolution))

        # Augmentation (helpful for FiveK - more diverse than real estate)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                input_img = np.fliplr(input_img).copy()
                target_img = np.fliplr(target_img).copy()

        # To tensor
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor


class TransferLearningTrainer:
    """
    Two-phase transfer learning trainer.

    Phase 1: Pretrain on FiveK
    - Learn general tone mapping and HDR handling
    - Use full learning rate
    - Longer training (more data)

    Phase 2: Fine-tune on Real Estate
    - Domain adaptation
    - Lower learning rate (10-20% of pretrain LR)
    - Early stopping
    - Optional: Gradual unfreezing
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 output_dir: str = 'outputs_transfer'):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'phase1_train_losses': [],
            'phase1_val_losses': [],
            'phase2_train_losses': [],
            'phase2_val_losses': [],
        }

    def pretrain_on_fivek(self,
                          fivek_dir: str,
                          resolution: int = 512,
                          batch_size: int = 16,
                          epochs: int = 30,
                          lr: float = 2e-4,
                          num_workers: int = 8,
                          criterion: nn.Module = None,
                          val_split: float = 0.1) -> Dict:
        """
        Phase 1: Pretrain on FiveK dataset.

        Args:
            fivek_dir: Path to FiveK dataset
            resolution: Training resolution
            batch_size: Batch size
            epochs: Number of epochs
            lr: Learning rate
            num_workers: DataLoader workers
            criterion: Loss function (default: L1)
            val_split: Validation split ratio

        Returns:
            Training history dict
        """
        print("=" * 70)
        print("PHASE 1: PRETRAINING ON FIVEK")
        print("=" * 70)

        # Create dataset
        full_dataset = FiveKDataset(
            fivek_dir,
            resolution=resolution,
            augment=True
        )

        # Split into train/val
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"FiveK Train: {len(train_dataset)}")
        print(f"FiveK Val: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Scheduler: Cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Loss
        if criterion is None:
            criterion = nn.L1Loss()

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_loss = 0
            for src, tar in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                src, tar = src.to(self.device), tar.to(self.device)

                optimizer.zero_grad()
                out = self.model(src)
                loss = criterion(out, tar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, tar in val_loader:
                    src, tar = src.to(self.device), tar.to(self.device)
                    out = self.model(src)
                    val_loss += criterion(out, tar).item()
            val_loss /= len(val_loader)

            scheduler.step()

            # Log
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.output_dir / 'checkpoint_fivek_best.pt')

            print(f"Epoch {epoch:3d}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f} "
                  f"{'(best)' if is_best else ''}")

            self.history['phase1_train_losses'].append(train_loss)
            self.history['phase1_val_losses'].append(val_loss)

        # Save pretrained checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, self.output_dir / 'checkpoint_fivek_pretrained.pt')

        print(f"\nPhase 1 complete! Best val loss: {best_val_loss:.4f}")

        return {'best_val_loss': best_val_loss}

    def finetune_on_real_estate(self,
                                 train_jsonl: str,
                                 val_jsonl: str,
                                 base_dir: str = '.',
                                 resolution: int = 512,
                                 batch_size: int = 8,
                                 epochs: int = 100,
                                 lr: float = 2e-5,  # Lower LR for fine-tuning!
                                 num_workers: int = 8,
                                 criterion: nn.Module = None,
                                 early_stopping_patience: int = 15,
                                 gradual_unfreeze: bool = False) -> Dict:
        """
        Phase 2: Fine-tune on real estate dataset.

        Key differences from pretraining:
        - Lower learning rate (10-20% of pretrain LR)
        - Early stopping to prevent overfitting
        - Optional gradual unfreezing

        Args:
            train_jsonl: Path to training JSONL
            val_jsonl: Path to validation JSONL
            base_dir: Base directory for image paths
            resolution: Training resolution
            batch_size: Batch size (often smaller than pretrain)
            epochs: Maximum epochs
            lr: Learning rate (should be lower than pretrain!)
            num_workers: DataLoader workers
            criterion: Loss function
            early_stopping_patience: Patience for early stopping
            gradual_unfreeze: Whether to gradually unfreeze layers

        Returns:
            Training history dict
        """
        print("\n" + "=" * 70)
        print("PHASE 2: FINE-TUNING ON REAL ESTATE")
        print("=" * 70)

        # Import real estate dataset
        from train_darkir_simple import RealEstateDataset

        train_dataset = RealEstateDataset(train_jsonl, base_dir, resolution)
        val_dataset = RealEstateDataset(val_jsonl, base_dir, resolution)

        print(f"Real Estate Train: {len(train_dataset)}")
        print(f"Real Estate Val: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # Optimizer with lower LR
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Loss
        if criterion is None:
            criterion = nn.L1Loss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            # Optional: Gradual unfreezing
            if gradual_unfreeze:
                self._gradual_unfreeze(epoch, epochs)

            # Train
            self.model.train()
            train_loss = 0
            for src, tar in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                src, tar = src.to(self.device), tar.to(self.device)

                optimizer.zero_grad()
                out = self.model(src)
                loss = criterion(out, tar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, tar in val_loader:
                    src, tar = src.to(self.device), tar.to(self.device)
                    out = self.model(src)
                    val_loss += criterion(out, tar).item()
            val_loss /= len(val_loader)

            scheduler.step()

            # Check for improvement
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.output_dir / 'checkpoint_best.pt')
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch:3d}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f} "
                  f"{'(best)' if is_best else ''}")

            self.history['phase2_train_losses'].append(train_loss)
            self.history['phase2_val_losses'].append(val_loss)

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break

        # Save final history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nPhase 2 complete! Best val loss: {best_val_loss:.4f}")

        return {'best_val_loss': best_val_loss, 'best_epoch': best_epoch}

    def _gradual_unfreeze(self, current_epoch: int, total_epochs: int):
        """
        Gradually unfreeze model layers during fine-tuning.

        Strategy:
        - Epoch 1-10%: Only last layer trainable
        - Epoch 10-30%: Last 2 decoder blocks
        - Epoch 30-50%: All decoder blocks
        - Epoch 50%+: Full model
        """
        progress = current_epoch / total_epochs

        # Get all parameters
        params = list(self.model.named_parameters())

        if progress < 0.1:
            # Only last layer
            for name, param in params[:-2]:
                param.requires_grad = False
            for name, param in params[-2:]:
                param.requires_grad = True
        elif progress < 0.3:
            # Last 25% of parameters
            cutoff = int(len(params) * 0.75)
            for name, param in params[:cutoff]:
                param.requires_grad = False
            for name, param in params[cutoff:]:
                param.requires_grad = True
        elif progress < 0.5:
            # Last 50% of parameters
            cutoff = int(len(params) * 0.5)
            for name, param in params[:cutoff]:
                param.requires_grad = False
            for name, param in params[cutoff:]:
                param.requires_grad = True
        else:
            # Full model
            for param in self.model.parameters():
                param.requires_grad = True


def prepare_fivek_dataset(
    raw_dir: str,
    output_dir: str,
    expert: str = 'C',
    resize: Optional[int] = None
) -> str:
    """
    Prepare FiveK dataset from raw downloaded files.

    Expected input structure (from MIT-Adobe FiveK download):
    raw_dir/
        fivek_dataset/
            raw_photos/         # Original DNG/RAW files
            expert{A,B,C,D,E}/  # Expert retouched TIFFs

    Output structure:
    output_dir/
        input/      # Converted input images
        expertC/    # Expert C images
        pairs.jsonl # Pair mappings

    Args:
        raw_dir: Path to raw FiveK download
        output_dir: Output directory
        expert: Which expert to use
        resize: Optional resize (e.g., 1024 for faster training)

    Returns:
        Path to output directory
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_out = output_path / 'input'
    expert_out = output_path / f'expert{expert}'
    input_out.mkdir(exist_ok=True)
    expert_out.mkdir(exist_ok=True)

    # Find input images
    raw_photos_dir = raw_path / 'raw_photos'
    expert_dir = raw_path / f'expert{expert}'

    if not raw_photos_dir.exists():
        # Try alternative structure
        raw_photos_dir = raw_path / 'input'

    if not raw_photos_dir.exists():
        raise ValueError(f"Could not find input directory in {raw_dir}")

    pairs = []

    # Process each image
    input_files = list(raw_photos_dir.glob('*.jpg')) + list(raw_photos_dir.glob('*.png'))

    for input_file in tqdm(input_files, desc="Processing FiveK"):
        # Find corresponding expert file
        stem = input_file.stem
        expert_file = None

        for ext in ['.tif', '.tiff', '.jpg', '.png']:
            candidate = expert_dir / f"{stem}{ext}"
            if candidate.exists():
                expert_file = candidate
                break

        if expert_file is None:
            continue

        # Load and optionally resize
        input_img = cv2.imread(str(input_file))
        expert_img = cv2.imread(str(expert_file))

        if input_img is None or expert_img is None:
            continue

        if resize:
            h, w = input_img.shape[:2]
            scale = resize / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            input_img = cv2.resize(input_img, (new_w, new_h))
            expert_img = cv2.resize(expert_img, (new_w, new_h))

        # Save
        output_input = input_out / f"{stem}.jpg"
        output_expert = expert_out / f"{stem}.jpg"

        cv2.imwrite(str(output_input), input_img)
        cv2.imwrite(str(output_expert), expert_img)

        pairs.append({
            'input': f"input/{stem}.jpg",
            'target': f"expert{expert}/{stem}.jpg"
        })

    # Save pairs
    with open(output_path / 'pairs.jsonl', 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Prepared {len(pairs)} pairs in {output_dir}")

    return str(output_path)


# Convenience function for quick transfer learning setup
def run_transfer_learning(
    model: nn.Module,
    fivek_dir: str,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str = 'outputs_transfer',
    resolution: int = 512,
    fivek_epochs: int = 30,
    finetune_epochs: int = 100,
    fivek_lr: float = 2e-4,
    finetune_lr: float = 2e-5,
    batch_size: int = 16,
    device: str = 'cuda',
    criterion: nn.Module = None
) -> Dict:
    """
    Run complete two-phase transfer learning.

    Args:
        model: Model to train
        fivek_dir: Path to FiveK dataset
        train_jsonl: Real estate training data
        val_jsonl: Real estate validation data
        output_dir: Output directory
        resolution: Training resolution
        fivek_epochs: Epochs for FiveK pretraining
        finetune_epochs: Max epochs for fine-tuning
        fivek_lr: Learning rate for FiveK
        finetune_lr: Learning rate for fine-tuning (should be lower!)
        batch_size: Batch size
        device: Device
        criterion: Loss function

    Returns:
        Training results
    """
    trainer = TransferLearningTrainer(model, device, output_dir)

    # Phase 1: Pretrain on FiveK
    phase1_results = trainer.pretrain_on_fivek(
        fivek_dir=fivek_dir,
        resolution=resolution,
        batch_size=batch_size,
        epochs=fivek_epochs,
        lr=fivek_lr,
        criterion=criterion
    )

    # Phase 2: Fine-tune on Real Estate
    phase2_results = trainer.finetune_on_real_estate(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        resolution=resolution,
        batch_size=batch_size // 2,  # Smaller batch for fine-tuning
        epochs=finetune_epochs,
        lr=finetune_lr,
        criterion=criterion,
        early_stopping_patience=15
    )

    return {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'output_dir': output_dir
    }


if __name__ == '__main__':
    print("FiveK Transfer Learning module loaded successfully")
    print("\nUsage:")
    print("  from fivek_transfer import run_transfer_learning")
    print("  results = run_transfer_learning(model, fivek_dir, train_jsonl, val_jsonl)")
