#!/usr/bin/env python3
"""
Create Proper Data Splits with Held-Out Test Set

Split strategy:
1. TEST SET: First 10 images from train.jsonl (completely held out)
2. Remaining 567 images split 90:10 into TRAIN:VAL
3. No data leakage - test set never seen during training/validation

This ensures:
- Fair evaluation on truly unseen data
- Consistent splits across all experiments
- Reproducible results
"""

import json
import os
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)

def main():
    # Paths
    source_jsonl = 'train.jsonl'
    output_dir = Path('data_splits/proper_split')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all samples
    print("Loading samples from train.jsonl...")
    samples = []
    with open(source_jsonl, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"Total samples: {len(samples)}")

    # Split 1: First 10 images -> TEST SET (held out)
    test_samples = samples[:10]
    remaining_samples = samples[10:]

    print(f"\n=== SPLIT CONFIGURATION ===")
    print(f"Test set (held out): {len(test_samples)} images")
    print(f"Remaining for train/val: {len(remaining_samples)} images")

    # Shuffle remaining samples
    random.shuffle(remaining_samples)

    # Split 2: 90:10 train:val from remaining
    val_size = int(len(remaining_samples) * 0.10)
    train_size = len(remaining_samples) - val_size

    train_samples = remaining_samples[:train_size]
    val_samples = remaining_samples[train_size:]

    print(f"\nTrain set: {len(train_samples)} images (90%)")
    print(f"Val set: {len(val_samples)} images (10%)")

    # Verify no overlap
    test_paths = set(s['src'] for s in test_samples)
    train_paths = set(s['src'] for s in train_samples)
    val_paths = set(s['src'] for s in val_samples)

    assert len(test_paths & train_paths) == 0, "Data leakage: test/train overlap!"
    assert len(test_paths & val_paths) == 0, "Data leakage: test/val overlap!"
    assert len(train_paths & val_paths) == 0, "Data leakage: train/val overlap!"
    print("\n✓ No data leakage detected")

    # Save splits
    def save_jsonl(samples, path):
        with open(path, 'w') as f:
            for s in samples:
                # Standardize keys to 'input'/'target' format
                out = {
                    'input': s.get('src', s.get('input')),
                    'target': s.get('tar', s.get('target'))
                }
                f.write(json.dumps(out) + '\n')

    # Save files
    save_jsonl(test_samples, output_dir / 'test.jsonl')
    save_jsonl(train_samples, output_dir / 'train.jsonl')
    save_jsonl(val_samples, output_dir / 'val.jsonl')

    print(f"\n=== FILES SAVED ===")
    print(f"  {output_dir}/test.jsonl  ({len(test_samples)} samples)")
    print(f"  {output_dir}/train.jsonl ({len(train_samples)} samples)")
    print(f"  {output_dir}/val.jsonl   ({len(val_samples)} samples)")

    # Save metadata
    metadata = {
        'description': 'Proper train/val/test split with held-out test set',
        'split_strategy': {
            'test': 'First 10 images from original train.jsonl (held out)',
            'train_val': '90:10 split of remaining 567 images',
            'seed': 42
        },
        'counts': {
            'test': len(test_samples),
            'train': len(train_samples),
            'val': len(val_samples),
            'total': len(samples)
        },
        'test_images': [s.get('src', s.get('input')) for s in test_samples],
        'no_data_leakage': True
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  {output_dir}/metadata.json")

    # Print test set images for reference
    print(f"\n=== HELD-OUT TEST SET (10 images) ===")
    for i, s in enumerate(test_samples):
        print(f"  {i+1}. {s.get('src', s.get('input'))}")

    print("\n✓ Data splits created successfully!")
    print("\nUSAGE:")
    print("  --train_jsonl data_splits/proper_split/train.jsonl")
    print("  --val_jsonl data_splits/proper_split/val.jsonl")
    print("  --test_jsonl data_splits/proper_split/test.jsonl  (for final evaluation only)")


if __name__ == '__main__':
    main()
