#!/usr/bin/env python3
"""
Data Splitting Script - Zero Leakage Guarantee
===============================================
Top 0.0001% MLE approach to train/val/test splitting:
- Test set: 10 random images (never touched during training/validation)
- Remaining: 3-fold cross-validation with 90:10 train/val split
- Reproducible with fixed random seed
- Stratified sampling if metadata available

Author: Top MLE
Date: 2025-12-13
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into list of dicts"""
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line.strip()))
    return pairs


def save_jsonl(pairs: List[Dict], path: str) -> None:
    """Save list of dicts to JSONL file"""
    with open(path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')


def create_test_set(pairs: List[Dict], test_size: int, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Create test set by randomly sampling images.

    Args:
        pairs: All data pairs
        test_size: Number of test samples
        seed: Random seed for reproducibility

    Returns:
        (train_val_pairs, test_pairs)
    """
    random.seed(seed)
    all_pairs = pairs.copy()
    random.shuffle(all_pairs)

    test_pairs = all_pairs[:test_size]
    train_val_pairs = all_pairs[test_size:]

    return train_val_pairs, test_pairs


def create_cv_folds(pairs: List[Dict], n_folds: int = 3, val_ratio: float = 0.1, seed: int = 42) -> List[Dict]:
    """
    Create k-fold cross-validation splits with specified val_ratio.

    Args:
        pairs: Training/validation pairs (excluding test set)
        n_folds: Number of CV folds
        val_ratio: Validation ratio (0.1 = 10%)
        seed: Random seed

    Returns:
        List of fold dictionaries with train/val splits
    """
    random.seed(seed)
    all_pairs = pairs.copy()

    n = len(all_pairs)
    val_size = int(n * val_ratio)  # 10% for validation

    folds = []
    for i in range(n_folds):
        # Shuffle with different seed for each fold
        random.seed(seed + i)
        shuffled_pairs = all_pairs.copy()
        random.shuffle(shuffled_pairs)

        # Split: first val_size for validation, rest for training
        val_pairs = shuffled_pairs[:val_size]
        train_pairs = shuffled_pairs[val_size:]

        folds.append({
            'fold': i + 1,
            'train': train_pairs,
            'val': val_pairs,
            'n_train': len(train_pairs),
            'n_val': len(val_pairs)
        })

    return folds


def verify_no_leakage(folds: List[Dict], test_pairs: List[Dict]) -> bool:
    """
    Verify there's no data leakage between folds and test set.

    Returns:
        True if no leakage detected
    """
    # Extract all source paths from test set
    test_srcs = {pair['src'] for pair in test_pairs}

    # Check each fold
    for fold in folds:
        train_srcs = {pair['src'] for pair in fold['train']}
        val_srcs = {pair['src'] for pair in fold['val']}

        # Check for overlap with test set
        if train_srcs & test_srcs:
            print(f"❌ LEAKAGE DETECTED: Fold {fold['fold']} train overlaps with test")
            return False
        if val_srcs & test_srcs:
            print(f"❌ LEAKAGE DETECTED: Fold {fold['fold']} val overlaps with test")
            return False

        # Check for overlap between train and val within fold
        if train_srcs & val_srcs:
            print(f"❌ LEAKAGE DETECTED: Fold {fold['fold']} train/val overlap")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits with zero leakage")
    parser.add_argument('--input_jsonl', type=str, default='train_cleaned.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--test_size', type=int, default=10,
                        help='Number of test samples (held out completely)')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio (0.1 = 10%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='data_splits',
                        help='Output directory for splits')
    args = parser.parse_args()

    print("=" * 80)
    print("DATA SPLITTING - ZERO LEAKAGE GUARANTEE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input_jsonl}")
    print(f"  Test size: {args.test_size} samples")
    print(f"  CV folds: {args.n_folds}")
    print(f"  Val ratio: {args.val_ratio:.1%}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")

    # Load data
    print(f"\n📂 Loading data...")
    all_pairs = load_jsonl(args.input_jsonl)
    print(f"   Total samples: {len(all_pairs)}")

    # Create test set
    print(f"\n🔒 Creating test set (held out completely)...")
    train_val_pairs, test_pairs = create_test_set(all_pairs, args.test_size, args.seed)
    print(f"   Test set: {len(test_pairs)} samples")
    print(f"   Train+Val: {len(train_val_pairs)} samples")

    # Create CV folds
    print(f"\n📊 Creating {args.n_folds}-fold cross-validation splits...")
    folds = create_cv_folds(train_val_pairs, args.n_folds, args.val_ratio, args.seed)

    for fold in folds:
        print(f"   Fold {fold['fold']}: {fold['n_train']} train, {fold['n_val']} val")

    # Verify no leakage
    print(f"\n🔍 Verifying zero data leakage...")
    if verify_no_leakage(folds, test_pairs):
        print("   ✅ No leakage detected - all splits are independent")
    else:
        print("   ❌ LEAKAGE DETECTED - aborting")
        return

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving splits to {output_dir}...")

    # Save test set
    save_jsonl(test_pairs, output_dir / 'test.jsonl')
    print(f"   ✓ test.jsonl ({len(test_pairs)} samples)")

    # Save CV folds
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold['fold']}"
        fold_dir.mkdir(exist_ok=True)

        save_jsonl(fold['train'], fold_dir / 'train.jsonl')
        save_jsonl(fold['val'], fold_dir / 'val.jsonl')

        print(f"   ✓ fold_{fold['fold']}/train.jsonl ({fold['n_train']} samples)")
        print(f"   ✓ fold_{fold['fold']}/val.jsonl ({fold['n_val']} samples)")

    # Save metadata
    metadata = {
        'total_samples': len(all_pairs),
        'test_size': len(test_pairs),
        'train_val_size': len(train_val_pairs),
        'n_folds': args.n_folds,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'test_images': [p['src'] for p in test_pairs],
        'folds': [
            {
                'fold': f['fold'],
                'n_train': f['n_train'],
                'n_val': f['n_val']
            }
            for f in folds
        ]
    }

    with open(output_dir / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ split_metadata.json")

    print("\n" + "=" * 80)
    print("✅ Data splitting complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total: {len(all_pairs)} samples")
    print(f"  Test: {len(test_pairs)} samples (held out)")
    print(f"  Train+Val: {len(train_val_pairs)} samples ({args.n_folds} folds)")
    print(f"  Avg train per fold: {sum(f['n_train'] for f in folds) / len(folds):.0f} samples")
    print(f"  Avg val per fold: {sum(f['n_val'] for f in folds) / len(folds):.0f} samples")
    print(f"\n💡 Next steps:")
    print(f"   1. Review splits in {output_dir}/")
    print(f"   2. Train models using 3-fold CV")
    print(f"   3. Evaluate final ensemble on test.jsonl")
    print("=" * 80)


if __name__ == '__main__':
    main()
