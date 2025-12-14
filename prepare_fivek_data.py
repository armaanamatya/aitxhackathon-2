#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
Prepare MIT-Adobe FiveK dataset for pre-training.
Converts FiveK format to our training format (src/tar pairs).
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# Paths
FIVEK_DIR = Path("/mmfs1/home/sww35/autohdr-real-estate-577/fivek_dataset")
OUTPUT_DIR = Path("/mmfs1/home/sww35/autohdr-real-estate-577/fivek_processed")

# FiveK has expert retouches A, B, C, D, E
# Expert C is most commonly used in research
EXPERT = "C"


def find_image_pairs(fivek_dir):
    """
    Find input/output image pairs from FiveK dataset.

    Expected structure after download:
    fivek_dataset/
        input/  (or photos_original/)
            image_001.jpg
            image_002.jpg
            ...
        output_expert_C/  (or photos_expert_C/)
            image_001.jpg
            image_002.jpg
            ...
    """
    print("🔍 Searching for FiveK image pairs...")

    # Try different possible directory names
    input_dirs = [
        fivek_dir / "input",
        fivek_dir / "photos_original",
        fivek_dir / "original",
        fivek_dir / "source",
    ]

    output_dirs = [
        fivek_dir / f"output_expert_{EXPERT}",
        fivek_dir / f"photos_expert_{EXPERT}",
        fivek_dir / f"expert_{EXPERT}",
        fivek_dir / "expertC",
    ]

    # Find existing directories
    input_dir = None
    for d in input_dirs:
        if d.exists():
            input_dir = d
            print(f"   Found input dir: {d}")
            break

    output_dir = None
    for d in output_dirs:
        if d.exists():
            output_dir = d
            print(f"   Found output dir: {d}")
            break

    if not input_dir or not output_dir:
        print("❌ Could not find FiveK image directories!")
        print(f"   Searched input: {[str(d) for d in input_dirs]}")
        print(f"   Searched output: {[str(d) for d in output_dirs]}")
        return []

    # Find matching pairs
    input_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    output_files = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))

    print(f"   Input images: {len(input_files)}")
    print(f"   Output images: {len(output_files)}")

    # Match by filename
    pairs = []
    input_dict = {f.stem: f for f in input_files}
    output_dict = {f.stem: f for f in output_files}

    for name in input_dict:
        if name in output_dict:
            pairs.append({
                'src': str(input_dict[name]),
                'tar': str(output_dict[name]),
                'name': name
            })

    print(f"   ✅ Found {len(pairs)} matching pairs")
    return pairs


def create_train_val_split(pairs, val_ratio=0.05):
    """Split into train/val (95/5 split for large dataset)"""
    random.seed(42)
    random.shuffle(pairs)

    val_size = int(len(pairs) * val_ratio)
    train_pairs = pairs[val_size:]
    val_pairs = pairs[:val_size]

    return train_pairs, val_pairs


def copy_images_and_create_jsonl(pairs, split_name, output_dir):
    """
    Copy images to output directory and create JSONL manifest.

    Creates:
    - fivek_processed/images/  (all images)
    - fivek_processed/train.jsonl
    - fivek_processed/val.jsonl
    """
    print(f"\n📋 Processing {split_name} split ({len(pairs)} pairs)...")

    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{split_name}.jsonl"

    # Process pairs
    jsonl_entries = []

    for pair in tqdm(pairs, desc=f"Copying {split_name}"):
        src_path = Path(pair['src'])
        tar_path = Path(pair['tar'])

        # Create unique filenames
        name = pair['name']
        src_dest = images_dir / f"{name}_src{src_path.suffix}"
        tar_dest = images_dir / f"{name}_tar{tar_path.suffix}"

        # Copy images
        if not src_dest.exists():
            shutil.copy2(src_path, src_dest)
        if not tar_dest.exists():
            shutil.copy2(tar_path, tar_dest)

        # Create JSONL entry with relative paths
        jsonl_entries.append({
            'src': f"images/{src_dest.name}",
            'tar': f"images/{tar_dest.name}"
        })

    # Write JSONL
    with open(jsonl_path, 'w') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"   ✅ Saved {len(jsonl_entries)} entries to {jsonl_path}")
    return jsonl_path


def create_dataset_stats(output_dir, train_jsonl, val_jsonl):
    """Create dataset statistics file"""

    with open(train_jsonl) as f:
        train_count = sum(1 for _ in f)

    with open(val_jsonl) as f:
        val_count = sum(1 for _ in f)

    stats = {
        'dataset': 'MIT-Adobe FiveK',
        'expert': EXPERT,
        'train_pairs': train_count,
        'val_pairs': val_count,
        'total_pairs': train_count + val_count,
        'split_ratio': f'{100*(1-val_count/(train_count+val_count)):.1f}% train / {100*val_count/(train_count+val_count):.1f}% val',
        'description': 'Professional photo retouching dataset for pre-training',
    }

    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total pairs: {stats['total_pairs']}")
    print(f"   Train: {stats['train_pairs']}")
    print(f"   Val: {stats['val_pairs']}")
    print(f"   Split: {stats['split_ratio']}")

    return stats


def main():
    print("=" * 70)
    print("PREPARING MIT-ADOBE FIVEK DATASET")
    print("=" * 70)

    # Find pairs
    pairs = find_image_pairs(FIVEK_DIR)

    if len(pairs) == 0:
        print("\n❌ No image pairs found!")
        print("\nPlease ensure FiveK dataset is downloaded and extracted to:")
        print(f"   {FIVEK_DIR}")
        print("\nExpected structure:")
        print("   fivek_dataset/")
        print("       input/  or  photos_original/")
        print("       output_expert_C/  or  photos_expert_C/")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split train/val
    train_pairs, val_pairs = create_train_val_split(pairs, val_ratio=0.05)

    # Process and copy images
    train_jsonl = copy_images_and_create_jsonl(train_pairs, "train", OUTPUT_DIR)
    val_jsonl = copy_images_and_create_jsonl(val_pairs, "val", OUTPUT_DIR)

    # Create stats
    stats = create_dataset_stats(OUTPUT_DIR, train_jsonl, val_jsonl)

    print("\n" + "=" * 70)
    print("✅ FIVEK DATASET READY!")
    print("=" * 70)
    print(f"\nProcessed dataset location: {OUTPUT_DIR}")
    print(f"\nTo use for pre-training:")
    print(f"   --train_jsonl {OUTPUT_DIR}/train.jsonl")
    print(f"   --val_split 0  (validation already split)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
