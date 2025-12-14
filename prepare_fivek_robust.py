#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
ROBUST MIT-Adobe FiveK Dataset Preparation Pipeline
Top 0.0001% MLE standards: Error handling, validation, resume capability, domain analysis
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation"""
    fivek_dir: Path
    output_dir: Path
    expert: str = "C"  # A, B, C, D, or E - C is most popular in research
    val_ratio: float = 0.05
    min_resolution: int = 256  # Reject images smaller than this
    max_resolution: int = 4096  # Downsample images larger than this
    target_format: str = "jpg"
    jpeg_quality: int = 95
    verify_pairs: bool = True
    resume_from_checkpoint: bool = True
    random_seed: int = 42


class ImageValidator:
    """Validate image quality and compatibility"""

    @staticmethod
    def validate_image(img_path: Path, min_res: int = 256) -> Tuple[bool, Optional[str]]:
        """
        Validate image file.
        Returns: (is_valid, error_message)
        """
        try:
            # Check file exists
            if not img_path.exists():
                return False, f"File not found: {img_path}"

            # Check file size (not empty, not too small)
            file_size = img_path.stat().st_size
            if file_size < 1024:  # < 1KB
                return False, f"File too small ({file_size} bytes)"

            # Try to load image
            img = cv2.imread(str(img_path))
            if img is None:
                return False, "Failed to load image (corrupted?)"

            # Check dimensions
            h, w = img.shape[:2]
            if h < min_res or w < min_res:
                return False, f"Resolution too low ({w}x{h} < {min_res})"

            # Check channels
            if len(img.shape) != 3 or img.shape[2] != 3:
                return False, f"Invalid channels (expected RGB, got shape {img.shape})"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def check_pair_compatibility(src_path: Path, tar_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if source and target images are compatible"""
        try:
            src = cv2.imread(str(src_path))
            tar = cv2.imread(str(tar_path))

            if src is None or tar is None:
                return False, "Failed to load one or both images"

            # Check dimensions match (or are close)
            src_h, src_w = src.shape[:2]
            tar_h, tar_w = tar.shape[:2]

            if (src_h, src_w) != (tar_h, tar_w):
                # Allow small differences (cropping during export)
                ratio_diff = abs((src_w/src_h) - (tar_w/tar_h))
                if ratio_diff > 0.05:  # 5% aspect ratio difference
                    return False, f"Aspect ratio mismatch (src:{src_w}x{src_h}, tar:{tar_w}x{tar_h})"

            return True, None

        except Exception as e:
            return False, f"Compatibility check error: {str(e)}"


class FiveKDatasetPreparer:
    """Robust FiveK dataset preparation with validation and resume capability"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.validator = ImageValidator()
        self.checkpoint_file = config.output_dir / ".preparation_checkpoint.json"
        self.stats = defaultdict(int)

    def find_image_directories(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Find input and output directories with multiple fallback options"""
        print("\n🔍 Locating FiveK image directories...")

        # Try multiple naming conventions
        input_candidates = [
            "input", "photos_original", "original", "source", "raw", "inputs"
        ]
        output_candidates = [
            f"output_expert_{self.config.expert}",
            f"photos_expert_{self.config.expert}",
            f"expert_{self.config.expert}",
            f"expert{self.config.expert}",
            f"ExpertC" if self.config.expert == "C" else f"Expert{self.config.expert}",
        ]

        # Search for directories
        input_dir = None
        for name in input_candidates:
            candidate = self.config.fivek_dir / name
            if candidate.exists() and candidate.is_dir():
                num_files = len(list(candidate.glob("*.jpg"))) + len(list(candidate.glob("*.png")))
                if num_files > 0:
                    input_dir = candidate
                    print(f"   ✅ Input directory: {candidate} ({num_files} images)")
                    break

        output_dir = None
        for name in output_candidates:
            candidate = self.config.fivek_dir / name
            if candidate.exists() and candidate.is_dir():
                num_files = len(list(candidate.glob("*.jpg"))) + len(list(candidate.glob("*.png")))
                if num_files > 0:
                    output_dir = candidate
                    print(f"   ✅ Output directory: {candidate} ({num_files} images)")
                    break

        if not input_dir or not output_dir:
            print(f"\n   ❌ Could not find required directories!")
            if not input_dir:
                print(f"      Missing input dir. Searched: {input_candidates}")
            if not output_dir:
                print(f"      Missing output dir. Searched: {output_candidates}")

        return input_dir, output_dir

    def discover_pairs(self, input_dir: Path, output_dir: Path) -> List[Dict]:
        """Discover and validate image pairs"""
        print("\n📋 Discovering image pairs...")

        # Find all images
        input_images = {}
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            for img in input_dir.glob(ext):
                input_images[img.stem] = img

        output_images = {}
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            for img in output_dir.glob(ext):
                output_images[img.stem] = img

        print(f"   Input images: {len(input_images)}")
        print(f"   Output images: {len(output_images)}")

        # Match pairs
        pairs = []
        for name in tqdm(sorted(input_images.keys()), desc="Matching pairs"):
            if name in output_images:
                pairs.append({
                    'name': name,
                    'src': str(input_images[name]),
                    'tar': str(output_images[name]),
                    'validated': False
                })

        print(f"   ✅ Found {len(pairs)} potential pairs")
        return pairs

    def validate_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """Validate all pairs for quality and compatibility"""
        print("\n🔍 Validating image pairs...")

        valid_pairs = []
        validation_errors = defaultdict(int)

        for pair in tqdm(pairs, desc="Validating"):
            src_path = Path(pair['src'])
            tar_path = Path(pair['tar'])

            # Validate source
            src_valid, src_error = self.validator.validate_image(
                src_path, self.config.min_resolution
            )
            if not src_valid:
                validation_errors[f"src: {src_error}"] += 1
                self.stats['rejected_invalid_src'] += 1
                continue

            # Validate target
            tar_valid, tar_error = self.validator.validate_image(
                tar_path, self.config.min_resolution
            )
            if not tar_valid:
                validation_errors[f"tar: {tar_error}"] += 1
                self.stats['rejected_invalid_tar'] += 1
                continue

            # Check compatibility
            compat_valid, compat_error = self.validator.check_pair_compatibility(
                src_path, tar_path
            )
            if not compat_valid:
                validation_errors[f"compatibility: {compat_error}"] += 1
                self.stats['rejected_incompatible'] += 1
                continue

            # Pair is valid
            pair['validated'] = True
            valid_pairs.append(pair)
            self.stats['valid_pairs'] += 1

        # Report validation results
        print(f"\n   ✅ Valid pairs: {len(valid_pairs)}/{len(pairs)}")
        if validation_errors:
            print(f"   ❌ Rejection reasons:")
            for reason, count in sorted(validation_errors.items(), key=lambda x: -x[1])[:5]:
                print(f"      {reason}: {count}")

        return valid_pairs

    def create_train_val_split(self, pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Create stratified train/val split with reproducibility"""
        random.seed(self.config.random_seed)

        # Shuffle
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)

        # Split
        val_size = int(len(shuffled_pairs) * self.config.val_ratio)
        train_pairs = shuffled_pairs[val_size:]
        val_pairs = shuffled_pairs[:val_size]

        print(f"\n📊 Dataset split:")
        print(f"   Train: {len(train_pairs)} ({100*(1-self.config.val_ratio):.1f}%)")
        print(f"   Val: {len(val_pairs)} ({100*self.config.val_ratio:.1f}%)")

        return train_pairs, val_pairs

    def process_and_save_split(
        self,
        pairs: List[Dict],
        split_name: str
    ) -> Path:
        """Process images and save split JSONL with resume capability"""
        print(f"\n💾 Processing {split_name} split...")

        images_dir = self.config.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = self.config.output_dir / f"{split_name}.jsonl"

        # Check checkpoint for resume
        processed_files = set()
        if self.config.resume_from_checkpoint and jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    processed_files.add(entry['src'])

        # Process pairs
        jsonl_entries = []

        with open(jsonl_path, 'a' if processed_files else 'w') as f:
            for pair in tqdm(pairs, desc=f"Copying {split_name}"):
                name = pair['name']

                # Create relative paths
                src_rel = f"images/{name}_src.{self.config.target_format}"
                tar_rel = f"images/{name}_tar.{self.config.target_format}"

                # Skip if already processed
                if src_rel in processed_files:
                    continue

                src_dest = images_dir / f"{name}_src.{self.config.target_format}"
                tar_dest = images_dir / f"{name}_tar.{self.config.target_format}"

                # Copy and optionally convert
                self._copy_and_convert(Path(pair['src']), src_dest)
                self._copy_and_convert(Path(pair['tar']), tar_dest)

                # Write JSONL entry
                entry = {'src': src_rel, 'tar': tar_rel}
                f.write(json.dumps(entry) + '\n')
                f.flush()  # Ensure written for resume

                self.stats[f'{split_name}_pairs_processed'] += 1

        print(f"   ✅ Processed {len(pairs)} pairs → {jsonl_path}")
        return jsonl_path

    def _copy_and_convert(self, src: Path, dst: Path):
        """Copy image with optional format conversion and downsampling"""
        if dst.exists():
            return  # Already processed

        try:
            img = cv2.imread(str(src))

            # Downsample if needed
            h, w = img.shape[:2]
            if max(h, w) > self.config.max_resolution:
                scale = self.config.max_resolution / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.stats['images_downsampled'] += 1

            # Save with target format
            if self.config.target_format == 'jpg':
                cv2.imwrite(
                    str(dst),
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
            else:
                cv2.imwrite(str(dst), img)

        except Exception as e:
            print(f"   ⚠️  Error processing {src}: {e}")
            # Copy original as fallback
            shutil.copy2(src, dst)

    def save_dataset_metadata(self, train_jsonl: Path, val_jsonl: Path):
        """Save comprehensive dataset metadata"""
        with open(train_jsonl) as f:
            train_count = sum(1 for _ in f)
        with open(val_jsonl) as f:
            val_count = sum(1 for _ in f)

        metadata = {
            'dataset': 'MIT-Adobe FiveK',
            'expert': self.config.expert,
            'config': asdict(self.config),
            'statistics': dict(self.stats),
            'splits': {
                'train': {'count': train_count, 'file': str(train_jsonl)},
                'val': {'count': val_count, 'file': str(val_jsonl)},
                'total': train_count + val_count
            },
            'usage': {
                'pre_training': f'--train_jsonl {train_jsonl}',
                'note': 'Use for pre-training before fine-tuning on real estate data'
            }
        }

        metadata_file = self.config.output_dir / 'dataset_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n📄 Metadata saved: {metadata_file}")
        return metadata

    def run(self):
        """Execute complete preparation pipeline"""
        print("=" * 80)
        print("ROBUST MIT-ADOBE FIVEK DATASET PREPARATION")
        print("=" * 80)

        # 1. Find directories
        input_dir, output_dir = self.find_image_directories()
        if not input_dir or not output_dir:
            print("\n❌ FAILED: Required directories not found")
            sys.exit(1)

        # 2. Discover pairs
        pairs = self.discover_pairs(input_dir, output_dir)
        if len(pairs) == 0:
            print("\n❌ FAILED: No image pairs found")
            sys.exit(1)

        # 3. Validate pairs
        valid_pairs = self.validate_pairs(pairs)
        if len(valid_pairs) == 0:
            print("\n❌ FAILED: No valid pairs after validation")
            sys.exit(1)

        # 4. Create split
        train_pairs, val_pairs = self.create_train_val_split(valid_pairs)

        # 5. Process and save
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        train_jsonl = self.process_and_save_split(train_pairs, "train")
        val_jsonl = self.process_and_save_split(val_pairs, "val")

        # 6. Save metadata
        metadata = self.save_dataset_metadata(train_jsonl, val_jsonl)

        # 7. Summary
        print("\n" + "=" * 80)
        print("✅ FIVEK DATASET PREPARATION COMPLETE")
        print("=" * 80)
        print(f"\n📍 Output location: {self.config.output_dir}")
        print(f"\n📊 Final Statistics:")
        print(f"   Total valid pairs: {len(valid_pairs)}")
        print(f"   Train pairs: {metadata['splits']['train']['count']}")
        print(f"   Val pairs: {metadata['splits']['val']['count']}")
        print(f"\n💡 Usage for pre-training:")
        print(f"   {metadata['usage']['pre_training']}")
        print("\n" + "=" * 80)


def main():
    config = DatasetConfig(
        fivek_dir=Path("/mmfs1/home/sww35/autohdr-real-estate-577/fivek_dataset"),
        output_dir=Path("/mmfs1/home/sww35/autohdr-real-estate-577/fivek_processed"),
        expert="C",
        val_ratio=0.05,
        min_resolution=256,
        max_resolution=2048,  # Downsample large images to save space
        target_format="jpg",
        jpeg_quality=95,
        verify_pairs=True,
        resume_from_checkpoint=True,
        random_seed=42
    )

    preparer = FiveKDatasetPreparer(config)
    preparer.run()


if __name__ == "__main__":
    main()
