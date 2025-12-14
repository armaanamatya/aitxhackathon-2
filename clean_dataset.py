#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
Data cleaning pipeline: Remove outliers and low-quality pairs
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy import stats

DATA_DIR = "/mmfs1/home/sww35/autohdr-real-estate-577"
TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
OUTPUT_JSONL = os.path.join(DATA_DIR, "train_cleaned.jsonl")
ANALYSIS_FILE = os.path.join(DATA_DIR, "analysis_results_v2/comprehensive_analysis.json")


def analyze_pair(src_path, tar_path):
    """Compute quality metrics for a single pair"""
    src = cv2.imread(src_path)
    tar = cv2.imread(tar_path)

    if src is None or tar is None:
        return None

    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

    # Resize for analysis
    src_small = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)
    tar_small = cv2.resize(tar, (512, 512), interpolation=cv2.INTER_AREA)

    src_f = src_small.astype(np.float32) / 255.0
    tar_f = tar_small.astype(np.float32) / 255.0

    src_gray = np.mean(src_f, axis=2)
    tar_gray = np.mean(tar_f, axis=2)

    # Key transformation metrics
    brightness_src = np.mean(src_gray)
    brightness_tar = np.mean(tar_gray)
    brightness_delta = brightness_tar - brightness_src

    shadow_src = np.percentile(src_gray, 5)
    shadow_tar = np.percentile(tar_gray, 5)
    shadow_lift = shadow_tar - shadow_src

    # Gamma estimation
    valid_mask = (src_gray > 0.05) & (src_gray < 0.95)
    if np.sum(valid_mask) > 100:
        try:
            slope, _, r_value, _, _ = stats.linregress(
                np.log(src_gray[valid_mask] + 1e-6),
                np.log(tar_gray[valid_mask] + 1e-6)
            )
            gamma = float(np.clip(slope, 0.3, 3.0))
            gamma_r2 = r_value ** 2
        except:
            gamma = 1.0
            gamma_r2 = 0.0
    else:
        gamma = 1.0
        gamma_r2 = 0.0

    # Blur detection (Laplacian variance)
    src_blur = cv2.Laplacian(cv2.cvtColor(src_small, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    tar_blur = cv2.Laplacian(cv2.cvtColor(tar_small, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

    # Spatial alignment (SSIM)
    alignment_ssim = ssim(src_gray, tar_gray, data_range=1.0)

    return {
        'brightness_delta': brightness_delta,
        'shadow_lift': shadow_lift,
        'gamma': gamma,
        'gamma_r2': gamma_r2,
        'src_blur': src_blur,
        'tar_blur': tar_blur,
        'alignment_ssim': alignment_ssim,
    }


def main():
    print("=" * 70)
    print("DATA CLEANING PIPELINE")
    print("=" * 70)

    # Load dataset statistics
    print("\nLoading comprehensive analysis statistics...")
    with open(ANALYSIS_FILE) as f:
        analysis = json.load(f)

    stats = analysis['aggregated']

    # Define filtering thresholds (mean ± 2 std)
    brightness_mean = stats['brightness_delta']['mean']
    brightness_std = stats['brightness_delta']['std']
    brightness_min = brightness_mean - 2 * brightness_std
    brightness_max = brightness_mean + 2 * brightness_std

    shadow_mean = stats['shadows_lift']['mean']
    shadow_std = stats['shadows_lift']['std']
    shadow_min = shadow_mean - 2 * shadow_std

    gamma_mean = stats['gamma_estimated']['mean']
    gamma_std = stats['gamma_estimated']['std']
    gamma_min = gamma_mean - 2 * gamma_std
    gamma_max = gamma_mean + 2 * gamma_std

    print(f"\n📊 Filtering Thresholds (mean ± 2σ):")
    print(f"   Brightness delta: [{brightness_min:.3f}, {brightness_max:.3f}]")
    print(f"   Shadow lift: [{shadow_min:.3f}, ∞]")
    print(f"   Gamma: [{gamma_min:.3f}, {gamma_max:.3f}]")

    # Load train pairs
    pairs = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line.strip()))

    print(f"\nAnalyzing {len(pairs)} training pairs...")

    # Analyze each pair
    results = []
    for pair in tqdm(pairs, desc="Analyzing"):
        src_path = os.path.join(DATA_DIR, pair['src'])
        tar_path = os.path.join(DATA_DIR, pair['tar'])

        metrics = analyze_pair(src_path, tar_path)

        if metrics is None:
            continue

        results.append({
            'pair': pair,
            'metrics': metrics
        })

    # Apply filters
    print("\n🔍 Applying filters...")

    filtered = []
    reject_reasons = {
        'bad_brightness': 0,
        'bad_shadow': 0,
        'bad_gamma': 0,
        'low_gamma_fit': 0,
        'too_blurry': 0,
        'poor_alignment': 0,
    }

    for item in results:
        m = item['metrics']
        reject = False

        # Filter 1: Brightness outliers
        if m['brightness_delta'] < brightness_min or m['brightness_delta'] > brightness_max:
            reject_reasons['bad_brightness'] += 1
            reject = True

        # Filter 2: Shadow darkening (should be lifting)
        if m['shadow_lift'] < shadow_min:
            reject_reasons['bad_shadow'] += 1
            reject = True

        # Filter 3: Gamma outliers
        if m['gamma'] < gamma_min or m['gamma'] > gamma_max:
            reject_reasons['bad_gamma'] += 1
            reject = True

        # Filter 4: Poor gamma fit (transformation inconsistent)
        if m['gamma_r2'] < 0.5:
            reject_reasons['low_gamma_fit'] += 1
            reject = True

        # Filter 5: Blurry source images
        if m['src_blur'] < 50:  # Very blurry
            reject_reasons['too_blurry'] += 1
            reject = True

        # Filter 6: Poor spatial alignment
        if m['alignment_ssim'] < 0.5:  # Different composition
            reject_reasons['poor_alignment'] += 1
            reject = True

        if not reject:
            filtered.append(item['pair'])

    # Report results
    print(f"\n📊 FILTERING RESULTS:")
    print(f"   Original pairs: {len(results)}")
    print(f"   Cleaned pairs: {len(filtered)}")
    print(f"   Removed: {len(results) - len(filtered)} ({(1 - len(filtered)/len(results))*100:.1f}%)")
    print(f"\n❌ Rejection reasons:")
    for reason, count in reject_reasons.items():
        if count > 0:
            print(f"   {reason}: {count}")

    # Save cleaned dataset
    print(f"\n💾 Saving cleaned dataset to: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, 'w') as f:
        for pair in filtered:
            f.write(json.dumps(pair) + '\n')

    print("\n✅ DONE!")
    print(f"   Use {OUTPUT_JSONL} for training")


if __name__ == "__main__":
    main()
