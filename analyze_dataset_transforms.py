#!/usr/bin/env /cm/local/apps/python39/bin/python3
"""
Analyze transformations between source and target images in the dataset.
This will help us understand what pre/post processing steps would be most effective.
"""

import numpy as np
from PIL import Image, ImageStat
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def analyze_image_pair(src_path, tar_path):
    """Analyze differences between source and target image."""
    src = Image.open(src_path).convert('RGB')
    tar = Image.open(tar_path).convert('RGB')

    # Convert to numpy arrays
    src_np = np.array(src).astype(np.float32) / 255.0
    tar_np = np.array(tar).astype(np.float32) / 255.0

    # Resize to same size if needed
    if src.size != tar.size:
        tar = tar.resize(src.size, Image.BICUBIC)
        tar_np = np.array(tar).astype(np.float32) / 255.0

    results = {}

    # 1. Brightness analysis (luminance)
    src_lum = 0.299 * src_np[:,:,0] + 0.587 * src_np[:,:,1] + 0.114 * src_np[:,:,2]
    tar_lum = 0.299 * tar_np[:,:,0] + 0.587 * tar_np[:,:,1] + 0.114 * tar_np[:,:,2]

    results['brightness_src_mean'] = float(np.mean(src_lum))
    results['brightness_tar_mean'] = float(np.mean(tar_lum))
    results['brightness_delta'] = float(np.mean(tar_lum) - np.mean(src_lum))
    results['brightness_ratio'] = float(np.mean(tar_lum) / (np.mean(src_lum) + 1e-8))

    # 2. Contrast analysis (std of luminance)
    results['contrast_src'] = float(np.std(src_lum))
    results['contrast_tar'] = float(np.std(tar_lum))
    results['contrast_delta'] = float(np.std(tar_lum) - np.std(src_lum))
    results['contrast_ratio'] = float(np.std(tar_lum) / (np.std(src_lum) + 1e-8))

    # 3. Saturation analysis
    src_hsv = np.array(src.convert('HSV')).astype(np.float32) / 255.0
    tar_hsv = np.array(tar.convert('HSV')).astype(np.float32) / 255.0

    results['saturation_src_mean'] = float(np.mean(src_hsv[:,:,1]))
    results['saturation_tar_mean'] = float(np.mean(tar_hsv[:,:,1]))
    results['saturation_delta'] = float(np.mean(tar_hsv[:,:,1]) - np.mean(src_hsv[:,:,1]))
    results['saturation_ratio'] = float(np.mean(tar_hsv[:,:,1]) / (np.mean(src_hsv[:,:,1]) + 1e-8))

    # 4. Color temperature (blue/red ratio)
    src_color_temp = np.mean(src_np[:,:,2]) / (np.mean(src_np[:,:,0]) + 1e-8)
    tar_color_temp = np.mean(tar_np[:,:,2]) / (np.mean(tar_np[:,:,0]) + 1e-8)

    results['color_temp_src'] = float(src_color_temp)
    results['color_temp_tar'] = float(tar_color_temp)
    results['color_temp_delta'] = float(tar_color_temp - src_color_temp)

    # 5. Gamma estimation (approximate)
    # Compare mid-tones to estimate gamma curve
    src_midtones = src_lum[(src_lum > 0.2) & (src_lum < 0.8)]
    tar_midtones = tar_lum[(src_lum > 0.2) & (src_lum < 0.8)]

    if len(src_midtones) > 0 and len(tar_midtones) > 0:
        # Estimate gamma: tar = src^gamma
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_est = np.log(np.mean(tar_midtones) + 1e-8) / np.log(np.mean(src_midtones) + 1e-8)
            results['estimated_gamma'] = float(gamma_est) if np.isfinite(gamma_est) else 1.0
    else:
        results['estimated_gamma'] = 1.0

    # 6. Per-channel analysis
    for i, channel in enumerate(['R', 'G', 'B']):
        results[f'{channel}_src_mean'] = float(np.mean(src_np[:,:,i]))
        results[f'{channel}_tar_mean'] = float(np.mean(tar_np[:,:,i]))
        results[f'{channel}_delta'] = float(np.mean(tar_np[:,:,i]) - np.mean(src_np[:,:,i]))

    # 7. Histogram analysis (percentiles)
    for pct in [5, 25, 50, 75, 95]:
        results[f'lum_src_p{pct}'] = float(np.percentile(src_lum, pct))
        results[f'lum_tar_p{pct}'] = float(np.percentile(tar_lum, pct))

    # 8. Shadow/Highlight changes
    shadows_src = src_lum[src_lum < 0.2]
    shadows_tar = tar_lum[src_lum < 0.2]
    highlights_src = src_lum[src_lum > 0.8]
    highlights_tar = tar_lum[src_lum > 0.8]

    if len(shadows_src) > 0:
        results['shadows_lift'] = float(np.mean(shadows_tar) - np.mean(shadows_src))
    else:
        results['shadows_lift'] = 0.0

    if len(highlights_src) > 0:
        results['highlights_adjust'] = float(np.mean(highlights_tar) - np.mean(highlights_src))
    else:
        results['highlights_adjust'] = 0.0

    return results


def main():
    """Analyze all training pairs and generate summary statistics."""

    # Load training samples
    train_jsonl = Path('train.jsonl')
    if not train_jsonl.exists():
        print("ERROR: train.jsonl not found!")
        return

    print("Loading training samples...")
    samples = []
    with open(train_jsonl) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    print(f"Found {len(samples)} training pairs")

    # Analyze all pairs
    all_results = []
    print("\nAnalyzing transformations...")
    for sample in tqdm(samples[:50]):  # Analyze first 50 for speed
        src_path = Path(sample['src'])
        tar_path = Path(sample['tar'])

        if src_path.exists() and tar_path.exists():
            try:
                results = analyze_image_pair(src_path, tar_path)
                all_results.append(results)
            except Exception as e:
                print(f"Error analyzing {src_path}: {e}")

    print(f"\nSuccessfully analyzed {len(all_results)} image pairs")

    # Compute aggregate statistics
    print("\n" + "="*60)
    print("DATASET TRANSFORMATION ANALYSIS")
    print("="*60)

    # Brightness
    brightness_deltas = [r['brightness_delta'] for r in all_results]
    brightness_ratios = [r['brightness_ratio'] for r in all_results]
    print(f"\n📊 BRIGHTNESS:")
    print(f"  Average delta: {np.mean(brightness_deltas):+.4f} ({np.mean(brightness_deltas)*100:+.1f}%)")
    print(f"  Average ratio: {np.mean(brightness_ratios):.4f}x")
    print(f"  Std dev delta: {np.std(brightness_deltas):.4f}")

    # Contrast
    contrast_deltas = [r['contrast_delta'] for r in all_results]
    contrast_ratios = [r['contrast_ratio'] for r in all_results]
    print(f"\n📈 CONTRAST:")
    print(f"  Average delta: {np.mean(contrast_deltas):+.4f} ({np.mean(contrast_deltas)*100:+.1f}%)")
    print(f"  Average ratio: {np.mean(contrast_ratios):.4f}x")
    print(f"  Std dev delta: {np.std(contrast_deltas):.4f}")

    # Saturation
    sat_deltas = [r['saturation_delta'] for r in all_results]
    sat_ratios = [r['saturation_ratio'] for r in all_results]
    print(f"\n🎨 SATURATION:")
    print(f"  Average delta: {np.mean(sat_deltas):+.4f} ({np.mean(sat_deltas)*100:+.1f}%)")
    print(f"  Average ratio: {np.mean(sat_ratios):.4f}x")
    print(f"  Std dev delta: {np.std(sat_deltas):.4f}")

    # Color temperature
    temp_deltas = [r['color_temp_delta'] for r in all_results]
    print(f"\n🌡️  COLOR TEMPERATURE (Blue/Red):")
    print(f"  Average delta: {np.mean(temp_deltas):+.4f}")
    print(f"  Std dev: {np.std(temp_deltas):.4f}")

    # Gamma
    gammas = [r['estimated_gamma'] for r in all_results]
    print(f"\n⚡ GAMMA CURVE:")
    print(f"  Average estimated gamma: {np.mean(gammas):.3f}")
    print(f"  Std dev: {np.std(gammas):.3f}")
    print(f"  Range: [{np.min(gammas):.3f}, {np.max(gammas):.3f}]")

    # Shadows/Highlights
    shadow_lifts = [r['shadows_lift'] for r in all_results]
    highlight_adjusts = [r['highlights_adjust'] for r in all_results]
    print(f"\n🌓 SHADOWS/HIGHLIGHTS:")
    print(f"  Shadow lift (avg): {np.mean(shadow_lifts):+.4f} ({np.mean(shadow_lifts)*100:+.1f}%)")
    print(f"  Highlight adjust (avg): {np.mean(highlight_adjusts):+.4f} ({np.mean(highlight_adjusts)*100:+.1f}%)")

    # Per-channel
    print(f"\n🔴🟢🔵 PER-CHANNEL CHANGES:")
    for channel in ['R', 'G', 'B']:
        deltas = [r[f'{channel}_delta'] for r in all_results]
        print(f"  {channel} delta (avg): {np.mean(deltas):+.4f} ({np.mean(deltas)*100:+.1f}%)")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDED PRE/POST PROCESSING")
    print("="*60)

    avg_brightness_ratio = np.mean(brightness_ratios)
    avg_contrast_ratio = np.mean(contrast_ratios)
    avg_sat_ratio = np.mean(sat_ratios)
    avg_gamma = np.mean(gammas)
    avg_shadow_lift = np.mean(shadow_lifts)

    print("\n✅ HIGH PRIORITY (Consistent patterns):")

    if abs(avg_brightness_ratio - 1.0) > 0.05:
        print(f"  • Brightness adjustment: {avg_brightness_ratio:.3f}x (Post-process)")

    if abs(avg_contrast_ratio - 1.0) > 0.05:
        print(f"  • Contrast enhancement: {avg_contrast_ratio:.3f}x (Post-process)")

    if abs(avg_sat_ratio - 1.0) > 0.05:
        print(f"  • Saturation boost: {avg_sat_ratio:.3f}x ({(avg_sat_ratio-1)*100:+.1f}%) (Post-process)")

    if abs(avg_gamma - 1.0) > 0.05:
        print(f"  • Gamma correction: γ={avg_gamma:.3f} (Pre-process)")

    if abs(avg_shadow_lift) > 0.02:
        print(f"  • Shadow lift: +{avg_shadow_lift:.3f} ({avg_shadow_lift*100:+.1f}%) (Post-process)")

    print("\n⚠️  MEDIUM PRIORITY (Variable patterns):")
    if np.std(brightness_deltas) > 0.05:
        print(f"  • Adaptive brightness (std: {np.std(brightness_deltas):.3f})")

    if np.std(temp_deltas) > 0.1:
        print(f"  • Adaptive white balance (std: {np.std(temp_deltas):.3f})")

    # Save detailed results
    output_file = Path('dataset_analysis.json')
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'brightness_ratio': float(avg_brightness_ratio),
                'contrast_ratio': float(avg_contrast_ratio),
                'saturation_ratio': float(avg_sat_ratio),
                'gamma': float(avg_gamma),
                'shadow_lift': float(avg_shadow_lift),
                'highlight_adjust': float(np.mean(highlight_adjusts)),
            },
            'per_image_results': all_results
        }, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")

    # Create visualization
    print("\n📊 Creating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Brightness distribution
    axes[0, 0].hist(brightness_deltas, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(brightness_deltas), color='red', linestyle='--', label=f'Mean: {np.mean(brightness_deltas):.3f}')
    axes[0, 0].set_xlabel('Brightness Delta')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Brightness Changes Distribution')
    axes[0, 0].legend()

    # Contrast distribution
    axes[0, 1].hist(contrast_deltas, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(contrast_deltas), color='red', linestyle='--', label=f'Mean: {np.mean(contrast_deltas):.3f}')
    axes[0, 1].set_xlabel('Contrast Delta')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Contrast Changes Distribution')
    axes[0, 1].legend()

    # Saturation distribution
    axes[0, 2].hist(sat_deltas, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(np.mean(sat_deltas), color='red', linestyle='--', label=f'Mean: {np.mean(sat_deltas):.3f}')
    axes[0, 2].set_xlabel('Saturation Delta')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Saturation Changes Distribution')
    axes[0, 2].legend()

    # Gamma distribution
    axes[1, 0].hist(gammas, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(gammas), color='red', linestyle='--', label=f'Mean: {np.mean(gammas):.3f}')
    axes[1, 0].set_xlabel('Estimated Gamma')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Gamma Curve Distribution')
    axes[1, 0].legend()

    # Shadow lift
    axes[1, 1].hist(shadow_lifts, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(shadow_lifts), color='red', linestyle='--', label=f'Mean: {np.mean(shadow_lifts):.3f}')
    axes[1, 1].set_xlabel('Shadow Lift')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Shadow Adjustments Distribution')
    axes[1, 1].legend()

    # Per-channel deltas
    r_deltas = [r['R_delta'] for r in all_results]
    g_deltas = [r['G_delta'] for r in all_results]
    b_deltas = [r['B_delta'] for r in all_results]

    axes[1, 2].hist([r_deltas, g_deltas, b_deltas], bins=30, alpha=0.6,
                    label=['Red', 'Green', 'Blue'], edgecolor='black')
    axes[1, 2].set_xlabel('Channel Delta')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Per-Channel Changes')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: dataset_analysis.png")

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
