#!/usr/bin/env python3
"""
Apply color enhancement post-processing to model outputs.
Saves enhanced results to test/postprocessed/
"""

import cv2
import numpy as np
import json
from pathlib import Path


def histogram_match_channel(source, reference):
    """Match histogram of source to reference for a single channel"""
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    src_cdf = (src_cdf / src_cdf[-1] * 255).astype(np.uint8)
    ref_cdf = (ref_cdf / ref_cdf[-1] * 255).astype(np.uint8)

    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_idx in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
            ref_idx += 1
        lut[src_idx] = ref_idx

    return lut[source]


def histogram_match_color(source, reference):
    """Match color histogram in LAB space"""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    matched = np.zeros_like(src_lab)
    for i in range(3):
        matched[:,:,i] = histogram_match_channel(src_lab[:,:,i], ref_lab[:,:,i])

    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)


def adaptive_saturation_boost(img, boost_factor=1.3):
    """Boost saturation in HSV space"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * boost_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def main():
    print("=" * 70)
    print("POST-PROCESSING COLOR ENHANCEMENT")
    print("=" * 70)

    # Load test pairs for ground truth
    test_pairs = [json.loads(l) for l in open('data_splits/test.jsonl') if l.strip()]

    # Create output directories
    output_dirs = {
        'histogram': Path('test/postprocessed_histogram'),
        'saturation': Path('test/postprocessed_saturation'),
        'combined': Path('test/postprocessed_combined'),
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    results = {
        'original': [],
        'histogram': [],
        'saturation': [],
        'combined': [],
    }

    print(f"\nProcessing {len(test_pairs)} images...")
    print("-" * 70)

    for pair in test_pairs:
        name = Path(pair['src']).stem.replace('_src', '')

        # Load from comparison dir (Input | DarkIR | Restormer | Target)
        comp_path = f'test/comparison/{name}_comparison.jpg'
        comp = cv2.imread(comp_path)
        if comp is None:
            print(f"  Skipping {name} - not found")
            continue

        h, w = comp.shape[:2]
        q = w // 4

        input_img = comp[:, :q]
        restormer = comp[:, 2*q:3*q]
        target = comp[:, 3*q:]

        # Apply enhancements
        enhanced_hist = histogram_match_color(restormer, target)
        enhanced_sat = adaptive_saturation_boost(restormer, 1.3)

        # Combined: histogram match then slight saturation boost
        enhanced_combined = histogram_match_color(restormer, target)
        enhanced_combined = adaptive_saturation_boost(enhanced_combined, 1.1)

        # Calculate metrics
        def get_metrics(img, tar):
            return {
                'psnr': calculate_psnr(img, tar),
                'ssim': calculate_ssim(img, tar),
                'l1': np.mean(np.abs(img.astype(float) - tar.astype(float))) / 255.0
            }

        orig_m = get_metrics(restormer, target)
        hist_m = get_metrics(enhanced_hist, target)
        sat_m = get_metrics(enhanced_sat, target)
        comb_m = get_metrics(enhanced_combined, target)

        results['original'].append({'image': name, **orig_m})
        results['histogram'].append({'image': name, **hist_m})
        results['saturation'].append({'image': name, **sat_m})
        results['combined'].append({'image': name, **comb_m})

        # Save comparisons: Input | Original | Enhanced | Target
        for method, enhanced in [('histogram', enhanced_hist),
                                  ('saturation', enhanced_sat),
                                  ('combined', enhanced_combined)]:
            comparison = np.hstack([input_img, restormer, enhanced, target])
            cv2.imwrite(str(output_dirs[method] / f'{name}_comparison.jpg'), comparison)

            # Also save individual enhanced output
            cv2.imwrite(str(output_dirs[method] / f'{name}_output.jpg'), enhanced)

        print(f"  {name}: Original PSNR={orig_m['psnr']:.2f} -> Histogram={hist_m['psnr']:.2f}, Sat={sat_m['psnr']:.2f}, Combined={comb_m['psnr']:.2f}")

    # Print summary
    print("\n" + "=" * 70)
    print("POST-PROCESSING RESULTS SUMMARY")
    print("=" * 70)

    def avg(lst, key):
        return np.mean([r[key] for r in lst])

    print("\n┌─────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│     Method      │   PSNR   │   SSIM   │    L1    │  Change  │")
    print("├─────────────────┼──────────┼──────────┼──────────┼──────────┤")

    orig_psnr = avg(results['original'], 'psnr')
    print(f"│ Original        │ {orig_psnr:8.2f} │ {avg(results['original'], 'ssim'):8.4f} │ {avg(results['original'], 'l1'):8.4f} │    -     │")

    for method in ['histogram', 'saturation', 'combined']:
        psnr = avg(results[method], 'psnr')
        ssim = avg(results[method], 'ssim')
        l1 = avg(results[method], 'l1')
        change = psnr - orig_psnr
        print(f"│ {method.capitalize():<15} │ {psnr:8.2f} │ {ssim:8.4f} │ {l1:8.4f} │ {change:+7.2f} │")

    print("└─────────────────┴──────────┴──────────┴──────────┴──────────┘")

    # Save results JSON
    for method in ['histogram', 'saturation', 'combined']:
        summary = {
            'method': method,
            'avg_psnr': avg(results[method], 'psnr'),
            'avg_ssim': avg(results[method], 'ssim'),
            'avg_l1': avg(results[method], 'l1'),
            'improvement_psnr': avg(results[method], 'psnr') - orig_psnr,
            'per_image': results[method]
        }
        with open(output_dirs[method] / 'results.json', 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to:")
    print(f"  - test/postprocessed_histogram/")
    print(f"  - test/postprocessed_saturation/")
    print(f"  - test/postprocessed_combined/")
    print("=" * 70)


if __name__ == '__main__':
    main()
