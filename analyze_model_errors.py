#!/usr/bin/env python3
"""
Deep Analysis: Model Output vs Ground Truth
============================================

Analyzes WHERE and HOW the model fails compared to ground truth.
This informs what losses/architectures to use.

Analysis includes:
1. Spatial error maps (where are errors?)
2. Brightness-stratified errors (highlights vs shadows vs midtones)
3. Color channel analysis (R vs G vs B errors)
4. Saturation/Hue error analysis
5. Edge vs flat region errors
6. Statistical summary of failure patterns
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class ErrorAnalysis:
    """Complete error analysis for one image."""
    image_id: str

    # Overall metrics
    psnr: float
    ssim: float
    mae: float  # Mean absolute error

    # Brightness-stratified errors (KEY INSIGHT)
    mae_shadows: float      # Brightness < 0.3
    mae_midtones: float     # Brightness 0.3-0.7
    mae_highlights: float   # Brightness > 0.7

    # What percentage of total error comes from each region
    error_pct_shadows: float
    error_pct_midtones: float
    error_pct_highlights: float

    # Color channel errors
    mae_red: float
    mae_green: float
    mae_blue: float

    # Saturation errors
    mae_saturation: float
    saturation_error_in_highlights: float

    # Hue errors (important for color accuracy)
    mae_hue: float
    hue_error_in_highlights: float

    # Edge vs flat regions
    mae_edges: float
    mae_flat: float

    # Region coverage
    pct_shadows: float
    pct_midtones: float
    pct_highlights: float


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255**2 / mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return structural_similarity(pred, target, channel_axis=2)


def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB to HSV."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    h = hsv[:,:,0] / 180.0  # Normalize to [0, 1]
    s = hsv[:,:,1] / 255.0
    v = hsv[:,:,2] / 255.0
    return h, s, v


def detect_edges(image: np.ndarray, threshold: float = 30) -> np.ndarray:
    """Detect edges using Sobel."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return magnitude > threshold


def analyze_single_image(
    output: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    image_id: str
) -> ErrorAnalysis:
    """
    Comprehensive error analysis for a single image.
    """
    # Ensure same size
    if output.shape != target.shape:
        output = cv2.resize(output, (target.shape[1], target.shape[0]))
    if source.shape != target.shape:
        source = cv2.resize(source, (target.shape[1], target.shape[0]))

    # Basic metrics
    psnr = compute_psnr(output, target)
    ssim = compute_ssim(output, target)

    # Convert to float [0, 1]
    out_f = output.astype(float) / 255.0
    tar_f = target.astype(float) / 255.0
    src_f = source.astype(float) / 255.0

    # Absolute error per pixel
    error = np.abs(out_f - tar_f)
    mae = error.mean()

    # Brightness from SOURCE (this is what we use for detection)
    src_brightness = 0.299 * src_f[:,:,0] + 0.587 * src_f[:,:,1] + 0.114 * src_f[:,:,2]

    # Brightness masks
    shadow_mask = src_brightness < 0.3
    midtone_mask = (src_brightness >= 0.3) & (src_brightness <= 0.7)
    highlight_mask = src_brightness > 0.7

    # Coverage percentages
    total_pixels = src_brightness.size
    pct_shadows = shadow_mask.sum() / total_pixels
    pct_midtones = midtone_mask.sum() / total_pixels
    pct_highlights = highlight_mask.sum() / total_pixels

    # Mean error per region
    error_magnitude = error.mean(axis=2)  # Average across RGB

    mae_shadows = error_magnitude[shadow_mask].mean() if shadow_mask.sum() > 0 else 0
    mae_midtones = error_magnitude[midtone_mask].mean() if midtone_mask.sum() > 0 else 0
    mae_highlights = error_magnitude[highlight_mask].mean() if highlight_mask.sum() > 0 else 0

    # What percentage of TOTAL error comes from each region
    total_error = error_magnitude.sum()
    error_pct_shadows = error_magnitude[shadow_mask].sum() / total_error if total_error > 0 else 0
    error_pct_midtones = error_magnitude[midtone_mask].sum() / total_error if total_error > 0 else 0
    error_pct_highlights = error_magnitude[highlight_mask].sum() / total_error if total_error > 0 else 0

    # Per-channel errors
    mae_red = error[:,:,0].mean()
    mae_green = error[:,:,1].mean()
    mae_blue = error[:,:,2].mean()

    # HSV analysis
    out_h, out_s, out_v = rgb_to_hsv(output)
    tar_h, tar_s, tar_v = rgb_to_hsv(target)

    # Saturation error
    sat_error = np.abs(out_s - tar_s)
    mae_saturation = sat_error.mean()
    saturation_error_in_highlights = sat_error[highlight_mask].mean() if highlight_mask.sum() > 0 else 0

    # Hue error (circular)
    hue_diff = np.abs(out_h - tar_h)
    hue_diff = np.minimum(hue_diff, 1 - hue_diff)  # Circular distance
    mae_hue = hue_diff.mean()
    hue_error_in_highlights = hue_diff[highlight_mask].mean() if highlight_mask.sum() > 0 else 0

    # Edge vs flat region analysis
    edge_mask = detect_edges(target)
    flat_mask = ~edge_mask

    mae_edges = error_magnitude[edge_mask].mean() if edge_mask.sum() > 0 else 0
    mae_flat = error_magnitude[flat_mask].mean() if flat_mask.sum() > 0 else 0

    return ErrorAnalysis(
        image_id=image_id,
        psnr=psnr,
        ssim=ssim,
        mae=mae,
        mae_shadows=mae_shadows,
        mae_midtones=mae_midtones,
        mae_highlights=mae_highlights,
        error_pct_shadows=error_pct_shadows,
        error_pct_midtones=error_pct_midtones,
        error_pct_highlights=error_pct_highlights,
        mae_red=mae_red,
        mae_green=mae_green,
        mae_blue=mae_blue,
        mae_saturation=mae_saturation,
        saturation_error_in_highlights=saturation_error_in_highlights,
        mae_hue=mae_hue,
        hue_error_in_highlights=hue_error_in_highlights,
        mae_edges=mae_edges,
        mae_flat=mae_flat,
        pct_shadows=pct_shadows,
        pct_midtones=pct_midtones,
        pct_highlights=pct_highlights,
    )


def create_error_visualization(
    output: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    save_path: Path
):
    """Create detailed error visualization."""
    # Ensure same size
    if output.shape != target.shape:
        output = cv2.resize(output, (target.shape[1], target.shape[0]))
    if source.shape != target.shape:
        source = cv2.resize(source, (target.shape[1], target.shape[0]))

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 1: Images
    axes[0, 0].imshow(source)
    axes[0, 0].set_title('Source', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(output)
    axes[0, 1].set_title('Model Output', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(target)
    axes[0, 2].set_title('Ground Truth', fontsize=12)
    axes[0, 2].axis('off')

    # Error map
    error = np.abs(output.astype(float) - target.astype(float))
    error_magnitude = error.mean(axis=2)
    im = axes[0, 3].imshow(error_magnitude, cmap='hot', vmin=0, vmax=50)
    axes[0, 3].set_title('Error Map (brighter=more error)', fontsize=12)
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Row 2: Brightness regions and their errors
    src_f = source.astype(float) / 255.0
    src_brightness = 0.299 * src_f[:,:,0] + 0.587 * src_f[:,:,1] + 0.114 * src_f[:,:,2]

    # Highlight mask
    highlight_mask = src_brightness > 0.7
    axes[1, 0].imshow(highlight_mask, cmap='Reds')
    axes[1, 0].set_title(f'Highlight Regions ({100*highlight_mask.mean():.1f}%)', fontsize=12)
    axes[1, 0].axis('off')

    # Error in highlights only
    error_in_highlights = error_magnitude * highlight_mask
    im = axes[1, 1].imshow(error_in_highlights, cmap='hot', vmin=0, vmax=50)
    axes[1, 1].set_title('Error in Highlights', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Shadow mask
    shadow_mask = src_brightness < 0.3
    axes[1, 2].imshow(shadow_mask, cmap='Blues')
    axes[1, 2].set_title(f'Shadow Regions ({100*shadow_mask.mean():.1f}%)', fontsize=12)
    axes[1, 2].axis('off')

    # Error in shadows only
    error_in_shadows = error_magnitude * shadow_mask
    im = axes[1, 3].imshow(error_in_shadows, cmap='hot', vmin=0, vmax=50)
    axes[1, 3].set_title('Error in Shadows', fontsize=12)
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)

    # Row 3: Color analysis
    # Per-channel errors
    axes[2, 0].imshow(error[:,:,0], cmap='Reds', vmin=0, vmax=50)
    axes[2, 0].set_title(f'Red Error (MAE={error[:,:,0].mean():.1f})', fontsize=12)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(error[:,:,1], cmap='Greens', vmin=0, vmax=50)
    axes[2, 1].set_title(f'Green Error (MAE={error[:,:,1].mean():.1f})', fontsize=12)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(error[:,:,2], cmap='Blues', vmin=0, vmax=50)
    axes[2, 2].set_title(f'Blue Error (MAE={error[:,:,2].mean():.1f})', fontsize=12)
    axes[2, 2].axis('off')

    # Saturation error
    out_hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV).astype(float)
    tar_hsv = cv2.cvtColor(target, cv2.COLOR_RGB2HSV).astype(float)
    sat_error = np.abs(out_hsv[:,:,1] - tar_hsv[:,:,1])
    im = axes[2, 3].imshow(sat_error, cmap='magma', vmin=0, vmax=50)
    axes[2, 3].set_title(f'Saturation Error (MAE={sat_error.mean():.1f})', fontsize=12)
    axes[2, 3].axis('off')
    plt.colorbar(im, ax=axes[2, 3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_folder(
    folder: Path,
    output_suffix: str = '_output',
    source_suffix: str = '_src',
    target_suffix: str = '_tar'
) -> List[ErrorAnalysis]:
    """Analyze all images in a folder."""
    results = []

    output_files = sorted(folder.glob(f'*{output_suffix}.jpg'))

    for output_path in output_files:
        stem = output_path.stem
        img_id = stem.replace(output_suffix, '')

        source_path = folder / f'{img_id}{source_suffix}.jpg'
        target_path = folder / f'{img_id}{target_suffix}.jpg'

        # Try alternate naming
        if not source_path.exists():
            source_path = folder / f'{img_id}_input.jpg'
        if not target_path.exists():
            target_path = folder / f'{img_id}_target.jpg'

        if not source_path.exists() or not target_path.exists():
            print(f"  Skipping {img_id}: missing source or target")
            continue

        output = np.array(Image.open(output_path).convert('RGB'))
        source = np.array(Image.open(source_path).convert('RGB'))
        target = np.array(Image.open(target_path).convert('RGB'))

        analysis = analyze_single_image(output, target, source, img_id)
        results.append(analysis)

        print(f"  {img_id}: PSNR={analysis.psnr:.2f}, "
              f"Highlight MAE={analysis.mae_highlights:.4f} ({100*analysis.error_pct_highlights:.1f}% of error), "
              f"Sat Error in Highlights={analysis.saturation_error_in_highlights:.4f}")

    return results


def print_summary(results: List[ErrorAnalysis], name: str):
    """Print summary statistics."""
    if not results:
        return

    print(f"\n{'='*70}")
    print(f"SUMMARY: {name}")
    print(f"{'='*70}")

    # Aggregate
    avg = lambda key: np.mean([getattr(r, key) for r in results])

    print(f"\nOverall Metrics:")
    print(f"  Avg PSNR: {avg('psnr'):.2f} dB")
    print(f"  Avg SSIM: {avg('ssim'):.4f}")
    print(f"  Avg MAE: {avg('mae'):.4f}")

    print(f"\n*** BRIGHTNESS-STRATIFIED ERRORS (KEY INSIGHT) ***")
    print(f"  Shadows (brightness < 0.3):   MAE = {avg('mae_shadows'):.4f}  |  {100*avg('error_pct_shadows'):.1f}% of total error")
    print(f"  Midtones (0.3-0.7):           MAE = {avg('mae_midtones'):.4f}  |  {100*avg('error_pct_midtones'):.1f}% of total error")
    print(f"  Highlights (> 0.7):           MAE = {avg('mae_highlights'):.4f}  |  {100*avg('error_pct_highlights'):.1f}% of total error")
    print(f"\n  Coverage: Shadows={100*avg('pct_shadows'):.1f}%, Midtones={100*avg('pct_midtones'):.1f}%, Highlights={100*avg('pct_highlights'):.1f}%")

    # Calculate error density (error per unit area)
    highlight_error_density = avg('mae_highlights') / (avg('pct_highlights') + 0.001)
    midtone_error_density = avg('mae_midtones') / (avg('pct_midtones') + 0.001)
    shadow_error_density = avg('mae_shadows') / (avg('pct_shadows') + 0.001)

    print(f"\n  Error DENSITY (normalized by area):")
    print(f"    Shadows:    {shadow_error_density:.4f}")
    print(f"    Midtones:   {midtone_error_density:.4f}")
    print(f"    Highlights: {highlight_error_density:.4f}")

    if highlight_error_density > midtone_error_density:
        ratio = highlight_error_density / midtone_error_density
        print(f"\n  >>> HIGHLIGHTS have {ratio:.1f}x MORE error density than midtones! <<<")

    print(f"\nColor Channel Errors:")
    print(f"  Red:   {avg('mae_red'):.4f}")
    print(f"  Green: {avg('mae_green'):.4f}")
    print(f"  Blue:  {avg('mae_blue'):.4f}")

    print(f"\nSaturation/Hue Errors:")
    print(f"  Overall Saturation MAE: {avg('mae_saturation'):.4f}")
    print(f"  Saturation Error in HIGHLIGHTS: {avg('saturation_error_in_highlights'):.4f}")
    print(f"  Overall Hue MAE: {avg('mae_hue'):.4f}")
    print(f"  Hue Error in HIGHLIGHTS: {avg('hue_error_in_highlights'):.4f}")

    print(f"\nEdge vs Flat Regions:")
    print(f"  Edge MAE: {avg('mae_edges'):.4f}")
    print(f"  Flat MAE: {avg('mae_flat'):.4f}")

    # Key recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS BASED ON ANALYSIS:")
    print(f"{'='*70}")

    if avg('mae_highlights') > avg('mae_midtones') * 1.2:
        print("1. [HIGH PRIORITY] Highlights have significantly higher error.")
        print("   -> Use highlight-weighted loss (3-5x weight on bright regions)")

    if avg('saturation_error_in_highlights') > avg('mae_saturation') * 1.5:
        print("2. [HIGH PRIORITY] Saturation error is concentrated in highlights.")
        print("   -> Add explicit saturation loss for highlight regions")

    if avg('hue_error_in_highlights') > avg('mae_hue') * 1.3:
        print("3. [MEDIUM PRIORITY] Hue errors higher in highlights.")
        print("   -> Add hue-matching loss for bright regions")

    max_channel = max([('Red', avg('mae_red')), ('Green', avg('mae_green')), ('Blue', avg('mae_blue'))], key=lambda x: x[1])
    print(f"4. [INFO] {max_channel[0]} channel has highest error ({max_channel[1]:.4f})")

    if avg('mae_edges') > avg('mae_flat') * 1.3:
        print("5. [MEDIUM PRIORITY] Edge regions have higher error.")
        print("   -> Keep gradient/edge preservation loss")


def main():
    print("="*70)
    print("Model Error Analysis: Output vs Ground Truth")
    print("="*70)

    # Install dependencies if needed
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-image', '-q'])

    test_dir = Path('test')
    analysis_dir = Path('analysis_results')
    analysis_dir.mkdir(exist_ok=True)

    all_results = {}

    # Analyze Restormer 896
    print("\n" + "="*70)
    print("Analyzing: Restormer 896")
    print("="*70)

    restormer_dir = test_dir / 'restormer_896'
    if restormer_dir.exists():
        results = analyze_folder(restormer_dir, '_output', '_src', '_tar')
        all_results['restormer_896'] = results
        print_summary(results, 'Restormer 896')

        # Create visualizations
        vis_dir = analysis_dir / 'restormer_896'
        vis_dir.mkdir(exist_ok=True)

        for output_path in sorted(restormer_dir.glob('*_output.jpg'))[:3]:  # First 3
            img_id = output_path.stem.replace('_output', '')
            output = np.array(Image.open(output_path).convert('RGB'))
            source = np.array(Image.open(restormer_dir / f'{img_id}_src.jpg').convert('RGB'))
            target = np.array(Image.open(restormer_dir / f'{img_id}_tar.jpg').convert('RGB'))
            create_error_visualization(output, target, source, vis_dir / f'{img_id}_error_analysis.png')
            print(f"  Created visualization: {vis_dir / f'{img_id}_error_analysis.png'}")

    # Analyze DarkIR
    print("\n" + "="*70)
    print("Analyzing: DarkIR Baseline")
    print("="*70)

    darkir_dir = test_dir / 'darkir_baseline'
    if darkir_dir.exists():
        results = analyze_folder(darkir_dir, '_output', '_input', '_target')
        all_results['darkir_baseline'] = results
        print_summary(results, 'DarkIR Baseline')

        # Create visualizations
        vis_dir = analysis_dir / 'darkir_baseline'
        vis_dir.mkdir(exist_ok=True)

        for output_path in sorted(darkir_dir.glob('*_output.jpg'))[:3]:
            img_id = output_path.stem.replace('_output', '')
            output = np.array(Image.open(output_path).convert('RGB'))
            source_path = darkir_dir / f'{img_id}_input.jpg'
            target_path = darkir_dir / f'{img_id}_target.jpg'
            if source_path.exists() and target_path.exists():
                source = np.array(Image.open(source_path).convert('RGB'))
                target = np.array(Image.open(target_path).convert('RGB'))
                create_error_visualization(output, target, source, vis_dir / f'{img_id}_error_analysis.png')
                print(f"  Created visualization: {vis_dir / f'{img_id}_error_analysis.png'}")

    # Save all results
    results_json = {}
    for name, results in all_results.items():
        results_json[name] = {
            'per_image': [asdict(r) for r in results],
            'summary': {
                'avg_psnr': np.mean([r.psnr for r in results]),
                'avg_ssim': np.mean([r.ssim for r in results]),
                'avg_mae_highlights': np.mean([r.mae_highlights for r in results]),
                'avg_error_pct_highlights': np.mean([r.error_pct_highlights for r in results]),
                'avg_saturation_error_highlights': np.mean([r.saturation_error_in_highlights for r in results]),
            }
        }

    with open(analysis_dir / 'error_analysis.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {analysis_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
