#!/usr/bin/env python3
"""
Compare All Models - Aggregated Results
========================================

Collects and compares results from all trained models.
Outputs a summary table for easy comparison.
"""

import json
from pathlib import Path
from typing import Dict, List
import sys

def load_comparison(output_dir: Path) -> Dict:
    """Load comparison summary from a model output directory."""
    summary_file = output_dir / 'comparison_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)

    # Try test_results.json as fallback
    test_file = output_dir / 'test_results.json'
    if test_file.exists():
        with open(test_file) as f:
            data = json.load(f)
            return {
                'model': data.get('model', output_dir.name),
                'psnr': data.get('avg_psnr', 0),
                'ssim': data.get('avg_ssim', 0),
                'highlight_psnr': data.get('avg_highlight_psnr', 0),
                'params_M': data.get('params', 0) / 1e6 if data.get('params') else 0
            }

    return None


def main():
    # Find all output directories
    output_dirs = [
        'outputs_restormer_896',
        'outputs_restormer_384',
        'outputs_nafnet_comprehensive',
        'outputs_retinexformer_comprehensive',
        'outputs_retinexformer_full',
        'outputs_highlight_aware',
        'outputs_darkir',
        'outputs_full_light_aug',
        'outputs_full_standard_aug',
        'outputs_full_normalize_exp',
    ]

    results = []

    for dir_name in output_dirs:
        output_dir = Path(dir_name)
        if output_dir.exists():
            summary = load_comparison(output_dir)
            if summary:
                summary['dir'] = dir_name
                results.append(summary)

    # Also check test/ directory for existing results
    test_dir = Path('test')
    if test_dir.exists():
        for subdir in test_dir.iterdir():
            if subdir.is_dir() and not subdir.name.endswith('_postprocessed'):
                # Try to find metrics
                metrics_file = subdir / 'metrics.json'
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        data = json.load(f)
                    if 'average' in data:
                        avg = data['average']
                        results.append({
                            'model': subdir.name,
                            'dir': str(subdir),
                            'psnr': avg.get('psnr_before', avg.get('avg_psnr', 0)),
                            'ssim': avg.get('ssim_before', avg.get('avg_ssim', 0)),
                            'highlight_psnr': avg.get('highlight_psnr', 0),
                            'params_M': 0
                        })

    if not results:
        print("No results found. Train some models first!")
        print("\nAvailable training scripts:")
        print("  sbatch train_nafnet_comprehensive.sh")
        print("  sbatch train_retinexformer_comprehensive.sh")
        return

    # Sort by PSNR
    results.sort(key=lambda x: x.get('psnr', 0), reverse=True)

    # Print table
    print("\n" + "=" * 90)
    print("MODEL COMPARISON - Real Estate HDR Enhancement")
    print("=" * 90)
    print(f"{'Model':<35} {'PSNR':>8} {'SSIM':>8} {'HL_PSNR':>10} {'Params':>10}")
    print("-" * 90)

    for r in results:
        model = r.get('model', r.get('dir', 'Unknown'))[:34]
        psnr = r.get('psnr', 0)
        ssim = r.get('ssim', 0)
        hl_psnr = r.get('highlight_psnr', 0)
        params = r.get('params_M', 0)

        print(f"{model:<35} {psnr:>8.2f} {ssim:>8.4f} {hl_psnr:>10.2f} {params:>9.2f}M")

    print("-" * 90)

    # Best model
    best = results[0]
    print(f"\nBest model: {best.get('model', best.get('dir'))} (PSNR: {best.get('psnr', 0):.2f} dB)")

    # Key insights
    print("\n" + "=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    print("- HL_PSNR (Highlight PSNR) measures performance on window/bright regions")
    print("- Based on error analysis: highlights have 46x more error density")
    print("- Good HL_PSNR indicates proper window/highlight color recovery")
    print("=" * 90)


if __name__ == '__main__':
    main()
