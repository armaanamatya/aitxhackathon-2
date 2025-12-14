#!/usr/bin/env python3
"""
Apply Window Post-Processing to Test Images
=============================================

Applies optimized window enhancement to Restormer and DarkIR outputs.
Uses adaptive parameters tuned for real estate HDR enhancement.

Output Structure:
    test/
        restormer_896_postprocessed/
            {id}_postprocessed.jpg
            {id}_comparison.jpg  (src | output | postprocessed | target)
        darkir_baseline_postprocessed/
            {id}_postprocessed.jpg
            {id}_comparison.jpg
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PostProcessConfig:
    """Post-processing configuration optimized for real estate HDR."""
    # Window detection
    brightness_threshold: float = 0.60  # Lower = detect more windows
    saturation_threshold: float = 0.18  # Low saturation = washed out
    min_window_area: int = 500  # Minimum pixels for a window region

    # Enhancement strengths
    saturation_boost: float = 1.6  # Boost color in windows
    contrast_boost: float = 1.25  # Enhance contrast
    vibrance_boost: float = 1.3  # Selective saturation boost

    # Color correction
    blue_boost: float = 1.15  # Windows often show sky
    warmth_reduction: float = 0.95  # Reduce yellow cast in windows

    # Blending
    blend_radius: int = 21  # Smooth mask edges
    blend_falloff: float = 0.7  # How fast mask fades at edges

    # Detail enhancement
    detail_sharpen: float = 0.3  # Light sharpening in windows
    clarity_boost: float = 1.1  # Local contrast


class WindowPostProcessor:
    """
    Robust window post-processor optimized for real estate HDR.
    """
    def __init__(self, config: PostProcessConfig = None):
        self.config = config or PostProcessConfig()

    def detect_windows(
        self,
        image: np.ndarray,
        source_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect window regions using brightness and saturation.

        Args:
            image: Model output [H, W, 3] RGB uint8
            source_image: Source input (better for detection)

        Returns:
            mask: Binary mask [H, W] uint8
        """
        # Use source for detection if available (more reliable)
        detect_img = source_image if source_image is not None else image
        detect_img = detect_img.astype(np.float32) / 255.0

        # Compute brightness (luminance)
        brightness = (
            0.299 * detect_img[:,:,0] +
            0.587 * detect_img[:,:,1] +
            0.114 * detect_img[:,:,2]
        )

        # Compute saturation
        max_rgb = detect_img.max(axis=2)
        min_rgb = detect_img.min(axis=2)
        saturation = max_rgb - min_rgb

        # Window criteria: bright AND low saturation
        bright_mask = brightness > self.config.brightness_threshold
        low_sat_mask = saturation < self.config.saturation_threshold

        window_mask = (bright_mask & low_sat_mask).astype(np.uint8) * 255

        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_CLOSE, kernel_close)
        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_OPEN, kernel_open)

        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(window_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.config.min_window_area:
                window_mask[labels == i] = 0

        return window_mask

    def create_soft_mask(self, hard_mask: np.ndarray) -> np.ndarray:
        """Create soft mask with feathered edges."""
        soft_mask = hard_mask.astype(np.float32) / 255.0

        if self.config.blend_radius > 0:
            ksize = self.config.blend_radius * 2 + 1
            soft_mask = cv2.GaussianBlur(soft_mask, (ksize, ksize), 0)

            # Apply falloff curve for smoother transitions
            soft_mask = np.power(soft_mask, self.config.blend_falloff)

        return soft_mask

    def enhance_saturation_selective(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Vibrance-style saturation boost (boosts low-saturation areas more).
        """
        img_float = image.astype(np.float32) / 255.0

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Current saturation (normalized)
        sat = hsv[:,:,1] / 255.0

        # Vibrance: boost less saturated pixels more
        # Low saturation gets bigger boost
        boost_factor = self.config.saturation_boost * (1 - sat * 0.5)

        # Apply boost only in masked regions
        mask_3d = mask[:,:,np.newaxis] if mask.ndim == 2 else mask

        new_sat = hsv[:,:,1] * (1 + (boost_factor - 1) * mask)
        hsv[:,:,1] = np.clip(new_sat, 0, 255)

        # Apply contrast boost to Value channel
        v_float = hsv[:,:,2] / 255.0
        v_boosted = 0.5 + (v_float - 0.5) * self.config.contrast_boost
        hsv[:,:,2] = np.clip(v_boosted * 255, 0, 255)

        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return enhanced

    def apply_color_correction(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply color correction to window regions.
        - Boost blue (sky often visible through windows)
        - Reduce warmth (yellow cast from interior lighting)
        """
        img_float = image.astype(np.float32)
        mask = mask[:,:,np.newaxis] if mask.ndim == 2 else mask

        # Blue boost
        blue_boosted = img_float.copy()
        blue_boosted[:,:,2] = np.clip(img_float[:,:,2] * self.config.blue_boost, 0, 255)

        # Warmth reduction (reduce red and green slightly)
        blue_boosted[:,:,0] = np.clip(img_float[:,:,0] * self.config.warmth_reduction, 0, 255)

        # Blend based on mask
        result = img_float * (1 - mask) + blue_boosted * mask

        return np.clip(result, 0, 255).astype(np.uint8)

    def enhance_local_contrast(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply CLAHE-style local contrast enhancement to windows.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])

        enhanced_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Blend based on mask
        mask = mask[:,:,np.newaxis] if mask.ndim == 2 else mask
        result = image.astype(np.float32) * (1 - mask * 0.5) + enhanced_lab.astype(np.float32) * mask * 0.5

        return np.clip(result, 0, 255).astype(np.uint8)

    def sharpen_windows(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Light sharpening in window regions for detail recovery."""
        if self.config.detail_sharpen <= 0:
            return image

        # Unsharp mask
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(
            image, 1.0 + self.config.detail_sharpen,
            blurred, -self.config.detail_sharpen,
            0
        )

        # Blend based on mask
        mask = mask[:,:,np.newaxis] if mask.ndim == 2 else mask
        result = image.astype(np.float32) * (1 - mask) + sharpened.astype(np.float32) * mask

        return np.clip(result, 0, 255).astype(np.uint8)

    def process(
        self,
        output_image: np.ndarray,
        source_image: Optional[np.ndarray] = None,
        return_mask: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full post-processing pipeline.

        Args:
            output_image: Model output [H, W, 3] RGB uint8
            source_image: Source input for better window detection
            return_mask: If True, return the window mask

        Returns:
            processed: Enhanced image
            mask: Window mask (if return_mask=True)
        """
        # Detect windows
        hard_mask = self.detect_windows(output_image, source_image)
        soft_mask = self.create_soft_mask(hard_mask)

        # Check if any windows detected
        window_coverage = soft_mask.sum() / soft_mask.size

        if window_coverage < 0.005:  # Less than 0.5% windows
            # No significant windows, return original
            if return_mask:
                return output_image, soft_mask
            return output_image

        # Apply enhancements sequentially
        result = output_image.copy()

        # 1. Saturation boost (vibrance-style)
        result = self.enhance_saturation_selective(result, soft_mask)

        # 2. Color correction (blue boost, warmth reduction)
        result = self.apply_color_correction(result, soft_mask)

        # 3. Local contrast enhancement
        result = self.enhance_local_contrast(result, soft_mask)

        # 4. Light sharpening
        result = self.sharpen_windows(result, soft_mask)

        if return_mask:
            return result, soft_mask
        return result


def create_comparison_image(
    source: np.ndarray,
    output: np.ndarray,
    postprocessed: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create side-by-side comparison: src | output | postprocessed | target
    With optional mask overlay.
    """
    h, w = source.shape[:2]

    # Resize all to same size
    def resize(img):
        if img.shape[:2] != (h, w):
            return cv2.resize(img, (w, h))
        return img

    source = resize(source)
    output = resize(output)
    postprocessed = resize(postprocessed)
    target = resize(target)

    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 30

    def add_label(img, text):
        labeled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
        labeled[label_h:] = img
        cv2.putText(labeled, text, (10, 22), font, 0.7, (255, 255, 255), 2)
        return labeled

    # Add labels (convert to BGR for cv2.putText, then back)
    source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    post_bgr = cv2.cvtColor(postprocessed, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

    source_labeled = add_label(source_bgr, "Source")
    output_labeled = add_label(output_bgr, "Model Output")
    post_labeled = add_label(post_bgr, "Post-Processed")
    target_labeled = add_label(target_bgr, "Target")

    comparison = np.concatenate([
        source_labeled, output_labeled, post_labeled, target_labeled
    ], axis=1)

    # Convert back to RGB
    comparison = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)

    return comparison


def process_folder(
    input_folder: Path,
    output_folder: Path,
    processor: WindowPostProcessor,
    source_suffix: str = '_src',
    output_suffix: str = '_output',
    target_suffix: str = '_tar',
):
    """
    Process all images in a folder.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all output images
    output_images = sorted(input_folder.glob(f'*{output_suffix}.jpg'))

    results = []

    for output_path in output_images:
        # Extract ID
        stem = output_path.stem
        img_id = stem.replace(output_suffix, '')

        # Find source and target
        source_path = input_folder / f'{img_id}{source_suffix}.jpg'
        target_path = input_folder / f'{img_id}{target_suffix}.jpg'

        if not source_path.exists():
            # Try alternate naming
            source_path = input_folder / f'{img_id}_input.jpg'
        if not target_path.exists():
            target_path = input_folder / f'{img_id}_target.jpg'

        # Load images
        output_img = np.array(Image.open(output_path).convert('RGB'))
        source_img = np.array(Image.open(source_path).convert('RGB')) if source_path.exists() else None
        target_img = np.array(Image.open(target_path).convert('RGB')) if target_path.exists() else None

        # Process
        postprocessed, mask = processor.process(output_img, source_img, return_mask=True)

        # Save postprocessed
        post_path = output_folder / f'{img_id}_postprocessed.jpg'
        Image.fromarray(postprocessed).save(post_path, quality=95)

        # Save comparison if target available
        if target_img is not None and source_img is not None:
            comparison = create_comparison_image(
                source_img, output_img, postprocessed, target_img, mask
            )
            comp_path = output_folder / f'{img_id}_comparison.jpg'
            Image.fromarray(comparison).save(comp_path, quality=95)

        # Calculate metrics
        if target_img is not None:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            # Resize if needed
            if target_img.shape != output_img.shape:
                target_img = cv2.resize(target_img, (output_img.shape[1], output_img.shape[0]))
            if target_img.shape != postprocessed.shape:
                postprocessed_resized = cv2.resize(postprocessed, (target_img.shape[1], target_img.shape[0]))
            else:
                postprocessed_resized = postprocessed

            psnr_before = peak_signal_noise_ratio(target_img, output_img)
            psnr_after = peak_signal_noise_ratio(target_img, postprocessed_resized)

            ssim_before = structural_similarity(target_img, output_img, channel_axis=2)
            ssim_after = structural_similarity(target_img, postprocessed_resized, channel_axis=2)

            results.append({
                'id': img_id,
                'psnr_before': float(psnr_before),
                'psnr_after': float(psnr_after),
                'psnr_delta': float(psnr_after - psnr_before),
                'ssim_before': float(ssim_before),
                'ssim_after': float(ssim_after),
                'ssim_delta': float(ssim_after - ssim_before),
                'window_coverage': float(mask.sum() / mask.size)
            })

            print(f"  {img_id}: PSNR {psnr_before:.2f} -> {psnr_after:.2f} ({psnr_after-psnr_before:+.2f}), "
                  f"SSIM {ssim_before:.4f} -> {ssim_after:.4f} ({ssim_after-ssim_before:+.4f})")
        else:
            print(f"  {img_id}: processed (no target for metrics)")

    # Save results
    if results:
        with open(output_folder / 'metrics.json', 'w') as f:
            json.dump({
                'per_image': results,
                'average': {
                    'psnr_before': np.mean([r['psnr_before'] for r in results]),
                    'psnr_after': np.mean([r['psnr_after'] for r in results]),
                    'psnr_delta': np.mean([r['psnr_delta'] for r in results]),
                    'ssim_before': np.mean([r['ssim_before'] for r in results]),
                    'ssim_after': np.mean([r['ssim_after'] for r in results]),
                    'ssim_delta': np.mean([r['ssim_delta'] for r in results]),
                }
            }, f, indent=2)

    return results


def main():
    print("=" * 70)
    print("Window Post-Processing for Real Estate HDR")
    print("=" * 70)

    # Install skimage if needed
    try:
        from skimage.metrics import peak_signal_noise_ratio
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-image', '-q'])
        from skimage.metrics import peak_signal_noise_ratio

    # Create optimized processor
    config = PostProcessConfig(
        brightness_threshold=0.58,  # Slightly lower to catch more windows
        saturation_boost=1.5,
        contrast_boost=1.2,
        blue_boost=1.12,
        warmth_reduction=0.96,
        detail_sharpen=0.25,
    )
    processor = WindowPostProcessor(config)

    test_dir = Path('test')

    # Process Restormer 896
    print("\n" + "=" * 70)
    print("Processing Restormer 896")
    print("=" * 70)

    restormer_input = test_dir / 'restormer_896'
    restormer_output = test_dir / 'restormer_896_postprocessed'

    if restormer_input.exists():
        process_folder(
            restormer_input,
            restormer_output,
            processor,
            source_suffix='_src',
            output_suffix='_output',
            target_suffix='_tar'
        )
    else:
        print(f"  Folder not found: {restormer_input}")

    # Process DarkIR baseline
    print("\n" + "=" * 70)
    print("Processing DarkIR Baseline")
    print("=" * 70)

    darkir_input = test_dir / 'darkir_baseline'
    darkir_output = test_dir / 'darkir_baseline_postprocessed'

    if darkir_input.exists():
        process_folder(
            darkir_input,
            darkir_output,
            processor,
            source_suffix='_input',
            output_suffix='_output',
            target_suffix='_target'
        )
    else:
        print(f"  Folder not found: {darkir_input}")

    # Process Restormer 384 if exists
    print("\n" + "=" * 70)
    print("Processing Restormer 384")
    print("=" * 70)

    restormer_384_input = test_dir / 'restormer_384'
    restormer_384_output = test_dir / 'restormer_384_postprocessed'

    if restormer_384_input.exists():
        process_folder(
            restormer_384_input,
            restormer_384_output,
            processor,
            source_suffix='_src',
            output_suffix='_output',
            target_suffix='_tar'
        )
    else:
        print(f"  Folder not found: {restormer_384_input}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for output_dir in [restormer_output, darkir_output, restormer_384_output]:
        metrics_file = output_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            avg = metrics['average']
            print(f"\n{output_dir.name}:")
            print(f"  PSNR: {avg['psnr_before']:.2f} -> {avg['psnr_after']:.2f} ({avg['psnr_delta']:+.2f})")
            print(f"  SSIM: {avg['ssim_before']:.4f} -> {avg['ssim_after']:.4f} ({avg['ssim_delta']:+.4f})")

    print("\n" + "=" * 70)
    print("Done! Check test/*_postprocessed/ folders for results")
    print("=" * 70)


if __name__ == '__main__':
    main()
