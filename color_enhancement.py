#!/usr/bin/env python3
"""
Color Enhancement Post-Processing for HDR Real Estate Images

Addresses undersaturation and dull colors from neural network outputs.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def histogram_match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of source to reference for a single channel"""
    src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    src_cdf = (src_cdf / src_cdf[-1] * 255).astype(np.uint8)
    ref_cdf = (ref_cdf / ref_cdf[-1] * 255).astype(np.uint8)

    # Create lookup table
    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_idx in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
            ref_idx += 1
        lut[src_idx] = ref_idx

    return lut[source]


def adaptive_saturation_boost(img: np.ndarray, target_saturation: float = None,
                              boost_factor: float = 1.3) -> np.ndarray:
    """
    Boost saturation adaptively in HSV space.

    Args:
        img: BGR image
        target_saturation: Target mean saturation (if None, use boost_factor)
        boost_factor: Multiplicative boost (default 1.3 = 30% increase)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if target_saturation is not None:
        current_sat = np.mean(hsv[:,:,1])
        if current_sat > 0:
            boost_factor = target_saturation / current_sat

    hsv[:,:,1] = np.clip(hsv[:,:,1] * boost_factor, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def lab_color_enhance(img: np.ndarray, l_boost: float = 1.0,
                      ab_boost: float = 1.2) -> np.ndarray:
    """
    Enhance colors in LAB space.

    Args:
        img: BGR image
        l_boost: Luminance adjustment (1.0 = no change)
        ab_boost: Color channel boost (>1 = more colorful)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # L channel: adjust around middle gray (128)
    lab[:,:,0] = np.clip((lab[:,:,0] - 128) * l_boost + 128, 0, 255)

    # a* and b* channels: boost deviation from neutral (128)
    lab[:,:,1] = np.clip((lab[:,:,1] - 128) * ab_boost + 128, 0, 255)
    lab[:,:,2] = np.clip((lab[:,:,2] - 128) * ab_boost + 128, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def histogram_match_color(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match color histogram of source to reference image.
    Works in LAB space for perceptually uniform matching.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    matched = np.zeros_like(src_lab)
    for i in range(3):
        matched[:,:,i] = histogram_match_channel(src_lab[:,:,i], ref_lab[:,:,i])

    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)


def auto_color_correct(output: np.ndarray, input_img: np.ndarray,
                       target: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Automatic color correction for neural network outputs.

    Strategy:
    1. If target available: histogram match to target
    2. Else: adaptive saturation boost based on input-output comparison
    """
    if target is not None:
        # Best option: match to target color distribution
        return histogram_match_color(output, target)

    # Fallback: boost saturation to match input's colorfulness
    input_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    output_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    input_sat = np.mean(input_hsv[:,:,1])
    output_sat = np.mean(output_hsv[:,:,1])

    # Boost to at least 80% of input saturation
    target_sat = max(output_sat, input_sat * 0.8)

    return adaptive_saturation_boost(output, target_saturation=target_sat)


def enhance_real_estate_hdr(img: np.ndarray,
                            saturation_boost: float = 1.25,
                            contrast_boost: float = 1.1,
                            warmth: float = 0.0) -> np.ndarray:
    """
    Real estate specific enhancement.

    Args:
        img: BGR image
        saturation_boost: Saturation multiplier (1.25 = 25% boost)
        contrast_boost: Contrast multiplier
        warmth: Color temperature shift (-1 to 1, positive = warmer)
    """
    # 1. Saturation boost in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_boost, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 2. Contrast in LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,0] = np.clip((lab[:,:,0] - 128) * contrast_boost + 128, 0, 255)

    # 3. Warmth adjustment (shift b* channel)
    if warmth != 0:
        lab[:,:,2] = np.clip(lab[:,:,2] + warmth * 20, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    import json
    from pathlib import Path

    print("=" * 70)
    print("COLOR ENHANCEMENT TEST")
    print("=" * 70)

    # Test on comparison outputs
    test_pairs = [json.loads(l) for l in open('data_splits/test.jsonl') if l.strip()]

    output_dir = Path('test/color_enhanced')
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair in test_pairs[:3]:
        name = Path(pair['src']).stem.replace('_src', '')

        # Load original comparison
        comp = cv2.imread(f'test/comparison/{name}_comparison.jpg')
        if comp is None:
            continue

        h, w = comp.shape[:2]
        q = w // 4

        input_img = comp[:, :q]
        restormer = comp[:, 2*q:3*q]
        target = comp[:, 3*q:]

        # Apply different enhancement methods
        enhanced_hist = histogram_match_color(restormer, target)
        enhanced_sat = adaptive_saturation_boost(restormer, boost_factor=1.3)
        enhanced_lab = lab_color_enhance(restormer, ab_boost=1.25)
        enhanced_auto = auto_color_correct(restormer, input_img, target)

        # Create comparison: Input | Original | Hist Match | Target
        comparison = np.hstack([input_img, restormer, enhanced_hist, target])
        cv2.imwrite(str(output_dir / f'{name}_hist_match.jpg'), comparison)

        # Analyze improvement
        def get_sat(img):
            return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1])

        print(f"\n{name}:")
        print(f"  Target saturation:    {get_sat(target):.1f}")
        print(f"  Original output:      {get_sat(restormer):.1f}")
        print(f"  Histogram matched:    {get_sat(enhanced_hist):.1f}")
        print(f"  Saturation boosted:   {get_sat(enhanced_sat):.1f}")
        print(f"  LAB enhanced:         {get_sat(enhanced_lab):.1f}")

    print(f"\nEnhanced images saved to: {output_dir}/")
