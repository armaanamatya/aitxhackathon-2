"""
Window Post-Processing for Real Estate HDR Enhancement
======================================================

Quick post-processing to enhance window regions in model outputs.
Use this for immediate improvements, but window-aware training is recommended
for best results.

Limitations:
- Cannot recover detail the model didn't generate
- Heuristic approach, not learned from ground truth
- May create edge artifacts
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Union
from pathlib import Path


class WindowPostProcessor:
    """
    Post-process model outputs to enhance window regions.
    """
    def __init__(
        self,
        brightness_threshold: float = 0.65,
        saturation_boost: float = 1.5,
        contrast_boost: float = 1.2,
        color_temperature_shift: float = 0.0,  # Positive = warmer, negative = cooler
        blend_radius: int = 15,
        min_window_area: int = 1000,  # Minimum pixels to be considered a window
    ):
        self.brightness_threshold = brightness_threshold
        self.saturation_boost = saturation_boost
        self.contrast_boost = contrast_boost
        self.color_temperature_shift = color_temperature_shift
        self.blend_radius = blend_radius
        self.min_window_area = min_window_area

    def detect_windows(
        self,
        image: np.ndarray,
        use_source: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect window regions.

        Args:
            image: RGB image [H, W, 3] in [0, 255]
            use_source: If provided, detect windows from source instead of output

        Returns:
            mask: Binary mask [H, W] with 1s for window regions
        """
        detect_img = use_source if use_source is not None else image
        detect_img = detect_img.astype(np.float32) / 255.0

        # Compute brightness
        brightness = 0.299 * detect_img[:,:,0] + 0.587 * detect_img[:,:,1] + 0.114 * detect_img[:,:,2]

        # Compute saturation
        max_rgb = detect_img.max(axis=2)
        min_rgb = detect_img.min(axis=2)
        saturation = max_rgb - min_rgb

        # Window: bright AND low saturation (overexposed)
        bright_mask = brightness > self.brightness_threshold
        low_sat_mask = saturation < 0.15

        window_mask = (bright_mask & low_sat_mask).astype(np.uint8)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_CLOSE, kernel)
        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_OPEN, kernel)

        # Filter small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(window_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_window_area:
                window_mask[labels == i] = 0

        return window_mask

    def create_soft_mask(self, hard_mask: np.ndarray) -> np.ndarray:
        """Create soft mask with feathered edges for smooth blending."""
        soft_mask = hard_mask.astype(np.float32)

        # Gaussian blur for soft edges
        if self.blend_radius > 0:
            soft_mask = cv2.GaussianBlur(
                soft_mask,
                (self.blend_radius * 2 + 1, self.blend_radius * 2 + 1),
                0
            )

        return soft_mask

    def enhance_windows(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply enhancement to window regions.

        Args:
            image: RGB image [H, W, 3] in [0, 255]
            mask: Soft mask [H, W] in [0, 1]

        Returns:
            enhanced: Enhanced image
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        # Convert to HSV for saturation boost
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Boost saturation in window regions
        enhanced_sat = img_hsv[:,:,1] * self.saturation_boost
        enhanced_sat = np.clip(enhanced_sat, 0, 255)

        # Boost contrast in window regions (around 0.5 midpoint)
        enhanced_v = img_hsv[:,:,2].astype(np.float32) / 255.0
        enhanced_v = 0.5 + (enhanced_v - 0.5) * self.contrast_boost
        enhanced_v = np.clip(enhanced_v * 255, 0, 255)

        # Color temperature shift
        if self.color_temperature_shift != 0:
            # Shift hue slightly
            enhanced_h = img_hsv[:,:,0] + self.color_temperature_shift * 10
            enhanced_h = enhanced_h % 180
        else:
            enhanced_h = img_hsv[:,:,0]

        # Reconstruct enhanced HSV
        enhanced_hsv = np.stack([enhanced_h, enhanced_sat, enhanced_v], axis=2).astype(np.uint8)
        enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

        # Blend original and enhanced using mask
        mask_3d = mask[:, :, np.newaxis]
        result = image * (1 - mask_3d) + enhanced_rgb * mask_3d

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_local_tone_mapping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        gamma: float = 0.7,  # < 1 brightens, > 1 darkens
    ) -> np.ndarray:
        """
        Apply local tone mapping to window regions.
        Can help recover some detail in overexposed areas.
        """
        img_float = image.astype(np.float32) / 255.0

        # Apply gamma correction to brighten/recover detail
        enhanced = np.power(img_float, gamma)

        # Blend
        mask_3d = mask[:, :, np.newaxis]
        result = img_float * (1 - mask_3d) + enhanced * mask_3d

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def process(
        self,
        output_image: np.ndarray,
        source_image: Optional[np.ndarray] = None,
        return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Full post-processing pipeline.

        Args:
            output_image: Model output RGB [H, W, 3] in [0, 255]
            source_image: Original input (better for window detection)
            return_mask: If True, also return the window mask

        Returns:
            processed: Enhanced image
            mask (optional): Window mask used
        """
        # Detect windows (use source if available for better detection)
        hard_mask = self.detect_windows(output_image, use_source=source_image)
        soft_mask = self.create_soft_mask(hard_mask)

        # Apply enhancements
        enhanced = self.enhance_windows(output_image, soft_mask)

        # Optional: apply local tone mapping for detail recovery
        # enhanced = self.apply_local_tone_mapping(enhanced, soft_mask, gamma=0.8)

        if return_mask:
            return enhanced, soft_mask
        return enhanced

    def process_batch(
        self,
        outputs: torch.Tensor,
        sources: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process a batch of images.

        Args:
            outputs: [B, 3, H, W] tensor in [-1, 1]
            sources: [B, 3, H, W] tensor in [-1, 1] (optional)

        Returns:
            processed: [B, 3, H, W] tensor in [-1, 1]
        """
        # Convert to numpy
        outputs_np = ((outputs + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        if sources is not None:
            sources_np = ((sources + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        else:
            sources_np = [None] * len(outputs_np)

        # Process each image
        processed = []
        for out, src in zip(outputs_np, sources_np):
            proc = self.process(out, src)
            processed.append(proc)

        # Convert back to tensor
        processed = np.stack(processed)
        processed = torch.from_numpy(processed).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1

        return processed.to(outputs.device)


class AdaptiveWindowPostProcessor(WindowPostProcessor):
    """
    Adaptive post-processor that learns optimal parameters from ground truth.

    Use this to find the best post-processing parameters for your dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_params = None

    def optimize_params(
        self,
        model_outputs: list,
        sources: list,
        targets: list,
        metric: str = 'psnr'
    ) -> dict:
        """
        Find optimal post-processing parameters using grid search.

        Args:
            model_outputs: List of model output images [H, W, 3]
            sources: List of source images [H, W, 3]
            targets: List of ground truth images [H, W, 3]
            metric: 'psnr' or 'ssim'

        Returns:
            best_params: Dict of optimal parameters
        """
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        best_score = -float('inf')
        best_params = {}

        # Grid search
        for sat_boost in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            for contrast in [1.0, 1.1, 1.2, 1.3]:
                for brightness_thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
                    self.saturation_boost = sat_boost
                    self.contrast_boost = contrast
                    self.brightness_threshold = brightness_thresh

                    scores = []
                    for out, src, tar in zip(model_outputs, sources, targets):
                        processed = self.process(out, src)

                        if metric == 'psnr':
                            score = peak_signal_noise_ratio(tar, processed)
                        else:
                            score = structural_similarity(tar, processed, channel_axis=2)
                        scores.append(score)

                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'saturation_boost': sat_boost,
                            'contrast_boost': contrast,
                            'brightness_threshold': brightness_thresh,
                            'score': avg_score,
                            'metric': metric,
                        }

        self.saturation_boost = best_params['saturation_boost']
        self.contrast_boost = best_params['contrast_boost']
        self.brightness_threshold = best_params['brightness_threshold']
        self.best_params = best_params

        return best_params


def postprocess_image(
    output_path: Union[str, Path],
    source_path: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> np.ndarray:
    """
    Convenience function to post-process a single image.

    Args:
        output_path: Path to model output image
        source_path: Path to source image (optional, helps with window detection)
        save_path: If provided, save the result here
        **kwargs: Parameters for WindowPostProcessor

    Returns:
        processed: Enhanced image as numpy array
    """
    processor = WindowPostProcessor(**kwargs)

    output_img = np.array(Image.open(output_path).convert('RGB'))
    source_img = np.array(Image.open(source_path).convert('RGB')) if source_path else None

    processed = processor.process(output_img, source_img)

    if save_path:
        Image.fromarray(processed).save(save_path)

    return processed


def postprocess_batch_folder(
    output_folder: Union[str, Path],
    source_folder: Optional[Union[str, Path]] = None,
    save_folder: Optional[Union[str, Path]] = None,
    **kwargs
):
    """
    Post-process all images in a folder.
    """
    output_folder = Path(output_folder)
    source_folder = Path(source_folder) if source_folder else None
    save_folder = Path(save_folder) if save_folder else output_folder / 'postprocessed'
    save_folder.mkdir(exist_ok=True, parents=True)

    processor = WindowPostProcessor(**kwargs)

    for output_path in sorted(output_folder.glob('*.jpg')) + sorted(output_folder.glob('*.png')):
        source_path = None
        if source_folder:
            # Try to find matching source file
            source_path = source_folder / output_path.name
            if not source_path.exists():
                source_path = None

        output_img = np.array(Image.open(output_path).convert('RGB'))
        source_img = np.array(Image.open(source_path).convert('RGB')) if source_path else None

        processed = processor.process(output_img, source_img)

        save_path = save_folder / output_path.name
        Image.fromarray(processed).save(save_path)
        print(f"Processed: {output_path.name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Window Post-Processing')
    parser.add_argument('--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('--source', type=str, help='Source image or folder (optional)')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--saturation', type=float, default=1.5, help='Saturation boost')
    parser.add_argument('--contrast', type=float, default=1.2, help='Contrast boost')
    parser.add_argument('--threshold', type=float, default=0.65, help='Brightness threshold')

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        postprocess_batch_folder(
            input_path,
            source_folder=args.source,
            save_folder=args.output,
            saturation_boost=args.saturation,
            contrast_boost=args.contrast,
            brightness_threshold=args.threshold
        )
    else:
        processed = postprocess_image(
            input_path,
            source_path=args.source,
            save_path=args.output,
            saturation_boost=args.saturation,
            contrast_boost=args.contrast,
            brightness_threshold=args.threshold
        )
        print(f"Processed: {input_path}")
