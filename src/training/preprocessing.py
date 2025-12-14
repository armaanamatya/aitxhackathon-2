"""
Modular, configurable preprocessing pipeline for training data.
MLE-grade design with composable transforms and easy experimentation.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random


# ============================================================================
# BASE TRANSFORM INTERFACE
# ============================================================================

class Transform(ABC):
    """Base class for all transforms"""

    @abstractmethod
    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transform to source and target images"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get transform configuration for logging"""
        pass


# ============================================================================
# COLOR SPACE TRANSFORMS
# ============================================================================

class ToLAB(Transform):
    """Convert RGB to LAB color space for perceptual uniformity"""

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        tar_lab = cv2.cvtColor(tar, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        return src_lab, tar_lab

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ToLAB"}


class ToLinearRGB(Transform):
    """Convert sRGB to linear RGB (gamma 2.2 correction)"""

    def __init__(self, gamma: float = 2.2):
        self.gamma = gamma

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        src_linear = np.power(src / 255.0, self.gamma)
        tar_linear = np.power(tar / 255.0, self.gamma)
        return (src_linear * 255).astype(np.uint8), (tar_linear * 255).astype(np.uint8)

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ToLinearRGB", "gamma": self.gamma}


# ============================================================================
# NORMALIZATION TRANSFORMS
# ============================================================================

class HistogramMatching(Transform):
    """Match source histogram to target (reduce dataset variance)"""

    def __init__(self, apply_to: str = "source"):
        """
        Args:
            apply_to: 'source' (normalize src to tar) or 'both' (mutual normalization)
        """
        self.apply_to = apply_to

    def _match_histograms(self, source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Match histogram of source to template"""
        matched = np.zeros_like(source)
        for i in range(3):  # RGB channels
            matched[:, :, i] = self._match_channel(source[:, :, i], template[:, :, i])
        return matched

    def _match_channel(self, source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Match single channel histogram"""
        src_values, src_counts = np.unique(source.ravel(), return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]
        tmpl_cdf = np.cumsum(tmpl_counts).astype(np.float64)
        tmpl_cdf /= tmpl_cdf[-1]

        interp_tmpl_values = np.interp(src_cdf, tmpl_cdf, tmpl_values)
        return interp_tmpl_values[np.searchsorted(src_values, source.ravel())].reshape(source.shape)

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.apply_to == "source":
            return self._match_histograms(src, tar), tar
        elif self.apply_to == "both":
            # Mutual normalization (experimental)
            mean_img = ((src.astype(np.float32) + tar.astype(np.float32)) / 2).astype(np.uint8)
            return self._match_histograms(src, mean_img), self._match_histograms(tar, mean_img)
        return src, tar

    def get_config(self) -> Dict[str, Any]:
        return {"name": "HistogramMatching", "apply_to": self.apply_to}


class ExposureNormalization(Transform):
    """Normalize exposure across dataset"""

    def __init__(self, target_mean: float = 0.45, apply_to: str = "source"):
        """
        Args:
            target_mean: Target brightness mean [0, 1]
            apply_to: 'source', 'target', or 'both'
        """
        self.target_mean = target_mean
        self.apply_to = apply_to

    def _normalize_exposure(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to target mean brightness"""
        img_f = img.astype(np.float32) / 255.0
        current_mean = np.mean(img_f)
        if current_mean < 0.01:
            return img
        scale = self.target_mean / current_mean
        normalized = np.clip(img_f * scale, 0, 1)
        return (normalized * 255).astype(np.uint8)

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.apply_to in ["source", "both"]:
            src = self._normalize_exposure(src)
        if self.apply_to in ["target", "both"]:
            tar = self._normalize_exposure(tar)
        return src, tar

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ExposureNormalization", "target_mean": self.target_mean, "apply_to": self.apply_to}


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class RandomHorizontalFlip(Transform):
    """Random horizontal flip (both src and tar)"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            return cv2.flip(src, 1), cv2.flip(tar, 1)
        return src, tar

    def get_config(self) -> Dict[str, Any]:
        return {"name": "RandomHorizontalFlip", "p": self.p}


class RandomCrop(Transform):
    """Random crop (paired for src and tar)"""

    def __init__(self, crop_size: Tuple[int, int], p: float = 1.0):
        self.crop_size = crop_size
        self.p = p

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return src, tar

        h, w = src.shape[:2]
        crop_h, crop_w = self.crop_size

        if h <= crop_h or w <= crop_w:
            return src, tar

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        src_crop = src[top:top+crop_h, left:left+crop_w]
        tar_crop = tar[top:top+crop_h, left:left+crop_w]

        return src_crop, tar_crop

    def get_config(self) -> Dict[str, Any]:
        return {"name": "RandomCrop", "crop_size": self.crop_size, "p": self.p}


class RandomRotation(Transform):
    """Random 90-degree rotations"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return src, tar

        k = random.randint(1, 3)  # 90, 180, or 270 degrees
        return cv2.rotate(src, k - 1), cv2.rotate(tar, k - 1)

    def get_config(self) -> Dict[str, Any]:
        return {"name": "RandomRotation", "p": self.p}


# ============================================================================
# QUALITY ENHANCEMENT
# ============================================================================

class DenoiseSource(Transform):
    """Denoise source images (can help model focus on structure)"""

    def __init__(self, strength: int = 5):
        self.strength = strength

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        src_denoised = cv2.fastNlMeansDenoisingColored(src, None, self.strength, self.strength, 7, 21)
        return src_denoised, tar

    def get_config(self) -> Dict[str, Any]:
        return {"name": "DenoiseSource", "strength": self.strength}


class SharpenTarget(Transform):
    """Sharpen target images (emphasize details)"""

    def __init__(self, strength: float = 0.5):
        self.strength = strength
        self.kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tar_sharpened = cv2.filter2D(tar, -1, self.kernel)
        tar_blended = cv2.addWeighted(tar, 1 - self.strength, tar_sharpened, self.strength, 0)
        return src, tar_blended

    def get_config(self) -> Dict[str, Any]:
        return {"name": "SharpenTarget", "strength": self.strength}


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline"""

    # Color space
    color_space: str = "RGB"  # RGB, LAB, Linear
    linear_gamma: float = 2.2

    # Normalization
    normalize_exposure: bool = False
    exposure_target_mean: float = 0.45
    exposure_apply_to: str = "source"  # source, target, both
    histogram_matching: bool = False
    histogram_apply_to: str = "source"

    # Augmentation
    random_flip: bool = True
    flip_p: float = 0.5
    random_crop: bool = False
    crop_size: Optional[Tuple[int, int]] = None
    crop_p: float = 1.0
    random_rotation: bool = False
    rotation_p: float = 0.5

    # Quality
    denoise_source: bool = False
    denoise_strength: int = 5
    sharpen_target: bool = False
    sharpen_strength: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'color_space': self.color_space,
            'linear_gamma': self.linear_gamma,
            'normalize_exposure': self.normalize_exposure,
            'exposure_target_mean': self.exposure_target_mean,
            'exposure_apply_to': self.exposure_apply_to,
            'histogram_matching': self.histogram_matching,
            'histogram_apply_to': self.histogram_apply_to,
            'random_flip': self.random_flip,
            'flip_p': self.flip_p,
            'random_crop': self.random_crop,
            'crop_size': self.crop_size,
            'crop_p': self.crop_p,
            'random_rotation': self.random_rotation,
            'rotation_p': self.rotation_p,
            'denoise_source': self.denoise_source,
            'denoise_strength': self.denoise_strength,
            'sharpen_target': self.sharpen_target,
            'sharpen_strength': self.sharpen_strength,
        }


class PreprocessingPipeline:
    """Composable preprocessing pipeline"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.transforms: List[Transform] = []
        self._build_pipeline()

    def _build_pipeline(self):
        """Build transform pipeline from config"""

        # Color space conversion
        if self.config.color_space == "LAB":
            self.transforms.append(ToLAB())
        elif self.config.color_space == "Linear":
            self.transforms.append(ToLinearRGB(self.config.linear_gamma))

        # Normalization
        if self.config.normalize_exposure:
            self.transforms.append(ExposureNormalization(
                self.config.exposure_target_mean,
                self.config.exposure_apply_to
            ))

        if self.config.histogram_matching:
            self.transforms.append(HistogramMatching(self.config.histogram_apply_to))

        # Quality enhancement
        if self.config.denoise_source:
            self.transforms.append(DenoiseSource(self.config.denoise_strength))

        if self.config.sharpen_target:
            self.transforms.append(SharpenTarget(self.config.sharpen_strength))

        # Augmentation (applied last)
        if self.config.random_flip:
            self.transforms.append(RandomHorizontalFlip(self.config.flip_p))

        if self.config.random_crop and self.config.crop_size:
            self.transforms.append(RandomCrop(self.config.crop_size, self.config.crop_p))

        if self.config.random_rotation:
            self.transforms.append(RandomRotation(self.config.rotation_p))

    def __call__(self, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all transforms in sequence"""
        for transform in self.transforms:
            src, tar = transform(src, tar)
        return src, tar

    def get_config(self) -> Dict[str, Any]:
        """Get full pipeline config"""
        return {
            'config': self.config.to_dict(),
            'transforms': [t.get_config() for t in self.transforms]
        }


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_preset_config(preset: str) -> PreprocessConfig:
    """Get preset configuration"""

    presets = {
        # No preprocessing (baseline)
        "none": PreprocessConfig(),

        # Light augmentation only
        "light_aug": PreprocessConfig(
            random_flip=True,
            flip_p=0.5,
        ),

        # Standard augmentation
        "standard_aug": PreprocessConfig(
            random_flip=True,
            flip_p=0.5,
            random_rotation=True,
            rotation_p=0.25,
        ),

        # Exposure normalization
        "normalize_exposure": PreprocessConfig(
            normalize_exposure=True,
            exposure_target_mean=0.45,
            exposure_apply_to="source",
            random_flip=True,
        ),

        # Histogram matching
        "histogram_match": PreprocessConfig(
            histogram_matching=True,
            histogram_apply_to="source",
            random_flip=True,
        ),

        # LAB color space
        "lab_colorspace": PreprocessConfig(
            color_space="LAB",
            random_flip=True,
        ),

        # Quality enhancement
        "quality_enhance": PreprocessConfig(
            denoise_source=True,
            denoise_strength=5,
            sharpen_target=True,
            sharpen_strength=0.3,
            random_flip=True,
        ),

        # Aggressive (all techniques)
        "aggressive": PreprocessConfig(
            normalize_exposure=True,
            exposure_target_mean=0.45,
            exposure_apply_to="source",
            denoise_source=True,
            denoise_strength=5,
            random_flip=True,
            random_rotation=True,
            rotation_p=0.25,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    return presets[preset]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import json

    # Example 1: Use preset
    config = get_preset_config("standard_aug")
    pipeline = PreprocessingPipeline(config)

    # Load example image pair
    src = cv2.imread("images/1_src.jpg")
    tar = cv2.imread("images/1_tar.jpg")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    src_processed, tar_processed = pipeline(src, tar)

    # Print config
    print(json.dumps(pipeline.get_config(), indent=2))

    # Example 2: Custom config
    custom_config = PreprocessConfig(
        normalize_exposure=True,
        exposure_target_mean=0.5,
        random_flip=True,
        flip_p=0.5,
    )
    custom_pipeline = PreprocessingPipeline(custom_config)
    print("\nCustom config:")
    print(json.dumps(custom_pipeline.get_config(), indent=2))
