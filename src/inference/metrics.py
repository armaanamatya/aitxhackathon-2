"""
Image Quality Metrics for Real Estate HDR Enhancement

Computes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Color histogram comparison
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Try to import optional dependencies
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. PSNR/SSIM will be computed manually.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")


class ImageMetrics:
    """Compute image quality metrics between generated and ground truth images."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize LPIPS model if available
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image [H, W, C] in [0, 255] to tensor [1, C, H, W] in [-1, 1]."""
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 255.0 * 2 - 1  # [0, 255] -> [-1, 1]
        return tensor.to(self.device)

    def psnr(self, generated: np.ndarray, target: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.

        Args:
            generated: Generated image [H, W, C] in [0, 255]
            target: Target image [H, W, C] in [0, 255]

        Returns:
            PSNR value in dB (higher is better)
        """
        if SKIMAGE_AVAILABLE:
            return peak_signal_noise_ratio(target, generated, data_range=255)
        else:
            # Manual computation
            mse = np.mean((generated.astype(float) - target.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * np.log10(255.0 / np.sqrt(mse))

    def ssim(self, generated: np.ndarray, target: np.ndarray) -> float:
        """
        Compute Structural Similarity Index.

        Args:
            generated: Generated image [H, W, C] in [0, 255]
            target: Target image [H, W, C] in [0, 255]

        Returns:
            SSIM value in [0, 1] (higher is better)
        """
        if SKIMAGE_AVAILABLE:
            return structural_similarity(
                target, generated,
                channel_axis=2,
                data_range=255,
            )
        else:
            # Simplified SSIM
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            gen = generated.astype(float)
            tar = target.astype(float)

            mu_gen = np.mean(gen)
            mu_tar = np.mean(tar)
            sigma_gen = np.var(gen)
            sigma_tar = np.var(tar)
            sigma_gen_tar = np.cov(gen.flatten(), tar.flatten())[0, 1]

            ssim = ((2 * mu_gen * mu_tar + C1) * (2 * sigma_gen_tar + C2)) / \
                   ((mu_gen ** 2 + mu_tar ** 2 + C1) * (sigma_gen + sigma_tar + C2))

            return float(ssim)

    @torch.no_grad()
    def lpips_score(self, generated: np.ndarray, target: np.ndarray) -> float:
        """
        Compute LPIPS (perceptual similarity).

        Args:
            generated: Generated image [H, W, C] in [0, 255]
            target: Target image [H, W, C] in [0, 255]

        Returns:
            LPIPS value (lower is better)
        """
        if self.lpips_model is None:
            return 0.0

        gen_tensor = self._to_tensor(generated)
        tar_tensor = self._to_tensor(target)

        # Resize if needed (LPIPS expects reasonable resolution)
        if gen_tensor.shape[2] > 512 or gen_tensor.shape[3] > 512:
            gen_tensor = F.interpolate(gen_tensor, size=(512, 512), mode='bilinear')
            tar_tensor = F.interpolate(tar_tensor, size=(512, 512), mode='bilinear')

        return self.lpips_model(gen_tensor, tar_tensor).item()

    def color_histogram_similarity(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        bins: int = 64
    ) -> float:
        """
        Compute color histogram similarity using histogram intersection.

        Args:
            generated: Generated image [H, W, C] in [0, 255]
            target: Target image [H, W, C] in [0, 255]
            bins: Number of histogram bins

        Returns:
            Similarity score in [0, 1] (higher is better)
        """
        similarities = []

        for c in range(3):
            hist_gen, _ = np.histogram(generated[:, :, c], bins=bins, range=(0, 255))
            hist_tar, _ = np.histogram(target[:, :, c], bins=bins, range=(0, 255))

            # Normalize histograms
            hist_gen = hist_gen.astype(float) / hist_gen.sum()
            hist_tar = hist_tar.astype(float) / hist_tar.sum()

            # Histogram intersection
            similarity = np.minimum(hist_gen, hist_tar).sum()
            similarities.append(similarity)

        return np.mean(similarities)

    def compute_all(
        self,
        generated: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            generated: Generated image [H, W, C] in [0, 255]
            target: Target image [H, W, C] in [0, 255]

        Returns:
            Dictionary of metric names to values
        """
        return {
            'psnr': self.psnr(generated, target),
            'ssim': self.ssim(generated, target),
            'lpips': self.lpips_score(generated, target),
            'color_hist': self.color_histogram_similarity(generated, target),
        }


def evaluate_model(
    model_path: str,
    data_root: str,
    jsonl_path: str,
    image_size: int = 512,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on the dataset.

    Args:
        model_path: Path to trained model
        data_root: Root directory with images
        jsonl_path: Path to train.jsonl
        image_size: Image size for inference
        num_samples: Number of samples to evaluate (None for all)

    Returns:
        Dictionary of average metrics
    """
    import json

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.inference.infer import HDREnhancer

    # Initialize
    enhancer = HDREnhancer(
        model_path=model_path,
        image_size=image_size,
        precision="fp16",
    )
    metrics = ImageMetrics()

    # Load dataset
    data_root = Path(data_root)
    with open(jsonl_path, 'r') as f:
        pairs = [json.loads(line) for line in f]

    if num_samples:
        pairs = pairs[:num_samples]

    # Compute metrics
    all_metrics = []

    for pair in tqdm(pairs, desc="Evaluating"):
        src_path = data_root / pair['src']
        tar_path = data_root / pair['tar']

        # Load images
        src_img = Image.open(src_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Generate enhanced image
        gen_img = enhancer.enhance(src_img, preserve_resolution=True)

        # Resize target to match generated (in case of resolution differences)
        tar_img = tar_img.resize(gen_img.size, Image.LANCZOS)

        # Convert to numpy
        gen_np = np.array(gen_img)
        tar_np = np.array(tar_img)

        # Compute metrics
        sample_metrics = metrics.compute_all(gen_np, tar_np)
        all_metrics.append(sample_metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"PSNR:       {avg_metrics['psnr']:.2f} +/- {avg_metrics['psnr_std']:.2f} dB")
    print(f"SSIM:       {avg_metrics['ssim']:.4f} +/- {avg_metrics['ssim_std']:.4f}")
    print(f"LPIPS:      {avg_metrics['lpips']:.4f} +/- {avg_metrics['lpips_std']:.4f} (lower is better)")
    print(f"Color Hist: {avg_metrics['color_hist']:.4f} +/- {avg_metrics['color_hist_std']:.4f}")

    return avg_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HDR enhancement model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        image_size=args.image_size,
        num_samples=args.num_samples,
    )
