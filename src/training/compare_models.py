"""
Model Comparison Script
=======================

Compares performance of different models:
1. Restormer (trained from scratch)
2. SwinRestormer (pretrained + fine-tuned)
3. Other models (INRetouch, DAT, Mamba, HAT)

Metrics:
- L1 Loss (MAE)
- PSNR
- SSIM
- LPIPS (perceptual)
- Inference time

Outputs:
- Comparison table
- Sample visualizations
- CSV with detailed results
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import csv

# Import models
try:
    from restormer import Restormer, restormer_base
except ImportError:
    restormer_base = None

try:
    from swin_restormer import SwinRestormer, swin_restormer_small
except ImportError:
    swin_restormer_small = None

try:
    from inretouch import INRetouch
except ImportError:
    INRetouch = None

try:
    from dat import DAT, dat_restormer_size
except ImportError:
    dat_restormer_size = None

try:
    from mamba_diffusion import MambaDiffusion, mamba_base
except ImportError:
    mamba_base = None


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute Structural Similarity Index."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(3, 1, window_size, window_size).contiguous().to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=3)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# =============================================================================
# Dataset
# =============================================================================

class EvalDataset(Dataset):
    """Dataset for evaluation."""

    def __init__(self, data_root: str, jsonl_path: str, image_size: int = 256):
        self.data_root = Path(data_root)
        self.image_size = image_size

        self.samples = []
        with open(self.data_root / jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        # Use validation split (last 10%)
        split_idx = int(len(self.samples) * 0.9)
        self.samples = self.samples[split_idx:]
        print(f"Loaded {len(self.samples)} validation samples")

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(self.data_root / path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img  # [0, 1] range

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        src_key = 'src' if 'src' in item else 'source'
        tar_key = 'tar' if 'tar' in item else 'target'

        source = self._load_image(item[src_key])
        target = self._load_image(item[tar_key])

        return {
            'source': source,
            'target': target,
            'name': Path(item[src_key]).stem,
        }


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_name: str, checkpoint_path: str, device: torch.device) -> Optional[torch.nn.Module]:
    """Load a model from checkpoint."""

    if model_name == 'restormer' and restormer_base:
        model = restormer_base()
    elif model_name == 'swin_restormer' and swin_restormer_small:
        model = swin_restormer_small(pretrained=False, freeze_encoder=False)
    elif model_name == 'dat' and dat_restormer_size:
        model = dat_restormer_size()
    elif model_name == 'mamba' and mamba_base:
        model = mamba_base()
    else:
        print(f"Model {model_name} not available")
        return None

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lpips_fn: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate a model on the dataset."""

    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_time = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        # Convert to [-1, 1] for model input
        source_model = source * 2 - 1

        # Measure inference time
        start_time = time.time()
        output = model(source_model)
        if isinstance(output, dict):
            output = output['output']
        torch.cuda.synchronize()
        total_time += time.time() - start_time

        # Convert output to [0, 1]
        output = output * 0.5 + 0.5
        output = output.clamp(0, 1)

        # Compute metrics
        for i in range(source.size(0)):
            pred = output[i:i+1]
            tgt = target[i:i+1]

            total_l1 += F.l1_loss(pred, tgt).item()
            total_psnr += compute_psnr(pred, tgt)
            total_ssim += compute_ssim(pred, tgt)

            if lpips_fn is not None:
                # LPIPS expects [-1, 1]
                total_lpips += lpips_fn(pred * 2 - 1, tgt * 2 - 1).item()

            num_samples += 1

    return {
        'l1': total_l1 / num_samples,
        'psnr': total_psnr / num_samples,
        'ssim': total_ssim / num_samples,
        'lpips': total_lpips / num_samples if lpips_fn else 0.0,
        'inference_time_ms': (total_time / num_samples) * 1000,
    }


# =============================================================================
# Visualization
# =============================================================================

def save_comparison_images(
    models: Dict[str, torch.nn.Module],
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device,
    num_samples: int = 5,
):
    """Save side-by-side comparison images."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break

            source = batch['source'].to(device)
            target = batch['target'].to(device)
            name = batch['name'][0]

            source_model = source * 2 - 1

            images = [
                ('Source', source[0]),
                ('Target', target[0]),
            ]

            for model_name, model in models.items():
                if model is not None:
                    output = model(source_model)
                    if isinstance(output, dict):
                        output = output['output']
                    output = (output * 0.5 + 0.5).clamp(0, 1)
                    images.append((model_name, output[0]))

            # Create comparison image
            num_images = len(images)
            fig_width = 256 * num_images
            combined = np.zeros((256, fig_width, 3), dtype=np.uint8)

            for i, (label, img) in enumerate(images):
                img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                combined[:, i*256:(i+1)*256] = img_np

            Image.fromarray(combined).save(output_dir / f'{name}_comparison.png')

    print(f"Saved comparison images to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare Models')
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--jsonl_path', type=str, default='train.jsonl')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='comparison_results')

    # Checkpoint paths
    parser.add_argument('--restormer_ckpt', type=str, default='outputs_restormer/checkpoint_best.pt')
    parser.add_argument('--swin_restormer_ckpt', type=str, default='outputs_swin_restormer/checkpoint_best.pt')
    parser.add_argument('--inretouch_ckpt', type=str, default='outputs_inretouch/checkpoint_best.pt')
    parser.add_argument('--dat_ckpt', type=str, default='outputs_dat/checkpoint_best.pt')
    parser.add_argument('--mamba_ckpt', type=str, default='outputs_mamba/checkpoint_best.pt')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = EvalDataset(args.data_root, args.jsonl_path, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load LPIPS
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
    except:
        lpips_fn = None
        print("LPIPS not available")

    # Load models
    models = {}

    print("\nLoading models...")

    # Restormer (scratch)
    if restormer_base and os.path.exists(args.restormer_ckpt):
        models['Restormer (scratch)'] = load_model('restormer', args.restormer_ckpt, device)

    # SwinRestormer (pretrained + fine-tuned)
    if swin_restormer_small and os.path.exists(args.swin_restormer_ckpt):
        models['SwinRestormer (fine-tuned)'] = load_model('swin_restormer', args.swin_restormer_ckpt, device)

    # DAT
    if dat_restormer_size and os.path.exists(args.dat_ckpt):
        models['DAT'] = load_model('dat', args.dat_ckpt, device)

    # Mamba
    if mamba_base and os.path.exists(args.mamba_ckpt):
        models['MambaDiffusion'] = load_model('mamba', args.mamba_ckpt, device)

    if not models:
        print("No models found! Please provide checkpoint paths.")
        return

    # Evaluate models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    results = {}
    for model_name, model in models.items():
        if model is not None:
            print(f"\nEvaluating {model_name}...")
            metrics = evaluate_model(model, dataloader, device, lpips_fn)
            results[model_name] = metrics

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'L1 ↓':<10} {'PSNR ↑':<10} {'SSIM ↑':<10} {'LPIPS ↓':<10} {'Time (ms)':<10}")
    print("-" * 80)

    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['l1']:.4f}    {metrics['psnr']:.2f}     {metrics['ssim']:.4f}    {metrics['lpips']:.4f}    {metrics['inference_time_ms']:.1f}")

    print("=" * 80)

    # Save results to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'comparison_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'L1', 'PSNR', 'SSIM', 'LPIPS', 'Inference_Time_ms'])
        for model_name, metrics in results.items():
            writer.writerow([
                model_name,
                f"{metrics['l1']:.4f}",
                f"{metrics['psnr']:.2f}",
                f"{metrics['ssim']:.4f}",
                f"{metrics['lpips']:.4f}",
                f"{metrics['inference_time_ms']:.1f}",
            ])

    print(f"\nResults saved to {output_dir / 'comparison_results.csv'}")

    # Save comparison images
    save_comparison_images(models, dataloader, output_dir / 'images', device)

    # Determine winner
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['l1'])
        print(f"\n🏆 Best model by L1: {best_model[0]} (L1: {best_model[1]['l1']:.4f})")


if __name__ == '__main__':
    main()
