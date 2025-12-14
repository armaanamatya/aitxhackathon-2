#!/usr/bin/env python3
"""
Ensemble Inference: Restormer + DarkIR + ControlNet
====================================================
Combines predictions from multiple models for best generalization.

Usage:
    python3 inference_ensemble.py \
        --input test_image.jpg \
        --output enhanced.jpg \
        --restormer_path outputs_restormer/checkpoint_best.pt \
        --darkir_path outputs_darkir_384_m_cv/fold_1/checkpoint_best.pt \
        --ensemble_mode average
"""

import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import sys

# Add model paths
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
sys.path.insert(0, str(Path(__file__).parent / 'DarkIR'))

from restormer import Restormer
from archs.DarkIR import DarkIR


class EnsembleModel:
    """Ensemble of multiple models for robust predictions."""

    def __init__(self, models: list, weights: list = None, mode: str = 'average'):
        """
        Args:
            models: List of model instances
            weights: Optional weights for weighted average (must sum to 1)
            mode: 'average', 'weighted', or 'median'
        """
        self.models = models
        self.mode = mode

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights

        # Set all models to eval
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble prediction.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Ensemble prediction [B, C, H, W]
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # Stack predictions
        predictions = torch.stack(predictions)  # [N_models, B, C, H, W]

        # Ensemble
        if self.mode == 'average':
            output = predictions.mean(dim=0)
        elif self.mode == 'weighted':
            weights = torch.tensor(self.weights, device=x.device).view(-1, 1, 1, 1, 1)
            output = (predictions * weights).sum(dim=0)
        elif self.mode == 'median':
            output = predictions.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return output


def load_restormer(checkpoint_path: str, device: str = 'cuda'):
    """Load Restormer model."""
    model = Restormer(
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        use_checkpointing=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_darkir(checkpoint_path: str, model_size: str = 'm', device: str = 'cuda'):
    """Load DarkIR model."""
    width = 32 if model_size == 'm' else 64
    model = DarkIR(
        img_channel=3,
        width=width,
        middle_blk_num_enc=2,
        middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3],
        dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser(description="Ensemble inference")
    parser.add_argument('--input', type=str, required=True, help='Input image')
    parser.add_argument('--output', type=str, required=True, help='Output image')
    parser.add_argument('--restormer_path', type=str, default=None, help='Restormer checkpoint')
    parser.add_argument('--darkir_path', type=str, default=None, help='DarkIR checkpoint')
    parser.add_argument('--darkir_size', type=str, default='m', choices=['m', 'l'])
    parser.add_argument('--ensemble_mode', type=str, default='average',
                        choices=['average', 'weighted', 'median'])
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Weights for weighted ensemble (must sum to 1)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("="*80)
    print("ENSEMBLE INFERENCE")
    print("="*80)

    # Load models
    models = []
    model_names = []

    if args.restormer_path:
        print(f"\n📥 Loading Restormer from {args.restormer_path}")
        models.append(load_restormer(args.restormer_path, args.device))
        model_names.append("Restormer")

    if args.darkir_path:
        print(f"📥 Loading DarkIR-{args.darkir_size} from {args.darkir_path}")
        models.append(load_darkir(args.darkir_path, args.darkir_size, args.device))
        model_names.append(f"DarkIR-{args.darkir_size}")

    if len(models) == 0:
        raise ValueError("Must provide at least one model checkpoint!")

    print(f"\n✅ Loaded {len(models)} models: {', '.join(model_names)}")

    # Create ensemble
    ensemble = EnsembleModel(models, weights=args.weights, mode=args.ensemble_mode)
    print(f"📊 Ensemble mode: {args.ensemble_mode}")
    if args.weights:
        print(f"⚖️  Weights: {args.weights}")

    # Load image
    print(f"\n📂 Loading input: {args.input}")
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # To tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(args.device)

    # Inference
    print(f"🚀 Running ensemble inference...")
    output = ensemble.predict(img_tensor)

    # To numpy
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Save
    cv2.imwrite(args.output, output)
    print(f"✅ Saved output: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
