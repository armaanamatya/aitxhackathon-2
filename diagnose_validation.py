#!/usr/bin/env python3
"""
Diagnostic script to check why validation loss is stuck
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from restormer import Restormer

print("=" * 70)
print("VALIDATION DIAGNOSTIC")
print("=" * 70)

# Load checkpoint
ckpt_path = 'outputs_restormer_optimal_v2/checkpoint_best.pt'
print(f"\nLoading checkpoint: {ckpt_path}")

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Val loss: {ckpt['val_loss']}")
    print(f"  Best val loss: {ckpt['best_val_loss']}")

    # Check model weights
    state_dict = ckpt['model_state_dict']
    print(f"\n  Model has {len(state_dict)} parameters")

    # Sample a few parameters
    param_stats = []
    for i, (name, param) in enumerate(list(state_dict.items())[:10]):
        stats = {
            'name': name,
            'shape': list(param.shape),
            'mean': float(param.mean()),
            'std': float(param.std()),
            'min': float(param.min()),
            'max': float(param.max())
        }
        param_stats.append(stats)

    print("\n  Sample parameter statistics:")
    for stats in param_stats[:5]:
        print(f"    {stats['name'][:40]:40s} mean={stats['mean']:8.5f} std={stats['std']:8.5f}")

    # Create model and load weights
    print("\n\nCreating model...")
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    )

    model.load_state_dict(state_dict)
    print("✓ Model weights loaded")

    # Test with dummy input
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output1 = model(dummy_input)
        output2 = model(dummy_input)

    # Check if outputs are identical
    diff = (output1 - output2).abs().max()
    print(f"\nDeterminism test (same input twice):")
    print(f"  Max difference: {diff:.10f}")
    if diff < 1e-6:
        print("  ✓ Model is deterministic")
    else:
        print("  ✗ Model is NON-deterministic!")

    # Check if model is actually changing outputs
    dummy_input2 = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output3 = model(dummy_input2)

    diff2 = (output1 - output3).abs().max()
    print(f"\nVariation test (different inputs):")
    print(f"  Max difference: {diff2:.10f}")
    if diff2 > 0.01:
        print("  ✓ Model produces different outputs for different inputs")
    else:
        print("  ✗ Model outputs are suspiciously similar!")
        print(f"    Output1 mean: {output1.mean():.6f}, std: {output1.std():.6f}")
        print(f"    Output3 mean: {output3.mean():.6f}, std: {output3.std():.6f}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
