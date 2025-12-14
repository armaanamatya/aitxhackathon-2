"""
Quick validation script to test Elite Color Refiner setup
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch

print("=" * 80)
print("ELITE COLOR REFINER SETUP VALIDATION")
print("=" * 80)
print()

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from models.color_refiner import create_elite_color_refiner
    from training.restormer import create_restormer
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create refiner
print("\nTest 2: Creating refiner...")
try:
    refiner = create_elite_color_refiner('medium')
    print(f"✓ Refiner created: {refiner.get_num_params() / 1e6:.2f}M params")
except Exception as e:
    print(f"✗ Refiner creation failed: {e}")
    sys.exit(1)

# Test 3: Load frozen backbone
print("\nTest 3: Loading frozen Restormer896 backbone...")
try:
    backbone_path = 'outputs_restormer_896/checkpoint_best.pt'
    checkpoint = torch.load(backbone_path, map_location='cpu')

    backbone = create_restormer('base')

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Try loading directly first
    try:
        backbone.load_state_dict(state_dict)
    except RuntimeError:
        # Handle key mismatch: add .blocks to keys if needed
        print("  Fixing key mismatch...")
        fixed_state_dict = {}
        for k, v in state_dict.items():
            # Fix encoder/decoder level keys: encoder_level1.0 -> encoder_level1.blocks.0
            if 'encoder_level' in k or 'decoder_level' in k or 'latent' in k or 'refinement' in k:
                parts = k.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    # Insert 'blocks' after level name
                    parts.insert(1, 'blocks')
                    k = '.'.join(parts)
            fixed_state_dict[k] = v
        backbone.load_state_dict(fixed_state_dict)

    num_params = sum(p.numel() for p in backbone.parameters())
    print(f"✓ Backbone loaded: {num_params / 1e6:.2f}M params")

    # Get val loss from checkpoint
    if 'val_loss' in checkpoint:
        print(f"  Checkpoint val_loss: {checkpoint['val_loss']:.4f}")

except Exception as e:
    print(f"✗ Backbone loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test forward pass
print("\nTest 4: Testing forward pass...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Move to device
    backbone = backbone.to(device)
    refiner = refiner.to(device)

    # Freeze backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Create dummy input
    x_input = torch.randn(1, 3, 512, 512).to(device)

    # Backbone forward
    with torch.no_grad():
        x_backbone = backbone(x_input)

    print(f"  Backbone output shape: {x_backbone.shape}")
    print(f"  Backbone output range: [{x_backbone.min():.3f}, {x_backbone.max():.3f}]")

    # Refiner forward
    x_refined = refiner(x_input, x_backbone)

    print(f"  Refiner output shape: {x_refined.shape}")
    print(f"  Refiner output range: [{x_refined.min():.3f}, {x_refined.max():.3f}]")

    print("✓ Forward pass successful")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test data loading
print("\nTest 5: Testing data loading...")
try:
    import json

    train_jsonl = 'data_splits/fold_1/train.jsonl'
    val_jsonl = 'data_splits/fold_1/val.jsonl'

    # Count samples
    with open(train_jsonl) as f:
        train_samples = sum(1 for _ in f)

    with open(val_jsonl) as f:
        val_samples = sum(1 for _ in f)

    print(f"  Train samples: {train_samples}")
    print(f"  Val samples: {val_samples}")
    print("✓ Data files accessible")

except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 6: Check GPU memory
print("\nTest 6: GPU memory check...")
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # Memory after loading models
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved: {mem_reserved:.2f} GB")
        print(f"  Total: {mem_total:.2f} GB")
        print(f"  Available: {mem_total - mem_allocated:.2f} GB")

        # Estimate training memory
        batch_size = 2
        resolution = 896

        # Estimate: 2 batches in GPU (input + backbone output) + gradients
        estimated_mem = mem_allocated + (batch_size * 3 * resolution * resolution * 4 * 4) / 1e9

        print(f"  Estimated training memory (batch_size=2): {estimated_mem:.2f} GB")

        if estimated_mem > mem_total * 0.9:
            print("  ⚠️  Warning: Tight memory! Consider reducing batch size or resolution")
        else:
            print(f"  ✓ Sufficient memory ({mem_total - estimated_mem:.2f} GB margin)")
    else:
        print("  CPU mode (no GPU)")

except Exception as e:
    print(f"✗ GPU check failed: {e}")

print()
print("=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print()
print("Ready to launch training:")
print("  sbatch train_elite_refiner.sh")
print()
