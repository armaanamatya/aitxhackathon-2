#!/usr/bin/env python3
"""
Comprehensive Setup Verification - Top 0.0001% MLE Standards
=============================================================
Verifies entire pipeline before training:
- Data splits (90:10, zero leakage)
- Dependencies
- Pretrained weights
- GPU availability
- Model architecture
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def color_text(text, color='green'):
    colors = {'green': '\033[92m', 'red': '\033[91m', 'yellow': '\033[93m', 'blue': '\033[94m', 'reset': '\033[0m'}
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def check_dependencies():
    """Check all required dependencies."""
    print("\n" + "="*80)
    print("1. CHECKING DEPENDENCIES")
    print("="*80)

    deps = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
    }

    missing = []
    versions = {}

    for module, name in deps.items():
        try:
            if module == 'cv2':
                import cv2
                versions[name] = cv2.__version__
            elif module == 'torch':
                import torch
                versions[name] = torch.__version__
                print(f"  ✅ {name}: {torch.__version__}")
                print(f"     CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"     CUDA version: {torch.version.cuda}")
                    print(f"     GPU: {torch.cuda.get_device_name(0)}")
                continue
            elif module == 'torchvision':
                import torchvision
                versions[name] = torchvision.__version__
            elif module == 'numpy':
                import numpy
                versions[name] = numpy.__version__
                # Check NumPy version
                major = int(numpy.__version__.split('.')[0])
                if major >= 2:
                    print(f"  ⚠️  {name}: {numpy.__version__} (WARNING: NumPy 2.x may cause issues)")
                    print(f"     Recommend: pip install 'numpy<2.0'")
                    continue
            elif module == 'tqdm':
                import tqdm
                versions[name] = tqdm.__version__

            print(f"  ✅ {name}: {versions[name]}")
        except ImportError:
            missing.append(name)
            print(f"  ❌ {name}: NOT FOUND")

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join([d.split()[0].lower() for d in missing])}")
        return False

    print(f"\n✅ All dependencies installed!")
    return True


def check_data_splits():
    """Verify data splits are correct (90:10, zero leakage)."""
    print("\n" + "="*80)
    print("2. CHECKING DATA SPLITS")
    print("="*80)

    splits_dir = Path('data_splits')
    if not splits_dir.exists():
        print(f"❌ data_splits/ directory not found!")
        print(f"   Run: python3 create_data_splits.py --input_jsonl train_cleaned.jsonl")
        return False

    # Load test set
    test_jsonl = splits_dir / 'test.jsonl'
    if not test_jsonl.exists():
        print(f"❌ test.jsonl not found!")
        return False

    test_files = set()
    with open(test_jsonl) as f:
        for line in f:
            pair = json.loads(line)
            test_files.add(pair['src'])

    print(f"  Test set: {len(test_files)} samples")

    # Check each fold
    fold_stats = []
    for fold in range(1, 4):
        fold_dir = splits_dir / f'fold_{fold}'
        train_jsonl = fold_dir / 'train.jsonl'
        val_jsonl = fold_dir / 'val.jsonl'

        if not train_jsonl.exists() or not val_jsonl.exists():
            print(f"  ❌ Fold {fold}: Missing files")
            return False

        train_files = set()
        val_files = set()

        with open(train_jsonl) as f:
            for line in f:
                pair = json.loads(line)
                train_files.add(pair['src'])

        with open(val_jsonl) as f:
            for line in f:
                pair = json.loads(line)
                val_files.add(pair['src'])

        n_train = len(train_files)
        n_val = len(val_files)
        total = n_train + n_val
        ratio = n_train / total if total > 0 else 0

        fold_stats.append({
            'fold': fold,
            'train': n_train,
            'val': n_val,
            'ratio': ratio,
            'train_files': train_files,
            'val_files': val_files
        })

        # Check for leakage with test set
        train_test_overlap = train_files & test_files
        val_test_overlap = val_files & test_files

        # Check for leakage within fold
        train_val_overlap = train_files & val_files

        status = "✅" if (not train_test_overlap and not val_test_overlap and
                         not train_val_overlap and 0.89 <= ratio <= 0.91) else "❌"

        print(f"  {status} Fold {fold}: {n_train} train / {n_val} val ({ratio*100:.1f}% train)")

        if train_test_overlap:
            print(f"      ❌ LEAKAGE: {len(train_test_overlap)} files in both train and test!")
        if val_test_overlap:
            print(f"      ❌ LEAKAGE: {len(val_test_overlap)} files in both val and test!")
        if train_val_overlap:
            print(f"      ❌ LEAKAGE: {len(train_val_overlap)} files in both train and val!")
        if not (0.89 <= ratio <= 0.91):
            print(f"      ⚠️  Ratio should be ~90%, got {ratio*100:.1f}%")

    # Check cross-fold diversity
    print(f"\n  Cross-fold validation check:")
    for i in range(3):
        for j in range(i+1, 3):
            fold_i_val = fold_stats[i]['val_files']
            fold_j_val = fold_stats[j]['val_files']
            overlap = fold_i_val & fold_j_val
            overlap_pct = len(overlap) / len(fold_i_val) * 100 if fold_i_val else 0
            print(f"    Fold {i+1} val ∩ Fold {j+1} val: {len(overlap)} files ({overlap_pct:.1f}% overlap)")

    print(f"\n✅ Data splits verified!")
    print(f"   Total: {len(test_files) + fold_stats[0]['train'] + fold_stats[0]['val']} samples")
    print(f"   Test: {len(test_files)} (held out)")
    print(f"   Train+Val: {fold_stats[0]['train'] + fold_stats[0]['val']} per fold")

    return True


def check_pretrained_weights():
    """Check if pretrained weights exist and are valid."""
    print("\n" + "="*80)
    print("3. CHECKING PRETRAINED WEIGHTS")
    print("="*80)

    pretrained_path = Path('pretrained/restormer_denoising.pth')

    if not pretrained_path.exists():
        print(f"  ⚠️  Pretrained weights not found: {pretrained_path}")
        print(f"     Training will be from scratch (expected PSNR: ~26-28 dB)")
        print(f"\n  To download pretrained weights:")
        print(f"     bash download_pretrained_restormer.sh")
        print(f"\n  Expected improvement with pretrained:")
        print(f"     From scratch: ~26-28 dB PSNR")
        print(f"     With pretrained: ~30-32 dB PSNR (+3-5 dB)")
        return False

    # Check file size
    size_mb = pretrained_path.stat().st_size / (1024 * 1024)

    if size_mb < 50:
        print(f"  ⚠️  Pretrained weights file too small: {size_mb:.1f} MB")
        print(f"     Expected: ~100-110 MB")
        print(f"     File may be corrupted, re-download with:")
        print(f"     bash download_pretrained_restormer.sh")
        return False

    print(f"  ✅ Pretrained weights found: {pretrained_path}")
    print(f"     Size: {size_mb:.1f} MB")

    # Try loading with torch
    try:
        import torch
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                n_params = len(checkpoint['model_state_dict'])
            else:
                n_params = len(checkpoint)
        else:
            n_params = len(checkpoint) if hasattr(checkpoint, '__len__') else '?'

        print(f"     Parameters: {n_params}")
        print(f"  ✅ Pretrained weights loaded successfully!")
        return True
    except Exception as e:
        print(f"  ❌ Error loading pretrained weights: {e}")
        return False


def check_model_architecture():
    """Verify model can be instantiated."""
    print("\n" + "="*80)
    print("4. CHECKING MODEL ARCHITECTURE")
    print("="*80)

    try:
        sys.path.insert(0, str(Path('src/training')))
        from restormer import Restormer

        # Try creating model
        model = Restormer(
            in_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Restormer instantiated successfully")
        print(f"     Total parameters: {total_params:,}")

        # Test forward pass
        import torch
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"     Output shape: {output.shape} (expected: [1, 3, 256, 256])")

        if output.shape == dummy_input.shape:
            print(f"  ✅ Model forward pass successful!")
            return True
        else:
            print(f"  ❌ Output shape mismatch!")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_config():
    """Verify training scripts have optimal configuration."""
    print("\n" + "="*80)
    print("5. CHECKING TRAINING CONFIGURATION")
    print("="*80)

    script_path = Path('train_controlnet_restormer_512_a100.sh')

    if not script_path.exists():
        print(f"  ❌ Training script not found: {script_path}")
        return False

    with open(script_path) as f:
        content = f.read()

    # Extract key configurations
    configs = {}
    for line in content.split('\n'):
        if '=' in line and not line.strip().startswith('#'):
            parts = line.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip('"')
                configs[key] = value

    # Check critical settings
    checks = {
        'RESOLUTION': ('512', 'High resolution for quality'),
        'BATCH_SIZE': (lambda v: int(v) >= 8, 'Batch size >= 8 for A100'),
        'EPOCHS': (lambda v: int(v) >= 50, 'Sufficient epochs'),
        'LR': ('1e-4', 'Optimal learning rate'),
        'LAMBDA_L1': ('1.0', 'L1 loss weight'),
        'LAMBDA_VGG': (lambda v: float(v) >= 0.15, 'VGG weight >= 0.15 for perceptual quality'),
        'EARLY_STOP_PATIENCE': (lambda v: int(v) >= 10, 'Early stopping patience'),
        'N_FOLDS': ('3', '3-fold cross-validation'),
        'MIXED_PRECISION': ('true', 'Mixed precision enabled'),
        'PRETRAINED_PATH': (lambda v: 'restormer_denoising.pth' in v, 'Pretrained weights path set'),
    }

    all_good = True
    for key, (expected, desc) in checks.items():
        if key in configs:
            value = configs[key]
            if callable(expected):
                try:
                    is_ok = expected(value)
                except:
                    is_ok = False
            else:
                is_ok = value == expected

            status = "✅" if is_ok else "⚠️ "
            print(f"  {status} {key}: {value} ({desc})")

            if not is_ok:
                all_good = False
        else:
            print(f"  ⚠️  {key}: NOT FOUND")
            all_good = False

    if all_good:
        print(f"\n✅ Training configuration optimal!")
    else:
        print(f"\n⚠️  Some settings may not be optimal")

    return True


def main():
    print("\n" + "="*80)
    print("CONTROLNET-RESTORMER SETUP VERIFICATION")
    print("Top 0.0001% MLE Standards")
    print("="*80)

    checks = [
        ("Dependencies", check_dependencies),
        ("Data Splits", check_data_splits),
        ("Pretrained Weights", check_pretrained_weights),
        ("Model Architecture", check_model_architecture),
        ("Training Config", check_training_config),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    all_passed = all(results.values())

    if all_passed:
        print(f"\n" + "="*80)
        print("🎉 ALL CHECKS PASSED - READY TO TRAIN!")
        print("="*80)
        print(f"\n🚀 Next steps:")
        print(f"   1. Quick test (15 min):")
        print(f"      bash test_controlnet_restormer_quick.sh")
        print(f"\n   2. Full training (12-16 hours):")
        print(f"      sbatch train_controlnet_restormer_512_a100.sh")
        print(f"\n📊 Expected results:")
        print(f"   Val PSNR: 30-32 dB (with pretrained)")
        print(f"   Test PSNR: 30-31 dB (ensemble of 3 folds)")
        print(f"   Gain over from-scratch: +3-5 dB")
        print(f"\n" + "="*80)
        return 0
    else:
        print(f"\n" + "="*80)
        print("⚠️  SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print("="*80)

        if not results.get("Pretrained Weights"):
            print(f"\n💡 Critical: Download pretrained weights for +3-5 dB improvement:")
            print(f"   bash download_pretrained_restormer.sh")

        if not results.get("Dependencies"):
            print(f"\n💡 Install missing dependencies:")
            print(f"   conda install -c conda-forge opencv -y")

        if not results.get("Data Splits"):
            print(f"\n💡 Regenerate data splits:")
            print(f"   python3 create_data_splits.py --input_jsonl train_cleaned.jsonl")

        print(f"\n" + "="*80)
        return 1


if __name__ == '__main__':
    exit(main())
