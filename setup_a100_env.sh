#!/bin/bash
# ============================================================================
# Create Fresh Environment with A100 Support
# ============================================================================

set -e

echo "========================================"
echo "Setting up A100-compatible environment"
echo "========================================"
echo ""

# Remove old env if exists
if conda env list | grep -q "controlnet_a100"; then
    echo "🗑️  Removing old controlnet_a100 environment..."
    conda env remove -n controlnet_a100 -y
fi

# Create new environment with Python 3.10
echo "📦 Creating fresh environment (Python 3.10)..."
conda create -n controlnet_a100 python=3.10 -y

# Activate
source ~/miniforge3/etc/profile.d/conda.sh
conda activate controlnet_a100

echo ""
echo "📥 Installing PyTorch with A100 support (CUDA 11.8)..."
# Install PyTorch with sm_80 support (A100)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "📥 Installing other dependencies..."
pip install opencv-python-headless tqdm "numpy<2.0" Pillow gdown

echo ""
echo "✅ Installation complete!"
echo ""

# Test
echo "🧪 Testing installation..."
python3 -c "
import torch
import torchvision
import cv2
import numpy as np

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ TorchVision: {torchvision.__version__}')
print(f'✅ OpenCV: {cv2.__version__}')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')

    # Check A100 compatibility
    cap = torch.cuda.get_device_capability(0)
    print(f'✅ Compute capability: sm_{cap[0]}{cap[1]}')

    if cap >= (8, 0):
        print(f'🎉 A100 (sm_80) SUPPORTED!')
    else:
        print(f'⚠️  Warning: GPU compute capability < 8.0')
"

echo ""
echo "========================================"
echo "✅ Environment ready!"
echo "========================================"
echo ""
echo "🚀 To use this environment:"
echo "   conda activate controlnet_a100"
echo ""
echo "🧪 Then run quick test:"
echo "   bash test_controlnet_restormer_quick.sh"
echo ""
echo "========================================"
