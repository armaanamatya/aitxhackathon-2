#!/bin/bash
# Fix ControlNet dependencies by removing conflicts

echo "=========================================="
echo "FIXING CONTROLNET DEPENDENCIES"
echo "=========================================="

# Uninstall problematic packages
echo "Removing torch_xla (not needed)..."
pip uninstall -y torch_xla torch-xla 2>/dev/null

echo "Removing conflicting packages..."
pip uninstall -y accelerate diffusers transformers 2>/dev/null

echo ""
echo "Installing clean versions..."
pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers==4.40.0 \
    accelerate==0.29.3 \
    peft==0.10.0

echo ""
echo "Verifying installation..."
python3 -c "from diffusers import AutoencoderKL, ControlNetModel; print('✅ diffusers OK')"
python3 -c "from transformers import CLIPTextModel; print('✅ transformers OK')"
python3 -c "from accelerate import Accelerator; print('✅ accelerate OK')"

echo "=========================================="
echo "✅ DEPENDENCIES FIXED"
echo "=========================================="
