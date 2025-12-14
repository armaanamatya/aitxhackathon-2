#!/bin/bash
# Setup clean virtual environment for ControlNet training

echo "=========================================="
echo "SETTING UP CONTROLNET ENVIRONMENT"
echo "=========================================="

VENV_NAME="controlnet_venv"

# Remove old venv if exists
if [ -d "$VENV_NAME" ]; then
    echo "Removing old venv..."
    rm -rf "$VENV_NAME"
fi

# Create new venv
echo "Creating virtual environment..."
/cm/local/apps/python39/bin/python3 -m venv "$VENV_NAME"

# Activate
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing ControlNet dependencies..."
pip install -r requirements_controlnet.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
print('✅ All imports successful!')
print('✅ Environment ready for ControlNet training')
"

echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To activate this environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To train ControlNet:"
echo "  sbatch train_controlnet_venv.sh"
echo "=========================================="
