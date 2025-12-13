#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o setup_env_%j.out
#SBATCH -e setup_env_%j.err
#SBATCH -J setup_env

# Setup fresh venv for AutoHDR training on GPU nodes

echo "=========================================="
echo "Setting up AutoHDR Virtual Environment"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load Python and CUDA modules
module load python39
module load cuda11.8/toolkit/11.8.0

# Check GPU
nvidia-smi

# Remove old venv if exists
if [ -d "venv_gpu" ]; then
    echo "Removing old venv_gpu..."
    rm -rf venv_gpu
fi

# Create fresh venv
echo "Creating fresh virtual environment..."
python3 -m venv venv_gpu

# Activate venv
source venv_gpu/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install lpips pillow tqdm numpy scikit-image pytorch-msssim

# Optional: Diffusers for SD-LoRA (comment out if not needed)
# pip install diffusers transformers peft accelerate

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "GPU check skipped"
python3 -c "import lpips; print('LPIPS: OK')"
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Venv location: $(pwd)/venv_gpu"
echo ""
echo "To use in SLURM scripts:"
echo "  module load python39"
echo "  module load cuda11.8/toolkit/11.8.0"
echo "  source venv_gpu/bin/activate"
