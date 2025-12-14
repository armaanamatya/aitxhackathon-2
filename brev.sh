#!/bin/bash
set -e

echo "=========================================="
echo "  Brev Setup: Restormer HDR Training"
echo "=========================================="

# System updates and essentials
echo "[1/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    git \
    tmux \
    htop \
    nvtop \
    unzip \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Upgrade pip
echo "[2/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
echo "[3/6] Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
echo "[4/6] Installing training dependencies..."
pip install \
    numpy \
    Pillow \
    opencv-python \
    scikit-image \
    tqdm \
    lpips \
    pytorch-msssim \
    kornia \
    wandb \
    tensorboard \
    accelerate

# Clone repository (skip if already in repo)
echo "[5/6] Setting up repository..."
cd ~
if [ ! -d "autohdr-real-estate-577" ]; then
    git clone https://github.com/sww35/autohdr-real-estate-577.git
fi
cd autohdr-real-estate-577

# Verify GPU setup
echo "[6/6] Verifying GPU..."
nvidia-smi
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
EOF

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload your dataset:"
echo "     scp -r /local/path/to/dataset brev-<instance>:~/autohdr-real-estate-577/"
echo ""
echo "  2. Start training:"
echo "     cd ~/autohdr-real-estate-577"
echo "     python src/training/train_restormer_large.py \\"
echo "       --data_root . \\"
echo "       --jsonl_path train.jsonl \\"
echo "       --image_size 1024 \\"
echo "       --batch_size 2 \\"
echo "       --model_size large"
echo ""
echo "  3. Use tmux for persistent sessions:"
echo "     tmux new -s train"
echo ""
