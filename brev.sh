#!/bin/bash
set -e

echo "=========================================="
echo "  Brev Setup: Restormer 3297x2201 Training"
echo "=========================================="

# =============================================================================
# 1. System Setup
# =============================================================================
echo "[1/7] Installing system packages..."
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

# =============================================================================
# 2. Python Setup
# =============================================================================
echo "[2/7] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# =============================================================================
# 3. PyTorch with CUDA
# =============================================================================
echo "[3/7] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# =============================================================================
# 4. Training Dependencies
# =============================================================================
echo "[4/7] Installing training dependencies..."
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

# =============================================================================
# 5. Clone Repository
# =============================================================================
echo "[5/7] Cloning repository..."
cd ~
if [ ! -d "aitxhackathon-2" ]; then
    git clone -b shell https://github.com/armaanamatya/aitxhackathon-2.git
fi
cd aitxhackathon-2

# =============================================================================
# 6. Download Dataset from Dropbox
# =============================================================================
echo "[6/7] Downloading dataset from Dropbox..."
if [ ! -d "images" ]; then
    wget --continue --timeout=0 -O dataset.zip "https://www.dropbox.com/scl/fo/fvr1xwtp89n3n7zewtbk1/AOIpVeRP3gs0x-sMx4dFPnE?rlkey=sdxwning0n70dgyuijmeqep6e&st=axxovbtt&dl=1"
    unzip dataset.zip
    rm dataset.zip
    echo "Dataset downloaded and extracted."
else
    echo "Dataset already exists, skipping download."
fi

# =============================================================================
# 7. Verify Setup
# =============================================================================
echo "[7/7] Verifying setup..."

# Check GPU
nvidia-smi
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

# Check data splits
echo ""
echo "Data splits:"
wc -l data_splits/proper_split/*.jsonl

# Check essential files
echo ""
echo "Essential files:"
ls -la train_restormer_512_combined_loss.py
ls -la finetune_encoder.py
ls -la src/training/restormer.py

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo ""
echo "  tmux new -s train"
echo ""
echo "  python3 train_restormer_512_combined_loss.py \\"
echo "      --train_jsonl data_splits/proper_split/train.jsonl \\"
echo "      --val_jsonl data_splits/proper_split/val.jsonl \\"
echo "      --output_dir outputs_restormer_3297 \\"
echo "      --resolution 3297 \\"
echo "      --batch_size 2 \\"
echo "      --lr 2e-4 \\"
echo "      --warmup_epochs 5 \\"
echo "      --patience 15 \\"
echo "      --epochs 100"
echo ""
echo "After training completes, finetune encoder:"
echo ""
echo "  python3 finetune_encoder.py \\"
echo "      --checkpoint outputs_restormer_3297/checkpoint_best.pt \\"
echo "      --resolution 3297 \\"
echo "      --batch_size 1 \\"
echo "      --epochs 50"
echo ""
echo "=========================================="
