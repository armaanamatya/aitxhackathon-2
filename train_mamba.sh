#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o train_mamba_%j.out
#SBATCH -e train_mamba_%j.err
#SBATCH -J mamba_retouch

# =============================================================================
# MambaDiffusion: State-of-the-Art Hybrid Model for Image Retouching
# =============================================================================
# Based on latest 2024-2025 research:
# - MambaIRv2 (CVPR 2025): State Space Models with linear complexity
# - DAT (ICCV 2023): Dual Aggregation Transformer
# - Diff-Mamba: Mamba + Diffusion hybrid
#
# Architecture:
# - Vision State Space (VSS) blocks with bidirectional selective scan
# - Local Enhancement for preserving local pixel details
# - Channel Attention for reducing redundancy
# - DAT-style dual aggregation (spatial + channel attention)
#
# Loss functions:
# - Charbonnier (L1) - pixel accuracy
# - SSIM - structural similarity
# - VGG Perceptual - feature matching
# - LPIPS - perceptual quality
# - LAB Color - color accuracy
# - Histogram - color distribution
# - Gradient - edge preservation
# =============================================================================

echo "=========================================="
echo "MambaDiffusion Training - Maximum Quality"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate any virtual environment
unset VIRTUAL_ENV
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "autohdr_venv" | tr '\n' ':' | sed 's/:$//')

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

# Print Python info
echo "Python: $(which python3)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify dependencies
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --user --quiet
python3 -c "import einops; print('einops OK')" || pip install einops --user --quiet
python3 -c "from PIL import Image; print('PIL OK')"

# =============================================================================
# Training Configuration - MAXIMUM QUALITY (Mamba-Large)
# =============================================================================
# Model: large (~80M params) - Maximum capacity with VSS + DAT
# Losses: All quality losses enabled
# EMA: Enabled for stable outputs
# =============================================================================

python3 src/training/train_mamba.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_mamba \
    --model_size large \
    --batch_size 2 \
    --num_epochs 200 \
    --image_size 128 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --lambda_char 10.0 \
    --lambda_ssim 1.0 \
    --lambda_perceptual 1.0 \
    --lambda_lpips 1.0 \
    --lambda_lab 1.0 \
    --lambda_hist 0.5 \
    --lambda_gradient 0.5 \
    --use_amp \
    --use_ema \
    --ema_decay 0.999 \
    --grad_accum_steps 2 \
    --num_workers 8 \
    --save_interval 10 \
    --sample_interval 5

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "End time: $(date)"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

echo "=========================================="
echo "MambaDiffusion Training Complete"
echo "End time: $(date)"
echo "=========================================="
