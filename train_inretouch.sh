#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o train_inretouch_%j.out
#SBATCH -e train_inretouch_%j.err
#SBATCH -J inretouch

# =============================================================================
# INRetouch: Context-Aware Retouching Network
# =============================================================================
# MAXIMUM QUALITY configuration for _src.jpg -> _tar.jpg matching
#
# Architecture:
# - Context-aware Implicit Neural Representation
# - Multi-scale context encoder (U-Net style)
# - FiLM modulation for adaptive editing
# - Global color transform + 3D LUT
# - Multi-scale discriminator (GAN training)
#
# Loss functions (all enabled for max quality):
# - Charbonnier (L1) - pixel accuracy
# - SSIM - structural similarity
# - VGG Perceptual - feature matching
# - LPIPS - perceptual quality
# - LAB Color - color accuracy
# - Histogram - color distribution
# - Gradient - edge preservation
# - GAN - sharpness
# =============================================================================

echo "=========================================="
echo "INRetouch Training - Maximum Quality Mode"
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
# Training Configuration - MAXIMUM QUALITY
# =============================================================================
# Model: large (20M+ params) for maximum capacity
# Losses: All quality losses enabled with optimized weights
# GAN: Enabled for sharper outputs
# EMA: High decay (0.9999) for stable results
# =============================================================================

python3 src/training/train_inretouch.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_inretouch \
    --model_size large \
    --batch_size 2 \
    --num_epochs 300 \
    --image_size 256 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --lambda_l1 10.0 \
    --lambda_ssim 1.0 \
    --lambda_perceptual 1.0 \
    --lambda_lpips 1.0 \
    --lambda_lab 1.0 \
    --lambda_hist 0.5 \
    --lambda_gradient 0.5 \
    --lambda_color 0.5 \
    --use_amp \
    --use_ema \
    --ema_decay 0.9999 \
    --grad_accum_steps 4 \
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
echo "INRetouch Training Complete"
echo "End time: $(date)"
echo "=========================================="
