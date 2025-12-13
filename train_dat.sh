#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o train_dat_%j.out
#SBATCH -e train_dat_%j.err
#SBATCH -J dat_retouch

# =============================================================================
# DAT: Dual Aggregation Transformer (ICCV 2023)
# =============================================================================
# Based on: "Dual Aggregation Transformer for Image Super-Resolution"
# Adapted for image retouching task (_src.jpg -> _tar.jpg)
#
# Key features:
# - Dual aggregation: alternating spatial + channel attention
# - Adaptive Interaction Module (AIM)
# - Spatial-Gate Feed-Forward Network (SGFN)
# - ~26M params (similar to Restormer-base to avoid overfitting)
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
echo "DAT Training - Restormer-Size Configuration"
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
# Training Configuration - Restormer-size (~26M params)
# =============================================================================

python3 src/training/train_dat.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_dat \
    --model_size restormer_size \
    --batch_size 2 \
    --num_epochs 200 \
    --image_size 256 \
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
echo "DAT Training Complete"
echo "End time: $(date)"
echo "=========================================="
