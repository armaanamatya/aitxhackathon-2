#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_retinexmamba_%j.out
#SBATCH -e train_retinexmamba_%j.err
#SBATCH -J autohdr_retinexmamba

# RetinexMamba Training for HDR Enhancement
# Features:
# - Retinex theory: decomposes image into illumination + reflectance
# - Mamba (State Space Models): efficient global context with O(n) complexity
# - Illumination-guided attention for adaptive enhancement
# - Using SMALL model (~5M params) to avoid overfitting on 550 images

echo "=========================================="
echo "RetinexMamba Training"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

# Print Python and CUDA info
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify dependencies
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --quiet
python3 -c "import einops; print('einops OK')" || pip install einops --quiet

# Try to install mamba-ssm (optional - will fall back to pure PyTorch if unavailable)
python3 -c "from mamba_ssm import Mamba; print('Mamba SSM OK')" 2>/dev/null || echo "Note: Using pure PyTorch Mamba implementation (slower but compatible)"

# Run RetinexMamba training
# Using SMALL model for 550 images to prevent overfitting
# Loss weights optimized for color grading task:
# - High Charbonnier (1.0) for pixel accuracy
# - SSIM (0.15) for structural similarity
# - FFT (0.05) for frequency domain matching
# - Perceptual (0.1) for feature-level similarity
# - LPIPS (0.15) for perceptual quality
# - LAB (0.15) for color accuracy
# - Histogram (0.05) for color distribution
# - Illumination (0.01) for Retinex consistency
python3 src/training/train_retinexmamba.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_retinexmamba \
    --model_size small \
    --batch_size 2 \
    --num_epochs 200 \
    --image_size 512 \
    --lr 3e-4 \
    --lambda_charbonnier 1.0 \
    --lambda_ssim 0.15 \
    --lambda_fft 0.05 \
    --lambda_perceptual 0.1 \
    --lambda_lpips 0.15 \
    --lambda_lab 0.15 \
    --lambda_hist 0.05 \
    --lambda_illum 0.01 \
    --ema_decay 0.999 \
    --save_interval 10 \
    --sample_interval 5 \
    --num_workers 8

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "End time: $(date)"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

echo "=========================================="
echo "RetinexMamba Training Complete"
echo "End time: $(date)"
echo "=========================================="
