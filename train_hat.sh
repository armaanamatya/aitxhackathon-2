#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_hat_%j.out
#SBATCH -e train_hat_%j.err
#SBATCH -J autohdr_hat

# HAT (Hybrid Attention Transformer) Training
# Architecture combines:
# - Window-based Self-Attention (like Swin)
# - Channel Attention
# - Overlapping Cross-Attention (key innovation)
# Currently SOTA for image restoration quality

echo "=========================================="
echo "HAT-Base Training (256px)"
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

# Run HAT-Base training
# Using base model (~20M params) for best quality/overfitting balance
# Loss weights optimized for color grading:
# - Strong LPIPS and LAB for perceptual color accuracy
# - SSIM for structural quality
# - Histogram for color distribution matching
python3 src/training/train_hat.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_hat \
    --model_size base \
    --batch_size 8 \
    --num_epochs 200 \
    --image_size 256 \
    --lr 2e-4 \
    --lambda_charbonnier 1.0 \
    --lambda_ssim 0.2 \
    --lambda_fft 0.05 \
    --lambda_perceptual 0.1 \
    --lambda_lpips 0.2 \
    --lambda_lab 0.2 \
    --lambda_hist 0.1 \
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
echo "HAT Training Complete"
echo "End time: $(date)"
echo "=========================================="
