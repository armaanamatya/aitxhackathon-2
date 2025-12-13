#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_restormer_large_%j.out
#SBATCH -e train_restormer_large_%j.err
#SBATCH -J autohdr_restormer_large

# Optimized Restormer-Large Training
# Features:
# - Restormer-large model (57M params)
# - EMA (Exponential Moving Average) for stable outputs
# - Enhanced losses: Charbonnier, SSIM, FFT, VGG, LPIPS, LAB, Histogram

echo "=========================================="
echo "Restormer-Large Optimized Training"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load modules (packages are in user's .local)
module load python39
module load cuda11.8/toolkit/11.8.0

# Print Python and CUDA info
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify lpips is installed
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --quiet

# Run Restormer-Large training with all optimizations
python3 src/training/train_restormer_large.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_restormer_large \
    --model_size large \
    --batch_size 1 \
    --num_epochs 150 \
    --image_size 512 \
    --lr 2e-4 \
    --lambda_charbonnier 1.0 \
    --lambda_ssim 0.1 \
    --lambda_fft 0.05 \
    --lambda_perceptual 0.1 \
    --lambda_lpips 0.1 \
    --lambda_lab 0.1 \
    --lambda_hist 0.05 \
    --ema_decay 0.999 \
    --save_interval 10 \
    --sample_interval 5 \
    --num_workers 4

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "End time: $(date)"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

echo "=========================================="
echo "Restormer-Large Training Complete"
echo "End time: $(date)"
echo "=========================================="
