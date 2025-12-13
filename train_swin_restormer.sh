#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_swin_restormer_%j.out
#SBATCH -e train_swin_restormer_%j.err
#SBATCH -J swin_rest

# =============================================================================
# SwinRestormer: Pretrained Swin Encoder + Restormer Decoder
# =============================================================================
# Fine-tuning strategy with progressive unfreezing:
# - Stage 1 (epochs 1-30): Freeze encoder, train decoder only
# - Stage 2 (epochs 31-60): Unfreeze last 2 encoder layers
# - Stage 3 (epochs 61-100): Full fine-tuning
#
# Anti-overfitting techniques:
# - Pretrained encoder (ImageNet-22K, 14M images)
# - Progressive unfreezing
# - Lower LR for encoder layers
# - Strong data augmentation
# - Mixup regularization
# - Early stopping
# - EMA
# =============================================================================

echo "=========================================="
echo "SwinRestormer Fine-Tuning"
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

# Install/verify dependencies
python3 -c "import timm; print(f'timm: {timm.__version__}')" || pip install timm --user --quiet
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --user --quiet
python3 -c "import einops; print('einops OK')" || pip install einops --user --quiet

# =============================================================================
# Training Configuration
# =============================================================================
# Model: swin_restormer_small (Swin-Small encoder + Restormer decoder)
# Total epochs: 100 (30 + 30 + 40)
# Batch size: 4 (effective 8 with grad accumulation)
# =============================================================================

python3 src/training/train_swin_restormer.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_swin_restormer \
    --image_size 256 \
    --model_size small \
    --batch_size 4 \
    --stage1_epochs 30 \
    --stage2_epochs 30 \
    --stage3_epochs 40 \
    --lr_stage1 2e-4 \
    --lr_stage2 5e-5 \
    --lr_stage3 1e-5 \
    --weight_decay 0.05 \
    --use_mixup \
    --mixup_alpha 0.2 \
    --use_amp \
    --use_ema \
    --patience 15 \
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
echo "SwinRestormer Training Complete"
echo "End time: $(date)"
echo "=========================================="
