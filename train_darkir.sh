#!/bin/bash
#SBATCH --job-name=darkir_cv
#SBATCH --output=darkir_cv_%j.out
#SBATCH --error=darkir_cv_%j.err
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ============================================================================
# DarkIR 3-Fold Cross-Validation Training
# ============================================================================
# Top 0.0001% MLE approach:
# - Train 3 models with 3-fold CV
# - Early stopping on Val PSNR (15 epoch patience)
# - Track PSNR/SSIM metrics
# - Zero data leakage (test set never touched)
# ============================================================================

echo "========================================="
echo "DarkIR Training - 3-Fold Cross-Validation"
echo "========================================="
echo "Start time: $(date)"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv

# Activate conda environment (you already have one!)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate autohdr

# Install DarkIR dependencies
pip install -q ptflops opencv-python 2>/dev/null || true

# Configuration - Optimal for A100 80GB
RESOLUTION=512  # User requested 512x512
MODEL_SIZE=m    # m (3.31M params) or l (12.96M params)
BATCH_SIZE=16   # Optimal for A100
EPOCHS=100
LR=1e-4
EARLY_STOP_PATIENCE=15
OUTPUT_DIR="outputs_darkir_512_cv"

echo "Configuration:"
echo "  Resolution: ${RESOLUTION}"
echo "  Model: DarkIR-${MODEL_SIZE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Early stopping: ${EARLY_STOP_PATIENCE} epochs"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Train fold 1 with simplified script (FP32 - no mixed precision for stability)
python3 train_darkir_simple.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --base_dir . \
    --resolution ${RESOLUTION} \
    --width 32 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --early_stopping_patience ${EARLY_STOP_PATIENCE} \
    --output_dir ${OUTPUT_DIR}/fold_1 \
    --save_every 10 \
    --num_workers 8

# Train fold 2
python3 train_darkir_simple.py \
    --train_jsonl data_splits/fold_2/train.jsonl \
    --val_jsonl data_splits/fold_2/val.jsonl \
    --base_dir . \
    --resolution ${RESOLUTION} \
    --width 32 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --early_stopping_patience ${EARLY_STOP_PATIENCE} \
    --output_dir ${OUTPUT_DIR}/fold_2 \
    --save_every 10 \
    --num_workers 8

# Train fold 3
python3 train_darkir_simple.py \
    --train_jsonl data_splits/fold_3/train.jsonl \
    --val_jsonl data_splits/fold_3/val.jsonl \
    --base_dir . \
    --resolution ${RESOLUTION} \
    --width 32 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --early_stopping_patience ${EARLY_STOP_PATIENCE} \
    --output_dir ${OUTPUT_DIR}/fold_3 \
    --save_every 10 \
    --num_workers 8

echo ""
echo "========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "========================================="
