#!/bin/bash
#SBATCH --job-name=darkir_window
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=darkir_window_%j.out

echo "========================================"
echo "DarkIR with Window-Aware Loss"
echo "========================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Configuration
RESOLUTION=512
BATCH_SIZE=8
EPOCHS=100
LOSS_CONFIG="aggressive"  # default, aggressive, or light
OUTPUT_DIR="outputs_darkir_window_aware"

echo ""
echo "Configuration:"
echo "  Resolution: ${RESOLUTION}"
echo "  Loss config: ${LOSS_CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Train on fold 1
python3 train_darkir_window_aware.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --base_dir . \
    --resolution ${RESOLUTION} \
    --width 32 \
    --loss_config ${LOSS_CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --early_stopping_patience 15 \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 8

echo ""
echo "========================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================"
