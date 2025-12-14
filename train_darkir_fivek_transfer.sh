#!/bin/bash
#SBATCH --job-name=darkir_fivek
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=darkir_fivek_%j.out

echo "========================================"
echo "DarkIR with FiveK Transfer Learning"
echo "========================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Configuration
RESOLUTION=512
FIVEK_DIR="fivek_dataset"
FIVEK_EPOCHS=30
FIVEK_BATCH=16
FIVEK_LR=2e-4

FINETUNE_EPOCHS=100
FINETUNE_BATCH=8
FINETUNE_LR=2e-5  # 10x lower than pretrain!

OUTPUT_DIR="outputs_darkir_fivek_transfer"

echo ""
echo "Configuration:"
echo "  Resolution: ${RESOLUTION}"
echo "  FiveK epochs: ${FIVEK_EPOCHS}"
echo "  Finetune epochs: ${FINETUNE_EPOCHS}"
echo "  FiveK LR: ${FIVEK_LR}"
echo "  Finetune LR: ${FINETUNE_LR} (10x lower)"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Check if FiveK dataset exists
if [ ! -d "${FIVEK_DIR}" ]; then
    echo "ERROR: FiveK dataset not found at ${FIVEK_DIR}"
    echo "Please run: python3 prepare_fivek_robust.py first"
    exit 1
fi

# Run transfer learning
python3 train_darkir_fivek_transfer.py \
    --fivek_dir ${FIVEK_DIR} \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --base_dir . \
    --resolution ${RESOLUTION} \
    --width 32 \
    --fivek_epochs ${FIVEK_EPOCHS} \
    --fivek_batch_size ${FIVEK_BATCH} \
    --fivek_lr ${FIVEK_LR} \
    --finetune_epochs ${FINETUNE_EPOCHS} \
    --finetune_batch_size ${FINETUNE_BATCH} \
    --finetune_lr ${FINETUNE_LR} \
    --early_stopping_patience 15 \
    --use_window_loss \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 8

echo ""
echo "========================================"
echo "Transfer learning complete!"
echo "Date: $(date)"
echo "========================================"
