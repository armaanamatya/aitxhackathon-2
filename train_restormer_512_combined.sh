#!/bin/bash
# Train Restormer at 3296x2192 resolution on Brev B300
# Usage: ./train_restormer_512_combined.sh

echo "========================================================================"
echo "RESTORMER 3008x2000 (6MP) - COMBINED LOSS (L1 + Window + Color)"
echo "========================================================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "CONFIG:"
echo "  - Resolution: 3008 x 2000 (~6MP, divisible by 16)"
echo "  - Batch size: 1"
echo "  - Gradient checkpointing: DISABLED (faster)"
echo "  - Data workers: 8"
echo "  - Expected memory: ~80-100GB"
echo ""
echo "DATA SPLIT (no leakage):"
echo "  - TEST: 10 images (HELD OUT - never seen)"
echo "  - TRAIN: 511 images (90%)"
echo "  - VAL: 56 images (10%)"
echo ""
echo "LOSS: L1(1.0) + Window(0.5) + BrightRegionSaturation(0.3)"
echo ""

# Verify data splits exist
if [ ! -f "data_splits/proper_split/train.jsonl" ]; then
    echo "ERROR: Proper data splits not found!"
    echo "Run: python3 create_proper_splits.py"
    exit 1
fi

echo "Data splits verified:"
wc -l data_splits/proper_split/*.jsonl
echo ""
echo "========================================================================"
echo "Starting training..."
echo "========================================================================"

# Train Restormer 3008x2000 (~6MP) with combined loss
python3 train_restormer_512_combined_loss.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_restormer_3008 \
    --resolution 3008 \
    --batch_size 1 \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --epochs 100 \
    --num_workers 8

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================================================"
echo ""
echo "Next step: Finetune encoder"
echo "python3 finetune_encoder.py --checkpoint outputs_restormer_3008/checkpoint_best.pt --resolution 3008 --batch_size 1 --epochs 50"
