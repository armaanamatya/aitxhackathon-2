#!/bin/bash
# Train Restormer at 3297x2201 resolution on Brev B300
# Usage: ./train_restormer_512_combined.sh

echo "========================================================================"
echo "RESTORMER 3297x2201 - COMBINED LOSS (L1 + Window + Color)"
echo "========================================================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "DATA SPLIT:"
echo "  - TEST: 10 images (HELD OUT - never seen)"
echo "  - TRAIN: 511 images (90%)"
echo "  - VAL: 56 images (10%)"
echo ""
echo "LOSS: L1(1.0) + Window(0.5) + BrightRegionSaturation(0.3)"
echo "GRADIENT CHECKPOINTING: ENABLED"
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

# Train Restormer 3297 with combined loss
python3 train_restormer_512_combined_loss.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_restormer_3297 \
    --resolution 3297 \
    --batch_size 1 \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --epochs 100 \
    --use_checkpointing \
    --num_workers 4

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================================================"
echo ""
echo "Next step: Finetune encoder"
echo "python3 finetune_encoder.py --checkpoint outputs_restormer_3297/checkpoint_best.pt --resolution 3297 --batch_size 1 --epochs 50"
