#!/bin/bash
# DDP Multi-GPU Training for Restormer at 7MP (3296x2192)
# Usage: ./train_ddp_512.sh
#
# For 2x H200 (282GB total):
#   - 7MP (3296x2192) without checkpointing
#   - batch_size=4 (2 per GPU)
#   - torch.compile enabled for speed

echo "========================================================================"
echo "RESTORMER DDP - 4MP (2448x1632) - MULTI-GPU TRAINING"
echo "========================================================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs detected: $NUM_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "CONFIG:"
echo "  - Resolution: 2448 x 1632 (~4MP, divisible by 16)"
echo "  - Total batch size: 4 (2 per GPU)"
echo "  - torch.compile: ENABLED (H100/H200 only)"
echo "  - DDP backend: NCCL"
echo "  - Data workers: 4 per process"
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
echo "Starting DDP training with $NUM_GPUS GPUs..."
echo "========================================================================"

# Launch with torchrun for DDP
torchrun --nproc_per_node=$NUM_GPUS train_restormer_ddp.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_restormer_ddp_4mp \
    --resolution 2448 \
    --batch_size 4 \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --epochs 100 \
    --compile \
    --num_workers 4

echo ""
echo "========================================================================"
echo "DDP Training complete!"
echo "Date: $(date)"
echo "========================================================================"
echo ""
echo "Next step: Finetune encoder"
echo "python3 finetune_encoder.py --checkpoint outputs_restormer_ddp_4mp/checkpoint_best.pt --resolution 2448 --batch_size 4 --epochs 50"
