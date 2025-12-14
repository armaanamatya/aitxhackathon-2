#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o train_restormer_512_%j.out
#SBATCH -e train_restormer_512_%j.err
#SBATCH -J rest_512

# =============================================================================
# Restormer 512x512 Training with DeepSpeed (3 GPUs)
# =============================================================================
# Uses DeepSpeed ZeRO-2 with CPU offloading for memory efficiency
# Resolution: 512x512
# GPUs: 3 x A100 80GB
# =============================================================================

echo "=========================================="
echo "Restormer 512x512 - 3 GPU DeepSpeed"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate any active virtualenv
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

echo ""
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Install DeepSpeed if needed
python3 -c "import deepspeed" 2>/dev/null || python3 -m pip install deepspeed --user --quiet

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=3

echo ""
echo "Training Configuration:"
echo "  - Resolution: 512x512"
echo "  - GPUs: 3"
echo "  - Batch per GPU: 1"
echo "  - Gradient Accumulation: 4"
echo "  - Effective Batch Size: 12"
echo "  - Epochs: 100"
echo ""

# Run with DeepSpeed launcher
deepspeed --num_gpus=3 \
    src/training/train_restormer_deepspeed_512.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_restormer_512 \
    --crop_size 512 \
    --batch_size 1 \
    --gradient_accumulation 4 \
    --num_epochs 100 \
    --lr 2e-4

TRAIN_EXIT=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Training Complete!"
    echo "Checkpoints saved to: outputs_restormer_512/"
    ls -la outputs_restormer_512/
else
    echo "Training failed with exit code: $TRAIN_EXIT"
fi
echo "End time: $(date)"
echo "=========================================="
