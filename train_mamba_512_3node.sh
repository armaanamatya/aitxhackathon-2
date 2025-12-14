#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o train_mamba_512_%j.out
#SBATCH -e train_mamba_512_%j.err
#SBATCH -J mamba_512

# =============================================================================
# MambaDiffusion 512x512 Training - Multi-Node (3 Nodes x 1 GPU)
# =============================================================================
# Uses DeepSpeed ZeRO-2 across 3 nodes for 3x memory
# Resolution: 512x512
# GPUs: 3 x A100 80GB (1 per node)
# =============================================================================

echo "=========================================="
echo "MambaDiffusion 512x512 - 3 Node Training"
echo "=========================================="
echo "Start time: $(date)"
echo "Master node: $(hostname)"
echo "All nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
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

# Get master node info for distributed training
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo ""
echo "Training Configuration:"
echo "  - Resolution: 512x512"
echo "  - Nodes: 3"
echo "  - GPUs per node: 1"
echo "  - Total GPUs: 3"
echo "  - Batch per GPU: 1"
echo "  - Gradient Accumulation: 4"
echo "  - Effective Batch Size: 12"
echo "  - Master: $MASTER_ADDR:$MASTER_PORT"
echo "  - Epochs: 100"
echo ""

# Create hostfile for DeepSpeed
HOSTFILE=/tmp/hostfile_$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "$host slots=1"
done > $HOSTFILE

echo "Hostfile contents:"
cat $HOSTFILE
echo ""

# Run with DeepSpeed multi-node launcher
deepspeed --hostfile=$HOSTFILE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/train_mamba_deepspeed_512.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_mamba_512 \
    --crop_size 512 \
    --batch_size 1 \
    --gradient_accumulation 4 \
    --num_epochs 100 \
    --lr 2e-4

TRAIN_EXIT=$?

# Cleanup
rm -f $HOSTFILE

echo ""
echo "=========================================="
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Training Complete!"
    echo "Checkpoints saved to: outputs_mamba_512/"
    ls -la outputs_mamba_512/
else
    echo "Training failed with exit code: $TRAIN_EXIT"
fi
echo "End time: $(date)"
echo "=========================================="
