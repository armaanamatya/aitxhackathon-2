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
# MambaDiffusion 512x512 - FSDP (3 Nodes, Shared Memory)
# =============================================================================
# Uses PyTorch FSDP to shard model across 3 GPUs
# Effective VRAM: 240GB (3 x 80GB)
# Resolution: 512x512
# =============================================================================

echo "=========================================="
echo "MambaDiffusion 512x512 - FSDP 3-Node"
echo "=========================================="
echo "Start time: $(date)"
echo "Master node: $(hostname)"
echo "All nodes: $SLURM_JOB_NODELIST"
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

# Get master node info
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo ""
echo "Training Configuration:"
echo "  - Mode: FSDP (Fully Sharded Data Parallel)"
echo "  - Sharding: FULL_SHARD (model split across GPUs)"
echo "  - Effective VRAM: 240GB (3 x 80GB)"
echo "  - Resolution: 512x512"
echo "  - Nodes: 3"
echo "  - Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

# Run with torchrun for multi-node
srun --ntasks=$SLURM_NTASKS \
    --ntasks-per-node=1 \
    bash -c "
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        export WORLD_SIZE=$SLURM_NTASKS
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=0

        python3 src/training/train_mamba_fsdp_512.py \
            --data_root . \
            --jsonl_path train.jsonl \
            --output_dir outputs_mamba_512 \
            --crop_size 512 \
            --batch_size 1 \
            --gradient_accumulation 4 \
            --num_epochs 100 \
            --lr 2e-4
    "

TRAIN_EXIT=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Training Complete!"
    ls -la outputs_mamba_512/
else
    echo "Training failed with exit code: $TRAIN_EXIT"
fi
echo "End time: $(date)"
echo "=========================================="
