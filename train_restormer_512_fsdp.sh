#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o train_restormer_512_%j.out
#SBATCH -e train_restormer_512_%j.err
#SBATCH -J rest_512

# =============================================================================
# Restormer 512x512 - FSDP (3 Nodes, Shared Memory)
# =============================================================================

echo "=========================================="
echo "Restormer 512x512 - FSDP 3-Node"
echo "=========================================="
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /mmfs1/home/sww35/autohdr-real-estate-577

if [ -n "$VIRTUAL_ENV" ]; then deactivate 2>/dev/null || true; fi
unset VIRTUAL_ENV

module load python39
module load cuda11.8/toolkit/11.8.0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29501

echo ""
echo "FSDP Config: FULL_SHARD, 240GB effective VRAM"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 \
    bash -c "
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        export WORLD_SIZE=$SLURM_NTASKS
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=0

        python3 src/training/train_restormer_fsdp_512.py \
            --data_root . \
            --jsonl_path train.jsonl \
            --output_dir outputs_restormer_512 \
            --crop_size 512 \
            --batch_size 1 \
            --gradient_accumulation 4 \
            --num_epochs 100 \
            --lr 2e-4
    "

echo "End time: $(date)"
