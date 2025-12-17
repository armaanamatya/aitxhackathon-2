#!/bin/bash
#SBATCH --job-name=prod_gain
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# ============================================================================
# PRODUCTION GAIN NETWORK TRAINING
# ============================================================================
# Formulation: output = input * gain (CANNOT produce black spots)
#
# Key differences from SOTA:
# - Multiplicative gain instead of additive residual
# - Dual-path processing (highlights vs shadows)
# - Zone-aware attention
# - Edge-aware gain smoothing
# ============================================================================

cd /mmfs1/home/sww35/autohdr-real-estate-577

export PATH="/cm/local/apps/python39/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "============================================================"
echo "PRODUCTION GAIN NETWORK"
echo "============================================================"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Formulation: output = input * gain"
echo "============================================================"

python3 -u train_production.py \
    --data_dir images \
    --output_dir outputs_production \
    --resolution 512 \
    --batch_size 4 \
    --epochs 100 \
    --lr 2e-4 \
    --patience 15 \
    --min_epochs 20

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "End: $(date)"
echo "Output: outputs_production/"
echo "============================================================"
