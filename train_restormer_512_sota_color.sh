#!/bin/bash
#SBATCH --job-name=restormer_sota_color
#SBATCH --output=logs/restormer_sota_color_%j.out
#SBATCH --error=logs/restormer_sota_color_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Restormer 512 with SOTA Color Enhancement Loss
# Robust optimal solution for real estate HDR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Starting at: $(date)"
echo ""

# Load environment
source ~/.bashrc
conda activate hdr

# GPU info
nvidia-smi
echo ""

# Training parameters
RESOLUTION=512
BATCH_SIZE=4
EPOCHS=50
LR=2e-4
OUTPUT_DIR="outputs_restormer_512_sota_color"

echo "Training Configuration:"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# Run training
python train_restormer_512_sota_color.py \
    --resolution ${RESOLUTION} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "Training complete at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
