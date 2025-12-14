#!/bin/bash
#SBATCH --job-name=restormer_clean
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=training_clean_%j.out
#SBATCH --error=training_clean_%j.err

set -e

echo "================================================================================"
echo "RESTORMER CLEAN TRAINING FROM SCRATCH"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate environment
source /mmfs1/home/sww35/miniforge3/etc/profile.d/conda.sh
conda activate controlnet_a100

echo "📋 Environment:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.version.cuda}')"
echo ""

echo "🔍 GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv
echo ""

echo "================================================================================"
echo "CONFIGURATION - SINGLE RESTORMER FROM SCRATCH"
echo "================================================================================"
echo ""
echo "📐 Model: Restormer-Base"
echo "  Dimension: 48"
echo "  Blocks: 4-6-6-8"
echo "  Params: ~26M (all trainable)"
echo ""
echo "🎯 Training:"
echo "  Resolution: 384px (optimal for memory)"
echo "  Batch size: 16"
echo "  Epochs: 150"
echo "  Learning rate: 2e-4"
echo "  Early stopping: 20 epochs"
echo ""
echo "📊 Data:"
echo "  3-fold CV, 90:10 split"
echo "  511 train / 56 val per fold"
echo "  10 test samples (held out)"
echo ""
echo "================================================================================"

# Execute training
python3 train_restormer_simple.py \
    --data_splits_dir data_splits \
    --resolution 384 \
    --dim 48 \
    --num_blocks 4 6 6 8 \
    --num_refinement_blocks 4 \
    --heads 1 2 4 8 \
    --ffn_expansion_factor 2.66 \
    --batch_size 16 \
    --epochs 150 \
    --lr 2e-4 \
    --warmup_epochs 15 \
    --early_stopping_patience 20 \
    --lambda_l1 1.0 \
    --lambda_vgg 0.2 \
    --lambda_ssim 0.1 \
    --n_folds 3 \
    --output_dir outputs_restormer_clean \
    --save_every 10 \
    --num_workers 32 \
    --device cuda \
    --mixed_precision

echo ""
echo "================================================================================"
echo "Training completed: $(date)"
echo "================================================================================"
