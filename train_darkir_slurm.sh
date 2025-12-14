#!/bin/bash
#SBATCH --job-name=darkir_cv
#SBATCH --output=darkir_cv_%j.out
#SBATCH --error=darkir_cv_%j.err
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ============================================================================
# DarkIR Training - SLURM Submission Script
# ============================================================================
# Submit with: sbatch train_darkir_slurm.sh
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "DarkIR Training - 3-Fold Cross-Validation"
echo "========================================="
echo "Start time: $(date)"
echo ""

# ============================================================================
# ACTIVATE YOUR ENVIRONMENT
# ============================================================================
# IMPORTANT: Uncomment and modify the line that matches your setup

# Option 1: Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate autohdr_venv

# Option 2: Alternative conda path
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate autohdr

# Option 3: Virtual environment
# source ~/autohdr-real-estate-577/venv/bin/activate

# Verify environment
echo "🔍 Using Python: $(which python3)"
echo "🔍 Python version: $(python3 --version)"
echo ""

# Install missing dependencies if needed
echo "📦 Installing dependencies..."
pip install -q opencv-python ptflops 2>/dev/null || true
echo ""

# GPU check
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

RESOLUTION=384          # 256, 384, 512, or 640
MODEL_SIZE=m            # m (3.31M params) or l (12.96M params)
BATCH_SIZE=8            # 4, 8, 12, or 16
EPOCHS=100              # Maximum epochs
LR=1e-4                 # Learning rate
WARMUP_EPOCHS=10        # Warmup epochs
EARLY_STOP_PATIENCE=15  # Early stopping patience

# Loss weights
LAMBDA_L1=1.0           # Pixel-level accuracy
LAMBDA_VGG=0.15         # Perceptual quality (optimized)
LAMBDA_SSIM=0.1         # Structural similarity

# Training options
N_FOLDS=3               # Number of CV folds
NUM_WORKERS=8           # DataLoader workers
MIXED_PRECISION=true    # Use mixed precision

# Output
OUTPUT_DIR="outputs_darkir_${RESOLUTION}_${MODEL_SIZE}_cv"
SAVE_EVERY=10           # Save checkpoint every N epochs

# Optional: Pretrained weights
PRETRAINED_PATH=""      # Leave empty for training from scratch

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "📋 Configuration:"
echo "  Resolution: ${RESOLUTION}px"
echo "  Model: DarkIR-${MODEL_SIZE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS} (max)"
echo "  Learning rate: ${LR}"
echo "  Early stopping: ${EARLY_STOP_PATIENCE} epochs"
echo "  Loss weights: L1=${LAMBDA_L1}, VGG=${LAMBDA_VGG}, SSIM=${LAMBDA_SSIM}"
echo "  CV folds: ${N_FOLDS}"
echo "  Workers: ${NUM_WORKERS}"
echo "  Mixed precision: ${MIXED_PRECISION}"
echo "  Output: ${OUTPUT_DIR}"
if [ -n "$PRETRAINED_PATH" ]; then
    echo "  Pretrained: ${PRETRAINED_PATH}"
else
    echo "  Pretrained: Training from scratch"
fi
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================

CMD="python3 train_darkir_cv.py \
    --data_splits_dir data_splits \
    --resolution ${RESOLUTION} \
    --model_size ${MODEL_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --early_stopping_patience ${EARLY_STOP_PATIENCE} \
    --lambda_l1 ${LAMBDA_L1} \
    --lambda_vgg ${LAMBDA_VGG} \
    --lambda_ssim ${LAMBDA_SSIM} \
    --n_folds ${N_FOLDS} \
    --output_dir ${OUTPUT_DIR} \
    --save_every ${SAVE_EVERY} \
    --num_workers ${NUM_WORKERS} \
    --device cuda"

# Add optional flags
if [ "$MIXED_PRECISION" = true ]; then
    CMD="${CMD} --mixed_precision"
fi

if [ -n "$PRETRAINED_PATH" ]; then
    CMD="${CMD} --pretrained_path ${PRETRAINED_PATH}"
fi

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "🚀 Starting training..."
echo ""

# Execute training
eval ${CMD}

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "========================================="
echo "✅ Training complete!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "📊 Results saved to: ${OUTPUT_DIR}"
echo ""
echo "📈 View summary:"
echo "   cat ${OUTPUT_DIR}/cv_summary.json | python3 -m json.tool"
echo ""
echo "========================================="
