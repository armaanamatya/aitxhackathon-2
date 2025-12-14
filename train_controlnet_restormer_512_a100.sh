#!/bin/bash
#SBATCH --job-name=cn_restormer_512
#SBATCH --output=cn_restormer_512_%j.out
#SBATCH --error=cn_restormer_512_%j.err
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00

# ============================================================================
# ControlNet-Restormer Training - 512x512 on A100 (80GB)
# ============================================================================
# Optimized for maximum quality on unseen test set
# Memory usage: ~15-20GB with batch size 16 (mixed precision)
# Training time: ~12-16 hours for 100 epochs
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "ControlNet-Restormer Training - 512x512"
echo "========================================="
echo "Start time: $(date)"
echo ""

# ============================================================================
# ACTIVATE ENVIRONMENT
# ============================================================================

source ~/miniforge3/etc/profile.d/conda.sh
conda activate autohdr_venv

echo "🔍 Using Python: $(which python3)"
echo "🔍 Python version: $(python3 --version)"
echo ""

# GPU check
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR A100 80GB
# ============================================================================

# Model architecture
DIM=48                      # Base dimension
NUM_BLOCKS="4 6 6 8"        # Restormer default
NUM_REFINEMENT_BLOCKS=4
HEADS="1 2 4 8"
FFN_EXPANSION=2.66

# Training - OPTIMIZED FOR A100
RESOLUTION=512              # High resolution for better quality
BATCH_SIZE=16               # A100 80GB can handle 16-24 @ 512px
EPOCHS=100                  # Maximum epochs
LR=1e-4                     # Learning rate
WARMUP_EPOCHS=10            # Warmup
EARLY_STOP_PATIENCE=15      # Early stopping

# Loss weights (optimized for perceptual quality)
LAMBDA_L1=1.0               # Pixel-level accuracy
LAMBDA_VGG=0.2              # Perceptual quality (higher for better visuals)
LAMBDA_SSIM=0.1             # Structural similarity

# Cross-validation
N_FOLDS=3                   # 3-fold CV
NUM_WORKERS=16              # Match CPU count
MIXED_PRECISION=true        # CRITICAL for memory efficiency

# Pretrained weights (HIGHLY RECOMMENDED)
# Download from: https://github.com/swz30/Restormer/releases
# Options:
#   - Motion_Deblurring.pth (GoPro trained)
#   - Denoising.pth (SIDD trained)
PRETRAINED_PATH="pretrained/restormer_denoising.pth"          # Set this to pretrained checkpoint path
# Example: PRETRAINED_PATH="pretrained/Motion_Deblurring.pth"

# Output
OUTPUT_DIR="outputs_controlnet_restormer_512_cv"
SAVE_EVERY=10

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "📋 Configuration:"
echo "  Architecture: ControlNet-Restormer"
echo "  Resolution: ${RESOLUTION}px"
echo "  Batch size: ${BATCH_SIZE} (optimized for A100 80GB)"
echo "  Epochs: ${EPOCHS} (max)"
echo "  Learning rate: ${LR}"
echo "  Early stopping: ${EARLY_STOP_PATIENCE} epochs"
echo "  Loss weights: L1=${LAMBDA_L1}, VGG=${LAMBDA_VGG}, SSIM=${LAMBDA_SSIM}"
echo "  CV folds: ${N_FOLDS}"
echo "  Workers: ${NUM_WORKERS}"
echo "  Mixed precision: ${MIXED_PRECISION}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

if [ -n "$PRETRAINED_PATH" ]; then
    echo "  ✅ Pretrained: ${PRETRAINED_PATH}"
    echo "     Expected PSNR gain: +3-5 dB over from-scratch"
else
    echo "  ⚠️  WARNING: No pretrained weights specified!"
    echo "     ControlNet-Restormer works MUCH better with pretrained base."
    echo "     Expected PSNR without pretraining: ~26-28 dB"
    echo "     Expected PSNR WITH pretraining: ~29-32 dB"
    echo ""
    echo "     To use pretrained weights:"
    echo "     1. Download from https://github.com/swz30/Restormer/releases"
    echo "     2. Set PRETRAINED_PATH variable above"
fi
echo ""

# ============================================================================
# MEMORY ESTIMATE
# ============================================================================

echo "💾 Estimated Memory Usage @ 512px:"
echo "  Model weights: ~400 MB (base + trainable)"
echo "  Activations (FP16, batch=${BATCH_SIZE}): ~8-12 GB"
echo "  Optimizer states: ~800 MB"
echo "  VGG loss model: ~550 MB"
echo "  Total: ~12-16 GB (well within 80GB A100)"
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================

CMD="python3 train_controlnet_restormer_cv.py \
    --data_splits_dir data_splits \
    --resolution ${RESOLUTION} \
    --dim ${DIM} \
    --num_blocks ${NUM_BLOCKS} \
    --num_refinement_blocks ${NUM_REFINEMENT_BLOCKS} \
    --heads ${HEADS} \
    --ffn_expansion_factor ${FFN_EXPANSION} \
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
echo "Expected training time:"
echo "  ~4-5 hours per fold"
echo "  ~12-16 hours total (3 folds)"
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
echo "🧪 Next step - Evaluate on test set:"
echo "   python3 evaluate_controlnet_restormer_test.py \\"
echo "       --model_dir ${OUTPUT_DIR} \\"
echo "       --resolution ${RESOLUTION}"
echo ""
echo "🎯 Expected results on test set:"
echo "   Single fold: ~29-31 dB PSNR"
echo "   Ensemble (3 folds avg): ~30-32 dB PSNR"
echo "   Gain over from-scratch: +3-5 dB"
echo ""
echo "========================================="
