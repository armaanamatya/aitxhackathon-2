#!/bin/bash
# ============================================================================
# DarkIR Training - Manual Execution Script
# ============================================================================
# Run this directly in your working environment:
#   bash train_darkir_manual.sh
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "DarkIR Training - 3-Fold Cross-Validation"
echo "========================================="
echo "Start time: $(date)"
echo ""

# GPU check
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# CONFIGURATION (Edit these as needed)
# ============================================================================

RESOLUTION=384          # 256, 384, 512, or 640
MODEL_SIZE=m            # m (3.31M params) or l (12.96M params)
BATCH_SIZE=8            # 4, 8, 12, or 16 (adjust based on GPU memory)
EPOCHS=100              # Maximum epochs (early stopping will likely stop earlier)
LR=1e-4                 # Learning rate
WARMUP_EPOCHS=10        # Warmup epochs
EARLY_STOP_PATIENCE=15  # Early stopping patience (epochs without improvement)

# Loss weights (higher = more emphasis)
LAMBDA_L1=1.0           # Pixel-level accuracy
LAMBDA_VGG=0.15         # Perceptual quality (increased from 0.1)
LAMBDA_SSIM=0.1         # Structural similarity

# Training options
N_FOLDS=3               # Number of CV folds (1, 2, or 3)
NUM_WORKERS=8           # DataLoader workers (0 if environment issues)
MIXED_PRECISION=true    # Use mixed precision (faster training)

# Output
OUTPUT_DIR="outputs_darkir_${RESOLUTION}_${MODEL_SIZE}_cv"
SAVE_EVERY=10           # Save checkpoint every N epochs

# Optional: Pretrained weights path (leave empty if training from scratch)
PRETRAINED_PATH=""      # e.g., "DarkIR/models/bests/LOLBlur_DarkIR_m.pth"

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
echo "Command:"
echo "${CMD}"
echo ""
echo "========================================="
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
echo "📈 View cross-validation summary:"
echo "   cat ${OUTPUT_DIR}/cv_summary.json | python3 -m json.tool"
echo ""
echo "🔍 Next steps:"
echo "   1. Check fold results in ${OUTPUT_DIR}/fold_*/"
echo "   2. Evaluate on test set:"
echo "      python3 evaluate_darkir_test.py \\"
echo "          --cv_dir ${OUTPUT_DIR} \\"
echo "          --test_jsonl data_splits/test.jsonl \\"
echo "          --resolution ${RESOLUTION} \\"
echo "          --model_size ${MODEL_SIZE} \\"
echo "          --n_folds ${N_FOLDS} \\"
echo "          --save_visuals \\"
echo "          --output_dir test_results"
echo ""
echo "========================================="
