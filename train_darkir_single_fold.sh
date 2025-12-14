#!/bin/bash
# ============================================================================
# DarkIR Training - SINGLE FOLD TEST
# ============================================================================
# Quick test script to verify everything works before running all 3 folds
# Run this first: bash train_darkir_single_fold.sh
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "DarkIR Training - Single Fold Test"
echo "========================================="
echo "Start time: $(date)"
echo ""

# GPU check
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

# ============================================================================
# TEST CONFIGURATION (Quick test)
# ============================================================================

FOLD=1                  # Which fold to train (1, 2, or 3)
RESOLUTION=384          # 384px for quick test
MODEL_SIZE=m            # DarkIR-m (smaller, faster)
BATCH_SIZE=8            # Standard batch size
EPOCHS=20               # Reduced for quick test
LR=1e-4
EARLY_STOP_PATIENCE=5   # Stricter for quick test

LAMBDA_L1=1.0
LAMBDA_VGG=0.15
LAMBDA_SSIM=0.1

NUM_WORKERS=8           # Set to 0 if environment issues
OUTPUT_DIR="outputs_darkir_test_fold${FOLD}"

echo "📋 Test Configuration:"
echo "  Training fold: ${FOLD}"
echo "  Resolution: ${RESOLUTION}px"
echo "  Epochs: ${EPOCHS} (test run)"
echo "  Early stopping: ${EARLY_STOP_PATIENCE} epochs"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# RUN SINGLE FOLD
# ============================================================================

echo "🚀 Starting single fold test..."
echo ""

python3 train_darkir_cv.py \
    --data_splits_dir data_splits \
    --fold ${FOLD} \
    --resolution ${RESOLUTION} \
    --model_size ${MODEL_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup_epochs 3 \
    --early_stopping_patience ${EARLY_STOP_PATIENCE} \
    --lambda_l1 ${LAMBDA_L1} \
    --lambda_vgg ${LAMBDA_VGG} \
    --lambda_ssim ${LAMBDA_SSIM} \
    --n_folds 3 \
    --output_dir ${OUTPUT_DIR} \
    --save_every 5 \
    --num_workers ${NUM_WORKERS} \
    --mixed_precision \
    --device cuda

echo ""
echo "========================================="
echo "✅ Test complete!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "📊 Check results:"
echo "   cat ${OUTPUT_DIR}/fold_${FOLD}/history.json | python3 -m json.tool"
echo ""
echo "✅ If test passed, run full training:"
echo "   bash train_darkir_manual.sh"
echo ""
