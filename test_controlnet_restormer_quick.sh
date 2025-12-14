#!/bin/bash
# ============================================================================
# Quick Test: ControlNet-Restormer Architecture
# ============================================================================
# Runs 1 fold for 5 epochs to verify everything works
# Runtime: ~15-20 minutes
# ============================================================================

set -e

echo "========================================"
echo "ControlNet-Restormer Quick Test"
echo "========================================"
echo ""

# ============================================================================
# CONFIGURATION - QUICK TEST
# ============================================================================

RESOLUTION=384              # Lower res for quick test
BATCH_SIZE=2                # Smaller batch (dual model uses 2x memory)
EPOCHS=5                    # Just 5 epochs to verify
EARLY_STOP_PATIENCE=999     # Disable for test
N_FOLDS=1                   # Only test fold 1
NUM_WORKERS=8
MIXED_PRECISION=true

# Loss weights
LAMBDA_L1=1.0
LAMBDA_VGG=0.2
LAMBDA_SSIM=0.1

# Pretrained (optional for quick test)
PRETRAINED_PATH=""          # Leave empty to test from scratch

# Output
OUTPUT_DIR="outputs_controlnet_restormer_quick_test"

echo "📋 Quick Test Configuration:"
echo "  Resolution: ${RESOLUTION}px"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS} (just to verify it works)"
echo "  Folds: ${N_FOLDS} (fold 1 only)"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU:"
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    echo ""
fi

# Build command
CMD="python3 train_controlnet_restormer_cv.py \
    --data_splits_dir data_splits \
    --resolution ${RESOLUTION} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --early_stopping_patience ${EARLY_STOP_PATIENCE} \
    --lambda_l1 ${LAMBDA_L1} \
    --lambda_vgg ${LAMBDA_VGG} \
    --lambda_ssim ${LAMBDA_SSIM} \
    --n_folds ${N_FOLDS} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers ${NUM_WORKERS} \
    --device cuda \
    --use_checkpointing"

if [ "$MIXED_PRECISION" = true ]; then
    CMD="${CMD} --mixed_precision"
fi

if [ -n "$PRETRAINED_PATH" ]; then
    CMD="${CMD} --pretrained_path ${PRETRAINED_PATH}"
fi

echo "🚀 Starting quick test..."
echo ""

# Execute
eval ${CMD}

echo ""
echo "========================================"
echo "✅ Quick test complete!"
echo "========================================"
echo ""
echo "If you see this message, the architecture works! ✅"
echo ""
echo "📊 Check results:"
echo "   cat ${OUTPUT_DIR}/cv_summary.json | python3 -m json.tool"
echo ""
echo "🚀 Ready for full training:"
echo "   sbatch train_controlnet_restormer_512_a100.sh"
echo ""
echo "Expected full training results:"
echo "  Val PSNR: ~30-32 dB (with pretrained)"
echo "  Test PSNR: ~30-31 dB (ensemble of 3 folds)"
echo ""
echo "========================================"
