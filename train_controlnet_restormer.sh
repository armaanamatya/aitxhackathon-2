#!/bin/bash
# ============================================================================
# ControlNet-Restormer Training Script
# ============================================================================
# Maximum quality on unseen test set using hybrid architecture
# ============================================================================

set -e  # Exit on error

echo "========================================"
echo "ControlNet-Restormer Training"
echo "========================================"
echo "Start time: $(date)"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model architecture
DIM=48                      # Base dimension (48 = 26.1M params)
NUM_BLOCKS="4 6 6 8"        # Number of blocks per stage
NUM_REFINEMENT_BLOCKS=4     # Refinement blocks
HEADS="1 2 4 8"            # Attention heads per stage
FFN_EXPANSION=2.66         # FFN expansion factor

# Training
RESOLUTION=512              # 384, 512, 768, 1024 (use 512+ for B200)
BATCH_SIZE=8                # 8 for 512px, 16 for 384px on B200
EPOCHS=100                  # Maximum epochs
LR=1e-4                     # Learning rate
WARMUP_EPOCHS=10           # Warmup epochs
EARLY_STOP_PATIENCE=15     # Early stopping patience

# Loss weights (optimized for perceptual quality)
LAMBDA_L1=1.0              # Pixel-level accuracy
LAMBDA_VGG=0.2             # Perceptual quality (higher than DarkIR)
LAMBDA_SSIM=0.1            # Structural similarity

# Cross-validation
N_FOLDS=3                  # Number of CV folds
NUM_WORKERS=8              # DataLoader workers
MIXED_PRECISION=true       # Use mixed precision

# Pretrained weights (OPTIONAL - leave empty to train from scratch)
# Download from: https://github.com/swz30/Restormer/releases
PRETRAINED_PATH="pretrained/restormer_denoising.pth"         # e.g., "pretrained/restormer_sidd.pth"

# Output
OUTPUT_DIR="outputs_controlnet_restormer_${RESOLUTION}_cv"
SAVE_EVERY=10              # Save checkpoint every N epochs

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "📋 Configuration:"
echo "  Architecture: ControlNet-Restormer"
echo "  Resolution: ${RESOLUTION}px"
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
    echo "  Pretrained: Training from scratch (not recommended)"
    echo "  ⚠️  WARNING: ControlNet works best with pretrained base!"
fi
echo ""

# ============================================================================
# GPU CHECK
# ============================================================================

if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

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

# Add gradient checkpointing for high-res
if [ "$RESOLUTION" -ge 768 ]; then
    CMD="${CMD} --use_checkpointing"
    echo "✅ Gradient checkpointing enabled for ${RESOLUTION}px"
    echo ""
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
echo "========================================"
echo "✅ Training complete!"
echo "End time: $(date)"
echo "========================================"
echo ""
echo "📊 Results saved to: ${OUTPUT_DIR}"
echo ""
echo "📈 View summary:"
echo "   cat ${OUTPUT_DIR}/cv_summary.json | python3 -m json.tool"
echo ""
echo "🧪 Next steps:"
echo "   1. Evaluate on test set:"
echo "      python3 evaluate_controlnet_restormer_test.py --model_dir ${OUTPUT_DIR}"
echo ""
echo "   2. Run ensemble inference (Restormer + DarkIR + ControlNet):"
echo "      python3 inference_ensemble.py \\"
echo "          --input test_image.jpg \\"
echo "          --output enhanced.jpg \\"
echo "          --restormer_path outputs_restormer/checkpoint_best.pt \\"
echo "          --darkir_path outputs_darkir_384_m_cv/fold_1/checkpoint_best.pt"
echo ""
echo "========================================"
