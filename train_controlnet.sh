#!/bin/bash
# =============================================================================
# ControlNet Training Script for AutoHDR Real Estate Photo Enhancement
# =============================================================================
#
# ControlNet uses zero-convolution for stable training and provides
# strong conditioning from the source image.
#
# Advantages over GAN-based approach:
# - More stable training (no adversarial loss)
# - Faster convergence
# - Better preservation of source image structure
#
# Usage:
#   ./train_controlnet.sh                     # Train full model
#   ./train_controlnet.sh --model_type lite   # Train lightweight version
#
# =============================================================================

set -e

# Default configuration
DATA_ROOT="${DATA_ROOT:-./data}"
JSONL_PATH="${JSONL_PATH:-train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/controlnet}"
MODEL_TYPE="${MODEL_TYPE:-full}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"

# Loss weights (optimized for real estate HDR)
LAMBDA_L1="${LAMBDA_L1:-100.0}"
LAMBDA_PERCEPTUAL="${LAMBDA_PERCEPTUAL:-10.0}"
LAMBDA_SSIM="${LAMBDA_SSIM:-5.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-2.0}"
LAMBDA_HIST="${LAMBDA_HIST:-1.0}"
LAMBDA_LAB="${LAMBDA_LAB:-10.0}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-5.0}"

echo "=============================================="
echo "ControlNet HDR Enhancement Training"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model type:     $MODEL_TYPE"
echo "  Data root:      $DATA_ROOT"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Image size:     ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch size:     $BATCH_SIZE"
echo "  Epochs:         $EPOCHS"
echo "  Warmup epochs:  $WARMUP_EPOCHS"
echo ""
echo "Loss weights:"
echo "  L1:             $LAMBDA_L1"
echo "  Perceptual:     $LAMBDA_PERCEPTUAL"
echo "  SSIM:           $LAMBDA_SSIM"
echo "  Edge:           $LAMBDA_EDGE"
echo "  Histogram:      $LAMBDA_HIST"
echo "  LAB:            $LAMBDA_LAB"
echo "  LPIPS:          $LAMBDA_LPIPS"
echo ""
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python src/training/train_controlnet.py \
    --data_root "$DATA_ROOT" \
    --jsonl_path "$JSONL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$EPOCHS" \
    --lr 2e-4 \
    --lambda_l1 "$LAMBDA_L1" \
    --lambda_perceptual "$LAMBDA_PERCEPTUAL" \
    --lambda_ssim "$LAMBDA_SSIM" \
    --lambda_edge "$LAMBDA_EDGE" \
    --lambda_hist "$LAMBDA_HIST" \
    --lambda_lab "$LAMBDA_LAB" \
    --lambda_lpips "$LAMBDA_LPIPS" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --min_lr 1e-6 \
    --grad_clip 1.0 \
    --save_interval 10 \
    --sample_interval 5 \
    "$@"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
echo "Samples saved to: $OUTPUT_DIR/samples/"
echo ""
echo "Best model: $OUTPUT_DIR/checkpoints/controlnet_best.pt"
echo ""
