#!/bin/bash
# =============================================================================
# Optimized Training Script for AutoHDR Real Estate Photo Enhancement
# =============================================================================
#
# This script uses enhanced loss functions and training optimizations:
# - SSIM loss for structural similarity
# - Edge-aware loss for sharp details
# - Color histogram loss for accurate color grading
# - Learning rate warmup for stable training
# - Gradient accumulation for larger effective batch sizes
# - FP16 mixed precision for faster training
#
# Usage:
#   ./train_optimized.sh                    # Use defaults
#   ./train_optimized.sh --epochs 200       # Override epochs
#   ./train_optimized.sh --model restormer  # Train Restormer instead
#
# =============================================================================

set -e

# Default configuration
DATA_ROOT="${DATA_ROOT:-./data}"
JSONL_PATH="${JSONL_PATH:-train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/optimized}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"

# Loss weights (optimized for real estate HDR enhancement)
LAMBDA_L1="${LAMBDA_L1:-100.0}"
LAMBDA_PERCEPTUAL="${LAMBDA_PERCEPTUAL:-10.0}"
LAMBDA_SSIM="${LAMBDA_SSIM:-5.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-2.0}"
LAMBDA_HIST="${LAMBDA_HIST:-1.0}"
LAMBDA_LAB="${LAMBDA_LAB:-10.0}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-5.0}"
LAMBDA_ADV="${LAMBDA_ADV:-1.0}"

echo "=============================================="
echo "AutoHDR Optimized Training"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data root:      $DATA_ROOT"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Image size:     ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch size:     $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
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
echo "  Adversarial:    $LAMBDA_ADV"
echo ""
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python3 src/training/train.py \
    --data_root "$DATA_ROOT" \
    --jsonl_path "$JSONL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$EPOCHS" \
    --lr_g 2e-4 \
    --lr_d 2e-4 \
    --lambda_l1 "$LAMBDA_L1" \
    --lambda_perceptual "$LAMBDA_PERCEPTUAL" \
    --lambda_ssim "$LAMBDA_SSIM" \
    --lambda_edge "$LAMBDA_EDGE" \
    --lambda_hist "$LAMBDA_HIST" \
    --lambda_lab "$LAMBDA_LAB" \
    --lambda_lpips "$LAMBDA_LPIPS" \
    --lambda_adv "$LAMBDA_ADV" \
    --grad_accum "$GRAD_ACCUM" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --min_lr 1e-6 \
    --use_spectral_norm \
    --num_disc_scales 2 \
    --label_smoothing 0.1 \
    --instance_noise 0.1 \
    --grad_clip_g 1.0 \
    --grad_clip_d 1.0 \
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
echo "Best model: $OUTPUT_DIR/checkpoints/best_generator.pt"
echo ""
