#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o restormer_hdr_512_%j.out
#SBATCH -e restormer_hdr_512_%j.err
#SBATCH -J rest_hdr

# =============================================================================
# Restormer 512x512 with HDR Losses for Real Estate Photo Enhancement
# =============================================================================
#
# Key Features:
#   - HDR-specific losses for window preservation
#   - Gradient loss to prevent cracks/artifacts at edges
#   - Highlight loss to preserve bright regions (windows)
#   - Laplacian pyramid for multi-scale edge preservation
#   - PSNR/SSIM tracking with early stopping
#   - Warmup + cosine annealing LR schedule
#
# Expected Results:
#   - Val PSNR: 28-32 dB
#   - Better window preservation than standard L1+VGG
#   - Fewer artifacts at high-contrast edges
#
# =============================================================================

echo "============================================================"
echo "Restormer 512x512 with HDR Losses"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Navigate to project
cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate any virtualenv
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Model settings
MODEL_SIZE="base"           # tiny, small, base, large
IMAGE_SIZE=512              # Training resolution

# Training settings
BATCH_SIZE=2                # Batch size per GPU
GRAD_ACCUM=4                # Gradient accumulation steps
EFFECTIVE_BATCH=8           # BATCH_SIZE * GRAD_ACCUM

NUM_EPOCHS=100              # Maximum epochs
LR="1e-4"                   # Initial learning rate
WARMUP_EPOCHS=5             # LR warmup epochs
EARLY_STOPPING=20           # Stop if no improvement for N epochs

# Standard loss weights
LAMBDA_L1=1.0               # Pixel-level L1 loss
LAMBDA_VGG=0.1              # VGG perceptual loss
LAMBDA_LPIPS=0.1            # LPIPS perceptual loss

# HDR-specific loss weights (KEY FOR WINDOW PRESERVATION)
LAMBDA_GRADIENT=0.15        # Edge preservation (prevents cracks)
LAMBDA_HIGHLIGHT=0.2        # Window/highlight preservation
LAMBDA_LAPLACIAN=0.1        # Multi-scale edge preservation
LAMBDA_SSIM=0.1             # Structural similarity

HIGHLIGHT_THRESH=0.3        # Brightness threshold for highlight loss (in [-1,1] space)

# Data
DATA_ROOT="."
JSONL_PATH="train.jsonl"
OUTPUT_DIR="outputs_restormer_hdr_512"

# Other
NUM_WORKERS=8
SEED=42

# =============================================================================
# Print configuration
# =============================================================================

echo "Training Configuration:"
echo "  Model: Restormer-${MODEL_SIZE}"
echo "  Resolution: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch size: ${BATCH_SIZE} x ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}"
echo "  Epochs: ${NUM_EPOCHS} (early stopping: ${EARLY_STOPPING})"
echo "  Learning rate: ${LR} (warmup: ${WARMUP_EPOCHS} epochs)"
echo ""
echo "Loss Weights:"
echo "  L1: ${LAMBDA_L1}"
echo "  VGG: ${LAMBDA_VGG}"
echo "  LPIPS: ${LAMBDA_LPIPS}"
echo "  Gradient (edge): ${LAMBDA_GRADIENT}"
echo "  Highlight (window): ${LAMBDA_HIGHLIGHT}"
echo "  Laplacian (multi-scale): ${LAMBDA_LAPLACIAN}"
echo "  SSIM: ${LAMBDA_SSIM}"
echo "  Highlight threshold: ${HIGHLIGHT_THRESH}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# =============================================================================
# Run training
# =============================================================================

python3 src/training/train_restormer_hdr.py \
    --data_root "${DATA_ROOT}" \
    --jsonl_path "${JSONL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_size "${MODEL_SIZE}" \
    --image_size ${IMAGE_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${GRAD_ACCUM} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --early_stopping_patience ${EARLY_STOPPING} \
    --lambda_l1 ${LAMBDA_L1} \
    --lambda_vgg ${LAMBDA_VGG} \
    --lambda_lpips ${LAMBDA_LPIPS} \
    --lambda_gradient ${LAMBDA_GRADIENT} \
    --lambda_highlight ${LAMBDA_HIGHLIGHT} \
    --lambda_laplacian ${LAMBDA_LAPLACIAN} \
    --lambda_ssim ${LAMBDA_SSIM} \
    --highlight_threshold ${HIGHLIGHT_THRESH} \
    --num_workers ${NUM_WORKERS} \
    --save_every 10 \
    --sample_every 5 \
    --seed ${SEED}

TRAIN_EXIT=$?

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✅ Training Complete!"
    echo ""
    echo "Outputs:"
    ls -lh ${OUTPUT_DIR}/checkpoint_best.pt 2>/dev/null || echo "  (no best checkpoint yet)"
    echo ""
    echo "View training history:"
    echo "  cat ${OUTPUT_DIR}/history.json | python3 -m json.tool"
    echo ""
    echo "Latest samples:"
    ls -lt ${OUTPUT_DIR}/samples/*.jpg 2>/dev/null | head -5
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT"
fi
echo ""
echo "End time: $(date)"
echo "============================================================"
