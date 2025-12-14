#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o retinexformer_hdr_512_%j.out
#SBATCH -e retinexformer_hdr_512_%j.err
#SBATCH -J retinex

# =============================================================================
# Retinexformer 512x512 with HDR Losses - Real Estate Photo Enhancement
# =============================================================================
#
# Why Retinexformer for your task:
#   - Physics-based: Retinex theory separates illumination from reflectance
#   - Illumination-guided attention: different processing for windows vs interior
#   - ICCV 2023 + ECCV 2024 enhanced: proven SOTA for low-light/HDR
#   - Very lightweight: 0.4M-3.7M params (perfect for 464 samples!)
#
# Key advantage for windows:
#   - Illumination Estimator learns to identify bright regions (windows)
#   - IG-MSA attention uses this to treat them differently
#   - Result: Windows preserved while lifting shadows
#
# =============================================================================

echo "======================================================================"
echo "Retinexformer 512x512 with HDR Losses"
echo "======================================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Navigate to project
cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate virtualenv if active
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Install einops if needed (required for Retinexformer)
pip install einops --quiet 2>/dev/null || true
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Model (very lightweight - great for small dataset!)
MODEL_SIZE="large"          # tiny (0.4M), small (1M), base (1.6M), large (3.7M)
IMAGE_SIZE=512

# Training
BATCH_SIZE=2
GRAD_ACCUM=4
EFFECTIVE_BATCH=8

NUM_EPOCHS=100
LR="2e-4"                   # Retinexformer uses slightly higher LR
WARMUP_EPOCHS=5
EARLY_STOPPING=20

# Standard losses
LAMBDA_L1=1.0
LAMBDA_VGG=0.1
LAMBDA_LPIPS=0.05           # Lower LPIPS for Retinexformer

# HDR losses (KEY for window preservation)
LAMBDA_GRADIENT=0.15        # Edge preservation
LAMBDA_HIGHLIGHT=0.25       # Window preservation (higher for Retinexformer!)
LAMBDA_LAPLACIAN=0.1        # Multi-scale edges
LAMBDA_SSIM=0.1             # Structural similarity
HIGHLIGHT_THRESH=0.3        # Brightness threshold

# Data
DATA_ROOT="."
JSONL_PATH="train.jsonl"
OUTPUT_DIR="outputs_retinexformer_hdr_512"

# Other
NUM_WORKERS=8
SEED=42

# =============================================================================
# Print configuration
# =============================================================================

echo "Configuration:"
echo "  Model: Retinexformer-${MODEL_SIZE}"
echo "  Resolution: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}"
echo "  Epochs: ${NUM_EPOCHS} (early stop: ${EARLY_STOPPING})"
echo "  LR: ${LR} (warmup: ${WARMUP_EPOCHS})"
echo ""
echo "Loss Weights (HDR-optimized):"
echo "  L1: ${LAMBDA_L1}"
echo "  VGG: ${LAMBDA_VGG}"
echo "  LPIPS: ${LAMBDA_LPIPS}"
echo "  Gradient (edges): ${LAMBDA_GRADIENT}"
echo "  Highlight (windows): ${LAMBDA_HIGHLIGHT} ← KEY"
echo "  Laplacian: ${LAMBDA_LAPLACIAN}"
echo "  SSIM: ${LAMBDA_SSIM}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"
echo ""

# =============================================================================
# Test imports first
# =============================================================================

echo "Testing imports..."
python3 -c "
from src.training.retinexformer import create_retinexformer, count_parameters
from src.training.hdr_losses import HDRLoss
model = create_retinexformer('${MODEL_SIZE}')
print(f'  Retinexformer-${MODEL_SIZE}: {count_parameters(model)/1e6:.2f}M params')
print('  HDR losses: OK')
print('  All imports successful!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Import test failed!"
    exit 1
fi

echo ""

# =============================================================================
# Run training
# =============================================================================

python3 src/training/train_retinexformer_hdr.py \
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
echo "======================================================================"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✅ Training Complete!"
    echo ""
    echo "Results:"
    if [ -f "${OUTPUT_DIR}/checkpoint_best.pt" ]; then
        ls -lh ${OUTPUT_DIR}/checkpoint_best.pt
    fi
    echo ""
    echo "View history:"
    echo "  cat ${OUTPUT_DIR}/history.json | python3 -m json.tool"
    echo ""
    echo "Samples:"
    ls -lt ${OUTPUT_DIR}/samples/*.jpg 2>/dev/null | head -5
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT"
    echo ""
    echo "Check logs:"
    echo "  tail -100 retinexformer_hdr_512_${SLURM_JOB_ID}.err"
fi
echo ""
echo "End time: $(date)"
echo "======================================================================"
