#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o retinexformer_full_%j.out
#SBATCH -e retinexformer_full_%j.err
#SBATCH -J retinex_full

# =============================================================================
# Retinexformer-Large Full Training with Proper Data Splits
# =============================================================================
#
# Data Split Strategy (NO DATA LEAKAGE):
#   - Test: 10 samples held out (data_splits/test.jsonl)
#   - Train: 511 samples (data_splits/fold_1/train.jsonl)
#   - Val: 56 samples (data_splits/fold_1/val.jsonl)
#   - Total: 577 samples = 10 + 511 + 56
#
# Why Retinexformer-Large for Real Estate HDR:
#   - Physics-based Retinex theory: separates illumination from reflectance
#   - Illumination-Guided MSA: different processing for windows vs interior
#   - 3.7M params: small enough for 567 training samples, avoids overfitting
#   - Designed for low-light/HDR enhancement (ICCV 2023 + ECCV 2024)
#
# HDR Losses for Window Preservation:
#   - Highlight loss: preserves bright regions (windows)
#   - Gradient loss: prevents edge artifacts/cracks
#   - Laplacian: multi-scale edge preservation
#   - SSIM: structural similarity
#
# =============================================================================

echo "======================================================================"
echo "Retinexformer-Large Full Training (Proper Splits)"
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

# Model
MODEL_SIZE="large"          # large = 3.7M params (optimal for 567 samples)
IMAGE_SIZE=512              # Training resolution

# Training
BATCH_SIZE=2                # Per-GPU batch size
GRAD_ACCUM=4                # Gradient accumulation
EFFECTIVE_BATCH=8           # 2 * 4 = 8

NUM_EPOCHS=100              # Max epochs
LR="2e-4"                   # Retinexformer prefers slightly higher LR
WARMUP_EPOCHS=5             # LR warmup
EARLY_STOPPING=20           # Stop if no improvement

# Standard losses
LAMBDA_L1=1.0
LAMBDA_VGG=0.1
LAMBDA_LPIPS=0.05           # Lower for Retinexformer

# HDR losses (KEY for window preservation)
LAMBDA_GRADIENT=0.15        # Edge preservation
LAMBDA_HIGHLIGHT=0.25       # Window preservation (CRITICAL)
LAMBDA_LAPLACIAN=0.1        # Multi-scale edges
LAMBDA_SSIM=0.1             # Structural similarity
HIGHLIGHT_THRESH=0.3        # Brightness threshold

# Data splits (pre-created, no leakage)
DATA_ROOT="."
SPLIT_DIR="data_splits"
FOLD=1                      # Using fold 1

OUTPUT_DIR="outputs_retinexformer_full"

# Other
NUM_WORKERS=8
SEED=42

# =============================================================================
# Verify data splits exist
# =============================================================================

echo "Verifying data splits..."
if [ ! -f "${SPLIT_DIR}/test.jsonl" ]; then
    echo "ERROR: ${SPLIT_DIR}/test.jsonl not found!"
    exit 1
fi
if [ ! -f "${SPLIT_DIR}/fold_${FOLD}/train.jsonl" ]; then
    echo "ERROR: ${SPLIT_DIR}/fold_${FOLD}/train.jsonl not found!"
    exit 1
fi
if [ ! -f "${SPLIT_DIR}/fold_${FOLD}/val.jsonl" ]; then
    echo "ERROR: ${SPLIT_DIR}/fold_${FOLD}/val.jsonl not found!"
    exit 1
fi

# Count samples
TEST_COUNT=$(wc -l < "${SPLIT_DIR}/test.jsonl")
TRAIN_COUNT=$(wc -l < "${SPLIT_DIR}/fold_${FOLD}/train.jsonl")
VAL_COUNT=$(wc -l < "${SPLIT_DIR}/fold_${FOLD}/val.jsonl")

echo "  Test samples: ${TEST_COUNT} (held out)"
echo "  Train samples: ${TRAIN_COUNT}"
echo "  Val samples: ${VAL_COUNT}"
echo "  Total: $((TEST_COUNT + TRAIN_COUNT + VAL_COUNT))"
echo ""

# =============================================================================
# Print configuration
# =============================================================================

echo "Configuration:"
echo "  Model: Retinexformer-${MODEL_SIZE} (3.7M params)"
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
echo "  Highlight (windows): ${LAMBDA_HIGHLIGHT} <- KEY"
echo "  Laplacian: ${LAMBDA_LAPLACIAN}"
echo "  SSIM: ${LAMBDA_SSIM}"
echo ""
echo "Data Splits (fold ${FOLD}):"
echo "  Test: ${SPLIT_DIR}/test.jsonl"
echo "  Train: ${SPLIT_DIR}/fold_${FOLD}/train.jsonl"
echo "  Val: ${SPLIT_DIR}/fold_${FOLD}/val.jsonl"
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

python3 src/training/train_retinexformer_full.py \
    --data_root "${DATA_ROOT}" \
    --split_dir "${SPLIT_DIR}" \
    --fold ${FOLD} \
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
    echo "Training Complete!"
    echo ""
    echo "Results:"
    if [ -f "${OUTPUT_DIR}/checkpoint_best.pt" ]; then
        ls -lh ${OUTPUT_DIR}/checkpoint_best.pt
    fi
    echo ""
    echo "Training history:"
    echo "  cat ${OUTPUT_DIR}/history.json | python3 -m json.tool"
    echo ""
    echo "Test results (10 held-out samples):"
    if [ -f "${OUTPUT_DIR}/test_results.json" ]; then
        cat ${OUTPUT_DIR}/test_results.json | python3 -m json.tool
    fi
    echo ""
    echo "Samples:"
    ls -lt ${OUTPUT_DIR}/samples/*.jpg 2>/dev/null | head -5
else
    echo "Training failed with exit code: $TRAIN_EXIT"
    echo ""
    echo "Check logs:"
    echo "  tail -100 retinexformer_full_${SLURM_JOB_ID}.err"
fi
echo ""
echo "End time: $(date)"
echo "======================================================================"
