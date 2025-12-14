#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o highlight_aware_%j.out
#SBATCH -e highlight_aware_%j.err
#SBATCH -J highlight

# =============================================================================
# Highlight-Aware Training for Real Estate HDR Enhancement
# =============================================================================
#
# Based on error analysis findings:
#   - Highlights have 46.5x MORE error density than midtones
#   - Highlights are only 2-3% of pixels but dominate error
#   - Hue error in highlights is 1.8-2.7x higher than average
#   - Standard L1 loss masks this problem
#
# Solution: Highlight-weighted losses that:
#   1. Detect ALL bright regions (windows, outdoor views, bright objects)
#   2. Apply 30-50x normalized weight on highlights
#   3. Add explicit hue/saturation matching for highlights
#   4. Force model to transform highlights aggressively
#
# =============================================================================

echo "======================================================================"
echo "Highlight-Aware Training"
echo "======================================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate virtualenv if active
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV

module load python39
module load cuda11.8/toolkit/11.8.0

echo "Python: $(which python3)"
pip install einops lpips --quiet 2>/dev/null
echo ""

# =============================================================================
# Configuration (Based on Error Analysis)
# =============================================================================

# Model
MODEL_TYPE="retinexformer"      # retinexformer, restormer, nafnet
MODEL_SIZE="large"              # 3.7M params

IMAGE_SIZE=512

# Training
BATCH_SIZE=2
GRAD_ACCUM=4
EFFECTIVE_BATCH=8

NUM_EPOCHS=100
LR="2e-4"
WARMUP_EPOCHS=5
EARLY_STOPPING=20

# Loss preset: 'aggressive' based on 46x error density finding
LOSS_PRESET="aggressive"

# Data
SPLIT_DIR="data_splits"
FOLD=1
OUTPUT_DIR="outputs_highlight_aware"

NUM_WORKERS=8
SEED=42

# =============================================================================
# Verify data
# =============================================================================

echo "Verifying data splits..."
TEST_COUNT=$(wc -l < "${SPLIT_DIR}/test.jsonl")
TRAIN_COUNT=$(wc -l < "${SPLIT_DIR}/fold_${FOLD}/train.jsonl")
VAL_COUNT=$(wc -l < "${SPLIT_DIR}/fold_${FOLD}/val.jsonl")

echo "  Test: ${TEST_COUNT}, Train: ${TRAIN_COUNT}, Val: ${VAL_COUNT}"
echo ""

# =============================================================================
# Print configuration
# =============================================================================

echo "Configuration:"
echo "  Model: ${MODEL_TYPE}-${MODEL_SIZE}"
echo "  Resolution: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}"
echo "  Loss Preset: ${LOSS_PRESET}"
echo ""
echo "Based on Error Analysis:"
echo "  - Highlights have 46.5x higher error DENSITY than midtones"
echo "  - Using 4x highlight weight + hue/saturation losses"
echo "  - Brightness threshold: 0.50 (aggressive detection)"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"
echo ""

# =============================================================================
# Test imports
# =============================================================================

echo "Testing imports..."
python3 -c "
from src.training.retinexformer import create_retinexformer, count_parameters
from src.training.highlight_aware_losses import create_highlight_aware_loss

model = create_retinexformer('${MODEL_SIZE}')
print(f'  Model: {count_parameters(model)/1e6:.2f}M params')

loss_fn = create_highlight_aware_loss('${LOSS_PRESET}')
print(f'  Loss: ${LOSS_PRESET} preset loaded')
print('  All imports OK!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Import test failed!"
    exit 1
fi

echo ""

# =============================================================================
# Run training
# =============================================================================

python3 src/training/train_window_aware.py \
    --data_root "." \
    --split_dir "${SPLIT_DIR}" \
    --fold ${FOLD} \
    --output_dir "${OUTPUT_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --model_size "${MODEL_SIZE}" \
    --image_size ${IMAGE_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${GRAD_ACCUM} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --early_stopping_patience ${EARLY_STOPPING} \
    --loss_preset "${LOSS_PRESET}" \
    --num_workers ${NUM_WORKERS} \
    --save_every 10 \
    --sample_every 5 \
    --seed ${SEED} \
    --use_amp

TRAIN_EXIT=$?

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "======================================================================"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Training Complete!"
    echo ""
    if [ -f "${OUTPUT_DIR}/checkpoint_best.pt" ]; then
        ls -lh ${OUTPUT_DIR}/checkpoint_best.pt
    fi
    echo ""
    echo "Test results:"
    if [ -f "${OUTPUT_DIR}/test_results.json" ]; then
        cat ${OUTPUT_DIR}/test_results.json | python3 -m json.tool | head -20
    fi
else
    echo "Training failed with exit code: $TRAIN_EXIT"
    echo "Check: tail -100 highlight_aware_${SLURM_JOB_ID}.err"
fi
echo ""
echo "End time: $(date)"
echo "======================================================================"
