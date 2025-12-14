#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o retinex_comprehensive_%j.out
#SBATCH -e retinex_comprehensive_%j.err
#SBATCH -J retinex

# =============================================================================
# Retinexformer Comprehensive Training - Top 0.0001% MLE Solution
# =============================================================================
#
# Why Retinexformer is ideal:
#   - Illumination-guided attention handles windows differently
#   - Physics-based Retinex decomposition
#   - Lightweight (3.7M) - perfect for 577 samples
#
# Optimized for:
#   - Windows/highlights (46x error density -> 4x weight)
#   - Edge preservation (gradient + edge-aware loss)
#   - Color accuracy (hue + saturation losses in highlights)
#
# =============================================================================

echo "======================================================================"
echo "Retinexformer Comprehensive Training"
echo "======================================================================"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

if [ -n "$VIRTUAL_ENV" ]; then deactivate 2>/dev/null; fi
module load python39
module load cuda11.8/toolkit/11.8.0

pip install einops lpips --quiet 2>/dev/null

# =============================================================================
# Configuration
# =============================================================================

MODEL_SIZE="large"          # Retinexformer-large (3.7M params)
IMAGE_SIZE=512
BATCH_SIZE=2
GRAD_ACCUM=4
EFFECTIVE_BATCH=8

NUM_EPOCHS=100
LR="2e-4"                   # Retinexformer prefers this LR
WARMUP=5
EARLY_STOP=15

# Loss weights (optimal based on error analysis)
LAMBDA_HIGHLIGHT_L1=4.0     # 46x error density -> 4x weight
LAMBDA_HIGHLIGHT_HUE=0.8    # 1.8-2.7x hue error in highlights
LAMBDA_HIGHLIGHT_SAT=0.6
LAMBDA_HIGHLIGHT_COLOR=1.5
LAMBDA_GRADIENT=0.2
LAMBDA_EDGE_AWARE=0.3
LAMBDA_SSIM=0.15
BRIGHTNESS_THRESH=0.50      # Catch more highlights

SPLIT_DIR="data_splits"
FOLD=1
OUTPUT_DIR="outputs_retinexformer_comprehensive"

# =============================================================================
# Verify splits
# =============================================================================

echo "Data splits:"
echo "  Test: $(wc -l < ${SPLIT_DIR}/test.jsonl)"
echo "  Train: $(wc -l < ${SPLIT_DIR}/fold_${FOLD}/train.jsonl)"
echo "  Val: $(wc -l < ${SPLIT_DIR}/fold_${FOLD}/val.jsonl)"
echo ""
echo "Configuration:"
echo "  Model: Retinexformer-${MODEL_SIZE}"
echo "  Resolution: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}"
echo ""
echo "Loss weights (optimal for windows/highlights/edges):"
echo "  Highlight L1: ${LAMBDA_HIGHLIGHT_L1}x"
echo "  Highlight Hue: ${LAMBDA_HIGHLIGHT_HUE}"
echo "  Highlight Saturation: ${LAMBDA_HIGHLIGHT_SAT}"
echo "  Gradient: ${LAMBDA_GRADIENT}"
echo "  Edge-aware: ${LAMBDA_EDGE_AWARE}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"

# =============================================================================
# Test imports
# =============================================================================

echo "Testing imports..."
python3 -c "
from src.training.retinexformer import create_retinexformer, count_parameters
model = create_retinexformer('${MODEL_SIZE}')
print(f'  Retinexformer-${MODEL_SIZE}: {count_parameters(model)/1e6:.2f}M params')
print('  All imports OK!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Import test failed!"
    exit 1
fi

# =============================================================================
# Run training
# =============================================================================

python3 src/training/train_retinexformer_comprehensive.py \
    --data_root "." \
    --split_dir "${SPLIT_DIR}" \
    --fold ${FOLD} \
    --output_dir "${OUTPUT_DIR}" \
    --model_size "${MODEL_SIZE}" \
    --image_size ${IMAGE_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${GRAD_ACCUM} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --warmup_epochs ${WARMUP} \
    --early_stopping_patience ${EARLY_STOP} \
    --lambda_highlight_l1 ${LAMBDA_HIGHLIGHT_L1} \
    --lambda_highlight_hue ${LAMBDA_HIGHLIGHT_HUE} \
    --lambda_highlight_saturation ${LAMBDA_HIGHLIGHT_SAT} \
    --lambda_highlight_color ${LAMBDA_HIGHLIGHT_COLOR} \
    --lambda_gradient ${LAMBDA_GRADIENT} \
    --lambda_edge_aware ${LAMBDA_EDGE_AWARE} \
    --lambda_ssim ${LAMBDA_SSIM} \
    --brightness_threshold ${BRIGHTNESS_THRESH} \
    --num_workers 8 \
    --use_amp

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Complete!"
    echo ""
    echo "Results:"
    cat ${OUTPUT_DIR}/comparison_summary.json 2>/dev/null | python3 -m json.tool
    echo ""
    echo "Test outputs in: ${OUTPUT_DIR}/test_outputs/"
else
    echo "Training failed: $EXIT_CODE"
fi
echo "End: $(date)"
echo "======================================================================"
