#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o nafnet_comprehensive_%j.out
#SBATCH -e nafnet_comprehensive_%j.err
#SBATCH -J nafnet

# =============================================================================
# NAFNet Comprehensive Training - Top 0.0001% MLE Solution
# =============================================================================
#
# Features:
#   - Highlight-aware losses (based on 46x error density finding)
#   - Same train/val/test splits for fair comparison
#   - Standardized output format for comparison with Restormer/Retinexformer
#   - Edge-aware loss for artifact prevention
#
# =============================================================================

echo "======================================================================"
echo "NAFNet Comprehensive Training"
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

MODEL_SIZE="base"           # NAFNet-base (~17M params)
IMAGE_SIZE=512
BATCH_SIZE=4
GRAD_ACCUM=2
EFFECTIVE_BATCH=8

NUM_EPOCHS=100
LR="1e-3"                   # NAFNet uses higher LR
WARMUP=5
EARLY_STOP=15

# Loss weights (based on error analysis)
LAMBDA_HIGHLIGHT=4.0        # 46x error density -> 4x weight
LAMBDA_GRADIENT=0.15
LAMBDA_COLOR=0.5

SPLIT_DIR="data_splits"
FOLD=1
OUTPUT_DIR="outputs_nafnet_comprehensive"

# =============================================================================
# Verify splits
# =============================================================================

echo "Data splits:"
echo "  Test: $(wc -l < ${SPLIT_DIR}/test.jsonl)"
echo "  Train: $(wc -l < ${SPLIT_DIR}/fold_${FOLD}/train.jsonl)"
echo "  Val: $(wc -l < ${SPLIT_DIR}/fold_${FOLD}/val.jsonl)"
echo ""
echo "Configuration:"
echo "  Model: NAFNet-${MODEL_SIZE}"
echo "  Resolution: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}"
echo "  Highlight weight: ${LAMBDA_HIGHLIGHT}x"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================================================"

# =============================================================================
# Test imports
# =============================================================================

echo "Testing imports..."
python3 -c "
from src.training.training.nafnet import create_nafnet
model = create_nafnet('${MODEL_SIZE}')
params = sum(p.numel() for p in model.parameters())
print(f'  NAFNet-${MODEL_SIZE}: {params/1e6:.2f}M params')
print('  All imports OK!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Import test failed!"
    exit 1
fi

# =============================================================================
# Run training
# =============================================================================

python3 src/training/train_nafnet_comprehensive.py \
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
    --lambda_highlight ${LAMBDA_HIGHLIGHT} \
    --lambda_gradient ${LAMBDA_GRADIENT} \
    --lambda_color ${LAMBDA_COLOR} \
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
