#!/bin/bash
#SBATCH --job-name=sota_hdr
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# ============================================================================
# SOTA HDR TRAINING + AUTOMATIC INFERENCE
# ============================================================================
#
# This script:
# 1. Trains the SOTA HDR model with early stopping
# 2. Automatically runs inference on test_final/ after training
# 3. Saves outputs to sota_hdr_out/
#    - outputs/: Enhanced images
#    - comparisons/: Source | Output side-by-side
#
# Usage:
#   sbatch train_sota_hdr.sh
# ============================================================================

cd /mmfs1/home/sww35/autohdr-real-estate-577

export PATH="/cm/local/apps/python39/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "============================================================"
echo "SOTA HDR TRAINING + INFERENCE PIPELINE"
echo "============================================================"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"

# ============================================================================
# STAGE 1: TRAINING WITH EARLY STOPPING
# ============================================================================
echo ""
echo ">>> STAGE 1: TRAINING"
echo "============================================================"

python3 -u train_sota_hdr.py \
    --data_dir images \
    --output_dir outputs_sota_hdr \
    --resolution 512 \
    --batch_size 2 \
    --epochs 100 \
    --lr 2e-4 \
    --early_stopping_patience 15 \
    --min_epochs 20

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo ">>> Training completed successfully"
echo ""

# ============================================================================
# STAGE 2: INFERENCE ON TEST SET
# ============================================================================
echo ""
echo ">>> STAGE 2: INFERENCE ON TEST SET"
echo "============================================================"

# Find best checkpoint
CHECKPOINT="outputs_sota_hdr/checkpoint_best.pt"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="outputs_sota_hdr/checkpoint_last.pt"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

python3 -u infer_sota_hdr.py \
    --checkpoint "$CHECKPOINT" \
    --input_dir test_final \
    --output_dir sota_hdr_out \
    --tile_size 768 \
    --overlap 192

INFER_EXIT_CODE=$?

if [ $INFER_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Inference failed with exit code $INFER_EXIT_CODE"
    exit $INFER_EXIT_CODE
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "End: $(date)"
echo ""
echo "Training output:  outputs_sota_hdr/"
echo "  - checkpoint_best.pt"
echo "  - checkpoint_last.pt"
echo "  - history.json"
echo ""
echo "Inference output: sota_hdr_out/"
echo "  - outputs/       (enhanced images)"
echo "  - comparisons/   (source | output side-by-side)"
echo "  - inference_summary.json"
echo ""

# Count output files
if [ -d "sota_hdr_out/outputs" ]; then
    NUM_OUTPUTS=$(ls -1 sota_hdr_out/outputs/*.png 2>/dev/null | wc -l)
    echo "Generated $NUM_OUTPUTS output images"
fi

if [ -d "sota_hdr_out/comparisons" ]; then
    NUM_COMPARISONS=$(ls -1 sota_hdr_out/comparisons/*.png 2>/dev/null | wc -l)
    echo "Generated $NUM_COMPARISONS comparison images"
fi

echo "============================================================"
