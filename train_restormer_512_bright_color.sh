#!/bin/bash
#SBATCH --job-name=rest_bcolor
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=restormer_512_bright_color_%j.out

echo "========================================================================"
echo "RESTORMER 512 - ENHANCED BRIGHT COLOR LOSS"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "FOCUS: Better matching colors in bright regions (windows, sky, plants)"
echo ""
echo "LOSS COMPONENTS:"
echo "  - L1 (1.0): Primary pixel accuracy"
echo "  - Window (0.3): Extra weight on bright regions"
echo "  - BrightColor (0.5): Color accuracy in bright regions"
echo "    - Hue: Exact color type (blue sky, green plants)"
echo "    - Chroma: Colorfulness/saturation"
echo "    - RGB: Per-channel accuracy in bright regions"
echo "    - Histogram: Color distribution matching"
echo ""
echo "DATA SPLIT:"
echo "  - TEST: First 10 images (HELD OUT)"
echo "  - TRAIN: 511 images (90%)"
echo "  - VAL: 56 images (10%)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Verify data splits exist
if [ ! -f "data_splits/proper_split/train.jsonl" ]; then
    echo "ERROR: Proper data splits not found!"
    echo "Run: python3 create_proper_splits.py"
    exit 1
fi

echo "Data splits verified:"
wc -l data_splits/proper_split/*.jsonl
echo ""

# Train with enhanced bright color loss
/cm/local/apps/python39/bin/python3 train_restormer_512_bright_color.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_restormer_512_bright_color \
    --resolution 512 \
    --l1_weight 1.0 \
    --window_weight 0.3 \
    --bright_color_weight 0.5 \
    --brightness_threshold 0.4 \
    --epochs 100 \
    --batch_size 2 \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --num_workers 4

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================================================"
