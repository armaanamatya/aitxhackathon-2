#!/bin/bash
#SBATCH --job-name=elite_comb
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=elite_refiner_combined_%j.out

echo "========================================================================"
echo "ELITE COLOR REFINER - COMBINED LOSS (SAME AS BACKBONE)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "DATA SPLIT (SAME as backbone):"
echo "  - TEST: First 10 images (HELD OUT)"
echo "  - TRAIN: 511 images (90%)"
echo "  - VAL: 56 images (10%)"
echo ""
echo "BACKBONE: outputs_restormer_512_combined"
echo "LOSS (SAME): L1(1.0) + Window(0.5) + BrightRegionSaturation(0.3)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Verify backbone exists
BACKBONE_PATH="outputs_restormer_512_combined/checkpoint_best.pt"
if [ ! -f "$BACKBONE_PATH" ]; then
    echo "ERROR: Backbone checkpoint not found: $BACKBONE_PATH"
    echo "       Train Restormer 512 Combined first!"
    echo "       Run: sbatch train_restormer_512_combined.sh"
    exit 1
fi

# Verify data splits exist
if [ ! -f "data_splits/proper_split/train.jsonl" ]; then
    echo "ERROR: Proper data splits not found!"
    echo "Run: python3 create_proper_splits.py"
    exit 1
fi

echo "Backbone checkpoint:"
ls -la $BACKBONE_PATH
echo ""

echo "Data splits:"
wc -l data_splits/proper_split/*.jsonl
echo ""

# Elite Color Refiner with SAME loss as backbone
# - Uses SAME data split (fair comparison)
# - Uses SAME loss function (L1 + Window + BrightRegionSaturation)
# - Frozen backbone, trainable refiner only
/cm/local/apps/python39/bin/python3 train_elite_refiner_combined.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --backbone_path $BACKBONE_PATH \
    --output_dir outputs_elite_refiner_combined \
    --resolution 512 \
    --refiner_size medium \
    --epochs 100 \
    --batch_size 2 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --grad_clip 1.0 \
    --patience 15 \
    --num_workers 4

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================================================"
