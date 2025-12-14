#!/bin/bash
#SBATCH --job-name=preproc_exp
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=preproc_experiments_%j.out

echo "Preprocessing Ablation Study (Baseline Last)"
echo "========================================"
date

# Base settings
RESOLUTION=512
BATCH_SIZE=4
EPOCHS=50
EARLY_STOP_PATIENCE=10  # Stop if no improvement for 10 epochs (20% of total)
BASE_DIR="/mmfs1/home/sww35/autohdr-real-estate-577"

# Experiment 1: Light augmentation
echo ""
echo "Experiment 1: Light augmentation (flip only)"
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOP_PATIENCE \
    --preprocess light_aug \
    --output_dir outputs_full_light_aug \
    --mixed_precision

# Experiment 2: Standard augmentation
echo ""
echo "Experiment 2: Standard augmentation (flip + rotation)"
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOP_PATIENCE \
    --preprocess standard_aug \
    --output_dir outputs_full_standard_aug \
    --mixed_precision

# Experiment 3: Exposure normalization
echo ""
echo "Experiment 3: Exposure normalization"
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOP_PATIENCE \
    --preprocess normalize_exposure \
    --output_dir outputs_full_normalize_exp \
    --mixed_precision

# Experiment 4: Histogram matching
echo ""
echo "Experiment 4: Histogram matching"
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOP_PATIENCE \
    --preprocess histogram_match \
    --output_dir outputs_full_histogram \
    --mixed_precision

# Experiment 5: Baseline (no preprocessing, full data) - RUN LAST
echo ""
echo "Experiment 5: Baseline (full 577 pairs, no preprocessing) - BASELINE"
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOP_PATIENCE \
    --preprocess none \
    --output_dir outputs_full_baseline \
    --mixed_precision

echo ""
echo "========================================"
echo "All experiments complete!"
date
echo "========================================"
