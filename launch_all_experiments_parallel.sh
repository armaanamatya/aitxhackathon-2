#!/bin/bash
# Launch all 5 preprocessing experiments in parallel (5 separate jobs)

RESOLUTION=512
BATCH_SIZE=16
EPOCHS=50

echo "Launching 5 preprocessing experiments in parallel..."

# Experiment 1: Baseline
sbatch --job-name=exp_baseline --output=exp_baseline_%j.out --wrap="\
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train_cleaned.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --preprocess none \
    --output_dir outputs_cleaned_baseline \
    --mixed_precision"

# Experiment 2: Light aug
sbatch --job-name=exp_light_aug --output=exp_light_aug_%j.out --wrap="\
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train_cleaned.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --preprocess light_aug \
    --output_dir outputs_cleaned_light_aug \
    --mixed_precision"

# Experiment 3: Standard aug
sbatch --job-name=exp_std_aug --output=exp_std_aug_%j.out --wrap="\
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train_cleaned.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --preprocess standard_aug \
    --output_dir outputs_cleaned_standard_aug \
    --mixed_precision"

# Experiment 4: Normalize exposure
sbatch --job-name=exp_norm_exp --output=exp_norm_exp_%j.out --wrap="\
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train_cleaned.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --preprocess normalize_exposure \
    --output_dir outputs_cleaned_normalize_exp \
    --mixed_precision"

# Experiment 5: Histogram matching
sbatch --job-name=exp_histogram --output=exp_histogram_%j.out --wrap="\
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train_cleaned.jsonl \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --preprocess histogram_match \
    --output_dir outputs_cleaned_histogram \
    --mixed_precision"

echo "All 5 experiments submitted!"
echo "Monitor with: squeue -u \$USER"
