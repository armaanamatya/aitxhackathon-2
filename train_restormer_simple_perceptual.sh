#!/bin/bash
#SBATCH --job-name=rest_perc
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=restormer_perc_%j.out

echo "========================================================================"
echo "SIMPLIFIED ROBUST RESTORMER (WITH VGG PERCEPTUAL)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# With VGG perceptual loss for better texture/color
/cm/local/apps/python39/bin/python3 train_restormer_simple_robust.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --output_dir outputs_restormer_simple_perceptual \
    --resolution 512 \
    --dim 48 \
    --num_blocks 4 6 6 8 \
    --epochs 100 \
    --batch_size 2 \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --num_workers 4 \
    --use_perceptual \
    --mixed_precision

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================================================"
