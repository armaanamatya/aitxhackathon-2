#!/bin/bash
#SBATCH --job-name=elite_fast
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=elite_refiner_fast_%j.out

echo "========================================================================"
echo "ELITE COLOR REFINER - FAST TRAINING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "SIMPLIFIED LOSS: 1.0×Charbonnier + 3.0×HSV"
echo "Expected Speed: 2.5 sec/batch (40% faster than full version)"
echo "Removed: VGG Perceptual, FFT, Histogram"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Elite Color Refiner - Fast Training
# - Frozen Restormer896 backbone
# - Trainable refiner (medium size = 1.2M params)
# - Simplified loss (no VGG overhead)
# - 40% faster than full version

/cm/local/apps/python39/bin/python3 train_elite_refiner_fast.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --backbone_path outputs_restormer_896/checkpoint_best.pt \
    --output_dir outputs_elite_refiner_fast \
    --resolution 896 \
    --refiner_size medium \
    --epochs 100 \
    --batch_size 2 \
    --grad_accum_steps 4 \
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
