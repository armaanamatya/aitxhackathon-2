#!/bin/bash
#SBATCH --job-name=elite_384
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=elite_refiner_384_%j.out

echo "========================================================================"
echo "ELITE COLOR REFINER - 384 FAST TRAINING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "RESOLUTION: 384x384 (5.5x fewer pixels than 896)"
echo "BACKBONE: Restormer384 (val_loss: 0.0588)"
echo "SIMPLIFIED LOSS: 1.0×Charbonnier + 3.0×HSV"
echo "Expected Speed: ~3x faster than 896 version"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Elite Color Refiner - 384 Fast Training
# - Frozen Restormer384 backbone (excellent quality: 0.0588)
# - Trainable refiner (medium size = 1.2M params)
# - Simplified loss (no VGG overhead)
# - Much faster due to smaller resolution

/cm/local/apps/python39/bin/python3 train_elite_refiner_fast.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --backbone_path outputs_restormer_384/checkpoint_best.pt \
    --output_dir outputs_elite_refiner_384 \
    --resolution 384 \
    --refiner_size medium \
    --epochs 100 \
    --batch_size 4 \
    --grad_accum_steps 2 \
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
