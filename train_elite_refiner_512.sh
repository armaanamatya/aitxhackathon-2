#!/bin/bash
#SBATCH --job-name=elite_512
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=elite_refiner_512_%j.out

echo "========================================================================"
echo "ELITE COLOR REFINER - 512 FROZEN BACKBONE"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "BACKBONE: Restormer512 Combined (from job 609756)"
echo "RESOLUTION: 512x512"
echo "LOSS: 1.0×Charbonnier + 3.0×HSV"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

/cm/local/apps/python39/bin/python3 train_elite_refiner_fast.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --backbone_path outputs_restormer_512_combined/checkpoint_best.pt \
    --output_dir outputs_elite_refiner_512 \
    --resolution 512 \
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
