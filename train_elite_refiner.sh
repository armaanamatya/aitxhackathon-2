#!/bin/bash
#SBATCH --job-name=elite_refiner
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=elite_refiner_%j.out

echo "========================================================================"
echo "ELITE COLOR REFINER TRAINING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Elite Color Refiner Training
# - Frozen Restormer896 backbone
# - Trainable refiner (medium size = 1.2M params)
# - Resolution: 896x896
# - Effective batch size: 2 * 4 = 8 (via gradient accumulation)
# - Mixed precision training
# - EMA for stable convergence
# - Adaptive multi-loss

/cm/local/apps/python39/bin/python3 train_elite_color_refiner.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --backbone_path outputs_restormer_896/checkpoint_best.pt \
    --output_dir outputs_elite_refiner_896 \
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
