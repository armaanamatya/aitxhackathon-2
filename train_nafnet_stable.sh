#!/bin/bash
#SBATCH --job-name=nafnet_stable
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=nafnet_stable_%j.out

echo "========================================================================"
echo "STABLE NAFNET TRAINING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Run stable NAFNet training
/cm/local/apps/python39/bin/python3 train_nafnet_stable.py \
    --train_jsonl data_splits/fold_1/train.jsonl \
    --val_jsonl data_splits/fold_1/val.jsonl \
    --output_dir outputs_nafnet_stable \
    --resolution 512 \
    --width 32 \
    --middle_blk_num 1 \
    --enc_blk_nums 1 1 1 28 \
    --dec_blk_nums 1 1 1 1 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --warmup_epochs 10 \
    --patience 15 \
    --grad_clip 0.5 \
    --num_workers 4

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================================================"
