#!/bin/bash
#SBATCH --job-name=rest_adv
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=restormer_512_adversarial_%j.out

echo "========================================================================"
echo "RESTORMER 512 - ADVERSARIAL + STYLE TRAINING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "PhD-Level Approach for Learning HDR Style:"
echo "  - Adversarial loss -> matches TARGET DISTRIBUTION (not average)"
echo "  - Style loss (Gram) -> captures color/texture patterns"
echo "  - Histogram loss -> exact color distribution matching"
echo ""
echo "LOSS COMPONENTS:"
echo "  - L1: 1.0 (pixel accuracy)"
echo "  - Perceptual: 0.1 (VGG features)"
echo "  - Style: 0.1 (Gram matrix)"
echo "  - Histogram: 0.1 (color distribution)"
echo "  - Adversarial: 0.01 (discriminator feedback)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

/cm/local/apps/python39/bin/python3 train_restormer_512_adversarial.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_restormer_512_adversarial \
    --resolution 512 \
    --l1_weight 1.0 \
    --perceptual_weight 0.1 \
    --style_weight 0.1 \
    --histogram_weight 0.1 \
    --adversarial_weight 0.01 \
    --epochs 100 \
    --batch_size 2 \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --num_workers 4

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
