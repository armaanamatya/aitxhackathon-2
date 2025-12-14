#!/bin/bash
#SBATCH --job-name=rest_colp
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=restormer_512_color_perceptual_%j.out

echo "========================================================================"
echo "RESTORMER 512 - COLOR + PERCEPTUAL + SSIM LOSS"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "FOCUS: Accurate color reproduction for ALL bright colors"
echo ""
echo "LOSS COMPONENTS:"
echo "  - L1 (1.0): Pixel accuracy"
echo "  - Perceptual (0.1): VGG feature similarity (textures, colors)"
echo "  - SSIM (0.1): Structural similarity"
echo "  - Color (0.3): HSV matching (Hue + Saturation + Value)"
echo "  - Lab (0.2): Perceptually uniform color space"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Verify data splits
if [ ! -f "data_splits/proper_split/train.jsonl" ]; then
    echo "ERROR: Data splits not found!"
    exit 1
fi

echo "Data splits:"
wc -l data_splits/proper_split/*.jsonl
echo ""

# Train with Color + Perceptual + SSIM loss
/cm/local/apps/python39/bin/python3 train_restormer_512_color_perceptual.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_restormer_512_color_perceptual \
    --resolution 512 \
    --l1_weight 1.0 \
    --perceptual_weight 0.1 \
    --ssim_weight 0.1 \
    --color_weight 0.3 \
    --lab_weight 0.2 \
    --epochs 100 \
    --batch_size 2 \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --num_workers 4

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================================================"
