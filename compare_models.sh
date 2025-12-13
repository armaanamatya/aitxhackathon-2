#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o compare_models_%j.out
#SBATCH -e compare_models_%j.err
#SBATCH -J compare

# =============================================================================
# Model Comparison Script
# =============================================================================
# Compares all trained models:
# - Restormer (scratch) vs SwinRestormer (fine-tuned)
# - DAT, MambaDiffusion, INRetouch, HAT
#
# Metrics: L1, PSNR, SSIM, LPIPS, Inference Time
# =============================================================================

echo "=========================================="
echo "Model Comparison"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

# Run comparison
python3 src/training/compare_models.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --image_size 256 \
    --batch_size 1 \
    --output_dir comparison_results \
    --restormer_ckpt outputs_restormer/checkpoint_best.pt \
    --swin_restormer_ckpt outputs_swin_restormer/checkpoint_best.pt \
    --inretouch_ckpt outputs_inretouch/checkpoint_best.pt \
    --dat_ckpt outputs_dat/checkpoint_best.pt \
    --mamba_ckpt outputs_mamba/checkpoint_best.pt

echo "=========================================="
echo "Comparison Complete"
echo "End time: $(date)"
echo "=========================================="
