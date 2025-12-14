#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=comparison_%j.out

echo "DarkIR vs Restormer Comparison"
echo "Date: $(date)"
echo "========================================"

cd /mmfs1/home/sww35/autohdr-real-estate-577

/cm/local/apps/python39/bin/python3 compare_darkir_vs_restormer.py \
    --darkir_ckpt outputs_darkir_512_cv/fold_1/checkpoint_best.pt \
    --restormer_ckpt outputs_full_light_aug/checkpoint_best.pt \
    --test_jsonl data_splits/test.jsonl \
    --output_dir test/comparison \
    --resolution 512

echo ""
echo "Comparison complete!"
echo "Date: $(date)"
