#!/bin/bash
#SBATCH --job-name=ab_compare
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=ab_comparison_%j.out

echo "========================================================================"
echo "A/B COMPARISON: BACKBONE vs BACKBONE + ELITE REFINER"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Check if refiner checkpoint exists
if [ ! -f "outputs_elite_refiner_512/checkpoint_best.pt" ]; then
    echo "ERROR: Elite Refiner 512 checkpoint not found!"
    echo "       Training may still be in progress."
    echo "       Check job status with: squeue -u $USER"
    exit 1
fi

echo "Found checkpoints:"
ls -la outputs_full_baseline/checkpoint_best.pt
ls -la outputs_elite_refiner_512/checkpoint_best.pt
echo ""

# Run comparison
/cm/local/apps/python39/bin/python3 compare_baseline_vs_refiner.py \
    --backbone_path outputs_full_baseline/checkpoint_best.pt \
    --refiner_path outputs_elite_refiner_512/checkpoint_best.pt \
    --data_jsonl data_splits/fold_1/val.jsonl \
    --num_samples 10 \
    --resolution 512 \
    --output_dir comparison_baseline_vs_refiner

echo ""
echo "========================================================================"
echo "Comparison complete!"
echo "Results: comparison_baseline_vs_refiner/"
echo "Date: $(date)"
echo "========================================================================"
