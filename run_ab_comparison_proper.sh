#!/bin/bash
#SBATCH --job-name=ab_proper
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=ab_comparison_proper_%j.out

echo "========================================================================"
echo "A/B COMPARISON - PROPER SPLIT (HELD-OUT TEST SET)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""
echo "COMPARISON:"
echo "  A: Restormer 512 Combined (backbone only)"
echo "  B: Restormer 512 Combined + Elite Refiner"
echo ""
echo "TEST SET: First 10 images (NEVER seen during training)"
echo ""

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Check if checkpoints exist
BACKBONE_PATH="outputs_restormer_512_combined/checkpoint_best.pt"
REFINER_PATH="outputs_elite_refiner_combined/checkpoint_best.pt"

if [ ! -f "$BACKBONE_PATH" ]; then
    echo "ERROR: Backbone checkpoint not found: $BACKBONE_PATH"
    exit 1
fi

if [ ! -f "$REFINER_PATH" ]; then
    echo "ERROR: Refiner checkpoint not found: $REFINER_PATH"
    exit 1
fi

echo "Checkpoints found:"
ls -la $BACKBONE_PATH
ls -la $REFINER_PATH
echo ""

# Run comparison on HELD-OUT TEST SET (10 images)
/cm/local/apps/python39/bin/python3 compare_baseline_vs_refiner.py \
    --backbone_path $BACKBONE_PATH \
    --refiner_path $REFINER_PATH \
    --data_jsonl data_splits/proper_split/test.jsonl \
    --num_samples 10 \
    --resolution 512 \
    --output_dir comparison_proper_test

echo ""
echo "========================================================================"
echo "Comparison complete!"
echo "Results: comparison_proper_test/"
echo "Date: $(date)"
echo "========================================================================"
