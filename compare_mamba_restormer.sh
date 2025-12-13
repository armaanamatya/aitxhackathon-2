#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o compare_mamba_restormer_%j.out
#SBATCH -e compare_mamba_restormer_%j.err
#SBATCH -J compare

# =============================================================================
# Comprehensive Model Comparison
# MambaDiffusion vs Restormer-Base (both at 128x128)
# =============================================================================
# Creates:
# - Side-by-side comparison images
# - Difference maps
# - Per-image metrics CSV
# - Statistical analysis
# - Detailed comparison log
# =============================================================================

echo "=========================================="
echo "Model Comparison: Mamba vs Restormer"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

# Install dependencies if needed
python3 -c "import lpips" || pip install lpips --user --quiet
python3 -c "from skimage.metrics import structural_similarity" || pip install scikit-image --user --quiet

# =============================================================================
# Run Comparison
# =============================================================================
# Using 128x128 for both models (fair comparison)
# Outputs to compare_restformerbase_mamba/
# =============================================================================

python3 src/training/compare_mamba_restormer.py \
    --restormer_ckpt outputs_restormer_128/checkpoint_best.pt \
    --mamba_ckpt outputs_mamba/checkpoint_best.pt \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir compare_restformerbase_mamba \
    --restormer_size 128 \
    --mamba_size 128 \
    --comparison_size 256

COMPARE_EXIT=$?

if [ $COMPARE_EXIT -eq 0 ]; then
    echo "=========================================="
    echo "Comparison Complete!"
    echo "=========================================="
    echo "Results saved to: compare_restformerbase_mamba/"
    echo ""
    echo "Contents:"
    echo "  - comparisons/: Side-by-side images (source, target, restormer, mamba)"
    echo "  - difference_maps/: Amplified error visualizations"
    echo "  - individual/: Per-model output images"
    echo "  - per_image_results.csv: Metrics for each image"
    echo "  - comparison_summary.json: Statistical summary"
    echo "  - comparison_log.txt: Detailed analysis"
    echo ""

    # Print summary if available
    if [ -f compare_restformerbase_mamba/comparison_summary.json ]; then
        echo "Quick Summary:"
        python3 -c "
import json
with open('compare_restformerbase_mamba/comparison_summary.json') as f:
    data = json.load(f)
print(f\"Overall Winner: {data['overall_winner']}\")
print(f\"Restormer PSNR: {data['metrics_stats']['PSNR']['restormer']['mean']:.2f} dB\")
print(f\"Mamba PSNR: {data['metrics_stats']['PSNR']['mamba']['mean']:.2f} dB\")
print(f\"Restormer SSIM: {data['metrics_stats']['SSIM']['restormer']['mean']:.4f}\")
print(f\"Mamba SSIM: {data['metrics_stats']['SSIM']['mamba']['mean']:.4f}\")
print(f\"Restormer Inference: {data['timing']['restormer_mean']*1000:.2f}ms\")
print(f\"Mamba Inference: {data['timing']['mamba_mean']*1000:.2f}ms\")
"
    fi
else
    echo "=========================================="
    echo "ERROR: Comparison failed with exit code $COMPARE_EXIT"
    echo "=========================================="
fi

echo "End time: $(date)"
echo "=========================================="
