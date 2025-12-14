#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o compare_mamba_rest_opt_%j.out
#SBATCH -e compare_mamba_rest_opt_%j.err
#SBATCH -J compare

# =============================================================================
# Comprehensive Model Comparison
# MambaDiffusion vs Optimized Restormer (both at 128x128)
# =============================================================================
# Output directory: compare_mamba_vs_restormer_opt/
#
# Creates:
# - Side-by-side comparison images (source | target | restormer | mamba)
# - Difference/error maps for each model
# - Per-image metrics CSV
# - Statistical analysis (mean, std, percentiles)
# - Detailed comparison log
# - Winner determination
# =============================================================================

echo "=========================================="
echo "Model Comparison"
echo "MambaDiffusion vs Optimized Restormer"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate any active virtualenv
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

# Verify python version
echo "Python: $(which python3)"
echo "Version: $(python3 --version)"

# Install dependencies if needed
python3 -c "import lpips" 2>/dev/null || python3 -m pip install lpips --user --quiet
python3 -c "from skimage.metrics import structural_similarity" 2>/dev/null || python3 -m pip install scikit-image --user --quiet
python3 -c "from PIL import Image" 2>/dev/null || python3 -m pip install Pillow --user --quiet

echo ""
echo "Comparing:"
echo "  - MambaDiffusion (outputs_mamba/checkpoint_best.pt)"
echo "  - Optimized Restormer (outputs_restormer_optimized/checkpoint_best.pt)"
echo "  - Both at 128x128 resolution"
echo ""

# =============================================================================
# Run Comparison
# =============================================================================

python3 src/training/compare_mamba_restormer.py \
    --restormer_ckpt outputs_restormer_optimized/checkpoint_best.pt \
    --mamba_ckpt outputs_mamba/checkpoint_best.pt \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir compare_mamba_vs_restormer_opt \
    --restormer_size 128 \
    --mamba_size 128 \
    --comparison_size 256

COMPARE_EXIT=$?

if [ $COMPARE_EXIT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Comparison Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: compare_mamba_vs_restormer_opt/"
    echo ""
    echo "Directory contents:"
    ls -la compare_mamba_vs_restormer_opt/
    echo ""
    echo "  - comparisons/     : Side-by-side images"
    echo "  - difference_maps/ : Error visualization"
    echo "  - individual/      : Per-model outputs"
    echo "  - per_image_results.csv"
    echo "  - comparison_summary.json"
    echo "  - comparison_log.txt"
    echo ""

    # Print summary
    if [ -f compare_mamba_vs_restormer_opt/comparison_summary.json ]; then
        echo "=========================================="
        echo "RESULTS SUMMARY"
        echo "=========================================="
        python3 << 'EOF'
import json

with open('compare_mamba_vs_restormer_opt/comparison_summary.json') as f:
    data = json.load(f)

print(f"\n{'='*50}")
print("METRIC COMPARISON")
print(f"{'='*50}")
print(f"{'Metric':<12} {'Restormer':<15} {'Mamba':<15} {'Winner'}")
print(f"{'-'*50}")

metrics = ['L1', 'PSNR', 'SSIM', 'LPIPS', 'DeltaE']
for m in metrics:
    r = data['metrics_stats'][m]['restormer']['mean']
    ma = data['metrics_stats'][m]['mamba']['mean']

    # Determine winner
    if m in ['PSNR', 'SSIM']:
        winner = 'Mamba' if ma > r else 'Restormer'
    else:
        winner = 'Mamba' if ma < r else 'Restormer'

    print(f"{m:<12} {r:<15.4f} {ma:<15.4f} {winner}")

print(f"\n{'='*50}")
print("INFERENCE PERFORMANCE")
print(f"{'='*50}")
print(f"Restormer: {data['timing']['restormer_mean']*1000:.2f}ms")
print(f"Mamba:     {data['timing']['mamba_mean']*1000:.2f}ms")

speedup = data['timing']['restormer_mean'] / data['timing']['mamba_mean']
faster = "Mamba" if speedup > 1 else "Restormer"
print(f"Speedup:   {max(speedup, 1/speedup):.2f}x ({faster} faster)")

print(f"\n{'='*50}")
print(f"OVERALL WINNER: {data['overall_winner']}")
print(f"{'='*50}\n")

# Model info
print(f"Model Parameters:")
print(f"  Restormer: {data['model_info']['restormer_params']:,}")
print(f"  Mamba:     {data['model_info']['mamba_params']:,}")
EOF
    fi
else
    echo "=========================================="
    echo "ERROR: Comparison failed with exit code $COMPARE_EXIT"
    echo "=========================================="

    # Check if checkpoints exist
    echo ""
    echo "Checking checkpoints:"
    ls -la outputs_mamba/checkpoint_best.pt 2>/dev/null || echo "  - Mamba checkpoint NOT FOUND"
    ls -la outputs_restormer_optimized/checkpoint_best.pt 2>/dev/null || echo "  - Restormer optimized checkpoint NOT FOUND"
fi

echo ""
echo "End time: $(date)"
echo "=========================================="
