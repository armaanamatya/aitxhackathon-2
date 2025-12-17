#!/bin/bash
#SBATCH --job-name=sota_window
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# ============================================================================
# SOTA WINDOW RECOVERY TRAINING
# ============================================================================
# Uses learned luminance attention maps (heat maps) for:
# - Adaptive exposure zone detection (not fixed thresholds)
# - Zone-specific losses (different treatment per zone)
# - Boundary preservation at window frames
# - Multi-scale processing
#
# Key for GENERALIZATION:
# - Learned attention adapts to different scenes
# - Multi-scale processing captures various window sizes
# - Data augmentation increases robustness
# - Longer training with cosine annealing
# ============================================================================

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate any conda/venv - use system Python with working torch
deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

# Use system Python which has working torch + torchvision
export PATH="/cm/local/apps/python39/bin:$PATH"

echo "============================================================"
echo "SOTA Window Recovery Training"
echo "============================================================"
echo "Start time: $(date)"
echo "Python: $(which python3)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"

# Set environment for optimal training
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run efficient training with adaptive zone loss
# Uses percentile-based thresholds + color direction loss
python3 -u train_efficient_window.py \
    --resolution 512 \
    --batch_size 4 \
    --accum_steps 1 \
    --epochs 100 \
    --lr 2e-4 \
    --preset aggressive \
    --output_dir outputs_efficient_window \
    2>&1 | tee outputs_efficient_window/training.log

echo "============================================================"
echo "Training Complete!"
echo "End time: $(date)"
echo "============================================================"
