#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_restormer_128_%j.out
#SBATCH -e train_restormer_128_%j.err
#SBATCH -J rest_128

# =============================================================================
# Restormer-Base Training at 128x128 (Fair Comparison with MambaDiffusion)
# =============================================================================
# This trains Restormer at 128x128 to fairly compare with MambaDiffusion
# which was also trained at 128x128.
# =============================================================================

echo "=========================================="
echo "Restormer-Base Training (128x128)"
echo "For fair comparison with MambaDiffusion"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Deactivate any virtual environment
unset VIRTUAL_ENV
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "autohdr_venv" | tr '\n' ':' | sed 's/:$//')

# Load modules
module load python39
module load cuda11.8/toolkit/11.8.0

# Print Python info
echo "Python: $(which python3)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check dependencies
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --user --quiet
python3 -c "import einops; print('einops OK')" || pip install einops --user --quiet
python3 -c "from PIL import Image; print('PIL OK')"

# =============================================================================
# Training Configuration
# =============================================================================
# Model: Restormer-base
# Image size: 128x128 (to match MambaDiffusion)
# Epochs: 200
# Batch size: 8 (can use larger batch at smaller resolution)
# =============================================================================

python3 src/training/train_restormer.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_restormer_128 \
    --image_size 128 \
    --model_size base \
    --batch_size 8 \
    --num_epochs 200 \
    --lr 2e-4 \
    --num_workers 8

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "End time: $(date)"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

echo "=========================================="
echo "Restormer-Base (128x128) Training Complete"
echo "End time: $(date)"
echo "=========================================="
