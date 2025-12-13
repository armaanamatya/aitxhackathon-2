#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_restormer_%j.out
#SBATCH -e train_restormer_%j.err
#SBATCH -J autohdr_restormer

# Restormer: State-of-the-art transformer for image restoration
# No adversarial training - more stable, often better quality

echo "=========================================="
echo "Restormer Training"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load modules and activate venv
module load python39
module load cuda11.8/toolkit/11.8.0
source venv_gpu/bin/activate

# Print info
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify lpips is installed
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --quiet

# Run Restormer training
# Using "base" size which balances quality and speed
python3 src/training/train_restormer.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_restormer \
    --model_size base \
    --batch_size 2 \
    --num_epochs 150 \
    --image_size 512 \
    --lr 3e-4 \
    --lambda_l1 1.0 \
    --lambda_perceptual 0.1 \
    --lambda_lpips 0.1 \
    --lambda_lab 0.1 \
    --save_interval 10 \
    --sample_interval 5 \
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
echo "Restormer Training Complete"
echo "End time: $(date)"
echo "=========================================="
