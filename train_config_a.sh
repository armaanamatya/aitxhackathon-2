#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_config_a_%j.out
#SBATCH -e train_config_a_%j.err
#SBATCH -J autohdr_config_a

# Config A: Enhanced Pix2Pix at 768px - Full setup with all losses
# Expected: Best quality, slower training

echo "=========================================="
echo "Config A: Full Enhanced Pix2Pix @ 768px"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load modules and activate venv
module load python39
module load cuda11.8/toolkit/11.8.0
source venv_gpu/bin/activate

# Print Python and CUDA info
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify lpips is installed
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --quiet

# Run training
python3 src/training/train.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_config_a \
    --batch_size 10 \
    --num_epochs 150 \
    --image_size 768 \
    --lr_g 2e-4 \
    --lr_d 2e-4 \
    --lambda_l1 100 \
    --lambda_perceptual 10 \
    --lambda_lpips 5 \
    --lambda_lab 10 \
    --lambda_hist 1 \
    --lambda_adv 1 \
    --num_disc_scales 2 \
    --save_interval 10 \
    --sample_interval 5 \
    --num_workers 8

echo "=========================================="
echo "Config A Complete"
echo "End time: $(date)"
echo "=========================================="
