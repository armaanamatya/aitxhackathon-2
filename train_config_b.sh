#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_config_b_%j.out
#SBATCH -e train_config_b_%j.err
#SBATCH -J autohdr_config_b

# Config B: Enhanced Pix2Pix at 512px - Faster training, more epochs
# Expected: Good baseline, faster convergence

echo "=========================================="
echo "Config B: Enhanced Pix2Pix @ 512px"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Load modules (packages are in user's .local)
module load python39
module load cuda11.8/toolkit/11.8.0

# Print Python and CUDA info
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify lpips is installed
python3 -c "import lpips; print('LPIPS OK')" || pip install lpips --quiet

# Run training with AGGRESSIVE GAN stabilization
# Previous runs crashed at epochs 16, 50, 67 due to D overpowering G
# Key fixes (more aggressive):
# - Much lower D learning rate (5:1 ratio)
# - Lower adversarial loss weight (0.3 instead of 1.0)
# - Train D every 4 steps (less frequent)
# - Very aggressive D gradient clipping (0.1)
# - Higher instance noise with slower decay
python3 src/training/train.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_config_b \
    --batch_size 16 \
    --num_epochs 200 \
    --image_size 512 \
    --lr_g 2e-4 \
    --lr_d 4e-5 \
    --lambda_l1 100 \
    --lambda_perceptual 10 \
    --lambda_lpips 5 \
    --lambda_lab 10 \
    --lambda_hist 1 \
    --lambda_adv 0.3 \
    --num_disc_scales 2 \
    --label_smoothing 0.15 \
    --instance_noise 0.15 \
    --instance_noise_decay 0.9998 \
    --grad_clip_g 1.0 \
    --grad_clip_d 0.1 \
    --d_update_freq 4 \
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
echo "Config B Complete"
echo "End time: $(date)"
echo "=========================================="
