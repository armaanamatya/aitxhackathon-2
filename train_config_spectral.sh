#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_config_spectral_%j.out
#SBATCH -e train_config_spectral_%j.err
#SBATCH -J autohdr_spectral

# Config Spectral: Pix2Pix with Spectral Normalization Discriminator
# Spectral Normalization constrains discriminator Lipschitz constant,
# preventing mode collapse and training instability without aggressive tuning.

echo "=========================================="
echo "Config Spectral: Pix2Pix with Spectral Norm"
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

# Run training with Spectral Normalization
# With SN, we can use more standard GAN settings:
# - Higher adversarial weight (0.5) - SN prevents D from overpowering G
# - Less aggressive D update frequency (2)
# - Standard gradient clipping
# - Less instance noise needed
python3 src/training/train.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_config_spectral \
    --batch_size 16 \
    --num_epochs 200 \
    --image_size 512 \
    --lr_g 2e-4 \
    --lr_d 1e-4 \
    --lambda_l1 100 \
    --lambda_perceptual 10 \
    --lambda_lpips 5 \
    --lambda_lab 10 \
    --lambda_hist 1 \
    --lambda_adv 0.5 \
    --num_disc_scales 2 \
    --use_spectral_norm \
    --label_smoothing 0.1 \
    --instance_noise 0.05 \
    --instance_noise_decay 0.9998 \
    --grad_clip_g 1.0 \
    --grad_clip_d 0.5 \
    --d_update_freq 2 \
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
echo "Config Spectral Complete"
echo "End time: $(date)"
echo "=========================================="
