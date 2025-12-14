#!/bin/bash
#SBATCH --job-name=controlnet_1024
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=controlnet_1024_%j.out

echo "=========================================="
echo "CONTROLNET 1024x1024 TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi
echo "=========================================="

cd /mmfs1/home/sww35/autohdr-real-estate-577

# Remove ALL problematic packages and reinstall latest compatible versions
echo "Cleaning environment..."
pip uninstall -y torch_xla torch-xla 2>/dev/null || true

echo "Installing latest compatible versions..."
pip install --quiet --upgrade --force-reinstall \
    diffusers \
    transformers \
    accelerate \
    peft \
    huggingface-hub

# Verify
echo "Verifying installation..."
python3 -c "from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel, DDPMScheduler; print('✅ All imports OK')" || {
    echo "❌ Import failed, trying alternate method..."
    exit 1
}

echo ""
echo "Starting training..."
echo ""

# Run training
/cm/local/apps/python39/bin/python3 train_controlnet_enhancement.py \
    --train_jsonl train.jsonl \
    --data_dir . \
    --resolution 1024 \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --epochs 100 \
    --lr 1e-5 \
    --lr_scheduler cosine \
    --warmup_steps 500 \
    --num_workers 12 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --enable_xformers \
    --save_every 10 \
    --output_dir outputs_controlnet_1024

echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
