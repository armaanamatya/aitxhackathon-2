#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o train_sd_lora_%j.out
#SBATCH -e train_sd_lora_%j.err
#SBATCH -J autohdr_sd_lora

# Stable Diffusion + LoRA: Leverage SD priors for high-quality editing
# Very efficient training with LoRA adapters

echo "=========================================="
echo "Stable Diffusion LoRA Training"
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

# Install SD dependencies (not in base venv)
pip install diffusers transformers peft accelerate --quiet

# Run SD LoRA training
python3 src/training/train_sd_lora.py \
    --data_root . \
    --jsonl_path train.jsonl \
    --output_dir outputs_sd_lora \
    --model_id runwayml/stable-diffusion-v1-5 \
    --image_size 512 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --save_interval 10 \
    --sample_interval 5 \
    --num_workers 4

echo "=========================================="
echo "SD LoRA Training Complete"
echo "End time: $(date)"
echo "=========================================="
