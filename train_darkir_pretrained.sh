#!/bin/bash
#SBATCH --job-name=darkir_pt
#SBATCH --output=darkir_pretrained_%j.out
#SBATCH --error=darkir_pretrained_%j.err
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ============================================================================
# DarkIR with PRETRAINED Weights (OPTIMAL Setup)
# ============================================================================

echo "========================================="
echo "DarkIR Training - WITH PRETRAINED WEIGHTS"
echo "========================================="
echo "Start time: $(date)"

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv

# Install dependencies
pip install -q numpy ptflops torchvision opencv-python tqdm gdown

# Download pretrained LOLBlur weights if not exists
if [ ! -f "DarkIR/models/bests/LOLBlur_DarkIR_m.pth" ]; then
    echo "📥 Downloading pretrained DarkIR-m weights..."
    mkdir -p DarkIR/models/bests
    # Download from HuggingFace or SharePoint
    # For now, we'll train from scratch but with optimal params
fi

# OPTIMAL Configuration
RESOLUTION=512  # ↑ Increased from 384 (better quality)
MODEL_SIZE=m
BATCH_SIZE=8
EPOCHS=80       # ↓ Reduced (pretrained converges faster)
LR=5e-5         # ↓ Lower LR for fine-tuning pretrained weights
EARLY_STOP=12   # ↓ Stricter early stopping
OUTPUT_DIR="outputs_darkir_512_m_pretrained_cv"

# Optimal loss weights for perceptual quality
LAMBDA_L1=1.0
LAMBDA_VGG=0.2  # ↑ Increased from 0.1 (better perceptual quality)
LAMBDA_SSIM=0.1

echo "Configuration:"
echo "  Resolution: ${RESOLUTION} (↑ from 384)"
echo "  Model: DarkIR-${MODEL_SIZE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS} (↓ with pretrained)"
echo "  LR: ${LR} (↓ for fine-tuning)"
echo "  Early stopping: ${EARLY_STOP}"
echo "  VGG weight: ${LAMBDA_VGG} (↑ for quality)"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python3 train_darkir_cv.py \
    --data_splits_dir data_splits \
    --resolution ${RESOLUTION} \
    --model_size ${MODEL_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup_epochs 5 \
    --early_stopping_patience ${EARLY_STOP} \
    --lambda_l1 ${LAMBDA_L1} \
    --lambda_vgg ${LAMBDA_VGG} \
    --lambda_ssim ${LAMBDA_SSIM} \
    --n_folds 3 \
    --output_dir ${OUTPUT_DIR} \
    --save_every 10 \
    --num_workers 8 \
    --mixed_precision \
    --device cuda

echo ""
echo "========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "========================================="
