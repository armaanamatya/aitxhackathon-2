#!/bin/bash
# Real Estate HDR Enhancement - Training Script
# Optimized for NVIDIA DGX Spark

set -e

echo "=============================================="
echo "Real Estate HDR Enhancement Training"
echo "=============================================="

# Configuration
DATA_ROOT="."
JSONL_PATH="train.jsonl"
OUTPUT_DIR="outputs"
IMAGE_SIZE=512
BATCH_SIZE=8
NUM_EPOCHS=200
LR_G=2e-4
LR_D=2e-4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo ""
echo "Configuration:"
echo "  Image size: ${IMAGE_SIZE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run training
RESUME_ARG=""
if [ ! -z "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

python3 src/training/train.py \
    --data_root "$DATA_ROOT" \
    --jsonl_path "$JSONL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --lr_g "$LR_G" \
    --lr_d "$LR_D" \
    --lambda_l1 100.0 \
    --lambda_perceptual 10.0 \
    --lambda_adv 1.0 \
    --num_workers 4 \
    --save_interval 10 \
    --sample_interval 5 \
    $RESUME_ARG

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo "Samples saved to: ${OUTPUT_DIR}/samples/"
