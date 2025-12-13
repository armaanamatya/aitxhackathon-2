#!/bin/bash
# Real Estate HDR Enhancement - TensorRT Optimization Script
# Key NVIDIA integration for hackathon

set -e

echo "=============================================="
echo "TensorRT Model Optimization"
echo "=============================================="
echo ""
echo "This script optimizes the trained model using NVIDIA TensorRT"
echo "for faster inference on DGX Spark."
echo ""

# Configuration
MODEL_PATH="${1:-outputs/checkpoints/best_generator.pt}"
OUTPUT_DIR="outputs/optimized"
IMAGE_SIZE=512
PRECISION="fp16"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please train a model first using: ./run_training.sh"
    exit 1
fi

echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Precision: $PRECISION"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run optimization
python3 src/optimization/tensorrt_optimize.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --image_size "$IMAGE_SIZE" \
    --precision "$PRECISION" \
    --method all

echo ""
echo "=============================================="
echo "Optimization Complete!"
echo "=============================================="
echo ""
echo "Optimized models saved to: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -la "$OUTPUT_DIR"
echo ""
echo "To run inference with TensorRT model:"
echo "  python3 src/inference/infer.py --model_path ${OUTPUT_DIR}/model_trt_fp16.ts --tensorrt --input <image> --output <output>"
