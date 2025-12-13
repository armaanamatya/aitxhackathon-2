#!/bin/bash
# Real Estate HDR Enhancement - Inference Script

set -e

echo "=============================================="
echo "Real Estate HDR Enhancement - Inference"
echo "=============================================="

# Default configuration
MODEL_PATH="outputs/checkpoints/best_generator.pt"
IMAGE_SIZE=512
PRECISION="fp16"

# Parse arguments
INPUT=""
OUTPUT=""
USE_TENSORRT=false
TILED=false

print_usage() {
    echo ""
    echo "Usage: $0 --input <path> --output <path> [options]"
    echo ""
    echo "Required:"
    echo "  --input <path>     Input image or directory"
    echo "  --output <path>    Output image or directory"
    echo ""
    echo "Options:"
    echo "  --model <path>     Model path (default: outputs/checkpoints/best_generator.pt)"
    echo "  --tensorrt         Use TensorRT optimized model"
    echo "  --tiled            Use tiled processing for high-res images"
    echo "  --precision <p>    Precision: fp16 or fp32 (default: fp16)"
    echo ""
    echo "Examples:"
    echo "  # Single image"
    echo "  $0 --input images/100_src.jpg --output enhanced.jpg"
    echo ""
    echo "  # Directory"
    echo "  $0 --input test_images/ --output enhanced_images/"
    echo ""
    echo "  # With TensorRT"
    echo "  $0 --input images/100_src.jpg --output enhanced.jpg --tensorrt"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tensorrt)
            USE_TENSORRT=true
            MODEL_PATH="outputs/optimized/model_trt_fp16.ts"
            shift
            ;;
        --tiled)
            TILED=true
            shift
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Error: --input and --output are required"
    print_usage
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please train a model first or provide a valid model path."
    exit 1
fi

# Build command
CMD="python3 src/inference/infer.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --input $INPUT"
CMD="$CMD --output $OUTPUT"
CMD="$CMD --image_size $IMAGE_SIZE"
CMD="$CMD --precision $PRECISION"
CMD="$CMD --preserve_resolution"

if [ "$USE_TENSORRT" = true ]; then
    CMD="$CMD --tensorrt"
fi

if [ "$TILED" = true ]; then
    CMD="$CMD --tiled"
fi

echo ""
echo "Running: $CMD"
echo ""

# Run inference
$CMD

echo ""
echo "Done! Output saved to: $OUTPUT"
