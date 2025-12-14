#!/bin/bash
# Optimized Inference Runner for DGX Spark
# Maximum speed with full metrics preservation

set -e

echo "=============================================="
echo "DGX Spark Optimized Inference"
echo "=============================================="
echo ""

# Configuration - Optimized for DGX Spark
BACKBONE_PATH="${1:-checkpoints/restormer_base.pt}"
REFINER_PATH="${2:-checkpoints/refiner.pt}"
INPUT_DIR="${3:-images/test}"
OUTPUT_DIR="${4:-outputs/inference_dgx_spark}"
TARGETS_DIR="${5:-images/targets}"  # Optional: for metrics computation
JSONL_PATH="${6:-data_splits/test.jsonl}"  # Optional: for metrics computation

# DGX Spark Optimized Settings
TILE_SIZE=768
OVERLAP=96
BATCH_SIZE=16  # Increased from 4 - DGX Spark has 128GB unified memory
FP16=true
COMPILE=true
TENSORRT=false  # Set to true if TensorRT model is available
NUM_WORKERS=8  # For async I/O

echo "Configuration:"
echo "  Backbone: $BACKBONE_PATH"
echo "  Refiner: $REFINER_PATH"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Tile size: $TILE_SIZE"
echo "  Overlap: $OVERLAP"
echo "  Batch size: $BATCH_SIZE (optimized for DGX Spark)"
echo "  FP16: $FP16"
echo "  Compiled: $COMPILE"
echo "  TensorRT: $TENSORRT"
echo ""

# Check if model exists
if [ ! -f "$BACKBONE_PATH" ]; then
    echo "Error: Backbone model not found at $BACKBONE_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python3 inference_dgx_spark_optimized.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR \
    --backbone $BACKBONE_PATH \
    --tile_size $TILE_SIZE \
    --overlap $OVERLAP \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS"

# Add optional refiner
if [ -f "$REFINER_PATH" ]; then
    CMD="$CMD --refiner $REFINER_PATH"
    echo "  Using refiner: $REFINER_PATH"
fi

# Add FP16 flag
if [ "$FP16" = true ]; then
    CMD="$CMD --fp16"
fi

# Add compile flag
if [ "$COMPILE" = true ]; then
    CMD="$CMD --compile"
fi

# Add TensorRT flag
if [ "$TENSORRT" = true ]; then
    CMD="$CMD --tensorrt"
fi

# Add metrics computation if targets are provided
if [ -d "$TARGETS_DIR" ] || [ -f "$JSONL_PATH" ]; then
    if [ -f "$JSONL_PATH" ]; then
        CMD="$CMD --jsonl $JSONL_PATH"
        if [ -d "$TARGETS_DIR" ]; then
            CMD="$CMD --targets $TARGETS_DIR"
        fi
        echo "  Metrics computation enabled via JSONL"
    elif [ -d "$TARGETS_DIR" ]; then
        CMD="$CMD --targets $TARGETS_DIR"
        echo "  Metrics computation enabled via targets directory"
    fi
fi

echo ""
echo "Running inference..."
echo ""

# Run inference
eval $CMD

echo ""
echo "=============================================="
echo "Inference Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Metrics summary: $OUTPUT_DIR/inference_results.json"
echo ""

