#!/bin/bash
# FastAPI Inference Server Launcher
# Compatible with CUDA 13.0 and Python 3.10

set -e

echo "=============================================="
echo "FastAPI Inference Server"
echo "=============================================="
echo ""

# Configuration
BACKBONE_PATH="${1:-checkpoints/restormer_base.pt}"
REFINER_PATH="${2:-checkpoints/refiner.pt}"
HOST="${3:-0.0.0.0}"
PORT="${4:-8000}"

# Set environment variables
export MODEL_BACKBONE_PATH="$BACKBONE_PATH"
export MODEL_REFINER_PATH="$REFINER_PATH"
export HOST="$HOST"
export PORT="$PORT"
export MODEL_DEVICE="cuda"
export MODEL_PRECISION="fp16"
export MODEL_BATCH_SIZE=16
export LOG_LEVEL="INFO"

# Optional: Enable Triton
# export TRITON_ENABLED=true
# export TRITON_URL="localhost:8001"
# export TRITON_MODEL_NAME="restormer"

echo "Configuration:"
echo "  Backbone: $BACKBONE_PATH"
echo "  Refiner: $REFINER_PATH"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Device: $MODEL_DEVICE"
echo "  Precision: $MODEL_PRECISION"
echo ""

# Check if model exists
if [ ! -f "$BACKBONE_PATH" ]; then
    echo "Warning: Backbone model not found at $BACKBONE_PATH"
    echo "Server will start but model won't be loaded"
fi

# Run server
echo "Starting server..."
echo ""

python3 -m api_server.main

