#!/bin/bash
#SBATCH --job-name=controlnet_restormer_fixed
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=training_fixed_%j.out
#SBATCH --error=training_fixed_%j.err

set -e

################################################################################
# CONTROLNET-RESTORMER WITH FIXED PRETRAINED LOADING
# Production-Ready Configuration
################################################################################

echo "================================================================================"
echo "CONTROLNET-RESTORMER WITH FIXED PRETRAINED WEIGHT LOADING"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate conda environment
source /mmfs1/home/sww35/miniforge3/etc/profile.d/conda.sh
conda activate controlnet_a100

echo "📋 Environment:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.version.cuda}')"
echo ""

# Check GPU
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv
echo ""

# Check pretrained weights
PRETRAINED_PATH="pretrained/restormer_denoising.pth"
if [ -f "$PRETRAINED_PATH" ]; then
    echo "✅ Pretrained weights found: $PRETRAINED_PATH"
    echo "   Size: $(du -h $PRETRAINED_PATH | cut -f1) (expected: ~100MB)"
else
    echo "❌ ERROR: Pretrained weights not found: $PRETRAINED_PATH"
    echo "   Run: bash download_pretrained_restormer.sh"
    exit 1
fi
echo ""

################################################################################
# HYPERPARAMETERS - OPTIMIZED WITH PRETRAINED WEIGHTS
################################################################################

echo "================================================================================"
echo "HYPERPARAMETER CONFIGURATION (WITH PRETRAINED WEIGHTS + FIXED LOADING)"
echo "================================================================================"
echo ""

# Model Architecture - MUST MATCH PRETRAINED CHECKPOINT
DIM=96  # Pretrained checkpoint uses dim=96 (Restormer-Large)
NUM_BLOCKS="4 6 6 8"
NUM_REFINEMENT_BLOCKS=4
HEADS="1 2 4 8"
FFN_EXPANSION=2.66

echo "📐 Model Architecture (matching pretrained checkpoint):"
echo "  Base dimension: $DIM (Restormer-Large)"
echo "  Blocks per stage: $NUM_BLOCKS"
echo "  Attention heads: $HEADS"
echo "  Total params: ~200M (100M base frozen + 100M trainable)"
echo "  Trainable ratio: 50% (base frozen, adaptation trainable)"
echo ""

# Training Configuration
RESOLUTION=512
BATCH_SIZE=6  # Further reduced due to larger model (dim=96)
EPOCHS=100
LR=1e-4  # Lower LR for finetuning pretrained weights
WARMUP_EPOCHS=10
EARLY_STOP_PATIENCE=15

echo "🎯 Training Configuration:"
echo "  Resolution: ${RESOLUTION}px"
echo "  Batch size: $BATCH_SIZE (reduced to avoid OOM with dual-path)"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR (standard for finetuning)"
echo "  Warmup epochs: $WARMUP_EPOCHS"
echo "  Early stopping: $EARLY_STOP_PATIENCE epochs"
echo ""

# Loss Weights - Perceptual Quality Focus
LAMBDA_L1=1.0
LAMBDA_VGG=0.2
LAMBDA_SSIM=0.1

echo "🎨 Loss Weights (Perceptual Quality):"
echo "  L1 loss: $LAMBDA_L1 (pixel accuracy)"
echo "  VGG loss: $LAMBDA_VGG (perceptual quality - HIGH)"
echo "  SSIM loss: $LAMBDA_SSIM (structural similarity)"
echo ""

# Cross-Validation
N_FOLDS=3
DATA_SPLITS_DIR="data_splits"

echo "📊 Cross-Validation:"
echo "  Folds: $N_FOLDS"
echo "  Split per fold: 511 train / 56 val (90:10)"
echo "  Test set: 10 samples (completely held out)"
echo ""

# Performance Optimizations
NUM_WORKERS=32
MIXED_PRECISION="--mixed_precision"
FREEZE_BASE="--freeze_base"  # Freeze pretrained base, train adaptation layer

echo "⚡ Performance Optimizations:"
echo "  Mixed precision (FP16): true"
echo "  Data workers: $NUM_WORKERS"
echo "  Freeze base model: true (ControlNet mode)"
echo "  Gradient checkpointing: false"
echo "  Persistent workers: true"
echo "  Pin memory: true"
echo ""

echo "💾 Memory Estimate @ ${RESOLUTION}px, batch=$BATCH_SIZE:"
echo "  Base model (frozen): ~200 MB"
echo "  Trainable model: ~200 MB"
echo "  Activations (FP16, batch=10): ~8-12 GB"
echo "  Optimizer (trainable only): ~800 MB"
echo "  VGG loss model: ~550 MB"
echo "  ─────────────────────────────"
echo "  Total: ~12-16 GB / 80 GB (20% utilization)"
echo "  Headroom: ~65 GB (safe!)"
echo ""

echo "⏱️  Expected Training Time:"
echo "  Per epoch: ~1-2 min"
echo "  Per fold (with early stopping): ~1-2 hours"
echo "  Total (3 folds): 3-6 hours"
echo ""

echo "🎯 Expected Results (WITH PRETRAINED WEIGHTS):"
echo "  Validation PSNR: 30-32 dB per fold"
echo "  Test PSNR (single fold): 30-31 dB"
echo "  Test PSNR (ensemble): 31-32 dB"
echo "  Gain over from-scratch: +3-5 dB"
echo ""

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

echo "================================================================================"
echo ""
echo "🔍 Pre-flight checks..."
echo ""

# Check data splits exist
if [ ! -d "$DATA_SPLITS_DIR" ]; then
    echo "❌ ERROR: Data splits directory not found: $DATA_SPLITS_DIR"
    exit 1
fi

echo "✅ Data splits verified:"
if [ -f "$DATA_SPLITS_DIR/test.jsonl" ]; then
    TEST_COUNT=$(python3 -c "import json; print(sum(1 for _ in open('$DATA_SPLITS_DIR/test.jsonl')))")
    echo "   Test: $TEST_COUNT samples (held out)"
fi
if [ -f "$DATA_SPLITS_DIR/fold_1/train.jsonl" ]; then
    TRAIN_COUNT=$(python3 -c "import json; print(sum(1 for _ in open('$DATA_SPLITS_DIR/fold_1/train.jsonl')))")
    VAL_COUNT=$(python3 -c "import json; print(sum(1 for _ in open('$DATA_SPLITS_DIR/fold_1/val.jsonl')))")
    echo "   Fold 1: $TRAIN_COUNT train / $VAL_COUNT val"
fi

# Check CUDA
python3 -c "import torch; assert torch.cuda.is_available()" || {
    echo "❌ ERROR: CUDA not available"
    exit 1
}
echo "✅ CUDA available"

# Check GPU compute capability
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "✅ GPU compute capability: $COMPUTE_CAP (A100 = 8.0)"

# Check dependencies
echo "✅ Checking dependencies..."
python3 -c "import torch, torchvision, cv2, tqdm, numpy, PIL" || {
    echo "❌ ERROR: Missing dependencies"
    exit 1
}
echo "   All dependencies installed"

echo ""
echo "================================================================================"
echo "🚀 ALL CHECKS PASSED - STARTING TRAINING"
echo "================================================================================"
echo ""

# Output directory
OUTPUT_DIR="outputs_controlnet_restormer_fixed"

# Build command
CMD="python3 train_controlnet_restormer_cv.py \
    --data_splits_dir $DATA_SPLITS_DIR \
    --resolution $RESOLUTION \
    --dim $DIM \
    --num_blocks $NUM_BLOCKS \
    --num_refinement_blocks $NUM_REFINEMENT_BLOCKS \
    --heads $HEADS \
    --ffn_expansion_factor $FFN_EXPANSION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --early_stopping_patience $EARLY_STOP_PATIENCE \
    --lambda_l1 $LAMBDA_L1 \
    --lambda_vgg $LAMBDA_VGG \
    --lambda_ssim $LAMBDA_SSIM \
    --n_folds $N_FOLDS \
    --output_dir $OUTPUT_DIR \
    --save_every 10 \
    --num_workers $NUM_WORKERS \
    --device cuda \
    --pretrained_path $PRETRAINED_PATH \
    $FREEZE_BASE \
    $MIXED_PRECISION"

echo "Command:"
echo "$CMD"
echo ""
echo "───────────────────────────────────────────────────────────────────────────────"
echo ""

# Execute training
$CMD

echo ""
echo "───────────────────────────────────────────────────────────────────────────────"
echo "Training completed: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
