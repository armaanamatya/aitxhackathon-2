#!/bin/bash
#SBATCH --job-name=controlnet_restormer_scratch
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=training_scratch_%j.out
#SBATCH --error=training_scratch_%j.err

set -e

################################################################################
# CONTROLNET-RESTORMER TRAINING FROM SCRATCH
# Production-Ready Configuration
################################################################################

echo "================================================================================"
echo "CONTROLNET-RESTORMER TRAINING FROM SCRATCH"
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

################################################################################
# HYPERPARAMETERS - OPTIMIZED FOR TRAINING FROM SCRATCH
################################################################################

echo "================================================================================"
echo "HYPERPARAMETER CONFIGURATION (FROM SCRATCH)"
echo "================================================================================"
echo ""

# Model Architecture
DIM=48
NUM_BLOCKS="4 6 6 8"
NUM_REFINEMENT_BLOCKS=4
HEADS="1 2 4 8"
FFN_EXPANSION=2.66

echo "📐 Model Architecture:"
echo "  Base dimension: $DIM"
echo "  Blocks per stage: $NUM_BLOCKS"
echo "  Attention heads: $HEADS"
echo "  Total params: ~26M (all trainable)"
echo "  Trainable ratio: 100% (training from scratch)"
echo ""

# Training Configuration
RESOLUTION=512
BATCH_SIZE=16
EPOCHS=150  # Increased from 100 since training from scratch
LR=2e-4     # Higher LR for from-scratch training (vs 1e-4 for finetuning)
WARMUP_EPOCHS=15  # Longer warmup for from-scratch
EARLY_STOP_PATIENCE=20  # More patience for from-scratch training

echo "🎯 Training Configuration:"
echo "  Resolution: ${RESOLUTION}px (optimal for quality vs speed)"
echo "  Batch size: $BATCH_SIZE (optimal for A100 80GB)"
echo "  Epochs: $EPOCHS (increased for from-scratch training)"
echo "  Learning rate: $LR (higher for from-scratch vs 1e-4 for finetuning)"
echo "  Warmup epochs: $WARMUP_EPOCHS (longer for stable from-scratch training)"
echo "  Early stopping: $EARLY_STOP_PATIENCE epochs (more patience for convergence)"
echo ""

# Loss Weights - Perceptual Quality Focus
LAMBDA_L1=1.0
LAMBDA_VGG=0.2  # High perceptual quality
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

echo "⚡ Performance Optimizations:"
echo "  Mixed precision (FP16): true (2-3x speedup, no accuracy loss)"
echo "  Data workers: $NUM_WORKERS (parallel loading)"
echo "  Gradient checkpointing: false (disabled - enough memory)"
echo "  Persistent workers: true (no restart between epochs)"
echo "  Pin memory: true (faster GPU transfer)"
echo ""

echo "💾 Memory Estimate @ ${RESOLUTION}px, batch=$BATCH_SIZE:"
echo "  Model weights: ~400 MB"
echo "  Activations (FP16): ~10-15 GB"
echo "  Optimizer states: ~800 MB"
echo "  VGG loss model: ~550 MB"
echo "  ─────────────────────────────"
echo "  Total: ~15-20 GB / 80 GB (25% utilization)"
echo "  Headroom: ~60 GB (very safe!)"
echo ""

echo "⏱️  Expected Training Time (FROM SCRATCH):"
echo "  Per epoch: ~1-2 min"
echo "  Per fold (with early stopping): ~2-4 hours"
echo "  Total (3 folds): 6-12 hours"
echo ""

echo "🎯 Expected Results (FROM SCRATCH):"
echo "  Validation PSNR: 28-30 dB per fold (vs 30-32 dB with pretrained)"
echo "  Test PSNR (single fold): 28-29 dB"
echo "  Test PSNR (ensemble): 29-30 dB"
echo "  Note: 2-3 dB lower than pretrained due to no transfer learning"
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
    echo "   Run: python3 create_data_splits.py --input_jsonl train.jsonl --output_dir data_splits"
    exit 1
fi

echo "✅ Data splits verified:"
if [ -f "$DATA_SPLITS_DIR/test.json" ]; then
    TEST_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_SPLITS_DIR/test.json'))))")
    echo "   Test: $TEST_COUNT samples (held out)"
fi
if [ -f "$DATA_SPLITS_DIR/fold_0.json" ]; then
    TRAIN_COUNT=$(python3 -c "import json; d=json.load(open('$DATA_SPLITS_DIR/fold_0.json')); print(len(d['train']))")
    VAL_COUNT=$(python3 -c "import json; d=json.load(open('$DATA_SPLITS_DIR/fold_0.json')); print(len(d['val']))")
    echo "   Fold 1: $TRAIN_COUNT train / $VAL_COUNT val"
fi

# Check CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
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
echo "🚀 ALL CHECKS PASSED - STARTING TRAINING FROM SCRATCH"
echo "================================================================================"
echo ""

# Output directory
OUTPUT_DIR="outputs_controlnet_restormer_scratch"

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
