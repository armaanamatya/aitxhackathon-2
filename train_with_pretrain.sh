#!/bin/bash
#SBATCH --job-name=pretrain_finetune
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=pretrain_finetune_%j.out

echo "========================================================================"
echo "OPTIMAL PRE-TRAINING → FINE-TUNING PIPELINE"
echo "Top 0.0001% MLE Strategy: FiveK → Real Estate"
echo "========================================================================"
date

# Paths
FIVEK_TRAIN="/mmfs1/home/sww35/autohdr-real-estate-577/fivek_processed/train.jsonl"
REAL_ESTATE_TRAIN="/mmfs1/home/sww35/autohdr-real-estate-577/train.jsonl"

# Pre-training hyperparameters (larger dataset, lower LR)
PRETRAIN_RESOLUTION=512
PRETRAIN_BATCH_SIZE=8
PRETRAIN_EPOCHS=30  # Fewer epochs on large dataset
PRETRAIN_LR=5e-5  # Lower LR for stability on diverse data
PRETRAIN_WARMUP=5
PRETRAIN_PATIENCE=10

# Fine-tuning hyperparameters (smaller dataset, careful tuning)
FINETUNE_RESOLUTION=896  # Higher resolution for final quality
FINETUNE_BATCH_SIZE=4
FINETUNE_EPOCHS=100
FINETUNE_LR=1e-5  # Much lower LR to preserve pre-trained features
FINETUNE_WARMUP=5
FINETUNE_PATIENCE=15  # More patience during fine-tuning

# ========================================================================
# PHASE 1: PRE-TRAIN ON FIVEK (5K general photo enhancement pairs)
# ========================================================================

echo ""
echo "========================================================================"
echo "PHASE 1: PRE-TRAINING ON FIVEK DATASET"
echo "========================================================================"
echo "Strategy: Learn general photo enhancement patterns"
echo "Dataset: ~5000 professionally retouched photos (Expert C)"
echo "Goal: Initialize weights with photo enhancement knowledge"
echo ""

if [ ! -f "$FIVEK_TRAIN" ]; then
    echo "❌ ERROR: FiveK dataset not found at $FIVEK_TRAIN"
    echo "Please run: python3 prepare_fivek_robust.py"
    exit 1
fi

/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl "$FIVEK_TRAIN" \
    --resolution $PRETRAIN_RESOLUTION \
    --batch_size $PRETRAIN_BATCH_SIZE \
    --epochs $PRETRAIN_EPOCHS \
    --lr $PRETRAIN_LR \
    --warmup_epochs $PRETRAIN_WARMUP \
    --early_stopping_patience $PRETRAIN_PATIENCE \
    --preprocess light_aug \
    --output_dir outputs_fivek_pretrained \
    --mixed_precision \
    --save_every 5

PRETRAIN_CHECKPOINT="outputs_fivek_pretrained/checkpoint_best.pt"

if [ ! -f "$PRETRAIN_CHECKPOINT" ]; then
    echo "❌ ERROR: Pre-training failed, checkpoint not found"
    exit 1
fi

echo ""
echo "✅ Phase 1 Complete: Pre-trained model saved"
echo "   Location: $PRETRAIN_CHECKPOINT"
echo ""

# ========================================================================
# PHASE 2: FINE-TUNE ON REAL ESTATE DATA
# ========================================================================

echo ""
echo "========================================================================"
echo "PHASE 2: FINE-TUNING ON REAL ESTATE DATASET"
echo "========================================================================"
echo "Strategy: Adapt general photo enhancement to real estate domain"
echo "Dataset: 577 real estate image pairs (training data)"
echo "Goal: Specialize for real estate transformations"
echo ""
echo "Key differences from scratch training:"
echo "  - Lower learning rate (1e-5 vs 1e-4)"
echo "  - Higher resolution (896 vs 512)"
echo "  - More patience (15 vs 10 epochs)"
echo "  - Careful to preserve pre-trained features"
echo ""

/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl "$REAL_ESTATE_TRAIN" \
    --resolution $FINETUNE_RESOLUTION \
    --batch_size $FINETUNE_BATCH_SIZE \
    --epochs $FINETUNE_EPOCHS \
    --lr $FINETUNE_LR \
    --warmup_epochs $FINETUNE_WARMUP \
    --early_stopping_patience $FINETUNE_PATIENCE \
    --preprocess light_aug \
    --resume "$PRETRAIN_CHECKPOINT" \
    --output_dir outputs_finetuned_from_fivek \
    --mixed_precision \
    --save_every 10

echo ""
echo "✅ Phase 2 Complete: Fine-tuned model saved"
echo "   Location: outputs_finetuned_from_fivek/checkpoint_best.pt"
echo ""

# ========================================================================
# PHASE 3: EVALUATION ON TEST SET
# ========================================================================

echo ""
echo "========================================================================"
echo "PHASE 3: EVALUATION"
echo "========================================================================"
echo "Comparing: Scratch vs Pre-trained+Fine-tuned"
echo ""

# Run test inference on both models
echo "Running test inference on fine-tuned model..."
/cm/local/apps/python39/bin/python3 run_test_inference.py \
    --model outputs_finetuned_from_fivek/checkpoint_best.pt \
    --resolution $FINETUNE_RESOLUTION \
    --output_dir test/finetuned_from_fivek

# Compare with baseline (if exists)
if [ -f "test/restormer_896/results.json" ]; then
    echo ""
    echo "📊 RESULTS COMPARISON:"
    echo "-----------------------------------"
    echo "Baseline (trained from scratch):"
    python3 -c "import json; r=json.load(open('test/restormer_896/results.json')); print(f'  L1 Loss: {r[\"avg_l1_loss\"]:.4f}')"

    echo ""
    echo "Pre-trained + Fine-tuned:"
    python3 -c "import json; r=json.load(open('test/finetuned_from_fivek/results.json')); print(f'  L1 Loss: {r[\"avg_l1_loss\"]:.4f}')"

    echo ""
    echo "Expected improvement: +10-20% (if FiveK pre-training helps)"
fi

echo ""
echo "========================================================================"
echo "✅ COMPLETE PRE-TRAINING PIPELINE FINISHED"
echo "========================================================================"
date
echo ""
echo "Final model: outputs_finetuned_from_fivek/checkpoint_best.pt"
echo "Test results: test/finetuned_from_fivek/results.json"
echo "========================================================================"
