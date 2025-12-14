#!/bin/bash
# Quick experiment: Compare FP32 vs FP16 accuracy
# Trains fold 1 for 20 epochs with both precisions

echo "=== Comparing FP32 vs FP16 (Mixed Precision) ==="
echo ""

# FP32 (no mixed precision)
echo "Training with FP32 (baseline)..."
python3 train_controlnet_restormer_cv.py \
    --data_splits_dir data_splits \
    --resolution 384 \
    --batch_size 8 \
    --epochs 20 \
    --n_folds 1 \
    --output_dir outputs_fp32_test \
    --num_workers 16 \
    --device cuda \
    --pretrained_path pretrained/restormer_denoising.pth

FP32_PSNR=$(python3 -c "import json; print(json.load(open('outputs_fp32_test/fold_1/history.json'))['best_val_psnr'])")

# FP16 (mixed precision)
echo ""
echo "Training with FP16 (mixed precision)..."
python3 train_controlnet_restormer_cv.py \
    --data_splits_dir data_splits \
    --resolution 384 \
    --batch_size 8 \
    --epochs 20 \
    --n_folds 1 \
    --output_dir outputs_fp16_test \
    --num_workers 16 \
    --device cuda \
    --pretrained_path pretrained/restormer_denoising.pth \
    --mixed_precision

FP16_PSNR=$(python3 -c "import json; print(json.load(open('outputs_fp16_test/fold_1/history.json'))['best_val_psnr'])")

# Compare
echo ""
echo "=== RESULTS ==="
echo "FP32 Val PSNR: $FP32_PSNR dB"
echo "FP16 Val PSNR: $FP16_PSNR dB"
echo "Difference: $(python3 -c "print(abs($FP16_PSNR - $FP32_PSNR))")" dB"
echo ""
echo "Typical difference: <0.1 dB (negligible)"
