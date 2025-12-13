#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o evaluate_%j.out
#SBATCH -J autohdr_eval

# Evaluate all trained models and select the best one

echo "=========================================="
echo "AutoHDR Model Evaluation"
echo "=========================================="

cd /mmfs1/home/sww35/autohdr-real-estate-577
source venv/bin/activate

# Function to evaluate a model
evaluate_model() {
    CONFIG=$1
    CHECKPOINT="outputs_$CONFIG/checkpoints/best_generator.pt"

    if [ -f "$CHECKPOINT" ]; then
        echo ""
        echo "Evaluating Config $CONFIG..."
        python3 src/inference/metrics.py \
            --model_path "$CHECKPOINT" \
            --data_root . \
            --jsonl_path train.jsonl \
            --image_size 512 \
            --num_samples 58 | tee "eval_$CONFIG.txt"
        echo "Config $CONFIG evaluation complete"
    else
        echo "Warning: $CHECKPOINT not found, skipping Config $CONFIG"
    fi
}

# Evaluate all configs
evaluate_model "config_a"
evaluate_model "config_b"
evaluate_model "config_c"

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo ""
echo "Compare PSNR/SSIM scores in eval_config_*/metrics.txt"
echo "Visually compare samples in outputs_config_*/samples/"
echo ""
echo "To run TensorRT optimization on best model:"
echo "  python3 src/optimization/tensorrt_optimize.py --model_path <best_checkpoint>"
