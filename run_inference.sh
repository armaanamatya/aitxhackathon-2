#!/bin/bash
#SBATCH --job-name=darkir_inf
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=inference_%j.out

echo "Running DarkIR inference on test set..."
echo "Date: $(date)"

/cm/local/apps/python39/bin/python3 run_darkir_test_inference.py \
    --checkpoint outputs_darkir_512_cv/fold_1/checkpoint_best.pt \
    --test_jsonl data_splits/test.jsonl \
    --output_dir test/darkir_baseline \
    --resolution 512

echo ""
echo "Inference complete!"
echo "Date: $(date)"
