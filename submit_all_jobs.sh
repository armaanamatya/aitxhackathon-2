#!/bin/bash
# Submit training configurations to run in parallel on 3 A100s
# Choose your preferred 3 configurations to run

echo "=========================================="
echo "AutoHDR Training Jobs Submission"
echo "=========================================="
echo ""
echo "Available Configurations:"
echo "  1. Config A - Enhanced Pix2Pix @ 768px (recommended baseline)"
echo "  2. Config B - Enhanced Pix2Pix @ 512px (faster, more epochs)"
echo "  3. Config C - Pix2Pix perceptual-focused @ 768px"
echo "  4. Restormer - Transformer-based (SOTA for restoration)"
echo "  5. SD-LoRA - Stable Diffusion + LoRA (creative, powerful)"
echo ""
echo "Recommended combination: Config A + Restormer + Config B"
echo ""

# Make all scripts executable
chmod +x train_config_a.sh train_config_b.sh train_config_c.sh train_restormer.sh train_sd_lora.sh

# Submit recommended 3 jobs
echo "Submitting 3 recommended configurations..."
echo ""

JOB_A=$(sbatch -p gpu1 train_config_a.sh 2>/dev/null | awk '{print $4}')
echo "Config A submitted: Job ID $JOB_A (Enhanced Pix2Pix @ 768px)"

JOB_RESTORMER=$(sbatch -p gpu1 train_restormer.sh 2>/dev/null | awk '{print $4}')
echo "Restormer submitted: Job ID $JOB_RESTORMER (Transformer @ 512px)"

JOB_B=$(sbatch -p gpu1 train_config_b.sh 2>/dev/null | awk '{print $4}')
echo "Config B submitted: Job ID $JOB_B (Pix2Pix @ 512px, 200 epochs)"

echo ""
echo "=========================================="
echo "Jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs with: tail -f train_*.out"
echo ""
echo "Expected outputs:"
echo "  - outputs_config_a/checkpoints/best_generator.pt"
echo "  - outputs_restormer/checkpoints/best_generator.pt"
echo "  - outputs_config_b/checkpoints/best_generator.pt"
echo ""
echo "Compare samples in outputs_*/samples/"
echo ""
echo "=========================================="
echo "Alternative: Submit SD-LoRA or Config C instead:"
echo "  sbatch -p gpu1 train_sd_lora.sh"
echo "  sbatch -p gpu1 train_config_c.sh"
echo "=========================================="
