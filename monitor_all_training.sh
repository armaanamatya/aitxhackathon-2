#!/bin/bash
# Monitor all training jobs

echo "========================================================================"
echo "TRAINING JOB MONITOR"
echo "========================================================================"
echo "Time: $(date)"
echo ""

echo "📊 Active SLURM Jobs:"
squeue -u $USER --format="%.10i %.20j %.10T %.12M %.6D %R" | head -20
echo ""

echo "========================================================================"
echo "🚀 SIMPLIFIED RESTORMER (FAST - NO VGG)"
echo "========================================================================"
if [ -f "outputs_restormer_simple_fast/training.log" ]; then
    echo "Last 15 lines:"
    tail -15 outputs_restormer_simple_fast/training.log
    echo ""
    if grep -q "Epoch.*Val=" outputs_restormer_simple_fast/training.log 2>/dev/null; then
        echo "Progress summary:"
        grep "Epoch.*Val=" outputs_restormer_simple_fast/training.log | tail -5
    fi
else
    echo "⏳ Not started yet"
fi
echo ""

echo "========================================================================"
echo "🎨 SIMPLIFIED RESTORMER (WITH VGG PERCEPTUAL)"
echo "========================================================================"
if [ -f "outputs_restormer_simple_perceptual/training.log" ]; then
    echo "Last 15 lines:"
    tail -15 outputs_restormer_simple_perceptual/training.log
    echo ""
    if grep -q "Epoch.*Val=" outputs_restormer_simple_perceptual/training.log 2>/dev/null; then
        echo "Progress summary:"
        grep "Epoch.*Val=" outputs_restormer_simple_perceptual/training.log | tail -5
    fi
else
    echo "⏳ Not started yet"
fi
echo ""

echo "========================================================================"
echo "🔧 PREVIOUS COMPLEX LOSS (FOR COMPARISON)"
echo "========================================================================"
if [ -f "outputs_restormer_optimal_v2/training.log" ]; then
    echo "Last issue (validation stuck):"
    grep "Epoch.*Val=" outputs_restormer_optimal_v2/training.log | tail -5
else
    echo "❌ No log file"
fi
echo ""

echo "========================================================================"
echo "💡 QUICK COMMANDS"
echo "========================================================================"
echo "Watch fast training:     tail -f outputs_restormer_simple_fast/training.log"
echo "Watch perceptual:        tail -f outputs_restormer_simple_perceptual/training.log"
echo "Launch perceptual:       sbatch train_restormer_simple_perceptual.sh"
echo "Check jobs:              squeue -u \$USER"
echo "Cancel job:              scancel JOB_ID"
echo "========================================================================"
