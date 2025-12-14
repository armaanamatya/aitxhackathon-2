#!/bin/bash
# Quick training monitoring script

echo "=========================================="
echo "TRAINING MONITOR"
echo "=========================================="

echo -e "\n📊 RUNNING JOBS:"
squeue -u sww35

echo -e "\n📈 PREPROCESSING PROGRESS:"
if [ -f preproc_experiments_609674.out ]; then
    tail -5 preproc_experiments_609674.out | grep "Epoch"
else
    echo "No output yet"
fi

echo -e "\n🎨 CONTROLNET PROGRESS:"
LATEST_CN=$(ls -t controlnet_1024_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_CN" ]; then
    tail -10 "$LATEST_CN" | grep -E "Epoch|loss|COMPLETE"
else
    echo "No ControlNet job yet"
fi

echo -e "\n📂 OUTPUT DIRECTORIES:"
for dir in outputs_full_baseline outputs_cleaned_* outputs_controlnet_*; do
    if [ -d "$dir" ]; then
        if [ -f "$dir/history.json" ]; then
            echo "  ✅ $dir (has checkpoint)"
        else
            echo "  🔄 $dir (training...)"
        fi
    fi
done

echo -e "\n=========================================="
echo "To watch live: tail -f <output_file>.out"
echo "=========================================="
