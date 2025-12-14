#!/bin/bash
# ============================================================================
# Download DarkIR Pre-trained Weights (LOLBlur - Indoor-focused)
# ============================================================================

set -e

echo "📥 Downloading DarkIR pre-trained weights..."
echo ""

# Create directory
mkdir -p DarkIR/models/bests

# DarkIR weights are hosted on HuggingFace
# https://huggingface.co/Cidaut/DarkIR

echo "🔗 Option 1: Download from HuggingFace (Recommended)"
echo ""
echo "Visit: https://huggingface.co/Cidaut/DarkIR/tree/main"
echo "Download: DarkIR_LOLBlur.pth (for DarkIR-m)"
echo "Save to: DarkIR/models/bests/"
echo ""

echo "🔗 Option 2: Use wget (if direct link available)"
echo ""
# Note: You'll need to get the actual download link from HuggingFace
# Example:
# wget -O DarkIR/models/bests/DarkIR_LOLBlur_m.pth \
#      "https://huggingface.co/Cidaut/DarkIR/resolve/main/DarkIR_LOLBlur.pth"

echo "🔗 Option 3: Use git-lfs to clone the model repo"
echo ""
echo "git lfs install"
echo "git clone https://huggingface.co/Cidaut/DarkIR DarkIR_weights"
echo "cp DarkIR_weights/*.pth DarkIR/models/bests/"
echo ""

echo "✅ After downloading, update train_darkir_manual.sh:"
echo "   Set: PRETRAINED_PATH=\"DarkIR/models/bests/DarkIR_LOLBlur.pth\""
echo ""
echo "📊 LOLBlur dataset characteristics:"
echo "   - 10,200 training pairs"
echo "   - Mostly INDOOR low-light scenes"
echo "   - Better match for real estate than FiveK"
echo ""
