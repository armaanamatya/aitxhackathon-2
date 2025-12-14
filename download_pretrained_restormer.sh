#!/bin/bash
# ============================================================================
# Download Pretrained Restormer Weights for ControlNet-Restormer
# ============================================================================

set -e

echo "========================================"
echo "Downloading Pretrained Restormer Weights"
echo "========================================"
echo ""

# Create directory
mkdir -p pretrained

# Download SIDD denoising weights (best for indoor real estate)
echo "📥 Downloading SIDD denoising weights..."
echo "   Source: https://github.com/swz30/Restormer"
echo "   Dataset: SIDD (indoor low-light denoising)"
echo "   Performance: 40.02 dB PSNR"
echo ""

# Try gdown (Google Drive downloader)
if ! command -v gdown &> /dev/null; then
    echo "📦 Installing gdown (Google Drive downloader)..."
    pip install -q gdown
fi

# Download from Google Drive
# File ID from: https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view
gdown --id 1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH \
    -O pretrained/restormer_denoising.pth

# Verify download
if [ -f "pretrained/restormer_denoising.pth" ]; then
    SIZE=$(ls -lh pretrained/restormer_denoising.pth | awk '{print $5}')
    echo ""
    echo "✅ Download complete!"
    echo "   File: pretrained/restormer_denoising.pth"
    echo "   Size: ${SIZE}"
    echo ""

    # Auto-configure training script
    echo "🔧 Auto-configuring training script..."

    if [ -f "train_controlnet_restormer_512_a100.sh" ]; then
        # Update the PRETRAINED_PATH line
        sed -i 's|PRETRAINED_PATH=""|PRETRAINED_PATH="pretrained/restormer_denoising.pth"|g' train_controlnet_restormer_512_a100.sh
        echo "   ✅ Updated train_controlnet_restormer_512_a100.sh"
    fi

    if [ -f "train_controlnet_restormer.sh" ]; then
        sed -i 's|PRETRAINED_PATH=""|PRETRAINED_PATH="pretrained/restormer_denoising.pth"|g' train_controlnet_restormer.sh
        echo "   ✅ Updated train_controlnet_restormer.sh"
    fi

    echo ""
    echo "========================================"
    echo "✅ Setup Complete!"
    echo "========================================"
    echo ""
    echo "🚀 Ready to train! Run:"
    echo "   sbatch train_controlnet_restormer_512_a100.sh"
    echo ""
    echo "Expected improvement with pretrained weights:"
    echo "   Without: ~26-28 dB PSNR (training from scratch)"
    echo "   With:    ~30-32 dB PSNR (pretrained + adaptation)"
    echo "   Gain:    +3-5 dB 🎯"
    echo ""
else
    echo "❌ Download failed!"
    echo ""
    echo "Manual download:"
    echo "   Visit: https://github.com/swz30/Restormer/releases/tag/v1.0"
    echo "   Download: denoising_sidd.pth"
    echo "   Save to: pretrained/restormer_denoising.pth"
    exit 1
fi
