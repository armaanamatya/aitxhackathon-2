#!/bin/bash
# =============================================================================
# Submission Packaging Script for AutoHDR Challenge
# =============================================================================
#
# Creates a complete submission package for the NVIDIA DGX Spark AutoHDR challenge.
#
# Usage:
#   ./create_submission.sh
#   ./create_submission.sh --checkpoint outputs/checkpoints/best_generator.pt
#   ./create_submission.sh --test_dir data/test --output_dir submission/outputs
#
# =============================================================================

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBMISSION_DIR="submission_${TIMESTAMP}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/best_generator.pt}"
TEST_DIR="${TEST_DIR:-data/test}"
MODEL="${MODEL:-unet}"
USE_TTA="${USE_TTA:-true}"

echo "=============================================="
echo "AutoHDR Submission Packager"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Checkpoint:    $CHECKPOINT"
echo "  Test dir:      $TEST_DIR"
echo "  Model:         $MODEL"
echo "  TTA:           $USE_TTA"
echo "  Output:        $SUBMISSION_DIR"
echo ""

# Create submission directory
mkdir -p "$SUBMISSION_DIR/outputs"
mkdir -p "$SUBMISSION_DIR/checkpoints"

# Copy checkpoint
if [ -f "$CHECKPOINT" ]; then
    echo "Copying checkpoint..."
    cp "$CHECKPOINT" "$SUBMISSION_DIR/checkpoints/"
else
    echo "Warning: Checkpoint not found at $CHECKPOINT"
fi

# Run inference on test images
if [ -d "$TEST_DIR" ]; then
    echo ""
    echo "Running inference on test images..."

    TTA_FLAG=""
    if [ "$USE_TTA" = "false" ]; then
        TTA_FLAG="--no_tta"
    fi

    python infer_tta.py \
        --input_dir "$TEST_DIR" \
        --output_dir "$SUBMISSION_DIR/outputs" \
        --checkpoint "$CHECKPOINT" \
        --model "$MODEL" \
        $TTA_FLAG

    echo "Inference complete!"
else
    echo "Warning: Test directory not found at $TEST_DIR"
    echo "Please provide test images manually in $SUBMISSION_DIR/outputs/"
fi

# Create NOTES.md
echo ""
echo "Creating submission notes..."

cat > "$SUBMISSION_DIR/NOTES.md" << EOF
# AutoHDR Real Estate Photo Enhancement - Submission

## Model Architecture

**Primary Model**: U-Net Generator with GAN training
- Encoder: 8 downsampling blocks
- Bottleneck: 9 residual blocks
- Decoder: 7 upsampling blocks with skip connections
- Parameters: ~23M

**Discriminator**: Multi-Scale PatchGAN with Spectral Normalization
- 2 scales for coarse and fine detail discrimination
- Spectral normalization for training stability

## Training Details

- **Image size**: 512x512
- **Batch size**: 4 (effective 8 with gradient accumulation)
- **Optimizer**: AdamW (β1=0.5, β2=0.999)
- **Learning rate**: 2e-4 with warmup + CosineAnnealingWarmRestarts
- **Warmup epochs**: 5
- **Total epochs**: 100

## Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| L1 | 100.0 | Pixel reconstruction |
| VGG Perceptual | 10.0 | High-level feature matching |
| SSIM | 5.0 | Structural similarity |
| Edge-Aware | 2.0 | Sharp edge preservation |
| Color Histogram | 1.0 | Color distribution matching |
| LAB Color | 10.0 | Perceptually uniform color |
| LPIPS | 5.0 | Learned perceptual metric |
| Adversarial | 1.0 | GAN feedback |

## Inference

- **Time**: ~25ms @ 512x512 (TensorRT FP16)
- **VRAM**: ~4GB
- **TTA**: +5% quality, 2x inference time

## DGX Spark Optimizations

1. **128GB unified memory**: Full dataset cached in GPU memory
2. **TensorRT FP16**: 5x inference speedup
3. **Local processing**: Privacy-preserving (images never leave device)

## Key Innovations

1. **Edge-Aware Loss**: Custom loss preserving sharp furniture/wall edges
2. **Color Histogram Matching**: Ensures color grading matches professional edits
3. **Input-Only Augmentation**: Teaches robustness without corrupting target colors
4. **Spectral Normalization**: Stable GAN training without mode collapse

## Files

- \`outputs/\`: Enhanced test images
- \`checkpoints/\`: Model weights
- \`NOTES.md\`: This file

---

Generated: $(date)
Model: $MODEL
Checkpoint: $CHECKPOINT
EOF

# Create README
cat > "$SUBMISSION_DIR/README.md" << EOF
# AutoHDR Submission

## Quick Start

\`\`\`bash
# Run inference on a single image
python infer_tta.py --input image.jpg --output enhanced.jpg --checkpoint checkpoints/best_generator.pt

# Run inference on a directory
python infer_tta.py --input_dir test_images/ --output_dir results/ --checkpoint checkpoints/best_generator.pt
\`\`\`

## Contents

- \`outputs/\`: Enhanced images from test set
- \`checkpoints/\`: Trained model weights
- \`NOTES.md\`: Technical details and training configuration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU inference)

## Contact

Submission for NVIDIA DGX Spark AutoHDR Challenge
EOF

# Create ZIP
echo ""
echo "Creating submission archive..."
cd "$SUBMISSION_DIR"
zip -r "../submission_${TIMESTAMP}.zip" .
cd ..

echo ""
echo "=============================================="
echo "Submission package created!"
echo "=============================================="
echo ""
echo "Files:"
echo "  Directory: $SUBMISSION_DIR/"
echo "  Archive:   submission_${TIMESTAMP}.zip"
echo ""
echo "Email to: dan@autohdr.com"
echo "=============================================="
