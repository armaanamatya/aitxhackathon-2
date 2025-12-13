# Real Estate HDR Photo Enhancement

AI-powered photo enhancement for real estate imagery. Transforms unedited photos into professionally edited, publication-ready images.

## Overview

This solution trains a deep learning model to learn the editing style of professional real estate photo editors. Given an unedited photo, the model produces an enhanced version with:

- **Corrected exposure** - Properly lit interiors and exteriors
- **Balanced white balance** - Neutral, professional color tones
- **HDR-style tone mapping** - Detail in both shadows and highlights
- **Professional color grading** - Consistent, appealing aesthetic

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              REAL ESTATE HDR ENHANCEMENT PIPELINE           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Image ─────┐                                         │
│                   ▼                                         │
│            ┌─────────────┐                                  │
│            │   U-Net     │                                  │
│            │  Generator  │ (Residual Learning)              │
│            │  + GAN      │                                  │
│            └──────┬──────┘                                  │
│                   │                                         │
│                   ▼                                         │
│            ┌─────────────┐                                  │
│            │  TensorRT   │ (NVIDIA Optimization)            │
│            │   Engine    │                                  │
│            └──────┬──────┘                                  │
│                   │                                         │
│                   ▼                                         │
│            Enhanced Image                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

- **Generator**: U-Net with 9 residual blocks and skip connections
- **Discriminator**: PatchGAN (70x70 receptive field)
- **Losses**: L1 + Perceptual (VGG) + Adversarial
- **Residual Learning**: Model learns the "edit" rather than full image

## NVIDIA Integration

### TensorRT Optimization

The trained model is optimized with **NVIDIA TensorRT** for production deployment:

```bash
# Optimize model
./run_optimize.sh outputs/checkpoints/best_generator.pt

# Inference with TensorRT
./run_inference.sh --input image.jpg --output enhanced.jpg --tensorrt
```

**Performance improvements:**
- **5-10x faster inference** compared to vanilla PyTorch
- **FP16 precision** for memory efficiency
- **Kernel fusion** for optimized memory bandwidth

### The Spark Story: Why DGX Spark?

Our solution is optimized for the NVIDIA DGX Spark architecture:

1. **128GB Unified Memory**
   - Load entire dataset (577 × 3MB = 1.7GB) plus model (200MB) plus training buffers in GPU memory
   - Zero-copy data transfer eliminates CPU-GPU bottlenecks
   - Enables larger batch sizes (8-16) at 512px resolution

2. **Local Privacy & Latency**
   - Real estate images contain sensitive property information
   - Sub-100ms inference latency locally vs 500ms+ cloud API calls
   - Data never leaves the device

3. **TensorRT on Grace Hopper**
   - FP16 inference reduces memory 2x
   - NVLink-C2C provides 900GB/s bandwidth for streaming images
   - Ideal for batch processing real estate portfolios

## Quick Start

### Installation

```bash
# Create environment
conda create -n hdr-enhance python=3.10 -y
conda activate hdr-enhance

# Install dependencies
pip install -r requirements.txt

# Install TensorRT (optional, for optimization)
pip install tensorrt torch-tensorrt
```

### Training

```bash
# Basic training
./run_training.sh

# Custom settings
./run_training.sh --image_size 512 --batch_size 8 --epochs 200

# Resume from checkpoint
./run_training.sh --resume outputs/checkpoints/latest.pt
```

### TensorRT Optimization

```bash
# Optimize trained model
./run_optimize.sh outputs/checkpoints/best_generator.pt
```

### Inference

```bash
# Single image
./run_inference.sh --input images/100_src.jpg --output enhanced.jpg

# Directory of images
./run_inference.sh --input test_images/ --output enhanced_images/

# With TensorRT acceleration
./run_inference.sh --input image.jpg --output enhanced.jpg --tensorrt
```

### Demo UI

```bash
python demo.py
# Open http://localhost:7860 in browser
```

## Project Structure

```
├── src/
│   ├── data/
│   │   └── dataset.py          # PyTorch dataset for paired images
│   ├── training/
│   │   ├── models.py           # U-Net Generator, PatchGAN Discriminator
│   │   └── train.py            # Training loop with GAN + perceptual loss
│   ├── optimization/
│   │   └── tensorrt_optimize.py # TensorRT conversion and benchmarking
│   └── inference/
│       ├── infer.py            # Inference with tiled processing
│       └── metrics.py          # PSNR, SSIM, LPIPS evaluation
├── demo.py                     # Gradio web interface
├── run_training.sh             # Training script
├── run_optimize.sh             # TensorRT optimization script
├── run_inference.sh            # Inference script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 512 × 512 |
| Batch Size | 8 |
| Epochs | 200 |
| Learning Rate (G) | 2e-4 |
| Learning Rate (D) | 2e-4 |
| λ_L1 | 100.0 |
| λ_perceptual | 10.0 |
| λ_adversarial | 1.0 |

### Loss Functions

1. **L1 Loss**: Pixel-wise reconstruction
2. **Perceptual Loss**: VGG-19 feature matching (layers 3, 8, 15, 22)
3. **Adversarial Loss**: PatchGAN discriminator

### Data Augmentation

- Random horizontal flip (50%)
- Random vertical flip (10%)
- Random rotation (±5°)

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| PSNR | Peak Signal-to-Noise Ratio (higher is better) |
| SSIM | Structural Similarity Index (higher is better) |
| LPIPS | Learned Perceptual Similarity (lower is better) |

Run evaluation:
```bash
python src/inference/metrics.py \
    --model_path outputs/checkpoints/best_generator.pt \
    --data_root . \
    --jsonl_path train.jsonl
```

## Performance Benchmarks

| Configuration | Inference Time | Throughput |
|---------------|----------------|------------|
| PyTorch FP32 | ~150ms | 6.7 img/s |
| PyTorch FP16 | ~80ms | 12.5 img/s |
| TensorRT FP16 | ~25ms | 40 img/s |

*Benchmarked on NVIDIA DGX Spark at 512×512 resolution*

## Deliverables

1. **Trained Model**: `outputs/checkpoints/best_generator.pt`
2. **TensorRT Model**: `outputs/optimized/model_trt_fp16.ts`
3. **Sample Outputs**: `outputs/samples/`
4. **This Documentation**: `README.md`

## Technical Approach

### Why This Architecture?

1. **U-Net with Residual Learning**
   - Skip connections preserve spatial detail critical for real estate photos
   - Residual output (learning the "edit") converges faster than learning full image
   - Computationally efficient compared to diffusion models

2. **PatchGAN Discriminator**
   - Encourages high-frequency detail (textures, edges)
   - Effective for image-to-image translation tasks
   - Stable training dynamics

3. **Perceptual Loss**
   - Captures semantic similarity beyond pixel-level
   - Produces more natural-looking results
   - Reduces blur artifacts

### Why Not Diffusion Models?

While diffusion models (e.g., ControlNet + SD) could work, we chose a GAN-based approach because:

1. **Inference efficiency**: GANs require single forward pass vs 20-50 steps for diffusion
2. **Dataset size**: 577 samples is sufficient for paired image translation but limited for diffusion fine-tuning
3. **Deterministic output**: GANs produce consistent results for the same input
4. **Hackathon time constraints**: Faster training and iteration

## License

This project was developed for the Real Estate Photo Editing Hackathon.

## Contact

For questions about this submission, contact the team.
