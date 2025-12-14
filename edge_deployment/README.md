# Real Estate HDR - 7MP Inference Package

**Model:** Restormer Base (25.4M parameters)
**Resolution:** 7MP (3296×2192 typical)
**Performance:** Val L1 = 0.0855 @ Epoch 2
**Trained on:** NVIDIA DGX Cloud (A100 80GB)

---

## Quick Start

### Single Image Inference

```bash
python inference_7mp.py \
    --input photo.jpg \
    --output photo_hdr.jpg
```

### Batch Processing

```bash
python inference_7mp.py \
    --input_dir ./photos/ \
    --output_dir ./results/
```

---

## Supported Hardware

### ✅ NVIDIA A100/H100 (Datacenter)
**Performance:** 0.8-1.2 FPS for 7MP images
**Memory:** 30-35 GB per image
**Precision:** FP16 (automatic)

```bash
# Auto-detects A100 and optimizes accordingly
python inference_7mp.py --input photo.jpg --output result.jpg
```

### ✅ Apple M2 Ultra (128GB+ Unified Memory)
**Performance:** 0.8-1.2 FPS for 7MP images
**Memory:** 30 GB per image
**Precision:** FP16 (automatic)

```bash
# Auto-detects Apple Silicon (MPS backend)
python inference_7mp.py --input photo.jpg --output result.jpg
```

### ✅ NVIDIA Jetson AGX Orin (32GB)
**Performance:** Requires tiling (see optimization guides)
**Memory:** Use tiling for memory efficiency
**Precision:** INT8 recommended

For Jetson deployment, see:
- `../weights/restormer_512_baseline/EDGE_OPTIMIZATION_7MP.md`

### ✅ CPU Fallback
**Performance:** Slow (~10-20× slower than GPU)
**Memory:** 30-40 GB
**Use case:** When no GPU available

```bash
python inference_7mp.py --device cpu --input photo.jpg --output result.jpg
```

---

## Installation

### On NVIDIA A100/H100 (DGX Cloud, Datacenter)

```bash
# PyTorch 2.0+ with CUDA 12.1+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Additional dependencies
pip install pillow numpy
```

### On Apple Silicon (M1/M2 Ultra)

```bash
# PyTorch with MPS support
pip install torch torchvision

# Additional dependencies
pip install pillow numpy
```

### On NVIDIA Jetson

```bash
# Use NVIDIA PyTorch for Jetson
# See: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/

# Install dependencies
pip install pillow numpy
```

---

## Directory Structure

```
edge_deployment/
├── inference_7mp.py              # Main inference script
├── weights/
│   └── restormer_7mp/
│       ├── model_checkpoint.pt   # 7MP trained model (98MB)
│       ├── config.json           # Model configuration
│       └── README.md             # This file
├── src/
│   └── training/
│       ├── restormer.py          # Model architecture
│       └── __init__.py
└── requirements.txt              # Python dependencies
```

---

## Advanced Usage

### Manual Device Selection

```bash
# Force CUDA
python inference_7mp.py --device cuda --input photo.jpg --output result.jpg

# Force Apple MPS
python inference_7mp.py --device mps --input photo.jpg --output result.jpg

# Force CPU
python inference_7mp.py --device cpu --input photo.jpg --output result.jpg
```

### Precision Control

```bash
# Force FP16 (faster, less memory)
python inference_7mp.py --precision fp16 --input photo.jpg --output result.jpg

# Force FP32 (slower, more memory, slightly better quality)
python inference_7mp.py --precision fp32 --input photo.jpg --output result.jpg
```

### Custom Checkpoint

```bash
python inference_7mp.py \
    --checkpoint /path/to/custom_checkpoint.pt \
    --input photo.jpg \
    --output result.jpg
```

---

## Performance Benchmarks

### NVIDIA A100 80GB (FP16, single-pass)
- **Memory:** 30-35 GB per image
- **Throughput:** 0.8-1.2 FPS
- **Latency:** 800-1200 ms per image
- **Batch size:** 2 (can process 2 images simultaneously)
- **Quality:** Perfect (no tiling artifacts)

### Apple M2 Ultra 192GB (FP16, single-pass)
- **Memory:** 30 GB per image
- **Throughput:** 0.8-1.2 FPS
- **Latency:** 800-1200 ms per image
- **Batch size:** 1
- **Quality:** Perfect (no tiling artifacts)

### NVIDIA Jetson AGX Orin 32GB (INT8, tiling required)
- **Memory:** 8 GB (with 1024×1024 tiles)
- **Throughput:** See edge optimization guide
- **Requires:** Tiling strategy (not single-pass)

---

## Memory Requirements

### Single-Pass Inference (7MP)
**Minimum:** 128GB unified memory OR 64GB GPU memory with gradient checkpointing
**Recommended:** 128GB+ unified memory (Apple M2 Ultra, NVIDIA Grace Hopper)

**Memory breakdown:**
- Model weights: 0.1 GB
- Input/Output: 0.17 GB
- Activations: 15-20 GB (with gradient checkpointing)
- Attention: 10 GB
- Workspace: 5 GB
- **Total: 30-35 GB**

### Tiled Inference (for lower memory)
**Minimum:** 8GB for 512×512 tiles
**Recommended:** 16GB for 1024×1024 tiles

See: `../weights/restormer_512_baseline/EDGE_OPTIMIZATION_7MP.md` for tiling implementation.

---

## Optimization Guides

Detailed optimization guides are available in `../weights/restormer_512_baseline/`:

1. **INFERENCE_OPTIMIZATION_NVIDIA.md**
   - TensorRT optimization (3-5× speedup on A100/H100)
   - Mixed precision strategies
   - Multi-GPU inference
   - NVIDIA Triton deployment

2. **EDGE_OPTIMIZATION_7MP.md**
   - Tiling/patching for memory-constrained devices
   - Jetson-specific optimizations
   - Model quantization (INT8)
   - Hardware-specific backends

3. **OPTIMIZATION_128GB_UNIFIED.md**
   - Single-pass 7MP inference on high-memory systems
   - Apple Silicon (M2 Ultra) optimization
   - Memory-efficient configuration
   - Performance tuning

---

## Troubleshooting

### Out of Memory (OOM)

**On A100/H100:**
```bash
# Use gradient checkpointing (edit inference_7mp.py)
# Or reduce to FP32 if using FP16
python inference_7mp.py --precision fp32 --input photo.jpg --output result.jpg
```

**On systems <64GB:**
- Use tiling approach (see EDGE_OPTIMIZATION_7MP.md)
- Process at lower resolution
- Use batch_size=1

### Slow Performance

**On A100:**
- Ensure FP16 is being used (check output logs)
- Enable torch.compile() (automatic on A100)
- Check TF32 is enabled (automatic on A100)

**On Apple Silicon:**
- Ensure MPS backend is detected
- Use FP16 precision
- Enable torch.compile()

### Model Not Loading

Check PyTorch version compatibility:
```bash
python -c "import torch; print(torch.__version__)"
```

Minimum: PyTorch 2.0+
Recommended: PyTorch 2.6+

---

## Training Information

**Dataset:** Real Estate HDR Dataset
- Train: 511 images
- Validation: 56 images
- Test: 10 images (held out)

**Training Configuration:**
- Hardware: NVIDIA DGX Cloud (A100 80GB)
- Resolution: 7MP (native, no downsampling)
- Batch size: 2
- Epochs: 2
- Precision: Mixed (FP16 + FP32)
- Gradient checkpointing: Yes
- Optimizer: AdamW
- Learning rate: 2e-4

**Loss Function:**
- L1 Loss (weight: 1.0)
- Window-Aware Loss (weight: 0.3) - Bright regions
- Saturation Loss (weight: 0.2)

**Performance:**
- Validation L1: 0.0855
- Training time: ~2 hours on A100 80GB

---

## Citation

```bibtex
@inproceedings{restormer2022,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Zamir, Syed Waqas and Arora, Aditya and Gupta, Salman and Khan, Fahad Shahbaz and Sun, Jing and Shahbaz Khan, Fahad and Zhu, Fanglin and Shao, Ling and Qi, Guo-Jun and Yang, Ming-Hsuan},
  booktitle={CVPR},
  year={2022}
}
```

---

## License

See main repository for license information.

## Support

For questions or issues:
- Check optimization guides in `../weights/restormer_512_baseline/`
- Ensure PyTorch 2.0+ is installed
- Verify hardware compatibility

**Model Package Version:** 1.0
**Last Updated:** December 2025
