# Training Scripts Documentation
## AutoHDR Real Estate Enhancement Project

This document catalogs all unique training scripts, architectures tried, and their distinguishing features.

---

## Quick Reference Table

| Script | Architecture | Input Channels | Key Innovation | Resolution | L1 |
|--------|-------------|----------------|----------------|------------|-----|
| `train_restormer_512_combined_loss.py` | **Restormer Base** | 3 | **L1 + window + saturation** | 512 | **0.0515** |
| `train_restormer_512_sota_window.py` | Restormer | 3 | Luminance attention maps | 512 | ~0.053 |
| `train_restormer_512_sota_color.py` | Restormer | 3 | SOTA color loss (LAB+FFT+histogram) | 512 | - |
| `train_sota_hdr.py` | Restormer | 3 | Zone-aware + VGG perceptual | 512 | - |
| `train_production.py` | Restormer | 3 | Architectural gain constraints | 512 | - |
| `train_restormer_512_adversarial.py` | Restormer + GAN | 3 | Adversarial + style loss | 512 | - |
| `train_midas_depth.py` | Multi-Feature Restormer | 6 (RGB+D+E+S) | MiDaS depth (experimental) | 512 | 0.0979 |
| `train_multi_feature.py` | Multi-Feature Restormer | 6 | Proxy depth (experimental) | 512 | 0.1139 |
| `train_darkir_cv.py` | DarkIR | 3 | FFT-based + 3-fold CV | 512 | - |
| `train_retinexformer_hdr.py` | RetinexFormer | 3 | Illumination-guided attention | 512 | - |

---

## Category 1: Multi-Feature / Depth-Aware (Latest)

### `train_midas_depth.py` - **MiDaS Optimal Solution**
- **Architecture**: Multi-Feature Restormer (6-channel input)
- **Input**: RGB(3) + MiDaS Depth(1) + Edge(1) + Saturation(1)
- **Loss**: Delta-aware loss (adaptive fixing vs preservation)
- **Key Innovation**: TRUE depth from Intel MiDaS model (not proxy)
- **Why Different**: Uses actual depth estimation to distinguish windows (far/sky) from interior surfaces

### `train_multi_feature.py` - **Multi-Feature Base**
- **Architecture**: Multi-Feature Restormer (6-channel input)
- **Input**: RGB(3) + Proxy Depth(1) + Edge(1) + Saturation(1)
- **Loss**: Delta-aware loss
- **Key Innovation**: All features computed on-the-fly (no external models)
- **Why Different**: Memory efficient, uses luminance-based depth proxy instead of MiDaS

### `train_depth_aware.py` - **Depth-Aware Window Recovery**
- **Architecture**: Depth Restormer (4-channel input)
- **Input**: RGB(3) + Depth(1)
- **Loss**: Delta-aware loss
- **Key Innovation**: Uses depth as 4th channel to help distinguish windows

---

## Category 2: Restormer Variants (Core Architecture)

### `train_production.py` - **Production HDR Model**
- **Architecture**: Restormer with architectural constraints
- **Loss**: L1 + window-aware
- **Key Innovation**: Gain constraints enforce windows can only darken [0.3-1.0], interior can only brighten [1.0-3.0]
- **Why Different**: Architectural enforcement prevents model from making incorrect transformations

### `train_sota_hdr.py` - **SOTA HDR with Zone-Aware Loss**
- **Architecture**: Restormer Base (25.4M params)
- **Loss**: Zone-aware loss + VGG perceptual + color consistency
- **Key Innovation**: Different weights for windows/shadows/midtones based on GT analysis
- **Why Different**: GT analysis showed windows only need -7 brightness (not aggressive darkening)

### `train_restormer_512_sota_window.py` - **SOTA Window Recovery**
- **Architecture**: Restormer Base
- **Loss**: Luminance attention loss with zone-specific weighting
- **Key Innovation**: Learned luminance attention maps (heat maps) instead of fixed thresholds
- **Why Different**: Adaptive exposure zone detection like OENet, AGCSNet papers

### `train_restormer_512_sota_color.py` - **SOTA Color Enhancement**
- **Architecture**: Restormer Base
- **Loss**: Focal Frequency Loss + LAB Perceptual + Multi-Scale Color + Histogram Matching
- **Key Innovation**: Complete SOTA color loss suite from CVPR 2021 papers
- **Why Different**: Focuses on color accuracy over structure

### `train_restormer_512_adversarial.py` - **Adversarial + Style Training**
- **Architecture**: Restormer + PatchDiscriminator (GAN)
- **Loss**: L1 + Adversarial + Gram matrix style + histogram
- **Key Innovation**: Forces outputs to match TARGET DISTRIBUTION, not just mean
- **Why Different**: Addresses L1 regression producing "average" muted colors

### `train_restormer_512_combined_loss.py` - **Combined Loss Baseline**
- **Architecture**: Restormer Base
- **Loss**: L1(1.0) + Window(0.5) + Color/HSV(0.3)
- **Key Innovation**: Baseline combined loss for comparison
- **Why Different**: Standard reference implementation

### `train_restormer_512_bright_color.py` - **Bright Region Color Focus**
- **Architecture**: Restormer Base
- **Loss**: L1 + Window + BrightColor (Hue + Chroma + RGB + Histogram)
- **Key Innovation**: Color accuracy specifically in bright regions (windows, sky, plants)
- **Why Different**: Targets the specific color problem in overexposed areas

### `train_restormer_512_color_perceptual.py` - **Color + Perceptual + SSIM**
- **Architecture**: Restormer Base
- **Loss**: L1 + VGG Perceptual + SSIM + HSV Color + LAB Color
- **Key Innovation**: Perceptually uniform color space matching (LAB)
- **Why Different**: Uses perceptually-accurate color spaces

### `train_robust_window.py` - **Robust Window Recovery**
- **Architecture**: Restormer Base
- **Loss**: Delta-aware loss
- **Key Innovation**: Identifies what NEEDS fixing (source != target) vs preservation
- **Why Different**: Per-region adaptive treatment based on input-target delta

### `train_efficient_window.py` - **Efficient Zone-Based Training**
- **Architecture**: Restormer Base
- **Loss**: Adaptive zone loss with percentile-based thresholds
- **Key Innovation**: Percentile-based thresholds (not fixed), color direction loss
- **Why Different**: Memory efficient with dynamic threshold computation

### `train_restormer_optimal.py` - **Optimal Unified HDR**
- **Architecture**: Restormer Base
- **Loss**: Unified HDR Loss (window + color + edge + perceptual + frequency)
- **Key Innovation**: All-in-one loss combining 5 components
- **Why Different**: Comprehensive loss covering all aspects

### `train_restormer_simple_robust.py` - **Simplified Robust Training**
- **Architecture**: Restormer Base
- **Loss**: L1 + Window-Aware L1 (3x weight) + optional VGG
- **Key Innovation**: Simplified 3-component loss to fix stuck validation
- **Why Different**: Removes numerical instability, more aggressive LR schedule

### `train_restormer_cleaned.py` - **Base Restormer with Preprocessing**
- **Architecture**: Restormer Base
- **Loss**: Standard L1
- **Key Innovation**: Configurable preprocessing pipeline
- **Why Different**: Focus on data preprocessing rather than loss functions

### `train_restormer_ddp.py` - **Multi-GPU Distributed Training**
- **Architecture**: Restormer Base
- **Loss**: Standard
- **Key Innovation**: Distributed Data Parallel for 2x H200 GPUs
- **Why Different**: Scales to 7MP resolution (3296x2192) without tiling

---

## Category 3: Elite Color Refiner (Frozen Backbone)

### `train_elite_color_refiner.py` - **Full Elite Refiner**
- **Architecture**: Frozen Restormer896 backbone + trainable refiner
- **Loss**: Multi-loss with adaptive weighting, EMA, curriculum learning
- **Key Innovation**: Frozen backbone prevents forgetting, refiner learns color adjustment
- **Why Different**: Two-stage approach for color refinement

### `train_elite_refiner_fast.py` - **Fast Elite Refiner**
- **Architecture**: Frozen backbone + trainable refiner
- **Loss**: Charbonnier + 3x HSV (simplified)
- **Key Innovation**: 40% faster training, removed VGG/FFT/Histogram
- **Why Different**: Streamlined for rapid iteration

### `train_elite_refiner_normalized.py` - **Normalized Loss Refiner**
- **Architecture**: Frozen backbone + trainable refiner
- **Loss**: Normalized Charbonnier + 0.6x HSV
- **Key Innovation**: Loss components scaled to [0,1] for interpretability
- **Why Different**: Makes loss values comparable across experiments

### `train_elite_refiner_combined.py` - **Combined Loss Refiner**
- **Architecture**: Frozen backbone + trainable refiner
- **Loss**: Same as backbone (L1 + Window + Saturation)
- **Key Innovation**: Fair comparison with backbone using identical loss
- **Why Different**: Apples-to-apples comparison

---

## Category 4: DarkIR (FFT-based)

### `train_darkir_cv.py` - **DarkIR with Cross-Validation**
- **Architecture**: DarkIR-m (3.31M) or DarkIR-l (12.96M params)
- **Loss**: L1 + VGG Perceptual + SSIM
- **Key Innovation**: 3-fold CV + ensemble inference + pretrained LOLBlur weights
- **Why Different**: FFT-based architecture designed for low-light

### `train_darkir_simple.py` - **Simple DarkIR**
- **Architecture**: DarkIR
- **Loss**: L1
- **Key Innovation**: Minimal, stable baseline
- **Why Different**: Debugging/baseline reference

### `train_darkir_window_aware.py` - **DarkIR + Window Loss**
- **Architecture**: DarkIR
- **Loss**: Window-aware loss
- **Key Innovation**: Combines FFT architecture with window-focused training
- **Why Different**: FFT helps with frequency-domain window recovery

### `train_darkir_fivek_transfer.py` - **DarkIR with FiveK Transfer**
- **Architecture**: DarkIR
- **Loss**: Window-aware
- **Key Innovation**: Two-phase: pretrain on FiveK, finetune on real estate
- **Why Different**: Domain adaptation from general tone mapping

---

## Category 5: RetinexFormer / RetinexMamba (Physics-based)

### `train_retinexformer_hdr.py` - **RetinexFormer HDR**
- **Architecture**: RetinexFormer (ICCV 2023 / ECCV 2024)
- **Loss**: HDR losses
- **Key Innovation**: Physics-based Retinex theory separates illumination from reflectance
- **Why Different**: Illumination-guided attention naturally handles bright/dark regions differently

### `train_retinexmamba.py` - **RetinexMamba Training**
- **Architecture**: RetinexMamba (State-space + Retinex)
- **Loss**: Charbonnier + SSIM + FFT + Perceptual + LPIPS + LAB
- **Key Innovation**: Combines Mamba efficiency with Retinex physics
- **Why Different**: Illumination consistency loss for Retinex decomposition

---

## Category 6: Other Transformer Architectures

### `train_mamba.py` - **MambaDiffusion**
- **Architecture**: MambaIR (ECCV 2024 / CVPR 2025) + optional diffusion
- **Loss**: Charbonnier + SSIM + LPIPS + LAB + Histogram
- **Key Innovation**: State-space model with O(N) complexity instead of O(N^2)
- **Why Different**: Linear complexity for high resolution

### `train_hat.py` - **HAT (Hybrid Attention Transformer)**
- **Architecture**: HAT with Window + Channel + Overlapping Cross-Attention
- **Loss**: Multi-loss for color accuracy
- **Key Innovation**: Three types of attention mechanisms combined
- **Why Different**: Stronger attention mechanism than standard transformers

### `train_dat.py` - **DAT (Dual Aggregation Transformer)**
- **Architecture**: DAT (ICCV 2023)
- **Loss**: Charbonnier + SSIM + LPIPS + LAB + Histogram
- **Key Innovation**: Dual aggregation for spatial and channel attention
- **Why Different**: ICCV 2023 state-of-the-art architecture

### `train_nafnet_stable.py` - **Stable NAFNet**
- **Architecture**: NAFNet (simplified baseline)
- **Loss**: Simple L1
- **Key Innovation**: FP32 only, conservative LR, aggressive gradient clipping
- **Why Different**: Fixes NaN loss issues, extremely stable training

---

## Category 7: ControlNet / Diffusion-based

### `train_controlnet.py` - **ControlNet Training**
- **Architecture**: Stable Diffusion 2.1 + ControlNet
- **Loss**: Diffusion loss
- **Key Innovation**: Photorealistic generation capability
- **Why Different**: 95-98% quality but 100x slower inference (2-5s vs 0.03s)
- **Note**: Not recommended for hackathon (cost penalty too high)

### `train_controlnet_enhancement.py` - **ControlNet Enhancement**
- **Architecture**: SD 2.1 + ControlNet at 1024x1024
- **Loss**: Diffusion loss
- **Key Innovation**: Max resolution on 80GB A100
- **Why Different**: Highest quality possible with diffusion

### `train_controlnet_restormer_cv.py` - **ControlNet-Restormer CV**
- **Architecture**: Restormer with ControlNet training strategy
- **Loss**: L1 + VGG + SSIM
- **Key Innovation**: Cross-validation + pretrained SIDD/GoPro weights
- **Why Different**: Uses ControlNet robustness tricks without diffusion overhead

### `train_restormer_controlnet_style.py` - **Restormer with Zero-Conv**
- **Architecture**: Locked Restormer + trainable copy with zero-convolutions
- **Loss**: Standard
- **Key Innovation**: Zero-convolution layers prevent catastrophic forgetting
- **Why Different**: Small dataset adaptation (464 samples) without forgetting

---

## Category 8: Distributed / Optimized Training

### `src/training/train_restormer_deepspeed_512.py` - **DeepSpeed Training**
- Uses DeepSpeed for memory optimization and multi-GPU scaling

### `src/training/train_restormer_fsdp_512.py` - **FSDP Training**
- Fully Sharded Data Parallel for extreme memory efficiency

### `src/training/train_mamba_deepspeed_512.py` - **Mamba + DeepSpeed**
- MambaIR with DeepSpeed optimization

### `src/training/train_mamba_fsdp_512.py` - **Mamba + FSDP**
- MambaIR with FSDP for multi-GPU

---

## Summary: Evolution of Approaches

1. **Phase 1 - Architecture Exploration**: Tested Restormer, DarkIR, RetinexFormer, MambaIR, HAT, DAT, NAFNet, ControlNet

2. **Phase 2 - Loss Function Exploration**: Combined loss, zone-aware loss, adversarial loss, SOTA color loss, luminance attention loss

3. **Phase 3 - Window-Specific Solutions**: Window-aware loss, delta-aware loss, production constraints

4. **Phase 4 - Multi-Feature Innovation**: Added depth (proxy then MiDaS), edge, saturation as additional input channels (experimental, did not improve results)

---

## Best Results

| Model | L1 Loss | PSNR | SSIM | Notes |
|-------|---------|------|------|-------|
| **Restormer 512 Baseline** | **0.0515** | **23.57 dB** | **0.9273** | BEST - Simple L1 + window-aware + saturation |
| MiDaS Optimal | 0.0979 | 18.17 dB | - | Multi-feature experiment |
| Multi-Feature Base | 0.1139 | 16.56 dB | - | Proxy depth experiment |

**Winner**: Restormer 512 Baseline with combined loss (L1 + window-aware + saturation)

---

## Key Insights

1. **Restormer Base** (25.4M params) is the best architecture for this task
2. **Simple losses win**: L1 + window-aware + saturation (0.0515) beats complex multi-feature approaches
3. **Multi-feature experiments did NOT improve results** - MiDaS depth and proxy depth both performed worse than baseline
4. **Don't over-engineer**: The baseline Restormer with straightforward loss outperformed all experimental approaches
5. **ControlNet** quality is excellent but inference cost is prohibitive for production
