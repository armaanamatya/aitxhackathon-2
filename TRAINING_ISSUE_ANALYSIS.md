# Training Issue Analysis & Solution

## Problem: Validation Loss Stuck at 1.2904

### Symptoms
```
Epoch 1: Train=1.3313, Val=1.2904
Epoch 2: Train=1.3249, Val=1.2904
Epoch 3: Train=1.3294, Val=1.2904
...
Epoch 9: Train=1.3306, Val=1.2904

Components (IDENTICAL across all epochs):
char=0.2787, win=0.3260, hsv=0.3108, lab=0.1049,
hist=0.0544, grad=0.0178, fft=0.0019, perc=0.1959
```

### Root Cause: **Over-Complicated Loss Function**

The `unified_hdr_loss.py` had **8 different loss components**:

1. **Charbonnier Loss** - OK
2. **Window-Aware Loss** - OK
3. **HSV Color Loss** - Problematic (hue discontinuity at 0/360°)
4. **LAB Color Loss** - **MAJOR ISSUE** (exploded to 17.5, 94% of total!)
5. **Histogram Loss** - Numerically unstable (soft binning, CDF)
6. **Gradient Loss** - OK but adds noise
7. **FFT Loss** - Questionable for this task
8. **VGG Perceptual** - Memory intensive

#### Why LAB Loss Exploded

In run 2:
```
Epoch 1: Train=18.7584, Val=18.5435
Components: ..., lab=17.5027, ...  (94% of total loss!)
```

The LAB color space conversion involves:
- RGB → Linear sRGB (power 2.4)
- Linear RGB → XYZ (matrix multiply)
- XYZ → LAB (cube root + scaling)

**Issues:**
1. Numerical instability in dark/bright regions
2. Even with normalization, the loss scale was wrong
3. Gradient flow was dominated by LAB, starving other components

#### Why Validation Was Stuck

The loss landscape became **non-convex and flat** due to:

1. **Conflicting gradients** from 8 components pulling in different directions
2. **Scale mismatch** - LAB dominated, other components had no influence
3. **Numerical precision issues** - mixed precision + complex math = trouble
4. **Constant validation outputs** - model couldn't find a gradient to follow

Evidence: Training loss changed (1.3313 → 1.3226) but validation was FROZEN.

## Solution: Simplified Robust Loss

### New Design Philosophy

**KISS Principle**: Only include losses that are:
1. ✅ Numerically stable
2. ✅ Properly scaled
3. ✅ Actually useful for the task
4. ✅ Differentiable without issues

### Simplified Loss Components

#### Core (Always Active)
1. **L1 Loss** - Base reconstruction
   - Simple, stable, well-understood
   - Scale: typically 0.1-0.3 for [0,1] images

2. **Window-Aware L1** - 3x weight on bright regions
   - Simple luminance threshold (>0.75)
   - Smooth sigmoid weighting
   - Scale: similar to L1

#### Optional
3. **VGG Perceptual** (0.1x weight)
   - Only up to conv3_4 (not full VGG19)
   - Pre-normalized for stability
   - Can disable for faster training

### What Was Removed & Why

| Removed | Reason |
|---------|--------|
| HSV Loss | Hue discontinuity at 0°/360°, unstable gradients |
| LAB Loss | Numerical instability, exploded to 17.5 |
| Histogram | Soft binning + CDF = unstable, high memory |
| FFT Loss | Questionable benefit, adds noise |
| Gradient (Sobel) | Edges already captured by L1 + perceptual |

### Expected Behavior

#### Good Training Looks Like:
```
Epoch  1: Train=0.245, Val=0.231, L1=0.105, Win=0.126, Grad=0.45
Epoch  2: Train=0.228, Val=0.219, L1=0.098, Win=0.121, Grad=0.42
Epoch  3: Train=0.215, Val=0.208, L1=0.092, Win=0.116, Grad=0.38
...
```

**Key indicators:**
- ✅ Validation loss DECREASES
- ✅ Gradient norm shows learning (decreases over time)
- ✅ L1 and Window components are balanced (similar scale)
- ✅ Individual components also decrease

#### Red Flags:
- ❌ Validation stuck (same value for 3+ epochs)
- ❌ Any component > 10.0 (scale mismatch)
- ❌ Gradient norm < 0.01 or > 100 (vanishing/exploding)
- ❌ Training loss increases

## Training Scripts

### Option 1: Fast (No Perceptual) - Recommended to Start
```bash
sbatch train_restormer_simple_fast.sh
```

**Pros:**
- Faster training (~30% speedup)
- Less memory
- Simpler to debug
- Still captures window importance

**Cons:**
- May miss some fine texture details
- Color might be slightly less accurate

### Option 2: With Perceptual (VGG)
```bash
sbatch train_restormer_simple_perceptual.sh
```

**Pros:**
- Better texture/color preservation
- Perceptual quality often better

**Cons:**
- Slower training
- More memory (VGG features)
- Slightly more complex

## Architecture Analysis

### Restormer Model
```
Input (3, 512, 512)
  ↓
Patch Embed (dim=48)
  ↓
Encoder Level 1: 4 Transformer Blocks (dim=48, heads=1)
  ↓ Downsample
Encoder Level 2: 6 Transformer Blocks (dim=96, heads=2)
  ↓ Downsample
Encoder Level 3: 6 Transformer Blocks (dim=192, heads=4)
  ↓ Downsample
Bottleneck: 8 Transformer Blocks (dim=384, heads=8)
  ↓ Upsample + Skip
Decoder Level 3: 6 Transformer Blocks (dim=192, heads=4)
  ↓ Upsample + Skip
Decoder Level 2: 6 Transformer Blocks (dim=96, heads=2)
  ↓ Upsample + Skip
Decoder Level 1: 6 Transformer Blocks (dim=48, heads=1)
  ↓
Refinement: 4 Transformer Blocks (dim=48)
  ↓
Output Conv (dim=48 → 3)
  ↓
Output (3, 512, 512)

Total params: ~21.9M
```

**Key Features:**
- U-Net architecture with skip connections
- Multi-Dconv Head Transposed Attention (MDTA) - efficient for high-res
- Gated-Dconv Feed-Forward Network (GDFN) - no ReLU, uses gating
- LayerNorm instead of BatchNorm (more stable)

**Why This Model Works for HDR:**
1. ✅ Handles high-res (512x512) efficiently via channel-wise attention
2. ✅ U-Net preserves spatial details through skip connections
3. ✅ Large receptive field captures global illumination
4. ✅ No BatchNorm = consistent train/eval behavior

## Monitoring Training

### Check Progress
```bash
tail -f outputs_restormer_simple_fast/training.log
```

### What to Look For

#### First 5 Epochs (Warmup)
- Learning rate ramps: 4e-5 → 8e-5 → 1.2e-4 → 1.6e-4 → 2e-4
- Loss should decrease steadily
- Gradient norm typically 0.3-1.0

#### After Warmup
- Cosine annealing LR schedule kicks in
- Validation should improve for 10-20 epochs
- Early stopping patience = 15 epochs

### Troubleshooting

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| Val stuck again | Loss still too complex | Try even simpler (pure L1) |
| Loss explodes (>10) | Learning rate too high | Reduce to 1e-4 |
| Underfitting | Model too small | Increase `--dim` to 64 |
| Overfitting quickly | Not enough data aug | Add more augmentations |
| OOM | Batch too large | Reduce `--batch_size` to 2 |

## Expected Results

Based on similar architectures on HDR enhancement:

**Simplified Loss (no perceptual):**
- PSNR: 22-25 dB (baseline)
- Training time: ~6-8 hours on A100

**With Perceptual:**
- PSNR: 23-26 dB
- Perceptual quality: better
- Training time: ~8-10 hours on A100

**Previous Complex Loss:**
- ❌ Validation stuck, unusable

## Next Steps

1. ✅ **Start with fast version** - validate the fix works
   ```bash
   sbatch train_restormer_simple_fast.sh
   ```

2. **Monitor first 10 epochs** - ensure validation improves

3. **If working well**, launch perceptual version:
   ```bash
   sbatch train_restormer_simple_perceptual.sh
   ```

4. **Compare results** on test set using `run_test_inference.py`

5. **If still having issues**, fall back to **pure L1**:
   - Remove window-aware loss
   - Just optimize pixel-wise L1
   - Simplest possible baseline

## Key Takeaways

1. **Simpler is better** - Complex losses don't always help
2. **Scale matters** - All components should be O(0.1-1.0)
3. **Validate numerics** - Check if any component dominates
4. **Monitor gradients** - They tell you if learning is happening
5. **Debug incrementally** - Start simple, add complexity only if needed

---
*Generated: 2025-12-14*
*Context: Fixing stuck validation loss in Restormer HDR training*
