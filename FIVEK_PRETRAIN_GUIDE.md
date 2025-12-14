# MIT-Adobe FiveK Pre-training Pipeline
## Top 0.0001% MLE Complete Guide

---

## 📊 Strategy Overview

**Objective:** Improve real estate photo enhancement through transfer learning

**Approach:** Pre-train on FiveK (5K general photos) → Fine-tune on real estate (577 photos)

**Expected Improvement:** +10-20% on test L1 loss

---

## 🎯 Why Pre-training Works

### Dataset Comparison

| Aspect | FiveK | Real Estate |
|--------|-------|-------------|
| **Size** | ~5,000 pairs | 577 pairs |
| **Domain** | General photography | Real estate interiors |
| **Transformations** | Professional retouching | HDR-style enhancement |
| **Variety** | High (landscapes, portraits, etc.) | Medium (interiors, exteriors) |

### Transfer Learning Benefits

1. **Overcomes data scarcity** - 577 pairs is small, FiveK adds 5K more
2. **Learns general patterns** - Brightness, contrast, color correction
3. **Better initialization** - Start from photo enhancement, not random weights
4. **Regularization effect** - Pre-training prevents overfitting on small dataset

### Domain Gap Analysis

**Similarities:**
- ✅ Both involve brightness/exposure adjustment
- ✅ Both involve color grading
- ✅ Both involve local tone mapping
- ✅ Both are image-to-image translation

**Differences:**
- ⚠️ FiveK: diverse scenes vs Real estate: specific interiors
- ⚠️ FiveK: artistic retouching vs Real estate: HDR-style
- ⚠️ FiveK: expert C subjective vs Real estate: consistent +33% midtone boost

**Verdict:** Domain gap is moderate → Transfer learning should help significantly

---

## 🔧 Pipeline Components

### 1. Download FiveK Dataset

```bash
# Download (in progress - 50GB, ~30min)
cd fivek_dataset
wget https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar
tar -xf fivek_dataset.tar
```

**Status:** Currently downloading (28% complete)

### 2. Prepare FiveK Data (Robust Pipeline)

```bash
chmod +x prepare_fivek_robust.py
/cm/local/apps/python39/bin/python3 prepare_fivek_robust.py
```

**Features:**
- ✅ **Validation**: Checks image quality, resolution, compatibility
- ✅ **Error handling**: Graceful failures, detailed error reporting
- ✅ **Resume capability**: Can resume interrupted processing
- ✅ **Format conversion**: Converts to JPG, downsamples large images
- ✅ **Metadata tracking**: Saves statistics and configuration

**Output:**
```
fivek_processed/
├── images/               # All image pairs
│   ├── image_001_src.jpg
│   ├── image_001_tar.jpg
│   └── ...
├── train.jsonl           # ~4750 pairs (95%)
├── val.jsonl             # ~250 pairs (5%)
└── dataset_metadata.json # Complete metadata
```

### 3. Pre-train on FiveK

**Hyperparameters (optimized for large dataset):**
- Resolution: 512 (faster training)
- Batch size: 8
- Epochs: 30 (fewer epochs on large dataset)
- Learning rate: **5e-5** (lower for stability)
- Warmup: 5 epochs
- Early stopping: 10 epoch patience
- Augmentation: Light (flip only)

**Why these settings:**
- Lower LR prevents catastrophic forgetting
- Fewer epochs sufficient for large dataset
- Light augmentation (FiveK already diverse)
- Early stopping saves compute

### 4. Fine-tune on Real Estate

**Hyperparameters (optimized for domain adaptation):**
- Resolution: **896** (higher for quality)
- Batch size: 4 (large images)
- Epochs: 100
- Learning rate: **1e-5** (very low to preserve features)
- Warmup: 5 epochs
- Early stopping: **15 epoch patience** (more conservative)
- Augmentation: Light (flip only)
- **Load from:** Pre-trained FiveK checkpoint

**Why these settings:**
- **10x lower LR** (1e-5 vs 1e-4) - critical!
  - Preserves pre-trained features
  - Small updates to adapt domain
  - Prevents overwriting general knowledge
- Higher resolution for final quality
- More patience (fine-tuning can be noisy)
- Resume from FiveK checkpoint

---

## 🚀 Execution

### Quick Start (Automated Pipeline)

```bash
# 1. Prepare FiveK (run once, after download completes)
/cm/local/apps/python39/bin/python3 prepare_fivek_robust.py

# 2. Run complete pre-train → fine-tune pipeline
chmod +x train_with_pretrain.sh
sbatch train_with_pretrain.sh
```

**Time estimates:**
- FiveK download: ~30 min (in progress)
- FiveK preparation: ~15-30 min
- Pre-training: ~3-4 hours
- Fine-tuning: ~5-6 hours
- **Total: ~9-11 hours**

### Manual Control (Step-by-step)

```bash
# Phase 1: Pre-train on FiveK
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl fivek_processed/train.jsonl \
    --resolution 512 \
    --batch_size 8 \
    --epochs 30 \
    --lr 5e-5 \
    --early_stopping_patience 10 \
    --preprocess light_aug \
    --output_dir outputs_fivek_pretrained \
    --mixed_precision

# Phase 2: Fine-tune on Real Estate
/cm/local/apps/python39/bin/python3 train_restormer_cleaned.py \
    --train_jsonl train.jsonl \
    --resolution 896 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-5 \
    --early_stopping_patience 15 \
    --preprocess light_aug \
    --resume outputs_fivek_pretrained/checkpoint_best.pt \
    --output_dir outputs_finetuned_from_fivek \
    --mixed_precision

# Phase 3: Test
/cm/local/apps/python39/bin/python3 run_test_inference.py \
    --model outputs_finetuned_from_fivek/checkpoint_best.pt \
    --resolution 896 \
    --output_dir test/finetuned_from_fivek
```

---

## 📊 Expected Results

### Baseline (No Pre-training)
- Restormer 896 trained from scratch
- Test L1 loss: **0.0514**
- Training time: ~6 hours

### With FiveK Pre-training
- Pre-train on 5K pairs → Fine-tune on 577 pairs
- Expected test L1 loss: **0.041-0.046** (+10-20% improvement)
- Training time: ~9-11 hours

### Why Improvement?
1. **Better initialization** - starts from photo-aware weights
2. **Regularization** - pre-training acts as regularizer
3. **More data** - effectively 5577 pairs instead of 577
4. **Domain adaptation** - gradually shifts from general → specific

---

## 🔬 Validation Strategy

### Monitor These Metrics

1. **Pre-training phase:**
   - FiveK val loss should decrease steadily
   - Should converge faster than scratch (good initialization)
   - Target: Val loss < 0.055 (similar to real estate)

2. **Fine-tuning phase:**
   - Initial val loss will be higher (domain gap)
   - Should decrease rapidly in first 10 epochs
   - Then slowly improve as model adapts
   - Watch train/val gap (should be smaller than scratch)

3. **Test evaluation:**
   - Compare test L1 loss vs baseline
   - Look for +10-20% improvement
   - Check visual quality (less artifacts)

### Red Flags

❌ **Pre-training val loss not decreasing** → LR too high or data issues
❌ **Fine-tuning val loss increases** → LR too high, overfitting
❌ **Test loss worse than baseline** → Domain gap too large, strategy failed
❌ **Large train/val gap** → Overfitting, need more regularization

---

## 🎓 Top 0.0001% MLE Considerations

### Hyperparameter Choices (Evidence-based)

**Learning Rate Ratio:**
- Pre-training: 5e-5
- Fine-tuning: 1e-5 (5x lower)
- **Why:** Fine-tuning requires smaller updates to preserve features
- **Source:** "How transferable are features in deep neural networks?" (Yosinski et al., 2014)

**Early Stopping Patience:**
- Pre-training: 10 epochs (large dataset converges fast)
- Fine-tuning: 15 epochs (small dataset is noisy)
- **Why:** Small datasets have noisier gradients, need more patience
- **Source:** Standard practice in fine-tuning (BERT, GPT, etc.)

**Resolution Strategy:**
- Pre-training: 512 (speed)
- Fine-tuning: 896 (quality)
- **Why:** Learn general patterns fast, then refine at high-res
- **Source:** Progressive training (ProGAN, StyleGAN)

### Alternative Strategies (If This Fails)

1. **Freeze early layers** - Only fine-tune last N layers
2. **Gradual unfreezing** - Unfreeze layers progressively
3. **Discriminative LR** - Different LR per layer (lower for early layers)
4. **Mix datasets** - Train on FiveK + Real estate simultaneously
5. **Data augmentation** - More aggressive augmentation on small dataset

### Measuring Success

**Quantitative:**
- Test L1 loss improvement: +10-20%
- Faster convergence: 30% fewer epochs
- Lower validation loss: Better generalization

**Qualitative:**
- Fewer artifacts (especially in shadows)
- Better color consistency
- More natural transitions

---

## 📁 File Structure

```
autohdr-real-estate-577/
├── fivek_dataset/              # Downloaded FiveK
│   ├── input/
│   └── output_expert_C/
├── fivek_processed/            # Processed FiveK
│   ├── images/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── dataset_metadata.json
├── outputs_fivek_pretrained/   # Pre-trained model
│   ├── checkpoint_best.pt
│   └── history.json
├── outputs_finetuned_from_fivek/  # Fine-tuned model
│   ├── checkpoint_best.pt
│   └── history.json
├── test/finetuned_from_fivek/  # Test results
│   └── results.json
├── prepare_fivek_robust.py     # Robust data preparation
└── train_with_pretrain.sh      # Complete pipeline
```

---

## ⏱️ Current Status

- ✅ Download script created
- 🔄 **FiveK downloading** (28% complete, ~20min remaining)
- ✅ Robust preparation pipeline ready
- ✅ Pre-training script ready
- ✅ Fine-tuning script ready
- ⏳ Waiting for download to complete

**Next steps:**
1. Wait for download (monitoring in background)
2. Run `prepare_fivek_robust.py` when complete
3. Launch `train_with_pretrain.sh`
4. Compare results with baseline

---

## 🎯 Success Criteria

**Minimum viable:** Test L1 < 0.046 (+10% improvement)
**Target:** Test L1 < 0.043 (+15% improvement)
**Stretch goal:** Test L1 < 0.041 (+20% improvement)

If we achieve these, FiveK pre-training is worth the extra compute!
