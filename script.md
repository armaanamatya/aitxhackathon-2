# AutoHDR Hackathon Pitch Script
## 2-3 Minutes | Team [Your Team Name]

---

### OPENING - OUR APPROACH (20 seconds)

"We asked ourselves: what makes professional real estate photo editing so hard to automate?

It's not just brightness or contrast. Professionals are doing **global scene understanding**—they know what should be visible through that window, how shadows should fall across the room, and how to preserve color accuracy while recovering highlights.

We chose **Restormer**—a transformer architecture designed specifically for this kind of **global reasoning**."

---

### WHY RESTORMER? (30 seconds)

"Most teams will reach for diffusion models or basic CNNs. Here's why we went different:

**The Problem with Diffusion:**
- Slow inference (20-50 steps)
- High VRAM (8-12GB minimum)
- Can hallucinate details that weren't there

**The Problem with Basic CNNs:**
- Only see local patches
- Miss global lighting relationships
- Window recovery requires understanding the entire scene

**Restormer's Advantage:**
- **Single forward pass** - no iterative sampling
- **Transposed self-attention** - sees the entire image at once
- **25M parameters** - lightweight but powerful
- **Production-proven** - state-of-the-art on image restoration benchmarks

One pass. Full context. Real details preserved."

---

### TECHNICAL DEEP DIVE (45 seconds)

"Let me show you what's under the hood:

**Architecture:**
- 4-level encoder-decoder
- Attention heads: [1, 2, 4, 8] at each level
- Channel progression: 48 → 96 → 192 → 384
- 25.4 million parameters total

**Custom Loss Function - This is Key:**

We didn't use vanilla L1 loss. We designed a **three-component loss**:

1. **L1 Loss (weight 1.0)** - Base pixel accuracy
2. **Window-Aware Loss (weight 0.3)** - Extra penalty for bright regions above 0.7 brightness. This is where professional edits matter most—recovering blown-out windows.
3. **Saturation Preservation Loss (weight 0.2)** - Prevents the washed-out look that ruins amateur HDR

**Why this matters:** Standard losses treat all pixels equally. Ours tells the model: 'Pay extra attention to windows and highlights—that's what professionals fix.'

**Training:**
- 512×512 resolution (balances quality vs. training speed)
- 38 epochs to convergence
- AdamW optimizer, cosine annealing
- Mixed precision FP16 training"

---

### RESULTS (30 seconds)

"On our held-out validation set:

| Metric | Score |
|--------|-------|
| **L1 Loss** | 0.051 |
| **PSNR** | 23.57 dB |
| **SSIM** | 0.927 |

But numbers don't tell the whole story. Look at the outputs:

*[Show before/after comparisons]*

- Windows: Recovered, not clipped
- Shadows: Lifted naturally, not artificially
- Colors: Preserved, not oversaturated
- Details: Sharp, no hallucinated artifacts

This looks like what a professional editor would produce—because we trained on exactly that."

---

### INFERENCE EFFICIENCY (20 seconds)

"Here's where we really shine on the **30% efficiency criteria**:

| Metric | Our Model | Typical Diffusion |
|--------|-----------|-------------------|
| **Inference Steps** | 1 | 20-50 |
| **Time per Image** | ~100ms | 2-5 seconds |
| **VRAM** | 2-4 GB | 8-12 GB |
| **Model Size** | 97 MB | 2-6 GB |

**Single forward pass. No sampling. No iteration.**

We can process **10 images per second** on consumer hardware. That's production-ready scale."

---

### PRODUCTION READINESS (15 seconds)

"This isn't a research prototype. We built:

- **Gradio demo** - Drag and drop interface
- **Batch CLI** - Process entire directories
- **Edge deployment package** - Runs on Jetson, Apple Silicon, or cloud
- **Lossless PNG output** - Maximum quality preservation

An agency could deploy this tomorrow."

---

### CLOSING (15 seconds)

"We combined:
- **State-of-the-art architecture** (Restormer)
- **Custom domain-specific loss** (Window-aware + Saturation)
- **Extreme efficiency** (100ms inference, 97MB model)

Professional quality. Production ready. Blazingly fast.

Thank you."

---

## TECHNICAL SUMMARY (For Submission)

**Model:** Restormer (Transformer-based image restoration)
- 25.4M parameters
- Single forward pass architecture

**Why This Approach:**
- Global attention captures room-wide lighting relationships
- Single-pass inference (no iterative sampling like diffusion)
- Proven architecture for image restoration tasks
- Custom loss function specifically targets real estate editing challenges

**Training:**
- Resolution: 512×512
- Epochs: 38
- Time: ~4 hours on A100
- Data: 511 train / 56 val / 10 test split

**Inference:**
- Time: ~100ms per image (512×512 on GPU)
- VRAM: 2-4 GB
- Model size: 97 MB

**Key Optimizations:**
- Window-aware loss for highlight recovery
- Saturation preservation loss
- FP16 mixed precision
- cuDNN benchmark auto-tuning
- torch.inference_mode (zero gradient overhead)

**Tradeoffs:**
- Trained at 512×512 (not full resolution) for speed
- No perceptual/GAN losses (prioritized pixel accuracy over "style")
- Single-pass architecture (no iterative refinement)

---

## DEMO CHECKLIST

1. Show bad input photo (blown windows, dark interior)
2. Run live inference (<1 second)
3. Before/after slider comparison
4. Zoom on window recovery
5. Show batch processing speed
6. Display VRAM usage (prove efficiency)
