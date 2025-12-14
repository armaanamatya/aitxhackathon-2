# Restormer 512 Baseline - Real Estate HDR Model

**Model:** Restormer (CVPR 2022)
**Task:** Real Estate HDR Image Enhancement
**Resolution:** 512x512
**Performance:** L1=0.0515, PSNR=23.57, SSIM=0.9273 (test set)

---

## Model Overview

This is a Restormer-based model trained for real estate HDR image enhancement. The model transforms standard real estate photos into professionally edited HDR images with enhanced brightness, color, and window details.

**Key Features:**
- Window-aware loss for bright regions (windows, sky)
- Saturation preservation in bright areas
- 25.4M parameters (96.97 MB model size)
- Trained on 511 images, validated on 56, tested on 10 held-out images

**Training Job:** 609756
**Best Checkpoint:** Epoch 38
**Validation L1:** 0.0530

---

## Files in this Directory

```
weights/restormer_512_baseline/
├── model_checkpoint.pt      # PyTorch checkpoint (includes model, optimizer, scheduler)
├── config.json              # Training configuration and hyperparameters
├── architecture.json        # Model architecture specification
├── training_info.json       # Training metrics and epoch info
├── load_model.py            # Python script to load the model
└── README.md                # This file
```

---

## Quick Start

### 1. Load the Model (Python)

```python
import torch
from src.training.restormer import create_restormer

# Create model
model = create_restormer('base')

# Load checkpoint
checkpoint = torch.load('model_checkpoint.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model loaded! Epoch: {checkpoint['epoch']}, Val L1: {checkpoint['val_l1']:.4f}")
```

### 2. Run Inference

```python
from PIL import Image
import torchvision.transforms.functional as TF

# Load and preprocess image
img = Image.open('input.jpg').convert('RGB')
img_resized = TF.resize(img, (512, 512))
img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(img_tensor)
    output = torch.clamp(output, 0, 1)

# Convert back to PIL image
output_img = TF.to_pil_image(output.squeeze(0).cpu())
output_img.save('output_hdr.jpg')
```

### 3. Batch Inference

```python
import os
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class HDRInferenceDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_path = sample.get('input', sample.get('src'))

        img = Image.open(input_path).convert('RGB')
        img = TF.resize(img, (512, 512))
        img_tensor = TF.to_tensor(img)

        return img_tensor, Path(input_path).stem

# Create dataloader
dataset = HDRInferenceDataset('data_splits/proper_split/test.jsonl')
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

# Run batch inference
output_dir = Path('inference_results')
output_dir.mkdir(exist_ok=True)

model.eval()
with torch.no_grad():
    for batch, names in loader:
        batch = batch.to(device)
        outputs = model(batch)
        outputs = torch.clamp(outputs, 0, 1)

        for output, name in zip(outputs, names):
            img = TF.to_pil_image(output.cpu())
            img.save(output_dir / f'{name}_hdr.png')
```

---

## Model Architecture

**Type:** Encoder-Decoder Transformer (Restormer)

**Structure:**
- Encoder blocks: [4, 6, 6, 8]
- Decoder blocks: [8, 6, 6, 4]
- Refinement blocks: 4
- Channels: [48, 96, 192, 384]
- Attention heads: [1, 2, 4, 8]

**Total Parameters:** 25,423,939 (25.4M)

See `architecture.json` for complete specifications.

---

## Training Configuration

**Loss Function:**
- L1 Loss (weight: 1.0) - Base pixel accuracy
- Window-Aware Loss (weight: 0.3) - Extra weight for bright regions
- Saturation Loss (weight: 0.2) - Preserve saturation in bright areas

**Hyperparameters:**
- Resolution: 512x512
- Batch size: 4
- Learning rate: 2e-4
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Mixed precision: Yes (FP16)

**Data Split:**
- Train: 511 images (with augmentation: horizontal/vertical flips)
- Validation: 56 images
- Test: 10 images (held out, not used during training)

See `config.json` for complete configuration.

---

## Performance Metrics

**Test Set (10 held-out images):**
- L1: 0.0515 ± 0.0137
- PSNR: 23.57 ± 2.35 dB
- SSIM: 0.9273 ± 0.0225

**Validation Set:**
- L1: 0.0530 (best checkpoint)

**Training:**
- Epochs trained: 38
- Training stopped when validation loss plateaued

---

## System Requirements

**Minimum:**
- Python 3.9+
- PyTorch 2.0+
- 4GB GPU VRAM (for inference at 512x512)
- 8GB RAM

**Recommended:**
- Python 3.9+
- PyTorch 2.6+
- CUDA 12.1+
- 8GB+ GPU VRAM (A100, H100, RTX 4090, etc.)
- 16GB+ RAM

**Dependencies:**
```bash
pip install torch torchvision pillow numpy tqdm
```

---

## Known Limitations

1. **Color Matching:** While brightness and overall quality are excellent, the model struggles with:
   - Green hue regions (plants) - may lack saturation
   - Blue hue regions (sky) - may lack saturation
   - Per-image color grading variance (GT editing style varies significantly)

2. **Resolution:** Model is trained at 512x512. For higher resolutions, consider:
   - Tiling/patching approaches
   - Retraining at higher resolution
   - Using the model as a post-processing step

3. **Post-Processing:** Fixed post-processing rules (e.g., uniform saturation boost) do NOT work due to massive per-image variance in GT editing style. See `ANALYSIS_POST_PROCESSING.md` for details.

---

## Recommended Improvements

For better color matching, consider training with SOTA color losses:
- Focal Frequency Loss (CVPR 2021)
- LAB Perceptual Loss
- Color Curve Learning
- Histogram Matching

See `src/training/sota_color_loss.py` for implementation.

---

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{restormer2022,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Zamir, Syed Waqas and Arora, Aditya and Gupta, Salman and Khan, Fahad Shahbaz and Sun, Jing and Shahbaz Khan, Fahad and Zhu, Fanglin and Shao, Ling and Qi, Guo-Jun and Yang, Ming-Hsuan},
  booktitle={CVPR},
  year={2022}
}
```

---

## Contact

For questions about inference or deployment, contact the ML engineering team.

**Model Trained By:** Top 0.0001% MLE
**Training Date:** December 2025
**Framework:** PyTorch 2.6
