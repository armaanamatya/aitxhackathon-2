# Brev Training Guide - AutoHDR Real Estate Enhancement

## Overview

Training guide for AutoHDR models on Brev with B300 GPU (288GB VRAM). Full resolution training at 3197x2201.

---

## Hardware Requirements

| GPU | VRAM | Max Resolution | Batch Size |
|-----|------|----------------|------------|
| B300 | 288GB | 3197x2201 | 2-4 |
| H100 | 80GB | 1024x1024 | 4-8 |
| A100 | 80GB | 1024x1024 | 4-8 |

**Target Resolution**: 3197 x 2201 (full resolution training on B300)

---

## Critical Files

### Training Scripts

| Script | Purpose |
|--------|---------|
| `train_restormer_512_combined_loss.py` | Initial Restormer training |
| `finetune_encoder.py` | Encoder finetuning on best weights |
| `train_elite_refiner_combined.py` | Elite Refiner on frozen backbone |

### Model Architecture

| File | Description |
|------|-------------|
| `src/training/restormer.py` | Restormer architecture (25.4M params) |
| `src/models/color_refiner.py` | Elite Color Refiner (0.07M params) |

### Setup

| File | Purpose |
|------|---------|
| `brev.sh` | Brev instance setup script |
| `requirements_gpu.txt` | GPU dependencies |

---

## Data Split (NO DATA LEAKAGE)

```
data_splits/proper_split/
├── test.jsonl   (10 samples)  ← HELD OUT, NEVER seen during training
├── train.jsonl  (511 samples) ← 90% for training
└── val.jsonl    (56 samples)  ← 10% for validation
```

---

## Training Pipeline (3-Stage)

### Stage 1: Initial Training (Full Resolution)

Train Restormer at full 3197x2201 resolution on B300.

```bash
python3 train_restormer_512_combined_loss.py \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_restormer_3197 \
    --resolution 3197 \
    --batch_size 2 \
    --lr 2e-4 \
    --warmup_epochs 5 \
    --patience 15 \
    --epochs 100
```

**Output**: `outputs_restormer_3197/checkpoint_best.pt`

**Memory**: ~60-80GB with batch_size=2

---

### Stage 2: Encoder Finetuning

Load best weights and finetune only the encoder (decoder frozen).

```bash
python3 finetune_encoder.py \
    --checkpoint outputs_restormer_3197/checkpoint_best.pt \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_encoder_finetuned \
    --resolution 3197 \
    --batch_size 1 \
    --lr 5e-5 \
    --finetune_mode encoder \
    --epochs 50 \
    --patience 10
```

**Output**: `outputs_encoder_finetuned/checkpoint_best.pt`

**Why finetune encoder?**
- Encoder learns feature extraction
- Finetuning at full resolution captures fine details
- Frozen decoder prevents catastrophic forgetting

---

### Stage 3: Elite Refiner (Optional)

Train lightweight color refiner on frozen backbone.

```bash
python3 train_elite_refiner_combined.py \
    --backbone_path outputs_encoder_finetuned/checkpoint_best.pt \
    --train_jsonl data_splits/proper_split/train.jsonl \
    --val_jsonl data_splits/proper_split/val.jsonl \
    --output_dir outputs_elite_refiner \
    --resolution 3197 \
    --batch_size 2 \
    --epochs 50
```

---

## Quick Start on Brev

### 1. SSH into Instance

```bash
brev shell <instance-name>
```

### 2. Clone and Setup

```bash
cd ~
git clone https://github.com/armaanamatya/aitxhackathon-2.git
cd aitxhackathon-2

# Run setup (if not auto-run)
chmod +x brev.sh
./brev.sh
```

### 3. Upload Dataset

```bash
# From local machine
scp -r /path/to/images brev-<instance>:~/aitxhackathon-2/
scp -r /path/to/data_splits brev-<instance>:~/aitxhackathon-2/
```

### 4. Start Training (in tmux)

```bash
tmux new -s train

# Stage 1: Initial training
python3 train_restormer_512_combined_loss.py \
    --resolution 3197 \
    --batch_size 2 \
    --epochs 100

# After Stage 1 completes...

# Stage 2: Encoder finetuning
python3 finetune_encoder.py \
    --checkpoint outputs_restormer_3197/checkpoint_best.pt \
    --resolution 3197 \
    --batch_size 1 \
    --epochs 50
```

Detach tmux: `Ctrl+B, D`
Reattach: `tmux attach -t train`

---

## Loss Function

```python
Total Loss = 1.0 × L1 + 0.5 × Window + 0.3 × BrightRegionSaturation
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| L1 | 1.0 | Primary metric (MAE) |
| Window | 0.5 | Extra weight on bright regions |
| Saturation | 0.3 | Color vibrancy in highlights |

---

## Memory Estimates (B300 288GB)

| Resolution | Batch | Training Memory | Finetuning Memory |
|------------|-------|-----------------|-------------------|
| 3197x2201 | 1 | ~50GB | ~35GB |
| 3197x2201 | 2 | ~80GB | ~55GB |
| 3197x2201 | 4 | ~140GB | ~95GB |

B300 has plenty of headroom at full resolution.

---

## Hyperparameters

### Initial Training

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-4 |
| Warmup Epochs | 5 |
| Weight Decay | 0.02 |
| Gradient Clip | 1.0 |
| Early Stopping | 15 epochs |

### Encoder Finetuning

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-5 (lower) |
| Warmup Epochs | 2 |
| Weight Decay | 0.01 |
| Early Stopping | 10 epochs |

---

## Inference (Post-Training)

After training, model can run inference on DGX Spark (128GB) at full resolution.

```python
import torch
from src.training.restormer import create_restormer

model = create_restormer('base')
checkpoint = torch.load('outputs_encoder_finetuned/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference at 3197x2201 uses ~10GB VRAM
```

---

## Troubleshooting

### OOM on B300
- Reduce batch_size to 1
- Enable gradient checkpointing in model

### Slow Training
- Increase num_workers (default 4)
- Ensure data is on fast storage (NVMe)

### Poor Color Reproduction
- Check saturation loss weight
- Verify bright region threshold (0.5)

---

## File Checklist

```bash
# Verify all files exist before training
ls -la train_restormer_512_combined_loss.py
ls -la finetune_encoder.py
ls -la src/training/restormer.py
ls -la data_splits/proper_split/*.jsonl
ls -la brev.sh
```
