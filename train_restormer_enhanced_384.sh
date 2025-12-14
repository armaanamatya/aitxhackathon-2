#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o train_restormer_enhanced_384_%j.out
#SBATCH -e train_restormer_enhanced_384_%j.err
#SBATCH -J rest_enh

echo "=========================================="
echo "Restormer 384 - Enhanced Multi-Loss"
echo "=========================================="
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /mmfs1/home/sww35/autohdr-real-estate-577

if [ -n "$VIRTUAL_ENV" ]; then deactivate 2>/dev/null || true; fi
unset VIRTUAL_ENV
export PATH=/cm/local/apps/python39/bin:$PATH

module load python39
module load cuda11.8/toolkit/11.8.0

PYTHON=/cm/local/apps/python39/bin/python3
$PYTHON -m pip install --user Pillow tqdm einops timm torchvision --quiet 2>&1 | grep -v "Cache entry"

$PYTHON -c "
import sys
sys.path.insert(0, 'src/training')

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.models as models

from restormer import create_restormer

# ============= Enhanced Loss Functions =============

class VGGPerceptualLoss(nn.Module):
    \"\"\"Perceptual loss using VGG19 features\"\"\"
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slices = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x, y):
        loss = 0.0
        for slice in self.slices:
            x = slice(x)
            y = slice(y)
            loss += F.l1_loss(x, y)
        return loss

class EdgeLoss(nn.Module):
    \"\"\"Edge-preserving loss using Sobel filters\"\"\"
    def __init__(self):
        super().__init__()
        # Sobel filters
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

    def forward(self, x, y):
        device = x.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        # Convert to grayscale
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        y_gray = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]

        # Compute edges
        x_edge_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        x_edge_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        y_edge_x = F.conv2d(y_gray, self.sobel_x, padding=1)
        y_edge_y = F.conv2d(y_gray, self.sobel_y, padding=1)

        x_edge = torch.sqrt(x_edge_x ** 2 + x_edge_y ** 2 + 1e-6)
        y_edge = torch.sqrt(y_edge_x ** 2 + y_edge_y ** 2 + 1e-6)

        return F.l1_loss(x_edge, y_edge)

class SSIMLoss(nn.Module):
    \"\"\"SSIM loss (1 - SSIM)\"\"\"
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, x, y):
        if self.window.device != x.device:
            self.window = self.window.to(x.device)

        mu1 = F.conv2d(x, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(y, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(x*x, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(y*y, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(x*y, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

# ============= Dataset =============

class HDRDataset(Dataset):
    def __init__(self, data_root, jsonl_path, split='train', crop_size=384, test_jsonl='test.jsonl'):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.augment = (split == 'train')

        all_samples = []
        with open(self.data_root / jsonl_path) as f:
            for line in f:
                all_samples.append(json.loads(line.strip()))

        test_samples = set()
        if (self.data_root / test_jsonl).exists():
            with open(self.data_root / test_jsonl) as f:
                for line in f:
                    item = json.loads(line.strip())
                    test_samples.add(item['src'])

        filtered_samples = [s for s in all_samples if s['src'] not in test_samples]
        print(f'Total: {len(all_samples)}, Excluded test: {len(test_samples)}, Remaining: {len(filtered_samples)}')

        n_val = max(1, int(len(filtered_samples) * 0.1))
        if split == 'val':
            self.samples = filtered_samples[-n_val:]
        else:
            self.samples = filtered_samples[:-n_val]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        src = Image.open(self.data_root / item['src']).convert('RGB')
        tar = Image.open(self.data_root / item['tar']).convert('RGB')
        src = src.resize((self.crop_size, self.crop_size), Image.LANCZOS)
        tar = tar.resize((self.crop_size, self.crop_size), Image.LANCZOS)
        src = torch.from_numpy(np.array(src)).permute(2,0,1).float() / 255.0
        tar = torch.from_numpy(np.array(tar)).permute(2,0,1).float() / 255.0
        if self.augment and random.random() > 0.5:
            src, tar = torch.flip(src, [2]), torch.flip(tar, [2])
        return {'source': src, 'target': tar}

# ============= Training Setup =============

device = torch.device('cuda')
print('Creating Restormer model...')
model = create_restormer('base').to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# Initialize loss functions
print('\\nInitializing enhanced loss functions...')
vgg_loss = VGGPerceptualLoss().to(device)
edge_loss = EdgeLoss().to(device)
ssim_loss = SSIMLoss().to(device)
print('Loss functions ready: L1 + VGG + Edge + SSIM')

train_ds = HDRDataset('.', 'train.jsonl', 'train', 384)
val_ds = HDRDataset('.', 'train.jsonl', 'val', 384)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# Optimized hyperparameters based on previous runs
base_lr = 1e-4
warmup_epochs = 5
total_epochs = 100

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.02)
scaler = GradScaler()

output_dir = Path('outputs_restormer_enhanced_384')
output_dir.mkdir(exist_ok=True)

best_val = float('inf')
grad_accum = 2

# Loss weights (tuned for quality)
w_l1 = 1.0
w_vgg = 0.1
w_edge = 0.05
w_ssim = 0.1

print(f'\\nLoss weights: L1={w_l1}, VGG={w_vgg}, Edge={w_edge}, SSIM={w_ssim}')

def get_lr(epoch):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

def compute_loss(output, target):
    \"\"\"Compute combined loss\"\"\"
    loss_l1 = F.l1_loss(output, target)
    loss_vgg = vgg_loss(output, target)
    loss_edge = edge_loss(output, target)
    loss_ssim = ssim_loss(output, target)

    total = w_l1 * loss_l1 + w_vgg * loss_vgg + w_edge * loss_edge + w_ssim * loss_ssim
    return total, loss_l1, loss_vgg, loss_edge, loss_ssim

print('\\n=== Starting Enhanced Training ===')

for epoch in range(total_epochs):
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    epoch_loss = 0
    epoch_l1 = 0
    epoch_vgg = 0
    epoch_edge = 0
    epoch_ssim = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
    for i, batch in enumerate(pbar):
        src = batch['source'].to(device)
        tar = batch['target'].to(device)

        with autocast():
            out = model(src)
            loss, l1, vgg, edge, ssim = compute_loss(out, tar)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * grad_accum
        epoch_l1 += l1.item()
        epoch_vgg += vgg.item()
        epoch_edge += edge.item()
        epoch_ssim += ssim.item()

        pbar.set_postfix({
            'loss': f'{loss.item()*grad_accum:.4f}',
            'l1': f'{l1.item():.4f}',
            'lr': f'{lr:.2e}'
        })

    # Validation
    model.eval()
    val_loss = 0
    val_l1 = 0
    with torch.no_grad():
        for batch in val_loader:
            with autocast():
                out = model(batch['source'].to(device))
            tar = batch['target'].to(device)
            loss, l1, vgg, edge, ssim = compute_loss(out, tar)
            val_loss += loss.item()
            val_l1 += l1.item()

    val_loss /= len(val_loader)
    val_l1 /= len(val_loader)

    is_best = val_loss < best_val
    if is_best:
        best_val = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_l1': val_l1
        }, output_dir / 'checkpoint_best.pt')

    avg_l1 = epoch_l1 / len(train_loader)
    avg_vgg = epoch_vgg / len(train_loader)
    avg_edge = epoch_edge / len(train_loader)
    avg_ssim = epoch_ssim / len(train_loader)

    print(f'Epoch {epoch+1}: Train={epoch_loss/len(train_loader):.4f} (L1={avg_l1:.4f}, VGG={avg_vgg:.4f}, Edge={avg_edge:.4f}, SSIM={avg_ssim:.4f}), Val={val_loss:.4f} (L1={val_l1:.4f}), LR={lr:.2e}' + (' (best)' if is_best else ''))

print(f'\\nDone! Best val: {best_val:.4f}')
print(f'Model saved to: {output_dir / \"checkpoint_best.pt\"}')
"

echo "End time: $(date)"
