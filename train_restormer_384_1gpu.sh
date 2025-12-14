#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o train_restormer_384_%j.out
#SBATCH -e train_restormer_384_%j.err
#SBATCH -J rest_384

echo "=========================================="
echo "Restormer 384x384 - 1 GPU"
echo "=========================================="
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /mmfs1/home/sww35/autohdr-real-estate-577

if [ -n "$VIRTUAL_ENV" ]; then deactivate 2>/dev/null || true; fi
unset VIRTUAL_ENV
export PATH=/cm/local/apps/python39/bin:$PATH

module load python39
module load cuda11.8/toolkit/11.8.0

# Use explicit python path
PYTHON=/cm/local/apps/python39/bin/python3

# Install dependencies
$PYTHON -m pip install --user Pillow tqdm einops timm --quiet 2>&1 | grep -v "Cache entry"

$PYTHON -c "
import sys
sys.path.insert(0, 'src/training')

import torch
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm

from restormer import create_restormer

class HDRDataset(Dataset):
    def __init__(self, data_root, jsonl_path, split='train', crop_size=384):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.augment = (split == 'train')
        all_samples = []
        with open(self.data_root / jsonl_path) as f:
            for line in f:
                all_samples.append(json.loads(line.strip()))
        n_val = max(1, int(len(all_samples) * 0.1))
        self.samples = all_samples[-n_val:] if split == 'val' else all_samples[:-n_val]

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

device = torch.device('cuda')
print('Creating Restormer model...')
model = create_restormer('base').to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

train_ds = HDRDataset('.', 'train.jsonl', 'train', 384)
val_ds = HDRDataset('.', 'train.jsonl', 'val', 384)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler()

output_dir = Path('outputs_restormer_384')
output_dir.mkdir(exist_ok=True)

best_val = float('inf')
grad_accum = 2

for epoch in range(100):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/100')
    for i, batch in enumerate(pbar):
        src = batch['source'].to(device)
        tar = batch['target'].to(device)

        with autocast():
            out = model(src)
            loss = F.l1_loss(out, tar) / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * grad_accum
        pbar.set_postfix({'loss': f'{loss.item()*grad_accum:.4f}'})

    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            with autocast():
                out = model(batch['source'].to(device))
            val_loss += F.l1_loss(out, batch['target'].to(device)).item()
    val_loss /= len(val_loader)

    is_best = val_loss < best_val
    if is_best:
        best_val = val_loss
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss}, output_dir / 'checkpoint_best.pt')

    print(f'Epoch {epoch+1}: Train={epoch_loss/len(train_loader):.4f}, Val={val_loss:.4f}' + (' (best)' if is_best else ''))

print(f'Done! Best val: {best_val:.4f}')
"

echo "End time: $(date)"
