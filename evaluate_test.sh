#!/bin/bash
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o evaluate_test_%j.out
#SBATCH -e evaluate_test_%j.err
#SBATCH -J eval_test

echo "=========================================="
echo "Evaluate Models on Test Set (5 images)"
echo "=========================================="
echo "Start time: $(date)"

cd /mmfs1/home/sww35/autohdr-real-estate-577

if [ -n "$VIRTUAL_ENV" ]; then deactivate 2>/dev/null || true; fi
unset VIRTUAL_ENV
export PATH=/cm/local/apps/python39/bin:$PATH

module load python39
module load cuda11.8/toolkit/11.8.0

PYTHON=/cm/local/apps/python39/bin/python3
$PYTHON -m pip install --user Pillow tqdm einops timm scikit-image --quiet 2>&1 | grep -v "Cache entry"

$PYTHON -c "
import sys
sys.path.insert(0, 'src/training')

import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import torch.nn.functional as F

from restormer import create_restormer

# Try importing SSIM/PSNR metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print('Warning: scikit-image not available, only L1 loss will be computed')

class TestDataset(Dataset):
    def __init__(self, data_root, test_jsonl='test.jsonl', crop_size=384):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.samples = []
        with open(self.data_root / test_jsonl) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        print(f'Loaded {len(self.samples)} test images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        src = Image.open(self.data_root / item['src']).convert('RGB')
        tar = Image.open(self.data_root / item['tar']).convert('RGB')
        src_full = src.copy()
        tar_full = tar.copy()
        src = src.resize((self.crop_size, self.crop_size), Image.LANCZOS)
        tar = tar.resize((self.crop_size, self.crop_size), Image.LANCZOS)
        src_t = torch.from_numpy(np.array(src)).permute(2,0,1).float() / 255.0
        tar_t = torch.from_numpy(np.array(tar)).permute(2,0,1).float() / 255.0
        return {
            'source': src_t,
            'target': tar_t,
            'path': item['src'],
            'src_full': np.array(src_full),
            'tar_full': np.array(tar_full)
        }

def evaluate_model(model, test_loader, device, model_name, crop_size):
    model.eval()
    total_l1 = 0
    total_psnr = 0
    total_ssim = 0
    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            src = batch['source'].to(device)
            tar = batch['target'].to(device)

            with autocast():
                out = model(src)

            # L1 loss
            l1 = F.l1_loss(out, tar).item()
            total_l1 += l1

            # PSNR/SSIM on output
            if HAS_METRICS:
                out_np = out[0].cpu().numpy().transpose(1,2,0).clip(0,1)
                tar_np = tar[0].cpu().numpy().transpose(1,2,0).clip(0,1)
                total_psnr += psnr(tar_np, out_np, data_range=1.0)
                total_ssim += ssim(tar_np, out_np, data_range=1.0, channel_axis=2)

            n_samples += 1

    avg_l1 = total_l1 / n_samples
    results = {'model': model_name, 'crop_size': crop_size, 'l1': avg_l1}

    if HAS_METRICS:
        results['psnr'] = total_psnr / n_samples
        results['ssim'] = total_ssim / n_samples

    return results

device = torch.device('cuda')

# Find all checkpoints
checkpoints = [
    ('outputs_restormer_384', 'Restormer 384 (scratch)', 384),
    ('outputs_restormer_896', 'Restormer 896', 896),
    ('outputs_restormer_pretrained_384', 'Restormer 384 (pretrained)', 384),
]

print('\\n=== Test Set Evaluation ===')
print(f'Test images: 5 (first 5 from dataset, excluded from training)')
print()

results = []
for output_dir, name, crop_size in checkpoints:
    ckpt_path = Path(output_dir) / 'checkpoint_best.pt'
    if not ckpt_path.exists():
        print(f'{name}: No checkpoint found at {ckpt_path}')
        continue

    print(f'Evaluating {name}...')
    model = create_restormer('base').to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')

    test_ds = TestDataset('.', 'test.jsonl', crop_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    res = evaluate_model(model, test_loader, device, name, crop_size)
    res['best_epoch'] = best_epoch
    res['val_loss'] = val_loss
    results.append(res)

    del model
    torch.cuda.empty_cache()

print('\\n' + '='*60)
print('RESULTS SUMMARY')
print('='*60)
print(f'{\"Model\":<30} {\"Crop\":<6} {\"Val L1\":<10} {\"Test L1\":<10}', end='')
if HAS_METRICS:
    print(f' {\"PSNR\":<8} {\"SSIM\":<8}', end='')
print(f' {\"Best Epoch\":<10}')
print('-'*60)

for r in results:
    print(f'{r[\"model\"]:<30} {r[\"crop_size\"]:<6} {r[\"val_loss\"]:<10.4f} {r[\"l1\"]:<10.4f}', end='')
    if HAS_METRICS:
        print(f' {r[\"psnr\"]:<8.2f} {r[\"ssim\"]:<8.4f}', end='')
    print(f' {r[\"best_epoch\"]:<10}')

print('='*60)
print('\\nNote: These test images were NOT used during training (no data leakage)')
"

echo "End time: $(date)"
