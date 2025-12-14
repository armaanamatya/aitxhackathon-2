#!/usr/bin/env python3
"""
Quick training profiler to find bottlenecks
"""
import time
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.insert(0, str(Path('src/training')))

# Simulate your dataset
class DummyDataset:
    def __init__(self, n=409, res=512):
        self.n = n
        self.res = res

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Simulate image loading + preprocessing
        import cv2
        import numpy as np

        # Dummy images
        src = np.random.randint(0, 255, (self.res, self.res, 3), dtype=np.uint8)
        tar = np.random.randint(0, 255, (self.res, self.res, 3), dtype=np.uint8)

        # To tensor
        src = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0

        return src, tar, f"img_{idx}"

# Test different configurations
configs = [
    {"batch_size": 2, "num_workers": 0, "pin_memory": False},
    {"batch_size": 2, "num_workers": 8, "pin_memory": False},
    {"batch_size": 2, "num_workers": 16, "pin_memory": True},
    {"batch_size": 8, "num_workers": 16, "pin_memory": True},
    {"batch_size": 16, "num_workers": 16, "pin_memory": True},
]

print("="*80)
print("TRAINING SPEED PROFILER")
print("="*80)

dataset = DummyDataset(n=409, res=512)

for i, config in enumerate(configs, 1):
    print(f"\nTest {i}/{len(configs)}: {config}")

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    # Warmup
    for j, batch in enumerate(loader):
        if j >= 2:
            break

    # Timed run
    start = time.time()
    n_batches = 0
    for batch in loader:
        n_batches += 1
        if n_batches >= 20:  # Test 20 batches
            break

    elapsed = time.time() - start
    batches_per_sec = n_batches / elapsed
    samples_per_sec = batches_per_sec * config["batch_size"]

    print(f"  ⏱️  Time: {elapsed:.2f}s for {n_batches} batches")
    print(f"  📊 Speed: {batches_per_sec:.1f} batches/sec, {samples_per_sec:.1f} samples/sec")

    if i == 1:
        baseline = batches_per_sec
    else:
        speedup = batches_per_sec / baseline
        print(f"  🚀 Speedup: {speedup:.2f}x vs baseline")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("Choose config with highest samples/sec that fits in GPU memory")
print("="*80)
