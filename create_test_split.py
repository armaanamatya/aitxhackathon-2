#!/usr/bin/env python3
"""
Create test split by taking 10 samples from validation set
"""

import json
import random

# Set seed for reproducibility
random.seed(42)

# Load current validation set
val_samples = []
with open('data_splits/fold_1/val.jsonl', 'r') as f:
    for line in f:
        val_samples.append(json.loads(line.strip()))

print(f"Original validation set: {len(val_samples)} samples")

# Shuffle and split
random.shuffle(val_samples)
test_samples = val_samples[:10]
new_val_samples = val_samples[10:]

print(f"New validation set: {len(new_val_samples)} samples")
print(f"Test set: {len(test_samples)} samples")

# Save new validation set
with open('data_splits/fold_1/val.jsonl', 'w') as f:
    for sample in new_val_samples:
        f.write(json.dumps(sample) + '\n')

# Save test set
with open('data_splits/fold_1/test.jsonl', 'w') as f:
    for sample in test_samples:
        f.write(json.dumps(sample) + '\n')

print("\n✓ Created data_splits/fold_1/test.jsonl")
print("✓ Updated data_splits/fold_1/val.jsonl")

print("\nTest set samples:")
for i, sample in enumerate(test_samples, 1):
    print(f"  {i}. {sample['src']}")
