# Strategy: Fix Window/Outdoor Quality Issues

## Problem
- Windows and outdoor shots (grass, sky) don't match ground truth
- Everything else looks good
- Restormer lacks examples for extreme dynamic range

## Solution: Transfer Learning (FiveK → Real Estate)

### Phase 1: Pre-train on FiveK (5K samples)
**Why this helps:**
- FiveK has diverse outdoor scenes, windows, high DR scenarios
- Learns general local tone mapping
- Better highlight recovery (windows!)
- Better shadow detail (grass, outdoor)

**Settings:**
- Resolution: 512 (fast training)
- Batch size: 20
- Epochs: 30
- LR: 5e-5 (lower for stability)
- **No flip augmentation** (FiveK already diverse)

### Phase 2: Fine-tune on Real Estate (577 samples)
**Settings:**
- Resolution: 896 (high quality)
- Batch size: 8
- Epochs: 100
- LR: 1e-5 (very low - preserve FiveK knowledge)
- **No flip augmentation** (you're right - not useful for RE)
- Early stopping: 15 epochs patience

### Phase 3: Test on Windows/Outdoor
- Compare window quality vs baseline
- Expected improvement: +15-25% on window/outdoor L1

## Timeline
- FiveK preparation: 30 min
- FiveK pre-training: ~4 hours
- Real estate fine-tuning: ~5 hours
- **Total: ~9-10 hours**

## Why This Works
1. **More training data** (5,577 samples vs 577)
2. **Diverse scenarios** (outdoor, windows, high DR)
3. **Transfer learning** (general → specific)
4. **Domain adaptation** (professional retouching → RE enhancement)

## Expected Results
- Better window detail recovery
- Better outdoor/grass quality
- Maintained quality on interior shots
- Overall test L1: **0.041-0.046** (vs current 0.0514)

## Next Steps
1. Run prepare_fivek_robust.py
2. Train on FiveK
3. Fine-tune on real estate
4. Test and compare window quality
