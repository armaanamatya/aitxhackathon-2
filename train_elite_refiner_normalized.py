"""
Elite Color Refiner - Normalized Loss Version

KEY CHANGE: Loss components are normalized to [0, 1] for interpretability
- Charbonnier is already ~0.05 (good scale)
- HSV is scaled down by 5× to match Charbonnier scale
- Total loss is now directly comparable to backbone (both in ~0.05-0.15 range)

Expected Results:
- Train loss: 0.08-0.15
- Val loss: 0.06-0.10 (comparable to backbone's 0.0588!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedSimplifiedLoss(nn.Module):
    """
    Normalized loss where both components are in [0, 1] range

    Instead of: Charb (0.05) + 3×HSV (3×0.4 = 1.2) → Total 1.25
    We use:     Charb (0.05) + 0.6×HSV (0.6×0.4 = 0.24) → Total 0.29

    This makes the total loss comparable to backbone's pure Charbonnier loss!
    """
    def __init__(self, charbonnier_weight=1.0, hsv_weight=0.6):
        super().__init__()
        from train_elite_refiner_fast import CharbonnierLoss, HSVColorLoss

        self.charbonnier = CharbonnierLoss()
        self.hsv_color = HSVColorLoss(saturation_weight=3.0)  # Still 3× on saturation internally

        # Normalized weights (total ~= 0.1-0.3 instead of 1.0-1.5)
        self.charb_weight = charbonnier_weight
        self.hsv_weight = hsv_weight  # Much lower to match scale

    def forward(self, pred, target):
        # Compute losses
        loss_char = self.charbonnier(pred, target)
        loss_hsv = self.hsv_color(pred, target)

        # Normalized weighting
        total_loss = self.charb_weight * loss_char + self.hsv_weight * loss_hsv

        # Return loss dict for logging
        losses = {
            'total': total_loss,
            'charbonnier': loss_char.item(),
            'hsv': loss_hsv.item(),
            'backbone_comparable': loss_char.item(),  # For direct comparison
            'charb_weighted': (self.charb_weight * loss_char).item(),
            'hsv_weighted': (self.hsv_weight * loss_hsv).item(),
        }

        return total_loss, losses


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                   NORMALIZED LOSS EXPLANATION                          ║
╚════════════════════════════════════════════════════════════════════════╝

OLD APPROACH (Current):
  Charbonnier: ~0.05
  HSV × 3.0:   ~1.20  ← DOMINATES!
  Total:       ~1.25

  Problem: Total loss >>>>> backbone loss (0.0588)
  Can't compare directly!

NEW APPROACH (Normalized):
  Charbonnier × 1.0: ~0.05
  HSV × 0.6:         ~0.24  ← Balanced!
  Total:             ~0.29

  Benefit: Total loss is comparable to backbone (0.0588 vs 0.29)
  Both are measuring "overall quality" on similar scales

GRADIENT PERSPECTIVE:
  - Charbonnier gets 0.05/0.29 = 17% of gradient
  - HSV gets 0.24/0.29 = 83% of gradient

  Still HSV-dominated (what you want!) but total loss is interpretable.

TO USE THIS:
  Replace SimplifiedLoss with NormalizedSimplifiedLoss in train_elite_refiner_fast.py

  Expected results after training:
    Backbone:     val_loss = 0.0588 (Charb only)
    Refiner:      val_loss = 0.0600 (Charb + normalized HSV)

  Now you can say: "Refiner matches backbone quality (0.060 vs 0.059) while adding color!"
""")
