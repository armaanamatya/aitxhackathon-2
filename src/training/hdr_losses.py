"""
HDR-Specific Loss Functions for Real Estate Photo Enhancement

Designed to solve:
1. Window dimming issue - Highlight preservation loss
2. Cracks/artifacts at edges - Gradient loss
3. Multi-scale edge preservation - Laplacian pyramid loss
4. Local structure preservation - Local contrast loss

Usage:
    from hdr_losses import HDRLoss
    criterion = HDRLoss().to(device)
    loss = criterion(pred, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GradientLoss(nn.Module):
    """
    Sobel-based gradient loss for edge preservation.
    Prevents cracks and artifacts at high-contrast boundaries (windows).
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # Repeat for RGB channels
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)

        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return (loss_x + loss_y) / 2


class HighlightLoss(nn.Module):
    """
    Highlight preservation loss for windows and bright regions.
    Applies extra penalty to errors in bright areas.

    This is CRITICAL for real estate HDR - windows should NOT be dimmed.
    """

    def __init__(self, threshold: float = 0.6, weight_bright: float = 2.0):
        """
        Args:
            threshold: Brightness threshold (in [-1, 1] normalized space, so 0.6 = bright)
            weight_bright: Extra weight for bright region errors
        """
        super().__init__()
        self.threshold = threshold
        self.weight_bright = weight_bright

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Create mask for bright regions (windows)
        # Target is in [-1, 1] range, so threshold should account for this
        brightness = target.mean(dim=1, keepdim=True)
        highlight_mask = (brightness > self.threshold).float()

        # Count of bright pixels (avoid division by zero)
        bright_pixels = highlight_mask.sum() + 1e-8
        total_pixels = target.numel() / target.size(1)  # Exclude channel dim

        # If no bright pixels, return 0
        if bright_pixels < 10:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Weighted L1 loss: higher weight for bright regions
        diff = torch.abs(pred - target)

        # Apply highlight mask with extra weight
        weighted_diff = diff * (1.0 + (self.weight_bright - 1.0) * highlight_mask)

        # Compute separate losses
        highlight_loss = (diff * highlight_mask).sum() / bright_pixels

        return highlight_loss


class LaplacianPyramidLoss(nn.Module):
    """
    Multi-scale edge preservation using Laplacian pyramid.
    Preserves edges at multiple resolutions - important for window frames.
    """

    def __init__(self, num_levels: int = 3, sigma: float = 1.0):
        super().__init__()
        self.num_levels = num_levels
        self.sigma = sigma

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Simple box blur as gaussian approximation."""
        return F.avg_pool2d(
            F.pad(x, (1, 1, 1, 1), mode='reflect'),
            kernel_size=3, stride=1
        )

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian (high-frequency detail)."""
        blurred = self._gaussian_blur(x)
        return x - blurred

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        pred_curr = pred
        target_curr = target

        for level in range(self.num_levels):
            # Compute Laplacian at current level
            pred_lap = self._laplacian(pred_curr)
            target_lap = self._laplacian(target_curr)

            # L1 loss on Laplacian, weighted by level
            weight = 1.0 / (2 ** level)
            total_loss += weight * F.l1_loss(pred_lap, target_lap)

            # Downsample for next level
            if level < self.num_levels - 1:
                pred_curr = F.avg_pool2d(pred_curr, 2)
                target_curr = F.avg_pool2d(target_curr, 2)

        return total_loss / self.num_levels


class LocalContrastLoss(nn.Module):
    """
    Preserve local contrast (local standard deviation).
    Important for maintaining window/interior contrast boundary.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Create averaging kernel
        kernel = torch.ones(
            1, 1, self.kernel_size, self.kernel_size,
            device=pred.device, dtype=pred.dtype
        ) / (self.kernel_size ** 2)
        kernel = kernel.repeat(3, 1, 1, 1)

        # Local mean
        pred_mean = F.conv2d(pred, kernel, padding=self.padding, groups=3)
        target_mean = F.conv2d(target, kernel, padding=self.padding, groups=3)

        # Local contrast = deviation from local mean
        pred_contrast = pred - pred_mean
        target_contrast = target - target_mean

        return F.l1_loss(pred_contrast, target_contrast)


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    Measures structural similarity between images.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Create Gaussian window
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2 * sigma**2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        # 2D window
        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert from [-1, 1] to [0, 1] for SSIM calculation
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        channels = pred.size(1)
        window = self.window.repeat(channels, 1, 1, 1)

        mu_pred = F.conv2d(pred, window, padding=self.window_size//2, groups=channels)
        mu_target = F.conv2d(target, window, padding=self.window_size//2, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=self.window_size//2, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=self.window_size//2, groups=channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=self.window_size//2, groups=channels) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        # Return 1 - SSIM as loss (we want to maximize SSIM)
        return 1 - ssim_map.mean()


class HDRLoss(nn.Module):
    """
    Complete HDR Loss for Real Estate Photo Enhancement.

    Combines multiple losses for:
    - Pixel accuracy (L1)
    - Perceptual quality (VGG - external)
    - Edge preservation (Gradient)
    - Window preservation (Highlight)
    - Multi-scale edges (Laplacian)
    - Local structure (Local Contrast)
    - Structural similarity (SSIM)

    Recommended weights for real estate HDR:
        lambda_l1=1.0,
        lambda_gradient=0.15,
        lambda_highlight=0.2,
        lambda_laplacian=0.1,
        lambda_local_contrast=0.05,
        lambda_ssim=0.1
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_gradient: float = 0.15,
        lambda_highlight: float = 0.2,
        lambda_laplacian: float = 0.1,
        lambda_local_contrast: float = 0.05,
        lambda_ssim: float = 0.1,
        highlight_threshold: float = 0.3,  # In [-1,1] space, 0.3 is bright
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_gradient = lambda_gradient
        self.lambda_highlight = lambda_highlight
        self.lambda_laplacian = lambda_laplacian
        self.lambda_local_contrast = lambda_local_contrast
        self.lambda_ssim = lambda_ssim

        # Initialize loss components
        self.gradient_loss = GradientLoss()
        self.highlight_loss = HighlightLoss(threshold=highlight_threshold)
        self.laplacian_loss = LaplacianPyramidLoss(num_levels=3)
        self.local_contrast_loss = LocalContrastLoss(kernel_size=7)
        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute total HDR loss.

        Args:
            pred: Predicted image [B, 3, H, W] in [-1, 1] range
            target: Target image [B, 3, H, W] in [-1, 1] range
            return_components: If True, return dict with individual losses

        Returns:
            Total loss (and optionally individual components)
        """
        losses = {}

        # L1 loss
        losses['l1'] = F.l1_loss(pred, target)

        # Gradient loss (edge preservation)
        losses['gradient'] = self.gradient_loss(pred, target)

        # Highlight loss (window preservation)
        losses['highlight'] = self.highlight_loss(pred, target)

        # Laplacian pyramid loss (multi-scale edges)
        losses['laplacian'] = self.laplacian_loss(pred, target)

        # Local contrast loss
        losses['local_contrast'] = self.local_contrast_loss(pred, target)

        # SSIM loss
        losses['ssim'] = self.ssim_loss(pred, target)

        # Total weighted loss
        total = (
            self.lambda_l1 * losses['l1'] +
            self.lambda_gradient * losses['gradient'] +
            self.lambda_highlight * losses['highlight'] +
            self.lambda_laplacian * losses['laplacian'] +
            self.lambda_local_contrast * losses['local_contrast'] +
            self.lambda_ssim * losses['ssim']
        )

        if return_components:
            return total, losses
        return total


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target."""
    # Convert from [-1, 1] to [0, 1]
    pred = (pred + 1) / 2
    target = (target + 1) / 2

    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')

    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM between prediction and target."""
    ssim_loss = SSIMLoss()
    ssim_loss = ssim_loss.to(pred.device)
    return 1 - ssim_loss(pred, target).item()


# Test the losses
if __name__ == "__main__":
    # Create dummy tensors
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    # Test individual losses
    print("Testing individual losses:")
    print(f"  Gradient Loss: {GradientLoss()(pred, target):.4f}")
    print(f"  Highlight Loss: {HighlightLoss()(pred, target):.4f}")
    print(f"  Laplacian Loss: {LaplacianPyramidLoss()(pred, target):.4f}")
    print(f"  Local Contrast Loss: {LocalContrastLoss()(pred, target):.4f}")
    print(f"  SSIM Loss: {SSIMLoss()(pred, target):.4f}")

    # Test combined HDR loss
    print("\nTesting HDR Loss:")
    hdr_loss = HDRLoss()
    total, components = hdr_loss(pred, target, return_components=True)
    print(f"  Total: {total:.4f}")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")

    print("\nAll tests passed!")
