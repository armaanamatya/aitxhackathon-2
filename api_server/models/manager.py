"""
Model Manager - Handles model loading and inference.
Modular design for easy extension.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn
import logging
from PIL import Image
import numpy as np

# Import torch with error handling
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None
    F = None

logger = logging.getLogger(__name__)

# Lazy imports - only import when actually needed
def _import_model_modules():
    """Lazy import of model modules to avoid import errors at startup."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    
    # Add src to path
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from training.restormer import create_restormer
    from models.color_refiner import create_elite_color_refiner
    
    return create_restormer, create_elite_color_refiner


class ModelManager:
    """
    Manages model loading, caching, and inference.
    Supports both direct PyTorch inference and Triton integration.
    """
    
    def __init__(
        self,
        backbone_path: Optional[str] = None,
        refiner_path: Optional[str] = None,
        device: str = "cuda",
        precision: str = "fp16",
        use_compile: bool = True
    ):
        """
        Initialize model manager.
        
        Args:
            backbone_path: Path to backbone model checkpoint
            refiner_path: Path to refiner model checkpoint
            device: Device to run on
            precision: Model precision (fp16/fp32)
            use_compile: Whether to use torch.compile
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Please install it first.")
        
        # Check CUDA availability and fallback to CPU if needed
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
            logging.warning("CUDA check failed, falling back to CPU")
        
        if device == "cuda" and not cuda_available:
            logging.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = torch.device(device)
        self.precision = precision
        self.use_fp16 = precision == "fp16" and self.device.type == "cuda"
        self.use_compile = use_compile
        
        self.backbone = None
        self.refiner = None
        
        if backbone_path:
            self.load_backbone(backbone_path)
        
        if refiner_path:
            self.load_refiner(refiner_path)
    
    def load_backbone(self, checkpoint_path: str) -> None:
        """Load backbone model from checkpoint."""
        logger.info(f"Loading backbone from {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
        
        # Lazy import model modules
        create_restormer, _ = _import_model_modules()
        
        # Create model
        self.backbone = create_restormer('base').to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.eval()
        
        # Convert to FP16 if needed
        if self.use_fp16:
            self.backbone = self.backbone.half()
        
        # Compile if requested
        if self.use_compile:
            try:
                self.backbone = torch.compile(
                    self.backbone,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                logger.info("Backbone compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Warmup
        self._warmup_model(self.backbone)
        logger.info("Backbone loaded successfully")
    
    def load_refiner(self, checkpoint_path: str) -> None:
        """Load refiner model from checkpoint."""
        logger.info(f"Loading refiner from {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Refiner checkpoint not found: {checkpoint_path}")
        
        # Lazy import model modules
        _, create_elite_color_refiner = _import_model_modules()
        
        # Create model
        self.refiner = create_elite_color_refiner(size='medium').to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'refiner_state_dict' in checkpoint:
            state_dict = checkpoint['refiner_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.refiner.load_state_dict(state_dict, strict=False)
        self.refiner.eval()
        
        # Convert to FP16 if needed
        if self.use_fp16:
            self.refiner = self.refiner.half()
        
        # Compile if requested
        if self.use_compile:
            try:
                self.refiner = torch.compile(
                    self.refiner,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                logger.info("Refiner compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Warmup
        self._warmup_model(self.refiner)
        logger.info("Refiner loaded successfully")
    
    def _warmup_model(self, model: Any, num_runs: int = 3) -> None:
        """Warmup model for accurate timing."""
        if not TORCH_AVAILABLE:
            return
        
        dummy = torch.randn(1, 3, 512, 512).to(self.device)
        if self.use_fp16:
            dummy = dummy.half()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def infer(
        self,
        image: Image.Image,
        use_refiner: bool = True
    ) -> Image.Image:
        """
        Run inference on a single image.
        
        Args:
            image: Input PIL Image
            use_refiner: Whether to use refiner if available
        
        Returns:
            Enhanced PIL Image
        """
        if self.backbone is None:
            raise RuntimeError("Backbone model not loaded")
        
        # Use torch.no_grad context if available
        if TORCH_AVAILABLE:
            with torch.no_grad():
                # Preprocess
                tensor = self._preprocess(image)
                
                # Run backbone
                output = self.backbone(tensor)
                
                # Run refiner if available and requested
                if use_refiner and self.refiner is not None:
                    output = self.refiner(output, tensor)
        else:
            # Preprocess
            tensor = self._preprocess(image)
            
            # Run backbone
            output = self.backbone(tensor)
            
            # Run refiner if available and requested
            if use_refiner and self.refiner is not None:
                output = self.refiner(output, tensor)
        
        # Postprocess
        output_image = self._postprocess(output, image.size)
        
        return output_image
    
    def _preprocess(self, image: Image.Image) -> Any:
        """Preprocess image for model input."""
        # Resize to model input size (or keep original for tiled inference)
        # For now, we'll resize to 512x512
        image = image.resize((512, 512), Image.LANCZOS)
        
        # Convert to tensor
        tensor = torch.from_numpy(np.array(image)).float()
        tensor = tensor.permute(2, 0, 1) / 255.0  # [H, W, C] -> [C, H, W], [0, 255] -> [0, 1]
        tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        if self.use_fp16:
            tensor = tensor.half()
        
        return tensor
    
    def _postprocess(
        self,
        tensor: Any,
        original_size: Tuple[int, int]
    ) -> Image.Image:
        """Postprocess model output to PIL Image."""
        # Convert to numpy
        tensor = tensor.squeeze(0).float().cpu()
        tensor = torch.clamp(tensor, 0, 1)
        tensor = tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        tensor = (tensor * 255).astype(np.uint8)
        
        # Create PIL Image
        image = Image.fromarray(tensor)
        
        # Resize to original size
        image = image.resize(original_size, Image.LANCZOS)
        
        return image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "backbone_loaded": self.backbone is not None,
            "refiner_loaded": self.refiner is not None,
            "device": str(self.device),
            "precision": self.precision,
            "compiled": self.use_compile
        }
        
        if self.backbone:
            info["backbone_params"] = sum(p.numel() for p in self.backbone.parameters())
        
        if self.refiner:
            info["refiner_params"] = sum(p.numel() for p in self.refiner.parameters())
        
        return info

