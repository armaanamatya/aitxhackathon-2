"""
API routes module.
"""

from .inference import router as inference_router
from .health import router as health_router

__all__ = ["inference_router", "health_router"]

