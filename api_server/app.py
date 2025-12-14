"""
Main FastAPI application.
Modular and scalable design.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_server.config import config
from api_server.models.manager import ModelManager
from api_server.models.triton_client import TritonClient
from api_server.routes import inference_router, health_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.log_file if config.log_file else None
)

logger = logging.getLogger(__name__)

# Global model manager and Triton client
_model_manager: Optional[ModelManager] = None
_triton_client: Optional[TritonClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _model_manager, _triton_client
    
    # Startup
    logger.info("Starting FastAPI inference server...")
    logger.info(f"Configuration: device={config.model_device}, precision={config.model_precision}")
    
    # Initialize model manager if backbone path is provided
    if config.model_backbone_path:
        try:
            # Try to determine device - fallback to CPU if CUDA fails
            device = config.model_device
            try:
                import torch
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, using CPU instead")
                    device = "cpu"
            except Exception:
                logger.warning("PyTorch CUDA check failed, using CPU")
                device = "cpu"
            
            _model_manager = ModelManager(
                backbone_path=config.model_backbone_path,
                refiner_path=config.model_refiner_path,
                device=device,
                precision=config.model_precision,
                use_compile=True
            )
            logger.info("Model manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            logger.error("Server will start but inference endpoints won't work")
            _model_manager = None
    
    # Initialize Triton client if enabled
    if config.triton_enabled:
        try:
            _triton_client = TritonClient(
                url=config.triton_url,
                model_name=config.triton_model_name,
                model_version=config.triton_model_version,
                timeout=config.triton_timeout
            )
            logger.info("Triton client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Triton client: {e}")
            _triton_client = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI inference server...")
    _model_manager = None
    _triton_client = None


# Create FastAPI app
app = FastAPI(
    title="Real Estate HDR Enhancement API",
    description="FastAPI inference server for HDR photo enhancement",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(inference_router)


# Global getters for model manager and Triton client
def get_model_manager() -> Optional[ModelManager]:
    """Get the global model manager instance."""
    return _model_manager


def get_triton_client() -> Optional[TritonClient]:
    """Get the global Triton client instance."""
    return _triton_client


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Real Estate HDR Enhancement API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

