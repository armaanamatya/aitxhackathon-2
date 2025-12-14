"""
Health check and model info endpoints.
"""

import logging
from fastapi import APIRouter
from api_server.schemas import HealthResponse, ModelInfoResponse
from api_server.config import config
from api_server.app import get_model_manager, get_triton_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_manager = get_model_manager()
    triton_client = get_triton_client()
    
    model_loaded = model_manager is not None and model_manager.backbone is not None
    triton_enabled = config.triton_enabled and triton_client is not None
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_loaded,
        triton_enabled=triton_enabled
    )


@router.get("/model", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    model_manager = get_model_manager()
    triton_client = get_triton_client()
    
    if model_manager is None:
        return ModelInfoResponse(
            backbone_loaded=False,
            refiner_loaded=False,
            device=config.model_device,
            precision=config.model_precision,
            compiled=False
        )
    
    info = model_manager.get_model_info()
    
    triton_info = None
    if triton_client:
        triton_info = triton_client.get_model_info()
    
    return ModelInfoResponse(
        backbone_loaded=info["backbone_loaded"],
        refiner_loaded=info["refiner_loaded"],
        device=info["device"],
        precision=info["precision"],
        compiled=info["compiled"],
        backbone_params=info.get("backbone_params"),
        refiner_params=info.get("refiner_params"),
        triton_info=triton_info
    )

