"""
Inference endpoints.
"""

import logging
import time
import base64
import io
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from api_server.schemas import InferenceRequest, InferenceResponse, BatchInferenceRequest, BatchInferenceResponse
from api_server.app import get_model_manager, get_triton_client
from api_server.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["inference"])


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")


def encode_image(image: Image.Image) -> str:
    """Encode image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


@router.post("/", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Single image inference endpoint.
    
    Accepts either base64 encoded image or image URL.
    """
    model_manager = get_model_manager()
    triton_client = get_triton_client()
    
    if model_manager is None and triton_client is None:
        raise HTTPException(
            status_code=503,
            detail="No model available. Please load a model first."
        )
    
    # Load image
    if request.image_base64:
        image = decode_image(request.image_base64)
    elif request.image_url:
        # TODO: Implement URL image loading
        raise HTTPException(status_code=400, detail="URL image loading not yet implemented")
    else:
        raise HTTPException(status_code=400, detail="Either image_base64 or image_url must be provided")
    
    # Run inference
    start_time = time.perf_counter()
    
    try:
        if triton_client and config.triton_enabled:
            output_image = triton_client.infer(image)
        else:
            output_image = model_manager.infer(image, use_refiner=request.use_refiner)
        
        inference_time = time.perf_counter() - start_time
        
        # Encode output
        output_base64 = encode_image(output_image)
        
        return InferenceResponse(
            success=True,
            message="Inference completed successfully",
            output_base64=output_base64,
            inference_time=inference_time
        )
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/upload")
async def inference_upload(file: UploadFile = File(...), use_refiner: bool = True):
    """
    Inference endpoint that accepts file upload.
    
    More convenient for testing with tools like curl or Postman.
    """
    model_manager = get_model_manager()
    triton_client = get_triton_client()
    
    if model_manager is None and triton_client is None:
        raise HTTPException(
            status_code=503,
            detail="No model available. Please load a model first."
        )
    
    # Read and decode image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")
    
    # Run inference
    start_time = time.perf_counter()
    
    try:
        if triton_client and config.triton_enabled:
            output_image = triton_client.infer(image)
        else:
            output_image = model_manager.infer(image, use_refiner=use_refiner)
        
        inference_time = time.perf_counter() - start_time
        
        # Encode output
        output_base64 = encode_image(output_image)
        
        return InferenceResponse(
            success=True,
            message="Inference completed successfully",
            output_base64=output_base64,
            inference_time=inference_time
        )
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/batch", response_model=BatchInferenceResponse)
async def batch_inference(request: BatchInferenceRequest):
    """
    Batch inference endpoint.
    
    Processes multiple images in a single request.
    """
    model_manager = get_model_manager()
    triton_client = get_triton_client()
    
    if model_manager is None and triton_client is None:
        raise HTTPException(
            status_code=503,
            detail="No model available. Please load a model first."
        )
    
    if not request.images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(request.images) > config.max_concurrent_requests:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images. Maximum is {config.max_concurrent_requests}"
        )
    
    # Process images
    start_time = time.perf_counter()
    results = []
    
    for i, image_data in enumerate(request.images):
        try:
            image = decode_image(image_data)
            
            if triton_client and config.triton_enabled:
                output_image = triton_client.infer(image)
            else:
                output_image = model_manager.infer(image, use_refiner=request.use_refiner)
            
            output_base64 = encode_image(output_image)
            
            results.append({
                "index": i,
                "success": True,
                "output_base64": output_base64
            })
        
        except Exception as e:
            logger.error(f"Failed to process image {i}: {e}")
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    total_time = time.perf_counter() - start_time
    
    return BatchInferenceResponse(
        success=True,
        results=results,
        total_time=total_time
    )

