"""
Pydantic schemas for request/response validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Server status")
    version: str = Field(description="API version")
    model_loaded: bool = Field(description="Whether model is loaded")
    triton_enabled: bool = Field(description="Whether Triton is enabled")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    backbone_loaded: bool
    refiner_loaded: bool
    device: str
    precision: str
    compiled: bool
    backbone_params: Optional[int] = None
    refiner_params: Optional[int] = None
    triton_info: Optional[dict] = None


class InferenceRequest(BaseModel):
    """Inference request schema."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    use_refiner: bool = Field(True, description="Whether to use refiner")
    preserve_resolution: bool = Field(True, description="Preserve original resolution")


class InferenceResponse(BaseModel):
    """Inference response schema."""
    success: bool = Field(description="Whether inference succeeded")
    message: Optional[str] = Field(None, description="Status message")
    output_base64: Optional[str] = Field(None, description="Base64 encoded output image")
    inference_time: Optional[float] = Field(None, description="Inference time in seconds")


class BatchInferenceRequest(BaseModel):
    """Batch inference request schema."""
    images: list[str] = Field(description="List of base64 encoded images")
    use_refiner: bool = Field(True, description="Whether to use refiner")
    preserve_resolution: bool = Field(True, description="Preserve original resolution")


class BatchInferenceResponse(BaseModel):
    """Batch inference response schema."""
    success: bool
    results: list[dict] = Field(description="List of inference results")
    total_time: float = Field(description="Total processing time in seconds")

