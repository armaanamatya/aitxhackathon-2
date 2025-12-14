"""
Configuration management for the inference server.
Modular and easy to extend.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class ServerConfig(BaseSettings):
    """Server configuration."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    
    # Model settings
    model_backbone_path: Optional[str] = Field(default=None, description="Path to backbone model checkpoint")
    model_refiner_path: Optional[str] = Field(default=None, description="Path to refiner model checkpoint")
    model_device: str = Field(default="cuda", description="Device to run inference on")
    model_precision: str = Field(default="fp16", description="Model precision (fp16/fp32)")
    model_tile_size: int = Field(default=768, description="Tile size for tiled inference")
    model_overlap: int = Field(default=96, description="Overlap between tiles")
    model_batch_size: int = Field(default=16, description="Batch size for inference")
    
    # Triton settings
    triton_enabled: bool = Field(default=False, description="Enable Triton Inference Server")
    triton_url: str = Field(default="localhost:8001", description="Triton server URL")
    triton_model_name: str = Field(default="restormer", description="Triton model name")
    triton_model_version: str = Field(default="1", description="Triton model version")
    triton_timeout: float = Field(default=30.0, description="Triton request timeout in seconds")
    
    # Performance settings
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    request_timeout: float = Field(default=60.0, description="Request timeout in seconds")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    class Config:
        env_file = Path(__file__).parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""  # No prefix needed, use exact variable names


# Global config instance
config = ServerConfig()

