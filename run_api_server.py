#!/usr/bin/env python3
"""
Simple launcher for the FastAPI inference server.
Works on both Windows and Linux.
"""

import os
import sys
from pathlib import Path

def main():
    """Launch the API server with configuration."""
    
    # Default configuration
    backbone_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/restormer_base.pt"
    refiner_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/refiner.pt"
    host = sys.argv[3] if len(sys.argv) > 3 else "0.0.0.0"
    port = sys.argv[4] if len(sys.argv) > 4 else "8000"
    
    print("="*60)
    print("FastAPI Inference Server")
    print("="*60)
    print()
    
    # Set environment variables
    os.environ["MODEL_BACKBONE_PATH"] = backbone_path
    os.environ["MODEL_REFINER_PATH"] = refiner_path
    os.environ["HOST"] = host
    os.environ["PORT"] = port
    os.environ["MODEL_DEVICE"] = "cuda"
    os.environ["MODEL_PRECISION"] = "fp16"
    os.environ["MODEL_BATCH_SIZE"] = "16"
    os.environ["LOG_LEVEL"] = "INFO"
    
    print("Configuration:")
    print(f"  Backbone: {backbone_path}")
    print(f"  Refiner: {refiner_path}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Device: cuda")
    print(f"  Precision: fp16")
    print()
    
    # Check if model exists
    if not Path(backbone_path).exists():
        print(f"Warning: Backbone model not found at {backbone_path}")
        print("Server will start but model won't be loaded")
        print()
    
    # Run server
    print("Starting server...")
    print(f"API docs will be available at: http://{host}:{port}/docs")
    print()
    
    # Import and run
    from api_server.main import app
    import uvicorn
    from api_server.config import config
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=config.reload,
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    main()

