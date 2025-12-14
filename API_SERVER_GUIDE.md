# FastAPI Inference Server Guide

## Overview

A modular, scalable FastAPI inference server for Real Estate HDR Enhancement, compatible with:
- **NVIDIA Triton Inference Server**
- **CUDA 13.0**
- **Python 3.10**

## Architecture

The server is designed with modularity and scalability in mind:

```
api_server/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── app.py               # Main FastAPI application
├── main.py              # Entry point
├── schemas.py           # Pydantic request/response models
├── models/              # Model management
│   ├── __init__.py
│   ├── manager.py       # PyTorch model manager
│   └── triton_client.py # Triton client
└── routes/              # API routes
    ├── __init__.py
    ├── health.py        # Health check endpoints
    └── inference.py     # Inference endpoints
```

## Installation

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements_api_server.txt

# Install PyTorch (compatible with CUDA 13.0)
# Note: You may need to install from source or use nightly builds
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Triton Client (Optional)

If using Triton Inference Server:

```bash
pip install tritonclient[http]==2.41.0
```

## Configuration

Configuration is managed via environment variables or a `.env` file:

```bash
# Server settings
HOST=0.0.0.0
PORT=8000
WORKERS=1
RELOAD=false

# Model settings
MODEL_BACKBONE_PATH=checkpoints/restormer_base.pt
MODEL_REFINER_PATH=checkpoints/refiner.pt
MODEL_DEVICE=cuda
MODEL_PRECISION=fp16
MODEL_TILE_SIZE=768
MODEL_OVERLAP=96
MODEL_BATCH_SIZE=16

# Triton settings (optional)
TRITON_ENABLED=false
TRITON_URL=localhost:8001
TRITON_MODEL_NAME=restormer
TRITON_MODEL_VERSION=1
TRITON_TIMEOUT=30.0

# Performance settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=60.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/server.log
```

## Usage

### Basic Usage

```bash
# Using the launcher script
./run_api_server.sh \
    checkpoints/restormer_base.pt \
    checkpoints/refiner.pt \
    0.0.0.0 \
    8000

# Or directly with Python
python3 -m api_server.main
```

### With Environment Variables

```bash
export MODEL_BACKBONE_PATH=checkpoints/restormer_base.pt
export MODEL_REFINER_PATH=checkpoints/refiner.pt
python3 -m api_server.main
```

### With Triton Inference Server

```bash
export TRITON_ENABLED=true
export TRITON_URL=localhost:8001
export TRITON_MODEL_NAME=restormer
python3 -m api_server.main
```

## API Endpoints

### Health Check

```bash
# Check server health
curl http://localhost:8000/health/

# Get model information
curl http://localhost:8000/health/model
```

### Single Image Inference

**Using JSON (base64):**
```bash
curl -X POST http://localhost:8000/inference/ \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "...",
    "use_refiner": true,
    "preserve_resolution": true
  }'
```

**Using file upload:**
```bash
curl -X POST http://localhost:8000/inference/upload \
  -F "file=@image.jpg" \
  -F "use_refiner=true"
```

### Batch Inference

```bash
curl -X POST http://localhost:8000/inference/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["base64_image_1", "base64_image_2"],
    "use_refiner": true,
    "preserve_resolution": true
  }'
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Triton Integration

### Setting Up Triton

1. **Export Model to ONNX:**
```python
python3 src/optimization/tensorrt_optimize.py \
    --model_path checkpoints/restormer_base.pt \
    --output_dir outputs/optimized \
    --method onnx
```

2. **Deploy to Triton:**
   - Create a model repository structure
   - Configure `config.pbtxt` for your model
   - Start Triton server

3. **Enable in API Server:**
```bash
export TRITON_ENABLED=true
export TRITON_URL=localhost:8001
export TRITON_MODEL_NAME=restormer
```

## Scalability

### Horizontal Scaling

The server can be scaled horizontally using:
- **Multiple workers**: Set `WORKERS` environment variable
- **Load balancer**: Use nginx or similar in front of multiple instances
- **Kubernetes**: Deploy multiple pods with auto-scaling

### Vertical Scaling

- Increase `MODEL_BATCH_SIZE` for larger batches
- Use larger GPUs with more memory
- Enable TensorRT for faster inference

### Async Processing

For high-throughput scenarios, consider:
- Using a task queue (Celery, RQ) for async processing
- Implementing WebSocket for streaming results
- Using background tasks with FastAPI

## Monitoring

### Health Checks

The `/health/` endpoint provides:
- Server status
- Model loading status
- Triton connection status

### Logging

Logs are written to:
- Console (default)
- File (if `LOG_FILE` is set)

### Metrics

Consider adding:
- Prometheus metrics endpoint
- Request timing middleware
- Model inference metrics

## Error Handling

The server includes:
- Global exception handler
- Request validation
- Model availability checks
- Graceful error responses

## Development

### Running in Development Mode

```bash
export RELOAD=true
export LOG_LEVEL=DEBUG
python3 -m api_server.main
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn api_server.app:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_api_server.txt .
RUN pip install -r requirements_api_server.txt

COPY api_server/ ./api_server/
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/

CMD ["python", "-m", "api_server.main"]
```

### Using Kubernetes

See `k8s/` directory for example deployments.

## Troubleshooting

### Model Not Loading

- Check model path is correct
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check model checkpoint format

### Triton Connection Failed

- Verify Triton server is running: `curl http://localhost:8001/v2/health/ready`
- Check Triton URL and model name
- Verify model is deployed in Triton

### Out of Memory

- Reduce `MODEL_BATCH_SIZE`
- Use FP16 precision
- Reduce number of workers

## Extending the Server

### Adding New Endpoints

1. Create route in `api_server/routes/`
2. Add schema in `api_server/schemas.py`
3. Register router in `api_server/app.py`

### Adding New Models

1. Extend `ModelManager` in `api_server/models/manager.py`
2. Add model loading logic
3. Update configuration

### Custom Preprocessing

Modify `ModelManager._preprocess()` or `TritonClient._preprocess()`

## License

See main project license.

