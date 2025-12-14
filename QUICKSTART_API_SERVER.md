# Quick Start: FastAPI Inference Server

## Prerequisites

- Python 3.10
- CUDA 13.0 compatible PyTorch
- Model checkpoints (backbone and optionally refiner)

## Installation

```bash
# 1. Install dependencies
pip install -r requirements_api_server.txt

# 2. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. (Optional) Install Triton client
pip install tritonclient[http]==2.41.0
```

## Quick Start

### Option 1: Using the launcher script

```bash
./run_api_server.sh \
    checkpoints/restormer_base.pt \
    checkpoints/refiner.pt \
    0.0.0.0 \
    8000
```

### Option 2: Using environment variables

```bash
export MODEL_BACKBONE_PATH=checkpoints/restormer_base.pt
export MODEL_REFINER_PATH=checkpoints/refiner.pt
python3 -m api_server.main
```

### Option 3: Using .env file

```bash
# Copy example .env file
cp api_server/.env.example api_server/.env

# Edit api_server/.env with your settings
# Then run:
python3 -m api_server.main
```

## Test the Server

### 1. Check health

```bash
curl http://localhost:8000/health/
```

### 2. Test inference with file upload

```bash
curl -X POST http://localhost:8000/inference/upload \
  -F "file=@test_image.jpg" \
  -F "use_refiner=true"
```

### 3. View API documentation

Open in browser: http://localhost:8000/docs

## Example Python Client

```python
import requests
import base64
from PIL import Image
import io

# Load image
image_path = "test_image.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Send request
response = requests.post(
    "http://localhost:8000/inference/",
    json={
        "image_base64": image_base64,
        "use_refiner": True,
        "preserve_resolution": True
    }
)

# Get result
result = response.json()
if result["success"]:
    # Decode output image
    output_bytes = base64.b64decode(result["output_base64"])
    output_image = Image.open(io.BytesIO(output_bytes))
    output_image.save("output.png")
    print(f"Inference time: {result['inference_time']:.2f}s")
```

## With Triton Inference Server

1. **Start Triton server** (separate process)
2. **Set environment variables:**
```bash
export TRITON_ENABLED=true
export TRITON_URL=localhost:8001
export TRITON_MODEL_NAME=restormer
```
3. **Start API server:**
```bash
python3 -m api_server.main
```

## Troubleshooting

### Model not loading
- Check model path is correct
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check checkpoint format

### Port already in use
- Change PORT in .env or environment variable
- Or kill existing process: `lsof -ti:8000 | xargs kill`

### Import errors
- Ensure `src/` directory is in Python path
- Install all dependencies: `pip install -r requirements_api_server.txt`

## Next Steps

- See `API_SERVER_GUIDE.md` for detailed documentation
- Customize configuration in `api_server/config.py`
- Add new endpoints in `api_server/routes/`
- Extend model manager in `api_server/models/manager.py`

