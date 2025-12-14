# How to Run the FastAPI Inference Server

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements_api_server.txt
```

If you don't have the requirements file yet, install manually:
```bash
pip install fastapi uvicorn[standard] pydantic pydantic-settings python-multipart Pillow numpy
```

### Step 2: Run the Server

**Option A: Using the launcher script (Easiest)**
```bash
python run_api_server.py
```

**Option B: With custom model paths**
```bash
python run_api_server.py checkpoints/restormer_base.pt checkpoints/refiner.pt 0.0.0.0 8000
```

**Option C: Direct Python module**
```bash
python -m api_server.main
```

**Option D: Using environment variables**
```bash
export MODEL_BACKBONE_PATH=checkpoints/restormer_base.pt
export MODEL_REFINER_PATH=checkpoints/refiner.pt
python -m api_server.main
```

### Step 3: Test the Server

Open in browser: http://localhost:8000/docs

Or test with curl:
```bash
curl http://localhost:8000/health/
```

## Detailed Instructions

### Prerequisites Check

1. **Check Python version** (needs 3.10+):
```bash
python --version
```

2. **Check if dependencies are installed**:
```bash
python -c "import fastapi, uvicorn; print('Dependencies OK')"
```

3. **Check if model checkpoints exist** (optional - server will start without them):
```bash
ls checkpoints/*.pt
```

### Running on Windows

```powershell
# In PowerShell or Command Prompt
python run_api_server.py
```

### Running on Linux/Mac

```bash
# Make sure you're in the project directory
cd /path/to/aitxhackathon-2

# Run the server
python3 run_api_server.py
```

### Running with Custom Configuration

Create a `.env` file in `api_server/` directory:

```bash
# api_server/.env
HOST=0.0.0.0
PORT=8000
MODEL_BACKBONE_PATH=checkpoints/restormer_base.pt
MODEL_REFINER_PATH=checkpoints/refiner.pt
MODEL_DEVICE=cuda
MODEL_PRECISION=fp16
LOG_LEVEL=INFO
```

Then run:
```bash
python -m api_server.main
```

## Troubleshooting

### Error: "No module named 'api_server'"

**Solution**: Make sure you're in the project root directory:
```bash
cd /path/to/aitxhackathon-2
python -m api_server.main
```

### Error: "ModuleNotFoundError: No module named 'fastapi'"

**Solution**: Install dependencies:
```bash
pip install -r requirements_api_server.txt
```

### Error: "Model not found"

**Solution**: The server will start without models, but inference won't work. Either:
1. Provide correct model paths: `python run_api_server.py path/to/model.pt`
2. Or set environment variable: `export MODEL_BACKBONE_PATH=path/to/model.pt`

### Error: "Port already in use"

**Solution**: Use a different port:
```bash
python run_api_server.py checkpoints/model.pt checkpoints/refiner.pt 0.0.0.0 8001
```

### Error: "CUDA not available"

**Solution**: The server will fall back to CPU. For GPU:
1. Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Set device to CPU: `export MODEL_DEVICE=cpu`

## Testing the Server

### 1. Health Check
```bash
curl http://localhost:8000/health/
```

### 2. Model Info
```bash
curl http://localhost:8000/health/model
```

### 3. Test Inference (with Python)
```python
import requests
import base64

# Read image
with open("test_image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/inference/",
    json={"image_base64": img_base64, "use_refiner": True}
)

print(response.json())
```

### 4. View API Documentation
Open in browser: http://localhost:8000/docs

## Common Use Cases

### Development Mode (with auto-reload)
```bash
export RELOAD=true
python -m api_server.main
```

### Production Mode (multiple workers)
```bash
export WORKERS=4
python -m api_server.main
```

### With Triton Inference Server
```bash
export TRITON_ENABLED=true
export TRITON_URL=localhost:8001
export TRITON_MODEL_NAME=restormer
python -m api_server.main
```

## Next Steps

Once the server is running:
1. Visit http://localhost:8000/docs for interactive API documentation
2. Test endpoints using the Swagger UI
3. Integrate with your application using the API endpoints

For more details, see `API_SERVER_GUIDE.md`

