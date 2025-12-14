# Fix PyTorch CUDA Issue

If you're getting this error:
```
OSError: [WinError 127] The specified procedure could not be found. Error loading "...\c10_cuda.dll"
```

## Quick Fix: Use CPU Mode

The server will now start even if PyTorch CUDA has issues. To force CPU mode:

```bash
export MODEL_DEVICE=cpu
python run_api_server.py
```

Or in Windows PowerShell:
```powershell
$env:MODEL_DEVICE="cpu"
python run_api_server.py
```

## Proper Fix: Reinstall PyTorch

1. **Uninstall current PyTorch:**
```bash
pip uninstall torch torchvision
```

2. **Install CPU-only version (if you don't have CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

3. **Or install CUDA version (if you have CUDA installed):**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Check Your Setup

```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"
```

If CUDA is not available but you have a GPU, you may need to:
1. Install CUDA Toolkit from NVIDIA
2. Install matching cuDNN
3. Reinstall PyTorch with correct CUDA version

## Server Will Still Work

The server is designed to start even if PyTorch has issues. It will:
- Start successfully
- Show health endpoints
- Only fail when you try to use inference endpoints (if model isn't loaded)

You can still test the API structure at http://localhost:8000/docs

