# Inference Optimization for NVIDIA DGX Systems
## SOTA Analysis for Real Estate HDR Model Deployment

**Target Hardware:** NVIDIA DGX (A100, H100, H200)
**Model:** Restormer Base (25.4M params, 512x512)
**Baseline Performance:** ~15-20 FPS (PyTorch eager mode, FP32, batch=1)

**Analysis Level:** Top 0.0001% PhD-level ML Engineer

---

## Table of Contents

1. [Quick Wins (Minimal Effort)](#1-quick-wins-minimal-effort)
2. [TensorRT Optimization](#2-tensorrt-optimization-sota-recommended)
3. [Mixed Precision Inference](#3-mixed-precision-inference)
4. [torch.compile() - PyTorch 2.0+](#4-torchcompile---pytorch-20)
5. [Batching Strategies](#5-batching-strategies)
6. [CUDA Graphs](#6-cuda-graphs)
7. [Multi-GPU Inference](#7-multi-gpu-inference)
8. [NVIDIA Triton Inference Server](#8-nvidia-triton-inference-server)
9. [Model Quantization (INT8)](#9-model-quantization-int8)
10. [Advanced: ONNX + TensorRT](#10-advanced-onnx--tensorrt)
11. [Comparison Matrix](#comparison-matrix)

---

## 1. Quick Wins (Minimal Effort)

### 1.1 Enable TF32 (Tensor Float 32)

**Speedup:** 1.3-1.5x on A100/H100
**Effort:** 1 line of code
**Quality Impact:** Negligible (automatic mixed precision)

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Run inference as normal
model.eval()
with torch.no_grad():
    output = model(input)
```

**Pros:**
- Zero code changes to model
- Free 30-50% speedup on Ampere+ GPUs
- Maintains accuracy (TF32 has same range as FP32)
- NVIDIA default for A100/H100 training

**Cons:**
- Only available on Ampere (A100) and newer (H100, H200)
- Slightly reduced precision vs FP32 (19-bit mantissa vs 24-bit)
- Not beneficial for older GPUs (V100, etc.)

**Recommendation:** ✅ **ALWAYS ENABLE** on DGX A100/H100 systems. Zero downside.

---

### 1.2 Batch Size Optimization

**Speedup:** 2-4x (batch=8 vs batch=1)
**Effort:** Low
**Quality Impact:** None

```python
# Instead of processing images one by one
for img in images:
    output = model(img.unsqueeze(0))  # Slow

# Process in batches
batch_size = 8  # Tune based on GPU memory
for i in range(0, len(images), batch_size):
    batch = torch.stack(images[i:i+batch_size])
    outputs = model(batch)  # Fast
```

**Optimal Batch Sizes (512x512, Restormer 25M):**
- A100 (40GB): batch=16-24
- A100 (80GB): batch=32-48
- H100 (80GB): batch=48-64
- H200 (141GB): batch=80-128

**Pros:**
- Simple to implement
- Excellent GPU utilization
- Amortizes kernel launch overhead
- Linear scaling up to memory limit

**Cons:**
- Requires batching logic (padding for uneven batches)
- Higher latency per image (if real-time required)
- Memory-bound (can't exceed VRAM)

**Recommendation:** ✅ **ALWAYS USE** for offline/batch inference. Critical for throughput.

---

## 2. TensorRT Optimization (SOTA Recommended)

**Speedup:** 3-5x over PyTorch eager mode
**Effort:** Medium
**Quality Impact:** Negligible (<0.1% difference)

TensorRT is NVIDIA's high-performance inference optimizer. It's the **industry standard** for production deployment.

### 2.1 Using torch-tensorrt (Easiest)

```python
import torch
import torch_tensorrt

# Load model
model = create_restormer('base').eval().cuda()
checkpoint = torch.load('model_checkpoint.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 3, 512, 512],  # Fixed shape (or dynamic)
        dtype=torch.float16,      # Use FP16 for speed
    )],
    enabled_precisions={torch.float16},  # FP16 inference
    workspace_size=1 << 30,  # 1GB workspace
)

# Run inference (same API as PyTorch)
with torch.no_grad():
    output = trt_model(input_fp16)

# Save compiled model
torch.jit.save(trt_model, "model_trt.ts")
```

**Installation:**
```bash
# DGX should have TensorRT pre-installed
pip install torch-tensorrt==2.6.0  # Match PyTorch version
```

### 2.2 Dynamic Batching

```python
# Support multiple batch sizes
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 3, 512, 512],
        opt_shape=[8, 3, 512, 512],   # Optimize for batch=8
        max_shape=[32, 3, 512, 512],
        dtype=torch.float16,
    )],
    enabled_precisions={torch.float16},
)
```

### Pros:
- **3-5x speedup** over PyTorch eager
- Automatic kernel fusion (Conv+BN+ReLU → single kernel)
- Automatic precision calibration (FP16/INT8)
- Optimized for NVIDIA GPUs (uses Tensor Cores)
- Graph optimization (dead code elimination, constant folding)
- Still uses PyTorch API (easy integration)

### Cons:
- Compilation time (1-10 minutes, one-time cost)
- Fixed input shapes (or limited dynamic range)
- Requires TensorRT installation (~2GB)
- Debugging harder (optimized graph is opaque)
- Version compatibility issues (TensorRT + PyTorch + CUDA)

### Recommendation:
✅ **HIGHLY RECOMMENDED** for production deployment on DGX systems.

**Expected Performance (512x512, batch=8, A100 80GB):**
- PyTorch FP32: ~15 FPS
- PyTorch FP16: ~35 FPS
- TensorRT FP16: ~80-100 FPS
- TensorRT INT8: ~150-200 FPS (with calibration)

---

## 3. Mixed Precision Inference

**Speedup:** 2-3x over FP32
**Effort:** Low
**Quality Impact:** Negligible (<0.5% L1 increase)

### 3.1 Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast

model = model.half()  # Convert to FP16
input = input.half()

with torch.no_grad(), autocast():
    output = model(input)
```

### 3.2 Full FP16 Conversion

```python
# Convert entire model to FP16
model = model.half()

# Inference
with torch.no_grad():
    input_fp16 = input.half()
    output = model(input_fp16)
    output = output.float()  # Convert back to FP32 if needed
```

### Pros:
- 2-3x faster on A100/H100 (Tensor Cores)
- 2x less memory (can double batch size)
- Simple to implement (1-2 lines)
- Minimal quality loss for this task

### Cons:
- Potential numerical instability (rare for inference)
- Some ops don't support FP16 (auto-upcasts)
- Need to verify accuracy on validation set

### Quality Validation:

```python
# Test FP16 vs FP32 accuracy
model_fp32 = model.float()
model_fp16 = model.half()

for img, target in val_loader:
    out_fp32 = model_fp32(img.cuda())
    out_fp16 = model_fp16(img.half().cuda()).float()

    diff = F.l1_loss(out_fp32, out_fp16)
    print(f"FP32 vs FP16 difference: {diff:.6f}")  # Should be <0.001
```

**Recommendation:** ✅ **STRONGLY RECOMMENDED**. Free 2-3x speedup with negligible quality loss.

---

## 4. torch.compile() - PyTorch 2.0+

**Speedup:** 1.5-2.5x over eager mode
**Effort:** 1 line of code
**Quality Impact:** None (identical results)

PyTorch 2.0+ includes `torch.compile()` which uses **TorchInductor** (graph optimization).

```python
import torch

model = create_restormer('base').eval().cuda()
checkpoint = torch.load('model_checkpoint.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Compile model (one line!)
model = torch.compile(model, mode='max-autotune')

# First run is slow (compilation)
with torch.no_grad():
    output = model(input)  # Compiles here

# Subsequent runs are fast
with torch.no_grad():
    output = model(input)  # Fast!
```

### Compilation Modes:

```python
# 'default' - Balanced compilation time vs runtime
model = torch.compile(model, mode='default')

# 'reduce-overhead' - Minimize overhead (good for small models)
model = torch.compile(model, mode='reduce-overhead')

# 'max-autotune' - Maximum optimization (best for production)
model = torch.compile(model, mode='max-autotune')
```

### Pros:
- **1.5-2.5x speedup** with 1 line of code
- No model changes required
- Bit-for-bit identical results to eager mode
- Automatic kernel fusion
- Works with dynamic shapes
- Free and built into PyTorch 2.0+

### Cons:
- First run is slow (compilation overhead)
- May not work with all custom ops
- Compilation cache can be large (~GB)
- Slightly higher memory usage during compilation
- Less speedup than TensorRT (but easier)

### Recommendation:
✅ **RECOMMENDED** for quick wins. Use before investing in TensorRT.

**Expected Performance (512x512, batch=8, A100):**
- Eager mode: ~15 FPS
- torch.compile('default'): ~25 FPS
- torch.compile('max-autotune'): ~30-35 FPS
- TensorRT FP16: ~80-100 FPS (still better)

---

## 5. Batching Strategies

### 5.1 Static Batching

**Use case:** Offline processing, known dataset size

```python
def batch_inference_static(model, images, batch_size=16):
    """Process images in fixed-size batches."""
    results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

        # Pad last batch if needed
        if len(batch) < batch_size:
            pad_size = batch_size - len(batch)
            batch = batch + [torch.zeros_like(batch[0])] * pad_size

        batch_tensor = torch.stack(batch).cuda()

        with torch.no_grad():
            outputs = model(batch_tensor)

        results.extend(outputs[:len(batch) - pad_size])

    return results
```

### 5.2 Dynamic Batching

**Use case:** Real-time inference, variable input rates

```python
import queue
import threading

class DynamicBatcher:
    """Accumulate inputs and batch when ready."""

    def __init__(self, model, max_batch_size=16, timeout_ms=100):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout = timeout_ms / 1000
        self.queue = queue.Queue()

        # Start batching thread
        self.thread = threading.Thread(target=self._batch_worker)
        self.thread.start()

    def _batch_worker(self):
        """Collect inputs and batch process."""
        while True:
            batch = []
            result_queues = []

            # Collect inputs until batch full or timeout
            deadline = time.time() + self.timeout
            while len(batch) < self.max_batch_size:
                try:
                    timeout = max(0, deadline - time.time())
                    img, result_queue = self.queue.get(timeout=timeout)
                    batch.append(img)
                    result_queues.append(result_queue)
                except queue.Empty:
                    break

            if batch:
                # Process batch
                batch_tensor = torch.stack(batch).cuda()
                with torch.no_grad():
                    outputs = self.model(batch_tensor)

                # Return results
                for output, result_queue in zip(outputs, result_queues):
                    result_queue.put(output)

    def infer(self, img):
        """Submit image for inference."""
        result_queue = queue.Queue()
        self.queue.put((img, result_queue))
        return result_queue.get()  # Wait for result
```

### Pros (Dynamic Batching):
- Optimal GPU utilization even with variable input rates
- Lower latency than waiting for full batches
- Automatic load balancing

### Cons:
- Complexity (threading, queues)
- Debugging harder
- Potential race conditions

### Recommendation:
- Static batching: ✅ Use for offline processing
- Dynamic batching: ⚠️ Use only if needed (real-time serving)

---

## 6. CUDA Graphs

**Speedup:** 1.2-1.5x over regular CUDA
**Effort:** Medium
**Quality Impact:** None

CUDA Graphs capture the entire operation sequence and replay it with minimal overhead.

```python
# Warm up
model.eval()
for _ in range(10):
    with torch.no_grad():
        _ = model(torch.randn(8, 3, 512, 512).cuda())

# Capture CUDA graph
static_input = torch.randn(8, 3, 512, 512).cuda()
static_output = model(static_input)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay graph (very fast)
def infer_with_graph(input_tensor):
    # Copy input to static tensor
    static_input.copy_(input_tensor)

    # Replay graph
    g.replay()

    # Output is in static_output
    return static_output.clone()

# Use
output = infer_with_graph(my_input)
```

### Pros:
- 1.2-1.5x speedup (reduces kernel launch overhead)
- Particularly effective for small models
- Deterministic execution

### Cons:
- **Fixed input/output shapes** (no dynamic batching)
- Memory overhead (graph captures memory)
- Complex to implement
- Doesn't work with all ops (control flow, dynamic shapes)
- Limited benefit for large models (Restormer is already compute-bound)

### Recommendation:
⚠️ **NOT RECOMMENDED** for this model. Better alternatives:
- Use TensorRT instead (more flexible, better speedup)
- Use torch.compile() (easier, similar speedup)

CUDA Graphs are best for:
- Small models (<10M params)
- Latency-critical applications (e.g., <10ms target)
- Fixed input shapes

---

## 7. Multi-GPU Inference

### 7.1 Data Parallel Inference

**Use case:** Very large datasets, multiple GPUs available

```python
import torch.nn as nn

# Wrap model with DataParallel
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # 4 GPUs

# Inference (batch automatically split across GPUs)
batch = torch.randn(32, 3, 512, 512).cuda()  # Will use all 4 GPUs
output = model(batch)
```

### 7.2 Manual Multi-GPU Distribution

```python
def multi_gpu_inference(model, images, gpus=[0, 1, 2, 3]):
    """Distribute images across multiple GPUs."""
    # Replicate model on each GPU
    models = [model.to(f'cuda:{i}') for i in gpus]

    # Split images
    chunks = [images[i::len(gpus)] for i in range(len(gpus))]

    # Process in parallel
    results = []
    for gpu_id, chunk in zip(gpus, chunks):
        device = f'cuda:{gpu_id}'
        with torch.no_grad():
            result = models[gpu_id](chunk.to(device))
        results.append(result.cpu())

    # Combine results
    return torch.cat(results, dim=0)
```

### Pros:
- Near-linear scaling (2x GPUs → 2x throughput)
- Simple to implement with DataParallel
- No model changes required

### Cons:
- Only helps with throughput, not latency
- GPU-to-GPU communication overhead
- Better to use single GPU with batching (if sufficient)
- Cost inefficient (4x GPUs for 3.5x throughput)

### Recommendation:
⚠️ **USE ONLY IF** single GPU can't handle throughput requirements.

**DGX A100 (8x A100 80GB):**
- Single A100 with TensorRT FP16: ~100 FPS @ batch=32
- 8x A100 with DataParallel: ~700 FPS (7x speedup, not 8x due to overhead)

**Better approach:** Maximize single GPU utilization first (TensorRT + FP16 + large batch).

---

## 8. NVIDIA Triton Inference Server

**Speedup:** 1.2-2x over raw PyTorch (with dynamic batching)
**Effort:** High (deployment complexity)
**Quality Impact:** None

Triton is NVIDIA's inference serving platform (production-grade).

### Features:
- Dynamic batching (automatic request batching)
- Model versioning
- Multi-model serving
- HTTP/gRPC APIs
- Prometheus metrics
- A/B testing support
- TensorRT backend support

### Setup:

```bash
# 1. Export model to TorchScript or ONNX
torch.jit.save(torch.jit.script(model), 'model.pt')

# 2. Create Triton model repository
mkdir -p models/restormer_hdr/1/
mv model.pt models/restormer_hdr/1/model.pt

# 3. Create config.pbtxt
cat > models/restormer_hdr/config.pbtxt <<EOF
name: "restormer_hdr"
platform: "pytorch_libtorch"
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100000
}
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, 512, 512]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [3, 512, 512]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
EOF

# 4. Run Triton server
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver --model-repository=/models
```

### Client Code:

```python
import tritonclient.http as httpclient
import numpy as np

# Create client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input
input_data = np.random.randn(1, 3, 512, 512).astype(np.float32)
inputs = [httpclient.InferInput("input__0", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# Run inference
outputs = [httpclient.InferRequestedOutput("output__0")]
response = client.infer("restormer_hdr", inputs, outputs=outputs)

# Get result
output_data = response.as_numpy("output__0")
```

### Pros:
- Production-ready (monitoring, versioning, scaling)
- Automatic dynamic batching
- Multi-model serving (one server, many models)
- REST/gRPC APIs (language-agnostic)
- TensorRT backend (best of both worlds)
- Battle-tested (used by NVIDIA, major cloud providers)

### Cons:
- High deployment complexity
- Overkill for simple use cases
- Learning curve (configuration, APIs)
- Debugging harder (server logs, network issues)
- Container overhead

### Recommendation:
✅ **RECOMMENDED** for production deployment at scale (>1000 QPS).

⚠️ **NOT RECOMMENDED** for:
- Research/prototyping
- Single-user inference
- <100 QPS throughput requirements

---

## 9. Model Quantization (INT8)

**Speedup:** 2-4x over FP16
**Effort:** High
**Quality Impact:** Moderate (0.5-2% L1 increase, depends on calibration)

INT8 quantization reduces model size by 4x and speeds up inference using INT8 Tensor Cores.

### 9.1 Post-Training Quantization (PTQ)

```python
import torch
from torch.quantization import quantize_dynamic

# Dynamic quantization (simple but less optimal)
model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8
)
```

### 9.2 TensorRT INT8 Calibration (SOTA)

```python
import torch_tensorrt

# Prepare calibration data
calibration_data = []
for i, (img, _) in enumerate(val_loader):
    if i >= 500:  # Use 500 images for calibration
        break
    calibration_data.append(img.cuda())

# Compile with INT8
trt_model_int8 = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[8, 3, 512, 512],
        dtype=torch.float32,
    )],
    enabled_precisions={torch.int8},
    calibrator=torch_tensorrt.ptq.DataLoaderCalibrator(
        calibration_data,
        use_cache=True,
        cache_file="calibration.cache",
    ),
)
```

### Validation:

```python
# Compare INT8 vs FP16 quality
model_fp16 = model.half()
model_int8 = trt_model_int8

l1_diffs = []
for img, target in val_loader:
    out_fp16 = model_fp16(img.half().cuda())
    out_int8 = model_int8(img.cuda())

    l1_diff = F.l1_loss(out_fp16.float(), out_int8.float())
    l1_diffs.append(l1_diff.item())

print(f"FP16 vs INT8 L1 diff: {np.mean(l1_diffs):.6f}")
# Should be <0.005 for good calibration
```

### Pros:
- 2-4x speedup over FP16
- 4x less memory (can 4x batch size)
- Uses INT8 Tensor Cores (A100+)
- Minimal quality loss with good calibration

### Cons:
- Requires calibration dataset (500-1000 images)
- Quality degradation (need validation)
- Not all ops support INT8
- Transformer models can be sensitive to quantization
- May need per-layer quantization tuning

### Expected Quality Impact (Restormer):
- FP32: L1 = 0.0515 (baseline)
- FP16: L1 = 0.0516 (+0.02%)
- INT8 (good calibration): L1 = 0.0525-0.0535 (+2-4%)
- INT8 (poor calibration): L1 = 0.055-0.060 (+7-16%)

### Recommendation:
⚠️ **USE WITH CAUTION**. Only if:
1. FP16 performance is insufficient
2. You have validation set for calibration
3. You can tolerate 2-4% quality loss

**Validation process:**
1. Calibrate with 500-1000 diverse images
2. Test on full validation set
3. Visual inspection of worst-case outputs
4. A/B test vs FP16 on production traffic

---

## 10. Advanced: ONNX + TensorRT

**Speedup:** 3-5x (similar to TensorRT direct)
**Effort:** High
**Quality Impact:** Negligible

ONNX provides model portability and optimization opportunities.

### Workflow:

```python
import torch
import onnx
import onnx_tensorrt

# 1. Export to ONNX
dummy_input = torch.randn(1, 3, 512, 512).cuda()
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    },
    opset_version=17,
)

# 2. Optimize ONNX
import onnx
from onnxsim import simplify

onnx_model = onnx.load("model.onnx")
onnx_model_simplified, check = simplify(onnx_model)
onnx.save(onnx_model_simplified, "model_simplified.onnx")

# 3. Convert to TensorRT
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model_simplified.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16

serialized_engine = builder.build_serialized_network(network, config)

# Save engine
with open("model.trt", "wb") as f:
    f.write(serialized_engine)
```

### Pros:
- Model portability (ONNX is framework-agnostic)
- Optimization passes (constant folding, operator fusion)
- Easier deployment to non-PyTorch environments
- Explicit control over TensorRT optimization

### Cons:
- Complex multi-step process
- ONNX export can fail for custom ops
- Version compatibility hell (PyTorch → ONNX → TensorRT)
- Debugging difficult (3 frameworks involved)
- Same speedup as direct TensorRT (more effort, same result)

### Recommendation:
⚠️ **NOT RECOMMENDED** unless you need ONNX for portability.

**Use instead:**
- Direct TensorRT via torch-tensorrt (easier, same performance)

---

## Comparison Matrix

| Method | Speedup | Effort | Quality | Memory | GPU Support | Recommended |
|--------|---------|--------|---------|--------|-------------|-------------|
| **TF32 Enable** | 1.3-1.5x | Minimal | No change | Same | A100+ | ✅ Always |
| **Batch Size ↑** | 2-4x | Low | No change | +Linear | All | ✅ Always |
| **FP16 Inference** | 2-3x | Low | Negligible | 0.5x | All (best on A100+) | ✅ Strongly Rec |
| **torch.compile()** | 1.5-2.5x | Minimal | No change | +10% | All | ✅ Recommended |
| **TensorRT FP16** | 3-5x | Medium | Negligible | Same | All (best on A100+) | ✅ SOTA Choice |
| **TensorRT INT8** | 5-8x | High | Moderate | 0.25x | A100+ | ⚠️ If needed |
| **CUDA Graphs** | 1.2-1.5x | High | No change | +20% | All | ⚠️ Not for this model |
| **Multi-GPU** | Nx (N GPUs) | Medium | No change | Nx | All | ⚠️ Only if needed |
| **Triton Server** | 1.2-2x | Very High | No change | Same | All | ✅ Production scale |
| **ONNX+TensorRT** | 3-5x | Very High | Negligible | Same | All | ⚠️ Only for portability |

---

## Recommended Pipeline (DGX A100/H100)

### Stage 1: Quick Wins (30 minutes)
```python
# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Convert to FP16
model = model.half()

# Increase batch size
batch_size = 32  # A100 80GB can handle this

# Result: ~60 FPS (4x over baseline)
```

### Stage 2: torch.compile() (1 hour)
```python
model = torch.compile(model, mode='max-autotune')

# First run compiles (slow)
warmup = model(torch.randn(32, 3, 512, 512).half().cuda())

# Result: ~80-100 FPS (5-6x over baseline)
```

### Stage 3: TensorRT (4 hours)
```python
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 3, 512, 512],
        opt_shape=[32, 3, 512, 512],
        max_shape=[64, 3, 512, 512],
        dtype=torch.float16,
    )],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30,
)

# Save for reuse
torch.jit.save(trt_model, "model_trt_fp16.ts")

# Result: ~150-200 FPS (10-13x over baseline)
```

### Stage 4 (Optional): INT8 Quantization (8 hours)
```python
# Only if FP16 performance insufficient
# Requires careful calibration and validation

trt_model_int8 = torch_tensorrt.compile(
    model,
    inputs=[...],
    enabled_precisions={torch.int8},
    calibrator=...,
)

# Result: ~300-400 FPS (20-26x over baseline)
# Quality: L1 increase of 2-4%
```

---

## Expected Performance Summary

**Hardware:** DGX A100 80GB
**Model:** Restormer Base (25.4M params)
**Input:** 512x512x3 RGB

| Configuration | Throughput (FPS) | Latency (ms) | Speedup | Quality (L1) |
|---------------|------------------|--------------|---------|--------------|
| Baseline (PyTorch FP32, batch=1) | 15 | 67 | 1.0x | 0.0515 |
| + TF32 | 20 | 50 | 1.3x | 0.0515 |
| + FP16 | 35 | 29 | 2.3x | 0.0516 |
| + FP16 + Batch=32 | 60 | 533 | 4.0x | 0.0516 |
| + torch.compile() | 90 | 356 | 6.0x | 0.0516 |
| **TensorRT FP16 (batch=32)** | **180** | **178** | **12x** | **0.0517** |
| TensorRT INT8 (batch=32) | 350 | 91 | 23x | 0.0535 |

**Recommended:** TensorRT FP16 (best performance/quality tradeoff)

---

## DGX-Specific Optimizations

### DGX A100 (8x A100 80GB)

**Single GPU (Recommended):**
- TensorRT FP16, batch=32: ~180 FPS
- Total throughput: 180 FPS per GPU × 8 GPUs = 1,440 FPS (if workload parallelizable)

**Multi-GPU (Only if needed):**
- Use NVIDIA Triton with multiple model instances
- Each GPU runs independent Triton instance
- Load balance via reverse proxy (NGINX)

### DGX H100 (8x H100 80GB)

**Additional benefits:**
- FP8 support (Transformer Engine)
- 2x faster Tensor Cores vs A100
- Expected: ~300-400 FPS per GPU with TensorRT FP16

### DGX H200 (8x H200 141GB)

**Additional benefits:**
- Massive memory: batch=128 possible
- Same compute as H100, more memory bandwidth
- Expected: ~400-500 FPS per GPU with TensorRT FP16

---

## Monitoring and Profiling

### NVIDIA Nsight Systems

```bash
# Profile inference
nsys profile -o profile.qdrep python infer.py

# View in GUI
nsys-ui profile.qdrep
```

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

### TensorRT Profiling

```python
import tensorrt as trt

# Enable profiling in TensorRT
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

# Build and run engine...

# View profile
# Use trtexec or Nsight Systems
```

---

## Summary: SOTA Recommendations

### For Maximum Throughput (Offline Batch Processing):
1. ✅ TensorRT FP16 (3-5x speedup)
2. ✅ Large batch size (32-64 on A100)
3. ✅ Enable TF32
4. ⚠️ Consider INT8 if quality acceptable (2x more speedup)

**Expected: 150-200 FPS per A100 GPU**

### For Low Latency (Real-Time):
1. ✅ TensorRT FP16 (3-5x speedup)
2. ✅ Batch size = 1-4
3. ✅ CUDA Graphs (if latency <10ms required)
4. ⚠️ Pin CPU cores, isolate GPU

**Expected: 5-10ms per image**

### For Production Deployment:
1. ✅ NVIDIA Triton Inference Server
2. ✅ TensorRT FP16 backend
3. ✅ Dynamic batching (8-32)
4. ✅ Model versioning
5. ✅ Prometheus monitoring

**Expected: 180+ FPS per GPU, auto-scaling, production-grade**

### For Research/Prototyping:
1. ✅ torch.compile('max-autotune')
2. ✅ FP16 inference
3. ✅ Batch size tuning

**Expected: 80-100 FPS per GPU, minimal code changes**

---

## Final Recommendation Matrix

| Use Case | Solution | Expected Performance | Effort |
|----------|----------|---------------------|--------|
| **Quick prototyping** | torch.compile + FP16 | 80-100 FPS | 1 hour |
| **Maximum throughput** | TensorRT FP16 + batch=32 | 150-200 FPS | 4 hours |
| **Extreme throughput** | TensorRT INT8 + batch=64 | 300-400 FPS | 8 hours + validation |
| **Production serving** | Triton + TensorRT FP16 | 180+ FPS + scaling | 1-2 days |
| **Ultra-low latency** | TensorRT FP16 + CUDA Graphs | <10ms | 8 hours |

---

**Analysis by:** Top 0.0001% PhD-level ML Engineer
**Date:** December 2025
**Hardware Target:** NVIDIA DGX A100/H100/H200 Systems
