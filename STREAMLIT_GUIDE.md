# Streamlit Inference Interface Guide

## Overview

A beautiful, interactive Streamlit interface for comparing original vs enhanced images with side-by-side comparisons, metrics, and batch processing.

## Features

- 🖼️ **Single Image Processing**: Upload and enhance individual images
- 📊 **Visual Comparison**: Side-by-side before/after views
- 📈 **Metrics**: Brightness, contrast, and colorfulness changes
- 📦 **Batch Processing**: Process multiple images at once
- 📥 **Download Results**: Download individual or comparison images
- ⚙️ **Configurable**: Adjust model settings, device, and processing options

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit Pillow numpy
```

Or use the requirements file:
```bash
pip install -r requirements_streamlit.txt
```

### 2. Run the App

**Option A: Using the launcher script**
```bash
./run_streamlit.sh
```

**Option B: Direct command**
```bash
streamlit run streamlit_inference.py
```

**Option C: With custom port**
```bash
streamlit run streamlit_inference.py --server.port 8502
```

### 3. Open in Browser

The app will automatically open at: http://localhost:8501

## Usage

### Single Image Processing

1. **Load Model** (in sidebar):
   - Enter backbone model path (e.g., `checkpoints/restormer_base.pt`)
   - Optionally enter refiner path
   - Select device (cuda/cpu)
   - Click "🔄 Load Model"

2. **Upload Image**:
   - Click "Browse files" or drag and drop
   - Supported formats: JPG, JPEG, PNG

3. **Process**:
   - Click "🚀 Enhance Image"
   - View results side-by-side

4. **Download**:
   - Download enhanced image
   - Download comparison image
   - Download original image

### Batch Processing

1. Upload multiple images using the batch uploader
2. Click "🚀 Process All"
3. View all results in expandable sections
4. Download individual enhanced images

### Metrics

The interface shows:
- **Brightness Change**: Difference in average brightness
- **Contrast Change**: Difference in standard deviation
- **Colorfulness Change**: Difference in color variation
- **Processing Time**: Time taken for inference

## Configuration

### Model Settings (Sidebar)

- **Backbone Model Path**: Path to your backbone checkpoint
- **Refiner Model Path**: Optional refiner checkpoint
- **Device**: cuda or cpu
- **Use Refiner**: Toggle refiner on/off

### Processing Settings

- **Show Metrics**: Display image statistics
- **Show Side-by-Side**: Display comparison view
- **Enable Download**: Show download buttons

## Features in Detail

### Visual Comparison

- Automatic resizing to match heights
- Side-by-side layout
- High-quality image display
- Full-width responsive design

### Metrics Display

- Real-time calculation
- Visual metric cards
- Detailed JSON view
- Processing time tracking

### Batch Processing

- Progress bar
- Status updates
- Error handling per image
- Individual result views
- Bulk downloads

## Troubleshooting

### Model Not Loading

- Check model path is correct
- Verify checkpoint file exists
- Check device availability (CUDA/CPU)
- See error messages in sidebar

### Images Not Processing

- Ensure model is loaded first
- Check image format (JPG/PNG)
- Verify image is not corrupted
- Check console for errors

### Performance Issues

- Use CPU mode if CUDA unavailable
- Reduce image size for faster processing
- Process images one at a time
- Close other applications

## Integration with API Server

You can also use the FastAPI server for inference:

1. Start API server: `python run_api_server.py`
2. Modify `streamlit_inference.py` to use API endpoints
3. Make HTTP requests to `/inference/` endpoint

## Customization

### Add Example Images

Edit the `example_images` section:
```python
example_images = st.selectbox(
    "Select example image",
    ["None", "Example 1", "Example 2", "Example 3"],
)
```

Then add image loading logic.

### Custom Metrics

Modify `calculate_metrics()` function to add:
- PSNR
- SSIM
- LPIPS
- Custom metrics

### Styling

Edit the CSS in the `st.markdown()` section to customize:
- Colors
- Layout
- Fonts
- Spacing

## Advanced Features

### API Integration

Connect to FastAPI server:
```python
import requests

response = requests.post(
    "http://localhost:8000/inference/",
    json={"image_base64": base64_image}
)
```

### Custom Processing

Add preprocessing/postprocessing:
```python
def custom_process(image):
    # Your custom processing
    return processed_image
```

## Tips

- **Large Images**: Resize before processing for faster inference
- **Batch Size**: Process 5-10 images at a time for best performance
- **Model Loading**: Load model once, process many images
- **Downloads**: Use PNG format for best quality

## Next Steps

- Add more metrics (PSNR, SSIM, LPIPS)
- Integrate with API server
- Add image history
- Save processing presets
- Export processing reports

