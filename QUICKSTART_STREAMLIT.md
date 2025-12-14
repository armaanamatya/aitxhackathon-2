# Quick Start: Streamlit Inference Interface

## Run the App (3 Steps)

### Step 1: Install Streamlit (if not already installed)
```bash
pip install streamlit Pillow numpy
```

### Step 2: Run the App

**Windows:**
```bash
streamlit run streamlit_inference.py
```

**Or double-click:** `run_streamlit.bat`

**Linux/Mac:**
```bash
./run_streamlit.sh
```

### Step 3: Use the Interface

1. **Load Model** (in left sidebar):
   - Enter model path: `checkpoints/restormer_base.pt`
   - Click "🔄 Load Model"

2. **Upload Image**:
   - Click "Browse files"
   - Select an image (JPG/PNG)

3. **Enhance**:
   - Click "🚀 Enhance Image"
   - See side-by-side comparison!

## Features

✅ **Visual Comparison** - See before/after side-by-side  
✅ **Metrics** - Brightness, contrast, colorfulness changes  
✅ **Batch Processing** - Process multiple images  
✅ **Download Results** - Save enhanced images  
✅ **Beautiful UI** - Clean, modern interface  

## What You'll See

- **Original Image** (left column)
- **Enhanced Image** (right column)  
- **Side-by-Side Comparison** (full width)
- **Metrics Dashboard** (brightness, contrast, time)
- **Download Buttons** (individual & comparison)

## Troubleshooting

**App won't start?**
```bash
pip install streamlit --upgrade
```

**Model won't load?**
- Check model path is correct
- Verify file exists
- Try CPU mode if CUDA issues

**Images not processing?**
- Make sure model is loaded first
- Check image format (JPG/PNG)
- Look for error messages

## Next Steps

- See `STREAMLIT_GUIDE.md` for full documentation
- Customize the interface
- Add your own metrics
- Integrate with API server

