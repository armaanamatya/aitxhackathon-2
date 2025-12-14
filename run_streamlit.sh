#!/bin/bash
# Streamlit Inference Interface Launcher

echo "=============================================="
echo "Streamlit HDR Enhancement Interface"
echo "=============================================="
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing Streamlit..."
    pip install streamlit Pillow numpy
fi

echo "Starting Streamlit app..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""

streamlit run streamlit_inference.py --server.port 8501 --server.address 0.0.0.0

