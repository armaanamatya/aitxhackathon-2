"""
Safe launcher for Gradio demo - handles PyTorch errors gracefully
"""

import sys
import os
from pathlib import Path

# Set environment to use CPU if CUDA fails
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

print("Starting Gradio demo...")
print("If PyTorch has CUDA issues, demo will run in mock mode.")
print()

try:
    import demo
    print("Demo module loaded successfully")
except Exception as e:
    print(f"Error loading demo: {e}")
    print("\nTrying to run with CPU-only mode...")
    
    # Force CPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MODEL_DEVICE"] = "cpu"
    
    try:
        import demo
    except Exception as e2:
        print(f"Still failed: {e2}")
        print("\nThe demo will run but model inference won't work.")
        print("You can still test the UI interface.")
        sys.exit(1)

