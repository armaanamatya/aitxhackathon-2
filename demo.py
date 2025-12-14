"""
Gradio Demo for Real Estate HDR Enhancement

Provides a web interface for:
- Single image enhancement
- Before/after comparison
- Batch processing
"""

import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Try to import HDREnhancer, but handle errors gracefully
try:
    from src.inference.infer import HDREnhancer
    ENHANCER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import HDREnhancer: {e}")
    print("Demo will run in mock mode (images will pass through unchanged)")
    ENHANCER_AVAILABLE = False
    HDREnhancer = None


# Global enhancer instance
enhancer = None


def load_model(model_path: str, image_size: int = 512, precision: str = "fp16"):
    """Load or reload the model."""
    global enhancer

    if not ENHANCER_AVAILABLE:
        return "Error: HDREnhancer not available. PyTorch may not be properly installed."

    if not os.path.exists(model_path):
        return f"Error: Model not found at {model_path}"

    try:
        enhancer = HDREnhancer(
            model_path=model_path,
            image_size=image_size,
            precision=precision,
        )
        return f"Model loaded successfully from {model_path}"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def enhance_image(
    input_image: np.ndarray,
    preserve_resolution: bool = True,
) -> tuple:
    """
    Enhance a single image.

    Returns:
        Tuple of (enhanced_image, processing_info)
    """
    global enhancer

    if not ENHANCER_AVAILABLE:
        # Mock mode - return original image
        if input_image is None:
            return None, "Error: No input image provided."
        return input_image, "Mock mode: Image passed through unchanged (PyTorch not available)"

    if enhancer is None:
        return None, "Error: Model not loaded. Please load a model first."

    if input_image is None:
        return None, "Error: No input image provided."

    try:
        # Convert numpy array to PIL Image
        input_pil = Image.fromarray(input_image)

        # Process
        start_time = time.perf_counter()
        enhanced_pil = enhancer.enhance(input_pil, preserve_resolution=preserve_resolution)
        processing_time = time.perf_counter() - start_time

        # Convert back to numpy
        enhanced_np = np.array(enhanced_pil)

        info = f"Processing time: {processing_time*1000:.2f} ms\n"
        info += f"Input size: {input_pil.size}\n"
        info += f"Output size: {enhanced_pil.size}"

        return enhanced_np, info

    except Exception as e:
        return None, f"Error: {str(e)}"


def create_comparison(
    input_image: np.ndarray,
    enhanced_image: np.ndarray,
) -> np.ndarray:
    """Create side-by-side comparison."""
    if input_image is None or enhanced_image is None:
        return None

    # Ensure same size
    h1, w1 = input_image.shape[:2]
    h2, w2 = enhanced_image.shape[:2]

    if (h1, w1) != (h2, w2):
        # Resize to match
        enhanced_pil = Image.fromarray(enhanced_image).resize((w1, h1), Image.LANCZOS)
        enhanced_image = np.array(enhanced_pil)

    # Create side-by-side
    comparison = np.concatenate([input_image, enhanced_image], axis=1)

    return comparison


def build_demo():
    """Build the Gradio interface."""

    with gr.Blocks(title="Real Estate HDR Enhancement") as demo:
        gr.Markdown("""
        # Real Estate HDR Photo Enhancement

        Transform unedited real estate photos into professionally edited images using AI.

        **Features:**
        - Automatic exposure correction
        - White balance adjustment
        - HDR-style tone mapping
        - Powered by NVIDIA TensorRT optimization

        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Model settings
                gr.Markdown("### Model Settings")
                model_path_input = gr.Textbox(
                    label="Model Path",
                    value="checkpoints/best_generator.pt",
                    placeholder="Path to model weights"
                )
                with gr.Row():
                    image_size_input = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=128,
                        label="Processing Size"
                    )
                    precision_input = gr.Dropdown(
                        choices=["fp16", "fp32"],
                        value="fp16",
                        label="Precision"
                    )
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                input_image = gr.Image(
                    label="Upload Real Estate Photo",
                    type="numpy",
                )
                preserve_res = gr.Checkbox(
                    label="Preserve Original Resolution",
                    value=True
                )
                enhance_btn = gr.Button("Enhance Photo", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Enhanced Output")
                output_image = gr.Image(
                    label="Enhanced Photo",
                    type="numpy",
                )
                processing_info = gr.Textbox(
                    label="Processing Info",
                    interactive=False
                )

        with gr.Row():
            gr.Markdown("### Side-by-Side Comparison")

        comparison_image = gr.Image(
            label="Before (Left) vs After (Right)",
            type="numpy",
        )

        # Example images
        gr.Markdown("---")
        gr.Markdown("### Example Images")
        gr.Examples(
            examples=[
                ["images/100_src.jpg"],
                ["images/101_src.jpg"],
                ["images/102_src.jpg"],
            ],
            inputs=input_image,
            label="Click an example to load",
        )

        # Event handlers
        load_btn.click(
            fn=load_model,
            inputs=[model_path_input, image_size_input, precision_input],
            outputs=model_status,
        )

        enhance_btn.click(
            fn=enhance_image,
            inputs=[input_image, preserve_res],
            outputs=[output_image, processing_info],
        ).then(
            fn=create_comparison,
            inputs=[input_image, output_image],
            outputs=comparison_image,
        )

        gr.Markdown("""
        ---
        ### Technical Details

        **Model Architecture:** U-Net with residual learning
        **Training:** Pix2Pix GAN with perceptual loss
        **Optimization:** NVIDIA TensorRT for fast inference
        **Hardware:** Optimized for NVIDIA DGX Spark

        **Performance:**
        - Processing time: ~50-100ms per image
        - Supports up to 4K resolution with tiled processing
        """)

    return demo


if __name__ == "__main__":
    # Try to load default model (only if enhancer is available)
    if ENHANCER_AVAILABLE:
        default_model = "checkpoints/best_generator.pt"
        if os.path.exists(default_model):
            try:
                load_model(default_model)
            except Exception as e:
                print(f"Warning: Could not load default model: {e}")
    else:
        print("\n" + "="*60)
        print("WARNING: Running in MOCK MODE")
        print("="*60)
        print("PyTorch is not available or has CUDA issues.")
        print("The demo will run but images will pass through unchanged.")
        print("To fix: Install PyTorch properly or use CPU mode.")
        print("="*60 + "\n")

    # Build and launch demo
    demo = build_demo()
    
    # Try to find an available port
    import socket
    def find_free_port(start_port=7860, max_attempts=10):
        for i in range(max_attempts):
            port = start_port + i
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return None
    
    # Find available port
    port = find_free_port(7860)
    if port is None:
        port = 7860  # Fallback, Gradio will handle error
    
    print(f"\n{'='*60}")
    print(f"Starting Gradio demo on port {port}")
    print(f"{'='*60}\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,  # Creates public link via Gradio
        show_error=True,  # Show errors in UI
        quiet=False,  # Show all output
    )
