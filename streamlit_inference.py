"""
Streamlit Inference Interface - Image Comparison
Compare original vs enhanced images side-by-side
"""

# IMPORTANT: st.set_page_config() MUST be the very first Streamlit command
# Import streamlit first, then immediately set page config
import streamlit as st

# Page config MUST be first - before ANY other Streamlit commands
st.set_page_config(
    page_title="HDR Enhancement Inference",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules (these won't execute Streamlit commands)
import sys
from pathlib import Path
import time
import base64
import io
from PIL import Image
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Try to import model components (after page config)
# Store error to show later, don't use st.warning() here
try:
    from api_server.models.manager import ModelManager
    MODEL_AVAILABLE = True
    MODEL_ERROR = None
except Exception as e:
    MODEL_AVAILABLE = False
    MODEL_ERROR = str(e)  # Store error, show it later in the UI

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .comparison-container {
        display: flex;
        justify-content: space-around;
        gap: 2rem;
        margin: 2rem 0;
    }
    .image-box {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(backbone_path, refiner_path=None, device="cuda"):
    """Load model with caching."""
    if not MODEL_AVAILABLE:
        return None
    
    try:
        model_manager = ModelManager(
            backbone_path=backbone_path,
            refiner_path=refiner_path,
            device=device,
            precision="fp16",
            use_compile=True
        )
        return model_manager
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def process_image(model_manager, image, use_refiner=True):
    """Process image through model."""
    if model_manager is None:
        # Mock processing for demo
        return image
    
    try:
        output = model_manager.infer(image, use_refiner=use_refiner)
        return output
    except Exception as e:
        st.error(f"Inference error: {e}")
        return image


def create_comparison(original, enhanced):
    """Create side-by-side comparison image."""
    # Ensure same height
    h1, w1 = original.size[1], original.size[0]
    h2, w2 = enhanced.size[1], enhanced.size[0]
    
    if h1 != h2:
        # Resize to match
        if h1 > h2:
            enhanced = enhanced.resize((int(w2 * h1 / h2), h1), Image.LANCZOS)
        else:
            original = original.resize((int(w1 * h2 / h1), h2), Image.LANCZOS)
    
    # Create side-by-side
    total_width = original.size[0] + enhanced.size[0]
    max_height = max(original.size[1], enhanced.size[1])
    
    comparison = Image.new('RGB', (total_width, max_height))
    comparison.paste(original, (0, 0))
    comparison.paste(enhanced, (original.size[0], 0))
    
    return comparison


def calculate_metrics(original, enhanced):
    """Calculate simple image metrics."""
    orig_arr = np.array(original).astype(float)
    enh_arr = np.array(enhanced.resize(original.size, Image.LANCZOS)).astype(float)
    
    # Mean brightness
    orig_brightness = orig_arr.mean()
    enh_brightness = enh_arr.mean()
    
    # Contrast (std dev)
    orig_contrast = orig_arr.std()
    enh_contrast = enh_arr.std()
    
    # Colorfulness (simplified)
    orig_color = np.std(orig_arr, axis=2).mean() if len(orig_arr.shape) == 3 else 0
    enh_color = np.std(enh_arr, axis=2).mean() if len(enh_arr.shape) == 3 else 0
    
    return {
        "brightness_change": f"{(enh_brightness - orig_brightness):.1f}",
        "contrast_change": f"{(enh_contrast - orig_contrast):.1f}",
        "colorfulness_change": f"{(enh_color - orig_color):.1f}",
        "original_brightness": f"{orig_brightness:.1f}",
        "enhanced_brightness": f"{enh_brightness:.1f}"
    }


def main():
    """Main Streamlit app."""
    
    # Show warning if model not available (after page config)
    if not MODEL_AVAILABLE:
        st.warning(f"⚠️ Model manager not available: {MODEL_ERROR}. Using mock mode (images will pass through unchanged).")
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ HDR Photo Enhancement</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        backbone_path = st.text_input(
            "Backbone Model Path",
            value="checkpoints/restormer_base.pt",
            help="Path to backbone model checkpoint"
        )
        
        refiner_path = st.text_input(
            "Refiner Model Path (Optional)",
            value="checkpoints/refiner.pt",
            help="Path to refiner model checkpoint (leave empty if not using)"
        )
        
        device = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0,
            help="Device to run inference on"
        )
        
        use_refiner = st.checkbox(
            "Use Refiner",
            value=True,
            help="Apply color refiner to output"
        )
        
        # Load model button
        load_model_btn = st.button("🔄 Load Model", type="primary", use_container_width=True)
        
        # Processing settings
        st.subheader("Processing Settings")
        show_metrics = st.checkbox("Show Metrics", value=True)
        show_comparison = st.checkbox("Show Side-by-Side", value=True)
        download_comparison = st.checkbox("Enable Download", value=True)
        
        # Info
        st.markdown("---")
        st.info("💡 **Tip**: Upload images to see before/after comparison")
    
    # Initialize session state
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Load model
    if load_model_btn:
        with st.spinner("Loading model..."):
            if Path(backbone_path).exists():
                st.session_state.model_manager = load_model(
                    backbone_path,
                    refiner_path if refiner_path and Path(refiner_path).exists() else None,
                    device
                )
                if st.session_state.model_manager:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.session_state.model_loaded = False
                    st.error("❌ Failed to load model")
            else:
                st.warning(f"⚠️ Model file not found: {backbone_path}")
                st.session_state.model_loaded = False
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to enhance"
        )
        
        # Or use example images
        st.subheader("Or Use Example")
        example_images = st.selectbox(
            "Select example image",
            ["None", "Example 1", "Example 2", "Example 3"],
            help="Select a pre-loaded example image"
        )
    
    with col2:
        st.header("📊 Status")
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
            if st.session_state.model_manager:
                info = st.session_state.model_manager.get_model_info()
                st.json(info)
        else:
            st.warning("⚠️ Model Not Loaded")
            st.info("Click 'Load Model' in sidebar to load a model")
    
    # Process image
    if uploaded_file is not None or example_images != "None":
        st.markdown("---")
        
        # Load image
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')
            image_name = uploaded_file.name
        else:
            # Load example image (you can add actual example images)
            st.info("Example images not implemented. Please upload an image.")
            original_image = None
        
        if original_image is not None:
            # Display original
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(original_image, use_container_width=True, caption=f"Original: {image_name}")
            
            # Process button
            process_btn = st.button("🚀 Enhance Image", type="primary", use_container_width=True)
            
            if process_btn:
                if not st.session_state.model_loaded:
                    st.error("❌ Please load a model first!")
                else:
                    with st.spinner("Processing image..."):
                        start_time = time.time()
                        
                        # Process image
                        enhanced_image = process_image(
                            st.session_state.model_manager,
                            original_image,
                            use_refiner=use_refiner
                        )
                        
                        processing_time = time.time() - start_time
                        
                        # Display enhanced
                        with col2:
                            st.subheader("✨ Enhanced Image")
                            st.image(enhanced_image, use_container_width=True, caption="Enhanced Result")
                        
                        # Metrics
                        if show_metrics:
                            st.markdown("---")
                            st.subheader("📊 Image Metrics")
                            
                            metrics = calculate_metrics(original_image, enhanced_image)
                            
                            metric_cols = st.columns(3)
                            with metric_cols[0]:
                                st.metric("Brightness Change", metrics["brightness_change"])
                            with metric_cols[1]:
                                st.metric("Contrast Change", metrics["contrast_change"])
                            with metric_cols[2]:
                                st.metric("Processing Time", f"{processing_time:.2f}s")
                            
                            # Detailed metrics
                            with st.expander("Detailed Metrics"):
                                st.json(metrics)
                        
                        # Side-by-side comparison
                        if show_comparison:
                            st.markdown("---")
                            st.subheader("🔄 Side-by-Side Comparison")
                            
                            comparison = create_comparison(original_image, enhanced_image)
                            st.image(comparison, use_container_width=True, caption="Original (Left) vs Enhanced (Right)")
                            
                            # Download button
                            if download_comparison:
                                buf = io.BytesIO()
                                comparison.save(buf, format='PNG')
                                buf.seek(0)
                                
                                st.download_button(
                                    label="📥 Download Comparison",
                                    data=buf,
                                    file_name=f"comparison_{image_name}",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        
                        # Individual downloads
                        st.markdown("---")
                        st.subheader("📥 Download Results")
                        
                        download_cols = st.columns(2)
                        
                        with download_cols[0]:
                            # Download enhanced
                            buf_enh = io.BytesIO()
                            enhanced_image.save(buf_enh, format='PNG')
                            buf_enh.seek(0)
                            st.download_button(
                                label="📥 Download Enhanced",
                                data=buf_enh,
                                file_name=f"enhanced_{image_name}",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        with download_cols[1]:
                            # Download original
                            buf_orig = io.BytesIO()
                            original_image.save(buf_orig, format='PNG')
                            buf_orig.seek(0)
                            st.download_button(
                                label="📥 Download Original",
                                data=buf_orig,
                                file_name=f"original_{image_name}",
                                mime="image/png",
                                use_container_width=True
                            )
            
            # Batch processing section
            st.markdown("---")
            st.subheader("📦 Batch Processing")
            
            batch_files = st.file_uploader(
                "Upload multiple images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
            
            if batch_files and st.button("🚀 Process All", use_container_width=True):
                if not st.session_state.model_loaded:
                    st.error("❌ Please load a model first!")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for i, file in enumerate(batch_files):
                        status_text.text(f"Processing {i+1}/{len(batch_files)}: {file.name}")
                        
                        try:
                            img = Image.open(file).convert('RGB')
                            enhanced = process_image(
                                st.session_state.model_manager,
                                img,
                                use_refiner=use_refiner
                            )
                            
                            results.append({
                                "name": file.name,
                                "original": img,
                                "enhanced": enhanced
                            })
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(batch_files))
                    
                    status_text.text("✅ Processing complete!")
                    
                    # Display batch results
                    st.markdown("### 📊 Batch Results")
                    
                    for result in results:
                        with st.expander(f"📷 {result['name']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(result['original'], caption="Original", use_container_width=True)
                            with col2:
                                st.image(result['enhanced'], caption="Enhanced", use_container_width=True)
                            
                            # Download buttons
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                buf = io.BytesIO()
                                result['enhanced'].save(buf, format='PNG')
                                buf.seek(0)
                                st.download_button(
                                    f"📥 Download {result['name']}",
                                    data=buf,
                                    file_name=f"enhanced_{result['name']}",
                                    mime="image/png",
                                    key=f"dl_{result['name']}"
                                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🖼️ HDR Photo Enhancement Inference Interface</p>
        <p>Powered by Restormer & FastAPI</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

