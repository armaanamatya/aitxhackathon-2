"""
Triton Inference Server client.
Handles communication with Triton for model inference.
"""

import logging
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

logger = logging.getLogger(__name__)


class TritonClient:
    """
    Client for Triton Inference Server.
    Handles model inference via Triton.
    """
    
    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "restormer",
        model_version: str = "1",
        timeout: float = 30.0
    ):
        """
        Initialize Triton client.
        
        Args:
            url: Triton server URL
            model_name: Name of the model in Triton
            model_version: Model version
            timeout: Request timeout in seconds
        """
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.timeout = timeout
        
        self.client = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Triton server."""
        try:
            self.client = httpclient.InferenceServerClient(
                url=self.url,
                verbose=False
            )
            
            # Check if server is ready
            if not self.client.is_server_ready():
                raise RuntimeError("Triton server is not ready")
            
            # Check if model is ready
            if not self.client.is_model_ready(self.model_name, self.model_version):
                raise RuntimeError(f"Model {self.model_name} is not ready")
            
            logger.info(f"Connected to Triton server at {self.url}")
            logger.info(f"Model {self.model_name} (version {self.model_version}) is ready")
            
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
    
    def infer(
        self,
        image: Image.Image,
        input_name: str = "input",
        output_name: str = "output"
    ) -> Image.Image:
        """
        Run inference via Triton.
        
        Args:
            image: Input PIL Image
            input_name: Name of the input tensor in Triton
            output_name: Name of the output tensor in Triton
        
        Returns:
            Enhanced PIL Image
        """
        if self.client is None:
            raise RuntimeError("Triton client not connected")
        
        # Get model metadata
        model_metadata = self.client.get_model_metadata(self.model_name, self.model_version)
        
        # Preprocess image
        input_data = self._preprocess(image, model_metadata, input_name)
        
        # Prepare inputs
        inputs = []
        for input_meta in model_metadata['inputs']:
            if input_meta['name'] == input_name:
                inputs.append(
                    httpclient.InferInput(
                        input_name,
                        input_data.shape,
                        triton_to_np_dtype(input_meta['datatype'])
                    )
                )
                inputs[-1].set_data_from_numpy(input_data)
        
        # Prepare outputs
        outputs = []
        for output_meta in model_metadata['outputs']:
            outputs.append(httpclient.InferRequestedOutput(output_meta['name']))
        
        # Run inference
        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
            request_id="",
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
            priority=0,
            timeout=self.timeout
        )
        
        # Get output
        output_data = response.as_numpy(output_name)
        
        # Postprocess
        output_image = self._postprocess(output_data, image.size)
        
        return output_image
    
    def _preprocess(
        self,
        image: Image.Image,
        model_metadata: Dict[str, Any],
        input_name: str
    ) -> np.ndarray:
        """Preprocess image for Triton input."""
        # Get input shape from metadata
        input_shape = None
        for input_meta in model_metadata['inputs']:
            if input_meta['name'] == input_name:
                input_shape = input_meta['shape']
                break
        
        if input_shape is None:
            raise ValueError(f"Input {input_name} not found in model metadata")
        
        # Resize image to match input shape
        # Assuming shape is [batch, channels, height, width]
        if len(input_shape) == 4:
            _, _, h, w = input_shape
            image = image.resize((w, h), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Convert to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # CHW -> BCHW
        
        return img_array
    
    def _postprocess(
        self,
        output_data: np.ndarray,
        original_size: tuple
    ) -> Image.Image:
        """Postprocess Triton output to PIL Image."""
        # Remove batch dimension if present
        if len(output_data.shape) == 4:
            output_data = output_data[0]  # Remove batch dimension
        
        # Convert from CHW to HWC
        if len(output_data.shape) == 3:
            output_data = np.transpose(output_data, (1, 2, 0))  # CHW -> HWC
        
        # Clamp to [0, 1] and convert to uint8
        output_data = np.clip(output_data, 0, 1)
        output_data = (output_data * 255).astype(np.uint8)
        
        # Create PIL Image
        image = Image.fromarray(output_data)
        
        # Resize to original size
        image = image.resize(original_size, Image.LANCZOS)
        
        return image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Triton model."""
        if self.client is None:
            return {"connected": False}
        
        try:
            model_metadata = self.client.get_model_metadata(self.model_name, self.model_version)
            model_config = self.client.get_model_config(self.model_name, self.model_version)
            
            return {
                "connected": True,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "inputs": model_metadata.get("inputs", []),
                "outputs": model_metadata.get("outputs", []),
                "config": model_config
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"connected": False, "error": str(e)}

