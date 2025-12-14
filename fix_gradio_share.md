# Fix Gradio Public Link (share=True)

## Issue: share=True not creating public link

If `share=True` isn't working, here are solutions:

## Solution 1: Authenticate with Hugging Face

Gradio needs Hugging Face authentication for public links:

```bash
# Install/upgrade gradio
pip install --upgrade gradio

# Login to Hugging Face (optional but recommended)
huggingface-cli login
```

Or set token as environment variable:
```bash
export HF_TOKEN=your_huggingface_token
```

## Solution 2: Check Network/Firewall

Public links require outbound internet connection:
- Check firewall settings
- Ensure port 7860 is accessible
- Try from different network

## Solution 3: Use Alternative Methods

### Option A: Use ngrok for tunneling
```bash
# Install ngrok
# Then run:
ngrok http 7860

# Use the ngrok URL
```

### Option B: Use localtunnel
```bash
npm install -g localtunnel
lt --port 7860
```

### Option C: Deploy to Hugging Face Spaces
Upload your code to Hugging Face Spaces for automatic public hosting.

## Solution 4: Check Gradio Version

```bash
pip install --upgrade gradio
```

Older versions may have issues with `share=True`.

## Solution 5: Manual Configuration

If share=True fails, you can manually expose:

```python
demo.launch(
    server_name="0.0.0.0",  # Listen on all interfaces
    server_port=7860,
    share=False,  # Disable Gradio share
    # Then use port forwarding or VPN
)
```

## Solution 6: Check Error Messages

Run with verbose output:
```python
demo.launch(
    share=True,
    show_error=True,
    quiet=False,
)
```

Check console for specific error messages.

## Quick Test

Run this to test if share works:
```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(share=True)
```

If this works, the issue is with your specific setup.

## Alternative: Use Streamlit with ngrok

If Gradio share doesn't work, use Streamlit + ngrok:
```bash
# Terminal 1: Run Streamlit
streamlit run streamlit_inference.py --server.port 8501

# Terminal 2: Create tunnel
ngrok http 8501
```

## Most Common Fix

Usually just updating Gradio fixes it:
```bash
pip install --upgrade gradio
python demo.py
```

The public link will appear in the console output.

