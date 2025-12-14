#!/usr/bin/env python3
"""
Simple test script for the API server.
"""

import requests
import base64
import sys
from pathlib import Path

def test_health(host="http://localhost:8000"):
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{host}/health/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info(host="http://localhost:8000"):
    """Test model info endpoint."""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{host}/health/model")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_inference(host="http://localhost:8000", image_path=None):
    """Test inference endpoint."""
    if image_path is None:
        print("\nSkipping inference test (no image provided)")
        return True
    
    print(f"\nTesting inference endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Send request
        response = requests.post(
            f"{host}/inference/",
            json={
                "image_base64": image_base64,
                "use_refiner": True,
                "preserve_resolution": True
            },
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        
        if result.get("success"):
            print(f"Inference time: {result.get('inference_time', 0):.2f}s")
            print("Inference successful!")
            return True
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*60)
    print("API Server Test Suite")
    print("="*60)
    print(f"Testing server at: {host}\n")
    
    results = []
    results.append(("Health Check", test_health(host)))
    results.append(("Model Info", test_model_info(host)))
    results.append(("Inference", test_inference(host, image_path)))
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

