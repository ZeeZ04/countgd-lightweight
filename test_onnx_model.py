#!/usr/bin/env python3
"""
Test script for CountGD ONNX model
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def test_onnx_model():
    """Test ONNX model inference"""
    print("Testing CountGD ONNX Model...")
    
    # Load ONNX model
    session = ort.InferenceSession("countgd_model.onnx")
    
    # Get input/output info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"✓ Model loaded - Inputs: {input_names}, Outputs: {output_names}")
    
    # Prepare test data
    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    outputs = session.run(output_names, {"image": image})
    count, heatmap = outputs
    
    print(f"✓ Inference successful")
    print(f"  - Count: {count[0][0]:.2f}")
    print(f"  - Heatmap shape: {heatmap.shape}")
    
    # Test with sample image if available
    try:
        img = Image.open("data/images/train_0000.jpg").convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).numpy()
        
        outputs = session.run(output_names, {"image": img_tensor})
        count, _ = outputs
        print(f"✓ Real image test - Count: {count[0][0]:.2f}")
        
    except FileNotFoundError:
        print("✓ No sample image found, using synthetic data only")
    
    print("🎉 ONNX model test completed!")

if __name__ == "__main__":
    test_onnx_model()
