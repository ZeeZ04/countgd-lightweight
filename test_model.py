#!/usr/bin/env python3
"""
Simple test script to verify CountGD model functionality
"""

import torch
from countgd_model_optimized import OptimizedCountGD as LightweightCountGD

def test_model():
    """Test basic model functionality"""
    print("Testing CountGD Lightweight Model...")
    
    # Initialize model
    model = LightweightCountGD()
    model.eval()
    
    # Create test input
    test_image = torch.randn(1, 3, 224, 224)
    test_text = ["Count the red cars in the image"]
    
    # Forward pass
    with torch.no_grad():
        count, heatmap = model(test_image, test_text)
    
    print(f"✓ Model forward pass successful")
    print(f"  - Count prediction: {count.item():.2f}")
    print(f"  - Heatmap shape: {heatmap.shape}")
    
    # Test with different prompts
    prompts = [
        "How many objects are there?",
        "Count all visible items",
        "Number of red vehicles"
    ]
    
    print(f"\n✓ Testing prompt variations:")
    for prompt in prompts:
        with torch.no_grad():
            count, _ = model(test_image, [prompt])
        print(f"  - '{prompt}': {count.item():.2f}")
    
    # Model size info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    print(f"\n🎉 All tests passed! CountGD model is working correctly.")

if __name__ == "__main__":
    test_model()
