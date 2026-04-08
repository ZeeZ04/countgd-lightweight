#!/usr/bin/env python3
"""Test script for explanation capabilities"""

import torch
from countgd_model import LightweightCountGD
import numpy as np

def test_explanation_capability():
    """Test the explanation functionality"""
    
    # Initialize model
    model = LightweightCountGD()
    model.eval()
    
    # Create dummy data
    dummy_image = torch.randn(1, 3, 224, 224)
    test_prompts = [
        "Count the red cars in the image",
        "How many people are walking?",
        "Find all the dogs in the scene"
    ]
    
    print("Testing explanation capability...")
    print("=" * 50)
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        print("-" * 40)
        
        # Run inference with explanations
        count, heatmap, explanations = model.inference_with_explanation(
            dummy_image, [prompt]
        )
        
        print(f"Final count prediction: {count.item():.2f}")
        print(f"Heatmap shape: {heatmap.shape}")

if __name__ == "__main__":
    test_explanation_capability()
