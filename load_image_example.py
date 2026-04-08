#!/usr/bin/env python3
"""
Example: How to load actual images for CountGD model
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

# Method 1: Basic image loading
def load_image_basic(image_path):
    """Load image and convert to tensor"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension

# Method 2: With normalization (recommended for models)
def load_image_normalized(image_path):
    """Load image with proper normalization"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

# Method 3: Using CountGD model directly
def test_with_real_image(image_path):
    """Test CountGD model with real image"""
    from countgd_model import LightweightCountGD
    
    # Load model
    model = LightweightCountGD()
    model.eval()
    
    # Load image
    image_tensor = load_image_normalized(image_path)
    text_prompt = ["Count the objects in this image"]
    
    # Run inference
    with torch.no_grad():
        count, heatmap = model(image_tensor, text_prompt)
    
    print(f"Predicted count: {count.item():.2f}")
    return count, heatmap

if __name__ == "__main__":
    # Example usage
    image_path = "./data/images/train_0000.jpg"  # Use any image path
    
    try:
        # Load image
        tensor = load_image_normalized(image_path)
        print(f"Image loaded successfully!")
        print(f"Shape: {tensor.shape}")  # Should be (1, 3, 224, 224)
        
        # Test with model
        count, heatmap = test_with_real_image(image_path)
        print(f"Model prediction: {count.item():.2f}")
        
    except FileNotFoundError:
        print(f"Image not found at {image_path}")
        print("Run 'python countgd_dataset.py' first to create sample images")
