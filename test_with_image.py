#!/usr/bin/env python3
"""
Test CountGD model with real images via command line
"""

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

def load_image(image_path):
    """Load and preprocess image for model input"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def load_model(model_path=None, model_type='full'):
    """Load appropriate model based on type"""
    if model_type == 'tiny':
        from countgd_model_tiny import TinyCountGD
        model = TinyCountGD()
    else:
        from countgd_model_optimized import OptimizedCountGD as LightweightCountGD
        model = LightweightCountGD()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading weights: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("✓ Weights loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load weights ({e})")
            print("   Using untrained model instead")
    
    return model

def test_model_with_image(image_path, prompt, model_path=None, model_type='full'):
    """Test model with real image and prompt"""
    print(f"Loading image: {image_path}")
    image_tensor = load_image(image_path)
    
    print(f"Initializing {model_type} model...")
    model = load_model(model_path, model_type)
    model.eval()
    
    print(f"Running inference with prompt: '{prompt}'")
    with torch.no_grad():
        count, heatmap = model(image_tensor, [prompt])
    
    print(f"\n✓ Results:")
    print(f"  - Predicted count: {count.item():.2f}")
    print(f"  - Heatmap shape: {heatmap.shape}")
    print(f"  - Image shape: {image_tensor.shape}")
    
    return count.item(), heatmap

def main():
    parser = argparse.ArgumentParser(description='Test CountGD model with real images')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--prompt', default='Count the objects in this image', 
                       help='Text prompt for counting (default: "Count the objects in this image")')
    parser.add_argument('--model', help='Path to trained model weights (optional)')
    parser.add_argument('--type', choices=['full', 'tiny'], default='full',
                       help='Model type: full (default) or tiny')
    
    args = parser.parse_args()
    
    try:
        count, heatmap = test_model_with_image(args.image, args.prompt, args.model, args.type)
        print(f"\n🎉 Test completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
