#!/usr/bin/env python3
"""Create and test different CountGD model sizes"""

import torch
from countgd_model import create_countgd_model, get_model_size_mb

def create_all_models():
    """Create all model variants and show their specs"""
    
    sizes = ['full', 'small', 'tiny', 'micro']
    models = {}
    
    print("Creating CountGD Models:")
    print("=" * 60)
    
    for size in sizes:
        try:
            model = create_countgd_model(size)
            models[size] = model
            
            size_mb = get_model_size_mb(model)
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"{size.upper():>6}: {size_mb:>6.1f} MB | {param_count:>10,} params")
            
        except Exception as e:
            print(f"{size.upper():>6}: ❌ Error - {e}")
    
    return models

def test_inference(models):
    """Test inference on all models"""
    
    print(f"\nInference Test:")
    print("-" * 40)
    
    # Test data
    dummy_image = torch.randn(1, 3, 224, 224)
    test_prompts = ["count red cars", "find dogs"]
    
    for size, model in models.items():
        try:
            model.eval()
            
            with torch.no_grad():
                for prompt in test_prompts:
                    if hasattr(model, 'inference_with_explanation'):
                        # Models with explanation capability
                        count, heatmap, explanations = model.inference_with_explanation(
                            dummy_image, [prompt]
                        )
                        print(f"{size.upper():>6} | {prompt}: {count.item():.2f}")
                    else:
                        # Simple models
                        count, heatmap = model(dummy_image, [prompt])
                        print(f"{size.upper():>6} | {prompt}: {count.item():.2f}")
                        
        except Exception as e:
            print(f"{size.upper():>6} | Error: {e}")

def save_models(models):
    """Save all models to disk"""
    
    print(f"\nSaving Models:")
    print("-" * 30)
    
    for size, model in models.items():
        try:
            save_path = f"countgd_{size}_untrained.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ {size.upper():>6}: {save_path}")
        except Exception as e:
            print(f"❌ {size.upper():>6}: {e}")

if __name__ == "__main__":
    # Create all models
    models = create_all_models()
    
    # Test inference
    test_inference(models)
    
    # Save untrained models
    save_models(models)
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Train models: python train_all_models.py --train-all")
    print(f"   2. Train single: python train_all_models.py --size tiny")
    print(f"   3. Test trained:  python main.py --mode explain")
