#!/usr/bin/env python3
"""
Compare original CountGD model with MLE-STAR optimized version
"""

import torch
import time
from countgd_model import LightweightCountGD
from countgd_model_optimized import OptimizedCountGD, get_model_size, benchmark_model

def compare_models():
    """Compare original vs MLE-STAR optimized models"""
    print("CountGD Model Comparison: Original vs MLE-STAR Optimized")
    print("=" * 60)
    
    # Load models
    print("Loading models...")
    original_model = LightweightCountGD()
    optimized_model = OptimizedCountGD()
    
    # Model statistics
    print("\n📊 Model Statistics:")
    print("-" * 30)
    
    orig_params = sum(p.numel() for p in original_model.parameters())
    opt_params = sum(p.numel() for p in optimized_model.parameters())
    
    orig_size = get_model_size(original_model)
    opt_size = get_model_size(optimized_model)
    
    print(f"Original Model:")
    print(f"  Parameters: {orig_params:,}")
    print(f"  Size: {orig_size:.2f} MB")
    
    print(f"\nMLE-STAR Optimized:")
    print(f"  Parameters: {opt_params:,}")
    print(f"  Size: {opt_size:.2f} MB")
    
    print(f"\nImprovement:")
    param_reduction = (orig_params - opt_params) / orig_params * 100
    size_reduction = (orig_size - opt_size) / orig_size * 100
    
    if param_reduction > 0:
        print(f"  Parameter reduction: {param_reduction:.1f}%")
    else:
        print(f"  Parameter increase: {abs(param_reduction):.1f}%")
        
    if size_reduction > 0:
        print(f"  Size reduction: {size_reduction:.1f}%")
    else:
        print(f"  Size increase: {abs(size_reduction):.1f}%")
    
    # Performance benchmark
    print("\n⚡ Performance Benchmark:")
    print("-" * 30)
    
    print("Original Model:")
    orig_time = benchmark_original_model(original_model)
    
    print("\nMLE-STAR Optimized:")
    opt_time = benchmark_model(optimized_model)
    
    speedup = orig_time / opt_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Feature comparison
    print("\n🚀 MLE-STAR Enhancements:")
    print("-" * 30)
    print("✓ Optimized MobileViT-XXS backbone")
    print("✓ Channel attention mechanism")
    print("✓ Cross-attention fusion layers")
    print("✓ Enhanced counting head with auxiliary loss")
    print("✓ Mobile deployment utilities (ONNX export)")
    print("✓ Quantization-aware training support")
    print("✓ Performance benchmarking tools")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 30)
    print("1. Use optimized model for mobile deployment")
    print("2. Export to ONNX for production inference")
    print("3. Apply quantization for further size reduction")
    print("4. Benchmark on target mobile devices")

def benchmark_original_model(model, num_runs=100):
    """Benchmark original model (simplified version)"""
    model.eval()
    dummy_images = torch.randn(1, 3, 224, 224)
    dummy_text = "count objects"
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_images, dummy_text)
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_images, dummy_text)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.1f}")
    
    return avg_time

if __name__ == "__main__":
    compare_models()
