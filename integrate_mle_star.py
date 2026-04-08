#!/usr/bin/env python3
"""
Integration script to apply MLE-STAR optimizations to CountGD project
"""

import os
import sys
import shutil
from mle_star import MLEStarAgent

def backup_original_model():
    """Backup original model before optimization"""
    if os.path.exists("countgd_model.py"):
        shutil.copy("countgd_model.py", "countgd_model_original.py")
        print("✓ Original model backed up to countgd_model_original.py")

def integrate_optimizations():
    """Apply MLE-STAR optimizations to CountGD"""
    print("🚀 Starting MLE-STAR integration with CountGD...")
    
    # Backup original
    backup_original_model()
    
    # Run MLE-STAR optimization
    agent = MLEStarAgent()
    optimized_path = agent.save_optimized_model("countgd_model_optimized.py")
    
    # Create integration summary
    summary = f"""
# MLE-STAR Integration Summary

## Optimizations Applied:
1. **Model Architecture**: Optimized MobileViT configuration based on performance evaluation
2. **Efficient Fusion**: Improved multimodal fusion layers
3. **Mobile Optimizations**: Added quantization support and ONNX export
4. **Benchmarking**: Included performance measurement utilities

## Files Created:
- `countgd_model_optimized.py`: MLE-STAR optimized model
- `countgd_model_original.py`: Backup of original model

## Next Steps:
1. Test the optimized model:
   ```bash
   python countgd_model_optimized.py
   ```

2. Update your training script to use the optimized model:
   ```python
   from countgd_model_optimized import OptimizedCountGD
   model = OptimizedCountGD()
   ```

3. Run benchmarks:
   ```python
   from countgd_model_optimized import benchmark_model
   benchmark_model(model)
   ```

4. Export for mobile deployment:
   ```python
   from countgd_model_optimized import export_to_onnx
   export_to_onnx(model, "countgd_mobile.onnx")
   ```

## Performance Improvements Expected:
- Reduced model size for mobile deployment
- Faster inference through optimized architecture
- Better accuracy through improved fusion layers
- Mobile-ready with quantization support
"""
    
    with open("MLE_STAR_INTEGRATION.md", "w") as f:
        f.write(summary)
    
    print("✓ MLE-STAR integration complete!")
    print("📋 See MLE_STAR_INTEGRATION.md for details")
    
    return optimized_path

def test_integration():
    """Test the integrated optimizations"""
    print("\n🧪 Testing MLE-STAR optimizations...")
    
    try:
        # Test import
        sys.path.insert(0, ".")
        from countgd_model_optimized import OptimizedCountGD, get_model_size
        
        # Create model
        model = OptimizedCountGD()
        
        # Check model size
        size_mb = get_model_size(model)
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Model created successfully")
        print(f"  - Size: {size_mb:.2f} MB")
        print(f"  - Parameters: {param_count:,}")
        
        # Test forward pass
        import torch
        dummy_images = torch.randn(1, 3, 224, 224)
        dummy_text = ["count the objects"]
        
        with torch.no_grad():
            counts, heatmaps = model(dummy_images, dummy_text)
            
        print(f"✓ Forward pass successful")
        print(f"  - Count output shape: {counts.shape}")
        print(f"  - Heatmap output shape: {heatmaps.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main integration workflow"""
    print("MLE-STAR Integration for CountGD")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("countgd_model.py"):
        print("❌ countgd_model.py not found. Please run from CountGD project directory.")
        return
    
    # Run integration
    optimized_path = integrate_optimizations()
    
    # Test integration
    if test_integration():
        print("\n🎉 MLE-STAR integration successful!")
        print("\nRecommended next steps:")
        print("1. Review the optimized model code")
        print("2. Update your training scripts")
        print("3. Run performance benchmarks")
        print("4. Export to ONNX for mobile deployment")
    else:
        print("\n⚠️  Integration completed but tests failed.")
        print("Please check the generated code and fix any issues.")

if __name__ == "__main__":
    main()
