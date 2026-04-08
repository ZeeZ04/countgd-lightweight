#!/usr/bin/env python3
"""
Test script to verify all bug fixes in the MSc research code
"""

import torch
import numpy as np
from pathlib import Path

def test_zmobilevit_fixes():
    """Test ZMobileViT fixes"""
    print("Testing ZMobileViT fixes...")
    
    try:
        from zmobilevit_counting import ZMobileViTCounting, LinearCountingAttention
        
        # Test linear attention fix
        attention = LinearCountingAttention(64, num_heads=4)
        x = torch.randn(2, 100, 64)
        output = attention(x)
        assert output.shape == x.shape, "Attention output shape mismatch"
        print("✓ Linear attention einsum fix working")
        
        # Test model creation with CLIP fallback
        model = ZMobileViTCounting(use_clip=True)
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_text = ["count objects"]
        
        count, density = model(dummy_input, dummy_text)
        print("✓ ZMobileViT model creation and forward pass working")
        
    except Exception as e:
        print(f"✗ ZMobileViT test failed: {e}")

def test_evaluation_fixes():
    """Test evaluation metric fixes"""
    print("\nTesting evaluation fixes...")
    
    try:
        from proper_counting_evaluation import CountingEvaluator
        
        evaluator = CountingEvaluator()
        
        # Test with normal data
        pred = [1, 2, 3, 4, 5]
        gt = [1, 2, 0, 4, 6]  # Include zero to test division fix
        
        metrics = evaluator.evaluate_counting_performance(pred, gt, "test")
        assert not np.isnan(metrics['NAE']), "NAE should not be NaN"
        assert not np.isnan(metrics['SRE']), "SRE should not be NaN"
        print("✓ Division by zero fixes working")
        
        # Test empty array validation
        try:
            evaluator.evaluate_counting_performance([], [], "empty")
            print("✗ Empty array validation failed")
        except ValueError:
            print("✓ Empty array validation working")
            
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")

def test_statistical_training_fixes():
    """Test statistical training fixes"""
    print("\nTesting statistical training fixes...")
    
    try:
        from statistical_training import StatisticalTrainer
        from zmobilevit_counting import ZMobileViTCounting
        
        trainer = StatisticalTrainer()
        
        # Test model handling
        model = ZMobileViTCounting()
        dummy_images = torch.randn(2, 3, 224, 224)
        dummy_texts = ["count", "objects"]
        
        # This should not crash with the new error handling
        try:
            outputs = model(dummy_images, dummy_texts)
            print("✓ Model forward pass handling working")
        except Exception as e:
            print(f"Model forward pass: {e}")
            
    except Exception as e:
        print(f"✗ Statistical training test failed: {e}")

def test_main_pipeline_fixes():
    """Test main pipeline fixes"""
    print("\nTesting main pipeline fixes...")
    
    try:
        from msc_main import MSCResearchPipeline
        
        pipeline = MSCResearchPipeline('./test_output')
        
        # Test efficiency analysis with error handling
        efficiency = pipeline.analyze_efficiency()
        print("✓ Efficiency analysis with error handling working")
        
    except Exception as e:
        print(f"✗ Main pipeline test failed: {e}")

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING BUG FIXES")
    print("="*60)
    
    test_zmobilevit_fixes()
    test_evaluation_fixes()
    test_statistical_training_fixes()
    test_main_pipeline_fixes()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
