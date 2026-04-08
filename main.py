#!/usr/bin/env python3
"""
CountGD: Lightweight Vision-Language Model for Object Counting and Grounding
Main execution script for the research project
"""

import torch
import argparse
from countgd_model_optimized import OptimizedCountGD as LightweightCountGD
from countgd_model import train_model, export_to_onnx
from countgd_dataset import get_countgd_dataloaders, download_countgd_dataset
from experiments import ExperimentRunner

def test_explanation_mode():
    """Test explanation capability with sample prompts"""
    print("Testing explanation capability...")
    print("=" * 50)
    
    model = LightweightCountGD()
    dummy_image = torch.randn(1, 3, 224, 224)
    
    test_prompts = [
        "Count the red cars in the image",
        "How many people are walking?", 
        "Find all the dogs in the scene"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting: '{prompt}'")
        count, heatmap, explanations = model.inference_with_explanation(dummy_image, [prompt])

def main():
    parser = argparse.ArgumentParser(description='CountGD Research Project')
    parser.add_argument('--mode', choices=['train', 'eval', 'export', 'experiment', 'explain'], 
                       default='experiment', help='Execution mode')
    parser.add_argument('--model-path', type=str, help='Path to saved model')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode == 'explain':
        test_explanation_mode()
        
    elif args.mode == 'train':
        print("Training lightweight CountGD model...")
        
        # Download and setup dataset
        data_dir = download_countgd_dataset(args.data_path)
        
        # Initialize model and dataset
        model = LightweightCountGD()
        train_loader, val_loader = get_countgd_dataloaders(data_dir, batch_size=8)
        
        # Train model
        train_model(model, train_loader, num_epochs=args.epochs)
        
        # Save model
        torch.save(model.state_dict(), 'countgd_lightweight.pth')
        print("Model saved as 'countgd_lightweight.pth'")
        
    elif args.mode == 'eval':
        print("Evaluating model performance...")
        
        runner = ExperimentRunner(args.model_path)
        results = runner.run_full_evaluation()
        
        print("Evaluation completed. Results saved.")
        
    elif args.mode == 'export':
        print("Exporting model to ONNX...")
        
        model = LightweightCountGD()
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path))
        
        dummy_input = (torch.randn(1, 3, 224, 224), ["Count the objects"])
        export_to_onnx(model, dummy_input, 'countgd_model.onnx')
        
        print("Model exported to 'countgd_model.onnx'")
        
    elif args.mode == 'experiment':
        print("Running comprehensive experiments...")
        
        runner = ExperimentRunner(args.model_path)
        results = runner.run_full_evaluation()
        
        print("\n" + "="*50)
        print("COUNTGD PROJECT RESULTS SUMMARY")
        print("="*50)
        
        # Performance summary
        perf = results['performance']
        print(f"\nPERFORMANCE METRICS:")
        print(f"- Average inference time: {torch.tensor(perf['inference_times']).mean():.4f}s")
        print(f"- Memory usage: {torch.tensor(perf['memory_usage']).mean():.2f} MB")
        print(f"- Mean Absolute Error: {torch.tensor(perf['mae_scores']).mean():.4f}")
        print(f"- Intersection over Union: {torch.tensor(perf['iou_scores']).mean():.4f}")
        
        # Model info
        info = results['model_info']
        print(f"\nMODEL CHARACTERISTICS:")
        print(f"- Total parameters: {info['total_params']:,}")
        print(f"- Model size: {info['model_size_mb']:.2f} MB")
        
        # Robustness
        robust = results['robustness']
        print(f"\nROBUSTNESS TO NOISE:")
        for noise_type, degradations in robust.items():
            print(f"- {noise_type.capitalize()}: {torch.tensor(degradations).mean():.4f} degradation")
        
        print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
