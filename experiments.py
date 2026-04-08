import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from countgd_model_optimized import OptimizedCountGD as LightweightCountGD
from countgd_model import ModelEvaluator, NoiseAugmentation
from countgd_dataset import get_countgd_dataloaders, download_countgd_dataset
import time
import psutil
import os

class ExperimentRunner:
    """Run comprehensive experiments for the CountGD project"""
    
    def __init__(self, model_path=None):
        self.model = LightweightCountGD()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.evaluator = ModelEvaluator()
        self.noise_aug = NoiseAugmentation()
        
    def benchmark_performance(self, test_images, test_texts, test_counts, test_heatmaps, device='cpu'):
        """Benchmark model performance metrics"""
        self.model.to(device)
        self.model.eval()
        
        results = {
            'inference_times': [],
            'memory_usage': [],
            'mae_scores': [],
            'iou_scores': []
        }
        
        with torch.no_grad():
            for img, text, true_count, true_heatmap in zip(test_images, test_texts, test_counts, test_heatmaps):
                # Memory before
                mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                
                # Inference timing
                start_time = time.time()
                pred_count, pred_heatmap = self.model(img.unsqueeze(0), [text])
                end_time = time.time()
                
                # Memory after
                mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                
                # Calculate metrics
                mae = self.evaluator.mean_absolute_error(pred_count, torch.tensor([[true_count]]))
                iou = self.evaluator.intersection_over_union(pred_heatmap.squeeze(), true_heatmap)
                
                results['inference_times'].append(end_time - start_time)
                results['memory_usage'].append(mem_after - mem_before)
                results['mae_scores'].append(mae.item())
                results['iou_scores'].append(iou.item())
        
        return results
    
    def test_prompt_variations(self, image, base_prompts):
        """Test model sensitivity to different prompt formulations"""
        variations = {
            'general': ["How many objects?", "Count objects", "Number of items"],
            'specific': ["Count the red cars", "How many red vehicles are there?", "Number of red automobiles"],
            'detailed': ["Count all the red cars visible in the image", "How many red colored cars can you see?"]
        }
        
        results = {}
        self.model.eval()
        
        with torch.no_grad():
            for category, prompts in variations.items():
                category_results = []
                for prompt in prompts:
                    count, _ = self.model(image.unsqueeze(0), [prompt])
                    category_results.append(count.item())
                results[category] = category_results
        
        return results
    
    def test_robustness_to_noise(self, test_images, test_texts):
        """Test model robustness to visual noise"""
        noise_types = ['blur', 'brightness', 'occlusion']
        results = {}
        
        for noise_type in noise_types:
            noise_results = []
            
            for img_tensor, text in zip(test_images, test_texts):
                # Convert to numpy for noise application
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # Apply noise
                if noise_type == 'blur':
                    noisy_img = self.noise_aug.add_gaussian_blur(img_np)
                elif noise_type == 'brightness':
                    noisy_img = self.noise_aug.adjust_brightness(img_np, factor=1.5)
                elif noise_type == 'occlusion':
                    noisy_img = self.noise_aug.add_occlusion(img_np)
                
                # Convert back to tensor
                noisy_tensor = torch.from_numpy(noisy_img).permute(2, 0, 1).float() / 255.0
                
                # Get predictions
                with torch.no_grad():
                    original_count, _ = self.model(img_tensor.unsqueeze(0), [text])
                    noisy_count, _ = self.model(noisy_tensor.unsqueeze(0), [text])
                
                # Calculate degradation
                degradation = abs(original_count.item() - noisy_count.item())
                noise_results.append(degradation)
            
            results[noise_type] = noise_results
        
        return results
    
    def generate_gradcam_visualization(self, image, text, target_layer=None):
        """Generate Grad-CAM++ visualization"""
        if target_layer is None:
            target_layer = self.model.vision_encoder.blocks[-1]
        
        cam = GradCAM(model=self.model, target_layers=[target_layer])
        
        # Generate heatmap
        grayscale_cam = cam(input_tensor=image.unsqueeze(0))
        
        return grayscale_cam[0]
    
    def compare_model_sizes(self):
        """Compare model sizes and parameters"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Model size estimation
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    def run_full_evaluation(self, data_dir='./data'):
        """Run complete evaluation suite"""
        print("Starting comprehensive evaluation...")
        
        # Setup real dataset
        download_countgd_dataset(data_dir)
        train_loader, val_loader = get_countgd_dataloaders(data_dir, batch_size=4)
        
        # Get real test data
        test_images, test_texts, test_counts, test_heatmaps = [], [], [], []
        for batch_idx, (images, texts, counts, heatmaps) in enumerate(val_loader):
            test_images.extend([img for img in images])
            test_texts.extend(texts)
            test_counts.extend([count.item() for count in counts])
            test_heatmaps.extend([hm for hm in heatmaps])
            if batch_idx >= 2:  # Use first 3 batches for testing
                break
        
        # 1. Performance benchmarking
        print("\n1. Performance Benchmarking...")
        perf_results = self.benchmark_performance(test_images, test_texts, test_counts, test_heatmaps)
        
        print(f"Average inference time: {np.mean(perf_results['inference_times']):.4f}s")
        print(f"Average MAE: {np.mean(perf_results['mae_scores']):.4f}")
        print(f"Average IoU: {np.mean(perf_results['iou_scores']):.4f}")
        
        # 2. Prompt variation testing
        print("\n2. Prompt Variation Testing...")
        prompt_results = self.test_prompt_variations(test_images[0], test_texts)
        for category, results in prompt_results.items():
            print(f"{category}: {results}")
        
        # 3. Robustness testing
        print("\n3. Robustness to Noise Testing...")
        robustness_results = self.test_robustness_to_noise(test_images[:3], test_texts[:3])
        for noise_type, degradations in robustness_results.items():
            print(f"{noise_type} degradation: {np.mean(degradations):.4f}")
        
        # 4. Model size analysis
        print("\n4. Model Size Analysis...")
        size_info = self.compare_model_sizes()
        print(f"Total parameters: {size_info['total_params']:,}")
        print(f"Model size: {size_info['model_size_mb']:.2f} MB")
        
        return {
            'performance': perf_results,
            'prompt_variations': prompt_results,
            'robustness': robustness_results,
            'model_info': size_info
        }

if __name__ == "__main__":
    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_full_evaluation()
    
    # Save results
    torch.save(results, 'experiment_results.pt')
    print("\nExperiment results saved to 'experiment_results.pt'")
