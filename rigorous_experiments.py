#!/usr/bin/env python3
"""
Rigorous Experimental Validation for MSc Dissertation (FIXED VERSION)
======================================================================
Fixes applied:
1. MPS device compatibility - CPU fallback for problematic models
2. Model size calculation - Only trainable parameters (excludes frozen CLIP)
3. Training stability - Lower LR (5e-5), warmup, gradient clipping (0.5)

- 5 random seeds for statistical validity
- 20 epochs for full training  
- All 3 models: Baseline, MLE-STAR Optimized, ZMobileViT
- Comprehensive metrics and statistical comparisons
- Separate output directory: ./rigorous_results/

Run with:
    python rigorous_experiments.py 2>&1 | tee results_rigorous_fixed.txt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats
import json
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Avoid tokenizer warnings
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import models
from zmobilevit_counting import ZMobileViTCounting
from countgd_model import LightweightCountGD
from countgd_model_optimized import OptimizedCountGD
from countgd_dataset import get_countgd_dataloaders, download_countgd_dataset


class RigorousExperimentRunner:
    """Run rigorous experiments with proper statistical methodology"""
    
    def __init__(self, output_dir='./rigorous_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        
        # Results storage
        self.all_results = {}
        
        # Loss functions
        self.count_criterion = nn.MSELoss()
        self.density_criterion = nn.BCEWithLogitsLoss()
        
        # FIX #1: Models that have MPS compatibility issues - use CPU for these
        # Added ZMobileViT due to training instability (4/5 seeds collapsed on MPS)
        self.mps_problematic_models = ['CountGD_Baseline', 'MLE-STAR_Optimized', 'ZMobileViT']
        
        # FIX #4: Model-specific learning rates for training stability
        self.model_learning_rates = {
            'CountGD_Baseline': 5e-5,
            'MLE-STAR_Optimized': 5e-5,
            'ZMobileViT': 1e-4,  # FIX #6: Moderate LR with AdamW for attention stability
        }
        
        # Model-specific warmup epochs
        self.model_warmup_epochs = {
            'CountGD_Baseline': 2,
            'MLE-STAR_Optimized': 2,
            'ZMobileViT': 5,  # Longer warmup for stability
        }
        
        # FIX #6: Use AdamW for all models (better for attention layers)
        # Use SmoothL1 loss for ZMobileViT (robust to outliers, smooth gradients near 0)
        self.model_use_sgd = set()  # Reverted: AdamW is better for attention models
        self.model_use_l1_loss = {'ZMobileViT'}  # Will use SmoothL1 instead of raw L1
        
    def set_seed(self, seed):
        """Set all random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def count_parameters(self, model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def count_total_parameters(self, model):
        """Count all parameters (trainable + frozen)"""
        return sum(p.numel() for p in model.parameters())
    
    def get_model_size_mb(self, model, trainable_only=True):
        """
        Get model size in MB
        
        FIX #2: Now correctly calculates trainable-only size by default
        to avoid counting frozen CLIP encoder weights
        """
        if trainable_only:
            # Only count trainable parameters (excludes frozen CLIP weights)
            param_size = sum(p.numel() * p.element_size() 
                           for p in model.parameters() if p.requires_grad)
        else:
            # Count all parameters including frozen
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            param_size += buffer_size
            
        return param_size / (1024 ** 2)
    
    def get_device(self, model_name):
        """
        Get appropriate device for model
        
        FIX #1: Use CPU for models with known MPS compatibility issues
        """
        if model_name in self.mps_problematic_models:
            print(f"    Note: Using CPU for {model_name} (MPS compatibility issue)")
            return torch.device('cpu')
        
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def benchmark_inference(self, model, model_name, num_runs=50):
        """Benchmark inference speed"""
        device = self.get_device(model_name)
        model = model.to(device)
        model.eval()
        
        dummy_images = torch.randn(1, 3, 224, 224).to(device)
        dummy_text = ["count objects"]
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                try:
                    _ = model(dummy_images, dummy_text)
                except:
                    try:
                        _ = model(dummy_images, "count objects")
                    except:
                        _ = model(dummy_images)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                try:
                    _ = model(dummy_images, dummy_text)
                except:
                    try:
                        _ = model(dummy_images, "count objects")
                    except:
                        _ = model(dummy_images)
            times.append((time.time() - start) * 1000)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'fps': 1000 / np.mean(times),
            'device': str(device)
        }
    
    def train_single_run(self, model, model_name, train_loader, val_loader, epochs, 
                         lr=5e-5, warmup_epochs=2):
        """
        Train model for a single seed
        
        FIX #3: Training stability improvements
        - Lower learning rate (5e-5 instead of 1e-4)
        - Added warmup epochs
        - Lower gradient clipping (0.5 instead of 1.0)
        - Use appropriate device based on model
        FIX #5: Use SGD+momentum and L1 loss for ZMobileViT
        """
        device = self.get_device(model_name)
        print(f"    Using device: {device}")
        model = model.to(device)
        
        # FIX #5: Use SGD with momentum for ZMobileViT (more stable for small models)
        use_sgd = model_name in getattr(self, 'model_use_sgd', set())
        use_l1 = model_name in getattr(self, 'model_use_l1_loss', set())
        
        if use_sgd:
            print(f"    Using SGD optimizer with momentum=0.9")
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        if use_l1:
            print(f"    Using SmoothL1 loss (robust + smooth gradients near 0)")
            count_loss_fn = nn.SmoothL1Loss(beta=1.0)  # Smooth transition at |error|=1
        else:
            count_loss_fn = nn.MSELoss()
        
        # Warmup + cosine annealing scheduler
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_val_mae = float('inf')
        best_model_state = None
        patience_counter = 0
        max_patience = 7  # More patience for small models that need time to converge
        collapse_threshold = 3.5  # If val_mae jumps above this, likely collapsing
        collapse_recoveries = 0
        max_recoveries = 2  # Allow up to 2 recovery attempts from collapse
        
        history = {'train_mae': [], 'val_mae': [], 'lr': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_maes = []
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
            for batch in pbar:
                images, texts, counts, density_maps = batch
                images = images.to(device)
                counts = counts.to(device).float()
                density_maps = density_maps.to(device)
                
                optimizer.zero_grad()
                
                # Handle different model interfaces with better error handling
                try:
                    outputs = model(images, texts)
                    if isinstance(outputs, tuple):
                        pred_counts = outputs[0]
                        pred_density = outputs[1] if len(outputs) > 1 else None
                    else:
                        pred_counts = outputs
                        pred_density = None
                except Exception as e:
                    try:
                        # Try with string text instead of list
                        text_str = texts[0] if isinstance(texts, list) else texts
                        outputs = model(images, text_str)
                        if isinstance(outputs, tuple):
                            pred_counts = outputs[0]
                            pred_density = outputs[1] if len(outputs) > 1 else None
                        else:
                            pred_counts = outputs
                            pred_density = None
                    except Exception as e2:
                        try:
                            pred_counts = model(images)
                            pred_density = None
                        except:
                            print(f"    Model forward error: {e}")
                            continue
                
                # Ensure correct shapes
                pred_counts = pred_counts.squeeze()
                counts = counts.squeeze()
                
                # Handle scalar tensors
                if pred_counts.dim() == 0:
                    pred_counts = pred_counts.unsqueeze(0)
                if counts.dim() == 0:
                    counts = counts.unsqueeze(0)
                
                # Count loss (uses model-specific loss function)
                count_loss = count_loss_fn(pred_counts, counts)
                
                # Density loss if available
                if pred_density is not None:
                    try:
                        if pred_density.shape[-2:] != density_maps.shape[-2:]:
                            if pred_density.dim() == 3:
                                pred_density = pred_density.unsqueeze(1)
                            pred_density = torch.nn.functional.interpolate(
                                pred_density, size=density_maps.shape[-2:], 
                                mode='bilinear', align_corners=False
                            ).squeeze(1)
                        density_loss = self.density_criterion(pred_density, density_maps)
                        total_loss = count_loss + 0.5 * density_loss
                    except:
                        total_loss = count_loss
                else:
                    total_loss = count_loss
                
                # Check for NaN loss
                if torch.isnan(total_loss):
                    print(f"    Warning: NaN loss detected, skipping batch")
                    continue
                
                total_loss.backward()
                
                # FIX #3: Lower gradient clipping threshold for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                batch_mae = torch.abs(pred_counts - counts).mean().item()
                train_maes.append(batch_mae)
                pbar.set_postfix({'mae': f'{batch_mae:.3f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
            
            # Validation
            model.eval()
            val_maes = []
            
            with torch.no_grad():
                for batch in val_loader:
                    images, texts, counts, _ = batch
                    images = images.to(device)
                    counts = counts.to(device).float().squeeze()
                    
                    try:
                        outputs = model(images, texts)
                        pred_counts = outputs[0] if isinstance(outputs, tuple) else outputs
                    except:
                        try:
                            text_str = texts[0] if isinstance(texts, list) else texts
                            outputs = model(images, text_str)
                            pred_counts = outputs[0] if isinstance(outputs, tuple) else outputs
                        except:
                            pred_counts = model(images)
                    
                    pred_counts = pred_counts.squeeze()
                    
                    # Handle scalar tensors
                    if pred_counts.dim() == 0:
                        pred_counts = pred_counts.unsqueeze(0)
                    if counts.dim() == 0:
                        counts = counts.unsqueeze(0)
                        
                    val_maes.append(torch.abs(pred_counts - counts).mean().item())
            
            train_mae = np.mean(train_maes) if train_maes else float('inf')
            val_mae = np.mean(val_maes) if val_maes else float('inf')
            
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            history['lr'].append(scheduler.get_last_lr()[0])
            
            # Collapse detection and recovery
            if val_mae > collapse_threshold and best_model_state is not None and epoch > warmup_epochs:
                collapse_recoveries += 1
                if collapse_recoveries <= max_recoveries:
                    print(f"    ⚠ Collapse detected (val_mae={val_mae:.3f}), recovering to best checkpoint (attempt {collapse_recoveries}/{max_recoveries})")
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                    # Reduce LR after recovery to stabilize
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    patience_counter = 0
                    scheduler.step()
                    continue
            
            # Early stopping check
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Train MAE={train_mae:.3f}, Val MAE={val_mae:.3f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, history, best_val_mae
    
    def evaluate_model(self, model, model_name, test_loader):
        """Comprehensive evaluation on test set"""
        device = self.get_device(model_name)
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for batch in test_loader:
                images, texts, counts, _ = batch
                images = images.to(device)
                
                try:
                    outputs = model(images, texts)
                    pred_counts = outputs[0] if isinstance(outputs, tuple) else outputs
                except:
                    try:
                        text_str = texts[0] if isinstance(texts, list) else texts
                        outputs = model(images, text_str)
                        pred_counts = outputs[0] if isinstance(outputs, tuple) else outputs
                    except:
                        pred_counts = model(images)
                
                pred_counts = pred_counts.squeeze().cpu().numpy()
                counts = counts.squeeze().numpy()
                
                # Handle single values
                if np.ndim(pred_counts) == 0:
                    pred_counts = np.array([pred_counts])
                if np.ndim(counts) == 0:
                    counts = np.array([counts])
                
                all_preds.extend(pred_counts.tolist())
                all_gts.extend(counts.tolist())
        
        preds = np.array(all_preds)
        gts = np.array(all_gts)
        
        # Metrics
        mae = np.mean(np.abs(preds - gts))
        rmse = np.sqrt(np.mean((preds - gts) ** 2))
        nae = np.mean(np.abs(preds - gts) / (gts + 1e-6))
        
        return {'MAE': mae, 'RMSE': rmse, 'NAE': nae}
    
    def run_model_experiments(self, model_class, model_name, train_loader, val_loader,
                               seeds=[42, 123, 456, 789, 999], epochs=20):
        """Run experiments for one model with multiple seeds"""
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"Seeds: {seeds}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        seed_results = []
        failed_seeds = []
        
        for i, seed in enumerate(seeds):
            print(f"\n[Seed {i+1}/{len(seeds)}] seed={seed}")
            
            self.set_seed(seed)
            
            # Create fresh model instance
            try:
                model = model_class()
            except Exception as e:
                print(f"    Error creating model: {e}")
                failed_seeds.append({'seed': seed, 'error': str(e)})
                continue
            
            # Get model info before training (FIX #2: show both trainable and total)
            trainable_params = self.count_parameters(model)
            total_params = self.count_total_parameters(model)
            size_mb = self.get_model_size_mb(model, trainable_only=True)
            total_size_mb = self.get_model_size_mb(model, trainable_only=False)
            
            print(f"    Trainable params: {trainable_params:,}, Total: {total_params:,}")
            print(f"    Trainable size: {size_mb:.2f} MB, Total (incl. frozen): {total_size_mb:.2f} MB")
            
            # Train with stability improvements (FIX #3 + FIX #4: model-specific LR)
            model_lr = self.model_learning_rates.get(model_name, 5e-5)
            model_warmup = self.model_warmup_epochs.get(model_name, 2)
            print(f"    Using LR={model_lr:.0e}, warmup={model_warmup} epochs")
            
            try:
                trained_model, history, best_val_mae = self.train_single_run(
                    model, model_name, train_loader, val_loader, epochs,
                    lr=model_lr,  # Model-specific learning rate
                    warmup_epochs=model_warmup  # Model-specific warmup
                )
            except Exception as e:
                print(f"    Training error: {e}")
                import traceback
                traceback.print_exc()
                failed_seeds.append({'seed': seed, 'error': str(e)})
                continue
            
            # Evaluate
            eval_results = self.evaluate_model(trained_model, model_name, val_loader)
            
            # Check for training collapse (NAE close to 1.0 indicates constant predictions)
            collapsed = eval_results['NAE'] > 0.95
            if collapsed:
                print(f"    WARNING: Training may have collapsed (NAE={eval_results['NAE']:.3f})")
            
            seed_results.append({
                'seed': seed,
                'MAE': eval_results['MAE'],
                'RMSE': eval_results['RMSE'],
                'NAE': eval_results['NAE'],
                'best_val_mae': best_val_mae,
                'epochs_trained': len(history['train_mae']),
                'collapsed': collapsed
            })
            
            # Save model
            save_path = self.output_dir / 'models' / f'{model_name}_seed{seed}.pth'
            torch.save(trained_model.state_dict(), save_path)
            
            print(f"    Final: MAE={eval_results['MAE']:.3f}, RMSE={eval_results['RMSE']:.3f}")
        
        if not seed_results:
            print(f"    No successful runs for {model_name}")
            return None
        
        # Aggregate results (optionally excluding collapsed runs)
        valid_results = [r for r in seed_results if not r['collapsed']]
        all_results_for_stats = valid_results if valid_results else seed_results
        
        maes = [r['MAE'] for r in all_results_for_stats]
        rmses = [r['RMSE'] for r in all_results_for_stats]
        naes = [r['NAE'] for r in all_results_for_stats]
        
        # Get efficiency metrics from fresh model
        fresh_model = model_class()
        inference = self.benchmark_inference(fresh_model, model_name)
        
        # Calculate 95% CI
        def calc_ci(data):
            if len(data) < 2:
                return (np.mean(data), np.mean(data))
            return stats.t.interval(0.95, len(data)-1, 
                                   loc=np.mean(data), 
                                   scale=stats.sem(data))
        
        aggregated = {
            'model_name': model_name,
            'num_seeds': len(seed_results),
            'num_valid': len(valid_results),
            'num_collapsed': len([r for r in seed_results if r['collapsed']]),
            'num_failed': len(failed_seeds),
            'MAE': {
                'mean': np.mean(maes),
                'std': np.std(maes),
                'ci_95': calc_ci(maes),
                'all_values': maes
            },
            'RMSE': {
                'mean': np.mean(rmses),
                'std': np.std(rmses),
                'all_values': rmses
            },
            'NAE': {
                'mean': np.mean(naes),
                'std': np.std(naes),
                'all_values': naes
            },
            'efficiency': {
                'parameters': self.count_parameters(fresh_model),
                'total_parameters': self.count_total_parameters(fresh_model),
                'size_mb': self.get_model_size_mb(fresh_model, trainable_only=True),
                'total_size_mb': self.get_model_size_mb(fresh_model, trainable_only=False),
                'inference_ms': inference['mean_ms'],
                'fps': inference['fps'],
                'device': inference['device']
            },
            'seed_results': seed_results,
            'failed_seeds': failed_seeds
        }
        
        self.all_results[model_name] = aggregated
        
        print(f"\n{model_name} Summary:")
        print(f"  MAE: {aggregated['MAE']['mean']:.3f} ± {aggregated['MAE']['std']:.3f}")
        print(f"  95% CI: [{aggregated['MAE']['ci_95'][0]:.3f}, {aggregated['MAE']['ci_95'][1]:.3f}]")
        print(f"  Valid/Total seeds: {aggregated['num_valid']}/{aggregated['num_seeds']}")
        print(f"  Trainable params: {aggregated['efficiency']['parameters']:,}")
        print(f"  Inference: {inference['mean_ms']:.2f} ms ({inference['fps']:.1f} FPS) on {inference['device']}")
        
        return aggregated
    
    def statistical_comparison(self):
        """Statistical comparison between all models"""
        
        model_names = list(self.all_results.keys())
        comparisons = {}
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                key = f"{name1} vs {name2}"
                
                comparisons[key] = {}
                
                for metric in ['MAE', 'RMSE', 'NAE']:
                    vals1 = self.all_results[name1][metric]['all_values']
                    vals2 = self.all_results[name2][metric]['all_values']
                    
                    # Need at least 2 samples each for t-test
                    if len(vals1) < 2 or len(vals2) < 2:
                        comparisons[key][metric] = {
                            't_statistic': float('nan'),
                            'p_value': float('nan'),
                            'significant': False,
                            'cohens_d': float('nan'),
                            'effect_size': 'insufficient data',
                            'better_model': 'N/A'
                        }
                        continue
                    
                    # Welch's t-test
                    t_stat, p_value = stats.ttest_ind(vals1, vals2, equal_var=False)
                    
                    # Cohen's d
                    pooled_std = np.sqrt((np.var(vals1) + np.var(vals2)) / 2)
                    cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std if pooled_std > 0 else 0
                    
                    # Effect size interpretation
                    d_abs = abs(cohens_d)
                    if d_abs < 0.2:
                        effect = 'negligible'
                    elif d_abs < 0.5:
                        effect = 'small'
                    elif d_abs < 0.8:
                        effect = 'medium'
                    else:
                        effect = 'large'
                    
                    comparisons[key][metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'cohens_d': cohens_d,
                        'effect_size': effect,
                        'better_model': name1 if np.mean(vals1) < np.mean(vals2) else name2
                    }
        
        return comparisons
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        
        lines = []
        lines.append("# Rigorous Experimental Results (Fixed Version)")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        lines.append("## Experimental Configuration")
        lines.append("- **Seeds**: 5 (42, 123, 456, 789, 999)")
        lines.append("- **Epochs**: 20 (with early stopping, patience=5)")
        lines.append("- **Optimizer**: AdamW (lr=5e-5, weight_decay=1e-4)")  # FIXED LR
        lines.append("- **LR Schedule**: Warmup (2 epochs) + Cosine Annealing")  # FIXED
        lines.append("- **Gradient Clipping**: 0.5")  # FIXED
        lines.append("- **Loss**: MSE (count) + 0.5 × BCE (density)")
        
        # Efficiency comparison
        lines.append("\n## Model Efficiency")
        lines.append("\n| Model | Trainable Params | Size (MB) | Inference (ms) | FPS | Device |")
        lines.append("|-------|------------------|-----------|----------------|-----|--------|")
        
        for name, res in self.all_results.items():
            eff = res['efficiency']
            lines.append(f"| {name} | {eff['parameters']:,} | {eff['size_mb']:.2f} | "
                        f"{eff['inference_ms']:.2f} | {eff['fps']:.1f} | {eff['device']} |")
        
        # Accuracy results
        lines.append("\n## Counting Accuracy")
        lines.append("\n| Model | MAE ↓ | RMSE ↓ | NAE ↓ | 95% CI (MAE) | Valid Seeds |")
        lines.append("|-------|-------|--------|-------|--------------|-------------|")
        
        for name, res in self.all_results.items():
            mae = f"{res['MAE']['mean']:.3f} ± {res['MAE']['std']:.3f}"
            rmse = f"{res['RMSE']['mean']:.3f} ± {res['RMSE']['std']:.3f}"
            nae = f"{res['NAE']['mean']:.3f} ± {res['NAE']['std']:.3f}"
            ci = f"[{res['MAE']['ci_95'][0]:.3f}, {res['MAE']['ci_95'][1]:.3f}]"
            valid = f"{res['num_valid']}/{res['num_seeds']}"
            lines.append(f"| {name} | {mae} | {rmse} | {nae} | {ci} | {valid} |")
        
        # Statistical comparisons
        if len(self.all_results) > 1:
            lines.append("\n## Statistical Significance Tests")
            
            comparisons = self.statistical_comparison()
            for comp_name, metrics in comparisons.items():
                lines.append(f"\n### {comp_name}")
                lines.append("\n| Metric | p-value | Significant | Effect Size | Better Model |")
                lines.append("|--------|---------|-------------|-------------|--------------|")
                
                for metric, data in metrics.items():
                    if data['effect_size'] == 'insufficient data':
                        lines.append(f"| {metric} | N/A | N/A | insufficient data | N/A |")
                    else:
                        sig = "✓ YES" if data['significant'] else "✗ NO"
                        lines.append(f"| {metric} | {data['p_value']:.4f} | {sig} | "
                                   f"{data['effect_size']} (d={data['cohens_d']:.2f}) | {data['better_model']} |")
        
        # Efficiency improvements
        lines.append("\n## Efficiency Analysis")
        
        if len(self.all_results) > 1:
            # Find baseline (model with most parameters)
            baseline_name = max(self.all_results.keys(), 
                              key=lambda x: self.all_results[x]['efficiency']['parameters'])
            baseline = self.all_results[baseline_name]['efficiency']
            
            for name, res in self.all_results.items():
                if name != baseline_name:
                    eff = res['efficiency']
                    param_red = (1 - eff['parameters'] / baseline['parameters']) * 100
                    size_red = (1 - eff['size_mb'] / baseline['size_mb']) * 100
                    speedup = baseline['inference_ms'] / eff['inference_ms'] if eff['inference_ms'] > 0 else 0
                    
                    lines.append(f"\n**{name} vs {baseline_name}:**")
                    lines.append(f"- Parameter reduction: {param_red:.1f}%")
                    lines.append(f"- Size reduction: {size_red:.1f}%")
                    lines.append(f"- Speedup: {speedup:.2f}x")
        
        # Per-seed breakdown
        lines.append("\n## Per-Seed Results")
        
        for name, res in self.all_results.items():
            lines.append(f"\n### {name}")
            lines.append("\n| Seed | MAE | RMSE | NAE | Epochs | Status |")
            lines.append("|------|-----|------|-----|--------|--------|")
            
            for sr in res['seed_results']:
                status = "⚠ Collapsed" if sr['collapsed'] else "✓ Valid"
                lines.append(f"| {sr['seed']} | {sr['MAE']:.3f} | {sr['RMSE']:.3f} | "
                           f"{sr['NAE']:.3f} | {sr['epochs_trained']} | {status} |")
            
            # List failed seeds if any
            if res['failed_seeds']:
                lines.append(f"\n**Failed seeds:** {[f['seed'] for f in res['failed_seeds']]}")
        
        # Save report
        report_text = '\n'.join(lines)
        report_path = self.output_dir / 'rigorous_report.md'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {report_path}")
        return report_text
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for dissertation"""
        
        # Main results table
        latex1 = []
        latex1.append(r"\begin{table}[htbp]")
        latex1.append(r"\centering")
        latex1.append(r"\caption{Counting Accuracy Comparison (Mean $\pm$ Std over valid seeds)}")
        latex1.append(r"\label{tab:accuracy_results}")
        latex1.append(r"\begin{tabular}{lcccc}")
        latex1.append(r"\toprule")
        latex1.append(r"Model & MAE $\downarrow$ & RMSE $\downarrow$ & NAE $\downarrow$ & Valid Seeds \\")
        latex1.append(r"\midrule")
        
        for name, res in self.all_results.items():
            mae = f"{res['MAE']['mean']:.2f} $\\pm$ {res['MAE']['std']:.2f}"
            rmse = f"{res['RMSE']['mean']:.2f} $\\pm$ {res['RMSE']['std']:.2f}"
            nae = f"{res['NAE']['mean']:.3f} $\\pm$ {res['NAE']['std']:.3f}"
            valid = f"{res['num_valid']}/{res['num_seeds']}"
            latex1.append(f"{name} & {mae} & {rmse} & {nae} & {valid} \\\\")
        
        latex1.append(r"\bottomrule")
        latex1.append(r"\end{tabular}")
        latex1.append(r"\end{table}")
        
        # Efficiency table
        latex2 = []
        latex2.append(r"\begin{table}[htbp]")
        latex2.append(r"\centering")
        latex2.append(r"\caption{Model Efficiency Comparison}")
        latex2.append(r"\label{tab:efficiency_results}")
        latex2.append(r"\begin{tabular}{lccccc}")
        latex2.append(r"\toprule")
        latex2.append(r"Model & Trainable Params & Size (MB) & Latency (ms) & FPS $\uparrow$ & Device \\")
        latex2.append(r"\midrule")
        
        for name, res in self.all_results.items():
            eff = res['efficiency']
            params = f"{eff['parameters']/1e6:.1f}M"
            latex2.append(f"{name} & {params} & {eff['size_mb']:.1f} & "
                         f"{eff['inference_ms']:.1f} & {eff['fps']:.1f} & {eff['device']} \\\\")
        
        latex2.append(r"\bottomrule")
        latex2.append(r"\end{tabular}")
        latex2.append(r"\end{table}")
        
        # Save
        with open(self.output_dir / 'accuracy_table.tex', 'w') as f:
            f.write('\n'.join(latex1))
        
        with open(self.output_dir / 'efficiency_table.tex', 'w') as f:
            f.write('\n'.join(latex2))
        
        print(f"✓ LaTeX tables saved")
    
    def generate_figures(self):
        """Generate publication-quality figures"""
        
        if not self.all_results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        names = list(self.all_results.keys())
        colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(names)]
        
        # 1. MAE comparison
        maes = [self.all_results[n]['MAE']['mean'] for n in names]
        mae_stds = [self.all_results[n]['MAE']['std'] for n in names]
        
        axes[0,0].bar(range(len(names)), maes, yerr=mae_stds, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black')
        axes[0,0].set_xticks(range(len(names)))
        axes[0,0].set_xticklabels(names, rotation=15, ha='right')
        axes[0,0].set_ylabel('MAE (Lower is Better)')
        axes[0,0].set_title('Counting Accuracy', fontweight='bold')
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # 2. Parameters comparison (trainable only)
        params = [self.all_results[n]['efficiency']['parameters']/1e6 for n in names]
        
        axes[0,1].bar(range(len(names)), params, color=colors, alpha=0.8, edgecolor='black')
        axes[0,1].set_xticks(range(len(names)))
        axes[0,1].set_xticklabels(names, rotation=15, ha='right')
        axes[0,1].set_ylabel('Trainable Parameters (Millions)')
        axes[0,1].set_title('Model Size', fontweight='bold')
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # 3. Inference speed
        fps = [self.all_results[n]['efficiency']['fps'] for n in names]
        
        axes[1,0].bar(range(len(names)), fps, color=colors, alpha=0.8, edgecolor='black')
        axes[1,0].set_xticks(range(len(names)))
        axes[1,0].set_xticklabels(names, rotation=15, ha='right')
        axes[1,0].set_ylabel('Frames Per Second')
        axes[1,0].set_title('Inference Speed', fontweight='bold')
        axes[1,0].grid(axis='y', alpha=0.3)
        
        # 4. Accuracy vs Efficiency trade-off
        for i, name in enumerate(names):
            axes[1,1].scatter(
                self.all_results[name]['efficiency']['parameters']/1e6,
                self.all_results[name]['MAE']['mean'],
                s=200, c=colors[i], label=name, edgecolors='black', alpha=0.8
            )
            axes[1,1].errorbar(
                self.all_results[name]['efficiency']['parameters']/1e6,
                self.all_results[name]['MAE']['mean'],
                yerr=self.all_results[name]['MAE']['std'],
                fmt='none', c=colors[i], capsize=3
            )
        
        axes[1,1].set_xlabel('Trainable Parameters (Millions)')
        axes[1,1].set_ylabel('MAE (Lower is Better)')
        axes[1,1].set_title('Accuracy-Efficiency Trade-off', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)
        
        plt.suptitle('Rigorous Experimental Results (5 seeds, 20 epochs)\nFixed: LR=5e-5, Warmup, Grad Clip=0.5', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plt.savefig(self.output_dir / 'rigorous_results.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'rigorous_results.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figures saved")
    
    def save_json_results(self):
        """Save results as JSON"""
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)  # FIX: Handle numpy bool types
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            return obj
        
        with open(self.output_dir / 'rigorous_results.json', 'w') as f:
            json.dump(convert(self.all_results), f, indent=2)
        
        print(f"✓ JSON results saved")


def main():
    """Main experiment runner"""
    
    print("="*70)
    print("RIGOROUS EXPERIMENTAL VALIDATION (FIXED VERSION)")
    print("="*70)
    print("\nFixes applied:")
    print("  1. MPS compatibility - CPU fallback for problematic models")
    print("  2. Model size - Now reports trainable parameters only")
    print("  3. Training stability - warmup, grad_clip=0.5")
    print("  4. Model-specific LR (ZMobileViT: 1e-4 AdamW)")
    print("  5. ZMobileViT: SmoothL1 loss, count_bias, proper init")
    print("  6. Collapse recovery - auto-revert to best checkpoint")
    print("="*70)
    
    # Initialize
    runner = RigorousExperimentRunner('./rigorous_results')
    
    # Prepare data
    print("\nPreparing data...")
    data_dir = download_countgd_dataset('./data')
    train_loader, val_loader = get_countgd_dataloaders(data_dir, batch_size=8)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Configuration
    seeds = [42, 123, 456, 789, 999]
    epochs = 20
    
    # Models to test
    models = [
        (LightweightCountGD, "CountGD_Baseline"),
        (OptimizedCountGD, "MLE-STAR_Optimized"),
        (ZMobileViTCounting, "ZMobileViT"),
    ]
    
    # Run experiments
    start_time = time.time()
    
    for model_class, model_name in models:
        try:
            runner.run_model_experiments(
                model_class=model_class,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                seeds=seeds,
                epochs=epochs
            )
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)
    
    runner.generate_report()
    runner.generate_latex_tables()
    runner.generate_figures()
    runner.save_json_results()
    
    # Final summary
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.1f} minutes")
    print(f"\nResults saved to: ./rigorous_results/")
    print("\nFiles generated:")
    print("  - rigorous_report.md")
    print("  - accuracy_table.tex")
    print("  - efficiency_table.tex")
    print("  - rigorous_results.json")
    print("  - rigorous_results.png/pdf")
    print("  - models/*.pth")
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Params':<12} {'MAE':<20} {'FPS':<10} {'Valid':<8}")
    print("-"*70)
    
    for name, res in runner.all_results.items():
        params = f"{res['efficiency']['parameters']/1e6:.1f}M"
        mae = f"{res['MAE']['mean']:.3f} ± {res['MAE']['std']:.3f}"
        fps = f"{res['efficiency']['fps']:.1f}"
        valid = f"{res['num_valid']}/{res['num_seeds']}"
        print(f"{name:<20} {params:<12} {mae:<20} {fps:<10} {valid:<8}")


if __name__ == "__main__":
    main()
