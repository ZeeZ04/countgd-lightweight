#!/usr/bin/env python3
"""
Statistical Training Pipeline for MSc Research
Implements multiple seeds, significance testing, and proper experimental controls
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
import json
import time
import inspect
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Import your models
from zmobilevit_counting import ZMobileViTCounting
from countgd_model import LightweightCountGD
from countgd_dataset import get_countgd_dataloaders, download_countgd_dataset
from proper_counting_evaluation import CountingEvaluator


class StatisticalTrainer:
    """
    Training with proper statistical methodology for research
    """
    
    def __init__(self, output_dir='./msc_statistical_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Loss functions
        self.count_criterion = nn.MSELoss()
        self.density_criterion = nn.BCEWithLogitsLoss()
        
        # Results storage
        self.all_results = {}
        
    def train_with_seeds(
        self,
        model_class,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        seeds: List[int] = [42, 123, 456, 789, 999],
        epochs: int = 20,
        lr: float = 1e-4
    ) -> Dict:
        """
        Train model with multiple random seeds for statistical validity
        """
        
        print(f"\n{'='*60}")
        print(f"Training {model_name} with {len(seeds)} seeds")
        print(f"{'='*60}")
        
        seed_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n[Seed {seed_idx+1}/{len(seeds)}] Training with seed={seed}")
            
            # Set all random seeds
            self._set_seed(seed)
            
            # Create fresh model instance
            model = model_class()
            
            # Train
            trained_model, metrics = self._train_single_seed(
                model, train_loader, val_loader, 
                epochs=epochs, lr=lr, seed=seed
            )
            
            # Evaluate
            eval_metrics = self._evaluate_model(trained_model, val_loader)
            
            # Store results
            seed_result = {
                'seed': seed,
                'final_mae': eval_metrics['MAE'],
                'final_rmse': eval_metrics['RMSE'],
                'final_nae': eval_metrics['NAE'],
                'training_history': metrics,
                'all_metrics': eval_metrics
            }
            seed_results.append(seed_result)
            
            # Save model
            torch.save(
                trained_model.state_dict(),
                self.output_dir / f'{model_name}_seed{seed}.pth'
            )
        
        # Aggregate results across seeds
        aggregated = self._aggregate_results(seed_results, model_name)
        
        # Store for comparison
        self.all_results[model_name] = aggregated
        
        return aggregated
    
    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _train_single_seed(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        seed: int
    ) -> Tuple[nn.Module, Dict]:
        """Train model for single seed"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': []
        }
        
        best_val_mae = float('inf')
        patience = 0
        max_patience = 5
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_mae = 0
            
            for images, texts, counts, density_maps in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                images = images.to(device)
                counts = counts.to(device)
                density_maps = density_maps.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass - Handle different model types
                try:
                    if hasattr(model, 'forward') and len(inspect.signature(model.forward).parameters) > 2:
                        pred_counts, pred_density = model(images, texts)
                    else:
                        pred_counts, pred_density = model(images, texts)
                except Exception as e:
                    print(f"Warning: Model forward pass failed: {e}")
                    # Fallback to basic forward
                    outputs = model(images, texts)
                    if isinstance(outputs, tuple):
                        pred_counts, pred_density = outputs[0], outputs[1] if len(outputs) > 1 else torch.zeros_like(density_maps)
                    else:
                        pred_counts, pred_density = outputs, torch.zeros_like(density_maps)
                
                # Calculate loss
                count_loss = self.count_criterion(pred_counts.squeeze(), counts.squeeze())
                
                # Resize pred_density to match density_maps if needed
                if pred_density.shape[-2:] != density_maps.shape[-2:]:
                    # Ensure pred_density has channel dimension for interpolation
                    if pred_density.dim() == 3:  # [B, H, W]
                        pred_density = pred_density.unsqueeze(1)  # [B, 1, H, W]
                    pred_density = torch.nn.functional.interpolate(
                        pred_density, size=density_maps.shape[-2:], mode='bilinear', align_corners=False
                    )
                    pred_density = pred_density.squeeze(1)  # Back to [B, H, W]
                
                density_loss = self.density_criterion(pred_density.squeeze(), density_maps)
                
                total_loss = count_loss + 0.5 * density_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_mae += torch.abs(pred_counts.squeeze() - counts.squeeze()).mean().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_mae = 0
            
            with torch.no_grad():
                for images, texts, counts, density_maps in val_loader:
                    images = images.to(device)
                    counts = counts.to(device)
                    density_maps = density_maps.to(device)
                    
                    # Forward pass - Handle different model types
                    try:
                        outputs = model(images, texts)
                        if isinstance(outputs, tuple):
                            pred_counts, pred_density = outputs[0], outputs[1] if len(outputs) > 1 else torch.zeros_like(density_maps)
                        else:
                            pred_counts, pred_density = outputs, torch.zeros_like(density_maps)
                    except Exception as e:
                        print(f"Warning: Model validation failed: {e}")
                        continue
                    
                    count_loss = self.count_criterion(pred_counts.squeeze(), counts.squeeze())
                    
                    # Resize pred_density to match density_maps if needed
                    if pred_density.shape[-2:] != density_maps.shape[-2:]:
                        if pred_density.dim() == 3:  # [B, H, W]
                            pred_density = pred_density.unsqueeze(1)  # [B, 1, H, W]
                        pred_density = torch.nn.functional.interpolate(
                            pred_density, size=density_maps.shape[-2:], mode='bilinear', align_corners=False
                        )
                        pred_density = pred_density.squeeze(1)  # Back to [B, H, W]
                    
                    density_loss = self.density_criterion(pred_density.squeeze(), density_maps)
                    
                    total_loss = count_loss + 0.5 * density_loss
                    
                    val_loss += total_loss.item()
                    val_mae += torch.abs(pred_counts.squeeze() - counts.squeeze()).mean().item()
            
            # Record metrics
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['train_mae'].append(train_mae / len(train_loader))
            history['val_mae'].append(val_mae / len(val_loader))
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            current_val_mae = val_mae / len(val_loader)
            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Progress
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Train MAE={train_mae/len(train_loader):.3f}, "
                      f"Val MAE={val_mae/len(val_loader):.3f}")
        
        return model, history
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for images, texts, counts, _ in test_loader:
                images = images.to(device)
                
                # Forward pass - Handle different model types
                try:
                    outputs = model(images, texts)
                    if isinstance(outputs, tuple):
                        pred_counts = outputs[0]
                    else:
                        pred_counts = outputs
                except Exception as e:
                    print(f"Warning: Model evaluation failed: {e}")
                    continue
                
                all_predictions.extend(pred_counts.cpu().numpy().flatten())
                all_ground_truth.extend(counts.numpy().flatten())
        
        # Calculate metrics
        evaluator = CountingEvaluator()
        metrics = evaluator.evaluate_counting_performance(
            all_predictions, all_ground_truth, 'temp'
        )
        
        return metrics
    
    def _aggregate_results(self, seed_results: List[Dict], model_name: str) -> Dict:
        """Aggregate results across seeds with statistical analysis"""
        
        # Extract metrics across seeds
        maes = [r['final_mae'] for r in seed_results]
        rmses = [r['final_rmse'] for r in seed_results]
        naes = [r['final_nae'] for r in seed_results]
        
        # Calculate statistics
        aggregated = {
            'model_name': model_name,
            'num_seeds': len(seed_results),
            'MAE': {
                'mean': np.mean(maes),
                'std': np.std(maes),
                'median': np.median(maes),
                'ci_95': self._confidence_interval(maes),
                'all_values': maes
            },
            'RMSE': {
                'mean': np.mean(rmses),
                'std': np.std(rmses),
                'median': np.median(rmses),
                'ci_95': self._confidence_interval(rmses),
                'all_values': rmses
            },
            'NAE': {
                'mean': np.mean(naes),
                'std': np.std(naes),
                'median': np.median(naes),
                'ci_95': self._confidence_interval(naes),
                'all_values': naes
            },
            'consistency': {
                'mae_cv': np.std(maes) / np.mean(maes),  # Coefficient of variation
                'rmse_cv': np.std(rmses) / np.mean(rmses)
            }
        }
        
        return aggregated
    
    def _confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval"""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return (mean - interval, mean + interval)
    
    def compare_models_statistically(self, model1_name: str, model2_name: str) -> Dict:
        """
        Perform statistical comparison between two models
        """
        
        if model1_name not in self.all_results or model2_name not in self.all_results:
            raise ValueError(f"Models must be trained first")
        
        results1 = self.all_results[model1_name]
        results2 = self.all_results[model2_name]
        
        comparisons = {}
        
        for metric in ['MAE', 'RMSE', 'NAE']:
            values1 = results1[metric]['all_values']
            values2 = results2[metric]['all_values']
            
            # Paired t-test (if same seeds) or independent t-test
            t_stat, p_value = stats.ttest_ind(values1, values2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
            
            # Interpretation
            significant = p_value < 0.05
            if abs(cohens_d) < 0.2:
                effect_size = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_size = "small"
            elif abs(cohens_d) < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            comparisons[metric] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': significant,
                'cohens_d': cohens_d,
                'effect_size': effect_size,
                'model1_better': np.mean(values1) < np.mean(values2)
            }
        
        return comparisons
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical report"""
        
        report_lines = [
            "# Statistical Analysis Report",
            "",
            "## Models Evaluated",
            ""
        ]
        
        # Model summaries
        for model_name, results in self.all_results.items():
            report_lines.extend([
                f"### {model_name}",
                f"- Seeds tested: {results['num_seeds']}",
                f"- MAE: {results['MAE']['mean']:.3f} ± {results['MAE']['std']:.3f}",
                f"- 95% CI: [{results['MAE']['ci_95'][0]:.3f}, {results['MAE']['ci_95'][1]:.3f}]",
                f"- RMSE: {results['RMSE']['mean']:.3f} ± {results['RMSE']['std']:.3f}",
                f"- Consistency (CV): {results['consistency']['mae_cv']:.3f}",
                ""
            ])
        
        # Pairwise comparisons
        if len(self.all_results) >= 2:
            report_lines.extend([
                "## Statistical Comparisons",
                ""
            ])
            
            model_names = list(self.all_results.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    comparison = self.compare_models_statistically(model_names[i], model_names[j])
                    
                    report_lines.extend([
                        f"### {model_names[i]} vs {model_names[j]}",
                        ""
                    ])
                    
                    for metric, stats in comparison.items():
                        sig_marker = "✓" if stats['significant'] else "✗"
                        better_model = model_names[i] if stats['model1_better'] else model_names[j]
                        
                        report_lines.extend([
                            f"**{metric}**:",
                            f"- p-value: {stats['p_value']:.6f} {sig_marker}",
                            f"- Effect size: {stats['effect_size']} (d={stats['cohens_d']:.3f})",
                            f"- Better model: {better_model}",
                            ""
                        ])
        
        # Save report
        report_path = self.output_dir / 'statistical_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save raw results as JSON
        json_path = self.output_dir / 'statistical_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for model_name, results in self.all_results.items():
                json_data[model_name] = {
                    'MAE_mean': float(results['MAE']['mean']),
                    'MAE_std': float(results['MAE']['std']),
                    'MAE_ci_lower': float(results['MAE']['ci_95'][0]),
                    'MAE_ci_upper': float(results['MAE']['ci_95'][1]),
                    'RMSE_mean': float(results['RMSE']['mean']),
                    'RMSE_std': float(results['RMSE']['std']),
                    'NAE_mean': float(results['NAE']['mean']),
                    'NAE_std': float(results['NAE']['std'])
                }
            json.dump(json_data, f, indent=2)
        
        print(f"\n✅ Statistical report saved to {report_path}")
        print(f"📊 Raw results saved to {json_path}")
        
        return report_path
    
    def plot_statistical_results(self):
        """Generate statistical visualization plots"""
        
        if not self.all_results:
            print("No results to plot")
            return
        
        # Setup plot style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. MAE comparison with error bars
        models = list(self.all_results.keys())
        mae_means = [self.all_results[m]['MAE']['mean'] for m in models]
        mae_stds = [self.all_results[m]['MAE']['std'] for m in models]
        
        axes[0, 0].bar(models, mae_means, yerr=mae_stds, capsize=5, color='steelblue', alpha=0.8)
        axes[0, 0].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[0, 0].set_title('MAE Comparison (with std error)', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add significance markers
        if len(models) == 2:
            comparison = self.compare_models_statistically(models[0], models[1])
            if comparison['MAE']['significant']:
                axes[0, 0].text(0.5, max(mae_means) * 1.1, 
                              f"p={comparison['MAE']['p_value']:.4f} *",
                              ha='center', fontsize=10)
        
        # 2. Box plot of all seed results
        mae_data = [self.all_results[m]['MAE']['all_values'] for m in models]
        bp = axes[0, 1].boxplot(mae_data, labels=models, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[0, 1].set_ylabel('MAE', fontsize=12)
        axes[0, 1].set_title('MAE Distribution Across Seeds', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Confidence intervals
        for i, model in enumerate(models):
            ci = self.all_results[model]['MAE']['ci_95']
            mean = self.all_results[model]['MAE']['mean']
            axes[1, 0].errorbar(i, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], 
                              fmt='o', markersize=8, capsize=5, label=model)
        
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].set_ylabel('MAE', fontsize=12)
        axes[1, 0].set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        
        # 4. Consistency analysis (CV)
        cvs = [self.all_results[m]['consistency']['mae_cv'] for m in models]
        axes[1, 1].bar(models, cvs, color='coral', alpha=0.8)
        axes[1, 1].set_ylabel('Coefficient of Variation', fontsize=12)
        axes[1, 1].set_title('Model Consistency (lower is better)', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good consistency')
        axes[1, 1].legend()
        
        plt.suptitle('Statistical Analysis of Model Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plot_path = self.output_dir / 'statistical_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'statistical_analysis.pdf', bbox_inches='tight')
        
        print(f"📈 Statistical plots saved to {plot_path}")
        
        return fig


def run_statistical_experiments(quick_mode=False):
    """
    Main function to run statistically rigorous experiments
    """
    
    print("="*60)
    print("STATISTICAL TRAINING PIPELINE")
    print("="*60)
    
    # Setup
    trainer = StatisticalTrainer()
    
    # Download and prepare data
    data_dir = download_countgd_dataset('./data')
    train_loader, val_loader = get_countgd_dataloaders(data_dir, batch_size=8)
    
    # Configure experiments
    if quick_mode:
        seeds = [42, 123, 456]  # Fewer seeds for testing
        epochs = 10
    else:
        seeds = [42, 123, 456, 789, 999]  # 5 seeds for proper evaluation
        epochs = 20
    
    # Train models
    models_to_train = [
        (ZMobileViTCounting, 'ZMobileViT'),
        (LightweightCountGD, 'CountGD_Baseline')
    ]
    
    for model_class, model_name in models_to_train:
        results = trainer.train_with_seeds(
            model_class=model_class,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            seeds=seeds,
            epochs=epochs,
            lr=1e-4
        )
        
        print(f"\n{model_name} Results:")
        print(f"MAE: {results['MAE']['mean']:.3f} ± {results['MAE']['std']:.3f}")
        print(f"95% CI: [{results['MAE']['ci_95'][0]:.3f}, {results['MAE']['ci_95'][1]:.3f}]")
    
    # Statistical comparison
    if len(trainer.all_results) >= 2:
        comparison = trainer.compare_models_statistically('ZMobileViT', 'CountGD_Baseline')
        
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        
        for metric, stats in comparison.items():
            print(f"\n{metric}:")
            print(f"  p-value: {stats['p_value']:.6f}")
            print(f"  Significant: {'YES' if stats['significant'] else 'NO'}")
            print(f"  Effect size: {stats['effect_size']} (Cohen's d = {stats['cohens_d']:.3f})")
    
    # Generate reports and plots
    trainer.generate_statistical_report()
    trainer.plot_statistical_results()
    
    print("\n✅ Statistical experiments complete!")
    print(f"📁 All results saved to {trainer.output_dir}")
    
    return trainer


if __name__ == "__main__":
    # Run experiments (use quick_mode=True for testing)
    trainer = run_statistical_experiments(quick_mode=True)
