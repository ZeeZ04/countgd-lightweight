#!/usr/bin/env python3
"""
Proper Counting Evaluation for MSc Project
This replaces the classification evaluation with counting-specific metrics
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import pandas as pd

class CountingEvaluator:
    """Proper evaluation metrics for counting tasks"""
    
    def __init__(self, output_dir='./msc_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def evaluate_counting_performance(self, predictions, ground_truth, model_name='model'):
        """Calculate all counting-specific metrics"""
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Validate inputs
        if len(predictions) == 0 or len(ground_truth) == 0:
            raise ValueError("Empty predictions or ground truth arrays")
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Mismatched lengths: predictions={len(predictions)}, ground_truth={len(ground_truth)}")
        
        metrics = {
            'MAE': mean_absolute_error(ground_truth, predictions),
            'RMSE': np.sqrt(mean_squared_error(ground_truth, predictions)),
            'NAE': self._normalized_absolute_error(predictions, ground_truth),
            'SRE': self._squared_relative_error(predictions, ground_truth),
            'undercount': np.sum(predictions < ground_truth) / len(predictions),
            'overcount': np.sum(predictions > ground_truth) / len(predictions),
            'accurate': np.sum(np.abs(predictions - ground_truth) <= 1) / len(predictions),
            'correlation': np.corrcoef(predictions, ground_truth)[0, 1] if len(predictions) > 1 else 0.0
        }
        
        # Store for later analysis
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }
        
        return metrics
    
    def _normalized_absolute_error(self, pred, gt):
        """NAE = mean(|pred - gt| / gt) - Fixed division by zero"""
        # Add small epsilon to avoid division by zero
        return np.mean(np.abs(pred - gt) / np.maximum(gt, 1e-7))
    
    def _squared_relative_error(self, pred, gt):
        """SRE = sqrt(mean((pred - gt)^2 / gt^2)) - Fixed division by zero"""
        # Add small epsilon to avoid division by zero
        return np.sqrt(np.mean(((pred - gt) ** 2) / np.maximum(gt ** 2, 1e-7)))
    
    def statistical_comparison(self, model1_results, model2_results, model1_name='Model1', model2_name='Model2'):
        """Perform statistical significance testing between two models"""
        
        # Multiple metrics comparison
        comparisons = {}
        
        for metric in ['MAE', 'RMSE', 'NAE']:
            if metric in model1_results and metric in model2_results:
                # If we have multiple runs (with seeds)
                if isinstance(model1_results[metric], list):
                    t_stat, p_value = stats.ttest_ind(model1_results[metric], model2_results[metric])
                    effect_size = self._cohens_d(model1_results[metric], model2_results[metric])
                    
                    comparisons[metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': effect_size,
                        'interpretation': self._interpret_effect_size(effect_size)
                    }
        
        return comparisons
    
    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_effect_size(self, d):
        """Interpret Cohen's d"""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_comprehensive_report(self, all_models_results):
        """Generate LaTeX-ready tables and comprehensive analysis"""
        
        # Create results DataFrame
        df_data = []
        for model_name, results in all_models_results.items():
            row = {
                'Model': model_name,
                'MAE↓': f"{results['metrics']['MAE']:.2f}",
                'RMSE↓': f"{results['metrics']['RMSE']:.2f}",
                'NAE↓': f"{results['metrics']['NAE']:.3f}",
                'Correlation↑': f"{results['metrics']['correlation']:.3f}"
            }
            
            # Add statistical significance if available
            if 'std' in results:
                row['MAE↓'] = f"{results['metrics']['MAE']:.2f} ± {results['std']['MAE']:.2f}"
                row['RMSE↓'] = f"{results['metrics']['RMSE']:.2f} ± {results['std']['RMSE']:.2f}"
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save as LaTeX table
        latex_table = df.to_latex(index=False, escape=False)
        with open(self.output_dir / 'results_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'results_table.csv', index=False)
        
        # Generate summary
        summary = {
            'best_model': df.loc[df['MAE↓'].astype(str).str.extract('([\d.]+)')[0].astype(float).idxmin(), 'Model'],
            'total_models': len(df),
            'metrics_computed': list(df.columns[1:]),
            'output_files': ['results_table.tex', 'results_table.csv']
        }
        
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return df, summary
    
    def plot_counting_performance(self, results_dict):
        """Generate publication-quality plots for counting evaluation"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. MAE Comparison
        models = list(results_dict.keys())
        maes = [results_dict[m]['metrics']['MAE'] for m in models]
        
        if 'std' in list(results_dict.values())[0]:
            errors = [results_dict[m]['std']['MAE'] for m in models]
            axes[0, 0].bar(models, maes, yerr=errors, capsize=5, color='steelblue')
        else:
            axes[0, 0].bar(models, maes, color='steelblue')
        
        axes[0, 0].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[0, 0].set_title('Counting Performance (MAE)', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Scatter plot: Predictions vs Ground Truth
        if 'predictions' in list(results_dict.values())[0]:
            for model_name in models[:3]:  # Plot top 3 models
                pred = results_dict[model_name]['predictions']
                gt = results_dict[model_name]['ground_truth']
                axes[0, 1].scatter(gt, pred, alpha=0.5, label=model_name, s=20)
            
            # Add perfect prediction line
            max_count = max([max(results_dict[m]['ground_truth']) for m in models])
            axes[0, 1].plot([0, max_count], [0, max_count], 'k--', alpha=0.5, label='Perfect')
            axes[0, 1].set_xlabel('Ground Truth Count', fontsize=12)
            axes[0, 1].set_ylabel('Predicted Count', fontsize=12)
            axes[0, 1].set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # 3. Error Distribution
        for model_name in models[:3]:
            if 'predictions' in results_dict[model_name]:
                errors = np.array(results_dict[model_name]['predictions']) - np.array(results_dict[model_name]['ground_truth'])
                axes[0, 2].hist(errors, alpha=0.5, label=model_name, bins=20)
        
        axes[0, 2].set_xlabel('Prediction Error', fontsize=12)
        axes[0, 2].set_ylabel('Frequency', fontsize=12)
        axes[0, 2].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # 4. Multiple Metrics Comparison
        metrics_to_plot = ['MAE', 'RMSE', 'NAE']
        metric_values = {metric: [results_dict[m]['metrics'][metric] for m in models] for metric in metrics_to_plot}
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            axes[1, 0].bar(x + i*width, metric_values[metric], width, label=metric)
        
        axes[1, 0].set_xlabel('Models', fontsize=12)
        axes[1, 0].set_ylabel('Error Value', fontsize=12)
        axes[1, 0].set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 5. Correlation Analysis
        correlations = [results_dict[m]['metrics']['correlation'] for m in models]
        axes[1, 1].bar(models, correlations, color='coral')
        axes[1, 1].set_ylabel('Correlation Coefficient', fontsize=12)
        axes[1, 1].set_title('Prediction Correlation', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].legend()
        
        # 6. Under/Over counting analysis
        under_over = {
            'Undercount': [results_dict[m]['metrics']['undercount'] * 100 for m in models],
            'Overcount': [results_dict[m]['metrics']['overcount'] * 100 for m in models],
            'Accurate (±1)': [results_dict[m]['metrics']['accurate'] * 100 for m in models]
        }
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, (label, values) in enumerate(under_over.items()):
            axes[1, 2].bar(x + i*width, values, width, label=label)
        
        axes[1, 2].set_xlabel('Models', fontsize=12)
        axes[1, 2].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 2].set_title('Counting Tendency Analysis', fontsize=14, fontweight='bold')
        axes[1, 2].set_xticks(x + width)
        axes[1, 2].set_xticklabels(models, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Comprehensive Counting Evaluation Results', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'counting_evaluation.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'counting_evaluation.pdf', bbox_inches='tight')
        
        return fig


def run_proper_evaluation(model, test_loader, model_name='ZMobileViT'):
    """Run complete counting evaluation"""
    
    evaluator = CountingEvaluator()
    
    all_predictions = []
    all_ground_truth = []
    
    model.eval()
    with torch.no_grad():
        for images, texts, counts, _ in test_loader:
            # Get predictions
            pred_counts, _ = model(images, texts)
            
            all_predictions.extend(pred_counts.cpu().numpy().flatten())
            all_ground_truth.extend(counts.cpu().numpy().flatten())
    
    # Calculate metrics
    metrics = evaluator.evaluate_counting_performance(
        all_predictions, all_ground_truth, model_name
    )
    
    print(f"\n{'='*50}")
    print(f"COUNTING EVALUATION RESULTS - {model_name}")
    print(f"{'='*50}")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"NAE:  {metrics['NAE']:.3f}")
    print(f"Correlation: {metrics['correlation']:.3f}")
    print(f"Accurate (±1): {metrics['accurate']*100:.1f}%")
    
    return metrics, evaluator


if __name__ == "__main__":
    # Example usage
    from countgd_dataset import get_countgd_dataloaders
    from countgd_model import LightweightCountGD
    
    # Load data
    _, test_loader = get_countgd_dataloaders('./data', batch_size=8)
    
    # Load model
    model = LightweightCountGD()
    
    # Run evaluation
    metrics, evaluator = run_proper_evaluation(model, test_loader, 'CountGD')
    
    print("\n✅ Proper counting evaluation complete!")
    print("📊 Results saved to ./msc_results/")
