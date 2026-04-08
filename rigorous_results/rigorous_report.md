# Rigorous Experimental Results (Fixed Version)

Generated: 2026-02-05 13:29:37

## Experimental Configuration
- **Seeds**: 5 (42, 123, 456, 789, 999)
- **Epochs**: 20 (with early stopping, patience=5)
- **Optimizer**: AdamW (lr=5e-5, weight_decay=1e-4)
- **LR Schedule**: Warmup (2 epochs) + Cosine Annealing
- **Gradient Clipping**: 0.5
- **Loss**: MSE (count) + 0.5 × BCE (density)

## Model Efficiency

| Model | Trainable Params | Size (MB) | Inference (ms) | FPS | Device |
|-------|------------------|-----------|----------------|-----|--------|
| CountGD_Baseline | 157,235,235 | 599.80 | 154.23 | 6.5 | cpu |
| MLE-STAR_Optimized | 64,530,691 | 246.17 | 46.08 | 21.7 | cpu |
| ZMobileViT | 2,895,652 | 11.05 | 48.13 | 20.8 | cpu |

## Counting Accuracy

| Model | MAE ↓ | RMSE ↓ | NAE ↓ | 95% CI (MAE) | Valid Seeds |
|-------|-------|--------|-------|--------------|-------------|
| CountGD_Baseline | 1.401 ± 0.003 | 1.727 ± 0.010 | 0.648 ± 0.004 | [1.398, 1.405] | 5/5 |
| MLE-STAR_Optimized | 1.402 ± 0.003 | 1.707 ± 0.034 | 0.639 ± 0.017 | [1.398, 1.406] | 5/5 |
| ZMobileViT | 1.423 ± 0.044 | 1.741 ± 0.130 | 0.648 ± 0.051 | [1.362, 1.483] | 5/5 |

## Statistical Significance Tests

### CountGD_Baseline vs MLE-STAR_Optimized

| Metric | p-value | Significant | Effect Size | Better Model |
|--------|---------|-------------|-------------|--------------|
| MAE | 0.8783 | ✗ NO | negligible (d=-0.11) | CountGD_Baseline |
| RMSE | 0.3325 | ✗ NO | medium (d=0.76) | MLE-STAR_Optimized |
| NAE | 0.3321 | ✗ NO | medium (d=0.77) | MLE-STAR_Optimized |

### CountGD_Baseline vs ZMobileViT

| Metric | p-value | Significant | Effect Size | Better Model |
|--------|---------|-------------|-------------|--------------|
| MAE | 0.3793 | ✗ NO | medium (d=-0.70) | CountGD_Baseline |
| RMSE | 0.8345 | ✗ NO | negligible (d=-0.16) | CountGD_Baseline |
| NAE | 0.9839 | ✗ NO | negligible (d=0.02) | ZMobileViT |

### MLE-STAR_Optimized vs ZMobileViT

| Metric | p-value | Significant | Effect Size | Better Model |
|--------|---------|-------------|-------------|--------------|
| MAE | 0.3855 | ✗ NO | medium (d=-0.69) | MLE-STAR_Optimized |
| RMSE | 0.6390 | ✗ NO | small (d=-0.35) | MLE-STAR_Optimized |
| NAE | 0.7589 | ✗ NO | small (d=-0.23) | MLE-STAR_Optimized |

## Efficiency Analysis

**MLE-STAR_Optimized vs CountGD_Baseline:**
- Parameter reduction: 59.0%
- Size reduction: 59.0%
- Speedup: 3.35x

**ZMobileViT vs CountGD_Baseline:**
- Parameter reduction: 98.2%
- Size reduction: 98.2%
- Speedup: 3.20x

## Per-Seed Results

### CountGD_Baseline

| Seed | MAE | RMSE | NAE | Epochs | Status |
|------|-----|------|-----|--------|--------|
| 42 | 1.400 | 1.717 | 0.644 | 13 | ✓ Valid |
| 123 | 1.400 | 1.719 | 0.645 | 15 | ✓ Valid |
| 456 | 1.400 | 1.727 | 0.648 | 13 | ✓ Valid |
| 789 | 1.407 | 1.744 | 0.656 | 17 | ✓ Valid |
| 999 | 1.400 | 1.725 | 0.647 | 18 | ✓ Valid |

### MLE-STAR_Optimized

| Seed | MAE | RMSE | NAE | Epochs | Status |
|------|-----|------|-----|--------|--------|
| 42 | 1.399 | 1.709 | 0.641 | 16 | ✓ Valid |
| 123 | 1.404 | 1.739 | 0.653 | 20 | ✓ Valid |
| 456 | 1.406 | 1.745 | 0.656 | 20 | ✓ Valid |
| 789 | 1.399 | 1.694 | 0.634 | 15 | ✓ Valid |
| 999 | 1.400 | 1.650 | 0.610 | 20 | ✓ Valid |

### ZMobileViT

| Seed | MAE | RMSE | NAE | Epochs | Status |
|------|-----|------|-----|--------|--------|
| 42 | 1.510 | 1.991 | 0.744 | 16 | ✓ Valid |
| 123 | 1.405 | 1.740 | 0.654 | 13 | ✓ Valid |
| 456 | 1.400 | 1.651 | 0.610 | 16 | ✓ Valid |
| 789 | 1.400 | 1.683 | 0.629 | 10 | ✓ Valid |
| 999 | 1.400 | 1.640 | 0.601 | 20 | ✓ Valid |