> These size variants are early experiments. The final comparison (baseline, MLE-STAR optimised, ZMobileViT)
> is in the main [README](README.md) and [`rigorous_results/`](rigorous_results/).

# CountGD Model Training Guide

## Quick Start

### 1. Create All Model Variants
```bash
python create_models.py
```
This creates untrained models of all sizes and shows their specifications.

### 2. Train a Single Model
```bash
# Train tiny model (recommended for testing)
python train_all_models.py --size tiny --epochs 5

# Train small model (balanced)
python train_all_models.py --size small --epochs 10

# Train micro model (ultra-lightweight)
python train_all_models.py --size micro --epochs 3
```

### 3. Train All Models
```bash
python train_all_models.py --train-all --epochs 5
```

### 4. Test Trained Models
```bash
# Test with explanations
python main.py --mode explain

# Test specific model
python test_explanations.py
```

## Model Sizes

| Size  | Memory | Parameters | Use Case |
|-------|--------|------------|----------|
| FULL  | 600 MB | 157M       | Baseline |
| SMALL | 586 MB | 153M       | Balanced |
| TINY  | 585 MB | 153M       | Fast training |
| MICRO | 0.3 MB | 72K        | Edge devices |

## Training Options

```bash
python train_all_models.py [OPTIONS]

Options:
  --size {full,small,tiny,micro}  Model size (default: small)
  --epochs INT                    Training epochs (default: 5)
  --lr FLOAT                     Learning rate (default: 1e-4)
  --data-path PATH               Dataset path (default: ./data)
  --train-all                    Train all model sizes
```

## Usage Examples

### Create and Train Micro Model (Ultra-fast)
```bash
python train_all_models.py --size micro --epochs 2
```

### Train All Models with Custom Settings
```bash
python train_all_models.py --train-all --epochs 10 --lr 5e-5
```

### Load and Use Trained Model
```python
from countgd_model import create_countgd_model
import torch

# Create model
model = create_countgd_model('tiny')

# Load trained weights
model.load_state_dict(torch.load('countgd_tiny.pth'))

# Use for inference
count, heatmap, explanations = model.inference_with_explanation(
    image, ["count red cars"]
)
```

## Files Created

- `countgd_{size}_untrained.pth` - Untrained model weights
- `countgd_{size}.pth` - Trained model weights
- Model configs in `model_configs.py`
- Size variants in `countgd_model_tiny.py`

## Recommendations

- **Development**: Use `tiny` model (fast training)
- **Production**: Use `small` model
- **Edge Deployment**: Use `micro` model (99.9% smaller)
- **Research**: Use `full` model as the baseline for comparisons
