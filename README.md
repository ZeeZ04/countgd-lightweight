# CountGD: Lightweight Vision-Language Model for Object Counting

An enhanced lightweight vision-language model for object counting and visual grounding using natural language instructions. This project implements MLE-STAR optimization techniques combined with ZMobileViT architecture for efficient deployment on edge devices.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![ONNX](https://img.shields.io/badge/ONNX-Export-005CED)
![License](https://img.shields.io/badge/License-Research-yellow)

## Overview

This research project investigates counting and locating objects in images using lightweight vision-language models with natural language instructions. The work was conducted as part of an MSc dissertation at the University of Southampton.

### Key Contributions

- **MLE-STAR Integration** — Enhanced training with Maximum Likelihood Estimation and Spatial-Temporal Attention Refinement
- **ZMobileViT Backbone** — Lightweight vision encoder optimized for mobile deployment
- **Dual-Task Learning** — Simultaneous object counting and visual grounding
- **Explainable AI** — Grad-CAM++ visualizations for model interpretability
- **ONNX Export** — Production-ready model deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CountGD Architecture                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Image     │    │   Text      │    │    Fusion       │ │
│  │  Encoder    │───▶│  Encoder    │───▶│    Module       │ │
│  │ (MobileViT) │    │   (CLIP)    │    │  (Projection)   │ │
│  └─────────────┘    └─────────────┘    └────────┬────────┘ │
│                                                  │          │
│                     ┌────────────────────────────┼────────┐ │
│                     │                            ▼        │ │
│                     │  ┌─────────────┐  ┌─────────────┐  │ │
│                     │  │   Count     │  │  Grounding  │  │ │
│                     │  │    Head     │  │    Head     │  │ │
│                     │  │ (Regression)│  │  (Heatmap)  │  │ │
│                     │  └─────────────┘  └─────────────┘  │ │
│                     └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU training)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MLE-STAR-Optimized.git
cd MLE-STAR-Optimized

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Experiment Suite

```bash
python main.py --mode experiment
```

### Train Model

```bash
# Quick training
python train_countgd.py

# Full training with custom parameters
python main.py --mode train --epochs 20 --data-path ./data
```

### Inference with Custom Image

```bash
# Basic usage
python test_with_image.py path/to/image.jpg

# Custom prompt
python test_with_image.py image.jpg --prompt "Count the red cars"

# With trained model
python test_with_image.py image.jpg --type tiny --model countgd_tiny.pth
```

### Export to ONNX

```bash
python main.py --mode export --model-path countgd_lightweight.pth
```

## Dataset

### CountGD Dataset Format

```
data/
├── train.json          # Training annotations
├── val.json            # Validation annotations  
└── images/             # Image directory
    ├── train_0000.jpg
    ├── val_0000.jpg
    └── ...
```

### Annotation Structure

```json
{
  "image": "train_0001.jpg",
  "text": "Count the red cars in the image", 
  "count": 3,
  "boxes": [[x1, y1, x2, y2], ...]
}
```

## Results

### Performance Metrics

| Model | MAE ↓ | IoU ↑ | Inference (ms) | Parameters |
|-------|-------|-------|----------------|------------|
| Baseline | 2.45 | 0.42 | 85 | 12.5M |
| + MLE-STAR | 1.82 | 0.51 | 82 | 12.5M |
| + ZMobileViT | 1.91 | 0.48 | 45 | 5.2M |
| **Full (Ours)** | **1.76** | **0.53** | **42** | **5.2M** |

### Robustness Analysis

| Noise Type | Degradation (%) |
|------------|-----------------|
| Gaussian Blur | 8.2 |
| Brightness | 5.1 |
| Occlusion | 12.4 |

## Project Structure

```
MLE-STAR-Optimized/
├── main.py                      # Main execution script
├── countgd_model.py             # Base model implementation
├── countgd_model_optimized.py   # MLE-STAR enhanced model
├── countgd_model_tiny.py        # Lightweight variant
├── countgd_dataset.py           # Dataset loader
├── mle_star.py                  # MLE-STAR implementation
├── experiments.py               # Evaluation suite
├── train_countgd.py             # Training script
├── test_with_image.py           # Inference script
├── zmobilevit_counting.py       # ZMobileViT backbone
├── requirements.txt             # Dependencies
├── data/                        # Dataset directory
├── msc_results/                 # Experiment results
└── README.md
```

## Key Technologies

- **PyTorch** — Deep learning framework
- **Transformers** — CLIP text encoder
- **TIMM** — Vision model library (MobileViT)
- **ONNX/ONNX Runtime** — Model deployment
- **Grad-CAM** — Explainability visualizations
- **Weights & Biases** — Experiment tracking

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{aris2025countgd,
  title={Lightweight Vision-Language Models for Object Counting with Natural Language Instructions},
  author={Aris, Abdul Azeez},
  year={2025},
  school={University of Southampton},
  type={MSc Dissertation}
}
```

## License

This project is for academic research purposes. See the dissertation for full details.

## Author

**Abdul Azeez Aris**
- MSc Artificial Intelligence, University of Southampton (2024-2026)
- BEng (Hons) Artificial Intelligence, Ulster University (2020-2024)

## Acknowledgments

- University of Southampton AI Research Group
- CountGD dataset authors
- MobileViT and CLIP research teams
