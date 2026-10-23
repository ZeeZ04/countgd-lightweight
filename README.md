# Lightweight CountGD: Vision–Language Object Counting for Constrained Hardware

MSc dissertation project, University of Southampton (2024–2026).

The project takes CountGD-style object counting — count and locate objects in an image from a natural-language
prompt such as *"count the red cars"* — and asks how small and fast the model can get before accuracy suffers.
Two compressed variants are compared against a full-size baseline:

1. **MLE-STAR Optimized** — produced by applying the MLE-STAR search-and-targeted-refinement methodology to the
   baseline architecture.
2. **ZMobileViT** — a much smaller custom MobileViT-based variant with linear attention.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![ONNX](https://img.shields.io/badge/ONNX-Export-005CED)

## About MLE-STAR

MLE-STAR is a machine-learning engineering agent methodology introduced by Google Research in 2025
(*"MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement"*). It works in two phases:
first **search** for strong candidate components, then use **ablation** to find the single component that matters
most and **refine that component** in a targeted way, rather than tuning everything at once.

Applied here (`mle_star.py`, `integrate_mle_star.py`):

1. **Search** — evaluated MobileViT variants (XXS, XS, S) as the image encoder.
2. **Ablation** — identified the vision encoder as the highest-impact component.
3. **Targeted refinement** — added channel attention to the encoder.
4. **Iteration** — added cross-attention vision–text fusion and an enhanced counting head with auxiliary loss.

## Architecture

```
 image ──► MobileViT image encoder ──┐
                                     ├─► vision–text fusion ─┬─► count head    (regression)
 prompt ─► distilled CLIP text enc. ─┘                       └─► grounding head (heatmap)
```

Both tasks are trained jointly: MSE loss on the count plus 0.5 × BCE loss on the density/grounding heatmap.

## Results

### Efficiency

| Model | Parameters | Size | CPU inference | Throughput |
|---|---|---|---|---|
| CountGD baseline | 157.2M | 599.8 MB | 154.2 ms | 6.5 FPS |
| MLE-STAR Optimized | 64.5M (−59%) | 246.2 MB | 46.1 ms (3.35× faster) | 21.7 FPS |
| ZMobileViT | 2.9M (−98%) | 11.0 MB | 48.1 ms (3.20× faster) | 20.8 FPS |

Source: `rigorous_results/rigorous_results.json`. An earlier standalone benchmark of the MLE-STAR model
(`MLE_STAR_RESULTS.md`) measured 100.5 ms → 29.9 ms per inference (3.36×) on a different run configuration.

### Counting error (5 seeds: 42, 123, 456, 789, 999)

| Model | MAE ↓ | RMSE ↓ | NAE ↓ |
|---|---|---|---|
| CountGD baseline | 1.401 ± 0.003 | 1.727 ± 0.010 | 0.648 ± 0.004 |
| MLE-STAR Optimized | 1.402 ± 0.003 | 1.707 ± 0.034 | 0.639 ± 0.017 |
| ZMobileViT | 1.423 ± 0.044 | 1.741 ± 0.130 | 0.648 ± 0.051 |

None of the pairwise differences are statistically significant (all p > 0.3; full tests in
`rigorous_results/rigorous_report.md`).

Training setup: AdamW (lr 5e-5, weight decay 1e-4), 2-epoch warm-up then cosine annealing, gradient clipping 0.5,
up to 20 epochs with early stopping (patience 5).

### Limitations

- The annotation files committed in `data/` are a 100-image training / 20-image validation **sample set** used to
  build and test the pipeline. Accuracy figures should be read as a like-for-like comparison between the three
  models on the same data, not as a benchmark against published CountGD results.
- Latency was measured on CPU. Edge-device numbers (Raspberry Pi, Jetson, mobile) have not been measured yet.

## Usage

```bash
git clone https://github.com/ZeeZ04/countgd-lightweight.git
cd countgd-lightweight
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the multi-seed comparison of all three models
python rigorous_experiments.py

# Inference on your own image
python test_with_image.py path/to/image.jpg --prompt "Count the red cars"

# Export to ONNX
python main.py --mode export --model-path countgd_lightweight.pth
```

Annotation format (`data/train.json`, `data/val.json`):

```json
{ "image": "train_0001.jpg", "text": "Count the red cars in the image", "count": 3,
  "boxes": [[x1, y1, x2, y2], ...] }
```

## Repository layout

| File | Purpose |
|---|---|
| `countgd_model.py` | Baseline model |
| `countgd_model_optimized.py` | MLE-STAR optimized model |
| `zmobilevit_counting.py` | ZMobileViT backbone and model |
| `mle_star.py`, `integrate_mle_star.py` | MLE-STAR search, ablation and refinement |
| `rigorous_experiments.py` | Multi-seed training, evaluation and significance tests |
| `rigorous_results/` | Final results (JSON, tables, figures) |
| `test_with_image.py`, `test_onnx_model.py` | Inference and ONNX checks |

## Stack

PyTorch · Hugging Face Transformers (CLIP) · timm (MobileViT) · ONNX / ONNX Runtime · Grad-CAM · Weights & Biases

## Author

**Abdul Azeez Aris** — University of Southampton, MSc Artificial Intelligence (2024–2026) · BEng (Hons) Artificial
Intelligence, Ulster University
