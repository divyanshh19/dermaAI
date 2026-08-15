# AI Skin Lesion Detection System — Training Guide

This guide details the step-by-step procedure to split datasets, validate data quality, train baseline and advanced vision models, fit probability calibration, evaluate test performance, and run error analysis.

---

## 1. Environment Setup

Ensure Python 3.10+ is installed with PyTorch and dependencies:

```bash
pip install -r backend/requirements.txt
```

---

## 2. Step-by-Step ML Pipeline Execution

### STEP 1: Generate Lesion-Aware Stratified Data Splits
Generate deterministic 70% Train / 15% Val / 15% Test CSV splits grouped by `lesion_id` to eliminate data leakage:

```bash
python backend/ml/datasets/split_dataset.py --config config.yaml
```

Output files:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

---

### STEP 2: Run Automated Data Validation Pipeline
Inspect dataset integrity, verify image channels, detect corrupted/duplicate images, and verify split isolation:

```bash
python backend/ml/data_validation.py --config config.yaml
```

Outputs:
- `reports/data_quality_report.json`
- `reports/class_distribution.png`

---

### STEP 3: Train Baseline EfficientNet-B0 Model
Train EfficientNet-B0 with Focal Loss, Automatic Mixed Precision (AMP), and Macro F1 early stopping:

```bash
python backend/ml/training/train.py --config config.yaml --model efficientnet_b0 --loss focal
```

Saved checkpoint:
- `backend/ml/saved_models/efficientnet_b0_best.pth`

---

### STEP 4: Train ConvNeXt-Tiny Model
Train ConvNeXt-Tiny for model architecture comparison:

```bash
python backend/ml/training/train.py --config config.yaml --model convnext_tiny --loss focal
```

Saved checkpoint:
- `backend/ml/saved_models/convnext_tiny_best.pth`

---

### STEP 5: Train Vision Transformer (DeiT-Tiny)
Train DeiT-Tiny for attention-based transformer representation:

```bash
python backend/ml/training/train.py --config config.yaml --model vit_deit_tiny --loss focal
```

Saved checkpoint:
- `backend/ml/saved_models/vit_deit_tiny_best.pth`

---

### STEP 6: Temperature Calibration & Ensemble Configuration
Fit temperature scaling parameters on validation logits to calibrate prediction probabilities:

```bash
python backend/ml/calibration/calibrate.py --config config.yaml
```

Saved calibration file:
- `backend/ml/saved_models/ensemble_config.json`

---

### STEP 7: Final Held-Out Test Set Evaluation
Evaluate the winning model checkpoint on the held-out test split:

```bash
python backend/ml/evaluation/evaluate.py --config config.yaml --model efficientnet_b0
```

Outputs:
- `backend/ml/evaluation/results/metrics.json`
- `backend/ml/evaluation/results/confusion_matrix.png`

---

### STEP 8: Run Diagnostic Error Analysis
Identify top confused class pairs, high-confidence misclassifications, and minority class failures:

```bash
python backend/ml/evaluation/error_analysis.py --config config.yaml --model efficientnet_b0
```

Output report:
- `reports/error_analysis.md`
