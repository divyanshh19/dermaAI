# MODEL AUDIT & ARCHITECTURAL REDESIGN REPORT

**Project:** AI Skin Lesion Detection System  
**Dataset:** HAM10000 (7 diagnostic classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`)  
**Date:** August 2026  

---

## 1. Executive Summary & Problem Diagnosis

The original implementation of the AI Skin Lesion Detection System suffered from poor classification accuracy, high variance, weak generalization, and uncalibrated confidence scores. A thorough code audit identified several root causes spanning data pipeline design, transfer learning execution, class imbalance strategy, inference serving disconnects, and evaluation metrics.

### Key Flaws Identified in Existing Codebase

1. **Dynamic / Unpersisted Dataset Splitting (Data Leakage Risk)**
   - `build_splits()` performed random splitting on the fly inside training scripts (`train_cnn.py`, `train_vit.py`, `evaluate.py`).
   - Split partitions were never exported to static files (e.g., `train.csv`, `val.csv`, `test.csv`). This caused slight variations in train/val/test membership across separate runs and model types, compromising reproducibility and introducing potential test set contamination.

2. **Flawed Transfer Learning & Frozen Backbone Strategy**
   - The initial training phase froze the entire backbone (`requires_grad = False`).
   - Fine-tuning only unfroze the single final MBConv block (EfficientNet-B0) or last 2 blocks (DeiT-tiny) for 3 epochs at a fixed low learning rate.
   - Low/mid-level feature extractors remained domain-locked to standard ImageNet textures rather than adapting to dermatoscopic skin features (e.g., pigment networks, streaks, dots/globules).

3. **Inappropriate Class Imbalance & Loss Function Combination**
   - Combined `WeightedRandomSampler` with class-weighted `CrossEntropyLoss`.
   - **Double-weighting flaw:** Oversampling minority samples in dataloaders *while simultaneously* scaling loss gradients created severe over-penalization for majority class (`nv`) samples, leading to high false-positive rates for rare classes and overall degradation.
   - No implementation of Focal Loss, Class-Balanced Loss, or Label Smoothing to address hard samples and overconfidence.

4. **Metrics Mismatch & Model Selection Bias**
   - Optimization and early stopping targeted overall dataset Accuracy.
   - In HAM10000, ~67% of all samples are `nv` (Melanocytic Nevi). A naive predictor outputting always `nv` achieves ~67% accuracy but 0% recall on life-threatening conditions like `mel` (Melanoma).
   - Validation selection must be driven by **Macro F1** and **Balanced Accuracy**.

5. **Inference Disconnect & Incomplete Architecture**
   - In `backend/app/services/inference.py`, the Vision Transformer loading code was commented out, returning hardcoded `"Not Trained"` warnings while claiming to run an ensemble.
   - Missing probability calibration (Temperature Scaling) led to overconfident raw softmax outputs.
   - Lacked an uncertainty/abstention threshold to flag ambiguous images for clinical review.

6. **Hardcoded Execution Hardware**
   - Scripts contained hardcoded `device = torch.device("cpu")` statements, preventing GPU acceleration (CUDA / MPS) even when hardware was available.

---

## 2. Re-Architectural Solutions Implemented

| Area | Old Implementation | Re-Architected Implementation |
|---|---|---|
| **Data Splitting** | On-the-fly random split grouping by `lesion_id` | Patient/lesion-level stratified split (70% train / 15% val / 15% test) saved deterministically to `data/splits/` CSV files. |
| **Data Quality** | Manual inspection | Automated `data_validation.py` checking duplicate hashes, corrupt files, split leakage, and class distributions. |
| **Backbone Models** | EfficientNet-B0 + DeiT-Tiny (frozen) | Modular model factory supporting `EfficientNet-B0`, `EfficientNet-B2`, `ConvNeXt-Tiny`, and `DeiT-Tiny / ViT`. |
| **Fine-Tuning Strategy** | 2-phase frozen/light unfreeze | Staged fine-tuning with discriminative learning rates ($10^{-3}$ for classifier head, $10^{-5}$ for backbone). |
| **Loss & Balancing** | Weighted Sampler + Weighted CE | Controlled experiments comparing Weighted CE, Focal Loss, and Label Smoothing; oversampling restricted strictly to training set. |
| **Target Metric** | Overall Accuracy | **Macro F1** primary metric, supported by **Balanced Accuracy** and OvR ROC-AUC. |
| **Hardware Execution** | Forced CPU | Automatic hardware detection: CUDA GPU with AMP (Automatic Mixed Precision) $\rightarrow$ Apple MPS $\rightarrow$ CPU fallback. |
| **Probability Calibration**| Raw Softmax outputs | Post-hoc Temperature Scaling fitted on validation set to minimize Expected Calibration Error (ECE). |
| **Safety / Abstention** | Forced prediction on all inputs | Configurable uncertainty threshold flagging low-confidence predictions for medical professional review. |
| **Serving & UI** | Partial inference; basic components | Fully integrated FastAPI backend with Grad-CAM overlays + responsive React medical-AI interface with live metrics dashboard. |

---

## 3. Class Mapping & Clinical Definitions

| Class Code | Full Medical Name | Category | Risk Level |
|---|---|---|---|
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma | Pre-cancerous | High Risk |
| `bcc` | Basal Cell Carcinoma | Malignant Skin Cancer | High Risk |
| `bkl` | Benign Keratosis-like Lesions | Benign | Low Risk |
| `df` | Dermatofibroma | Benign | Low Risk |
| `mel` | Melanoma | Malignant Skin Cancer | Critical Risk |
| `nv` | Melanocytic Nevi (Moles) | Benign | Low Risk |
| `vasc` | Vascular Lesions | Benign | Low Risk |

---

## 4. Conclusion

This re-architected design transforms the pipeline into a rigorous, reproducible Machine Learning system adhering to medical imaging best practices.
