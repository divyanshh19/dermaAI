"""
Comprehensive Evaluation Pipeline for Skin Lesion Classification.

Evaluates trained model checkpoints on the held-out test split (test.csv).
Computes:
- Overall Accuracy
- Balanced Accuracy
- Macro Precision, Recall, F1
- Weighted F1
- Multi-class One-vs-Rest ROC-AUC
- Per-class Precision, Recall, F1
Generates:
- Confusion Matrix & Normalized Confusion Matrix plots
- Per-class evaluation report JSON (backend/ml/evaluation/results/metrics.json)
"""
import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import SkinLesionDataset, CLASSES, IDX_TO_CLASS
from models.factory import build_model

def evaluate_checkpoint(
    config_path: str = "config.yaml",
    checkpoint_path: str = None,
    model_name: str = None,
    out_dir: str = "backend/ml/evaluation/results"
) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    md_cfg = cfg.get("model", {})

    model_name = model_name or md_cfg.get("name", "efficientnet_b0")
    checkpoint_path = checkpoint_path or os.path.join("backend/ml/saved_models", f"{model_name}_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    print(f"\n=== Evaluating Model: {model_name} on Held-Out Test Set ===")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Evaluation Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load Test Split
    splits_dir = ds_cfg.get("splits_dir", "data/splits")
    test_csv = os.path.join(splits_dir, "test.csv")
    images_dir = ds_cfg.get("images_dir", "data/ham10000/images")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv}")

    test_df = pd.read_csv(test_csv)
    test_ds = SkinLesionDataset(test_df, images_dir, image_size=cfg.get("training", {}).get("image_size", 224), split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load Model
    model = build_model(model_name=model_name, num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)

    logits_tensor = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits_tensor, dim=1).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_pred = probs.argmax(axis=1)

    # Compute Core Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    try:
        y_true_onehot = np.eye(len(CLASSES))[y_true]
        roc_auc = float(roc_auc_score(y_true_onehot, probs, average="macro", multi_class="ovr"))
    except Exception as e:
        roc_auc = 0.0
        print(f"Warning: OvR ROC-AUC computation failed: {e}")

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class_metrics = {}
    for idx, name in enumerate(CLASSES):
        per_class_metrics[name] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1_score": float(f1_class[idx]),
            "support": int(support_class[idx])
        }

    results = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "test_samples": len(test_df),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "roc_auc": roc_auc,
        "per_class_metrics": per_class_metrics
    }

    print("\n-----------------------------------------------------")
    print(f" Overall Accuracy:      {acc*100:.2f}%")
    print(f" Balanced Accuracy:     {bal_acc*100:.2f}%")
    print(f" Macro Precision:       {p_macro:.4f}")
    print(f" Macro Recall:          {r_macro:.4f}")
    print(f" Macro F1 Score:        {f1_macro:.4f}")
    print(f" Multi-Class ROC-AUC:   {roc_auc:.4f}")
    print("-----------------------------------------------------\n")

    # Save JSON Report
    os.makedirs(out_dir, exist_ok=True)
    metrics_json_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics JSON to {metrics_json_path}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix ({model_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title(f"Normalized Confusion Matrix ({model_name})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    cm_plot_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    evaluate_checkpoint(config_path=args.config, checkpoint_path=args.checkpoint, model_name=args.model)
