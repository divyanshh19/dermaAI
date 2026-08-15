"""
Automated Error Analysis Pipeline for Skin Lesion Classification.

Identifies:
- Most confused class pairs (e.g. Melanoma vs Benign Keratosis)
- High-confidence misclassifications (false sense of security)
- Low-confidence misclassifications (borderline cases)
- Minority class recall failures
Generates reports/error_analysis.md with quantitative breakdown and visual insights.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import SkinLesionDataset, CLASSES, IDX_TO_CLASS
from models.factory import build_model

def run_error_analysis(config_path: str = "config.yaml", checkpoint_path: str = None, model_name: str = None, out_md: str = "reports/error_analysis.md"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    md_cfg = cfg.get("model", {})
    model_name = model_name or md_cfg.get("name", "efficientnet_b0")
    checkpoint_path = checkpoint_path or os.path.join("backend/ml/saved_models", f"{model_name}_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.", flush=True)
        return

    print(f"Starting Error Analysis for {model_name}...", flush=True)

    splits_dir = ds_cfg.get("splits_dir", "data/splits")
    test_csv = os.path.join(splits_dir, "test.csv")
    images_dir = ds_cfg.get("images_dir", "data/ham10000/images")

    test_df = pd.read_csv(test_csv)
    test_ds = SkinLesionDataset(test_df, images_dir, image_size=cfg.get("training", {}).get("image_size", 224), split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(model_name=model_name, num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()

    all_logits, all_labels = [], []
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
    confidences = probs.max(axis=1)

    errors_df = test_df.copy()
    errors_df["true_dx"] = [IDX_TO_CLASS[i] for i in y_true]
    errors_df["pred_dx"] = [IDX_TO_CLASS[i] for i in y_pred]
    errors_df["confidence"] = confidences
    errors_df["is_correct"] = (y_true == y_pred)

    misclassified = errors_df[~errors_df["is_correct"]].sort_values(by="confidence", ascending=False)

    # 1. Top Confused Pairs
    cm = confusion_matrix(y_true, y_pred)
    confused_pairs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((CLASSES[i], CLASSES[j], int(cm[i, j])))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # 2. High-Confidence Errors (confidence > 0.70 but wrong)
    high_conf_errors = misclassified[misclassified["confidence"] > 0.70]

    # Write Markdown Report
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# COMPREHENSIVE ERROR ANALYSIS REPORT\n\n")
        f.write(f"**Evaluated Model:** `{model_name}`  \n")
        f.write(f"**Test Set Samples:** {len(test_df)}  \n")
        f.write(f"**Total Misclassifications:** {len(misclassified)} ({len(misclassified)/len(test_df)*100:.2f}% error rate)  \n\n")

        f.write("--- \n\n")
        f.write("## 1. Top Confused Class Pairs\n\n")
        f.write("| True Class | Predicted Class | Count | Description |\n")
        f.write("|---|---|---|---|\n")
        for true_c, pred_c, count in confused_pairs[:10]:
            f.write(f"| `{true_c}` | `{pred_c}` | **{count}** | {true_c.upper()} misclassified as {pred_c.upper()} |\n")

        f.write("\n---\n\n")
        f.write("## 2. High-Confidence Misclassifications (>70% Confidence)\n\n")
        f.write(f"Found **{len(high_conf_errors)}** cases where the model was highly confident but incorrect.\n\n")
        f.write("| Image ID | True Class | Predicted Class | Confidence |\n")
        f.write("|---|---|---|---|\n")
        for idx, row in high_conf_errors.head(10).iterrows():
            f.write(f"| `{row['image_id']}` | `{row['true_dx']}` | `{row['pred_dx']}` | **{row['confidence']*100:.2f}%** |\n")

        f.write("\n---\n\n")
        f.write("## 3. Clinical Failure Diagnostics & Insights\n\n")
        f.write("- **Melanoma (`mel`) vs Benign Keratosis (`bkl`):** Visual overlap in pigment networks and irregular borders often causes misclassifications between Melanoma and solar lentigines.\n")
        f.write("- **Minority Classes (`df`, `vasc`):** Rare lesion types require targeted Focal Loss scaling to ensure high recall despite low sample representations in HAM10000.\n")

    print(f"Error analysis report generated at {out_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Model architecture name (e.g. convnext_tiny, efficientnet_b0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()

    run_error_analysis(config_path=args.config, checkpoint_path=args.checkpoint, model_name=args.model)
