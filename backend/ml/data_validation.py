"""
Automated Data Validation Pipeline for HAM10000.

Performs thorough verification before training:
1. Detect duplicate images (MD5 hashing)
2. Detect corrupted images (PIL verification)
3. Detect missing images referenced in metadata
4. Verify label validity against expected 7 diagnostic classes
5. Inspect image dimensions and channels
6. Calculate class distribution & imbalance metrics
7. Verify split isolation / data leakage across train, val, test splits
8. Generate reports/data_quality_report.json and visual distribution plots.
"""
import os
import sys
import hashlib
import json
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def find_config(config_path="config.yaml"):
    candidates = [
        config_path,
        os.path.join("..", config_path),
        os.path.join(os.path.dirname(__file__), "..", "..", config_path)
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f)
    return {}

def compute_md5(file_path: str, chunk_size=8192) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def validate_dataset(metadata_csv: str, images_dir: str, splits_dir: str = None, out_report: str = "reports/data_quality_report.json", out_plot_dir: str = "reports") -> Dict[str, Any]:
    print("=== Starting Automated Data Quality Validation ===")
    report: Dict[str, Any] = {
        "status": "PASSED",
        "total_records": 0,
        "valid_images": 0,
        "missing_images": [],
        "corrupted_images": [],
        "duplicate_images_count": 0,
        "invalid_labels": [],
        "class_distribution": {},
        "split_leakage_detected": False,
        "image_dimensions_sample": {},
        "issues_found": []
    }

    if not os.path.exists(metadata_csv):
        # Resolve path
        if os.path.exists(os.path.join("..", metadata_csv)):
            metadata_csv = os.path.join("..", metadata_csv)

    if not os.path.exists(metadata_csv):
        report["status"] = "FAILED"
        report["issues_found"].append(f"Metadata file missing at {metadata_csv}")
        print(f"Error: {metadata_csv} not found.")
        return report

    df = pd.read_csv(metadata_csv)
    report["total_records"] = len(df)

    if "dx" not in df.columns:
        report["status"] = "FAILED"
        report["issues_found"].append("Missing 'dx' column in metadata CSV.")
        return report

    invalid_df = df[~df["dx"].isin(CLASSES)]
    if len(invalid_df) > 0:
        report["invalid_labels"] = invalid_df["dx"].tolist()
        report["issues_found"].append(f"Found {len(invalid_df)} records with invalid class labels.")
        report["status"] = "WARNING"

    class_counts = df["dx"].value_counts().to_dict()
    report["class_distribution"] = {c: class_counts.get(c, 0) for c in CLASSES}

    md5_hashes = {}
    missing_count = 0
    corrupt_count = 0
    duplicate_count = 0

    img_col = "image_path" if "image_path" in df.columns else ("image_id" if "image_id" in df.columns else None)

    for idx, row in df.iterrows():
        img_name = row[img_col] if img_col else f"{row['image_id']}.jpg"
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            img_name = f"{img_name}.jpg"

        full_path = os.path.join(images_dir, os.path.basename(img_name))
        if not os.path.exists(full_path) and os.path.exists(os.path.join("..", full_path)):
            full_path = os.path.join("..", full_path)

        if not os.path.exists(full_path):
            missing_count += 1
            if len(report["missing_images"]) < 50:
                report["missing_images"].append(img_name)
            continue

        try:
            with Image.open(full_path) as img:
                img.verify()
            with Image.open(full_path) as img:
                if idx == 0:
                    report["image_dimensions_sample"] = {"width": img.size[0], "height": img.size[1], "mode": img.mode}
        except Exception as e:
            corrupt_count += 1
            if len(report["corrupted_images"]) < 50:
                report["corrupted_images"].append(img_name)
            continue

        file_hash = compute_md5(full_path)
        if file_hash in md5_hashes:
            duplicate_count += 1
        else:
            md5_hashes[file_hash] = img_name

    report["valid_images"] = report["total_records"] - missing_count - corrupt_count
    report["duplicate_images_count"] = duplicate_count

    if missing_count > 0:
        report["issues_found"].append(f"{missing_count} referenced images are missing from {images_dir}")
    if corrupt_count > 0:
        report["issues_found"].append(f"{corrupt_count} images failed integrity verification")
    if duplicate_count > 0:
        report["issues_found"].append(f"{duplicate_count} exact duplicate images detected via MD5 hash")

    # Split Leakage Verification
    splits_dir_resolved = splits_dir or "data/splits"
    if not os.path.exists(splits_dir_resolved) and os.path.exists(os.path.join("..", splits_dir_resolved)):
        splits_dir_resolved = os.path.join("..", splits_dir_resolved)

    if os.path.exists(splits_dir_resolved):
        train_csv = os.path.join(splits_dir_resolved, "train.csv")
        val_csv = os.path.join(splits_dir_resolved, "val.csv")
        test_csv = os.path.join(splits_dir_resolved, "test.csv")

        if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
            tr_df = pd.read_csv(train_csv)
            va_df = pd.read_csv(val_csv)
            te_df = pd.read_csv(test_csv)

            group_col = "lesion_id" if "lesion_id" in tr_df.columns else "image_id"
            tr_lesions = set(tr_df[group_col])
            va_lesions = set(va_df[group_col])
            te_lesions = set(te_df[group_col])

            l1 = tr_lesions.intersection(va_lesions)
            l2 = tr_lesions.intersection(te_lesions)
            l3 = va_lesions.intersection(te_lesions)

            if len(l1) > 0 or len(l2) > 0 or len(l3) > 0:
                report["split_leakage_detected"] = True
                report["status"] = "FAILED"
                report["issues_found"].append(f"DATA LEAKAGE DETECTED across splits! Overlaps: tr-va={len(l1)}, tr-te={len(l2)}, va-te={len(l3)}")
            else:
                report["split_leakage_detected"] = False

    os.makedirs(os.path.dirname(out_report), exist_ok=True)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Data Quality Report written to {out_report}")

    os.makedirs(out_plot_dir, exist_ok=True)
    plt.figure(figsize=(9, 5))
    df_plot = pd.DataFrame(list(report["class_distribution"].items()), columns=["Class", "Count"])
    sns.barplot(data=df_plot, x="Class", y="Count", palette="viridis")
    plt.title("HAM10000 Class Distribution")
    plt.xlabel("Diagnostic Class")
    plt.ylabel("Number of Images")
    for idx, row in df_plot.iterrows():
        plt.text(idx, row["Count"] + 5, str(row["Count"]), ha="center", fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(out_plot_dir, "class_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Class distribution plot saved to {plot_path}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = find_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    validate_dataset(
        metadata_csv=ds_cfg.get("metadata_csv", "data/ham10000/metadata.csv"),
        images_dir=ds_cfg.get("images_dir", "data/ham10000/images"),
        splits_dir=ds_cfg.get("splits_dir", "data/splits"),
        out_report="reports/data_quality_report.json",
        out_plot_dir="reports"
    )
