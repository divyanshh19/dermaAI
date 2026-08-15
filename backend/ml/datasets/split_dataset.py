"""
Deterministic Lesion-Aware Stratified Dataset Splitter for HAM10000.

Performs a 70% Train / 15% Validation / 15% Test split grouped by lesion_id / patient_id
to guarantee zero data leakage between splits. Saves persistent CSV split files.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def find_config(config_path="config.yaml"):
    candidates = [
        config_path,
        os.path.join("..", config_path),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", config_path)
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f), candidate
    return {}, config_path

def find_metadata_csv(metadata_path):
    candidates = [
        metadata_path,
        os.path.join("..", metadata_path),
        "data/ham10000/metadata.csv",
        "../data/ham10000/metadata.csv",
        "data/ham10000/HAM10000_metadata.csv",
        "../data/ham10000/HAM10000_metadata.csv"
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

def generate_sample_metadata(output_path):
    """Generates sample metadata for demonstration & testing if HAM10000 dataset is not downloaded yet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    sample_size = 350
    lesion_ids = [f"HAM_{i:05d}" for i in range(1, 250)]
    
    # Ensure every class has sufficient representations for stratification (minimum 10 per class)
    dx_list = (
        ["nv"] * 150 +
        ["mel"] * 60 +
        ["bkl"] * 50 +
        ["bcc"] * 30 +
        ["akiec"] * 20 +
        ["vasc"] * 20 +
        ["df"] * 20
    )

    records = []
    for i in range(sample_size):
        lesion = np.random.choice(lesion_ids)
        img_id = f"ISIC_{i+24300:07d}"
        dx = dx_list[i % len(dx_list)]
        records.append({
            "lesion_id": lesion,
            "image_id": img_id,
            "dx": dx,
            "dx_type": "histo",
            "age": float(np.random.randint(20, 80)),
            "sex": np.random.choice(["male", "female"]),
            "localization": np.random.choice(["back", "lower extremity", "trunk", "upper extremity"]),
            "image_path": f"images/{img_id}.jpg"
        })
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Notice: Generated demo metadata CSV at: {output_path} ({len(df)} records across 7 classes)")
    return output_path

def split_dataset(metadata_csv: str, splits_dir: str, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    resolved_metadata = find_metadata_csv(metadata_csv)
    if not resolved_metadata:
        print(f"Metadata file not found at: '{metadata_csv}'.")
        print("To download the full HAM10000 dataset, setup Kaggle API credentials and run:")
        print("    python backend/ml/datasets/download_ham10000.py --out data/ham10000")
        print("\nCreating a sample metadata CSV now so you can immediately test the pipeline...")
        resolved_metadata = generate_sample_metadata(metadata_csv)

    df = pd.read_csv(resolved_metadata)
    print(f"Loaded metadata from {resolved_metadata}. Total records: {len(df)}")

    if "dx" not in df.columns:
        raise KeyError("Metadata CSV must contain 'dx' column for labels.")

    group_col = "lesion_id" if "lesion_id" in df.columns else ("patient_id" if "patient_id" in df.columns else "image_id")
    print(f"Grouping by '{group_col}' to prevent data leakage across splits...")

    lesion_df = df.groupby(group_col)["dx"].agg(lambda x: x.mode()[0]).reset_index()

    val_test_ratio = val_ratio + test_ratio
    train_lesions, val_test_lesions = train_test_split(
        lesion_df,
        test_size=val_test_ratio,
        stratify=lesion_df["dx"],
        random_state=seed
    )

    val_relative_ratio = val_ratio / val_test_ratio
    val_lesions, test_lesions = train_test_split(
        val_test_lesions,
        test_size=1.0 - val_relative_ratio,
        stratify=val_test_lesions["dx"],
        random_state=seed
    )

    train_groups = set(train_lesions[group_col])
    val_groups = set(val_lesions[group_col])
    test_groups = set(test_lesions[group_col])

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    os.makedirs(splits_dir, exist_ok=True)
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n--- Split Generation Summary ---")
    print(f"Train split: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%) -> {train_path}")
    print(f"Val split:   {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%) -> {val_path}")
    print(f"Test split:  {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%) -> {test_path}")

    overlap_train_val = set(train_df[group_col]).intersection(set(val_df[group_col]))
    overlap_train_test = set(train_df[group_col]).intersection(set(test_df[group_col]))
    overlap_val_test = set(val_df[group_col]).intersection(set(test_df[group_col]))
    
    assert len(overlap_train_val) == 0, "DATA LEAKAGE DETECTED between Train and Val!"
    assert len(overlap_train_test) == 0, "DATA LEAKAGE DETECTED between Train and Test!"
    assert len(overlap_val_test) == 0, "DATA LEAKAGE DETECTED between Val and Test!"
    print("Zero lesion leakage verified across all splits!")

    return train_df, val_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg, _ = find_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    
    metadata_path = ds_cfg.get("metadata_csv", "data/ham10000/metadata.csv")
    splits_path = ds_cfg.get("splits_dir", "data/splits")
    seed = ds_cfg.get("seed", 42)

    split_dataset(
        metadata_csv=metadata_path,
        splits_dir=splits_path,
        train_ratio=ds_cfg.get("train_ratio", 0.70),
        val_ratio=ds_cfg.get("val_ratio", 0.15),
        test_ratio=ds_cfg.get("test_ratio", 0.15),
        seed=seed
    )
