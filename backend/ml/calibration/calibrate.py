"""
Post-Hoc Probability Calibration via Temperature Scaling.

Fits optimal scalar temperature T per model on validation set using L-BFGS to optimize
Negative Log-Likelihood (NLL). Ensures T is strictly positive in [0.1, 5.0].
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import SkinLesionDataset, CLASSES
from models.factory import build_model

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Finds optimal scalar temperature T using L-BFGS optimization on NLL loss, strictly clamped to [0.1, 5.0]."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def eval_fn():
        optimizer.zero_grad()
        t_clamped = torch.clamp(temperature, min=0.1, max=5.0)
        loss = criterion(logits_t / t_clamped, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    final_t = float(torch.clamp(temperature, min=0.1, max=5.0).item())
    return final_t

def calibrate_and_tune_ensemble(config_path: str = "config.yaml", out_config: str = "backend/ml/saved_models/ensemble_config.json") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    splits_dir = ds_cfg.get("splits_dir", "data/splits")
    val_csv = os.path.join(splits_dir, "val.csv")
    images_dir = ds_cfg.get("images_dir", "data/ham10000/images")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Val CSV missing at {val_csv}")

    val_df = pd.read_csv(val_csv)
    val_ds = SkinLesionDataset(val_df, images_dir, image_size=cfg.get("training", {}).get("image_size", 224), split="val")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    ckpt_dir = cfg.get("paths", {}).get("saved_models_dir", "backend/ml/saved_models")
    
    temperatures = {}
    models_to_calibrate = ["convnext_tiny", "efficientnet_b0", "vit_deit_tiny"]

    for m_name in models_to_calibrate:
        ckpt_path = os.path.join(ckpt_dir, f"{m_name}_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, f"{m_name}_last.pth")

        if os.path.exists(ckpt_path):
            model = build_model(m_name, num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
            model.to(device).eval()

            logits_list, labels_list = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    logits_list.append(model(x.to(device)).cpu().numpy())
                    labels_list.append(y.numpy())

            logits = np.concatenate(logits_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            t_val = fit_temperature(logits, labels)
            temperatures[m_name] = t_val
            print(f"Fitted Temperature for {m_name}: {t_val:.3f}")
        else:
            temperatures[m_name] = 1.0

    ensemble_config = {
        "cnn_weight": 0.60,
        "vit_weight": 0.40,
        "temperatures": temperatures
    }

    os.makedirs(os.path.dirname(out_config), exist_ok=True)
    with open(out_config, "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"Saved ensemble calibration config to {out_config}")
    return ensemble_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    calibrate_and_tune_ensemble(config_path=args.config)
