"""
Unified Model Training Framework with Hardware Auto-Detection & Experiment Tracking.

Features:
- Hardware detection (CUDA GPU with AMP -> Apple MPS -> CPU fallback)
- Staged fine-tuning (Classifier head training -> Backbone fine-tuning with discriminative LR)
- Macro F1 metric early stopping
- Configurable loss functions (CrossEntropy, Weighted CE, Focal Loss, Label Smoothing)
- Checkpoint saving (best_model.pth, last_model.pth)
- Persistent experiment logging to experiments/results.csv
"""
import os
import sys
import time
import json
import csv
import argparse
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import SkinLesionDataset, make_weighted_sampler, CLASSES
from models.factory import build_model, unfreeze_backbone_stage
from training.losses import get_loss_function

def find_file(path_str):
    candidates = [
        path_str,
        os.path.join("..", path_str),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", path_str)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path_str

def load_config(config_path="config.yaml"):
    resolved_config = find_file(config_path)
    if os.path.exists(resolved_config):
        with open(resolved_config, "r") as f:
            return yaml.safe_load(f), resolved_config
    return {}, config_path

def get_device_info() -> Tuple[torch.device, bool, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        use_amp = True
        desc = f"CUDA GPU ({gpu_name}, VRAM: {vram})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False
        desc = "Apple MPS (Metal Performance Shaders)"
    else:
        device = torch.device("cpu")
        use_amp = False
        desc = f"CPU (Threads: {torch.get_num_threads()})"

    return device, use_amp, desc

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = -1.0
        self.counter = 0
        self.should_stop = False

    def step(self, val_f1: float) -> bool:
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                if is_train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    total_samples = len(loader.dataset)
    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1

def train_experiment(config_path: str = "config.yaml", model_name_override: str = None, loss_override: str = None):
    cfg, _ = load_config(config_path)

    ds_cfg = cfg.get("dataset", {})
    tr_cfg = cfg.get("training", {})
    md_cfg = cfg.get("model", {})
    pt_cfg = cfg.get("paths", {})

    model_name = model_name_override or md_cfg.get("name", "efficientnet_b0")
    loss_name = loss_override or tr_cfg.get("loss_function", "focal")
    batch_size = tr_cfg.get("batch_size", 64)
    image_size = tr_cfg.get("image_size", 224)
    head_epochs = tr_cfg.get("head_epochs", 3)
    fine_tune_epochs = tr_cfg.get("fine_tune_epochs", 5)
    head_lr = tr_cfg.get("head_lr", 1e-3)
    backbone_lr = tr_cfg.get("backbone_lr", 1e-4)
    patience = tr_cfg.get("patience", 4)

    device, use_amp, device_desc = get_device_info()
    print("\n=======================================================", flush=True)
    print(f"       AI SKIN DISEASE DETECTION MODEL TRAINING        ", flush=True)
    print("=======================================================", flush=True)
    print(f"Device:               {device_desc}", flush=True)
    print(f"Model Architecture:   {model_name}", flush=True)
    print(f"Loss Function:        {loss_name}", flush=True)
    print(f"Batch Size:           {batch_size}", flush=True)
    print(f"Mixed Precision AMP:  {use_amp}", flush=True)
    print("=======================================================\n", flush=True)

    splits_dir = find_file(ds_cfg.get("splits_dir", "data/splits"))
    images_dir = find_file(ds_cfg.get("images_dir", "data/ham10000/images"))

    train_csv = find_file(os.path.join(splits_dir, "train.csv"))
    val_csv = find_file(os.path.join(splits_dir, "val.csv"))

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("Split files missing. Run split_dataset.py first!")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_ds = SkinLesionDataset(train_df, images_dir, image_size=image_size, split="train")
    val_ds = SkinLesionDataset(val_df, images_dir, image_size=image_size, split="val")

    if loss_name == "weighted_sampling":
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        class_weights = None
        criterion = get_loss_function("cross_entropy")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        class_weights = train_ds.get_class_weights()
        criterion = get_loss_function(loss_name, class_weights=class_weights, gamma=tr_cfg.get("focal_gamma", 2.0), smoothing=tr_cfg.get("label_smoothing", 0.1))

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name=model_name, num_classes=len(CLASSES), pretrained=True, freeze_backbone=True).to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    optimizer_head = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr, weight_decay=tr_cfg.get("weight_decay", 1e-4))
    scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_head, T_max=head_epochs)

    saved_models_dir = find_file(pt_cfg.get("saved_models_dir", "backend/ml/saved_models"))
    os.makedirs(saved_models_dir, exist_ok=True)
    best_ckpt_path = os.path.join(saved_models_dir, f"{model_name}_best.pth")
    last_ckpt_path = os.path.join(saved_models_dir, f"{model_name}_last.pth")

    best_val_f1 = -1.0
    best_epoch = 0
    stopper = EarlyStopping(patience=patience)

    print("--- Stage 1: Classifier Head Training ---", flush=True)
    for epoch in range(1, head_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer_head, scaler, device, use_amp)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, None, None, device, use_amp)
        scheduler_head.step()
        elapsed = time.time() - t0

        print(f"[Stage 1 - Epoch {epoch}/{head_epochs}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} ({elapsed:.1f}s)", flush=True)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            torch.save({"model_name": model_name, "model_state_dict": model.state_dict(), "val_f1": va_f1, "val_acc": va_acc, "epoch": epoch}, best_ckpt_path)

    print("\n--- Stage 2: Unfreezing Top Backbone Layers & Fine-Tuning ---", flush=True)
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        print(f"Loaded best Stage 1 model weights (Val F1: {best_val_f1:.4f}) before starting Stage 2.", flush=True)

    model = unfreeze_backbone_stage(model, model_name=model_name, stage=2)
    
    if hasattr(model, "classifier"):
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]
    elif hasattr(model, "head"):
        backbone_params = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    else:
        backbone_params = [p for p in model.parameters() if p.requires_grad]
        head_params = []

    optimizer_ft = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr * 0.1}
    ], weight_decay=tr_cfg.get("weight_decay", 1e-4))

    scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode="max", factor=0.5, patience=2)

    for epoch in range(head_epochs + 1, head_epochs + fine_tune_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer_ft, scaler, device, use_amp)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, None, None, device, use_amp)
        scheduler_ft.step(va_f1)
        elapsed = time.time() - t0

        print(f"[Stage 2 - Epoch {epoch}/{head_epochs + fine_tune_epochs}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} ({elapsed:.1f}s)", flush=True)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            torch.save({"model_name": model_name, "model_state_dict": model.state_dict(), "val_f1": va_f1, "val_acc": va_acc, "epoch": epoch}, best_ckpt_path)
            print(f"  -> New best Macro F1: {best_val_f1:.4f} (Saved to {best_ckpt_path})", flush=True)

        if stopper.step(va_f1):
            print(f"Early stopping triggered at epoch {epoch}.", flush=True)
            break

    torch.save({"model_name": model_name, "model_state_dict": model.state_dict(), "val_f1": va_f1, "val_acc": va_acc, "epoch": epoch}, last_ckpt_path)

    exp_dir = find_file(pt_cfg.get("experiments_dir", "experiments"))
    os.makedirs(exp_dir, exist_ok=True)
    results_csv = os.path.join(exp_dir, "results.csv")
    file_exists = os.path.exists(results_csv)

    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "loss", "batch_size", "image_size", "best_epoch", "best_val_macro_f1", "checkpoint_path"])
        writer.writerow([model_name, loss_name, batch_size, image_size, best_epoch, f"{best_val_f1:.4f}", best_ckpt_path])

    print(f"\nExperiment logged to {results_csv}. Training Complete!", flush=True)
    return best_ckpt_path, best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model architecture")
    parser.add_argument("--loss", type=str, default=None, help="Override loss function")
    args = parser.parse_args()

    train_experiment(config_path=args.config, model_name_override=args.model, loss_override=args.loss)
