import os
import sys
import pytest
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))

from datasets.dataset import SkinLesionDataset, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

def test_label_mapping():
    assert len(CLASSES) == 7
    assert CLASS_TO_IDX["mel"] == 4
    assert IDX_TO_CLASS[4] == "mel"

def test_dataset_initialization():
    dummy_df = pd.DataFrame({
        "image_id": ["ISIC_0024306", "ISIC_0024307"],
        "image_path": ["ISIC_0024306.jpg", "ISIC_0024307.jpg"],
        "dx": ["mel", "nv"]
    })
    
    dataset = SkinLesionDataset(dummy_df, images_dir="data/ham10000/images", image_size=224, split="train")
    assert len(dataset) == 2
    
    weights = dataset.get_class_weights()
    assert len(weights) == 7
    assert isinstance(weights, torch.Tensor)
