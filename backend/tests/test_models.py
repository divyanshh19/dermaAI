import os
import sys
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))

from models.factory import build_model, unfreeze_backbone_stage

@pytest.mark.parametrize("model_name", ["efficientnet_b0", "efficientnet_b2", "convnext_tiny", "vit_deit_tiny"])
def test_model_forward_pass(model_name):
    model = build_model(model_name=model_name, num_classes=7, pretrained=False, freeze_backbone=True)
    model.eval()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (2, 7)

def test_staged_unfreezing():
    model = build_model("efficientnet_b0", num_classes=7, pretrained=False, freeze_backbone=True)
    model = unfreeze_backbone_stage(model, "efficientnet_b0", stage=2)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0
