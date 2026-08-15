"""
Model Factory Architecture for Skin Lesion Classification.

Supports:
- CNN Models: EfficientNet-B0, EfficientNet-B2, ConvNeXt-Tiny
- Vision Transformer Models: DeiT-Tiny / ViT-Tiny

Provides pretrained backbone loading, classification head creation, and staged fine-tuning.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

def build_model(model_name: str = "efficientnet_b0", num_classes: int = 7, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    model_name = model_name.lower().strip()
    
    if model_name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    elif model_name == "efficientnet_b2":
        weights = tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b2(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    elif model_name == "convnext_tiny":
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    elif model_name in ("vit_deit_tiny", "deit_tiny", "vit"):
        model = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        class ViTWrapper(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self.num_classes = num_classes

            def forward(self, x):
                feats = self.backbone(x)
                return self.head(feats)

        model = ViTWrapper(model, head)

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Supported: efficientnet_b0, efficientnet_b2, convnext_tiny, vit_deit_tiny")

    return model

def unfreeze_backbone_stage(model: nn.Module, model_name: str, stage: int = 2) -> nn.Module:
    """
    Unfreezes specified blocks/layers for staged fine-tuning:
    - Stage 2: Unfreezes top feature blocks / last 2 transformer blocks.
    - Stage 3: Unfreezes entire backbone for full fine-tuning.
    """
    model_name = model_name.lower().strip()

    if stage >= 3:
        for param in model.parameters():
            param.requires_grad = True
        return model

    if "efficientnet" in model_name:
        # Unfreeze last 2 MBConv blocks (features[-2:])
        for block in model.features[-2:]:
            for param in block.parameters():
                param.requires_grad = True

    elif "convnext" in model_name:
        # Unfreeze last stage block (features[7])
        for param in model.features[7].parameters():
            param.requires_grad = True

    elif "vit" in model_name or "deit" in model_name:
        # Unfreeze last 2 transformer blocks
        for block in model.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.backbone.norm.parameters():
            param.requires_grad = True

    return model
