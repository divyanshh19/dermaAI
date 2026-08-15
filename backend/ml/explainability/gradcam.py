"""
Architecture-Aware Grad-CAM & Attention Visualization Service.

Supports:
- CNNs (EfficientNet-B0, EfficientNet-B2, ConvNeXt-Tiny): target last feature block
- ViTs (DeiT-Tiny): spatial patch token transformation

Outputs:
- Raw normalized heatmap [0, 1]
- Colorized RGB overlay (uint8)
- Base64 encoded PNG string
"""
import base64
import io
from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

def _vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Reshapes DeiT patch tokens (B, N, C) -> (B, C, H, W), dropping [CLS] token."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_target_layer(model: nn.Module, model_name: str) -> list:
    model_name = model_name.lower().strip()
    
    if "efficientnet" in model_name:
        return [model.features[-1]]
    elif "convnext" in model_name:
        return [model.features[7]]
    elif "vit" in model_name or "deit" in model_name:
        return [model.backbone.blocks[-1].norm1]
    else:
        # Fallback to last convolutional layer found
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]
        raise ValueError(f"Could not automatically locate target layer for {model_name}")

def generate_gradcam(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    display_rgb_float: np.ndarray,
    target_class: int
) -> Tuple[np.ndarray, str]:
    """
    Generate Grad-CAM visualization overlay.
    - display_rgb_float: HxWx3 array in [0, 1] range (unnormalized image)
    - target_class: integer class index
    Returns: (overlay_rgb_uint8, base64_png_str)
    """
    if not HAS_GRAD_CAM:
        raise RuntimeError("pytorch_grad_cam library not installed.")

    model.eval()
    target_layers = get_target_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_class)]

    is_vit = "vit" in model_name or "deit" in model_name
    reshape_tf = _vit_reshape_transform if is_vit else None

    cam_kwargs = {"model": model, "target_layers": target_layers}
    if reshape_tf:
        cam_kwargs["reshape_transform"] = reshape_tf

    with GradCAM(**cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Mask thresholding to highlight lesion-specific regions
    grayscale_cam = np.clip(grayscale_cam, 0, 1)

    # Generate color overlay
    overlay_uint8 = show_cam_on_image(display_rgb_float, grayscale_cam, use_rgb=True)

    # Encode to Base64
    pil_img = Image.fromarray(overlay_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return overlay_uint8, b64_str
