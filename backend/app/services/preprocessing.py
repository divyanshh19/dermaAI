"""
API Preprocessing Service for Dermoscopic Image Inputs.
"""
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml"))

from preprocessing.pipeline import get_val_transforms, preprocess_opencv

def preprocess_for_inference(image_bytes: bytes, target_size: int = 224):
    """
    Decodes raw image bytes and prepares:
    1. input_tensor: (1, 3, target_size, target_size) standardized tensor
    2. display_rgb_float: (target_size, target_size, 3) float32 [0, 1] RGB array for Grad-CAM
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode uploaded image file. Ensure valid JPEG, PNG, or WEBP.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    transform = get_val_transforms(target_size)
    input_tensor = transform(pil_img).unsqueeze(0)

    display_resized = pil_img.resize((target_size, target_size))
    display_rgb_float = np.array(display_resized, dtype=np.float32) / 255.0

    return input_tensor, display_rgb_float
