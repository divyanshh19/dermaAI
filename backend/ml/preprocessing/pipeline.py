"""
Dermoscopic Image Preprocessing and Augmentation Pipeline.

Provides deterministic validation/testing transforms and moderate, medically realistic
training augmentations (flips, slight rotations, color jitter, ImageNet normalization).
Includes fast OpenCV preprocessing (hair removal, CLAHE contrast enhancement).
"""
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def remove_hair_dullrazor(img_bgr: np.ndarray) -> np.ndarray:
    """DullRazor morphological hair removal filter."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return inpainted

def enhance_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on L channel in LAB color space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocess_opencv(img_bgr: np.ndarray, target_size: int = 224, apply_hair_removal: bool = True) -> np.ndarray:
    """Runs OpenCV hair removal + CLAHE contrast enhancement + BGR to RGB."""
    if apply_hair_removal:
        img_bgr = remove_hair_dullrazor(img_bgr)
    img_bgr = enhance_clahe(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Moderate, medically appropriate augmentations for training."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic evaluation transforms."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def get_test_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic test transforms."""
    return get_val_transforms(image_size)
