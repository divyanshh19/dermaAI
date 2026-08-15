"""
Production Core Inference Service with Multi-Model Ensemble & Calibration.

Handles image loading, OpenCV preprocessing, tensor normalization, calibrated ensemble execution,
uncertainty thresholding, and architecture-aware Grad-CAM explainability.
"""
import os
import sys
import json
import io
from typing import Dict, Any, Tuple, Optional, List

import torch
import numpy as np
import cv2
from PIL import Image
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml"))

from models.factory import build_model
from preprocessing.pipeline import get_val_transforms, preprocess_opencv
from explainability.gradcam import generate_gradcam

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

DISEASE_INFO = {
    "akiec": {
        "display_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "category": "Pre-cancerous",
        "risk_level": "High Risk",
        "description": "Scaly or crusty precancerous lesion caused by UV exposure; can progress to Squamous Cell Carcinoma if untreated.",
        "recommendation": "Consult a dermatologist for evaluation and potential cryotherapy or topical treatment."
    },
    "bcc": {
        "display_name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "risk_level": "High Risk",
        "description": "Most common form of skin cancer; slow-growing but can cause localized tissue destruction.",
        "recommendation": "Consult a dermatologist promptly for surgical evaluation or biopsy."
    },
    "bkl": {
        "display_name": "Benign Keratosis-like Lesions",
        "category": "Benign",
        "risk_level": "Low Risk",
        "description": "Non-cancerous growths including seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses.",
        "recommendation": "Generally harmless. Monitor for changes in color, size, or shape."
    },
    "df": {
        "display_name": "Dermatofibroma",
        "category": "Benign",
        "risk_level": "Low Risk",
        "description": "Benign fibrous nodule commonly found on lower extremities; firm button-like feel.",
        "recommendation": "Harmless; no medical intervention required unless symptomatic."
    },
    "mel": {
        "display_name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "risk_level": "Critical Risk",
        "description": "Aggressive, life-threatening skin cancer arising from melanocytes; capability to metastasize rapidly.",
        "recommendation": "URGENT: Schedule an immediate dermatological evaluation and biopsy."
    },
    "nv": {
        "display_name": "Melanocytic Nevi",
        "category": "Benign",
        "risk_level": "Low Risk",
        "description": "Common benign skin mole formed by melanocyte clusters.",
        "recommendation": "Normal lesion. Perform routine self-exams using the ABCDE rule."
    },
    "vasc": {
        "display_name": "Vascular Lesions",
        "category": "Benign",
        "risk_level": "Low Risk",
        "description": "Benign blood vessel proliferations including cherry angiomas and pyogenic granulomas.",
        "recommendation": "Harmless benign lesion; consult a physician if bleeding or expanding."
    }
}

def find_file(path_str: str) -> str:
    candidates = [
        path_str,
        os.path.join("..", path_str),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", path_str),
        os.path.join(os.path.dirname(__file__), "..", "..", path_str)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path_str

class InferenceService:
    def __init__(self, config_path: str = "config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        
        self.config_path = find_file(config_path)
        self.config = self._load_config(self.config_path)
        
        self.primary_model_name = self.config.get("model", {}).get("name", "efficientnet_b0")
        self.confidence_threshold = self.config.get("model", {}).get("confidence_threshold", 0.40)
        self.image_size = self.config.get("training", {}).get("image_size", 224)
        
        self.models: Dict[str, torch.nn.Module] = {}
        self.temperatures: Dict[str, float] = {}
        self.transform = get_val_transforms(self.image_size)

        self._load_all_models()
        self._load_calibration()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def _find_checkpoint_for_model(self, m_name: str) -> Optional[str]:
        candidates = [
            os.path.join("backend/ml/saved_models", f"{m_name}_best.pth"),
            os.path.join("ml/saved_models", f"{m_name}_best.pth"),
            os.path.join("backend/ml/saved_models", f"{m_name}_last.pth"),
            os.path.join("ml/saved_models", f"{m_name}_last.pth"),
            os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models", f"{m_name}_best.pth"),
            os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models", f"{m_name}_last.pth")
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _load_all_models(self):
        target_models = ["efficientnet_b0", "convnext_tiny", "vit_deit_tiny"]
        # Ensure primary model is loaded first
        if self.primary_model_name not in target_models:
            target_models.insert(0, self.primary_model_name)

        for m_name in target_models:
            ckpt_path = self._find_checkpoint_for_model(m_name)
            if ckpt_path and os.path.exists(ckpt_path):
                try:
                    m = build_model(model_name=m_name, num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
                    m.to(self.device).eval()
                    self.models[m_name] = m
                    print(f"SUCCESS: Loaded active model '{m_name}' weights from {ckpt_path}")
                except Exception as e:
                    print(f"Error loading model {m_name}: {e}")

        if not self.models:
            print("Warning: No checkpoints found. Initializing default baseline model.")
            m = build_model(model_name="efficientnet_b0", num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
            m.to(self.device).eval()
            self.models["efficientnet_b0"] = m

    def _load_calibration(self):
        calib_path = find_file("backend/ml/saved_models/ensemble_config.json")
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r") as f:
                    data = json.load(f)
                temps = data.get("temperatures", {})
                for m_name in self.models.keys():
                    t_val = float(temps.get(m_name, 1.0))
                    self.temperatures[m_name] = max(0.1, t_val)
                    print(f"Loaded temperature scaling T={self.temperatures[m_name]:.3f} for {m_name}")
            except Exception as e:
                print(f"Error loading calibration config: {e}")
        else:
            for m_name in self.models.keys():
                self.temperatures[m_name] = 1.0

    @property
    def is_ready(self) -> bool:
        return len(self.models) > 0

    @property
    def primary_model(self) -> torch.nn.Module:
        if self.primary_model_name in self.models:
            return self.models[self.primary_model_name]
        return list(self.models.values())[0]

    @property
    def primary_model_key(self) -> str:
        if self.primary_model_name in self.models:
            return self.primary_model_name
        return list(self.models.keys())[0]

    def process_image(self, image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        display_img_resized = pil_img.resize((self.image_size, self.image_size))
        display_rgb_float = np.array(display_img_resized, dtype=np.float32) / 255.0

        input_tensor = self.transform(pil_img).unsqueeze(0)
        return input_tensor, display_rgb_float

    def predict(self, image_bytes: bytes, generate_explanation: bool = True) -> Dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Model service is not loaded.")

        input_tensor, display_rgb_float = self.process_image(image_bytes)
        input_tensor = input_tensor.to(self.device)

        model_probs_list = []

        with torch.no_grad():
            for m_name, model_inst in self.models.items():
                raw_logits = model_inst(input_tensor)
                t_val = self.temperatures.get(m_name, 1.0)
                calibrated_logits = raw_logits / t_val
                p = torch.softmax(calibrated_logits, dim=1).cpu().numpy()[0]
                model_probs_list.append(p)

        # Average ensemble probabilities across loaded models
        ensemble_probs = np.mean(model_probs_list, axis=0)

        pred_idx = int(np.argmax(ensemble_probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(ensemble_probs[pred_idx])

        is_uncertain = confidence < self.confidence_threshold

        top_indices = np.argsort(ensemble_probs)[::-1][:3]
        top_predictions = [
            {
                "class_code": CLASSES[idx],
                "display_name": DISEASE_INFO[CLASSES[idx]]["display_name"],
                "probability": float(ensemble_probs[idx]),
                "risk_level": DISEASE_INFO[CLASSES[idx]]["risk_level"]
            }
            for idx in top_indices
        ]

        probabilities = {CLASSES[i]: float(ensemble_probs[i]) for i in range(len(CLASSES))}

        explanation_b64 = None
        if generate_explanation:
            try:
                _, explanation_b64 = generate_gradcam(
                    model=self.primary_model,
                    model_name=self.primary_model_key,
                    input_tensor=input_tensor,
                    display_rgb_float=display_rgb_float,
                    target_class=pred_idx
                )
            except Exception as e:
                print(f"Grad-CAM generation warning: {e}")

        uncertainty_message = None
        if is_uncertain:
            uncertainty_message = (
                "Low confidence — image should be reviewed by a qualified medical professional. "
                "The system cannot make a definitive assessment with high certainty."
            )

        active_model_desc = f"Ensemble ({', '.join(self.models.keys())})" if len(self.models) > 1 else self.primary_model_key

        return {
            "prediction": pred_class,
            "prediction_display_name": DISEASE_INFO[pred_class]["display_name"],
            "category": DISEASE_INFO[pred_class]["category"],
            "risk_level": DISEASE_INFO[pred_class]["risk_level"],
            "confidence": confidence,
            "uncertain": is_uncertain,
            "uncertainty_message": uncertainty_message,
            "top_predictions": top_predictions,
            "probabilities": probabilities,
            "disease_info": DISEASE_INFO[pred_class],
            "model_name": active_model_desc,
            "explanation_available": explanation_b64 is not None,
            "gradcam_base64": explanation_b64
        }

# Singleton instance
inference_service = InferenceService()
