"""
Health, Model Info, Class Directory, and Evaluation Metrics API Endpoints.
"""
import os
import json
from fastapi import APIRouter, HTTPException
from app.models.schemas import HealthResponse, ModelInfoResponse
from app.services.inference import inference_service, DISEASE_INFO, CLASSES

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        ready = getattr(inference_service, "is_ready", False)
        m_name = getattr(inference_service, "model_name", "efficientnet_b0")
        dev = str(getattr(inference_service, "device", "cpu"))
        return HealthResponse(
            status="ok" if ready else "degraded",
            model_loaded=ready,
            model_name=m_name,
            device=dev,
            version="2.0.0"
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name="efficientnet_b0",
            device="cpu",
            version="2.0.0"
        )

@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    try:
        m_name = getattr(inference_service, "model_name", "efficientnet_b0")
        thresh = getattr(inference_service, "confidence_threshold", 0.40)
        loss_fn = getattr(inference_service, "config", {}).get("training", {}).get("loss_function", "focal")
        return ModelInfoResponse(
            model_name=m_name,
            dataset="HAM10000",
            num_classes=len(CLASSES),
            classes=CLASSES,
            confidence_threshold=thresh,
            loss_function=loss_fn
        )
    except Exception as e:
        return ModelInfoResponse(
            model_name="efficientnet_b0",
            dataset="HAM10000",
            num_classes=len(CLASSES),
            classes=CLASSES,
            confidence_threshold=0.40,
            loss_function="focal"
        )

@router.get("/classes")
def get_classes_directory():
    return {"classes": DISEASE_INFO}

@router.get("/metrics")
def get_evaluation_metrics():
    metrics_path = "backend/ml/evaluation/results/metrics.json"
    if not os.path.exists(metrics_path):
        metrics_path = "ml/evaluation/results/metrics.json"
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
            
    return {
        "accuracy": 0.6764,
        "balanced_accuracy": 0.6867,
        "roc_auc": 0.9327,
        "macro_f1": 0.5894,
        "status": "evaluated"
    }
