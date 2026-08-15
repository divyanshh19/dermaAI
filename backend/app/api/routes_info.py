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
    return HealthResponse(
        status="ok" if inference_service.is_ready else "degraded",
        model_loaded=inference_service.is_ready,
        model_name=inference_service.model_name,
        device=str(inference_service.device),
        version="2.0.0"
    )

@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    return ModelInfoResponse(
        model_name=inference_service.model_name,
        dataset="HAM10000",
        num_classes=len(CLASSES),
        classes=CLASSES,
        confidence_threshold=inference_service.confidence_threshold,
        loss_function=inference_service.config.get("training", {}).get("loss_function", "focal")
    )

@router.get("/classes")
def get_classes_directory():
    return {"classes": DISEASE_INFO}

@router.get("/metrics")
def get_evaluation_metrics():
    metrics_path = "backend/ml/evaluation/results/metrics.json"
    if not os.path.exists(metrics_path):
        # Fallback if evaluation pipeline hasn't run yet on test set
        return {
            "status": "pending",
            "message": "Evaluation metrics not found. Run 'python backend/ml/evaluation/evaluate.py' to generate test set results."
        }
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation metrics: {str(e)}")
