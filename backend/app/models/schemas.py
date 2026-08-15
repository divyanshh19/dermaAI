"""
Pydantic API Schemas for AI Skin Lesion Detection System.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class TopPrediction(BaseModel):
    class_code: str
    display_name: str
    probability: float
    risk_level: str

class DiseaseInfo(BaseModel):
    display_name: str
    category: str
    risk_level: str
    description: str
    recommendation: str

class PredictionResponse(BaseModel):
    prediction: str
    prediction_display_name: str
    category: str
    risk_level: str
    confidence: float
    uncertain: bool
    uncertainty_message: Optional[str] = None
    top_predictions: List[TopPrediction]
    probabilities: Dict[str, float]
    disease_info: DiseaseInfo
    model_name: str
    explanation_available: bool
    gradcam_base64: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    version: str = "2.0.0"

class ModelInfoResponse(BaseModel):
    model_name: str
    dataset: str = "HAM10000"
    num_classes: int = 7
    classes: List[str]
    confidence_threshold: float
    loss_function: str
