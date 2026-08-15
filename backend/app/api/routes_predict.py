"""
Prediction and Explainability API Endpoints with History Persistence.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import PredictionResponse
from app.services.inference import inference_service
from app.services.history_service import save_prediction, get_history, delete_prediction
from app.core.config import settings

router = APIRouter()

ALLOWED_MIME_TYPES = ("image/jpeg", "image/png", "image/webp", "image/jpg")
MAX_FILE_SIZE_MB = 10

def validate_uploaded_file(file: UploadFile, contents: bytes):
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP."
        )
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB."
        )

@router.post("/predict", response_model=PredictionResponse)
async def predict_lesion(file: UploadFile = File(...)):
    contents = await file.read()
    validate_uploaded_file(file, contents)

    if not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="Inference engine is not loaded.")

    try:
        result = inference_service.predict(contents, generate_explanation=False)
        saved = save_prediction(result)
        result["predictionId"] = saved["id"]
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/explain", response_model=PredictionResponse)
async def predict_and_explain_lesion(file: UploadFile = File(...)):
    contents = await file.read()
    validate_uploaded_file(file, contents)

    if not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="Inference engine is not loaded.")

    try:
        result = inference_service.predict(contents, generate_explanation=True)
        saved = save_prediction(result)
        result["predictionId"] = saved["id"]
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction & explainability failed: {str(e)}")

@router.get("/history")
def get_prediction_history():
    return get_history()

@router.delete("/history/{record_id}")
def delete_prediction_history(record_id: int):
    success = delete_prediction(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return {"status": "success", "deleted_id": record_id}
