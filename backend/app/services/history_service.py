"""
Prediction History Service for Python FastAPI backend.
Persists predictions into data/prediction_history.json.
"""
import os
import json
import time
from typing import List, Dict, Any

HISTORY_FILE = "data/prediction_history.json"

def _ensure_history_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

def get_history() -> List[Dict[str, Any]]:
    _ensure_history_file()
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_prediction(result: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_history_file()
    history = get_history()
    
    new_id = len(history) + 1
    record = {
        "id": new_id,
        "prediction": result.get("prediction"),
        "predictionDisplayName": result.get("prediction_display_name"),
        "category": result.get("category"),
        "riskLevel": result.get("risk_level"),
        "confidence": result.get("confidence"),
        "modelName": result.get("model_name"),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    # Prepend newest prediction first
    history.insert(0, record)
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
        
    return record

def delete_prediction(record_id: int) -> bool:
    _ensure_history_file()
    history = get_history()
    filtered = [item for item in history if item.get("id") != record_id]
    if len(filtered) < len(history):
        with open(HISTORY_FILE, "w") as f:
            json.dump(filtered, f, indent=2)
        return True
    return False
