"""
FastAPI Chatbot Microservice Entry Point (Port 8001).
Serves POST /chat for NVIDIA Nemotron Medical Assistant.
"""
import os
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.nemotron_service import generate_nemotron_response

app = FastAPI(
    title="NVIDIA Nemotron AI Medical Chatbot Microservice",
    description="Dedicated microservice serving Nemotron 70B LLM inference with medical safety guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionContextDTO(BaseModel):
    prediction: Optional[str] = None
    prediction_display_name: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    risk_level: Optional[str] = None
    top_predictions: Optional[list] = None

class ChatRequestDTO(BaseModel):
    message: str
    conversationId: Optional[str] = None
    predictionContext: Optional[Dict[str, Any]] = None

@app.get("/health")
def health():
    return {"status": "healthy", "service": "NVIDIA Nemotron Chatbot Microservice"}

@app.post("/chat")
def chat(req: ChatRequestDTO):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")
    
    conv_id = req.conversationId or "default"
    response = generate_nemotron_response(
        user_message=req.message,
        conversation_id=conv_id,
        prediction_context=req.predictionContext
    )
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
