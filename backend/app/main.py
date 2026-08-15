"""
FastAPI Application Server Entry Point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import routes_predict, routes_info

app = FastAPI(
    title="AI Skin Lesion Detection API",
    description="Medical-Grade AI Skin Lesion Classification & Explainability Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_info.router, prefix=settings.api_v1_prefix, tags=["Information & Health"])
app.include_router(routes_predict.router, prefix=settings.api_v1_prefix, tags=["Prediction & Grad-CAM"])

@app.get("/")
def root():
    return {
        "system": "AI Skin Lesion Detection API",
        "version": "2.0.0",
        "docs_url": "/docs"
    }
