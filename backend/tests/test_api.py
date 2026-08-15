import os
import sys
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data

def test_model_info_endpoint():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "HAM10000"
    assert data["num_classes"] == 7

def test_classes_endpoint():
    response = client.get("/api/v1/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert "mel" in data["classes"]

def test_invalid_image_type():
    file_bytes = b"fake binary data"
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", file_bytes, "text/plain")}
    )
    assert response.status_code == 415

def test_prediction_endpoint():
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code in (200, 503)
