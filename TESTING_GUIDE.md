# AI Skin Lesion Detection System — Testing Guide

This guide details how to execute automated unit, integration, and API tests using `pytest`.

---

## 1. Test Suite Overview

The test suite covers:
1. **Dataset Integrity (`test_dataset.py`)**: Validates class mappings, dataset loading, and class weight tensor calculations.
2. **Model Architecture & Forward Pass (`test_models.py`)**: Tests tensor shapes `(batch_size, 7)` for EfficientNet, ConvNeXt, and ViT, and verifies staged backbone unfreezing.
3. **API & Input Validation (`test_api.py`)**: Tests FastAPI `/health`, `/model-info`, `/classes`, `/predict`, file format verification (rejects non-image files with `415 Unsupported Media Type`), and file size bounds.

---

## 2. Running Automated Tests

Run the full pytest suite:

```bash
pytest backend/tests/ -v
```

To run a specific test file:

```bash
pytest backend/tests/test_api.py -v
```

---

## 3. Launching Application Services for Manual Verification

### Start FastAPI Backend Server
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Swagger API documentation will be available at: `http://localhost:8000/docs`

### Start React Frontend Server
```bash
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.
