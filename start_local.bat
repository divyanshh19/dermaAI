@echo off
TITLE DermaAI System - Local Microservices Launcher
COLOR 0A

echo =========================================================================
echo                 DermaAI System Local Launcher
echo =========================================================================
echo.
echo Starting 3 Microservices + React Frontend UI...
echo.

:: 1. Start Python PyTorch ML Microservice on Port 8000
echo [1/3] Launching Python PyTorch ML Service (Port 8000)...
start "DermaAI - PyTorch ML Service (8000)" cmd /k "uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend"

:: 2. Start NVIDIA Nemotron AI Chatbot Service on Port 8001
echo [2/3] Launching NVIDIA Nemotron AI Chatbot Service (Port 8001)...
start "DermaAI - Nemotron AI Chatbot (8001)" cmd /k "python chatbot-service/main.py"

:: 3. Start React Frontend UI on Port 5173
echo [3/3] Launching React UI (Port 5173)...
start "DermaAI - React Frontend UI (5173)" cmd /k "cd frontend && npm run dev"

echo.
echo =========================================================================
echo SUCCESS! All microservices launched in separate terminals.
echo Opening http://localhost:5173 in your default browser...
echo =========================================================================
timeout /t 3 >nul
start http://localhost:5173
