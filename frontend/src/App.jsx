import React, { useState } from "react";
import { ImageUploader } from "./components/ImageUploader";
import { PredictionResult } from "./components/PredictionResult";
import { MetricsDashboard } from "./components/MetricsDashboard";
import { ClinicalDirectory } from "./components/ClinicalDirectory";
import { ChatBot } from "./components/ChatBot";
import { PredictionHistory } from "./components/PredictionHistory";
import { api } from "./services/api";

export default function App() {
  const [activeTab, setActiveTab] = useState("demo");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [chatPredictionContext, setChatPredictionContext] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (file, explainable) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.predictImage(file, explainable);
      setAnalysisResult(data);
    } catch (err) {
      console.error("Analysis Error:", err);
      setError(
        err.response?.data?.message || err.response?.data?.detail || "Prediction request failed. Please wait a moment while the cloud microservices wake up, then try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleAskAI = (result) => {
    setChatPredictionContext(result);
    setActiveTab("chat");
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-title">
          <div className="brand-logo">AI</div>
          <div>
            <h1 className="brand-name">DermaAI System</h1>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Enterprise Spring Boot Gateway & NVIDIA Nemotron AI Medical Assistant
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === "demo" ? "active" : ""}`}
            onClick={() => setActiveTab("demo")}
          >
            Diagnostic Demo
          </button>
          <button
            className={`nav-tab ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            🤖 AI Chatbot
          </button>
          <button
            className={`nav-tab ${activeTab === "history" ? "active" : ""}`}
            onClick={() => setActiveTab("history")}
          >
            📜 Prediction History
          </button>
          <button
            className={`nav-tab ${activeTab === "metrics" ? "active" : ""}`}
            onClick={() => setActiveTab("metrics")}
          >
            Model Info & Metrics
          </button>
          <button
            className={`nav-tab ${activeTab === "directory" ? "active" : ""}`}
            onClick={() => setActiveTab("directory")}
          >
            Clinical Directory
          </button>
        </nav>
      </header>

      {/* Main Content Area */}
      <main>
        {error && (
          <div className="warning-box" style={{ marginBottom: "1.5rem" }}>
            <span>⚠️ {error}</span>
          </div>
        )}

        {activeTab === "demo" && (
          <div className="grid-2col">
            <ImageUploader onAnalyze={handleAnalyze} loading={loading} />
            <PredictionResult result={analysisResult} onAskAI={handleAskAI} />
          </div>
        )}

        {activeTab === "chat" && (
          <ChatBot
            initialPredictionContext={chatPredictionContext}
            onClearContext={() => setChatPredictionContext(null)}
          />
        )}

        {activeTab === "history" && <PredictionHistory />}

        {activeTab === "metrics" && <MetricsDashboard />}

        {activeTab === "directory" && <ClinicalDirectory />}
      </main>

      {/* Medical Disclaimer Footer */}
      <footer className="disclaimer-footer">
        <p style={{ fontWeight: 700, color: "#f8fafc", marginBottom: "0.25rem" }}>
          MEDICAL RESEARCH & EDUCATIONAL DISCLAIMER
        </p>
        <p>
          This system is an AI-based research and educational decision-support prototype trained on the HAM10000 dataset and does NOT constitute a clinical diagnosis. Always consult a licensed healthcare professional or dermatologist for clinical evaluation.
        </p>
      </footer>
    </div>
  );
}
