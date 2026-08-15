import React, { useState } from "react";

export function ImageUploader({ onAnalyze, loading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [explainable, setExplainable] = useState(true);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onAnalyze(selectedFile, explainable);
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>Dermoscopic Image Upload</span>
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>JPEG / PNG / WEBP</span>
      </div>

      {!previewUrl ? (
        <div
          className="dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => document.getElementById("fileInput").click()}
        >
          <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p style={{ fontWeight: 600, color: "white", marginBottom: "0.25rem" }}>
            Click or Drag & Drop Image Here
          </p>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            High-resolution dermoscopic image recommended
          </p>
        </div>
      ) : (
        <div>
          <div className="preview-box">
            <img src={previewUrl} alt="Lesion Preview" className="preview-img" />
          </div>
          <button
            onClick={() => { setSelectedFile(null); setPreviewUrl(null); }}
            style={{
              background: "transparent",
              border: "none",
              color: "#ef4444",
              fontSize: "0.8rem",
              fontWeight: 600,
              cursor: "pointer",
              marginBottom: "1rem"
            }}
          >
            ← Remove / Change Image
          </button>
        </div>
      )}

      <input
        type="file"
        id="fileInput"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />

      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", margin: "1rem 0" }}>
        <input
          type="checkbox"
          id="explainable"
          checked={explainable}
          onChange={(e) => setExplainable(e.target.checked)}
          style={{ width: "16px", height: "16px", accentColor: "var(--accent-primary)", cursor: "pointer" }}
        />
        <label htmlFor="explainable" style={{ fontSize: "0.85rem", color: "var(--text-muted)", cursor: "pointer" }}>
          Generate Grad-CAM Visual Heatmap
        </label>
      </div>

      <button
        className="btn-primary"
        onClick={handleSubmit}
        disabled={!selectedFile || loading}
      >
        {loading ? (
          <>
            <span className="spinner" /> Analyzing Image...
          </>
        ) : (
          "Run AI Lesion Analysis"
        )}
      </button>
    </div>
  );
}
