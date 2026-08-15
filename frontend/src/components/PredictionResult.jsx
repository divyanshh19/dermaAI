import React from "react";

export function PredictionResult({ result, onAskAI }) {
  if (!result) {
    return (
      <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "360px", color: "var(--text-muted)", textAlign: "center" }}>
        <svg style={{ width: "64px", height: "64px", opacity: 0.3, marginBottom: "1rem" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p style={{ fontWeight: 600 }}>No Analysis Performed Yet</p>
        <p style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>Upload a skin lesion image and click "Run AI Lesion Analysis" to view clinical predictions.</p>
      </div>
    );
  }

  const {
    prediction_display_name,
    category,
    risk_level,
    confidence,
    uncertain,
    uncertainty_message,
    top_predictions,
    disease_info,
    gradcam_base64,
    model_name,
  } = result;

  const getRiskBadgeClass = (level) => {
    if (level === "Critical Risk") return "badge badge-critical";
    if (level === "High Risk") return "badge badge-high";
    return "badge badge-low";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Primary Diagnosis Header Card */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
          <div>
            <span style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", fontWeight: 700 }}>
              Primary Diagnosis
            </span>
            <h2 style={{ fontSize: "1.5rem", fontWeight: 800, color: "white", marginTop: "0.2rem" }}>
              {prediction_display_name}
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>{category}</p>
          </div>
          <span className={getRiskBadgeClass(risk_level)}>{risk_level}</span>
        </div>

        {/* Confidence Progress Meter */}
        <div style={{ marginTop: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.35rem" }}>
            <span>Calibrated Confidence</span>
            <span style={{ color: "var(--accent-primary)" }}>{(confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill" style={{ width: `${confidence * 100}%` }} />
          </div>
        </div>

        {/* Ask AI About This Result Action Button */}
        <div style={{ marginTop: "1.25rem" }}>
          <button
            className="nav-tab active"
            style={{ width: "100%", padding: "0.75rem 1rem", borderRadius: "8px", fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem", fontSize: "0.9rem" }}
            onClick={() => onAskAI && onAskAI(result)}
          >
            <span>🤖 Ask AI About This Result</span>
          </button>
        </div>

        {/* Low Confidence Uncertainty Warning Banner */}
        {uncertain && (
          <div className="warning-box" style={{ marginTop: "1.25rem" }}>
            <svg style={{ width: "24px", height: "24px", flexShrink: 0 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <p style={{ fontWeight: 700, marginBottom: "0.2rem" }}>Low Confidence Prediction Alert</p>
              <p>{uncertainty_message}</p>
            </div>
          </div>
        )}
      </div>

      {/* Top 3 Predictions Breakdown */}
      <div className="card">
        <h3 className="card-title">Top Differential Diagnoses</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {top_predictions && top_predictions.map((item, idx) => (
            <div key={idx}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.25rem" }}>
                <span style={{ fontWeight: 600, color: "white" }}>
                  {idx + 1}. {item.display_name}
                </span>
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                  {(item.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="progress-bar-bg">
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${item.probability * 100}%`,
                    background: idx === 0 ? "linear-gradient(90deg, #6366f1, #a855f7)" : "#475569"
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Clinical Details & Recommendation */}
      {disease_info && (
        <div className="card">
          <h3 className="card-title">Clinical Recommendations</h3>
          <p style={{ fontSize: "0.9rem", color: "#e2e8f0", marginBottom: "1rem", lineHeight: 1.6 }}>
            {disease_info.description}
          </p>
          <div style={{ padding: "1rem", backgroundColor: "rgba(99, 102, 241, 0.1)", borderRadius: "10px", borderLeft: "4px solid var(--accent-primary)" }}>
            <p style={{ fontSize: "0.85rem", fontWeight: 700, color: "white", marginBottom: "0.2rem" }}>Action Plan:</p>
            <p style={{ fontSize: "0.85rem", color: "#c7d2fe" }}>{disease_info.recommendation}</p>
          </div>
        </div>
      )}

      {/* Grad-CAM Heatmap Explainability */}
      {gradcam_base64 && (
        <div className="card">
          <div className="card-title">
            <span>Grad-CAM Visual Heatmap</span>
            <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--accent-primary)" }}>
              Model: {model_name}
            </span>
          </div>

          <div style={{ width: "100%", height: "280px", borderRadius: "12px", overflow: "hidden", backgroundColor: "var(--bg-dark)" }}>
            <img
              src={`data:image/png;base64,${gradcam_base64}`}
              alt="Grad-CAM Overlay"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          </div>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.75rem", textAlign: "center" }}>
            Highlighted warm regions (red/yellow) indicate anatomical visual features influencing the neural network's classification.
          </p>
        </div>
      )}
    </div>
  );
}
