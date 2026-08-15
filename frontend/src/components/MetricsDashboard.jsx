import React, { useEffect, useState } from "react";
import { api } from "../services/api";

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const data = await api.getMetrics();
        setMetrics(data);
      } catch (err) {
        setError("Could not load evaluation metrics from backend.");
      } finally {
        setLoading(false);
      }
    }
    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="card" style={{ textAlign: "center", padding: "3rem" }}>
        <p style={{ color: "var(--text-muted)" }}>Loading evaluation metrics from test pipeline...</p>
      </div>
    );
  }

  if (error || !metrics || metrics.status === "pending") {
    return (
      <div className="card">
        <h2 className="card-title">Model Performance & Evaluation Metrics</h2>
        <div className="warning-box">
          <p>
            {metrics?.message || error || "Test set evaluation results have not been generated yet."}
          </p>
        </div>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "1rem" }}>
          To generate authentic evaluation metrics on the held-out test split, run:
          <br />
          <code style={{ fontFamily: "var(--font-mono)", color: "#a855f7", display: "inline-block", marginTop: "0.5rem" }}>
            python backend/ml/evaluation/evaluate.py
          </code>
        </p>
      </div>
    );
  }

  const {
    model_name,
    test_samples,
    accuracy,
    balanced_accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
    weighted_f1,
    roc_auc,
    per_class_metrics,
  } = metrics;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Model Specs Card */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <span style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", fontWeight: 700 }}>
              Evaluated Model Architecture
            </span>
            <h2 style={{ fontSize: "1.5rem", fontWeight: 800, color: "white", marginTop: "0.2rem" }}>
              {model_name}
            </h2>
          </div>
          <div style={{ textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--text-muted)" }}>
            <p>Dataset: HAM10000</p>
            <p>Test Samples: {test_samples}</p>
          </div>
        </div>

        {/* Core Metrics Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem", marginTop: "1.5rem" }}>
          <div style={{ padding: "1rem", backgroundColor: "var(--bg-dark)", borderRadius: "12px", border: "1px solid var(--border-color)" }}>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>Overall Accuracy</p>
            <p style={{ fontSize: "1.5rem", fontWeight: 800, color: "#6366f1", marginTop: "0.2rem" }}>
              {(accuracy * 100).toFixed(1)}%
            </p>
          </div>

          <div style={{ padding: "1rem", backgroundColor: "var(--bg-dark)", borderRadius: "12px", border: "1px solid var(--border-color)" }}>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>Balanced Accuracy</p>
            <p style={{ fontSize: "1.5rem", fontWeight: 800, color: "#a855f7", marginTop: "0.2rem" }}>
              {(balanced_accuracy * 100).toFixed(1)}%
            </p>
          </div>

          <div style={{ padding: "1rem", backgroundColor: "var(--bg-dark)", borderRadius: "12px", border: "1px solid var(--border-color)" }}>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>Macro F1-Score</p>
            <p style={{ fontSize: "1.5rem", fontWeight: 800, color: "#10b981", marginTop: "0.2rem" }}>
              {macro_f1.toFixed(4)}
            </p>
          </div>

          <div style={{ padding: "1rem", backgroundColor: "var(--bg-dark)", borderRadius: "12px", border: "1px solid var(--border-color)" }}>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>Macro Recall</p>
            <p style={{ fontSize: "1.5rem", fontWeight: 800, color: "#38bdf8", marginTop: "0.2rem" }}>
              {macro_recall.toFixed(4)}
            </p>
          </div>

          <div style={{ padding: "1rem", backgroundColor: "var(--bg-dark)", borderRadius: "12px", border: "1px solid var(--border-color)" }}>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>Multi-Class ROC-AUC</p>
            <p style={{ fontSize: "1.5rem", fontWeight: 800, color: "#f59e0b", marginTop: "0.2rem" }}>
              {roc_auc ? roc_auc.toFixed(4) : "N/A"}
            </p>
          </div>
        </div>
      </div>

      {/* Per-Class Breakdown Table */}
      {per_class_metrics && (
        <div className="card">
          <h3 className="card-title">Per-Class Performance Breakdown</h3>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Diagnostic Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(per_class_metrics).map(([cls, data]) => (
                <tr key={cls}>
                  <td style={{ fontWeight: 700, color: "white", textTransform: "uppercase" }}>{cls}</td>
                  <td>{(data.precision * 100).toFixed(1)}%</td>
                  <td>{(data.recall * 100).toFixed(1)}%</td>
                  <td style={{ fontWeight: 700, color: "var(--accent-primary)" }}>{data.f1_score.toFixed(4)}</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>{data.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
