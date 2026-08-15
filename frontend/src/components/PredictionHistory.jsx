import React, { useState, useEffect } from "react";
import { api } from "../services/api";

export function PredictionHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getPredictionHistory();
      setHistory(data);
    } catch (err) {
      console.error("History fetch error:", err);
      setError("Failed to fetch prediction history from Spring Boot backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this prediction record from MySQL?")) return;
    try {
      await api.deletePrediction(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      alert("Failed to delete record.");
    }
  };

  const getRiskBadgeClass = (level) => {
    if (level === "Critical Risk") return "badge badge-critical";
    if (level === "High Risk") return "badge badge-high";
    return "badge badge-low";
  };

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "white", margin: 0 }}>
            Prediction History Database
          </h2>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
            Persisted in MySQL via Spring Data JPA
          </p>
        </div>
        <button className="nav-tab" style={{ fontSize: "0.75rem", padding: "0.4rem 0.8rem" }} onClick={fetchHistory}>
          🔄 Refresh
        </button>
      </div>

      {error && <div className="warning-box" style={{ marginBottom: "1rem" }}>⚠️ {error}</div>}

      {loading ? (
        <p style={{ color: "var(--text-muted)" }}>Loading records from Spring Boot backend...</p>
      ) : history.length === 0 ? (
        <div style={{ textAlign: "center", padding: "2rem 0", color: "var(--text-muted)" }}>
          <p>No historical predictions saved yet.</p>
          <p style={{ fontSize: "0.8rem", marginTop: "0.25rem" }}>Perform a diagnostic scan on the main tab to save your first prediction into MySQL.</p>
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", color: "white", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #334155", textAlign: "left", color: "var(--text-muted)" }}>
                <th style={{ padding: "0.75rem" }}>ID</th>
                <th style={{ padding: "0.75rem" }}>Primary Diagnosis</th>
                <th style={{ padding: "0.75rem" }}>Risk Level</th>
                <th style={{ padding: "0.75rem" }}>Confidence</th>
                <th style={{ padding: "0.75rem" }}>Model</th>
                <th style={{ padding: "0.75rem" }}>Date</th>
                <th style={{ padding: "0.75rem", textAlign: "center" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.id} style={{ borderBottom: "1px solid #1e293b" }}>
                  <td style={{ padding: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>#{item.id}</td>
                  <td style={{ padding: "0.75rem", fontWeight: 700 }}>{item.predictionDisplayName || item.prediction}</td>
                  <td style={{ padding: "0.75rem" }}>
                    <span className={getRiskBadgeClass(item.riskLevel)}>{item.riskLevel}</span>
                  </td>
                  <td style={{ padding: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--accent-primary)" }}>
                    {(item.confidence * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: "0.75rem", color: "var(--text-muted)" }}>{item.modelName}</td>
                  <td style={{ padding: "0.75rem", color: "var(--text-muted)" }}>
                    {new Date(item.createdAt).toLocaleString()}
                  </td>
                  <td style={{ padding: "0.75rem", textAlign: "center" }}>
                    <button
                      style={{ background: "rgba(239, 68, 68, 0.2)", border: "1px solid #ef4444", color: "#fca5a5", borderRadius: "4px", padding: "0.25rem 0.5rem", cursor: "pointer", fontSize: "0.75rem" }}
                      onClick={() => handleDelete(item.id)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
