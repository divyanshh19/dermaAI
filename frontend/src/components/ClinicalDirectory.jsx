import React, { useEffect, useState } from "react";
import { api } from "../services/api";

export function ClinicalDirectory() {
  const [classes, setClasses] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchClasses() {
      try {
        const data = await api.getClasses();
        setClasses(data.classes);
      } catch (err) {
        console.error("Failed to load class directory:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchClasses();
  }, []);

  if (loading) {
    return <div className="card"><p style={{ color: "var(--text-muted)" }}>Loading clinical directory...</p></div>;
  }

  const getBadge = (risk) => {
    if (risk === "Critical Risk") return "badge badge-critical";
    if (risk === "High Risk") return "badge badge-high";
    return "badge badge-low";
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.5rem" }}>
      {classes &&
        Object.entries(classes).map(([code, info]) => (
          <div key={code} className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", fontWeight: 700, color: "var(--accent-primary)", textTransform: "uppercase" }}>
                  {code}
                </span>
                <span className={getBadge(info.risk_level)}>{info.risk_level}</span>
              </div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 800, color: "white", marginBottom: "0.5rem" }}>
                {info.display_name}
              </h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1rem", lineHeight: 1.5 }}>
                {info.description}
              </p>
            </div>
            <div style={{ padding: "0.75rem", backgroundColor: "var(--bg-dark)", borderRadius: "8px", border: "1px solid var(--border-color)", fontSize: "0.8rem", color: "#cbd5e1" }}>
              <strong style={{ color: "white" }}>Recommendation: </strong> {info.recommendation}
            </div>
          </div>
        ))}
    </div>
  );
}
