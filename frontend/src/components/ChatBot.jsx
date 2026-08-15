import React, { useState, useEffect, useRef } from "react";
import { api } from "../services/api";

export function ChatBot({ initialPredictionContext, onClearContext }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      content:
        "Hello! I am your **DermaAI Nemotron Health Assistant**. You can ask me general questions about skin health, skin lesion predictions, or confidence scores.\n\n⚠️ *Disclaimer: I provide educational decision-support information only and do not provide medical diagnoses.*",
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeContext, setActiveContext] = useState(initialPredictionContext);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (initialPredictionContext) {
      setActiveContext(initialPredictionContext);
      const predName = initialPredictionContext.prediction_display_name || initialPredictionContext.prediction;
      const conf = (initialPredictionContext.confidence * 100).toFixed(1);
      
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          content: `📌 **Lesion Context Auto-Loaded:**\n- **Primary Diagnosis:** ${predName}\n- **Calibrated Confidence:** ${conf}%\n- **Risk Level:** ${initialPredictionContext.risk_level}\n\nYou can now ask me any questions regarding this screening prediction!`,
        },
      ]);
    }
  }, [initialPredictionContext]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (customPrompt = null) => {
    const textToSend = customPrompt || inputMessage;
    if (!textToSend.trim()) return;

    const userMsg = { id: Date.now(), role: "user", content: textToSend };
    setMessages((prev) => [...prev, userMsg]);
    if (!customPrompt) setInputMessage("");
    setLoading(true);
    setError(null);

    try {
      const response = await api.sendChatMessage(textToSend, conversationId, activeContext);
      setConversationId(response.conversationId);

      const botMsg = {
        id: Date.now() + 1,
        role: "assistant",
        content: response.message,
        model: response.model,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error("Chat Error:", err);
      setError(
        err.response?.data?.message || "Failed to reach Nemotron Chatbot microservice. Ensure Spring Boot & Chatbot services are running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleNewConversation = () => {
    setConversationId(null);
    setActiveContext(null);
    if (onClearContext) onClearContext();
    setMessages([
      {
        id: Date.now(),
        role: "assistant",
        content: "Started a new conversation session. How can I help you today?",
      },
    ]);
  };

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", minHeight: "560px" }}>
      {/* Chatbot Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #334155", paddingBottom: "1rem", marginBottom: "1rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem" }}>🤖</span>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "white", margin: 0 }}>
              DermaAI Health Assistant
            </h2>
          </div>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
            Powered by NVIDIA Nemotron 70B LLM & Medical Safety Guardrails
          </p>
        </div>
        <button className="nav-tab" style={{ fontSize: "0.75rem", padding: "0.4rem 0.8rem" }} onClick={handleNewConversation}>
          ➕ New Chat
        </button>
      </div>

      {/* Loaded Prediction Context Badge */}
      {activeContext && (
        <div style={{ backgroundColor: "rgba(99, 102, 241, 0.15)", border: "1px solid var(--accent-primary)", borderRadius: "8px", padding: "0.5rem 0.75rem", marginBottom: "1rem", fontSize: "0.8rem", color: "#c7d2fe", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>
            📋 <strong>Active Screening Context:</strong> {activeContext.prediction_display_name || activeContext.prediction} ({(activeContext.confidence * 100).toFixed(1)}%)
          </span>
          <button style={{ background: "none", border: "none", color: "#94a3b8", cursor: "pointer", fontSize: "0.8rem" }} onClick={() => setActiveContext(null)}>
            ✕ Clear
          </button>
        </div>
      )}

      {/* Quick Action Chips */}
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        <button className="badge" style={{ backgroundColor: "#1e293b", border: "1px solid #334155", color: "#e2e8f0", cursor: "pointer" }} onClick={() => handleSend("What does my skin lesion screening prediction mean?")}>
          ❓ What does my result mean?
        </button>
        <button className="badge" style={{ backgroundColor: "#1e293b", border: "1px solid #334155", color: "#e2e8f0", cursor: "pointer" }} onClick={() => handleSend("Is an AI prediction a confirmed medical diagnosis?")}>
          🩺 Is this a diagnosis?
        </button>
        <button className="badge" style={{ backgroundColor: "#1e293b", border: "1px solid #334155", color: "#e2e8f0", cursor: "pointer" }} onClick={() => handleSend("When should I seek professional evaluation from a dermatologist?")}>
          👨‍⚕️ When to see a doctor?
        </button>
      </div>

      {/* Messages Scroll Area */}
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "1rem", paddingRight: "0.5rem", maxHeight: "380px" }}>
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "82%",
              padding: "0.85rem 1.1rem",
              borderRadius: msg.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
              backgroundColor: msg.role === "user" ? "var(--accent-primary)" : "#1e293b",
              color: "white",
              fontSize: "0.9rem",
              lineHeight: 1.5,
              whiteSpace: "pre-line",
              boxShadow: "0 2px 8px rgba(0,0,0,0.2)"
            }}
          >
            {msg.content}
            {msg.model && (
              <div style={{ fontSize: "0.65rem", color: "#94a3b8", marginTop: "0.4rem", fontFamily: "var(--font-mono)" }}>
                Model: {msg.model}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: "flex-start", backgroundColor: "#1e293b", padding: "0.75rem 1rem", borderRadius: "12px", color: "var(--text-muted)", fontSize: "0.85rem" }}>
            🤖 Nemotron is thinking...
          </div>
        )}
        {error && (
          <div className="warning-box" style={{ marginTop: "0.5rem" }}>
            ⚠️ {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Field & Send Button */}
      <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem", paddingTop: "0.75rem", borderTop: "1px solid #334155" }}>
        <input
          type="text"
          placeholder="Ask a question about skin health or your screening result..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          style={{
            flex: 1,
            backgroundColor: "#0f172a",
            border: "1px solid #334155",
            borderRadius: "8px",
            padding: "0.75rem 1rem",
            color: "white",
            fontSize: "0.9rem",
            outline: "none"
          }}
        />
        <button className="nav-tab active" style={{ padding: "0.75rem 1.25rem", borderRadius: "8px" }} onClick={() => handleSend()} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}
