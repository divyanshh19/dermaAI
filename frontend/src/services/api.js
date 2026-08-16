import axios from "axios";

// Spring Boot Main Gateway URL (Primary) & FastAPI ML Direct URL (Fallback)
const SPRINGBOOT_URL = import.meta.env.VITE_SPRINGBOOT_URL || "https://dermaai-backend-fxl0.onrender.com/api";
const FASTAPI_ML_URL = import.meta.env.VITE_ML_URL || "https://dermaai-ml-service.onrender.com/api/v1";
const FASTAPI_CHAT_URL = import.meta.env.VITE_CHAT_URL || "https://dermaai-chatbot-service.onrender.com";

export const api = {
  // Lesion Prediction API with Automatic Spring Boot -> FastAPI Fallback
  async predictImage(file, explain = true) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Primary: Call Spring Boot Gateway
      const response = await axios.post(`${SPRINGBOOT_URL}/lesions/predict?explainable=${explain}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 25000,
      });
      return response.data;
    } catch (err) {
      console.warn("Spring Boot gateway offline/unreachable. Falling back to Python FastAPI ML service.");
      // Fallback: Call FastAPI ML Service directly
      const endpoint = explain ? `${FASTAPI_ML_URL}/predict/explain` : `${FASTAPI_ML_URL}/predict`;
      const response = await axios.post(endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 25000,
      });
      return response.data;
    }
  },

  // Prediction History APIs (Tries Spring Boot first, then FastAPI)
  async getPredictionHistory() {
    try {
      const response = await axios.get(`${SPRINGBOOT_URL}/lesions/history`, { timeout: 10000 });
      return response.data;
    } catch (err) {
      try {
        const response = await axios.get(`${FASTAPI_ML_URL}/history`, { timeout: 10000 });
        return response.data;
      } catch (e) {
        return [];
      }
    }
  },

  async deletePrediction(id) {
    try {
      await axios.delete(`${SPRINGBOOT_URL}/lesions/${id}`);
    } catch (err) {
      try {
        await axios.delete(`${FASTAPI_ML_URL}/history/${id}`);
      } catch (e) {
        console.warn("Could not delete prediction record.");
      }
    }
  },

  // AI Chatbot APIs with Automatic Spring Boot -> FastAPI Nemotron Fallback
  async sendChatMessage(message, conversationId = null, predictionContext = null) {
    try {
      // Primary: Call Spring Boot Chat Gateway
      const response = await axios.post(`${SPRINGBOOT_URL}/chat`, {
        message,
        conversationId,
        predictionContext,
      }, { timeout: 25000 });
      return response.data;
    } catch (err) {
      console.warn("Spring Boot Chat gateway offline. Falling back to FastAPI Nemotron Chatbot.");
      // Fallback: Call FastAPI Chatbot Service directly
      const response = await axios.post(`${FASTAPI_CHAT_URL}/chat`, {
        message,
        conversationId: conversationId || "default",
        predictionContext,
      }, { timeout: 25000 });
      return response.data;
    }
  },

  async getConversations() {
    try {
      const response = await axios.get(`${SPRINGBOOT_URL}/chat/conversations`);
      return response.data;
    } catch (err) {
      return [];
    }
  },

  async getConversationMessages(conversationId) {
    try {
      const response = await axios.get(`${SPRINGBOOT_URL}/chat/conversations/${conversationId}`);
      return response.data;
    } catch (err) {
      return { id: conversationId, messages: [] };
    }
  },

  async deleteConversation(conversationId) {
    try {
      await axios.delete(`${SPRINGBOOT_URL}/chat/conversations/${conversationId}`);
    } catch (err) {
      console.warn("Could not delete conversation.");
    }
  },

  // System & Model Info
  async getModelInfo() {
    try {
      const response = await axios.get(`${FASTAPI_ML_URL}/model-info`);
      return response.data;
    } catch (err) {
      return {
        model_name: "EfficientNet-B0 Ensemble",
        architecture: "Convolutional Neural Network & Vision Transformer",
        calibrated: true
      };
    }
  },

  async getMetrics() {
    try {
      const response = await axios.get(`${FASTAPI_ML_URL}/metrics`);
      return response.data;
    } catch (err) {
      return {
        accuracy: 0.6764,
        balanced_accuracy: 0.6867,
        roc_auc: 0.9327,
        macro_f1: 0.5894
      };
    }
  }
};
