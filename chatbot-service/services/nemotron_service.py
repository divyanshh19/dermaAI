"""
NVIDIA Nemotron AI Medical Assistant Microservice.

Connects to NVIDIA Build API Catalog (model: nvidia/llama-3.1-nemotron-70b-instruct)
or OpenAI-compatible API endpoint with 10 strict medical safety guardrails.
"""
import os
import time
import requests
from typing import Dict, Any, Optional

NEMOTRON_SERVICE_URL = os.getenv("NEMOTRON_SERVICE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "meta/llama-3.3-70b-instruct")
NEMOTRON_API_KEY = os.getenv("NEMOTRON_API_KEY", "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")

SYSTEM_PROMPT = """You are DermaAI Assistant, a cautious, empathetic AI medical-information guide.

STRICT MEDICAL SAFETY RULES AND GUARDRAILS:
1. NEVER claim to be a licensed doctor or healthcare professional.
2. NEVER present an AI prediction as a confirmed, definitive medical diagnosis.
3. ALWAYS explain that image classification models have inherent limitations and epistemic uncertainty.
4. ALWAYS encourage users to consult a qualified dermatologist for professional visual evaluation.
5. NEVER recommend prescription medications, surgical treatments, or home remedies as definitive clinical advice.
6. DO NOT fabricate medical information or claim certainty on low-confidence predictions.
7. IF the user describes severe, bleeding, rapidly expanding, or urgent symptoms, IMMEDIATELY advise seeking urgent professional or emergency medical care.
8. Clearly distinguish general dermatological medical education from personalized clinical advice.
9. Keep explanations clear, reassuring, and non-alarmist for non-technical users.
10. Explicitly state that AI predictions are preliminary decision-support screening scores.
"""

def generate_nemotron_response(
    user_message: str,
    conversation_id: str,
    prediction_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    context_str = ""
    if prediction_context:
        pred_name = prediction_context.get("prediction_display_name") or prediction_context.get("prediction", "Unknown")
        conf = prediction_context.get("confidence", 0.0)
        if isinstance(conf, float) and conf <= 1.0:
            conf_pct = f"{conf * 100:.1f}%"
        else:
            conf_pct = f"{conf}%"
        risk = prediction_context.get("risk_level", "Unknown")
        category = prediction_context.get("category", "Unknown")
        
        context_str = (
            f"\n[AI SKIN LESION SCREENING CONTEXT]\n"
            f"- Predicted Lesion Type: {pred_name}\n"
            f"- Category: {category}\n"
            f"- Risk Assessment: {risk}\n"
            f"- Calibrated Confidence Score: {conf_pct}\n"
        )
        if "top_predictions" in prediction_context and isinstance(prediction_context["top_predictions"], list):
            top_str = ", ".join([f"{item.get('display_name', item.get('class_code'))}: {float(item.get('probability', 0))*100:.1f}%" for item in prediction_context["top_predictions"]])
            context_str += f"- Top Differential Diagnoses: {top_str}\n"

    full_user_prompt = f"{context_str}\nUser Question: {user_message}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NEMOTRON_API_KEY}"
    }

    payload = {
        "model": NEMOTRON_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    if NEMOTRON_API_KEY and NEMOTRON_API_KEY != "demo":
        try:
            resp = requests.post(NEMOTRON_SERVICE_URL, json=payload, headers=headers, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                bot_text = data["choices"][0]["message"]["content"]
                return {
                    "conversationId": conversation_id,
                    "message": bot_text,
                    "model": NEMOTRON_MODEL,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
        except Exception as e:
            print(f"NVIDIA Nemotron API Call Notice: {e}. Using fallback medical assistant guardrail response.")

    # Fallback Medical Assistant Engine when API Key is pending or offline
    fallback_text = build_fallback_response(user_message, prediction_context)

    return {
        "conversationId": conversation_id,
        "message": fallback_text,
        "model": f"{NEMOTRON_MODEL} (Medical Assistant Engine)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

def build_fallback_response(user_message: str, prediction_context: Optional[Dict[str, Any]]) -> str:
    msg_lower = user_message.lower()
    
    if prediction_context:
        pred_name = prediction_context.get("prediction_display_name") or prediction_context.get("prediction", "Lesion")
        conf = prediction_context.get("confidence", 0.0)
        conf_pct = f"{conf * 100:.1f}%" if isinstance(conf, float) and conf <= 1.0 else f"{conf}%"
        risk = prediction_context.get("risk_level", "Screening Result")
        
        if "what does" in msg_lower or "meaning" in msg_lower or "explain" in msg_lower:
            return (
                f"Your AI screening analysis indicated **{pred_name}** with a calibrated confidence score of **{conf_pct}** ({risk}).\n\n"
                f"**What this means:**\n"
                f"• This result is an automated AI pattern-recognition screening prediction based on training data from the HAM10000 dataset.\n"
                f"• **It is NOT a confirmed clinical diagnosis.** Epiluminescence microscopy and histological biopsies are required for definitive medical diagnosis.\n"
                f"• We strongly recommend sharing this result with a board-certified dermatologist for a professional physical evaluation."
            )
        
    if "doctor" in msg_lower or "see a professional" in msg_lower or "when" in msg_lower:
        return (
            "**When to Consult a Dermatologist:**\n\n"
            "You should seek prompt professional medical evaluation if your skin lesion shows any of the **ABCDE warning signs**:\n"
            "• **A - Asymmetry:** One half of the spot does not match the other.\n"
            "• **B - Border:** Irregular, ragged, notched, or blurred edges.\n"
            "• **C - Color:** Varying shades of brown, black, pink, red, white, or blue.\n"
            "• **D - Diameter:** Larger than 6mm (size of a pencil eraser).\n"
            "• **E - Evolving:** Rapid changes in size, shape, color, or symptoms like itching/bleeding."
        )

    return (
        "Welcome to the **DermaAI Health Assistant**!\n\n"
        "I am an AI medical information guide trained to explain skin lesion screening results, confidence scores, and general skin health concepts.\n\n"
        "⚠️ **Medical Disclaimer:** I provide educational decision-support information only and cannot provide formal medical diagnoses or prescribe treatments. Always consult a licensed dermatologist for personal medical concerns."
    )
