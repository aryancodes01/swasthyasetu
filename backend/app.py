"""
SwasthyaSetu AI - Backend
=========================
Two things live here:

1. /api/analyze-symptoms
   Called by the web frontend's symptom checker box.

2. /sms/webhook
   Called by an SMS/WhatsApp gateway (e.g. Twilio) whenever a villager
   without a smartphone TEXTS a toll-free number. We read the message
   body, run it through the same ML model, and reply with a plain-text
   SMS diagnosis + triage advice. No internet or app needed on their end.

Run with:  python app.py   (defaults to http://localhost:5000)
"""

import os
import re
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
condition_model = joblib.load(BASE_DIR / "ml" / "condition_model.pkl")
triage_model = joblib.load(BASE_DIR / "ml" / "triage_model.pkl")

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin) to call this API

EMERGENCY_KEYWORDS = [
    "chest pain", "unconscious", "not breathing", "seizure", "fits",
    "snake bite", "heavy bleeding", "severe bleeding", "accident",
    "cant breathe", "can't breathe", "blue lips",
]

TRIAGE_ADVICE = {
    "High": "This sounds urgent. Please go to the nearest hospital or call 108 for an ambulance NOW.",
    "Medium": "Please see a doctor within 24 hours. Rest, stay hydrated, and monitor your symptoms.",
    "Low": "This looks manageable at home. Rest and monitor. See a doctor if it gets worse.",
}


def clean_text(raw: str) -> str:
    """Lowercase and strip punctuation so 'FEVER!!' and 'fever' match the same way."""
    raw = raw.lower()
    raw = re.sub(r"[^a-z\s]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def is_emergency_override(text: str) -> bool:
    return any(kw in text for kw in EMERGENCY_KEYWORDS)


def analyze(symptom_text: str) -> dict:
    cleaned = clean_text(symptom_text)
    if not cleaned:
        return {
            "condition": "Unknown",
            "triage": "Low",
            "advice": "Please describe your symptoms, e.g. 'fever and headache'.",
        }

    condition = condition_model.predict([cleaned])[0]
    triage = triage_model.predict([cleaned])[0]

    # Safety net: certain keywords always force High triage,
    # regardless of what the ML model predicts. Never let a
    # small demo model under-triage an obvious emergency.
    if is_emergency_override(cleaned):
        triage = "High"

    return {
        "condition": condition,
        "triage": triage,
        "advice": TRIAGE_ADVICE[triage],
    }


@app.route("/api/analyze-symptoms", methods=["POST"])
def analyze_symptoms():
    data = request.get_json(silent=True) or {}
    symptom_text = data.get("symptoms", "")
    if not symptom_text.strip():
        return jsonify({"error": "symptoms field is required"}), 400

    result = analyze(symptom_text)
    return jsonify(result)


@app.route("/sms/webhook", methods=["POST"])
def sms_webhook():
    """
    Twilio-style incoming SMS webhook. Twilio (or Gupshup/Exotel/etc,
    India-focused SMS gateways) will POST form data here whenever
    someone texts your toll-free number. Field names below match
    Twilio's format ('Body', 'From') — adjust if you use a different
    provider.
    """
    incoming_msg = request.values.get("Body", "")
    sender = request.values.get("From", "unknown")

    result = analyze(incoming_msg)

    reply_text = (
        f"SwasthyaSetu AI\n"
        f"Possible: {result['condition']}\n"
        f"Urgency: {result['triage']}\n"
        f"{result['advice']}\n"
        f"Ambulance: 108"
    )

    # TwiML response format expected by Twilio's SMS webhook
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Message>{reply_text}</Message></Response>"""

    return app.response_class(twiml, mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
