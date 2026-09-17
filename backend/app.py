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
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # makes langdetect's output consistent every run

BASE_DIR = Path(__file__).resolve().parent
condition_model = joblib.load(BASE_DIR / "ml" / "condition_model.pkl")
triage_model = joblib.load(BASE_DIR / "ml" / "triage_model.pkl")

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin) to call this API

# Languages we actively support translating to/from.
# Add more ISO 639-1 codes here any time — GoogleTranslator supports 100+.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Urdu",
}


def detect_language(text: str) -> str:
    """Best-effort language detection. Falls back to English on failure
    (e.g. very short text, or text that's just an emoji)."""
    try:
        code = detect(text)
        return code if code in SUPPORTED_LANGUAGES else "en"
    except LangDetectException:
        return "en"


def translate_text(text: str, source: str, target: str) -> str:
    """Translate text between languages. Returns the original text
    unchanged if translation fails for any reason (e.g. no internet) —
    we never want a translation hiccup to break the whole response."""
    if not text or source == target:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception:
        return text

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


def analyze(symptom_text: str, user_language: str = None) -> dict:
    """
    Core pipeline, now multilingual:
      1. Detect what language the person wrote in (or use user_language if given)
      2. Translate their text to English (our ML model only understands English)
      3. Run the existing ML model exactly as before
      4. Translate the condition + advice back into the person's language
    """
    original_text = symptom_text
    detected_lang = user_language or detect_language(original_text)

    english_text = translate_text(original_text, source=detected_lang, target="en")
    cleaned = clean_text(english_text)

    if not cleaned:
        fallback_advice = "Please describe your symptoms, e.g. 'fever and headache'."
        return {
            "condition": "Unknown",
            "triage": "Low",
            "advice": translate_text(fallback_advice, source="en", target=detected_lang),
            "language": detected_lang,
            "language_name": SUPPORTED_LANGUAGES.get(detected_lang, "English"),
        }

    condition = condition_model.predict([cleaned])[0]
    triage = triage_model.predict([cleaned])[0]

    # Safety net: certain keywords always force High triage,
    # regardless of what the ML model predicts. Never let a
    # small demo model under-triage an obvious emergency.
    if is_emergency_override(cleaned):
        triage = "High"

    advice_en = TRIAGE_ADVICE[triage]

    return {
        "condition": translate_text(condition, source="en", target=detected_lang),
        "triage": triage,  # keep Low/Medium/High in English — used for UI colors
        "advice": translate_text(advice_en, source="en", target=detected_lang),
        "language": detected_lang,
        "language_name": SUPPORTED_LANGUAGES.get(detected_lang, "English"),
    }


@app.route("/api/analyze-symptoms", methods=["POST"])
def analyze_symptoms():
    data = request.get_json(silent=True) or {}
    symptom_text = data.get("symptoms", "")
    # Optional: frontend can pass a language code (e.g. "hi") to skip
    # auto-detection, useful when voice input already tells us the language.
    user_language = data.get("language")
    if not symptom_text.strip():
        return jsonify({"error": "symptoms field is required"}), 400

    result = analyze(symptom_text, user_language=user_language)
    return jsonify(result)


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """So the frontend's language dropdown always matches what the backend supports."""
    return jsonify(SUPPORTED_LANGUAGES)


@app.route("/sms/webhook", methods=["POST"])
def sms_webhook():
    """
    Twilio-style incoming SMS webhook. Twilio (or Gupshup/Exotel/etc,
    India-focused SMS gateways) will POST form data here whenever
    someone texts your toll-free number. Field names below match
    Twilio's format ('Body', 'From') — adjust if you use a different
    provider. Language is auto-detected from the SMS text itself, so
    a villager can text in Hindi, Bengali, Tamil, etc. with no setup.
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
