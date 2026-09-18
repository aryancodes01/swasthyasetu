"""
SwasthyaSetu AI - Backend
=========================
Four things live here:

1. POST /api/analyze-symptoms
   Called by the web frontend's symptom checker box. Now a short back-and-forth:
     - First message  -> model asks 2-3 follow-up questions (unless it's an
                          obvious emergency, or the first message was already
                          detailed and the model is confident).
     - Second message  -> user's answers are merged with the original text and
                          the model returns a final condition + triage + advice
                          + home remedies.
   See ANALYZE ENDPOINT CONTRACT below for the exact request/response shape.

2. GET /api/ui-strings
   Returns the app's static UI text (buttons, labels, headings) translated
   into whatever language the frontend asks for, so switching the language
   dropdown can re-translate the WHOLE page, not just the diagnosis result.

3. GET /api/languages
   Unchanged - lets the frontend's language dropdown match what we support.

4. POST /sms/webhook
   Called by an SMS/WhatsApp gateway (e.g. Twilio) whenever a villager
   without a smartphone TEXTS a toll-free number. Runs the same follow-up
   conversation as the web flow, keyed by the sender's phone number, so a
   villager can text once, get asked a quick follow-up question by SMS,
   reply, and get back a diagnosis + home remedies + emergency advice.

Run with:  python app.py   (defaults to http://localhost:5000)

DEMO DISCLAIMER: the ML model is trained on a small synthetic dataset for
hackathon purposes. Nothing in this file is a substitute for a real
clinical decision-support system - keep the "not a doctor" disclaimers in
every response that reaches an end user.
"""

import os
import re
import time
import uuid
import logging
import threading
from pathlib import Path
from functools import wraps

import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory, LangDetectException
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator

from remedies import get_remedies
from followup import get_followup_questions, should_ask_followup, MAX_FOLLOWUP_QUESTIONS

DetectorFactory.seed = 0  # makes langdetect's output consistent every run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("swasthyasetu")

BASE_DIR = Path(__file__).resolve().parent

try:
    condition_model = joblib.load(BASE_DIR / "ml" / "condition_model.pkl")
    triage_model = joblib.load(BASE_DIR / "ml" / "triage_model.pkl")
except Exception as exc:  # pragma: no cover - fails fast & loud on boot
    raise RuntimeError(
        f"Could not load ML models from {BASE_DIR / 'ml'}. "
        f"Run train_model.py first. Original error: {exc}"
    ) from exc

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin) to call this API

# Almost every host (Render, Railway, Heroku, a VPS behind nginx, etc.)
# terminates HTTPS at a reverse proxy and forwards to Flask over plain
# HTTP. Without this, request.url/request.scheme look like "http://..."
# even though the public URL is "https://..." - which breaks Twilio
# signature validation below (the signature is computed over the exact
# public URL Twilio called). ProxyFix reads the standard X-Forwarded-*
# headers those hosts already set, so request.url is correct again.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Languages we actively support translating to/from.
# Add more ISO 639-1 codes here any time - GoogleTranslator supports 100+.
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

DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Twilio SMS webhook security. Set the TWILIO_AUTH_TOKEN environment
# variable (found in your Twilio Console dashboard) to turn this on - it
# then rejects any POST to /sms/webhook that didn't genuinely come from
# Twilio, so strangers can't spam your model or spoof SMS replies for
# free. Left unset, /sms/webhook behaves exactly as before (useful for
# local testing with curl before you've wired up Twilio at all).
# ---------------------------------------------------------------------------
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")


def validate_twilio_request(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not TWILIO_AUTH_TOKEN:
            return view_func(*args, **kwargs)  # validation not configured - allow through

        validator = RequestValidator(TWILIO_AUTH_TOKEN)
        signature = request.headers.get("X-Twilio-Signature", "")
        is_valid = validator.validate(request.url, request.form, signature)
        if not is_valid:
            logger.warning("Rejected /sms/webhook request with invalid Twilio signature")
            return ("Forbidden", 403)
        return view_func(*args, **kwargs)

    return wrapped



# ---------------------------------------------------------------------------
# In-memory conversation store, used to hold the "awaiting follow-up answers"
# state between a user's first message and their reply. Fine for a demo /
# single-process deploy; swap for Redis if you run multiple workers.
# ---------------------------------------------------------------------------
CONVERSATIONS = {}
CONVERSATIONS_LOCK = threading.Lock()
CONVERSATION_TTL_SECONDS = 15 * 60  # abandon a stalled conversation after 15 min


def _purge_stale_conversations():
    cutoff = time.time() - CONVERSATION_TTL_SECONDS
    stale_keys = [k for k, v in CONVERSATIONS.items() if v["created_at"] < cutoff]
    for k in stale_keys:
        CONVERSATIONS.pop(k, None)


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Best-effort language detection. Falls back to English on failure
    (e.g. very short text, or text that's just an emoji)."""
    try:
        code = detect(text)
        return code if code in SUPPORTED_LANGUAGES else "en"
    except LangDetectException:
        return "en"
    except Exception:
        return "en"



# Cache of (text, source, target) -> translated text. The condition names,
# advice sentences, and remedy tips are all drawn from a small fixed set of
# English strings (not user free-text), so once a given piece of content has
# been translated into a language successfully, we never need to hit the
# translation service for that exact combination again - it's reused for
# every user from then on. This is the same trick /api/ui-strings already
# uses for the page labels; applying it here too is what actually fixes
# symptom/remedy translations, since analyze-symptoms makes many more
# translation calls per request than ui-strings does, so it was hitting the
# free Google Translate endpoint's rate limits far more easily.
_TRANSLATION_CACHE = {}
_TRANSLATION_CACHE_LOCK = threading.Lock()
TRANSLATION_RETRY_ATTEMPTS = 2
TRANSLATION_RETRY_DELAY_SECONDS = 0.4


def translate_text(text: str, source: str, target: str) -> str:
    """Translate text between languages. Returns the original text
    unchanged if translation fails for any reason (e.g. no internet, or the
    translation service rate-limiting us) - we never want a translation
    hiccup to break the whole response. Successful translations are cached
    and transient failures are retried once before giving up."""
    if not text or source == target:
        return text

    cache_key = (text, source, target)
    with _TRANSLATION_CACHE_LOCK:
        cached = _TRANSLATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    last_exc = None
    for attempt in range(TRANSLATION_RETRY_ATTEMPTS):
        try:
            translated = GoogleTranslator(source=source, target=target).translate(text)
        except Exception as exc:
            last_exc = exc
            translated = None

        if translated:
            with _TRANSLATION_CACHE_LOCK:
                _TRANSLATION_CACHE[cache_key] = translated
            return translated

        if attempt < TRANSLATION_RETRY_ATTEMPTS - 1:
            time.sleep(TRANSLATION_RETRY_DELAY_SECONDS)

    logger.warning(
        "Translation failed (%s -> %s) after %d attempt(s): %s",
        source, target, TRANSLATION_RETRY_ATTEMPTS, last_exc,
    )
    return text


def translate_list(items, source: str, target: str) -> list:
    if source == target:
        return list(items)
    return [translate_text(item, source, target) for item in items]


EMERGENCY_KEYWORDS = [
    "chest pain", "unconscious", "not breathing", "seizure", "fits",
    "snake bite", "heavy bleeding", "severe bleeding", "accident",
    "cant breathe", "can't breathe", "blue lips", "suicide", "poison",
    "poisoning", "overdose",
]

TRIAGE_ADVICE = {
    "High": "This sounds urgent. Please go to the nearest hospital or call 108 for an ambulance NOW.",
    "Medium": "Please see a doctor within 24 hours. Rest, stay hydrated, and monitor your symptoms.",
    "Low": "This looks manageable at home. Rest and monitor. See a doctor if it gets worse.",
}

DISCLAIMER_EN = (
    "This is an AI demo, not a doctor. For anything serious or if you're "
    "unsure, please consult a qualified medical professional."
)


def clean_text(raw: str) -> str:
    """Lowercase and strip punctuation so 'FEVER!!' and 'fever' match the same way."""
    if not raw:
        return ""
    raw = raw.lower()
    raw = re.sub(r"[^a-z\s]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def is_emergency_override(text: str) -> bool:
    return any(kw in text for kw in EMERGENCY_KEYWORDS)


TRIAGE_ORDER = ["Low", "Medium", "High"]
# With only ~47 training rows, the triage model is sometimes a near
# coin-flip between classes. Under-triaging is the dangerous failure mode
# here, so when it's this unsure we round UP to the more cautious level.
TRIAGE_MIN_CONFIDENCE = 0.45
TRIAGE_MIN_MARGIN = 0.08


def _escalate_triage(triage: str) -> str:
    idx = TRIAGE_ORDER.index(triage) if triage in TRIAGE_ORDER else 1
    return TRIAGE_ORDER[min(idx + 1, len(TRIAGE_ORDER) - 1)]


def classify(cleaned_text_en: str):
    """
    Runs both models and returns (condition, triage, confidence).
    Falls back to safe defaults if the model throws for any reason -
    we'd rather hand back a generic "see a doctor" than a 500 error.
    """
    try:
        condition = condition_model.predict([cleaned_text_en])[0]

        try:
            triage_classes = triage_model.classes_
            triage_proba = triage_model.predict_proba([cleaned_text_en])[0]
            triage = triage_classes[triage_proba.argmax()]
        except Exception:
            triage = triage_model.predict([cleaned_text_en])[0]
            triage_proba = None

        try:
            confidence = float(max(condition_model.predict_proba([cleaned_text_en])[0]))
        except Exception:
            confidence = 1.0  # model has no predict_proba - don't block on it
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc)
        return "Unknown", "Medium", 0.0

    if triage_proba is not None:
        sorted_proba = sorted(triage_proba, reverse=True)
        top = sorted_proba[0]
        margin = top - sorted_proba[1] if len(sorted_proba) > 1 else 1.0
        if top < TRIAGE_MIN_CONFIDENCE or margin < TRIAGE_MIN_MARGIN:
            triage = _escalate_triage(triage)

    if is_emergency_override(cleaned_text_en):
        triage = "High"

    return condition, triage, confidence


def build_result(cleaned_text_en: str, detected_lang: str) -> dict:
    """Runs classification + remedies lookup and translates everything
    back into the user's language. Shared by the web and SMS flows."""
    condition, triage, _confidence = classify(cleaned_text_en)
    advice_en = TRIAGE_ADVICE.get(triage, TRIAGE_ADVICE["Medium"])
    remedy_info = get_remedies(condition, triage)

    return {
        "stage": "result",
        "condition": translate_text(condition, "en", detected_lang),
        "triage": triage,  # kept in English on purpose - used for UI colors/logic
        "advice": translate_text(advice_en, "en", detected_lang),
        "is_emergency": remedy_info["is_emergency"],
        "home_remedies": translate_list(remedy_info["remedies"], "en", detected_lang),
        "remedies_note": translate_text(remedy_info["note"], "en", detected_lang),
        "disclaimer": translate_text(DISCLAIMER_EN, "en", detected_lang),
        "language": detected_lang,
        "language_name": SUPPORTED_LANGUAGES.get(detected_lang, "English"),
    }


def build_followup_response(session_id: str, questions_en: list, detected_lang: str) -> dict:
    return {
        "stage": "followup",
        "session_id": session_id,
        "questions": [
            {"id": q["id"], "text": translate_text(q["text_en"], "en", detected_lang)}
            for q in questions_en
        ],
        "language": detected_lang,
        "language_name": SUPPORTED_LANGUAGES.get(detected_lang, "English"),
    }


def empty_message_response(detected_lang: str) -> dict:
    fallback_advice = "Please describe your symptoms, e.g. 'fever and headache'."
    return {
        "stage": "result",
        "condition": "Unknown",
        "triage": "Low",
        "advice": translate_text(fallback_advice, "en", detected_lang),
        "is_emergency": False,
        "home_remedies": [],
        "remedies_note": "",
        "disclaimer": translate_text(DISCLAIMER_EN, "en", detected_lang),
        "language": detected_lang,
        "language_name": SUPPORTED_LANGUAGES.get(detected_lang, "English"),
    }


def process_message(session_key: str, raw_text: str, user_language: str = None,
                     force_skip_followup: bool = False) -> dict:
    """
    Core conversational pipeline, shared by the web endpoint and the SMS
    webhook. `session_key` is whatever uniquely identifies this
    conversation - a generated session_id for the web, or the sender's
    phone number for SMS.

    Stage 1 (no open conversation for this key):
      detect language -> translate to English -> emergency check ->
      decide whether to ask follow-up questions or answer straight away.

    Stage 2 (there IS an open conversation for this key):
      treat `raw_text` as the answer(s) to the follow-up questions,
      merge with the original message, and give a final result.
    """
    with CONVERSATIONS_LOCK:
        _purge_stale_conversations()
        pending = CONVERSATIONS.get(session_key)

    raw_text = (raw_text or "").strip()

    # ---- Stage 2: continuing an existing conversation -------------------
    if pending is not None:
        detected_lang = pending["detected_lang"]
        answer_en = translate_text(raw_text, source=user_language or detected_lang, target="en")
        combined_text_en = f"{pending['original_text_en']} {clean_text(answer_en)}".strip()

        with CONVERSATIONS_LOCK:
            CONVERSATIONS.pop(session_key, None)

        if not combined_text_en:
            return empty_message_response(detected_lang)

        return build_result(combined_text_en, detected_lang)

    # ---- Stage 1: brand new conversation ---------------------------------
    detected_lang = user_language or detect_language(raw_text)
    english_text = translate_text(raw_text, source=detected_lang, target="en")
    cleaned = clean_text(english_text)

    if not cleaned:
        return empty_message_response(detected_lang)

    # Obvious emergencies skip the questionnaire entirely - never delay
    # urgent advice to ask clarifying questions.
    if is_emergency_override(cleaned):
        return build_result(cleaned, detected_lang)

    if not force_skip_followup:
        _condition, _triage, confidence = classify(cleaned)
        if should_ask_followup(cleaned, confidence):
            questions = get_followup_questions(cleaned, max_questions=MAX_FOLLOWUP_QUESTIONS)
            if questions:
                with CONVERSATIONS_LOCK:
                    CONVERSATIONS[session_key] = {
                        "created_at": time.time(),
                        "detected_lang": detected_lang,
                        "original_text_en": cleaned,
                    }
                return build_followup_response(session_key, questions, detected_lang)

    return build_result(cleaned, detected_lang)


# ---------------------------------------------------------------------------
# ANALYZE ENDPOINT CONTRACT
# ---------------------------------------------------------------------------
# POST /api/analyze-symptoms
# Request JSON:
#   {
#     "symptoms": "<free text>",        required on every call
#     "language": "hi",                 optional ISO code, skips auto-detect
#     "session_id": "<uuid>",           required from the 2nd call onward
#     "skip_followup": false            optional - true forces an immediate
#                                        diagnosis with no follow-up questions
#   }
#
# Response JSON - one of two shapes, distinguished by "stage":
#
#   {"stage": "followup", "session_id": "...", "questions": [{"id","text"}, ...], ...}
#   -> show these questions, let the user answer (as free text is fine -
#      "3 days, no blood" works), then POST again with the SAME session_id
#      and the answer text as "symptoms".
#
#   {"stage": "result", "condition", "triage", "advice", "is_emergency",
#    "home_remedies": [...], "remedies_note", "disclaimer", ... }
#   -> final diagnosis. Conversation is now closed; a new call with no
#      session_id (or an expired one) starts a fresh conversation.
# ---------------------------------------------------------------------------

@app.route("/api/analyze-symptoms", methods=["POST"])
def analyze_symptoms():
    data = request.get_json(silent=True) or {}
    symptom_text = data.get("symptoms", "")
    user_language = data.get("language")
    skip_followup = bool(data.get("skip_followup", False))
    session_id = data.get("session_id")

    if not str(symptom_text).strip():
        return jsonify({"error": "symptoms field is required"}), 400

    if user_language and user_language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"unsupported language code: {user_language}"}), 400

    # A fresh conversation gets a fresh id up front, so the "followup"
    # response always has a session_id to hand back to us next time.
    session_key = session_id or str(uuid.uuid4())

    try:
        result = process_message(
            session_key, symptom_text, user_language=user_language,
            force_skip_followup=skip_followup,
        )
    except Exception as exc:  # last-resort safety net - never 500 to the UI
        logger.exception("analyze_symptoms failed")
        return jsonify({
            "stage": "result",
            "condition": "Unknown",
            "triage": "Medium",
            "advice": "Something went wrong on our end. Please consult a doctor directly.",
            "is_emergency": False,
            "home_remedies": [],
            "remedies_note": "",
            "disclaimer": DISCLAIMER_EN,
            "language": user_language or "en",
            "error": str(exc),
        }), 200

    # Make sure the session_id we generated is always in the response,
    # even on a same-call "result" (so the client can reuse it if it wants
    # a running conversation, e.g. "and one more thing...").
    result.setdefault("session_id", session_key)
    return jsonify(result)


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """So the frontend's language dropdown always matches what the backend supports."""
    return jsonify(SUPPORTED_LANGUAGES)


# ---------------------------------------------------------------------------
# UI STRINGS - lets the frontend translate the WHOLE page, not just the
# diagnosis, whenever the user switches languages.
# ---------------------------------------------------------------------------
UI_STRINGS_EN = {
    "app_title": "SwasthyaSetu AI",
    "app_tagline": "Describe your symptoms in your own words - we'll help you figure out what to do next.",
    "language_label": "Language",
    "input_placeholder": "e.g. fever and body pain since yesterday",
    "submit_button": "Check my symptoms",
    "followup_heading": "Just a couple of quick questions:",
    "followup_placeholder": "Type your answer here",
    "followup_submit_button": "Submit answers",
    "result_condition_label": "Possible condition",
    "result_triage_label": "Urgency level",
    "result_advice_label": "What to do",
    "result_remedies_label": "Home care tips",
    "result_disclaimer_label": "Please note",
    "emergency_banner": "This looks urgent - please seek emergency care right away.",
    "call_ambulance": "Call 108 for an ambulance",
    "new_check_button": "Check different symptoms",
    "footer_text": "Demo project - not a substitute for professional medical advice.",
    "loading_text": "Thinking...",
    "error_text": "Something went wrong. Please try again.",
}

_UI_STRINGS_CACHE = {}
_UI_STRINGS_CACHE_LOCK = threading.Lock()


@app.route("/api/ui-strings", methods=["GET"])
def ui_strings():
    lang = request.args.get("lang", DEFAULT_LANGUAGE)
    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"unsupported language code: {lang}"}), 400

    if lang == "en":
        return jsonify(UI_STRINGS_EN)

    with _UI_STRINGS_CACHE_LOCK:
        cached = _UI_STRINGS_CACHE.get(lang)
    if cached:
        return jsonify(cached)

    translated = {key: translate_text(value, "en", lang) for key, value in UI_STRINGS_EN.items()}

    with _UI_STRINGS_CACHE_LOCK:
        _UI_STRINGS_CACHE[lang] = translated

    return jsonify(translated)


# ---------------------------------------------------------------------------
# SMS / WhatsApp webhook
# ---------------------------------------------------------------------------

@app.route("/sms/webhook", methods=["POST"])
@validate_twilio_request
def sms_webhook():
    """
    Twilio-style incoming SMS webhook. Twilio (or Gupshup/Exotel/etc,
    India-focused SMS gateways) will POST form data here whenever
    someone texts your toll-free number. Field names below match
    Twilio's format ('Body', 'From') - adjust if you use a different
    provider. Language is auto-detected from the SMS text itself, so a
    villager can text in Hindi, Bengali, Tamil, etc. with no setup.

    Follow-up questions work over SMS too: if we need to ask a clarifying
    question, we text it back and wait for their next message (keyed by
    phone number) instead of giving a diagnosis immediately.
    """
    incoming_msg = request.values.get("Body", "")
    sender = request.values.get("From", "unknown")

    try:
        result = process_message(f"sms:{sender}", incoming_msg)
    except Exception as exc:
        logger.exception("sms_webhook failed")
        result = {
            "stage": "result", "condition": "Unknown", "triage": "Medium",
            "advice": "Something went wrong. Please consult a doctor directly.",
            "is_emergency": False, "home_remedies": [], "remedies_note": "",
        }

    if result["stage"] == "followup":
        lines = ["SwasthyaSetu AI - a couple of quick questions:"]
        lines += [f"- {q['text']}" for q in result["questions"]]
        lines.append("(Reply with your answers in one message)")
        reply_text = "\n".join(lines)
    else:
        lines = [
            "SwasthyaSetu AI",
            f"Possible: {result.get('condition', 'Unknown')}",
            f"Urgency: {result.get('triage', 'Medium')}",
            result.get("advice", ""),
        ]
        remedies = result.get("home_remedies") or []
        if remedies:
            lines.append("Tips: " + "; ".join(remedies[:3]))
        lines.append("Ambulance: 108")
        reply_text = "\n".join(lines)

    # Built with Twilio's own MessagingResponse rather than a hand-rolled
    # XML string, so characters like & or < in a translated reply (Hindi
    # punctuation, etc.) get escaped correctly instead of producing
    # malformed TwiML that Twilio would silently fail to deliver.
    twiml = MessagingResponse()
    twiml.message(reply_text)
    return app.response_class(str(twiml), mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(404)
def not_found(_exc):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(500)
def server_error(_exc):
    return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
