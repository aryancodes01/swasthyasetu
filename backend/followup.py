"""
followup.py
===========
Generates 2-3 clarifying follow-up questions from the user's (already
translated-to-English, cleaned) symptom text, so the model gets more to
work with before giving a diagnosis — the same way a nurse doing phone
triage would ask a couple of quick questions first.

Kept deliberately simple (keyword matching) so it's fast, dependency-free,
and easy to extend — add a new dict entry to grow the question bank.
"""

# Each entry: match on any keyword appearing in the cleaned text ->
# ask this one clarifying question. First match per keyword group wins;
# duplicates (by id) are removed automatically.
KEYWORD_QUESTIONS = [
    {"keywords": ["fever"], "id": "fever_duration",
     "text_en": "How many days have you had the fever, and is it constant or does it come and go?"},
    {"keywords": ["cough"], "id": "cough_type",
     "text_en": "Is the cough dry, or are you coughing up mucus/phlegm?"},
    {"keywords": ["vomit", "vomiting", "nausea", "nauseous"], "id": "vomiting_frequency",
     "text_en": "How many times have you vomited, and are you able to keep water down?"},
    {"keywords": ["rash", "itch", "itching", "allergic", "allergy"], "id": "rash_spread",
     "text_en": "Is the rash spreading, and is there any swelling of your face, lips, or throat?"},
    {"keywords": ["breath", "breathless", "breathing", "wheeze", "wheezing"], "id": "breathing_difficulty",
     "text_en": "Are you finding it hard to breathe even while sitting still, or only during activity?"},
    {"keywords": ["chest"], "id": "chest_pain_detail",
     "text_en": "Does the chest pain spread to your arm, jaw, or back, and are you sweating or feeling dizzy?"},
    {"keywords": ["stomach", "abdomen", "abdominal", "belly"], "id": "stomach_location",
     "text_en": "Is the stomach pain in one specific spot, or spread all over?"},
    {"keywords": ["diarrhea", "diarrhoea", "loose motion", "loose motions"], "id": "diarrhea_duration",
     "text_en": "How many days has this lasted, and is there any blood in it?"},
    {"keywords": ["headache"], "id": "headache_type",
     "text_en": "Did the headache come on suddenly and severely, or has it been a dull ache building up?"},
    {"keywords": ["pregnan"], "id": "pregnancy_weeks",
     "text_en": "How many weeks pregnant are you, and is there any bleeding?"},
    {"keywords": ["child", "baby", "kid", "infant"], "id": "child_age",
     "text_en": "How old is the child, and are they eating/drinking and behaving normally?"},
    {"keywords": ["back pain", "back"], "id": "back_injury",
     "text_en": "Did this start after lifting something heavy or an injury, or did it come on its own?"},
    {"keywords": ["urin"], "id": "urination_symptoms",
     "text_en": "Is there any burning while urinating, and do you also have fever or back pain?"},
    {"keywords": ["joint"], "id": "joint_swelling",
     "text_en": "Is the joint swollen, red, or warm to touch, and is it just one joint or several?"},
    {"keywords": ["bite", "bitten"], "id": "bite_source",
     "text_en": "What bit you (snake, insect, animal), and when did it happen?"},
    {"keywords": ["bleed", "bleeding"], "id": "bleeding_amount",
     "text_en": "Is the bleeding heavy or slowing down, and can you apply pressure to it right now?"},
]

GENERIC_QUESTIONS = [
    {"id": "duration", "text_en": "How many days have you had these symptoms?"},
    {"id": "other_symptoms",
     "text_en": "Do you have any other symptoms too — like fever, vomiting, or difficulty breathing?"},
    {"id": "severity", "text_en": "On a scale of 1 to 10, how severe would you say this is?"},
]

MAX_FOLLOWUP_QUESTIONS = 3
MIN_WORDS_TO_SKIP_FOLLOWUP = 8   # a long, detailed message may not need any
CONFIDENCE_TO_SKIP_FOLLOWUP = 0.55  # model already fairly sure -> skip asking


def get_followup_questions(cleaned_text_en: str, max_questions: int = MAX_FOLLOWUP_QUESTIONS) -> list:
    """
    Returns up to `max_questions` question dicts: [{"id": ..., "text_en": ...}, ...]
    Keyword matches are prioritized; generic questions fill any remaining slots.
    """
    seen_ids = set()
    questions = []

    for entry in KEYWORD_QUESTIONS:
        if entry["id"] in seen_ids:
            continue
        if any(kw in cleaned_text_en for kw in entry["keywords"]):
            questions.append({"id": entry["id"], "text_en": entry["text_en"]})
            seen_ids.add(entry["id"])
        if len(questions) >= max_questions:
            break

    if len(questions) < max_questions:
        for entry in GENERIC_QUESTIONS:
            if entry["id"] in seen_ids:
                continue
            questions.append({"id": entry["id"], "text_en": entry["text_en"]})
            seen_ids.add(entry["id"])
            if len(questions) >= max_questions:
                break

    return questions


def should_ask_followup(cleaned_text_en: str, condition_confidence: float) -> bool:
    """
    Decide whether it's worth interrupting the user with follow-up
    questions, or whether their first message already had enough detail.
    """
    word_count = len(cleaned_text_en.split())
    if word_count >= MIN_WORDS_TO_SKIP_FOLLOWUP and condition_confidence >= CONFIDENCE_TO_SKIP_FOLLOWUP:
        return False
    return True
