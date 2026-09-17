"""
remedies.py
===========
Maps each condition the ML model can predict to plain-language, non-drug
home-care advice (for Low/Medium triage) or first-aid-only guidance
(for High triage / emergencies).

IMPORTANT SAFETY RULE baked into this module:
  - We NEVER name specific medicines or dosages. That requires a doctor
    or pharmacist. Everything here is rest / fluids / hygiene / positioning
    / when-to-escalate style advice, which is safe to show to a general
    audience.
  - For High triage, "remedies" are reframed as first-aid + "do this while
    you get to a hospital" — never a substitute for emergency care.
"""

GENERIC_LOW_MEDIUM_REMEDIES = [
    "Rest as much as possible and avoid strenuous activity.",
    "Drink plenty of water or ORS to stay hydrated.",
    "Monitor your symptoms and note if they get worse.",
    "See a doctor if symptoms don't improve in 2-3 days.",
]

GENERIC_EMERGENCY_STEPS = [
    "This is not something to treat at home — get emergency medical help now.",
    "Call 108 for an ambulance or get to the nearest hospital immediately.",
    "Stay with the person, keep them calm, and keep them warm.",
    "Do not give food, water, or any medication unless a doctor tells you to.",
]

# Condition -> home-care advice, used when triage is Low or Medium.
HOME_REMEDIES = {
    "Viral Fever": [
        "Rest and drink plenty of fluids (water, coconut water, soup).",
        "Use a lukewarm sponge bath if the fever feels very high.",
        "Wear light, breathable clothing rather than covering up heavily.",
        "Eat light, easy-to-digest food like khichdi or dal.",
    ],
    "Common Cold": [
        "Drink warm fluids like tea, soup, or plain warm water.",
        "Try steam inhalation to ease a blocked nose.",
        "Gargle with warm salt water for a scratchy throat.",
        "Get extra sleep to help your body recover.",
    ],
    "Upper Respiratory Infection": [
        "Steam inhalation 2-3 times a day can loosen congestion.",
        "Warm salt-water gargles help a sore throat.",
        "Stay hydrated and rest your voice if it's hoarse.",
        "Avoid cold drinks and smoky or dusty environments.",
    ],
    "Tension Headache": [
        "Rest in a quiet, dimly lit room.",
        "Apply a cold or warm compress to your forehead/neck, whichever feels better.",
        "Make sure you're drinking enough water — dehydration is a common trigger.",
        "Gently massage your neck and shoulders to release tension.",
    ],
    "Gastroenteritis": [
        "Sip ORS or salted rice water frequently to replace lost fluids.",
        "Eat bland food — plain rice, banana, toast — once you can keep food down.",
        "Avoid dairy, oily, or spicy food until you feel better.",
        "Wash hands frequently to avoid spreading it to others.",
    ],
    "Mild Gastroenteritis": [
        "Sip ORS or coconut water to stay hydrated.",
        "Stick to a bland diet (rice, banana, toast) for a day or two.",
        "Avoid dairy, caffeine, and spicy/oily food.",
        "Rest and wash hands often.",
    ],
    "Allergic Reaction": [
        "Avoid the substance you think triggered it (food, plant, product, etc.).",
        "A cool compress can soothe itching or mild swelling.",
        "Avoid scratching, as it can break the skin and cause infection.",
        "Watch closely for swelling of the face, lips, or throat — if that happens, treat it as an emergency.",
    ],
    "Urinary Tract Infection": [
        "Drink plenty of water to help flush the urinary tract.",
        "Urinate as soon as you feel the urge; don't hold it in.",
        "A warm compress on the lower abdomen can ease discomfort.",
        "Avoid caffeine and alcohol until symptoms clear up.",
    ],
    "Dental Issue": [
        "Rinse your mouth with warm salt water a few times a day.",
        "Avoid very hot, cold, or sugary food and drinks on the affected tooth.",
        "A cold compress on the cheek can reduce swelling and pain.",
        "See a dentist soon — home care only manages pain temporarily.",
    ],
    "Ear Infection": [
        "A warm compress against the ear can ease pain.",
        "Keep the ear dry — avoid swimming or getting water in it.",
        "Avoid inserting cotton buds or anything into the ear canal.",
        "See a doctor if there's discharge, hearing loss, or high fever.",
    ],
    "Conjunctivitis": [
        "Wipe discharge gently with a clean, damp cloth, wiping outward.",
        "Avoid touching or rubbing your eyes.",
        "Wash your hands often and avoid sharing towels/pillows.",
        "A cool compress can soothe irritation.",
    ],
    "Muscle Strain": [
        "Rest the affected muscle and avoid heavy lifting.",
        "Apply ice for the first 24-48 hours, then switch to warm compress.",
        "Gently stretch once the sharp pain eases — don't push through pain.",
        "Over-the-counter pain relief gels can help; ask a pharmacist what's suitable.",
    ],
    "Normal Pregnancy Symptom": [
        "Eat small, frequent meals to help with nausea.",
        "Stay hydrated and get plenty of rest.",
        "Ginger tea or lemon water can help settle mild nausea.",
        "Keep up with your regular antenatal checkups.",
    ],
    "Possible Measles": [
        "Keep the child isolated from others — measles spreads very easily.",
        "Encourage plenty of fluids and rest.",
        "Keep the room dim if bright light bothers their eyes.",
        "See a doctor promptly — measles needs medical monitoring, especially in young children.",
    ],
    "Pediatric Concern": [
        "Keep offering small sips of fluids even if the child refuses food.",
        "Keep the child comfortable and monitor their temperature.",
        "Watch closely for lethargy, refusal to drink, or breathing changes.",
        "When in doubt with a child, it's always safer to get them checked by a doctor.",
    ],
    "Possible Asthma": [
        "Sit upright rather than lying down to make breathing easier.",
        "Move away from known triggers — dust, smoke, cold air, strong smells.",
        "Use your prescribed inhaler if you have one, exactly as your doctor advised.",
        "Track how often this happens — frequent episodes need a doctor's review.",
    ],
    "Malaria (suspected)": [
        "Rest and stay hydrated.",
        "Use a mosquito net and repellent to avoid further bites.",
        "You'll need a blood test to confirm this — please see a doctor soon.",
        "Don't wait it out at home for more than a day or two.",
    ],
    "Possible Arthritis / Dengue": [
        "Rest the affected joints and avoid strain.",
        "A cold compress can reduce joint swelling.",
        "Stay hydrated and watch closely for rash, high fever, or bleeding gums.",
        "Since this can also be early dengue, get a blood test if fever persists.",
    ],
    "Kidney Infection": [
        "Drink plenty of water unless a doctor has told you to limit fluids.",
        "Rest and avoid strenuous activity.",
        "A warm compress on the back/side may ease discomfort.",
        "This needs medical evaluation soon — kidney infections can worsen quickly.",
    ],
}

# Condition -> extra condition-specific first-aid steps for High triage,
# shown ALONGSIDE the generic emergency steps above (never instead of
# telling the person to seek emergency care).
EMERGENCY_FIRST_AID = {
    "Cardiac Emergency": [
        "Help the person sit up in a comfortable, half-reclined position.",
        "Loosen any tight clothing around the neck and chest.",
        "Do not let them exert themselves — no walking around.",
    ],
    "Severe Allergic Reaction (Anaphylaxis)": [
        "If they have a prescribed auto-injector (e.g. EpiPen), help them use it.",
        "Have them lie flat with legs raised, unless they're having trouble breathing — then let them sit up.",
        "Watch their breathing closely until help arrives.",
    ],
    "Severe Bleeding / Trauma": [
        "Apply firm, direct pressure to the wound with a clean cloth.",
        "Keep the injured area raised above heart level if possible.",
        "Don't remove the cloth if it soaks through — add more on top.",
    ],
    "Snake Bite Emergency": [
        "Keep the person still and keep the bitten limb below heart level.",
        "Remove tight jewellery/clothing near the bite before swelling starts.",
        "Do NOT cut the wound, apply ice, or try to suck out venom — these make it worse.",
    ],
    "Medical Emergency - Unconscious": [
        "Check if they're breathing; if not, begin CPR if you're trained.",
        "If breathing, place them in the recovery position (on their side).",
        "Do not give them anything to eat or drink.",
    ],
    "Possible Seizure Disorder": [
        "Clear the area around them so they don't hit anything.",
        "Do not hold them down or put anything in their mouth.",
        "Time the seizure — if it lasts more than 5 minutes, treat it as an emergency.",
    ],
    "Possible Shock": [
        "Lay the person down and raise their legs slightly (unless injured).",
        "Keep them warm with a blanket.",
        "Do not give them food or water.",
    ],
    "Possible Fracture": [
        "Don't move or straighten the injured limb.",
        "Support it gently in the position you found it, using a splint if available.",
        "Apply a cold pack wrapped in cloth to reduce swelling.",
    ],
    "Possible Hypertensive Emergency": [
        "Have the person sit down calmly and rest.",
        "Loosen tight clothing.",
        "Don't give them any blood-pressure medication that isn't already prescribed.",
    ],
    "Possible Meningitis": [
        "Keep the person in a dim, quiet room while arranging transport.",
        "Watch closely for worsening confusion, stiffness, or rash.",
        "This can progress fast — don't wait to see if it improves.",
    ],
    "Possible GI Bleed": [
        "Have them lie down and stay still.",
        "Don't give food or drink.",
        "Save a sample or photo of the vomit/stool if possible — it helps the doctor.",
    ],
    "Possible Appendicitis": [
        "Don't give food, water, or pain medication — it can complicate diagnosis.",
        "Have them lie still; movement often worsens the pain.",
        "This usually needs surgery soon — go directly to a hospital.",
    ],
    "Obstetric Emergency": [
        "Have her lie down on her left side.",
        "Keep her calm and warm while arranging transport.",
        "Note how much bleeding there is and how long it's lasted — tell the medical team.",
    ],
    "Severe Gastroenteritis": [
        "Offer small sips of ORS if they're able to keep it down.",
        "Watch for signs of severe dehydration: dizziness, very little urine, extreme weakness.",
        "Blood in the stool needs urgent medical attention — don't wait it out.",
    ],
    "Possible Asthma Attack": [
        "Help them sit upright, leaning slightly forward.",
        "Use their prescribed inhaler/reliever if available.",
        "Loosen tight clothing and keep them as calm as possible.",
    ],
    "Possible Pneumonia / COVID": [
        "Keep them upright rather than lying flat — it's easier to breathe.",
        "Keep the room well-ventilated.",
        "Isolate them from others where possible, in case it's contagious.",
    ],
    "Dengue (suspected)": [
        "Encourage fluids/ORS but avoid aspirin or ibuprofen-type painkillers — they increase bleeding risk.",
        "Watch closely for bleeding gums, black stool, or severe abdominal pain.",
        "This needs a platelet count and medical monitoring — go in today.",
    ],
}


def get_remedies(condition: str, triage: str) -> dict:
    """
    Returns a dict:
      {
        "is_emergency": bool,
        "remedies": [ ... plain-language steps ... ],
        "note": "<one-line framing sentence>"
      }
    Content is always in English here — the caller translates it, same as
    condition/advice, before sending it back to the user.
    """
    if triage == "High":
        steps = list(GENERIC_EMERGENCY_STEPS)
        extra = EMERGENCY_FIRST_AID.get(condition)
        if extra:
            steps = steps[:2] + extra + steps[2:]
        return {
            "is_emergency": True,
            "remedies": steps,
            "note": "These are first-aid steps only, not a treatment — please get emergency care right away.",
        }

    remedies = HOME_REMEDIES.get(condition, GENERIC_LOW_MEDIUM_REMEDIES)
    return {
        "is_emergency": False,
        "remedies": remedies,
        "note": "General self-care tips — not a substitute for a doctor's advice, especially if symptoms persist or worsen.",
    }
