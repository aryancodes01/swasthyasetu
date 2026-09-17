# SwasthyaSetu AI — Symptom Checker + Offline SMS Mode

Trimmed down to exactly the two features you asked for:
1. AI symptom checker (web)
2. Offline mode for villagers without a smartphone — they text a
   toll-free number and get an AI triage reply back as a normal SMS.

```
swasthyasetu/
├── frontend/
│   └── index.html          ← standalone web page, calls the backend API
├── backend/
│   ├── app.py               ← Flask API + SMS webhook
│   ├── requirements.txt
│   ├── data/
│   │   └── symptom_data.csv ← training data (REPLACE with real data)
│   └── ml/
│       ├── train_model.py
│       ├── condition_model.pkl  (generated)
│       └── triage_model.pkl     (generated)
└── README.md
```

## How it fits together

```
Villager (any phone, no internet)
      │  sends SMS "fever and cough"
      ▼
SMS Gateway (Twilio / Gupshup / Exotel — see step 4)
      │  POSTs the message to your server
      ▼
backend/app.py  →  /sms/webhook
      │  runs the same ML model as the web checker
      ▼
SMS Gateway sends the reply back to the villager's inbox


Smartphone user
      │  opens frontend/index.html, types symptoms
      ▼
backend/app.py  →  /api/analyze-symptoms  (JSON)
```

Both paths share one ML model, so behavior stays consistent whether
someone uses the web page or plain SMS.

---

## Step 1 — Run the backend locally

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# train the ML model (creates condition_model.pkl and triage_model.pkl)
python ml/train_model.py

# start the API
python app.py
```

You now have `http://localhost:5000` with:
- `POST /api/analyze-symptoms` — `{"symptoms": "fever and cough"}` → condition + triage
- `POST /sms/webhook` — Twilio-format webhook, returns TwiML
- `GET /health` — health check

Test it:
```bash
curl -X POST http://localhost:5000/api/analyze-symptoms \
  -H "Content-Type: application/json" \
  -d '{"symptoms":"fever and headache"}'
```

## Step 2 — Run the frontend

`frontend/index.html` is a static file — open it directly in a browser,
or serve it:
```bash
cd frontend
python3 -m http.server 8080
```
Visit `http://localhost:8080`. If your backend runs somewhere other
than `localhost:5000`, edit the `API_BASE_URL` constant near the
bottom of `index.html`.

## Step 3 — Improve the ML model (do this before real use)

`backend/data/symptom_data.csv` right now has ~45 hand-written rows —
enough to prove the pipeline works end-to-end, not enough to be
clinically useful (you saw this in testing: it under-called a "chest
pain" case correctly only because of the hard-coded emergency-keyword
safety net in `app.py`, not because the model itself was confident).
To make it real:
- Use a public symptom-disease dataset (e.g. Kaggle's "Disease
  Symptom Prediction" dataset, or India-specific datasets like those
  published by ICMR/NIH if accessible) — aim for thousands of rows,
  not dozens.
- Add Hindi/Hinglish training examples, since real users will text in
  mixed language.
- Consider swapping the plain TF-IDF + Logistic Regression baseline
  for a stronger multilingual text classifier once you have more data
  (e.g. a fine-tuned multilingual embedding model), but don't do that
  until the data is solid — a better model on bad data won't help.
- Always keep a rule-based safety net (like `EMERGENCY_KEYWORDS` in
  `app.py`) in front of the ML output. For a health-triage tool, a
  missed emergency is far worse than a model with modest accuracy.
- Get an actual clinician to review the condition→advice mapping
  before this touches real patients.

## Step 4 — Get a toll-free SMS number (India)

You need an SMS/telephony gateway that can (a) receive inbound SMS to
a number and (b) forward it to your server as a webhook. Options used
in India:

| Provider | Notes |
|---|---|
| **Twilio** | Easiest to prototype with (code in `app.py` already matches Twilio's webhook format). International; for a genuine India toll-free number you'll eventually want a local provider for TRAI compliance. |
| **Gupshup** | India-focused, handles SMS/WhatsApp, used by many govt/health projects. |
| **Exotel** | India telephony platform, supports SMS + voice + IVR, good if you also want a voice/IVR fallback. |
| **Government 108/104 integration** | For a real state-level rollout you'd eventually coordinate with the state health department / NHM to plug into or complement the existing 108 ambulance and 104 helpline systems rather than compete with them. |

Setup pattern (same for all of the above):
1. Sign up, buy/request a number (toll-free numbers usually need
   business verification — expect this to take longest).
2. In the provider's dashboard, set the **inbound SMS webhook URL**
   to `https://your-server.com/sms/webhook`.
3. Deploy `backend/app.py` somewhere public (Step 5) so the provider
   can reach it — `localhost` won't work here.
4. Send a test SMS to the number and confirm you get a reply.

If you use a provider other than Twilio, the field names in
`sms_webhook()` (`Body`, `From`) will differ — check that provider's
webhook payload docs and adjust `request.values.get(...)` accordingly.

## Step 5 — Deploy the backend so the SMS gateway can reach it

Any host works (Railway, Render, a VPS, AWS/GCP/Azure). Rough shape:
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```
Put it behind HTTPS (most platforms give you this for free) since SMS
gateways generally require `https://` webhook URLs.

## Step 6 — Deploy the frontend

`frontend/index.html` is static — Netlify, Vercel, GitHub Pages, or
any static host works. Just make sure `API_BASE_URL` points at your
deployed backend's public HTTPS URL, not `localhost`.

## Step 7 — Ongoing

- Log every SMS/API query (anonymized) so you can keep expanding the
  training dataset with real usage patterns.
- Rate-limit the SMS webhook to avoid abuse driving up your gateway
  costs.
- Add a disclaimer in every reply (already included) that this is
  guidance, not a diagnosis.
