"""
Train a simple ML pipeline that maps free-text symptom descriptions
(English, Hinglish, or simple Hindi transliteration) to:
  - a likely condition (multi-class classification)
  - a triage level: Low / Medium / High (multi-class classification)

Two separate classifiers share the same TF-IDF features.
This is a DEMO model trained on a small synthetic dataset
(backend/data/symptom_data.csv). Replace with a real, clinically
validated dataset before using this for actual medical decisions.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "symptom_data.csv"
MODEL_DIR = BASE_DIR / "ml"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df["text"]
    y_condition = df["condition"]
    y_triage = df["triage"]

    X_train, X_test, yc_train, yc_test, yt_train, yt_test = train_test_split(
        X, y_condition, y_triage, test_size=0.2, random_state=42
    )

    condition_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    condition_pipeline.fit(X_train, yc_train)

    triage_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    triage_pipeline.fit(X_train, yt_train)

    print("=== Condition classifier report ===")
    print(classification_report(yc_test, condition_pipeline.predict(X_test), zero_division=0))
    print("=== Triage classifier report ===")
    print(classification_report(yt_test, triage_pipeline.predict(X_test), zero_division=0))

    joblib.dump(condition_pipeline, MODEL_DIR / "condition_model.pkl")
    joblib.dump(triage_pipeline, MODEL_DIR / "triage_model.pkl")
    print(f"\nSaved models to {MODEL_DIR}")

if __name__ == "__main__":
    main()
