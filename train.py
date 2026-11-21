import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def clean_text(text: str) -> str:
    """Lowercase text and remove special characters/extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    # 1. Load dataset
    df = pd.read_csv("telecom_complaints.csv")
    df["clean_text"] = df["complaint_text"].apply(clean_text)

    # 2. Split into Train (80%) and Test (20%)
    X_train, X_test, y_dept_train, y_dept_test, y_urg_train, y_urg_test = train_test_split(
        df["clean_text"],
        df["department"],
        df["urgency"],
        test_size=0.2,
        random_state=42,
        stratify=df["department"]
    )

    # 3. Build Pipeline for Department Classification
    dept_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    dept_pipeline.fit(X_train, y_dept_train)

    # 4. Build Pipeline for Urgency Classification
    urg_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    urg_pipeline.fit(X_train, y_urg_train)

    # 5. Evaluate Models
    print("=== DEPARTMENT CLASSIFICATION REPORT ===")
    dept_preds = dept_pipeline.predict(X_test)
    print(classification_report(y_dept_test, dept_preds))

    print("=== URGENCY CLASSIFICATION REPORT ===")
    urg_preds = urg_pipeline.predict(X_test)
    print(classification_report(y_urg_test, urg_preds))

    # 6. Save trained models for the FastAPI service
    artifacts = {
        "dept_model": dept_pipeline,
        "urg_model": urg_pipeline
    }
    joblib.dump(artifacts, "triage_models.joblib")
    print("Models saved successfully to triage_models.joblib!")

if __name__ == "__main__":
    main()