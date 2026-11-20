import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)            # Keep only alphabets and spaces
    text = re.sub(r"\s+", " ", text).strip()        # Remove extra whitespace
    return text

def main():
    # 1. Load dataset
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["text", "sentiment"], inplace=True)
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["sentiment"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build Classical ML Pipeline (TF-IDF + Logistic Regression)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 4. Train Model
    pipeline.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2%}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # 6. Save Pipeline
    joblib.dump(pipeline, "sentiment_pipeline.joblib")
    print("Saved trained pipeline to 'sentiment_pipeline.joblib'")

if __name__ == "__main__":
    main()