import re
import joblib

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load saved pipeline
model = joblib.load("sentiment_pipeline.joblib")

sample_reviews = [
    "The battery life is super long and the screen looks crisp!",
    "Awful product, broke within an hour and support ignored me.",
    "The item arrived in a standard box."
]

for review in sample_reviews:
    cleaned = clean_text(review)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    confidence = max(probabilities) * 100
    print(f"Text: {review}")
    print(f" -> Sentiment: {prediction} ({confidence:.1f}% confidence)\n")