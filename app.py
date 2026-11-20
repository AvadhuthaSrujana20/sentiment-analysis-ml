import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sentiment Analyzer (ML)", page_icon="💬", layout="centered")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.joblib")

model = load_model()

st.title("💬 Text Sentiment Analysis")
st.caption("Powered by Classical Machine Learning (TF-IDF + Logistic Regression)")

tab1, tab2 = st.tabs(["Single Text Analysis", "Batch CSV Analysis"])

# --- TAB 1: Single Text ---
with tab1:
    user_input = st.text_area(
        "Enter a review, tweet, or feedback:",
        placeholder="e.g., The service was super fast and the quality is top-notch!",
        height=130
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            cleaned = clean_text(user_input)
            prediction = model.predict([cleaned])[0]
            probs = model.predict_proba([cleaned])[0]
            classes = model.classes_

            prob_df = pd.DataFrame({
                "Sentiment": classes,
                "Confidence (%)": [round(p * 100, 2) for p in probs]
            }).sort_values(by="Confidence (%)", ascending=False)

            if prediction == "Positive":
                st.success(f"**Predicted Sentiment:** {prediction} 😊")
            elif prediction == "Negative":
                st.error(f"**Predicted Sentiment:** {prediction} 😞")
            else:
                st.info(f"**Predicted Sentiment:** {prediction} 😐")

            st.subheader("Confidence Breakdown")
            st.bar_chart(prob_df.set_index("Sentiment"))

# --- TAB 2: Batch CSV ---
with tab2:
    st.write("Upload a CSV file containing a **`text`** column to analyze multiple rows at once.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            df["cleaned_text"] = df["text"].apply(clean_text)
            df["Predicted_Sentiment"] = model.predict(df["cleaned_text"])
            df["Confidence (%)"] = [round(max(p) * 100, 2) for p in model.predict_proba(df["cleaned_text"])]

            st.dataframe(df[["text", "Predicted_Sentiment", "Confidence (%)"]], use_container_width=True)

            csv_out = df[["text", "Predicted_Sentiment", "Confidence (%)"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_out,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )