import streamlit as st
from transformers import pipeline

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="NeuroNews AI",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# LOAD MODELS (STABLE + FIXED)
# -----------------------------
@st.cache_resource
def load_models():

    # Sentiment model (explicit to avoid pipeline registry issues)
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Summarization model (FIXED: explicit model required)
    summarizer_model = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn"
    )

    return sentiment_model, summarizer_model


sentiment_model, summarizer = load_models()

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🧠 NeuroNews AI Powered News Intelligence Platform")
st.write("Analyze sentiment and generate AI-powered summaries of any text.")

# -----------------------------
# INPUT AREA
# -----------------------------
text = st.text_area("📄 Paste your news/article text below:", height=250)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("🚀 Analyze"):

    if not text.strip():
        st.warning("Please enter some text to analyze.")
        st.stop()

    try:
        # -----------------------------
        # SENTIMENT ANALYSIS
        # -----------------------------
        sentiment = sentiment_model(text)

        # -----------------------------
        # SAFE SUMMARIZATION
        # (prevents token overflow crashes)
        # -----------------------------
        cleaned_text = text[:1000]

        summary = summarizer(
            cleaned_text,
            max_length=120,
            min_length=30,
            do_sample=False
        )

        # -----------------------------
        # OUTPUT SECTION
        # -----------------------------
        st.subheader("📊 Sentiment Result")
        st.json(sentiment)

        st.subheader("📝 AI Summary")
        st.success(summary[0]["summary_text"])

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Made by Muhammad Ali Kahoot")
