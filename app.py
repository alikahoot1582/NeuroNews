import streamlit as st
import requests
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Smart News Analyzer Pro", layout="wide")

API_KEY = "ca3137d727584cd49683422c57c69891"

@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis")
    summarizer = pipeline("summarization")
    return sentiment, summarizer

sentiment_model, summarizer = load_models()

# -----------------------
# FUNCTIONS
# -----------------------
def get_news(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}&pageSize=10"
    return requests.get(url).json()["articles"]

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    return result["label"], result["score"]

def summarize_text(text):
    if not text:
        return ""
    summary = summarizer(text[:1000], max_length=50, min_length=15, do_sample=False)
    return summary[0]["summary_text"]

# 🔹 Fake News Score (simple heuristic)
def credibility_score(text):
    if not text:
        return 50
    suspicious_words = ["shocking", "unbelievable", "secret", "exposed", "click here"]
    score = 100
    for word in suspicious_words:
        if word in text.lower():
            score -= 15
    return max(score, 10)

# 🔹 Clustering
def cluster_articles(texts, k=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("⚙️ Controls")
topic = st.sidebar.text_input("Topic", "AI")

# -----------------------
# MAIN
# -----------------------
st.title("🧠 Smart News Analyzer PRO")

if st.sidebar.button("Analyze News"):
    articles = get_news(topic)

    descriptions = [a["description"] or "" for a in articles]

    # 🔹 Sentiment Analysis
    sentiments = [analyze_sentiment(text)[0] for text in descriptions]

    # -----------------------
    # 📊 Sentiment Dashboard
    # -----------------------
    st.subheader("📊 Sentiment Overview")

    counts = {
        "POSITIVE": sentiments.count("POSITIVE"),
        "NEGATIVE": sentiments.count("NEGATIVE"),
        "NEUTRAL": sentiments.count("NEUTRAL"),
    }

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    st.pyplot(fig)

    # -----------------------
    # 🧩 Clustering
    # -----------------------
    st.subheader("🧩 Topic Clusters")

    labels = cluster_articles(descriptions, k=3)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(articles[i])

    for cluster_id, items in clusters.items():
        st.markdown(f"### Cluster {cluster_id + 1}")
        for art in items:
            st.write("•", art["title"])

    # -----------------------
    # 📰 Articles Display
    # -----------------------
    st.subheader("📰 Articles")

    for i, article in enumerate(articles):
        st.markdown("---")
        st.subheader(article["title"])

        if article["urlToImage"]:
            st.image(article["urlToImage"])

        st.write(article["description"])

        # Sentiment
        sentiment, score = analyze_sentiment(article["description"] or "")
        st.write(f"Sentiment: {sentiment} ({score:.2f})")

        # Credibility
        cred = credibility_score(article["description"])
        st.write(f"🛡️ Credibility Score: {cred}%")

        # Summary
        summary = summarize_text(article["content"])
        st.write("📝", summary)

    # -----------------------
    # ⚖️ Source Comparison
    # -----------------------
    st.subheader("⚖️ Source Comparison")

    source_map = {}

    for article in articles:
        source = article["source"]["name"]
        sentiment, _ = analyze_sentiment(article["description"] or "")
        source_map.setdefault(source, []).append(sentiment)

    for source, sentiments in source_map.items():
        st.write(f"### {source}")
        st.write("Articles:", len(sentiments))
        st.write("Positive:", sentiments.count("POSITIVE"))
        st.write("Negative:", sentiments.count("NEGATIVE"))
