import streamlit as st
import requests
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="NeuroNews AI", layout="wide")

API_KEY = "ca3137d727584cd49683422c57c69891"

# -----------------------
# LOAD MODELS (FIXED)
# -----------------------
@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    summarizer = pipeline(
        "text2text-generation",
        model="sshleifer/distilbart-cnn-12-6"
    )

    return sentiment, summarizer

sentiment_model, summarizer = load_models()

# -----------------------
# API FUNCTION
# -----------------------
def get_news(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}&pageSize=10"
    return requests.get(url).json().get("articles", [])

# -----------------------
# NLP FUNCTIONS
# -----------------------
def analyze_sentiment(text):
    if not text:
        return "NEUTRAL", 0.0
    result = sentiment_model(text[:512])[0]
    return result["label"], result["score"]

def summarize_text(text):
    if not text:
        return "No summary available."
    result = summarizer(text[:800], max_length=60, min_length=20, do_sample=False)
    return result[0]["generated_text"]

# -----------------------
# CREDIBILITY SCORE
# -----------------------
def credibility_score(text):
    if not text:
        return 50
    suspicious = ["shocking", "unbelievable", "secret", "exposed", "click here"]
    score = 100
    for word in suspicious:
        if word in text.lower():
            score -= 12
    return max(score, 10)

# -----------------------
# CLUSTERING
# -----------------------
def cluster_articles(texts, k=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels

# -----------------------
# UI
# -----------------------
st.title("🧠 NeuroNews AI - Smart News Analyzer")
st.caption("AI-powered news intelligence with sentiment, clustering & credibility scoring")

topic = st.sidebar.text_input("Enter Topic", "AI")
run = st.sidebar.button("Analyze News")

# -----------------------
# MAIN APP
# -----------------------
if run:
    articles = get_news(topic)
    descriptions = [a.get("description") or "" for a in articles]

    # -----------------------
    # 📊 SENTIMENT DASHBOARD
    # -----------------------
    st.subheader("📊 Sentiment Dashboard")

    sentiments = [analyze_sentiment(text)[0] for text in descriptions]

    counts = {
        "POSITIVE": sentiments.count("POSITIVE"),
        "NEGATIVE": sentiments.count("NEGATIVE"),
        "NEUTRAL": sentiments.count("NEUTRAL"),
    }

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    st.pyplot(fig)

    # -----------------------
    # 🧩 CLUSTERING
    # -----------------------
    st.subheader("🧩 Topic Clusters")

    if len(descriptions) > 3:
        labels = cluster_articles(descriptions, k=3)

        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(articles[i])

        for cluster_id, items in clusters.items():
            st.markdown(f"### Cluster {cluster_id + 1}")
            for art in items:
                st.write("•", art.get("title"))
    else:
        st.warning("Not enough articles for clustering.")

    # -----------------------
    # 📰 ARTICLES
    # -----------------------
    st.subheader("📰 Articles")

    for article in articles:
        st.markdown("---")
        st.subheader(article.get("title"))

        if article.get("urlToImage"):
            st.image(article["urlToImage"])

        desc = article.get("description") or ""
        st.write(desc)

        sentiment, score = analyze_sentiment(desc)
        st.write(f"📌 Sentiment: {sentiment} ({score:.2f})")

        cred = credibility_score(desc)
        st.write(f"🛡️ Credibility Score: {cred}%")

        summary = summarize_text(article.get("content") or desc)
        st.write("📝 Summary:", summary)

    # -----------------------
    # ⚖️ SOURCE COMPARISON
    # -----------------------
    st.subheader("⚖️ Source Comparison")

    source_map = {}

    for article in articles:
        source = article.get("source", {}).get("name", "Unknown")
        sentiment, _ = analyze_sentiment(article.get("description") or "")
        source_map.setdefault(source, []).append(sentiment)

    for source, sentiments in source_map.items():
        st.markdown(f"### {source}")
        st.write("Total Articles:", len(sentiments))
        st.write("POSITIVE:", sentiments.count("POSITIVE"))
        st.write("NEGATIVE:", sentiments.count("NEGATIVE"))
        st.write("NEUTRAL:", sentiments.count("NEUTRAL"))
