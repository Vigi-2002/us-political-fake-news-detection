import streamlit as st
import numpy as np
import pickle
import random
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# ======================================================
# MUST BE FIRST STREAMLIT COMMAND
# ======================================================
st.set_page_config(
    page_title="US Political Fake News Detection",
    layout="centered"
)


# ======================================================
# Load model & tokenizer
# ======================================================
@st.cache_resource
def load_artifacts():
    model = load_model("fake_news_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_artifacts()


# ======================================================
# Text cleaning (same logic as training)
# ======================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


# ======================================================
# Example US political articles
# ======================================================
example_articles = [
    """Washington — U.S. Senate leaders on Tuesday continued negotiations over a bipartisan
    spending bill aimed at preventing a government shutdown. Lawmakers said discussions
    had made progress but several issues remain unresolved ahead of the deadline.""",

    """U.S. officials said on Wednesday the administration is reviewing new border security
    measures as Congress debates immigration reform proposals. The Department of Homeland
    Security stated that coordination with lawmakers is ongoing.""",

    """BREAKING: Sources claim Washington elites secretly approved a hidden election plan
    that the media refuses to report. Millions of Americans are unaware of what is about
    to happen, insiders say.""",

    """The White House confirmed Thursday that the president will meet congressional leaders
    next week to discuss budget priorities and national security concerns."""
]


# ======================================================
# UI
# ======================================================
st.title("📰 US Political Fake News Detection")

st.write(
    "This application uses a **Bi-LSTM deep learning model** trained on "
    "**US political news articles**.\n\n"
    "The model analyzes **linguistic and stylistic writing patterns**, "
    "not factual truth."
)

st.divider()

# Example button
if st.button("🎲 Load Example Article"):
    st.session_state.example_text = random.choice(example_articles)

article_text = st.text_area(
    "Paste a US political news article below:",
    height=260,
    value=st.session_state.get("example_text", ""),
    placeholder="Paste a political news article here..."
)

MAX_LEN = 300


# ======================================================
# Prediction
# ======================================================
if st.button("Analyze Article"):

    if len(article_text.strip()) < 100:
        st.warning("Please enter a longer article (minimum 100 characters).")

    else:
        cleaned = clean_text(article_text)

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        prob = float(model.predict(padded)[0][0])

        st.subheader("Prediction Result")
        st.write(f"**Fake News Probability:** `{prob:.2f}`")

        # ---------------------------------------
        # Probability interpretation
        # ---------------------------------------
        if prob < 0.30:
            st.success("🟢 Likely REAL political news")
            st.caption(
                "The writing style resembles legitimate US political reporting."
            )

        elif prob < 0.60:
            st.warning("🟡 Uncertain / Mixed signals")
            st.caption(
                "The model is not confident. "
                "The article contains mixed linguistic patterns."
            )

        else:
            st.error("🔴 Likely FAKE political news")
            st.caption(
                "The writing style resembles patterns commonly found in fake political news."
            )

        st.divider()

        st.info(
            "⚠️ **Disclaimer:**\n\n"
            "This system does NOT verify factual accuracy.\n"
            "It analyzes language patterns learned from historical US political news datasets.\n\n"
            "Predictions may not be reliable for international or non-political articles."
        )
