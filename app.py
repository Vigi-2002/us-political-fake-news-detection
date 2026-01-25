import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page config (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="U.S. Political Fake News Detection",
    layout="centered"
)

# -------------------------------
# Load model & tokenizer
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("fake_news_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# -------------------------------
# App UI
# -------------------------------
st.title("📰 U.S. Political Fake News Detection")
st.write(
    "This application uses a deep learning BiLSTM model trained on U.S. political news articles "
    "to classify whether a news article is **Real** or **Fake**."
)

# -------------------------------
# Example articles
# -------------------------------
sample_articles = [

    "Donald Trump on Wednesday slammed what he described as  thugs  and  criminals  who clashed with police outside an Albuquerque campaign event, just hours after police in riot gear and mounted patrol units faced off against the violent crowd.The protesters in New Mexico were thugs who were flying the Mexican flag. The rally inside was big and beautiful, but outside, criminals!  Donald J. Trump (@realDonaldTrump) May 25, 2016Watch here:The clashes erupted overnight, after Trump and some 4,000 of his supporters left the Albuquerque Convention Center. Approximately 100 demonstrators remained in downtown.",

    "A newspaper report that British Prime Minister Theresa May is preparing to offer up to 20 billion pounds more to the European Union as part of Brexit divorce bill is speculation, her spokesman said on Thursday. The EU has told Britain to spell out what it will pay when it leaves the bloc in 2019 or it may miss a deadline next month to move the talks to a discussion of future trade ties, which businesses say is vital for them to make investment decisions.",

    "BREAKING: Sources claim Washington elites secretly approved a hidden election plan that the media refuses to report. Millions of Americans are unaware of what is about to happen, insiders say.",

    "U.S. Representative Trent Franks said on Friday that he would resign from Congress effective immediately, instead of the Jan. 31 date he previously had set following the announcement of a probe into accusations of sexual harassment against him. Last night, my wife was admitted to the hospital in Washington, D.C., due to an ongoing ailment. After discussing options with my family, we came to the conclusion that the best thing for our family now would be for me to tender my previous resignation effective today, December 8th, 2017. Franks said in an emailed statement.  Late on Thursday, Franks, who has represented a district in the Phoenix, Arizona, area since 2003, issued a statement saying that two women on his staff complained that he had discussed with them his efforts to find a surrogate mother, but he denied he had ever physically intimidated, coerced, or had, or attempted to have, any sexual contact with any member of my congressional staff."
]

if "example_text" not in st.session_state:
    st.session_state.example_text = ""

if st.button("🎲 Load Example Article"):
    st.session_state.example_text = np.random.choice(sample_articles)

# -------------------------------
# Text input
# -------------------------------
user_input = st.text_area(
    "Paste a U.S. political news article:",
    value=st.session_state.example_text,
    height=250
)

# -------------------------------
# Prediction
# -------------------------------

MAX_LEN = 300

if st.button("Analyze Article"):

    if len(user_input.strip()) < 100:
        st.warning("Please enter a longer article (minimum 100 characters).")

    else:
        cleaned = clean_text(user_input)

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
