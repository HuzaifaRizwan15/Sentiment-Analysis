from pathlib import Path
import re
import joblib
import streamlit as st
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "sentiment_model.joblib"

st.set_page_config(
    page_title="VERDICT",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #0c0f12;
    --panel: #13171c;
    --border: #252b33;
    --text: #edf0f3;
    --muted: #858e9a;
    --accent: #b8f36b;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.1rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 2px;
    color: var(--text);
}

.brand-mark {
    display: inline-block;
    width: 9px;
    height: 9px;
    background: var(--accent);
    border-radius: 2px;
    margin-right: 10px;
}

.top-meta {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1px;
    text-align: right;
}

.hero {
    padding: 54px 0 32px 0;
}

.eyebrow {
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.hero h1 {
    color: var(--text);
    font-size: clamp(36px, 5vw, 57px);
    font-weight: 600;
    line-height: 1.06;
    letter-spacing: -2.5px;
    margin: 0;
}

.hero p {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.8;
    max-width: 570px;
    margin-top: 17px;
}

.section-label {
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 23px;
}

.panel-title {
    font-size: 17px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 5px;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
    margin-bottom: 20px;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.2px;
    color: var(--muted);
    text-transform: uppercase;
}

.metric-value {
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -1px;
    margin-top: 8px;
    color: var(--text);
}

.result-card {
    background: #151b18;
    border: 1px solid #354434;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 14px;
}

.result-caption {
    color: #9cae99;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-value {
    color: var(--accent);
    font-size: 31px;
    font-weight: 600;
    letter-spacing: -1px;
    margin-top: 9px;
}

.result-note {
    color: #a3afa1;
    font-size: 12px;
    margin-top: 7px;
    line-height: 1.6;
}

div[data-testid="stTextArea"] textarea {
    background: #0f1317 !important;
    color: #edf0f3 !important;
    border: 1px solid #303741 !important;
    border-radius: 9px !important;
    font-size: 15px !important;
    line-height: 1.8 !important;
    padding: 15px !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #a8db6c !important;
    box-shadow: 0 0 0 1px #a8db6c !important;
}

.stButton > button {
    background: var(--accent);
    color: #10140d;
    border: 1px solid var(--accent);
    border-radius: 7px;
    font-weight: 600;
    min-height: 43px;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: #d0ff95;
    border-color: #d0ff95;
    color: #10140d;
}

div[data-testid="stDownloadButton"] button {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
}

div[data-testid="stDownloadButton"] button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

div[data-testid="stProgress"] > div {
    background: #252b33;
    border-radius: 3px;
}

div[data-testid="stProgress"] > div > div {
    background: var(--accent);
    border-radius: 3px;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 15px;
    border-radius: 9px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
}

div[data-testid="stMetricValue"] {
    color: var(--text);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}

hr {
    border-color: var(--border);
    margin: 27px 0;
}

.footer {
    color: #626c77;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    text-align: center;
    padding-top: 20px;
}

@media (max-width: 700px) {
    .hero { padding-top: 36px; }
    .hero h1 { letter-spacing: -1.5px; }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def clean_text(text):
    """Match the preprocessing used during model training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze_sentiment(text):
    model = load_model()
    processed_text = clean_text(text)

    prediction = model.predict([processed_text])[0]
    probabilities = model.predict_proba([processed_text])[0]

    scores = {
        str(label).lower(): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    return prediction.lower(), scores


if "history" not in st.session_state:
    st.session_state.history = []

if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None



header_left, header_right = st.columns([2, 1])

with header_left:
    st.markdown(
        '<div class="brand"><span class="brand-mark"></span>'
        'VERDICT <span style="color:#858e9a;">/ LAB</span></div>',
        unsafe_allow_html=True,
    )



st.markdown("""
<div class="hero">
    <div class="eyebrow">Text classification / Machine learning</div>
    <h1>Read between<br>the sentiments.</h1>
    <p>
        Put a tweet to the test. See what the model detects,
        how confident it is, and where its prediction gets interesting.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()



left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.markdown('<div class="section-label">01 / Input workspace</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Your text, under the microscope.</div>
        <div class="panel-subtitle">
            Paste a tweet or write something of your own.
            Slang, strong opinions, and ambiguous sentences are welcome.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("**Choose a test case**")

    examples = {
        "Write my own": "",
        "Strong positive": "This update is incredible. The developers absolutely nailed it!",
        "Strong negative": "What a disappointing experience. Nothing works properly.",
        "Neutral / factual": "The new update was released yesterday with three additional features.",
        "Sarcasm test": "Amazing, another update that breaks everything. Just what I needed.",
        "Mixed sentiment": "The graphics are fantastic, but the gameplay is repetitive.",
    }

    selected_example = st.selectbox(
        "Example library",
        options=list(examples.keys()),
        label_visibility="collapsed",
    )

    if st.button("Load selected example", use_container_width=True):
        st.session_state.tweet_input = examples[selected_example]
        st.rerun()

    tweet = st.text_area(
        "Tweet text",
        key="tweet_input",
        height=190,
        max_chars=2000,
        placeholder="Start typing here...\n\nWhat sentiment will the model find?",
        label_visibility="collapsed",
    )

    word_count = len(tweet.split()) if tweet.strip() else 0
    char_count = len(tweet)

    m1, m2 = st.columns(2)
    m1.caption(f"{word_count} words")
    m2.caption(f"{char_count} / 2000 characters")

    analyze_clicked = st.button(
        "Analyze text  →",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "The model classifies text. It does not understand context "
        "the way a human does."
    )

    if analyze_clicked:
        if not tweet.strip():
            st.warning("Enter some text before running the analysis.")
        else:
            try:
                with st.spinner("Running the classifier..."):
                    sentiment, scores = analyze_sentiment(tweet)

                result = {
                    "text": tweet.strip(),
                    "sentiment": sentiment,
                    "scores": scores,
                    "words": word_count,
                }

                st.session_state.last_result = result
                st.session_state.history.insert(0, {
                    "Text": tweet.strip(),
                    "Prediction": sentiment.title(),
                    "Confidence": scores[sentiment],
                })
                st.session_state.history = st.session_state.history[:20]

            except FileNotFoundError:
                st.error(
                    "Model file not found. Run `python train_model.py` "
                    "and check the artifacts folder."
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")



with right:
    st.markdown('<div class="section-label">02 / Model response</div>',
                unsafe_allow_html=True)

    result = st.session_state.last_result

    if result is None:
        st.markdown("""
        <div class="panel" style="min-height:235px;">
            <div class="metric-label">AWAITING INPUT</div>
            <div style="font-size:23px; font-weight:600; margin-top:20px;">
                Nothing to report. Yet.
            </div>
            <p style="color:#858e9a; font-size:13px; line-height:1.8;">
                Your prediction will appear here after analysis.
                The probability breakdown will show how the classifier
                weighs each of the three sentiment classes.
            </p>
            <div style="font-family:'DM Mono'; color:#59636d; font-size:11px;
                        margin-top:25px;">
                NEGATIVE &nbsp; / &nbsp; NEUTRAL &nbsp; / &nbsp; POSITIVE
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        sentiment = result["sentiment"]
        scores = result["scores"]
        confidence = scores[sentiment]

        display_names = {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        }

        descriptions = {
            "positive": "The text leans toward a favorable opinion.",
            "negative": "The text leans toward an unfavorable opinion.",
            "neutral": "The model detects a more neutral tone.",
        }

        st.markdown(f"""
        <div class="result-card">
            <div class="result-caption">Predicted sentiment</div>
            <div class="result-value">{display_names.get(sentiment, sentiment.title())}</div>
            <div class="result-note">{descriptions.get(sentiment, "")}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probability breakdown**")
        st.caption("How the model distributes its prediction across classes.")

        for label in ["positive", "neutral", "negative"]:
            probability = scores.get(label, 0.0)
            col1, col2 = st.columns([1, 0.28])

            with col1:
                st.write(label.title())

            with col2:
                st.markdown(
                    f"<div style='text-align:right; font-family:DM Mono; "
                    f"font-size:12px;'>{probability:.1%}</div>",
                    unsafe_allow_html=True,
                )

            st.progress(probability)

        st.divider()

        c1, c2 = st.columns(2)
        c1.metric("Top-class score", f"{confidence:.1%}")
        c2.metric("Text length", f"{result['words']} words")

        if confidence >= 0.75:
            st.caption("The model strongly favors its top prediction.")
        elif confidence >= 0.50:
            st.caption("The model favors this class, but alternatives remain plausible.")
        else:
            st.caption("The model's top class has relatively low confidence.")

        with st.expander("View analyzed text"):
            st.write(result["text"])



st.divider()

history_left, history_right = st.columns([1.5, 1])

with history_left:
    st.markdown('<div class="section-label">03 / Experiment history</div>',
                unsafe_allow_html=True)
    st.markdown("### Your recent tests")
    st.caption("Your last 20 analyses are stored for this session only.")

with history_right:
    st.write("")
    st.write("")
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    history_df["Confidence"] = history_df["Confidence"].map(
        lambda value: f"{value:.1%}"
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Text": st.column_config.TextColumn("Text", width="large"),
            "Prediction": st.column_config.TextColumn("Prediction", width="small"),
            "Confidence": st.column_config.TextColumn("Top-class score", width="small"),
        },
    )

    csv_df = pd.DataFrame(st.session_state.history)
    st.download_button(
        "Export experiment history (CSV)",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_experiments.csv",
        mime="text/csv",
    )
else:
    st.markdown("""
    <div class="panel">
        <div style="color:#858e9a; font-size:13px;">
            No experiments yet. Analyze a tweet to start building your history.
        </div>
    </div>
    """, unsafe_allow_html=True)



