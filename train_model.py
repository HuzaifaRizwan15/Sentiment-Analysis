from pathlib import Path
import re
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "twitter_training.csv"
VALIDATION_FILE = DATA_DIR / "twitter_validation.csv"

COLUMNS = ["tweet_id", "entity", "sentiment", "text"]


def load_tweets(file_path):
    df = pd.read_csv(
        file_path,
        header=None,
        names=COLUMNS,
        encoding="utf-8",
        on_bad_lines="skip"
    )

    # Remove accidental header rows, if present
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype("string").str.strip()

    
    df["sentiment"] = df["sentiment"].replace({
        "irrelevant": "neutral"
    })

    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]
    df = df.dropna(subset=["text"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text", "sentiment"])

    return df.reset_index(drop=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("Loading dataset...")

train_df = load_tweets(TRAIN_FILE)
val_df = load_tweets(VALIDATION_FILE)

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("\nTraining class distribution:")
print(train_df["sentiment"].value_counts())

X_train = train_df["text"].map(clean_text)
y_train = train_df["sentiment"]

X_val = val_df["text"].map(clean_text)
y_val = val_df["sentiment"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
print(f"\nValidation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, predictions, zero_division=0))

model_path = ARTIFACTS_DIR / "sentiment_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_val,
    predictions,
    labels=["negative", "neutral", "positive"],
    cmap="Blues",
    xticks_rotation=45
)
plt.title("Twitter Sentiment Analysis - Confusion Matrix")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("Confusion matrix saved successfully.")
