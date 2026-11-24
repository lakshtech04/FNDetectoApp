"""Fake news classification web app."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "news.csv"
MODEL_PATH = Path(os.environ.get("FNDETECTO_MODEL_PATH", BASE_DIR / "fake_news_pipeline.joblib"))


def train_pipeline() -> Pipeline:
    """Train and cache the complete vectorizer + classifier pipeline."""
    if not DATA_PATH.is_file():
        raise RuntimeError(
            "The trained model is missing and news.csv was not found. "
            "Keep news.csv in the project root so the model can be trained on first launch."
        )

    try:
        frame = pd.read_csv(DATA_PATH, usecols=["text", "label"])
    except Exception as exc:
        raise RuntimeError("news.csv must contain 'text' and 'label' columns.") from exc

    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["label"] = frame["label"].fillna("").astype(str).str.strip().str.upper()
    frame = frame[(frame["text"] != "") & frame["label"].isin({"FAKE", "REAL"})]
    if frame.empty or frame["label"].nunique() != 2:
        raise RuntimeError("news.csv must include non-empty examples for both FAKE and REAL labels.")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
            ("classifier", PassiveAggressiveClassifier(max_iter=50, random_state=42)),
        ]
    )
    pipeline.fit(frame["text"], frame["label"])

    # The app remains usable if the host filesystem is read-only; in that case
    # the newly trained pipeline stays in memory and is rebuilt on the next start.
    temporary_path = None
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, suffix=".joblib", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        joblib.dump(pipeline, temporary_path)
        os.replace(temporary_path, MODEL_PATH)
    except OSError:
        logging.getLogger(__name__).warning("Could not cache model at %s", MODEL_PATH, exc_info=True)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return pipeline


def load_or_train_pipeline() -> Pipeline:
    if MODEL_PATH.is_file():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            logging.getLogger(__name__).exception("Cached model could not be loaded; training a fresh pipeline")
    return train_pipeline()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
model = load_or_train_pipeline()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=model is not None)


@app.post("/predict")
def predict():
    message = request.form.get("message", "").strip()
    if not message:
        return render_template("index.html", error="Please enter a news story to check.", news_text=message)

    try:
        prediction = str(model.predict([message])[0]).upper()
        if prediction not in {"FAKE", "REAL"}:
            raise ValueError("The model returned an unsupported label.")
        return render_template("index.html", prediction=prediction, news_text=message)
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="We could not analyze this text. Please try again.",
            news_text=message,
        ), 500


@app.errorhandler(413)
def request_too_large(_error):
    return render_template("index.html", error="Please use text shorter than 2 MB."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
