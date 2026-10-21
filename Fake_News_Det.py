
from flask import Flask, render_template, request
from pathlib import Path
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_news_pipeline.joblib"

# Load trained model and TF-IDF pipeline
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()

    if not message:
        return render_template(
            "index.html",
            error="Please enter news text."
        )

    try:
        prediction = model.predict([message])[0]

        return render_template(
            "index.html",
            prediction=str(prediction),
            news_text=message
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Prediction failed. Please check the model and input."
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
