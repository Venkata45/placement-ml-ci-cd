from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    jsonify,
    request,
)


# =========================
# Flask application
# =========================

app = Flask(__name__)


# =========================
# Model configuration
# =========================

MODEL_PATH = Path(
    "placement_model.pkl"
)

FEATURES = [
    "cgpa",
    "placement_exam_marks",
]


# =========================
# Load model
# =========================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "placement_model.pkl was not found. "
            "Run train_model.py first."
        )

    return joblib.load(
        MODEL_PATH
    )


# =========================
# Health endpoint
# =========================

@app.get("/")
def health_check():

    return jsonify({
        "status": "ok",
        "service": "placement-prediction",
    })


# =========================
# Prediction endpoint
# =========================

@app.post("/predict")
def predict():

    data = request.get_json(
        silent=True
    )

    # Check JSON body
    if not data:

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    # Check required fields
    missing_fields = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:

        # Create input dataframe
        sample = pd.DataFrame([{
            feature: float(
                data[feature]
            )
            for feature in FEATURES
        }])

        # Load model
        model = load_model()

        # Prediction
        prediction_code = int(
            model.predict(sample)[0]
        )

        # Probability
        probability = float(
            model.predict_proba(sample)[0][1]
        )

        prediction = (
            "PLACED"
            if prediction_code == 1
            else "NOT PLACED"
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
            "placement_probability": round(
                probability,
                4,
            ),
        })

    except (ValueError, TypeError):

        return jsonify({
            "error": "Input values must be numeric"
        }), 400


# =========================
# Application entry point
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
    )
