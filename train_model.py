import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =========================
# File paths
# =========================

DATA_PATH = Path("placement_data.csv")
MODEL_PATH = Path("placement_model.pkl")
METRICS_PATH = Path("metrics.json")


# =========================
# Dataset columns
# =========================

FEATURES = [
    "cgpa",
    "placement_exam_marks",
]

TARGET = "placed"


def main():

    # Check dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "placement_data.csv was not found."
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # Check required columns
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Remove missing values
    df = df.dropna(
        subset=required_columns
    )

    # Features and target
    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Create model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    # Save model
    joblib.dump(
        model,
        MODEL_PATH,
    )

    # Save metrics
    with open(
        METRICS_PATH,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Display results
    print("\n==============================")
    print("MODEL TRAINING COMPLETED")
    print("==============================")

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    print("\nModel saved:")
    print(MODEL_PATH)

    print("\nMetrics saved:")
    print(METRICS_PATH)


if __name__ == "__main__":
    main()
