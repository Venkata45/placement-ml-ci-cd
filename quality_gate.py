import json
from pathlib import Path


# =========================
# Metrics file
# =========================

METRICS_PATH = Path("metrics.json")


# =========================
# Minimum acceptable values
# =========================

MIN_ACCURACY = 0.50
MIN_PRECISION = 0.60
MIN_RECALL = 0.10
MIN_F1 = 0.20
MIN_ROC_AUC = 0.50


# =========================
# Quality Gate
# =========================

def main():

    # Check metrics file
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "metrics.json was not found."
        )

    # Load metrics
    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    # Read metrics
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    roc_auc = metrics["roc_auc"]

    # Display metrics
    print("==============================")
    print("PLACEMENT ML QUALITY GATE")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nMinimum Required Values")
    print("------------------------------")

    print(f"Accuracy >= {MIN_ACCURACY}")
    print(f"Precision >= {MIN_PRECISION}")
    print(f"Recall >= {MIN_RECALL}")
    print(f"F1 Score >= {MIN_F1}")
    print(f"ROC-AUC >= {MIN_ROC_AUC}")

    # Store failures
    failures = []

    # Accuracy check
    if accuracy < MIN_ACCURACY:
        failures.append(
            f"Accuracy below {MIN_ACCURACY}"
        )

    # Precision check
    if precision < MIN_PRECISION:
        failures.append(
            f"Precision below {MIN_PRECISION}"
        )

    # Recall check
    if recall < MIN_RECALL:
        failures.append(
            f"Recall below {MIN_RECALL}"
        )

    # F1 check
    if f1 < MIN_F1:
        failures.append(
            f"F1 score below {MIN_F1}"
        )

    # ROC-AUC check
    if roc_auc < MIN_ROC_AUC:
        failures.append(
            f"ROC-AUC below {MIN_ROC_AUC}"
        )

    # =========================
    # Final Quality Gate
    # =========================

    if failures:

        print("\n==============================")
        print("QUALITY GATE FAILED")
        print("==============================")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    else:

        print("\n==============================")
        print("QUALITY GATE PASSED")
        print("==============================")

        print(
            "\nThe placement prediction model "
            "has passed the ML quality gate."
        )


# =========================
# Run
# =========================

if __name__ == "__main__":
    main()
