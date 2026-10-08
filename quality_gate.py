import json
from pathlib import Path


METRICS_PATH = Path("metrics.json")


# Minimum acceptable values
MIN_ACCURACY = 0.50
MIN_PRECISION = 0.40
MIN_RECALL = 0.30
MIN_F1 = 0.35
MIN_ROC_AUC = 0.50


def main():

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "metrics.json was not found."
        )

    with open(METRICS_PATH) as file:
        metrics = json.load(file)

    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    roc_auc = metrics["roc_auc"]

    print("==============================")
    print("PLACEMENT ML QUALITY GATE")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    failures = []

    if accuracy < MIN_ACCURACY:
        failures.append(
            f"Accuracy below {MIN_ACCURACY}"
        )

    if precision < MIN_PRECISION:
        failures.append(
            f"Precision below {MIN_PRECISION}"
        )

    if recall < MIN_RECALL:
        failures.append(
            f"Recall below {MIN_RECALL}"
        )

    if f1 < MIN_F1:
        failures.append(
            f"F1 score below {MIN_F1}"
        )

    if roc_auc < MIN_ROC_AUC:
        failures.append(
            f"ROC-AUC below {MIN_ROC_AUC}"
        )

    if failures:

        print("\nQUALITY GATE FAILED")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    print("\nQUALITY GATE PASSED")


if __name__ == "__main__":
    main()
