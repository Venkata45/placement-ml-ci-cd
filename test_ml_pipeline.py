import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):

        self.assertTrue(
            Path("placement_data.csv").exists()
        )

    def test_dataset_columns(self):

        df = pd.read_csv(
            "placement_data.csv"
        )

        required_columns = {
            "cgpa",
            "placement_exam_marks",
            "placed",
        }

        self.assertTrue(
            required_columns.issubset(
                df.columns
            )
        )

    def test_dataset_not_empty(self):

        df = pd.read_csv(
            "placement_data.csv"
        )

        self.assertGreater(
            len(df),
            0,
        )

    def test_target_values(self):

        df = pd.read_csv(
            "placement_data.csv"
        )

        unique_values = set(
            df["placed"].dropna().unique()
        )

        self.assertTrue(
            unique_values.issubset({0, 1})
        )

    def test_model_exists(self):

        self.assertTrue(
            Path("placement_model.pkl").exists()
        )

    def test_model_loads(self):

        model = joblib.load(
            "placement_model.pkl"
        )

        self.assertIsNotNone(
            model
        )

    def test_metrics_exists(self):

        self.assertTrue(
            Path("metrics.json").exists()
        )

    def test_metrics_load(self):

        with open(
            "metrics.json"
        ) as file:

            metrics = json.load(file)

        required_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        }

        self.assertTrue(
            required_metrics.issubset(
                metrics.keys()
            )
        )


if __name__ == "__main__":
    unittest.main()
