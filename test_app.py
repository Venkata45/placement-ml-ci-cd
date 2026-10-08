import unittest

from app import app


class TestPlacementApplication(unittest.TestCase):

    def setUp(self):

        self.client = app.test_client()

    # =========================
    # Health test
    # =========================

    def test_health_endpoint(self):

        response = self.client.get("/")

        self.assertEqual(
            response.status_code,
            200,
        )

        data = response.get_json()

        self.assertEqual(
            data["status"],
            "ok",
        )

    # =========================
    # Prediction test
    # =========================

    def test_prediction_endpoint(self):

        response = self.client.post(
            "/predict",
            json={
                "cgpa": 7.46,
                "placement_exam_marks": 38,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        data = response.get_json()

        self.assertIn(
            "prediction",
            data,
        )

        self.assertIn(
            "prediction_code",
            data,
        )

        self.assertIn(
            "placement_probability",
            data,
        )

        self.assertIn(
            data["prediction_code"],
            [0, 1],
        )

    # =========================
    # Missing field test
    # =========================

    def test_missing_field_validation(self):

        response = self.client.post(
            "/predict",
            json={
                "cgpa": 7.46,
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        data = response.get_json()

        self.assertIn(
            "missing_fields",
            data,
        )

    # =========================
    # Invalid input test
    # =========================

    def test_invalid_input_validation(self):

        response = self.client.post(
            "/predict",
            json={
                "cgpa": "abc",
                "placement_exam_marks": 38,
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )


if __name__ == "__main__":

    unittest.main()
