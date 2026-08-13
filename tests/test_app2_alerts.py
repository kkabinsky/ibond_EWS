import unittest

import numpy as np
import pandas as pd

import app


class _FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        assert len(features) == len(self.probabilities)
        return np.column_stack(
            [1.0 - self.probabilities, self.probabilities]
        )


class Approach2AlertTests(unittest.TestCase):
    def test_compute_alerts_assigns_all_risk_bands(self):
        frame = pd.DataFrame({
            "account_id": [1, 2, 3, 4],
            "default_3m": [0, 0, 0, 1],
            "feature": [10.0, 20.0, 30.0, 40.0],
        })
        model = _FixedProbabilityModel([0.02, 0.10, 0.20, 0.80])

        alerts = app.compute_alerts(frame, model, ["feature"])

        self.assertEqual(
            set(alerts["alert"]),
            {"LOW", "WATCH", "ELEVATED", "HIGH RISK"},
        )
        self.assertEqual(alerts.iloc[0]["account_id"], 4)

    def test_alert_distribution_returns_embeddable_png(self):
        alerts = pd.DataFrame({
            "alert": [
                "LOW",
                "LOW",
                "WATCH",
                "ELEVATED",
                "HIGH RISK",
            ]
        })

        encoded = app.fig_alert_dist(alerts)
        image_uri = app._uri(encoded)

        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 10_000)
        self.assertTrue(image_uri.startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
