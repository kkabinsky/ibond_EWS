import unittest

import numpy as np
import pandas as pd

import lead_metrics
import realtime_ews


class RealtimeLeadTimeTests(unittest.TestCase):
    def test_attach_firm_names_keeps_id_and_uses_clear_fallback(self):
        alerts = pd.DataFrame({"firm_id": [3, 13, 999]})
        mapping = pd.DataFrame({
            "firm_id": [3, 13],
            "firm_name": ["Areeya Property PCL", "Advanced Info Service PCL"],
        })

        result = realtime_ews.attach_firm_names(alerts, mapping=mapping)

        self.assertEqual(result["firm_id"].tolist(), [3, 13, 999])
        self.assertEqual(
            result["firm_name"].tolist(),
            [
                "Areeya Property PCL",
                "Advanced Info Service PCL",
                "Firm ID 999",
            ],
        )

    def test_historical_samples_exclude_alarms_outside_1_3m(self):
        history = pd.DataFrame([
            {
                "firm_id": 1, "month_index": 0, "month_year": "2025-01-01",
                "event": 0, "PD_3M": 0.90, "Momentum": 1.5, "h": 0.40,
                "flag_RS": 1,
            },
            {
                "firm_id": 1, "month_index": 1, "month_year": "2026-01-01",
                "event": 0, "PD_3M": 0.55, "Momentum": 1.1, "h": 0.20,
                "flag_RS": 0,
            },
            {
                "firm_id": 1, "month_index": 2, "month_year": "2026-02-01",
                "event": 0, "PD_3M": 0.65, "Momentum": 1.2, "h": 0.25,
                "flag_RS": 0,
            },
            {
                "firm_id": 1, "month_index": 3, "month_year": "2026-03-01",
                "event": 0, "PD_3M": 0.75, "Momentum": 1.3, "h": 0.30,
                "flag_RS": 0,
            },
            {
                "firm_id": 1, "month_index": 4, "month_year": "2026-04-01",
                "event": 1, "PD_3M": 0.95, "Momentum": 1.4, "h": 0.50,
                "flag_RS": 1,
            },
        ])

        samples = realtime_ews.historical_actionable_samples(history)

        self.assertEqual(set(samples["observation_date"].dt.year), {2026})
        self.assertEqual(len(samples), 3)
        self.assertLessEqual(float(samples["lead_days"].max()), 92.0)
        self.assertGreaterEqual(float(samples["lead_days"].min()), 28.0)

    def test_risk_matched_estimates_vary_and_ok_is_not_applicable(self):
        risk = np.linspace(0.51, 0.98, 16)
        samples = pd.DataFrame({
            "firm_id": np.arange(16),
            "alert": "HIGH RISK",
            "lead_days": np.linspace(31.0, 92.0, 16),
            "PD_3M": risk,
            "Momentum": np.linspace(1.0, 2.0, 16),
            "h": np.linspace(0.20, 0.70, 16),
        })
        latest = pd.DataFrame([
            {
                "firm_id": 101, "alert": "HIGH RISK",
                "PD_3M": 0.53, "Momentum": 1.05, "h": 0.22,
            },
            {
                "firm_id": 102, "alert": "HIGH RISK",
                "PD_3M": 0.96, "Momentum": 1.95, "h": 0.68,
            },
            {
                "firm_id": 103, "alert": "OK",
                "PD_3M": 0.01, "Momentum": 1.00, "h": 0.003,
            },
        ])

        result = realtime_ews.estimate_expected_actionable_lead(
            latest, samples
        )

        active = result[result["alert"] == "HIGH RISK"]
        self.assertEqual(active["expected_lead_days"].nunique(), 2)
        self.assertTrue(active["expected_lead_days"].between(31, 92).all())
        self.assertTrue(
            result.loc[result["alert"] == "OK", "expected_lead_days"]
            .isna()
            .all()
        )
        self.assertEqual(
            set(result["lead_metric_version"]),
            {lead_metrics.LEAD_METRIC_VERSION},
        )


if __name__ == "__main__":
    unittest.main()
