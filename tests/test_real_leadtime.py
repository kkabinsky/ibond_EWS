# -*- coding: utf-8 -*-
"""Unit tests for the shared actionable and persistent lead metrics."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import lead_metrics


def panel(dates, alarms):
    return pd.DataFrame({
        "month": pd.to_datetime(dates),
        "alarm": alarms,
    })


class LeadMetricsTests(unittest.TestCase):
    def compute(self, rows, event="2024-06-15"):
        return lead_metrics.compute_lead_metrics(
            rows,
            event_date=pd.Timestamp(event) if event is not None else pd.NaT,
            date_col="month",
            alarm_mask=rows["alarm"],
        )

    def test_alarm_inside_actionable_window_uses_earliest_qualifying_alarm(self):
        rows = panel(
            ["2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"],
            [True, True, True, True],
        )
        metrics = self.compute(rows)

        self.assertTrue(metrics["actionable_alarm_found"])
        self.assertEqual(metrics["first_alarm"], "2024-03-15")
        self.assertEqual(metrics["lead_days"], 92.0)
        self.assertLessEqual(
            metrics["lead_days"], 92,
            "Actionable 1-3M lead must not become a multi-year duration",
        )

    def test_old_alarm_only_is_persistent_but_not_actionable(self):
        rows = panel(
            ["2022-01-15", "2022-02-15", "2022-03-15"],
            [True, True, True],
        )
        metrics = self.compute(rows)
        status, kind = lead_metrics.status_and_kind(metrics, has_event=True)

        self.assertFalse(metrics["actionable_alarm_found"])
        self.assertTrue(np.isnan(metrics["lead_days"]))
        self.assertEqual(metrics["persistent_alarm_start"], "2022-01-15")
        self.assertGreater(metrics["persistent_alarm_days"], 365)
        self.assertEqual((status, kind), ("missed", "earlier-only"))

    def test_final_continuous_alarm_episode_is_reported_separately(self):
        rows = panel(
            [
                "2024-01-15", "2024-02-15", "2024-03-15",
                "2024-04-15", "2024-05-15", "2024-06-15",
            ],
            [True, False, True, True, True, True],
        )
        metrics = self.compute(rows, event="2024-07-15")

        self.assertEqual(metrics["first_alarm"], "2024-04-15")
        self.assertEqual(metrics["persistent_alarm_start"], "2024-03-15")
        self.assertEqual(metrics["persistent_alarm_end"], "2024-06-15")
        self.assertGreater(
            metrics["persistent_alarm_days"], metrics["lead_days"]
        )

    def test_missing_calendar_month_breaks_persistent_episode(self):
        rows = panel(
            ["2024-01-15", "2024-02-15", "2024-04-15", "2024-05-15"],
            [True, True, True, True],
        )
        metrics = self.compute(rows)

        self.assertEqual(metrics["persistent_alarm_start"], "2024-04-15")
        self.assertEqual(metrics["persistent_alarm_end"], "2024-05-15")

    def test_no_alarm_has_no_actionable_or_persistent_metric(self):
        rows = panel(
            ["2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"],
            [False, False, False, False],
        )
        metrics = self.compute(rows)
        status, kind = lead_metrics.status_and_kind(metrics, has_event=True)

        self.assertFalse(metrics["actionable_alarm_found"])
        self.assertIsNone(metrics["persistent_alarm_start"])
        self.assertTrue(np.isnan(metrics["lead_days"]))
        self.assertEqual((status, kind), ("missed", "missed"))

    def test_event_and_censor_status_are_distinct(self):
        rows = panel(
            ["2024-03-15", "2024-04-15", "2024-05-15"],
            [True, True, True],
        )
        event_metrics = self.compute(rows)
        censored_metrics = self.compute(rows, event=None)

        self.assertEqual(
            lead_metrics.status_and_kind(event_metrics, has_event=True),
            ("detected", "qualifying"),
        )
        self.assertEqual(
            lead_metrics.status_and_kind(censored_metrics, has_event=False),
            ("censored", "N/A"),
        )
        self.assertTrue(np.isnan(censored_metrics["lead_days"]))
        self.assertIsNone(censored_metrics["persistent_alarm_start"])

    def test_all_actionable_values_respect_the_defined_window(self):
        for event in pd.date_range("2023-01-15", periods=24, freq="MS"):
            event = event + pd.Timedelta(days=14)
            dates = [
                event - pd.DateOffset(months=4),
                event - pd.DateOffset(months=3),
                event - pd.DateOffset(months=2),
                event - pd.DateOffset(months=1),
            ]
            metrics = self.compute(panel(dates, [True] * 4), event=str(event.date()))
            self.assertTrue(metrics["actionable_alarm_found"])
            self.assertGreaterEqual(metrics["lead_days"], 28)
            self.assertLessEqual(metrics["lead_days"], 92)
            self.assertLessEqual(
                metrics["actionable_months_before_default"],
                lead_metrics.LEAD_WINDOW_MAX_MONTHS,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
