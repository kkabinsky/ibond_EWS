import inspect
import unittest
from unittest.mock import patch

import matplotlib.axes
import pandas as pd

import app
import load_bond
import survival


class SurvivalVisualPipelineTests(unittest.TestCase):
    def test_survivor2_images_use_cross_version_flet_src_api(self):
        main_source = inspect.getsource(app.main)

        self.assertNotIn("src_base64", main_source)
        self.assertIn(
            "s2ews_hazard_img.src = _uri(fig_hazard(df_surv))",
            main_source,
        )
        self.assertIn(
            "s2ews_boundary_img.src = _uri(fig_boundary(df_surv, meta))",
            main_source,
        )

    def test_normalize_uses_observation_month_and_true_event_only(self):
        raw = pd.DataFrame({
            "issuer_code": ["AAA", "AAA", "AAA", "BBB"],
            "month": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-01-01"],
            "event": [0, 0, 1, 0],
            "y_fwd": [1, 1, 0, 1],
        })

        result = app.normalize_ews_panel(raw)

        self.assertEqual(result["firm_id"].tolist(), ["AAA", "AAA", "AAA", "BBB"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["month_year"]))
        self.assertEqual(result["month_index"].tolist(), [1, 2, 3, 1])
        self.assertEqual(result["event"].tolist(), [0, 0, 1, 0])

    def test_prepare_panel_censors_rows_after_first_event(self):
        raw = pd.DataFrame({
            "firm_id": ["AAA"] * 4 + ["BBB"] * 2,
            "month": [
                "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01",
                "2025-01-01", "2025-02-01",
            ],
            "event": [0, 0, 1, 0, 0, 0],
            "feature": [1.0, 1.1, 1.2, 9.9, 2.0, 2.1],
        })

        result = survival.prepare_panel(raw)

        aaa = result[result["firm_id"] == "AAA"]
        self.assertEqual(len(aaa), 3)
        self.assertEqual(aaa["event"].tolist(), [0, 0, 1])
        self.assertNotIn(9.9, aaa["feature"].tolist())

    def test_bond_panel_selects_exact_33_model_features(self):
        frame = pd.DataFrame({
            **{name: [1.0, 2.0] for name in load_bond.BOND_FEATURES},
            "unrelated_numeric": [99.0, 100.0],
        })

        covariates = survival._get_covs(frame)

        self.assertEqual(covariates, load_bond.BOND_FEATURES)
        self.assertNotIn("unrelated_numeric", covariates)

    def test_boundary_plot_contains_normal_flagged_and_distress_classes(self):
        frame = pd.DataFrame({
            "firm_id": ["NORMAL", "FLAGGED", "DISTRESS", "NORMAL2"],
            "PD_prev": [0.25, 0.50, 0.80, 0.10],
            "Momentum": [0.50, 2.00, 1.50, 0.80],
            "y_fwd": [0, 0, 1, 0],
        })
        meta = {"boundary": {"K": 0.50, "alpha": 0.50}}

        diagnostics = app.boundary_plot_diagnostics(frame, meta)
        image = app.fig_boundary(frame, meta, max_pts=100, label_top=4)

        self.assertGreater(diagnostics["normal_rows"], 0)
        self.assertGreater(diagnostics["flagged_rows"], 0)
        self.assertGreater(diagnostics["distress_rows"], 0)
        self.assertGreater(diagnostics["right_edge_boundary"], 0)
        self.assertGreater(len(image), 10_000)

    def test_trajectory_does_not_label_pd_as_hazard_when_h_is_absent(self):
        frame = pd.DataFrame({
            "firm_id": ["AAA", "AAA", "AAA"],
            "month_index": [1, 2, 3],
            "PD_3M": [0.05, 0.10, 0.20],
            "Momentum": [1.0, 2.0, 2.0],
            "event": [0, 0, 1],
        })
        labels = []
        original_plot = matplotlib.axes.Axes.plot

        def capture_plot(axis, *args, **kwargs):
            labels.append(kwargs.get("label"))
            return original_plot(axis, *args, **kwargs)

        with patch.object(matplotlib.axes.Axes, "plot", new=capture_plot):
            image = app.fig_firm_trajectory(frame, "AAA", pd_threshold=0.12)

        self.assertNotIn("Monthly hazard h(t) (%)", labels)
        self.assertIn("Forward PD3M (%)", [
            str(label).replace("\u2083", "3") for label in labels if label
        ])
        self.assertGreater(len(image), 10_000)

    def test_firm_hazard_survival_uses_discrete_survival_product(self):
        frame = pd.DataFrame({
            "firm_id": ["AAA", "AAA", "AAA"],
            "month_index": [1, 2, 3],
            "h": [0.10, 0.20, 0.50],
            "PD_3M": [0.05, 0.10, 0.25],
            "event": [0, 0, 1],
        })

        result = app.firm_hazard_survival_frame(frame, "AAA")
        image = app.fig_firm_hazard_survival(frame, "AAA")

        self.assertEqual(
            result["_survival_plot"].round(4).tolist(),
            [0.9, 0.72, 0.36],
        )
        self.assertTrue(result["_survival_plot"].is_monotonic_decreasing)
        self.assertGreater(len(image), 10_000)


if __name__ == "__main__":
    unittest.main()
