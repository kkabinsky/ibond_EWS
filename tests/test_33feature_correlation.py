import unittest

import numpy as np
import pandas as pd

from build_33feature_correlation import sliding_window_correlation


class CorrelationCopyOnWriteTests(unittest.TestCase):
    def test_sliding_correlation_handles_read_only_pandas_arrays(self):
        panel = pd.DataFrame({
            "month": np.repeat(
                ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"],
                2,
            ),
            "ROE": np.arange(10, dtype=float),
            "DE": np.arange(10, dtype=float)[::-1],
        })
        previous = pd.options.mode.copy_on_write
        try:
            pd.options.mode.copy_on_write = True
            corr, meta = sliding_window_correlation(panel, window_months=5)
        finally:
            pd.options.mode.copy_on_write = previous

        self.assertEqual(meta["windows"], 1)
        self.assertTrue(np.allclose(np.diag(corr), 1.0))
        self.assertTrue(np.isfinite(corr.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
