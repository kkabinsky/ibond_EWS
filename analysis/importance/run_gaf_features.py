"""Generate Gramian Angular Field (GAF) panels for each of the 34 bond
predictors in `feature_bond.xlsx`.

For every feature we build a representative time series by taking the
cross-sectional median over firms on each date, then encode it as a polar
trajectory under the exponential angular mapping motivated in
Karami et al. (2026, ExpoGAF-AnoNet) and as the corresponding GASF and GADF
images. A four-panel JPG (time series, polar embedding, GASF, GADF) is saved
per feature for inclusion in the report appendix.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCEL_PATH = Path("feature_bond.xlsx")
TIMESERIES_SHEET = "Bond_RS_TimeSeries_34"
FEATURE_SHEET = "Feature_List_34"
OUTPUT_DIR = Path("gaf_outputs")
RESAMPLE_LEN = 96
DPI = 140


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def normalize_series(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    v = values.copy()
    median = np.nanmedian(v[finite])
    v[~finite] = median
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def resample_series(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.size == target_len:
        return values
    src = np.linspace(0.0, 1.0, num=values.size)
    dst = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(dst, src, values)


def exponential_phi(x_tilde: np.ndarray) -> np.ndarray:
    return math.pi * (np.exp(x_tilde) - 1.0) / (math.e - 1.0)


def gasf(phi: np.ndarray) -> np.ndarray:
    return np.cos(phi[:, None] + phi[None, :])


def gadf(phi: np.ndarray) -> np.ndarray:
    return np.sin(phi[:, None] - phi[None, :])


def render_feature_panel(
    feature_name: str,
    dates: np.ndarray,
    series: np.ndarray,
    out_path: Path,
) -> None:
    x_tilde = normalize_series(series)
    x_tilde_resampled = resample_series(x_tilde, RESAMPLE_LEN)
    phi = exponential_phi(x_tilde_resampled)
    gasf_img = gasf(phi)
    gadf_img = gadf(phi)

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6))
    ax_ts, ax_polar, ax_gasf, ax_gadf = axes

    ax_ts.plot(dates, x_tilde, color="#1f77b4", linewidth=0.9)
    ax_ts.set_title(f"Time series\n{feature_name} (median across firms)", fontsize=9)
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Normalized value")
    for label in ax_ts.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    radii = np.linspace(0.0, 1.0, num=RESAMPLE_LEN)
    ax_polar.remove()
    ax_polar = fig.add_subplot(1, 4, 2, projection="polar")
    ax_polar.plot(phi, radii, color="#d62728", linewidth=0.9)
    ax_polar.set_title("Exponential polar embedding", fontsize=9)
    ax_polar.set_yticklabels([])

    im_gasf = ax_gasf.imshow(gasf_img, cmap="viridis", origin="lower",
                              vmin=-1.0, vmax=1.0)
    ax_gasf.set_title("GASF (cos(phi_i + phi_j))", fontsize=9)
    ax_gasf.set_xticks([])
    ax_gasf.set_yticks([])
    plt.colorbar(im_gasf, ax=ax_gasf, fraction=0.046, pad=0.04)

    im_gadf = ax_gadf.imshow(gadf_img, cmap="coolwarm", origin="lower",
                              vmin=-1.0, vmax=1.0)
    ax_gadf.set_title("GADF (sin(phi_i - phi_j))", fontsize=9)
    ax_gadf.set_xticks([])
    ax_gadf.set_yticks([])
    plt.colorbar(im_gadf, ax=ax_gadf, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Exponential GAF encoding of '{feature_name}' for anomaly-based "
        "bond default monitoring",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_excel(EXCEL_PATH, sheet_name=FEATURE_SHEET)
    features = [f for f in feature_df["feature"].tolist() if isinstance(f, str)]

    ts = pd.read_excel(EXCEL_PATH, sheet_name=TIMESERIES_SHEET)
    ts["dt"] = pd.to_datetime(ts["dt"])

    grouped = ts.groupby("dt")[features].median(numeric_only=True).sort_index()
    dates = grouped.index.to_numpy()

    print(f"Loaded {len(features)} features over {len(grouped)} dates")
    for idx, feature in enumerate(features, start=1):
        if feature not in grouped.columns:
            print(f"[skip {idx:02d}] {feature}: not in time-series sheet")
            continue
        series = grouped[feature].to_numpy(dtype=float)
        out_name = f"gaf_{idx:02d}_{safe_filename(feature)}.jpg"
        out_path = OUTPUT_DIR / out_name
        render_feature_panel(feature, dates, series, out_path)
        print(f"[ok   {idx:02d}] {feature} -> {out_path.name}")


if __name__ == "__main__":
    main()
