"""Generate firm-level GAF panels for important bond features.

This script creates one image for each non-constant firm-feature trajectory
among the 34 important features available in `feature_bond.xlsx`.
Output is grouped by feature:

    out_put_gaf_feature_important/<feature>/<feature>_<firm_id>.jpg
"""

from __future__ import annotations

import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXCEL_PATH = Path("feature_bond.xlsx")
IMPORTANCE_PATH = Path("feature_importance_outputs") / "feature_importance_results.xlsx"
TIMESERIES_SHEET = "Bond_RS_TimeSeries_34"
FEATURE_SHEET = "Feature_List_34"
IMPORTANCE_SHEET = "All_Features"
OUTPUT_DIR = Path("out_put_gaf_feature_important")
TOP_K = 34
RESAMPLE_LEN = 96
DPI = 140
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


def safe_filename(name: object) -> str:
    text = str(name)
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


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


def is_nonconstant(series: pd.Series) -> bool:
    clean = series.dropna()
    return clean.size >= 2 and clean.nunique(dropna=True) > 1


def render_feature_panel(
    feature_name: str,
    firm_id: object,
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
    ax_ts.set_title(f"Time series\n{feature_name} (firm {firm_id})", fontsize=9)
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

    im_gasf = ax_gasf.imshow(
        gasf_img, cmap="viridis", origin="lower", vmin=-1.0, vmax=1.0
    )
    ax_gasf.set_title("GASF (cos(phi_i + phi_j))", fontsize=9)
    ax_gasf.set_xticks([])
    ax_gasf.set_yticks([])
    plt.colorbar(im_gasf, ax=ax_gasf, fraction=0.046, pad=0.04)

    im_gadf = ax_gadf.imshow(
        gadf_img, cmap="coolwarm", origin="lower", vmin=-1.0, vmax=1.0
    )
    ax_gadf.set_title("GADF (sin(phi_i - phi_j))", fontsize=9)
    ax_gadf.set_xticks([])
    ax_gadf.set_yticks([])
    plt.colorbar(im_gadf, ax=ax_gadf, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Firm-level exponential GAF encoding of '{feature_name}' "
        f"for firm {firm_id}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def render_task(task: tuple[str, object, np.ndarray, np.ndarray, str]) -> str:
    feature, firm_id, dates, values, out_path_str = task
    render_feature_panel(
        feature_name=feature,
        firm_id=firm_id,
        dates=dates,
        series=values,
        out_path=Path(out_path_str),
    )
    return out_path_str


def load_top_features() -> list[str]:
    available = pd.read_excel(EXCEL_PATH, sheet_name=FEATURE_SHEET)["feature"]
    available_set = set(available.dropna().astype(str))
    importance = pd.read_excel(IMPORTANCE_PATH, sheet_name=IMPORTANCE_SHEET)
    ranked = importance["Feature"].dropna().astype(str).tolist()
    return [feature for feature in ranked if feature in available_set][:TOP_K]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_top_features()
    ts = pd.read_excel(EXCEL_PATH, sheet_name=TIMESERIES_SHEET)
    ts["dt"] = pd.to_datetime(ts["dt"])

    grouped = (
        ts.groupby(["firm_id", "dt"])[features]
        .median(numeric_only=True)
        .sort_index()
    )
    firms = grouped.index.get_level_values("firm_id").unique()

    for feature in features:
        (OUTPUT_DIR / safe_filename(feature)).mkdir(parents=True, exist_ok=True)

    expected_pairs = len(firms) * len(features)
    written = 0
    skipped_constant = 0
    skipped_missing = 0

    print(f"Top {len(features)} available important features: {', '.join(features)}")
    print(f"Firms: {len(firms)}")
    print(f"Firm-feature pairs: {expected_pairs}")

    tasks = []
    for firm_id, firm_df in grouped.groupby(level="firm_id", sort=True):
        firm_df = firm_df.reset_index(level="firm_id", drop=True).sort_index()
        dates = firm_df.index.to_numpy()
        firm_tag = safe_filename(firm_id)

        for feature in features:
            series_s = firm_df[feature]
            if series_s.dropna().empty:
                skipped_missing += 1
                continue
            if not is_nonconstant(series_s):
                skipped_constant += 1
                continue

            feature_tag = safe_filename(feature)
            out_path = OUTPUT_DIR / feature_tag / f"{feature_tag}_{firm_tag}.jpg"
            tasks.append(
                (
                    feature,
                    firm_id,
                    dates,
                    series_s.to_numpy(dtype=float),
                    str(out_path),
                )
            )

    print(f"Images to write: {len(tasks)}")
    print(f"Skipped constant images: {skipped_constant}")
    print(f"Skipped missing images: {skipped_missing}")
    print(f"Workers: {MAX_WORKERS}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(render_task, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
            written += 1
            if written % 500 == 0 or written == len(tasks):
                print(f"Written {written}/{len(tasks)} images...", flush=True)

    summary = pd.DataFrame(
        [
            {
                "top_k": len(features),
                "n_firms": len(firms),
                "expected_pairs": expected_pairs,
                "written_nonconstant_images": written,
                "skipped_constant": skipped_constant,
                "skipped_missing": skipped_missing,
                "output_dir": str(OUTPUT_DIR),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("Done")
    print(f"Written non-constant images: {written}")
    print(f"Skipped constant images: {skipped_constant}")
    print(f"Skipped missing images: {skipped_missing}")
    print(f"Summary: {OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
