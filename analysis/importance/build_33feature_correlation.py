"""Build a 33-feature sliding-window correlation image for the iBond panel.

The app renders the same idea in memory for the 33-feature dashboard. This
script is a small command-line version so the image can also be regenerated
and inspected outside the UI.
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
DEFAULT_OUT = os.path.join(HERE, "corr_matrix_33features_sw5.png")

BOND_33_FEATURES = [
    "amihud_monthly", "amihud_monthly_100", "adj_illiq_kz", "scaled_amihud",
    "ln_amihud", "percent_zero_days", "zero_days", "n_days",
    "ROA", "ROE", "DE", "CurrentRatio", "QuickRatio", "CashRatio",
    "EBITtoTA", "REtoTA", "WorkingCapitaltoTA", "TDTA", "LTDtoTA", "STDtoTA",
    "cf_Interestcoverageratio", "acc_DebtServiceCoverageRatio",
    "lnTotalAssets", "lnAge",
    "Policyrate", "GDPgrowth", "UnemploymentratemodeledILOe",
    "ESGScore", "GovernancePillarScore", "EnvironmentalPillarScore",
    "SocialPillarScore", "IndependentBoardMembers", "AverageBoardTenure",
]


def load_panel(db_path: str, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)


def sliding_window_correlation(
    panel: pd.DataFrame,
    window_months: int = 5,
    method: str = "spearman",
) -> tuple[pd.DataFrame, dict[str, object]]:
    feat_cols = [c for c in BOND_33_FEATURES if c in panel.columns]
    if not feat_cols:
        raise ValueError("No 33-feature columns found in the input table.")
    if "month" not in panel.columns:
        raise ValueError("Input table must contain a 'month' column.")

    p = panel.copy()
    p["_month_period"] = pd.PeriodIndex(
        pd.to_datetime(p["month"].astype(str) + "-01", errors="coerce"),
        freq="M",
    )
    p = p[p["_month_period"].notna()].copy()
    months = sorted(p["_month_period"].dropna().unique())
    if len(months) < window_months:
        raise ValueError(f"Need at least {window_months} months of data.")

    corr_mats: list[np.ndarray] = []
    window_labels: list[tuple[str, str]] = []
    for end_month in months:
        start_month = end_month - (window_months - 1)
        win = pd.period_range(start_month, end_month, freq="M")
        sub = p[p["_month_period"].isin(win)]
        if sub["_month_period"].nunique() < 3 or len(sub) < max(10, len(feat_cols)):
            continue

        x_win = sub[feat_cols].apply(pd.to_numeric, errors="coerce")
        x_win = x_win.replace([np.inf, -np.inf], np.nan)
        corr = x_win.corr(method=method, min_periods=3)
        corr = corr.reindex(index=feat_cols, columns=feat_cols)
        corr = corr.where(np.isfinite(corr), np.nan)
        # Pandas Copy-on-Write exposes .values as read-only. Make a writable
        # NumPy copy before setting the self-correlation diagonal.
        corr_values = corr.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(corr_values, 1.0)
        corr_mats.append(corr_values)
        window_labels.append((str(start_month), str(end_month)))

    if not corr_mats:
        raise ValueError("No valid sliding windows after filtering.")

    avg = np.nanmean(np.stack(corr_mats, axis=0), axis=0)
    avg = np.nan_to_num(avg, nan=0.0, posinf=1.0, neginf=-1.0)
    avg = np.array(
        np.clip((avg + avg.T) / 2.0, -1.0, 1.0),
        dtype=float,
        copy=True,
    )
    np.fill_diagonal(avg, 1.0)

    meta = {
        "rows": int(len(panel)),
        "features": int(len(feat_cols)),
        "months": int(len(months)),
        "windows": int(len(corr_mats)),
        "first_window": window_labels[0],
        "last_window": window_labels[-1],
        "min_corr": float(np.min(avg)),
        "max_corr": float(np.max(avg)),
        "method": method,
        "window_months": int(window_months),
    }
    return pd.DataFrame(avg, index=feat_cols, columns=feat_cols), meta


def save_correlation_image(corr: pd.DataFrame, meta: dict[str, object], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=120)
    fig.patch.set_facecolor("#f8fafc")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")

    first = meta["first_window"][0]
    last = meta["last_window"][1]
    ax.set_title(
        f"33-Feature Spearman Correlation - Sliding 5-Month Windows "
        f"({meta['windows']} windows, {first} to {last})",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6.2)
    ax.set_yticklabels(corr.index, fontsize=6.2)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Average Spearman correlation (-1 to +1)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--table", default="ibond_33features_panel")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    panel = load_panel(args.db, args.table)
    corr, meta = sliding_window_correlation(panel, window_months=args.window)
    save_correlation_image(corr, meta, args.out)

    print(f"saved={args.out}")
    print(
        "rows={rows} features={features} months={months} windows={windows} "
        "first={first_window} last={last_window} min={min_corr:.3f} max={max_corr:.3f}"
        .format(**meta)
    )


if __name__ == "__main__":
    main()
