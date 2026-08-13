# -*- coding: utf-8 -*-
"""
curve_xgb.py -- XGBoost forecasts of the yield-curve factors (Level / Slope / Curvature)
with a full set of diagnostic plots.

WHAT IT DOES
    1. Loads the DNS factors produced by yield_curve_dns.py (real iBond data).
    2. Builds a lagged, leak-free design matrix: every column is known at time t.
    3. Walk-forward (expanding window) out-of-sample forecasts with XGBoost at
       horizons 1 / 3 / 6 / 12 months, benchmarked against a random walk and AR(1).
    4. Explains the model with SHAP (which lag actually drives Slope and Curvature).
    5. Forecasts forward from the last observed month.
    6. Draws five figures and saves everything to SQLite.

WHY A RANDOM-WALK BENCHMARK MATTERS
    Yield factors are close to unit-root processes. "Tomorrow = today" is a very
    strong baseline; a model that cannot beat it has learned nothing useful. Every
    accuracy number here is therefore reported as a ratio to the random walk
    (rel_RW < 1 means the model beat it).

RUN
    python curve_xgb.py                 # walk-forward + plots + save
    python curve_xgb.py --quick         # horizons 1 and 3 only (faster)
    python curve_xgb.py --no-save       # do not write to SQLite
"""
from __future__ import annotations

import os
import sqlite3
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")

T_PRED = "curve_xgb_prediction"
T_METRIC = "curve_xgb_metrics"
T_SHAP = "curve_xgb_shap"
T_FUTURE = "curve_xgb_future"
T_SUMMARY = "curve_xgb_summary"

FACTORS = ["Level", "Slope", "Curvature"]
DEFAULT_HORIZONS = (1, 3, 6, 12)
LAGS = 6                       # months of history fed to the model
MIN_TRAIN = 48                 # months required before the first OOS forecast

FC = {"Level": "#9d174d", "Slope": "#0369a1", "Curvature": "#b45309"}


# ================================================================= features ===
def build_features(f: pd.DataFrame, lags: int = LAGS) -> pd.DataFrame:
    """Lagged design matrix. Column *_lK is the factor K months before t, so a row
    dated t contains only information available at t (no look-ahead)."""
    d = f.sort_values("date").reset_index(drop=True).copy()
    cols = {}
    for fac in FACTORS:
        for L in range(1, lags + 1):                  # start at 1: never use time t
            cols[f"{fac}_l{L}"] = d[fac].shift(L)
        cols[f"{fac}_d1"] = d[fac].diff().shift(1)    # last observed change
        cols[f"{fac}_ma3"] = d[fac].rolling(3).mean().shift(1)
    for extra in ("y_3m", "y_1y", "y_2y", "y_10y", "y_15y"):
        if extra in d.columns and d[extra].notna().any():
            cols[f"{extra}_l1"] = d[extra].shift(1)
    if {"y_1y", "y_10y"}.issubset(d.columns):
        cols["spread_l1"] = (d["y_10y"] - d["y_1y"]).shift(1)
    cols["month"] = d["date"].dt.month
    X = pd.DataFrame(cols, index=d.index)
    X.insert(0, "date", d["date"])
    for fac in FACTORS:
        X[f"target_{fac}"] = d[fac]
    return X


def _xgb(seed: int = 0):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
        min_child_weight=2, n_jobs=4, random_state=seed, verbosity=0)


def _ar1(y: np.ndarray, h: int) -> float:
    """AR(1) iterated h steps -- the classic yield-factor benchmark."""
    if len(y) < 8 or np.std(y[:-1]) < 1e-12:
        return float(y[-1])
    b = np.polyfit(y[:-1], y[1:], 1)
    p = float(y[-1])
    for _ in range(h):
        p = b[0] * p + b[1]
    return p


# ============================================================ walk-forward ====
def walk_forward(f: pd.DataFrame, horizons=DEFAULT_HORIZONS, lags: int = LAGS,
                 verbose: bool = True, log=None):
    """Expanding-window OOS forecasts. At each origin t the model is refit on data
    up to t only, then asked for t+h. Returns (metrics, predictions).

    `log` is an optional callable(str) used by the GUI to stream progress while the
    walk-forward is running (it can take a while: one model fit per origin month).
    """
    def emit(msg):
        if verbose:
            print(msg)
        if log:
            log(msg)

    X = build_features(f, lags).dropna().reset_index(drop=True)
    feat_cols = [c for c in X.columns if c != "date" and not c.startswith("target_")]
    total_cells = len(FACTORS) * len(horizons)
    emit(f"design matrix: {len(X)} usable months x {len(feat_cols)} features")
    emit(f"training {total_cells} factor-horizon cells (expanding window) ...")
    rows, preds = [], []
    cell = 0
    for fac in FACTORS:
        y_all = X[f"target_{fac}"].to_numpy()
        for h in horizons:
            cell += 1
            n = len(X) - h
            if n <= MIN_TRAIN + 5:
                continue
            emit(f"[{cell}/{total_cells}] {fac} h={h}: fitting {n - MIN_TRAIN} models ...")
            recs = []
            for t in range(MIN_TRAIN, n):
                Xtr = X.loc[:t - 1, feat_cols].to_numpy()
                ytr = y_all[h:t + h]                    # target is h months ahead
                ytr = ytr[:len(Xtr)]
                if len(Xtr) != len(ytr) or len(Xtr) < 20:
                    continue
                m = _xgb()
                m.fit(Xtr, ytr)
                pred = float(m.predict(X.loc[[t], feat_cols].to_numpy())[0])
                hist = y_all[:t + 1]
                recs.append({
                    "factor": fac, "horizon": h,
                    "origin": X.loc[t, "date"],
                    "target_date": X.loc[t + h, "date"],
                    "actual": float(y_all[t + h]),
                    "xgb": pred,
                    "rw": float(hist[-1]),              # random walk
                    "ar1": _ar1(hist, h),
                })
            if not recs:
                continue
            p = pd.DataFrame(recs)
            preds.append(p)

            def rmse(col):
                return float(np.sqrt(np.mean((p[col] - p["actual"]) ** 2)))

            def mae(col):
                return float(np.mean(np.abs(p[col] - p["actual"])))
            r_x, r_r, r_a = rmse("xgb"), rmse("rw"), rmse("ar1")
            # directional hit rate: did we get the sign of the change right?
            dir_act = np.sign(p["actual"] - p["rw"])
            dir_pred = np.sign(p["xgb"] - p["rw"])
            hit = float(np.mean((dir_act == dir_pred)[dir_act != 0])) if (dir_act != 0).any() else np.nan
            rows.append({"factor": fac, "horizon": h, "n_oos": len(p),
                         "RMSE_xgb": r_x, "RMSE_rw": r_r, "RMSE_ar1": r_a,
                         "MAE_xgb": mae("xgb"), "MAE_rw": mae("rw"),
                         "rel_RW": r_x / r_r if r_r > 0 else np.nan,
                         "rel_AR1": r_x / r_a if r_a > 0 else np.nan,
                         "beats_RW": bool(r_x < r_r), "hit_rate": hit})
            flag = "BEATS RW" if r_x < r_r else "loses to RW"
            emit(f"    -> {fac} h={h}: RMSE {r_x:.4f} vs rw {r_r:.4f} "
                 f"| rel {r_x/r_r:.3f} | hit {hit:.1%} | {flag}")
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return metrics, predictions


# =================================================================== SHAP =====
def shap_importance(f: pd.DataFrame, horizon: int = 3, lags: int = LAGS) -> pd.DataFrame:
    """Mean |SHAP| per feature, per factor, for a model fit on the full sample."""
    X = build_features(f, lags).dropna().reset_index(drop=True)
    feat_cols = [c for c in X.columns if c != "date" and not c.startswith("target_")]
    out = []
    for fac in FACTORS:
        y = X[f"target_{fac}"].to_numpy()
        Xtr = X.loc[:len(X) - horizon - 1, feat_cols]
        ytr = y[horizon:]
        if len(Xtr) != len(ytr) or len(Xtr) < 30:
            continue
        m = _xgb()
        m.fit(Xtr.to_numpy(), ytr)
        try:
            import shap
            expl = shap.TreeExplainer(m)
            sv = expl.shap_values(Xtr.to_numpy())
            imp = np.abs(sv).mean(axis=0)
            src = "shap"
        except Exception:                       # fall back to XGBoost's own gain
            imp = m.feature_importances_
            src = "gain"
        for c, v in zip(feat_cols, imp):
            out.append({"factor": fac, "horizon": horizon, "feature": c,
                        "importance": float(v), "method": src})
    d = pd.DataFrame(out)
    if not d.empty:
        d["rank"] = d.groupby("factor")["importance"].rank(ascending=False)
    return d


# ================================================================= forecast ===
def forecast_future(f: pd.DataFrame, horizons=DEFAULT_HORIZONS, lags: int = LAGS) -> pd.DataFrame:
    """Fit on everything, then predict h months past the last observation."""
    X = build_features(f, lags).dropna().reset_index(drop=True)
    feat_cols = [c for c in X.columns if c != "date" and not c.startswith("target_")]
    last_date = X["date"].iloc[-1]
    rows = []
    for fac in FACTORS:
        y = X[f"target_{fac}"].to_numpy()
        for h in horizons:
            Xtr = X.loc[:len(X) - h - 1, feat_cols].to_numpy()
            ytr = y[h:]
            if len(Xtr) != len(ytr) or len(Xtr) < 30:
                continue
            m = _xgb()
            m.fit(Xtr, ytr)
            pred = float(m.predict(X.loc[[len(X) - 1], feat_cols].to_numpy())[0])
            rows.append({"factor": fac, "horizon": h,
                         "last_date": last_date,
                         "target_date": last_date + pd.DateOffset(months=h),
                         "last_value": float(y[-1]), "pred_xgb": pred,
                         "change": pred - float(y[-1])})
    return pd.DataFrame(rows)


# ==================================================================== plots ===
def _save(fig, name, outdir):
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_actual_vs_pred(preds, outdir, horizon=3):
    """Time series of actual vs XGBoost vs random walk, one panel per factor."""
    d = preds[preds["horizon"] == horizon]
    if d.empty:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, fac in zip(axes, FACTORS):
        g = d[d["factor"] == fac].sort_values("target_date")
        if g.empty:
            continue
        ax.plot(g["target_date"], g["actual"], lw=2.2, color=FC[fac], label="actual")
        ax.plot(g["target_date"], g["xgb"], lw=1.6, ls="--", color="#111827", label="XGBoost")
        ax.plot(g["target_date"], g["rw"], lw=1.0, ls=":", color="#9ca3af", label="random walk")
        r_x = np.sqrt(np.mean((g["xgb"] - g["actual"]) ** 2))
        r_r = np.sqrt(np.mean((g["rw"] - g["actual"]) ** 2))
        ax.set_ylabel(fac)
        ax.legend(fontsize=8, loc="upper left", ncol=3)
        ax.grid(alpha=0.25)
        ax.set_title(f"{fac} — RMSE xgb {r_x:.3f} vs rw {r_r:.3f} "
                     f"({'beats' if r_x < r_r else 'loses to'} RW)", fontsize=9, loc="left")
    axes[-1].set_xlabel("target date")
    fig.suptitle(f"Out-of-sample forecasts, horizon = {horizon} months", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, f"xgb_actual_vs_pred_h{horizon}.png", outdir)


def plot_rel_rw(metrics, outdir):
    """Accuracy relative to the random walk. Below 1.0 = the model adds value."""
    if metrics.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.2))
    hs = sorted(metrics["horizon"].unique())
    w = 0.8 / max(len(FACTORS), 1)
    for i, fac in enumerate(FACTORS):
        g = metrics[metrics["factor"] == fac].set_index("horizon").reindex(hs)
        x = np.arange(len(hs)) + i * w - 0.4 + w / 2
        bars = ax.bar(x, g["rel_RW"], width=w, color=FC[fac], label=fac, alpha=0.9)
        for b, v in zip(bars, g["rel_RW"]):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", fontsize=7)
    ax.axhline(1.0, color="#dc2626", lw=1.4, ls="--")
    ax.text(len(hs) - 0.5, 1.02, "random walk", color="#dc2626", fontsize=8, ha="right")
    ax.set_xticks(np.arange(len(hs)))
    ax.set_xticklabels([f"{h}m" for h in hs])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("RMSE ratio  (XGBoost / random walk)")
    ax.set_title("Below the dashed line = XGBoost beats the random walk")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, "xgb_rel_rw.png", outdir)


def plot_shap(shap_df, outdir, top=12):
    """Which lagged inputs drive each factor."""
    if shap_df.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, fac in zip(axes, FACTORS):
        g = (shap_df[shap_df["factor"] == fac]
             .sort_values("importance", ascending=False).head(top).iloc[::-1])
        if g.empty:
            ax.axis("off"); continue
        ax.barh(g["feature"], g["importance"], color=FC[fac], alpha=0.9)
        ax.set_title(fac, fontsize=10, color=FC[fac], fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.25)
    meth = shap_df["method"].iloc[0]
    fig.suptitle(f"Feature importance ({'mean |SHAP|' if meth == 'shap' else 'XGBoost gain'})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, "xgb_shap.png", outdir)


def plot_scatter(preds, outdir, horizon=3):
    """Predicted vs actual scatter with the 45-degree line."""
    d = preds[preds["horizon"] == horizon]
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, fac in zip(axes, FACTORS):
        g = d[d["factor"] == fac]
        if g.empty:
            ax.axis("off"); continue
        ax.scatter(g["actual"], g["xgb"], s=22, color=FC[fac], alpha=0.7, edgecolor="none")
        lo = float(min(g["actual"].min(), g["xgb"].min()))
        hi = float(max(g["actual"].max(), g["xgb"].max()))
        ax.plot([lo, hi], [lo, hi], color="#6b7280", lw=1.2, ls="--")
        r = float(np.corrcoef(g["actual"], g["xgb"])[0, 1]) if len(g) > 2 else np.nan
        ax.set_title(f"{fac}   corr = {r:.3f}", fontsize=10, color=FC[fac], fontweight="bold")
        ax.set_xlabel("actual"); ax.set_ylabel("predicted")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Predicted vs actual, horizon = {horizon} months", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, f"xgb_scatter_h{horizon}.png", outdir)


def plot_future(factors, future, outdir, tail=36):
    """History plus the forward forecast points."""
    if future.empty:
        return None
    hist = factors.sort_values("date").tail(tail)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True)
    for ax, fac in zip(axes, FACTORS):
        ax.plot(hist["date"], hist[fac], lw=2.0, color=FC[fac], marker="o", ms=3,
                label="observed")
        g = future[future["factor"] == fac].sort_values("target_date")
        if not g.empty:
            ax.plot(g["target_date"], g["pred_xgb"], lw=1.6, ls="--", marker="s", ms=6,
                    color="#111827", label="XGBoost forecast")
            last = hist.iloc[-1]
            ax.plot([last["date"], g["target_date"].iloc[0]],
                    [last[fac], g["pred_xgb"].iloc[0]], lw=1.0, ls=":", color="#111827")
            for _, r in g.iterrows():
                ax.annotate(f"{r['horizon']}m", (r["target_date"], r["pred_xgb"]),
                            textcoords="offset points", xytext=(0, 8), fontsize=7,
                            ha="center")
        ax.axvline(hist["date"].iloc[-1], color="#9ca3af", lw=0.9, ls=":")
        ax.set_ylabel(fac)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("date")
    fig.suptitle("Forward forecast from the last observed month", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "xgb_future.png", outdir)


# ================================================================== storage ===
def save_to_sqlite(metrics, preds, shap_df, future, summary, db=DB):
    con = sqlite3.connect(db)
    try:
        for df, t in ((metrics, T_METRIC), (preds, T_PRED),
                      (shap_df, T_SHAP), (future, T_FUTURE)):
            if df is not None and not df.empty:
                d = df.copy()
                # stringify datetimes for SQLite. Use pandas' own check: np.issubdtype
                # raises on extension dtypes such as StringDtype.
                for c in d.columns:
                    if pd.api.types.is_datetime64_any_dtype(d[c]):
                        d[c] = d[c].astype(str)
                d.to_sql(t, con, if_exists="replace", index=False)
        pd.DataFrame([summary]).to_sql(T_SUMMARY, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    out = []
    try:
        for t in (T_METRIC, T_PRED, T_SHAP, T_FUTURE, T_SUMMARY):
            try:
                d = pd.read_sql(f"select * from {t}", con)
                for c in ("origin", "target_date", "last_date"):
                    if c in d.columns:
                        d[c] = pd.to_datetime(d[c], errors="coerce")
                out.append(d)
            except Exception:
                out.append(pd.DataFrame())
    finally:
        con.close()
    return tuple(out)


# ====================================================================== run ===
def run(horizons=DEFAULT_HORIZONS, save=True, plots=True, verbose=True, outdir=HERE,
        log=None):
    """Full pipeline. `log` is an optional callable(str) so a GUI can show the
    training progress live while this runs on a background thread."""
    def emit(msg):
        if verbose:
            print(msg)
        if log:
            log(msg)

    import yield_curve_dns as ycd
    _c, factors, summ, _fc = ycd.load_from_sqlite(DB)
    if factors is None or factors.empty:
        raise RuntimeError("no DNS factors in SQLite — run yield_curve_dns.py or "
                           "download_bound.py first")
    factors = factors.sort_values("date").reset_index(drop=True)
    factors["date"] = pd.to_datetime(factors["date"])
    src = str(summ.iloc[0]["source"]) if summ is not None and not summ.empty else "unknown"
    is_demo = src.upper().startswith("DEMO")
    if verbose:
        print("=" * 88)
        print("XGBoost forecasts of the yield-curve factors")
        print("=" * 88)
    emit(f"source   : {src}")
    if is_demo:
        emit("*** SYNTHETIC DEMO DATA -- results are not meaningful ***")
    emit(f"periods  : {len(factors)}  "
         f"({factors['date'].min():%Y-%m} .. {factors['date'].max():%Y-%m})")
    emit(f"horizons : {list(horizons)}   lags: {LAGS}   min train: {MIN_TRAIN}")

    metrics, preds = walk_forward(factors, horizons, verbose=verbose, log=log)
    emit("computing SHAP feature importance ...")
    shap_df = shap_importance(factors, horizon=min(horizons, key=lambda h: abs(h - 3)))
    emit("forecasting forward from the last observed month ...")
    future = forecast_future(factors, horizons)

    figs = []
    if plots:
        emit("rendering figures ...")
        h_show = 3 if 3 in horizons else list(horizons)[0]
        for p in (plot_actual_vs_pred(preds, outdir, h_show),
                  plot_rel_rw(metrics, outdir),
                  plot_shap(shap_df, outdir),
                  plot_scatter(preds, outdir, h_show),
                  plot_future(factors, future, outdir)):
            if p:
                figs.append(p)

    n_beat = int(metrics["beats_RW"].sum()) if not metrics.empty else 0
    summary = {
        "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": src, "is_demo": int(is_demo),
        "n_periods": int(len(factors)),
        "date_min": str(factors["date"].min().date()),
        "date_max": str(factors["date"].max().date()),
        "horizons": ",".join(map(str, horizons)),
        "lags": LAGS, "min_train": MIN_TRAIN,
        "n_cells": int(len(metrics)), "n_beats_rw": n_beat,
        "best_rel_rw": float(metrics["rel_RW"].min()) if not metrics.empty else np.nan,
        "mean_rel_rw": float(metrics["rel_RW"].mean()) if not metrics.empty else np.nan,
        "n_figures": len(figs),
    }
    if save:
        save_to_sqlite(metrics, preds, shap_df, future, summary, DB)
        emit(f"saved to SQLite: {T_METRIC}, {T_PRED}, {T_SHAP}, {T_FUTURE}, {T_SUMMARY}")
    emit(f"DONE — XGBoost beat the random walk in {n_beat}/{len(metrics)} cells")

    if verbose:
        print("\n" + "-" * 88)
        print("ACCURACY RELATIVE TO THE RANDOM WALK  (rel_RW < 1 means XGBoost wins)")
        print("-" * 88)
        if not metrics.empty:
            show = metrics[["factor", "horizon", "n_oos", "RMSE_xgb", "RMSE_rw",
                            "rel_RW", "rel_AR1", "hit_rate", "beats_RW"]]
            print(show.to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))
            print(f"\nXGBoost beat the random walk in {n_beat}/{len(metrics)} cells")
            if n_beat == 0:
                print("  NOTE: beating a random walk on yield factors is genuinely hard.")
                print("        Reporting this honestly is the correct outcome.")
        if not future.empty:
            print("\nFORWARD FORECAST")
            print(future[["factor", "horizon", "target_date", "last_value",
                          "pred_xgb", "change"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if figs:
            print("\nFIGURES")
            for p in figs:
                print("  " + os.path.basename(p))
        if save:
            print(f"\nSaved: {T_METRIC}, {T_PRED}, {T_SHAP}, {T_FUTURE}, {T_SUMMARY}")
    return metrics, preds, shap_df, future, summary


def main():
    horizons = (1, 3) if "--quick" in sys.argv else DEFAULT_HORIZONS
    run(horizons=horizons, save="--no-save" not in sys.argv,
        plots="--no-plots" not in sys.argv)
    print("\nDone.")


if __name__ == "__main__":
    main()
