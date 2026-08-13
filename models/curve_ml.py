# -*- coding: utf-8 -*-
"""
curve_ml.py -- Machine-learning forecasting of the yield-curve factors
               (Level, Slope, Curvature) extracted by yield_curve_dns.py.

Implements the model comparison described in the CMDF-0128-2568 progress report
(sections 2.3.1 / 3.1): DNS factors are forecast h = 1, 3, 6, 12 months ahead and
several estimators are compared under a *recursive expanding-window out-of-sample*
design -- the same protocol the report uses, so nothing is fitted on the future.

Models
    RW          random walk (no-change) -- the benchmark every forecaster must beat
    AR(1)       univariate autoregression on the factor itself
    VAR(1)      the three factors jointly (captures Level<->Slope<->Curvature spillover)
    Ridge       penalised linear regression on the lag block
    LASSO       L1 -- the report finds it strong at 3-6 months
    ElasticNet  L1 + L2
    RandomForest, XGBoost, LightGBM   non-linear ensembles

Features (all strictly lagged, no look-ahead)
    p lags of Level, Slope and Curvature (default p = 3)
    plus, when available, the raw yields (1Y / 10Y) and their spread.

Accuracy is reported as RMSE, MAE and RMSE relative to the random walk
(< 1 means the model beats the random walk).

Run:
    python curve_ml.py                 # uses the DNS factors already in SQLite
    python curve_ml.py --horizons 1,3,6,12 --lags 3
"""
from __future__ import annotations

import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from thaibma_paths import DATA_ROOT  # data lives outside the repo

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
T_FORECAST = "curve_ml_forecast"
T_PRED = "curve_ml_prediction"
T_SUMMARY = "curve_ml_summary"

FACTORS = ["Level", "Slope", "Curvature"]
DEFAULT_HORIZONS = (1, 3, 6, 12)
DEFAULT_LAGS = 3
MIN_TRAIN = 48                      # months before the first out-of-sample forecast


# ================================================================ features ====
def build_features(f: pd.DataFrame, lags: int = DEFAULT_LAGS) -> pd.DataFrame:
    """Lagged design matrix. Every column is known at time t."""
    d = f.sort_values("date").reset_index(drop=True).copy()
    cols = {}
    for fac in FACTORS:
        for L in range(lags):
            cols[f"{fac}_l{L}"] = d[fac].shift(L)
    for extra in ("y_1y", "y_10y"):
        if extra in d.columns and d[extra].notna().any():
            cols[f"{extra}_l0"] = d[extra].shift(0)
    if {"y_1y", "y_10y"}.issubset(d.columns):
        cols["spread_l0"] = (d["y_10y"] - d["y_1y"]).shift(0)
    X = pd.DataFrame(cols, index=d.index)
    X.insert(0, "date", d["date"])
    for fac in FACTORS:
        X[f"target_{fac}"] = d[fac]
    return X


# ================================================================== models ====
def _models(seed: int = 0) -> dict:
    import lightgbm as lgb
    import xgboost as xgb
    return {
        "Ridge": lambda: Ridge(alpha=1.0, random_state=seed),
        "LASSO": lambda: Lasso(alpha=0.01, max_iter=5000, random_state=seed),
        "ElasticNet": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,
                                         random_state=seed),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=6,
                                                      n_jobs=4, random_state=seed),
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                            subsample=0.85, colsample_bytree=0.85,
                                            reg_lambda=2.0, n_jobs=4, random_state=seed,
                                            verbosity=0),
        "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                              subsample=0.85, colsample_bytree=0.85,
                                              n_jobs=4, random_state=seed, verbose=-1),
    }


def _var1_forecast(hist: np.ndarray, h: int) -> np.ndarray:
    """VAR(1) on the 3 factors, iterated h steps. hist = (T, 3)."""
    Y, X = hist[1:], np.column_stack([np.ones(len(hist) - 1), hist[:-1]])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)          # (4, 3)
    x = hist[-1].copy()
    for _ in range(h):
        x = B[0] + x @ B[1:]
    return x


def _ar1_forecast(y: np.ndarray, h: int) -> float:
    if len(y) < 6 or np.std(y[:-1]) < 1e-12:
        return float(y[-1])
    b = np.polyfit(y[:-1], y[1:], 1)
    p = float(y[-1])
    for _ in range(h):
        p = b[0] * p + b[1]
    return p


# =========================================================== walk-forward =====
def walk_forward(f: pd.DataFrame, horizons=DEFAULT_HORIZONS, lags: int = DEFAULT_LAGS,
                 min_train: int = MIN_TRAIN, verbose: bool = True):
    """Recursive expanding-window out-of-sample forecasting of every factor."""
    X = build_features(f, lags).dropna().reset_index(drop=True)
    feat_cols = [c for c in X.columns if c not in ("date",) and not c.startswith("target_")]
    n = len(X)
    if n <= min_train + max(horizons) + 5:
        raise RuntimeError(f"only {n} usable periods; need > {min_train + max(horizons) + 5}")

    makers = _models()
    rows, preds = [], []
    for h in horizons:
        for t in range(min_train, n - h):
            tr = X.iloc[:t + 1]
            cur = X.iloc[[t]]
            hist = tr[[f"target_{k}" for k in FACTORS]].to_numpy(dtype=float)
            actual = {k: float(X.iloc[t + h][f"target_{k}"]) for k in FACTORS}
            date_t = X.iloc[t]["date"]
            date_h = X.iloc[t + h]["date"]

            # --- benchmarks -------------------------------------------------
            for k_i, k in enumerate(FACTORS):
                preds.append({"model": "RW", "factor": k, "horizon": h,
                              "origin": date_t, "target_date": date_h,
                              "pred": float(hist[-1, k_i]), "actual": actual[k]})
                preds.append({"model": "AR(1)", "factor": k, "horizon": h,
                              "origin": date_t, "target_date": date_h,
                              "pred": _ar1_forecast(hist[:, k_i], h), "actual": actual[k]})
            v = _var1_forecast(hist, h)
            for k_i, k in enumerate(FACTORS):
                preds.append({"model": "VAR(1)", "factor": k, "horizon": h,
                              "origin": date_t, "target_date": date_h,
                              "pred": float(v[k_i]), "actual": actual[k]})

            # --- ML: direct h-step, one model per factor --------------------
            Xtr = tr[feat_cols].to_numpy(dtype=float)[:-h] if h > 0 else tr[feat_cols].to_numpy(float)
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xcur_s = sc.transform(Xtr), sc.transform(cur[feat_cols].to_numpy(float))
            for k in FACTORS:
                ytr = tr[f"target_{k}"].to_numpy(dtype=float)[h:]
                if len(ytr) < 24:
                    continue
                for mname, mk in makers.items():
                    try:
                        mdl = mk().fit(Xtr_s, ytr)
                        p = float(mdl.predict(Xcur_s)[0])
                    except Exception:
                        continue
                    preds.append({"model": mname, "factor": k, "horizon": h,
                                  "origin": date_t, "target_date": date_h,
                                  "pred": p, "actual": actual[k]})
        if verbose:
            print(f"  horizon {h:2d}m done")

    P = pd.DataFrame(preds)
    P["err"] = P["pred"] - P["actual"]
    g = P.groupby(["model", "factor", "horizon"])
    res = g.agg(RMSE=("err", lambda e: float(np.sqrt(np.mean(np.square(e))))),
                MAE=("err", lambda e: float(np.mean(np.abs(e)))),
                n=("err", "size")).reset_index()
    rw = res[res["model"] == "RW"].set_index(["factor", "horizon"])["RMSE"]
    res["rel_RW"] = [r["RMSE"] / rw.get((r["factor"], r["horizon"]), np.nan)
                     for _, r in res.iterrows()]
    return res.sort_values(["factor", "horizon", "RMSE"]).reset_index(drop=True), P


def latest_forecast(f: pd.DataFrame, horizons=DEFAULT_HORIZONS, lags: int = DEFAULT_LAGS,
                    model: str = "LightGBM") -> pd.DataFrame:
    """Fit on ALL available history and forecast forward from the last observation."""
    X = build_features(f, lags).dropna().reset_index(drop=True)
    feat_cols = [c for c in X.columns if c != "date" and not c.startswith("target_")]
    makers = _models()
    mk = makers.get(model)
    out = []
    last_date = pd.Timestamp(X.iloc[-1]["date"])
    hist = X[[f"target_{k}" for k in FACTORS]].to_numpy(dtype=float)
    for h in horizons:
        Xtr = X[feat_cols].to_numpy(dtype=float)[:-h]
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xcur_s = sc.transform(X[feat_cols].to_numpy(dtype=float)[[-1]])
        v = _var1_forecast(hist, h)
        for k_i, k in enumerate(FACTORS):
            ytr = X[f"target_{k}"].to_numpy(dtype=float)[h:]
            pred = float("nan")
            if mk is not None and len(ytr) >= 24:
                try:
                    pred = float(mk().fit(Xtr_s, ytr).predict(Xcur_s)[0])
                except Exception:
                    pred = float("nan")
            out.append({"factor": k, "horizon": h,
                        "target_date": (last_date + pd.DateOffset(months=h)).strftime("%Y-%m"),
                        "last_value": float(hist[-1, k_i]),
                        f"pred_{model}": pred,
                        "pred_VAR(1)": float(v[k_i]),
                        "pred_RW": float(hist[-1, k_i])})
    return pd.DataFrame(out)


# ============================================================ persistence =====
def save_to_sqlite(res, preds, latest, meta, db=DB):
    con = sqlite3.connect(db)
    res.to_sql(T_FORECAST, con, if_exists="replace", index=False)
    p = preds.copy()
    for c in ("origin", "target_date"):
        p[c] = p[c].astype(str)
    p.tail(20000).to_sql(T_PRED, con, if_exists="replace", index=False)
    latest.to_sql("curve_ml_latest", con, if_exists="replace", index=False)
    pd.DataFrame([meta]).to_sql(T_SUMMARY, con, if_exists="replace", index=False)
    con.commit(); con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    try:
        res = pd.read_sql_query(f"SELECT * FROM {T_FORECAST}", con)
        latest = pd.read_sql_query("SELECT * FROM curve_ml_latest", con)
        summ = pd.read_sql_query(f"SELECT * FROM {T_SUMMARY} LIMIT 1", con)
    except Exception:
        res = latest = summ = pd.DataFrame()
    finally:
        con.close()
    return res, latest, summ


def run(horizons=DEFAULT_HORIZONS, lags=DEFAULT_LAGS, save=True, verbose=True):
    import yield_curve_dns as ycd
    _curve, factors, summary, _fc = ycd.load_from_sqlite(DB)
    if factors.empty:
        raise RuntimeError("no DNS factors in SQLite -- run the Yield Curve menu first "
                           "(Estimate DNS or Demo curve).")
    if verbose:
        print(f"factors: {len(factors)} periods "
              f"({factors['date'].min():%Y-%m} .. {factors['date'].max():%Y-%m})")
    res, preds = walk_forward(factors, horizons, lags, verbose=verbose)
    # the champion must be a model `latest_forecast` can actually refit; AR(1)/VAR(1)/RW
    # are benchmarks computed inline and are reported separately.
    ml_names = set(_models().keys())
    best = (res[res["model"].isin(ml_names)].sort_values("RMSE")
            .groupby(["factor", "horizon"]).head(1))
    champion = (best.groupby("model").size().sort_values(ascending=False).index[0]
                if not best.empty else "LightGBM")
    overall_best = (res[res["model"] != "RW"].sort_values("RMSE").iloc[0]["model"]
                    if (res["model"] != "RW").any() else "RW")
    latest = latest_forecast(factors, horizons, lags, model=champion)
    src = str(summary.iloc[0]["source"]) if not summary.empty else "n/a"
    meta = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": src, "is_demo": int(str(src).upper().startswith("DEMO")),
            "n_periods": int(len(factors)), "lags": int(lags),
            "horizons": ",".join(map(str, horizons)),
            "last_obs": f"{factors['date'].max():%Y-%m}",
            "champion": champion,                 # best ML model (refittable)
            "overall_best": overall_best,          # best of all non-RW incl. AR/VAR
            "n_models": int(res["model"].nunique()),
            "beats_rw": int((res["rel_RW"] < 1).sum()),
            "n_cells": int(len(res))}
    if save:
        save_to_sqlite(res, preds, latest, meta)
    return res, preds, latest, meta


def main():
    hz = DEFAULT_HORIZONS
    if "--horizons" in sys.argv:
        hz = tuple(int(x) for x in sys.argv[sys.argv.index("--horizons") + 1].split(","))
    lags = DEFAULT_LAGS
    if "--lags" in sys.argv:
        lags = int(sys.argv[sys.argv.index("--lags") + 1])
    print("=" * 96)
    print("ML forecasting of yield-curve factors (Level / Slope / Curvature)")
    print("recursive expanding-window out-of-sample -- CMDF-0128-2568 sec 2.3.1")
    print("=" * 96)
    res, _p, latest, meta = run(horizons=hz, lags=lags, save="--no-save" not in sys.argv)
    if meta["is_demo"]:
        print("*** factors came from SYNTHETIC DEMO data -- not real market data ***")

    for fac in FACTORS:
        print(f"\n{fac.upper()}  (RMSE, and RMSE relative to random walk)")
        sub = res[res["factor"] == fac]
        models = sub["model"].unique()
        print(f"  {'model':14s}" + "".join(f"{f'h={h}m':>16s}" for h in hz))
        for m in models:
            s = sub[sub["model"] == m].set_index("horizon")
            cells = []
            for h in hz:
                if h in s.index:
                    cells.append(f"{s.loc[h,'RMSE']:.4f} ({s.loc[h,'rel_RW']:.2f})")
                else:
                    cells.append("-")
            print(f"  {m:14s}" + "".join(f"{c:>16s}" for c in cells))
        b = sub[sub["model"] != "RW"].sort_values("RMSE").iloc[0]
        print(f"  -> best: {b['model']} at h={int(b['horizon'])}m "
              f"(RMSE {b['RMSE']:.4f}, {b['rel_RW']:.2f}x random walk)")

    print(f"\nBEST ML MODEL: {meta['champion']}  |  best overall (incl. AR/VAR): "
          f"{meta['overall_best']}")
    print(f"  {meta['beats_rw']}/{meta['n_cells']} model-factor-horizon cells beat the random walk")
    if meta["is_demo"] and meta["beats_rw"] * 4 < meta["n_cells"]:
        print("  NOTE: the demo factors are generated as random walks, so almost nothing can")
        print("        beat RW by construction. Re-run on a real exported curve to judge this.")
    print(f"\nFORWARD FORECAST from {meta['last_obs']}")
    print(latest.to_string(index=False))
    if "--no-save" not in sys.argv:
        print(f"\nSaved: {T_FORECAST}, {T_PRED}, curve_ml_latest, {T_SUMMARY}  ({DB})")
    print("\nDone.")


if __name__ == "__main__":
    main()
