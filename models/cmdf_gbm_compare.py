# -*- coding: utf-8 -*-
"""
cmdf_gbm_compare.py -- CatBoost / XGBoost / LightGBM comparison for ln(PD)_12m.

Extends the model comparison in CMDF_threshold.py (Ridge, Random Forest, XGBoost,
LightGBM) with CatBoost, and reproduces Table 1 of the progress report in the same
layout: Model | R2 | RMSE | MAE | Spearman, for the Expanded and the ESG samples.

TARGET
    ln_pd12m -- the log of the Merton-implied 12-month probability of default.
    This is a REGRESSION problem, not a default/no-default classification: the
    report ranks issuers by implied risk rather than predicting realised events.

VALIDATION
    Expanding-window out-of-time, exactly as in CMDF_threshold.step_model_comparison:
        train on year <= t, test on t < year <= t+3, for t = 2013, 2016, 2019, ...
    R2 is the MEDIAN across folds; RMSE / MAE / Spearman are means. Reporting the
    median R2 follows the original script (one bad fold should not dominate).

INPUTS
    Rev01_Database_final.dta. The winsorised *_w features and the sample flags that
    CMDF_threshold expects are not stored in that file, so they are rebuilt here:
      *_w          1st/99th percentile winsorisation
      sample_noESG every firm-month that has the full feature set (Expanded)
      sample       firm-months that also carry an ESG score (ESG)

RUN
    python cmdf_gbm_compare.py
    python cmdf_gbm_compare.py --expanded-only
    python cmdf_gbm_compare.py --no-save
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import warnings

import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
DTA = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"

TARGET = "ln_pd12m"
ANALYSIS_START = 2010
EXPAND_FIRST_CUT = 2013
EXPAND_STEP = 3
MIN_TRAIN, MIN_TEST = 500, 100

T_TABLE = "cmdf_gbm_comparison"
T_FOLDS = "cmdf_gbm_folds"
T_IMPORT = "cmdf_gbm_importance"

# same 22 determinants as CMDF_threshold.FEATURES
FEATURES = [
    "lnTotalAssets", "AgeYear", "ROA", "ROE", "EBITtoTA", "REtoTA", "DE_w", "TDTA",
    "LTDtoTA", "STDtoTA", "CurrentRatio", "QuickRatio", "CashRatio",
    "WorkingCapitaltoTA_w", "cf_Interestcoverageratio_w",
    "cf_DebtServiceCoverageRatio_w", "amihud_monthly_100_w", "adj_illiq_kz_w",
    "ln_amihud", "Policyrate", "GDPgrowth", "Unemploymentratenationalesti",
]
# columns that CMDF_threshold reads as *_w but that the raw .dta stores unwinsorised
WINSOR = {
    "DE_w": "DE", "WorkingCapitaltoTA_w": "WorkingCapitaltoTA",
    "cf_Interestcoverageratio_w": "cf_Interestcoverageratio",
    "cf_DebtServiceCoverageRatio_w": "cf_DebtServiceCoverageRatio",
    "amihud_monthly_100_w": "amihud_monthly_100", "adj_illiq_kz_w": "adj_illiq_kz",
}

PRETTY = {
    "L_lnTotalAssets": "ln(Assets)", "L_AgeYear": "Firm Age", "L_ROA": "ROA",
    "L_ROE": "ROE", "L_EBITtoTA": "EBIT/TA", "L_REtoTA": "RE/TA", "L_DE_w": "D/E",
    "L_TDTA": "Total Debt/TA", "L_LTDtoTA": "LT Debt/TA", "L_STDtoTA": "ST Debt/TA",
    "L_CurrentRatio": "Current Ratio", "L_QuickRatio": "Quick Ratio",
    "L_CashRatio": "Cash Ratio", "L_WorkingCapitaltoTA_w": "WC/TA",
    "L_cf_Interestcoverageratio_w": "Interest Coverage",
    "L_cf_DebtServiceCoverageRatio_w": "DSCR",
    "L_amihud_monthly_100_w": "Amihud", "L_adj_illiq_kz_w": "Kang-Zhang",
    "L_ln_amihud": "ln(Amihud)", "L_Policyrate": "Policy Rate",
    "L_GDPgrowth": "GDP Growth", "L_Unemploymentratenationalesti": "Unemployment",
    "L_ESGScore": "ESG Score",
}


# ================================================================== data =====
def _winsorise(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def load_full(path=DTA, verbose=True) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    if verbose:
        print(f"  reading {os.path.basename(path)} ...")
    df = pd.read_stata(path, convert_categoricals=False)
    if verbose:
        print(f"  {len(df):,} rows x {len(df.columns)} columns")

    for w, raw in WINSOR.items():
        if w not in df.columns:
            if raw not in df.columns:
                raise KeyError(f"neither {w} nor {raw} is in the dataset")
            df[w] = _winsorise(df[raw])

    if "month_year" not in df.columns:
        # CMDF_threshold sorts on month_year; rebuild it if the raw file lacks it
        for cand in ("q_date", "Date_12m", "Date_Financial"):
            if cand in df.columns:
                df["month_year"] = pd.to_datetime(df[cand], errors="coerce")
                break
        else:
            df["month_year"] = pd.to_datetime(df["year"].astype(str) + "-12-31",
                                              errors="coerce")

    # Sample flags. The Expanded sample is every firm-month with the full feature
    # set; the ESG sample additionally requires an ESG score.
    have_feat = df[FEATURES].notna().all(axis=1) & df[TARGET].notna()
    df["sample_noESG"] = have_feat.astype(int)
    df["sample"] = (have_feat & df.get("ESGScore", pd.Series(np.nan,
                                                             index=df.index)).notna()).astype(int)
    if verbose:
        print(f"  sample_noESG = {int(df['sample_noESG'].sum()):,} rows | "
              f"sample (ESG) = {int(df['sample'].sum()):,} rows")
    return df


def build_sample(df_full, flag, add_esg=False, start=ANALYSIS_START):
    """Identical construction to CMDF_threshold.build_sample: lag every feature by
    one firm-month so nothing contemporaneous leaks into the prediction."""
    df = df_full.sort_values(["firm_id", "month_year"]).copy()
    feats = FEATURES + (["ESGScore"] if add_esg else [])
    lagged = {f"L_{v}": df.groupby("firm_id", observed=True)[v].shift(1) for v in feats}
    df = pd.concat([df, pd.DataFrame(lagged, index=df.index)], axis=1)
    df = df[df[flag] == 1].copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= start].copy()
    lag = [f"L_{v}" for v in feats]
    keep = lag + [TARGET, "firm_id", "year", "month_year"]
    df = df[keep].dropna(subset=lag + [TARGET]).reset_index(drop=True)
    return df, lag


# ================================================================ models =====
def models(seed=42):
    """Ridge / RF / XGBoost / LightGBM use the exact settings from
    CMDF_threshold.step_model_comparison so the existing rows stay comparable;
    CatBoost is added with a matching depth / rate / tree budget."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    import xgboost as xgb
    m = {
        "Ridge (linear)": lambda: Ridge(alpha=1.0),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=120, max_depth=10, min_samples_leaf=30,
            n_jobs=-1, random_state=seed),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
            n_jobs=-1, random_state=seed, verbosity=0),
        # importance_type="gain" is set explicitly because LightGBM defaults to
        # "split", the raw count of times a feature is used to branch. Split counts
        # reward features with many distinct values regardless of how much error they
        # remove, which is what put Firm Age above D/E in the importance table while
        # the same model's SHAP ranking agreed with XGBoost at a Spearman of 0.994.
        # The setting only changes what feature_importances_ reports; training and
        # every prediction are unaffected.
        "LightGBM": lambda: lgb.LGBMRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0,
            importance_type="gain",
            n_jobs=-1, random_state=seed, verbose=-1),
    }
    try:
        from catboost import CatBoostRegressor
        m["CatBoost"] = lambda: CatBoostRegressor(
            iterations=400, depth=4, learning_rate=0.05, l2_leaf_reg=3.0,
            random_seed=seed, verbose=0, allow_writing_files=False)
    except ImportError:
        print("  NOTE: catboost is not installed — install it with "
              "`pip install catboost` to include that row.")
    return m


def compare(df_full, samples=(("Expanded", "sample_noESG", False),
                              ("ESG", "sample", True)), verbose=True):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    rows, folds = [], []
    for label, flag, esg in samples:
        df, lag = build_sample(df_full, flag, esg)
        if df.empty:
            if verbose:
                print(f"  {label}: no rows — skipped")
            continue
        X = df[lag].reset_index(drop=True)
        y = df[TARGET].reset_index(drop=True)
        yr = df["year"].reset_index(drop=True)
        cuts = [c for c in range(EXPAND_FIRST_CUT, int(yr.max()), EXPAND_STEP)]
        if verbose:
            print(f"\n  {label} sample: {len(df):,} firm-months, "
                  f"{df['firm_id'].nunique():,} firms, {len(lag)} features, "
                  f"{yr.min()}-{yr.max()}")
        for name, ctor in models().items():
            r2, rm, ma, sp, nfold = [], [], [], [], 0
            t0 = time.time()
            for t in cuts:
                tr = (yr <= t).values
                te = ((yr > t) & (yr <= t + EXPAND_STEP)).values
                if te.sum() < MIN_TEST or tr.sum() < MIN_TRAIN:
                    continue
                Xtr, Xte = X[tr], X[te]
                if name == "Ridge (linear)":
                    sc = StandardScaler().fit(Xtr)
                    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                m = ctor()
                m.fit(Xtr, y[tr])
                p = m.predict(Xte)
                f_r2 = r2_score(y[te], p)
                f_rm = float(np.sqrt(mean_squared_error(y[te], p)))
                f_ma = mean_absolute_error(y[te], p)
                f_sp = spearmanr(y[te], p).correlation
                r2.append(f_r2); rm.append(f_rm); ma.append(f_ma); sp.append(f_sp)
                nfold += 1
                folds.append(dict(sample=label, model=name, cut_year=t,
                                  n_train=int(tr.sum()), n_test=int(te.sum()),
                                  R2=float(f_r2), RMSE=f_rm, MAE=float(f_ma),
                                  Spearman=float(f_sp)))
            if not r2:
                continue
            rows.append(dict(
                sample=label, model=name, n_folds=nfold,
                R2=float(np.median(r2)), RMSE=float(np.mean(rm)),
                MAE=float(np.mean(ma)), Spearman=float(np.nanmean(sp)),
                R2_mean=float(np.mean(r2)), R2_min=float(np.min(r2)),
                R2_max=float(np.max(r2)), seconds=round(time.time() - t0, 1),
                n_obs=int(len(df)), n_firms=int(df["firm_id"].nunique())))
            if verbose:
                print(f"    {name:16s} R2={np.median(r2):6.3f}  "
                      f"RMSE={np.mean(rm):6.3f}  MAE={np.mean(ma):6.3f}  "
                      f"Spearman={np.nanmean(sp):6.3f}  ({nfold} folds, "
                      f"{time.time()-t0:.0f}s)")
    return pd.DataFrame(rows), pd.DataFrame(folds)


def importance(df_full, top=22, verbose=True):
    """Gain importance from each gradient-boosting model, fitted on the full
    Expanded sample. Kept separate from SHAP, which CMDF_threshold already does."""
    df, lag = build_sample(df_full, "sample_noESG", False)
    X, y = df[lag], df[TARGET]
    out = []
    for name, ctor in models().items():
        if name in ("Ridge (linear)", "Random Forest"):
            continue
        try:
            m = ctor()
            m.fit(X, y)
            imp = np.asarray(m.feature_importances_, dtype=float)
            imp = imp / (imp.sum() or 1.0)
            for c, v in zip(lag, imp):
                out.append({"model": name, "feature": c,
                            "pretty": PRETTY.get(c, c), "importance": float(v)})
        except Exception as ex:
            if verbose:
                print(f"    importance for {name} failed: {ex}")
    d = pd.DataFrame(out)
    if not d.empty:
        d["rank"] = d.groupby("model")["importance"].rank(ascending=False)
    return d


# =============================================================== storage =====
def save_to_sqlite(table, folds, imp, db=DB):
    con = sqlite3.connect(db)
    try:
        for d, t in ((table, T_TABLE), (folds, T_FOLDS), (imp, T_IMPORT)):
            if d is not None and not d.empty:
                d.to_sql(t, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    out = []
    try:
        for t in (T_TABLE, T_FOLDS, T_IMPORT):
            try:
                out.append(pd.read_sql(f"select * from {t}", con))
            except Exception:
                out.append(pd.DataFrame())
    finally:
        con.close()
    return tuple(out)


def report_table(table: pd.DataFrame, sample="Expanded") -> str:
    """Table 1 of the report, same columns and ordering."""
    d = table[table["sample"] == sample].copy()
    if d.empty:
        return f"(no rows for the {sample} sample)"
    order = ["Ridge (linear)", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    d["_o"] = d["model"].map({m: i for i, m in enumerate(order)}).fillna(99)
    d = d.sort_values("_o")
    best = d.loc[d["R2"].idxmax(), "model"]
    lines = [f"Table 1. Model Performance Comparison ({sample} Sample)",
             f"{'Model':<18}{'R2':>8}{'RMSE':>9}{'MAE':>9}{'Spearman':>11}",
             "-" * 55]
    for _, r in d.iterrows():
        star = "  <-- best" if r["model"] == best else ""
        lines.append(f"{r['model']:<18}{r['R2']:>8.3f}{r['RMSE']:>9.3f}"
                     f"{r['MAE']:>9.3f}{r['Spearman']:>11.3f}{star}")
    n = d.iloc[0]
    lines.append("-" * 55)
    lines.append(f"{int(n['n_obs']):,} firm-months from {int(n['n_firms']):,} firms; "
                 f"{int(n['n_folds'])} expanding-window folds")
    return "\n".join(lines)


# =================================================================== run =====
def run(expanded_only=False, save=True, verbose=True):
    if verbose:
        print("=" * 78)
        print("CatBoost / XGBoost / LightGBM comparison for ln(PD)_12m")
        print("=" * 78)
    df_full = load_full(verbose=verbose)
    samples = (("Expanded", "sample_noESG", False),)
    if not expanded_only:
        samples = samples + (("ESG", "sample", True),)
    table, folds = compare(df_full, samples, verbose=verbose)
    if verbose:
        print("\n  feature importance (gradient-boosting models) ...")
    imp = importance(df_full, verbose=verbose)
    if save:
        save_to_sqlite(table, folds, imp, DB)
        table.to_csv(os.path.join(HERE, "cmdf_gbm_comparison.csv"), index=False)
        if verbose:
            print(f"\n  saved: {T_TABLE}, {T_FOLDS}, {T_IMPORT} "
                  f"+ cmdf_gbm_comparison.csv")
    return table, folds, imp


def main():
    table, folds, imp = run(expanded_only="--expanded-only" in sys.argv,
                            save="--no-save" not in sys.argv)
    print()
    for s in table["sample"].unique():
        print(report_table(table, s))
        print()
    if not imp.empty:
        print("Top 10 determinants by gain (mean across boosting models)")
        g = (imp.groupby("pretty")["importance"].mean()
             .sort_values(ascending=False).head(10))
        for k, v in g.items():
            print(f"  {k:<22} {v:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
