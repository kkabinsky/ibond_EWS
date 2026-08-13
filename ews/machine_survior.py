# -*- coding: utf-8 -*-
"""
machine_survior.py  --  Logistic vs XGBoost as the Approach-1 hazard estimator.

Same survival pipeline and output as survivor2.py (Cox-style discrete-time hazard
-> PD_3M -> Momentum -> hyperbolic boundary -> lead time), but the hazard estimator
is SWAPPABLE:

    Logistic : h = sigmoid(baseline(t) + beta . X)     (interpretable, odds ratios)
    XGBoost  : h = gradient-boosted trees(X)           (non-linear, explained by SHAP)

The two models are trained on the SAME data / features / representation, then compared
head-to-head:
  * PD_3M ROC-AUC (in-sample + out-of-sample), MCC, precision, recall, F1, flagged vol
  * lead-time table (final sustained alarm run) side-by-side per firm
  * SHAP feature ranking for XGBoost (odds ratios for logistic)
  * STATISTICAL TESTS on the common out-of-sample set:
      - bootstrap paired ROC-AUC difference (XGBoost - Logistic) with 95% CI + p-value
      - McNemar test on the two models' flag decisions
    -> a verdict on which model is better.

Run:  python machine_survior.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy import stats as sps
import xgboost as xgb

import survival
import survivor2
from survivor2 import (BASELINE_COVS, P_STAR_DEFAULT as P_STAR, DAYS_PER_MONTH,
                       lead_time_table, print_table)

roc_auc_score = survival.roc_auc_score
mcc = survival.matthews_corrcoef


# ------------------------------------------------------------ estimators ------
def fit_hazard_logistic(df, covs=None):
    return survival.fit_hazard(df, covs)          # the existing logistic hazard


def fit_hazard_xgb(df, covs=None):
    """Same feature representation as survival.fit_hazard, but the estimator is an
    XGBoost classifier (binary:logistic) with scale_pos_weight for the rare events."""
    covs = survival._get_covs(df, covs)
    d = df.copy()
    for c in covs:                                   # standardize (keeps _haz compatible)
        vals = pd.to_numeric(d[c], errors="coerce")
        med = vals.median()
        d[c] = (vals.fillna(med).fillna(0.0) - vals.mean()) / (vals.std() + 1e-9)
    tref = {"mu": d["month_index"].mean(), "sd": d["month_index"].std() + 1e-9}
    X = np.column_stack([survival._base(d["month_index"].values, tref), d[covs].values])
    X = np.nan_to_num(X, nan=0.0)
    y = d["event"].astype(int).values
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3,
        scale_pos_weight=spw, objective="binary:logistic", eval_metric="logloss",
        n_jobs=4, random_state=0, verbosity=0)
    clf.fit(X, y)
    stats = {c: (df[c].mean(), df[c].std() + 1e-9) for c in covs}
    return {"clf": clf, "tref": tref, "covs": covs, "stats": stats}


# --------------------------------------------------------- run one model ------
def run_with(panel, fit_fn, train_frac=0.7):
    """survival.run() with a pluggable hazard fit function. Returns the in-sample
    signals df, a meta dict (in-sample + out-of-sample metrics), and the OOS test
    dataframe (for the statistical comparison)."""
    model = fit_fn(panel)
    df = survival.add_signals(panel, model)
    bnd = survival.tune_boundary(df)
    tau_pd = float(np.nanquantile(df["PD_3M"], 0.85))
    df = survival.apply_signals(df, bnd["alpha"], bnd["logK"], tau_pd)
    d_in = df.dropna(subset=["y_fwd"])
    auc_in = survival._auc(d_in["y_fwd"], d_in["PD_3M"])

    months = np.sort(panel["month_index"].unique())
    meta = dict(model=model, boundary=bnd, tau_pd=tau_pd, pd_auc=auc_in,
                pd_auc_oos=float("nan"), persistence_auc=float("nan"),
                cut_month=None, n_test=0, oos_pd=None, oos_rs=None)
    te = None
    onsets = np.sort(panel.loc[panel["event"] == 1, "month_index"].unique())
    if len(onsets) >= 4:
        cut = int(onsets[max(1, int(len(onsets) * train_frac)) - 1])
    elif len(months) >= 8:
        cut = int(months[int(len(months) * train_frac)])
    else:
        cut = None
    if cut is not None:
        tr_panel = panel[panel["month_index"] <= cut]
        if tr_panel["event"].sum() >= 2 and tr_panel["event"].nunique() > 1:
            m_tr = fit_fn(tr_panel)
            full = survival.add_signals(panel, m_tr)
            tr = full[full["month_index"] <= cut]
            bnd_o = survival.tune_boundary(tr) or bnd
            tau_o = float(np.nanquantile(tr["PD_3M"], 0.85))
            te = full[full["month_index"] > cut].dropna(subset=["y_fwd"]).copy()
            if len(te) and te["y_fwd"].nunique() > 1:
                te = survival.apply_signals(te, bnd_o["alpha"], bnd_o["logK"], tau_o)
                meta.update(pd_auc_oos=survival._auc(te["y_fwd"], te["PD_3M"]),
                            persistence_auc=survival._auc(te["y_fwd"], te["event"]),
                            cut_month=cut, n_test=int(len(te)),
                            oos_pd=survival.evaluate(te, "flag_PD"),
                            oos_rs=survival.evaluate(te, "flag_RS"))
    return df, meta, te


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# --------------------------------------------------------- explainability -----
def odds_ratios_top(model, top=8):
    hr = survivor2.odds_ratios(model)
    return [(k, v) for k, v in list(hr.items())[:top]], "odds ratio exp(beta)"


def shap_top(model, panel, top=8, n_sample=1500, seed=0):
    """mean|SHAP| feature ranking for the XGBoost hazard (TreeExplainer)."""
    covs = model["covs"]
    d = panel.sample(min(n_sample, len(panel)), random_state=seed)
    cols = []
    for c in covs:
        vals = pd.to_numeric(d[c], errors="coerce")
        mu, sd = model["stats"][c]
        cols.append(np.nan_to_num((vals.fillna(mu) - mu) / sd, nan=0.0))
    Xstd = np.column_stack(cols)
    X = np.column_stack([survival._base(d["month_index"].values, model["tref"]), Xstd])
    X = np.nan_to_num(X, nan=0.0)
    try:
        import shap
        sv = shap.TreeExplainer(model["clf"]).shap_values(X)
        imp = np.abs(sv).mean(axis=0)[survival._DEG:]      # drop baseline-time columns
        s = pd.Series(imp, index=covs).sort_values(ascending=False)
        return [(k, v) for k, v in list(s.items())[:top]], "mean |SHAP|"
    except Exception:                                       # fallback: XGBoost gain
        imp = model["clf"].feature_importances_[survival._DEG:]
        s = pd.Series(imp, index=covs).sort_values(ascending=False)
        return [(k, v) for k, v in list(s.items())[:top]], "XGBoost gain (shap unavailable)"


# ------------------------------------------------------------ report ----------
def report_model(label, sub, panel, fit_fn, is_xgb):
    print("\n" + "=" * 78)
    print(f"MODEL: {label}   ({sub})")
    print("=" * 78)
    df, meta, te = run_with(panel, fit_fn)

    imp, imp_name = (shap_top(meta["model"], panel) if is_xgb
                     else odds_ratios_top(meta["model"]))
    print(f"[Stage 2-3] top features by {imp_name}:")
    for k, v in imp:
        print(f"      {k:34s} {v:10.4f}")

    e = meta["oos_pd"] or survival.evaluate(df, "flag_PD")
    f1 = _f1(e["precision"], e["recall"])
    print("\nPerformance (PD_3M):")
    print(f"  ROC-AUC  in-sample {meta['pd_auc']:.3f}   out-of-sample {meta['pd_auc_oos']:.3f}"
          f"   persistence {meta['persistence_auc']:.3f}")
    print(f"  OOS PD signal:  MCC {e['MCC']:.3f}  precision {e['precision']:.2f}"
          f"  recall {e['recall']:.2f}  F1 {f1:.2f}  flagged {e['volume']*100:.1f}%")

    lt = lead_time_table(df, P_STAR)
    det = lt[lt["status"] == "detected"]; leads = det["lead_days"].astype(float).values
    n_ev = int(panel["event"].sum())
    print(f"\nLead time (final sustained alarm run, p*={P_STAR:.2f}):")
    print(f"  detected {len(det)}/{n_ev}   missed {int((lt['status']=='missed').sum())}")
    if len(leads):
        print(f"  median {np.median(leads):.0f} d ({np.median(leads)/DAYS_PER_MONTH:.1f} mo)"
              f"  mean {leads.mean():.0f} d")
    return dict(label=label, df=df, meta=meta, te=te, lt=lt, f1=f1,
                auc_oos=meta["pd_auc_oos"], eval=e, leads=leads)


# ---------------------------------------------------- statistical tests -------
def bootstrap_auc_diff(y, p_lr, p_xgb, B=3000, seed=1):
    """paired bootstrap of AUC(XGB) - AUC(Logistic) on the same OOS rows."""
    y = np.asarray(y, int); p_lr = np.asarray(p_lr, float); p_xgb = np.asarray(p_xgb, float)
    rng = np.random.default_rng(seed); n = len(y); idx = np.arange(n)
    obs = roc_auc_score(y, p_xgb) - roc_auc_score(y, p_lr)
    diffs = []
    for _ in range(B):
        s = rng.choice(idx, n, replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        diffs.append(roc_auc_score(y[s], p_xgb[s]) - roc_auc_score(y[s], p_lr[s]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pval = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, lo, hi, float(pval)


def mcnemar_test(y, f_lr, f_xgb):
    y = np.asarray(y, int); c_lr = (np.asarray(f_lr, int) == y); c_xgb = (np.asarray(f_xgb, int) == y)
    b = int((c_lr & ~c_xgb).sum())          # logistic right, xgb wrong
    c = int((~c_lr & c_xgb).sum())          # logistic wrong, xgb right
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return b, c, float(sps.chi2.sf(stat, 1))


# ------------------------------------------------------------ main ------------
def main():
    print("=" * 78)
    print("APPROACH 1 HAZARD ESTIMATOR SHOWDOWN:  Logistic  vs  XGBoost")
    print("=" * 78)
    print("[Stage 1] Loading 33 features + dates + credit-event onsets ...")
    panel = survivor2.load_bond_dated()
    n_ev = int(panel["event"].sum())
    print(f"          {len(panel):,} firm-months | {panel['firm_id'].nunique()} firms | "
          f"{panel['month_year'].min().date()}..{panel['month_year'].max().date()} | events {n_ev}")
    # clean panel so only the 33 features can enter the hazard covariates
    clean = panel.drop(columns=survivor2.HAZARD_DROP, errors="ignore")

    R_lr = report_model("LOGISTIC", "discrete-time logistic hazard", clean, fit_hazard_logistic, is_xgb=False)
    R_xgb = report_model("XGBOOST", "gradient-boosted-trees hazard", clean, fit_hazard_xgb, is_xgb=True)

    # ---- head-to-head table -------------------------------------------------
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 78)
    el, ex = R_lr["eval"], R_xgb["eval"]
    rows = [
        ("PD_3M AUC (out-of-sample)", f"{R_lr['auc_oos']:.3f}", f"{R_xgb['auc_oos']:.3f}"),
        ("PD_3M AUC (in-sample)", f"{R_lr['meta']['pd_auc']:.3f}", f"{R_xgb['meta']['pd_auc']:.3f}"),
        ("MCC (OOS)", f"{el['MCC']:.3f}", f"{ex['MCC']:.3f}"),
        ("Precision (OOS)", f"{el['precision']:.2f}", f"{ex['precision']:.2f}"),
        ("Recall (OOS)", f"{el['recall']:.2f}", f"{ex['recall']:.2f}"),
        ("F1 (OOS)", f"{R_lr['f1']:.2f}", f"{R_xgb['f1']:.2f}"),
        ("Flagged volume (OOS)", f"{el['volume']*100:.1f}%", f"{ex['volume']*100:.1f}%"),
        ("Lead time median (mo)",
         f"{np.median(R_lr['leads'])/DAYS_PER_MONTH:.1f}" if len(R_lr['leads']) else "-",
         f"{np.median(R_xgb['leads'])/DAYS_PER_MONTH:.1f}" if len(R_xgb['leads']) else "-"),
        ("Detected before default",
         f"{(R_lr['lt']['status']=='detected').sum()}/{n_ev}",
         f"{(R_xgb['lt']['status']=='detected').sum()}/{n_ev}"),
    ]
    print(f"  {'Metric':30s} {'Logistic':>12s} {'XGBoost':>12s}")
    print("  " + "-" * 56)
    for m, a, b in rows:
        print(f"  {m:30s} {a:>12s} {b:>12s}")

    # ---- lead-time table side-by-side (defaulted firms) ---------------------
    a = R_lr["lt"][["firm_id", "default_date", "lead_days", "status"]].rename(
        columns={"lead_days": "lead_LR"})
    b = R_xgb["lt"][["firm_id", "lead_days"]].rename(columns={"lead_days": "lead_XGB"})
    lead_cmp = a.merge(b, on="firm_id")
    lead_cmp = lead_cmp[lead_cmp["status"].isin(["detected", "missed"])].sort_values("default_date")
    print("\nLead-time per defaulted firm (days):  Logistic vs XGBoost")
    print(f"  {'Firm':>6s} {'Default':>12s} {'lead_LR':>9s} {'lead_XGB':>9s}")
    print("  " + "-" * 40)
    for _, r in lead_cmp.iterrows():
        lr = "MISS" if pd.isna(r["lead_LR"]) else str(int(r["lead_LR"]))
        xg = "MISS" if pd.isna(r["lead_XGB"]) else str(int(r["lead_XGB"]))
        print(f"  {int(r['firm_id']):>6d} {str(r['default_date']):>12s} {lr:>9s} {xg:>9s}")

    # ---- statistical tests on the common OOS set ----------------------------
    print("\n" + "=" * 78)
    print("STATISTICAL TEST  (common out-of-sample rows)")
    print("=" * 78)
    tl, tx = R_lr["te"], R_xgb["te"]
    if tl is None or tx is None:
        print("  out-of-sample set unavailable -- cannot run the test."); return
    key = ["firm_id", "month_index"]
    m = (tl[key + ["y_fwd", "PD_3M", "flag_PD"]].rename(columns={"PD_3M": "p_lr", "flag_PD": "f_lr"})
         .merge(tx[key + ["PD_3M", "flag_PD"]].rename(columns={"PD_3M": "p_xgb", "flag_PD": "f_xgb"}), on=key))
    y = m["y_fwd"].astype(int).values
    print(f"  paired OOS rows: {len(m):,}   positives (event in 3m): {int(y.sum())}")

    obs, lo, hi, pval = bootstrap_auc_diff(y, m["p_lr"].values, m["p_xgb"].values)
    print(f"\n  [Bootstrap] AUC(XGBoost) - AUC(Logistic) = {obs:+.3f}"
          f"   95% CI [{lo:+.3f}, {hi:+.3f}]   p = {pval:.3f}")
    b, c, pmc = mcnemar_test(y, m["f_lr"].values, m["f_xgb"].values)
    print(f"  [McNemar]  logistic-only-correct = {b}, xgboost-only-correct = {c}, p = {pmc:.3f}")

    # ---- verdict ------------------------------------------------------------
    alpha = 0.05
    if pval < alpha:
        winner = "XGBoost" if obs > 0 else "Logistic"
        print(f"\n  VERDICT: {winner} has a STATISTICALLY significant higher OOS AUC "
              f"(p={pval:.3f} < {alpha}).")
    else:
        better = "XGBoost" if obs > 0 else "Logistic"
        print(f"\n  VERDICT: no statistically significant AUC difference (p={pval:.3f}). "
              f"{better} is nominally higher by {abs(obs):.3f}, but within noise -- "
              f"prefer the simpler/ more interpretable Logistic unless XGBoost wins on other criteria.")
    print("\nDone.")


if __name__ == "__main__":
    main()
