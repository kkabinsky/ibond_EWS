"""Corporate Bond Early Warning System (Approach 2 — Machine Learning XGBoost Survival Hazard).

Fits a non-linear XGBoost discrete-time hazard model on the iBond corporate bond panel
(16,686 issuer-months, 3,366 bond issues, 46 payment default records across 10 defaulted issuers).
Applies Prior Correction (King & Zeng 2001) to calibrate logit probabilities to the true sample
base rate (0.00192).

Outputs:
  - SQLite tables: bond_ews_xgb_alert, bond_ews_xgb_leadtime, bond_ews_xgb_summary
  - Out-of-sample AUC, caught default count, lead times, and hyperbolic alarm boundaries
"""

import os, sqlite3, re
import numpy as np, pandas as pd
from scipy.special import expit, logit
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import download_bond as dbnd
import bond_ews as bews
import lead_metrics
from bond_ews import (
    DB, T_UNIVERSE, T_PANEL, T_SUMMARY,
    FEATURES, CURVE_FEATURES, K_BOUND, ALPHA_BOUND, ALARM_BUDGET
)

T_ALERT_XGB = "bond_ews_xgb_alert"
T_LEAD_XGB = "bond_ews_xgb_leadtime"
T_SUMMARY_XGB = "bond_ews_xgb_summary"


def fit_hazard_xgb(p: pd.DataFrame, verbose=True):
    """Fit non-linear XGBoost discrete-time hazard on issuer-month panel."""
    feats = [c for c in FEATURES if c in p.columns]
    X = p[feats].fillna(0.0).to_numpy(dtype=np.float32)
    y = p["y_fwd"].to_numpy(dtype=np.int32)
    base_rate = float(y.mean()) if len(y) > 0 else 0.00192

    # GroupKFold by issuer for out-of-sample evaluation
    issuers = p["issuer_code"].unique()
    np.random.seed(42)
    shuffled = np.random.permutation(issuers)
    folds = np.array_split(shuffled, 5)

    oof_logits = np.zeros(len(p), dtype=np.float64)
    for fold_idx, test_iss in enumerate(folds):
        test_mask = p["issuer_code"].isin(test_iss).to_numpy()
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]

        if y_tr.sum() == 0:
            continue

        pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
        model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="logloss"
        )
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_te)[:, 1]
        oof_logits[test_mask] = logit(np.clip(preds, 1e-7, 1 - 1e-7))

    # Prior Correction (King & Zeng 2001) logit intercept shift
    prior_offset = np.log((1.0 - base_rate) / max(base_rate, 1e-6))
    calibrated_logits = oof_logits - prior_offset
    h_cal = expit(calibrated_logits)

    auc_in = float(roc_auc_score(y, h_cal)) if len(np.unique(y)) > 1 else np.nan
    auc_oos = auc_in

    # Fit final model on full dataset
    pos_w = float((len(y) - y.sum()) / max(y.sum(), 1))
    full_model = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w,
        random_state=42, eval_metric="logloss"
    )
    full_model.fit(X, y)
    full_preds = full_model.predict_proba(X)[:, 1]
    full_cal_logits = logit(np.clip(full_preds, 1e-7, 1 - 1e-7)) - prior_offset
    h_full = expit(full_cal_logits)

    res = p.copy()
    res["h"] = h_full
    res.attrs["meta"] = {
        "auc_in": auc_in, "auc_oos": auc_oos,
        "n_pos": int(y.sum()), "base_rate": base_rate,
        "prior_offset": float(prior_offset), "model": full_model,
        "features": feats
    }
    return res, res.attrs["meta"]


def run_xgb(refresh=False, save=True, verbose=True):
    """Run Approach 2 XGBoost Corporate Bond EWS pipeline."""
    uni, panel, alerts, lt_old, bsum = bews.load_from_sqlite(DB)
    _, _, defaults, _ = dbnd.load_from_sqlite(DB)
    if panel.empty:
        raise RuntimeError("Panel is empty. Run download_bond.py / bond_ews.py first.")

    d, meta = fit_hazard_xgb(panel, verbose=verbose)
    d = bews.add_signals(d)
    if not isinstance(d["month"].dtype, pd.PeriodDtype):
        d["month"] = pd.to_datetime(d["month"].astype(str), errors="coerce").dt.to_period("M")
    d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce")

    latest = (d.sort_values("month").groupby("issuer_code").tail(1)
              .sort_values("PD_3M", ascending=False).reset_index(drop=True))
    thr = float(d.attrs.get("p_star", np.nan))
    latest["alert"] = bews._levels(latest, thr)
    lt = bews.lead_time(d, thr, verbose=verbose)

    counts = latest["alert"].value_counts()
    summary = {
        "n_issues": int(len(uni)), "n_issuers": int(panel["issuer_code"].nunique()),
        "n_issuer_months": int(len(panel)),
        "n_defaulted_issuers": int(panel["event_date"].notna().groupby(panel["issuer_code"]).any().sum()),
        "n_positive_months": int(meta["n_pos"]),
        "auc_in": meta["auc_in"], "auc_oos": meta["auc_oos"],
        "p_star": thr, "alarm_budget": ALARM_BUDGET,
        "base_rate": meta.get("base_rate", np.nan),
        "prior_offset": meta.get("prior_offset", np.nan), "K": K_BOUND, "alpha": ALPHA_BOUND,
        "n_high": int(counts.get("HIGH RISK", 0)),
        "n_elevated": int(counts.get("ELEVATED", 0)),
        "n_watch": int(counts.get("WATCH", 0)),
        "n_ok": int(counts.get("OK", 0)),
        "n_events": int(len(lt)),
        "flagged_pct": float(counts.get("HIGH RISK", 0) / max(len(latest), 1) * 100),
        "verdict": bews._verdict(meta["auc_oos"], counts.get("HIGH RISK", 0), len(latest)),
        **lead_metrics.summarize_lead_table(lt),
        "as_of": str(latest["month"].max()) if not latest.empty else "",
        "approach": "Approach 2 (XGBoost Hazard)"
    }

    if save:
        con = sqlite3.connect(DB)
        d_save = d.copy()
        d_save["month"] = d_save["month"].astype(str)
        d_save["event_date"] = d_save["event_date"].astype(str)
        d_save.to_sql(T_ALERT_XGB, con, if_exists="replace", index=False)
        lt.to_sql(T_LEAD_XGB, con, if_exists="replace", index=False)
        pd.DataFrame([summary]).to_sql(T_SUMMARY_XGB, con, if_exists="replace", index=False)
        con.close()

    print(f"DONE (Approach 2 XGBoost) - {summary['n_high']} HIGH RISK, caught {summary['n_caught']}/{summary['n_events']} defaulted issuers, AUC OOS: {summary['auc_oos']:.3f}")
    return uni, panel, d, lt, meta, summary


if __name__ == "__main__":
    run_xgb(refresh=False, save=True, verbose=True)
