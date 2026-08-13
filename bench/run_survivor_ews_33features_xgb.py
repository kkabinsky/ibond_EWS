# -*- coding: utf-8 -*-
"""
run_survivor_ews_33features_xgb.py
================================================================================
Executes Approach 2 (Calibrated XGBoost Hazard Engine) on the full 33-feature iBond panel
dataset (`ibond_33features_panel`).

Calculates Actionable 1-3M Lead Time and separate Persistent Alarm Duration for
all true defaulted corporate bond issuers.
Outputs smooth calibrated PD_3M probabilities so Green Dots (248 issuers) and Red Dots (41 issuers)
render with high contrast and zero text overflow on the Hyperbola plot.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

import lead_metrics
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_ROOT, "cmdf_credit.db")
DTA_PATH = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"

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

ALARM_PD = 0.05      # PD_3M at or above this counts as an alarm


def run_33features_xgb_ews(db_path=DB_PATH, verbose=True):
    if verbose:
        print("=== [1/5] Loading 33-Feature iBond Panel Dataset for Approach 2 XGBoost ===")
    conn = sqlite3.connect(db_path)
    
    try:
        panel = pd.read_sql_query("SELECT * FROM ibond_33features_panel", conn)
    except Exception:
        import build_ibond_33features as b33
        panel = b33.build_ibond_33features(verbose=False)

    try:
        df_def = pd.read_sql_query("SELECT * FROM ibond_default_payment", conn)
    except Exception:
        df_def = pd.DataFrame()

    avail_cols = [c for c in BOND_33_FEATURES if c in panel.columns]
    
    X = panel[avail_cols].copy()
    for c in avail_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        if pd.isna(med): med = 0.0
        X[c] = X[c].fillna(med)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_def["issuer_code"] = df_def["symbol"].astype(str).str.extract(r"^([A-Z]+)")[0]
    def_issuers = set(df_def["issuer_code"].unique()) if not df_def.empty else set()

    # ---- TARGET -------------------------------------------------------------
    # The old target was
    #     y = issuer_code.isin(def_issuers) | (ROA < -8.0) | (DE > 5.5)
    # which is wrong twice over:
    #   1. `isin(def_issuers)` labels EVERY month of a defaulted issuer as positive,
    #      including months years before the event. The model then learns "this is
    #      one of the eight names", not "a default is approaching" -- pure label
    #      leakage, and it makes the alarm fire from the issuer's first month, which
    #      is what produced the multi-thousand-day lead times.
    #   2. ROA and DE are themselves model features, so the rest of the rule is the
    #      model re-learning its own threshold.
    # The target must be the real event: default within the next 3 months.
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    y = pd.Series(0, index=panel.index, dtype=int)
    if not df_def.empty:
        dd = df_def.copy()
        dd["payment_date"] = pd.to_datetime(dd["payment_date"], errors="coerce")
        first_def = (dd.dropna(subset=["payment_date"])
                     .groupby("issuer_code")["payment_date"].min())
        ev = panel["issuer_code"].map(first_def)
        gap_m = (ev.dt.year - panel["month_dt"].dt.year) * 12 + \
                (ev.dt.month - panel["month_dt"].dt.month)
        y = ((gap_m >= 0) & (gap_m <= 3)).fillna(False).astype(int)
        panel["event_date"] = ev
    if y.sum() < 5:
        raise RuntimeError(f"only {int(y.sum())} default-linked months found — run "
                           "download_bond.py --defaults first.")
    if verbose:
        print(f"    target: default within 3 months -> {int(y.sum())} positive months "
              f"({y.mean()*100:.2f}% of rows)")


    if verbose:
        print(f"=== [2/5] Training Calibrated XGBoost Classifier Hazard Engine on All 33 Features ({len(avail_cols)} features) ===")
    base_model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.03,
        eval_metric="logloss",
        random_state=42
    )
    
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_scaled, y)
    
    pd_3m = calibrated_model.predict_proba(X_scaled)[:, 1]
    panel["PD_3M"] = pd_3m
    
    panel["month_dt"] = pd.to_datetime(panel["month"])
    panel = panel.sort_values(["issuer_code", "month_dt"]).reset_index(drop=True)
    
    panel["PD_prev"] = panel.groupby("issuer_code")["PD_3M"].shift(1).fillna(panel["PD_3M"])
    
    panel["PD_lag_med"] = (
        panel.groupby("issuer_code")["PD_3M"]
        .transform(lambda s: s.shift(1).rolling(12, min_periods=1).median())
    )
    panel["Momentum"] = (panel["PD_3M"] / (panel["PD_lag_med"] + 1e-4)).clip(0.0, 10.0)
    
    score = np.log(np.clip(panel["Momentum"], 1e-9, None)) + 0.55 * np.log(np.clip(panel["PD_prev"], 1e-9, None))
    panel["flag_hyper"] = (score >= np.log(0.35)).astype(int)
    
    panel["alert_level"] = "OK"
    panel.loc[panel["PD_3M"] >= 0.05, "alert_level"] = "WATCH"
    panel.loc[panel["PD_3M"] >= 0.15, "alert_level"] = "ELEVATED"
    panel.loc[(panel["flag_hyper"] == 1) | ((panel["PD_3M"] >= 0.15) & (panel["Momentum"] >= 1.2)), "alert_level"] = "HIGH RISK"

    if verbose:
        print("=== [3/5] Computing REAL Empirical Lead-Times for True Defaulted Issuers (XGBoost) ===")
        
    leadtime_rows = []
    if not df_def.empty:
        def_issuers_grp = df_def.groupby("issuer_code")["payment_date"].min().reset_index()
        
        for _, row in def_issuers_grp.iterrows():
            icode = row["issuer_code"]
            def_date_str = str(row["payment_date"])[:10]
            def_dt = pd.to_datetime(def_date_str, errors="coerce")
            if pd.isna(def_dt): continue
            
            sub_a = panel[panel["issuer_code"] == icode].sort_values("month_dt")
            if sub_a.empty: continue
            
            pre = sub_a[sub_a["month_dt"] < def_dt]
            metrics = lead_metrics.compute_lead_metrics(
                sub_a,
                event_date=def_dt,
                date_col="month_dt",
                alarm_mask=(
                    (sub_a["PD_3M"] >= ALARM_PD)
                    | (sub_a["flag_hyper"] == 1)
                ),
            )
            status, kind = lead_metrics.status_and_kind(metrics, has_event=True)
            selected_idx = (
                metrics.get("actionable_alarm_index")
                if metrics.get("actionable_alarm_index") is not None
                else metrics.get("persistent_alarm_start_index")
            )
            peak_row = (
                sub_a.loc[selected_idx]
                if selected_idx is not None and selected_idx in sub_a.index
                else (pre.iloc[-1] if not pre.empty else sub_a.iloc[0])
            )
            if status == "detected":
                alert = "HIGH RISK"
                verdict_row = f"XGBOOST ACTIONABLE 1-3M ALARM ({metrics['lead_months']:.1f}m lead)"
            elif kind == "earlier-only":
                alert = "EARLIER ALARM ONLY"
                verdict_row = "PERSISTENT ALARM OUTSIDE ACTIONABLE 1-3M WINDOW"
            else:
                alert = "MISSED"
                verdict_row = "NO ALARM BEFORE DEFAULT"
            leadtime_rows.append({
                "issuer_code": icode,
                "PD_3M": float(peak_row["PD_3M"]),
                "Momentum": float(peak_row["Momentum"]),
                "alert": alert,
                "status": status,
                "kind": kind,
                "verdict": verdict_row,
                **lead_metrics.strip_internal_fields(metrics),
            })
            
    # No synthetic "STARK" fallback: an invented issuer with a made-up 364-day lead
    # would be reported to the user as a real measurement.
    df_leadtime = pd.DataFrame(leadtime_rows)

    if verbose:
        print("=== [4/5] Evaluating XGBoost Lead-Time Metrics & ROC-AUC ===")
    auc_in = float(roc_auc_score(y, pd_3m)) if y.nunique() > 1 else float("nan")

    # Honest out-of-sample estimate: hold out one defaulted issuer at a time. The old
    # code set auc_oos = auc_in and called it validated.
    auc_oos = float("nan")
    oy, op = [], []
    for held in sorted(panel.loc[y == 1, "issuer_code"].dropna().unique()):
        tr = (panel["issuer_code"] != held).to_numpy()
        if y[tr].sum() < 2:
            continue
        try:
            b = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.03,
                                  eval_metric="logloss", random_state=42)
            m_i = CalibratedClassifierCV(estimator=b, method="sigmoid", cv=3)
            m_i.fit(X_scaled[tr], y[tr])
            oy.append(y[~tr].to_numpy())
            op.append(m_i.predict_proba(X_scaled[~tr])[:, 1])
        except Exception:
            continue
    if oy:
        yy, pp = np.concatenate(oy), np.concatenate(op)
        if 0 < yy.sum() < len(yy):
            auc_oos = float(roc_auc_score(yy, pp))

    n_high_risk = int((panel["alert_level"] == "HIGH RISK").sum())
    caught = int(df_leadtime["lead_days"].notna().sum()) if not df_leadtime.empty else 0
    med_days = (float(df_leadtime["lead_days"].median()) if caught else float("nan"))
    mean_days = (float(df_leadtime["lead_days"].mean()) if caught else float("nan"))
    n_def = len(df_leadtime)

    flag_pct = n_high_risk / max(len(panel), 1) * 100
    if np.isnan(auc_oos):
        verdict_x = "NOT VALIDATED - no out-of-sample estimate could be produced"
    elif auc_oos < 0.55:
        verdict_x = (f"NO PREDICTIVE SKILL - out-of-sample AUC {auc_oos:.3f} is at or "
                     f"below chance; in-sample {auc_in:.3f} is overfitting")
    elif auc_oos < 0.65 or flag_pct > 20:
        verdict_x = f"WEAK - out-of-sample AUC {auc_oos:.3f}, {flag_pct:.0f}% flagged"
    else:
        verdict_x = f"USABLE - out-of-sample AUC {auc_oos:.3f}"

    summary_df = pd.DataFrame([{
        "n_issuers": int(panel["issuer_code"].nunique()),
        "n_issuer_months": int(len(panel)),
        "n_defaulted_issuers": n_def,
        "n_positive_months": int(y.sum()),
        "n_events": n_def,
        "auc_in": auc_in,
        "auc_oos": auc_oos,
        "verdict": verdict_x,
        "n_high_risk": n_high_risk,
        "n_high": n_high_risk,
        **lead_metrics.summarize_lead_table(df_leadtime),
    }])

    # Approach 2 writes only its own *_xgb_* tables.
    panel.to_sql("bond_ews_xgb_alert_33", conn, if_exists="replace", index=False)
    df_leadtime.to_sql("bond_ews_xgb_leadtime_33", conn, if_exists="replace", index=False)
    summary_df.to_sql("bond_ews_xgb_summary_33", conn, if_exists="replace", index=False)

    import build_ibond_33features_latest as b33l
    b33l.build_ibond_33features_latest(db_path=db_path, dta_path=DTA_PATH, verbose=False)
    
    conn.close()
    if verbose:
        print(f"=== [5/5] Done. {n_def} defaulted issuers, actionable 1-3M "
              f"caught {caught}, median "
              f"{med_days:.0f} days ===")
        print(f"    AUC in-sample {auc_in:.4f} | out-of-sample {auc_oos:.4f}")
        print(f"    {verdict_x}")
        if not df_leadtime.empty:
            cols = [c for c in ("issuer_code","first_alarm_date","default_date",
                                "lead_days","lead_months","PD_3M","alert")
                    if c in df_leadtime.columns]
            print()
            print(df_leadtime.sort_values("lead_days")[cols].to_string(index=False))
    return panel, summary_df

if __name__ == "__main__":
    run_33features_xgb_ews(verbose=True)
