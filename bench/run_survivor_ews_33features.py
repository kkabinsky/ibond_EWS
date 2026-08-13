# -*- coding: utf-8 -*-
"""
run_survivor_ews_33features.py
================================================================================
Executes the Survivor-2 EWS Engine on the full 33-feature iBond panel
dataset (`ibond_33features_panel`).

Calculates Actionable 1-3M Lead Time and separate Persistent Alarm Duration for
all true defaulted corporate bond issuers.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def run_33features_ews(db_path=DB_PATH, verbose=True):
    if verbose:
        print("=== [1/5] Loading 33-Feature iBond Panel Dataset from SQLite ===")
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
    
    # ---- TARGET -------------------------------------------------------------
    # The old fallback was  y = (ROA < -1.5) | (DE > 2.5).  That is not default:
    # it is a hand-made distress rule built from TWO OF THE MODEL'S OWN FEATURES,
    # so the model simply re-learned its own threshold. It fired on 3,662 of 16,686
    # rows (22%) and produced a meaningless AUC of 0.99. The target must be the real
    # payment-default event: does this issuer default within the next 3 months?
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    y = pd.Series(0, index=panel.index, dtype=int)
    if not df_def.empty:
        dd = df_def.copy()
        dd["issuer_code"] = dd["symbol"].astype(str).str.extract(r"^([A-Z]+)")[0]
        dd["payment_date"] = pd.to_datetime(dd["payment_date"], errors="coerce")
        first_def = dd.dropna(subset=["payment_date"]).groupby("issuer_code")["payment_date"].min()
        ev = panel["issuer_code"].map(first_def)
        gap_m = (ev.dt.year - panel["month_dt"].dt.year) * 12 + \
                (ev.dt.month - panel["month_dt"].dt.month)
        y = ((gap_m >= 0) & (gap_m <= 3)).fillna(False).astype(int)
        panel["event_date"] = ev
    if y.sum() < 5:
        raise RuntimeError(
            f"only {int(y.sum())} default-linked months found — cannot fit a hazard "
            "model. Run download_bond.py --defaults first.")
    if verbose:
        print(f"    target: default within 3 months -> {int(y.sum())} positive months "
              f"({y.mean()*100:.2f}% of rows)")


    if verbose:
        print(f"=== [2/5] Fitting Hazard Engine on All 33 Features ({len(avail_cols)} features) ===")
    model = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y)
    
    pd_3m = model.predict_proba(X_scaled)[:, 1]
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
        print("=== [3/5] Computing REAL Empirical Lead-Times for True Defaulted Issuers ===")
        
    leadtime_rows = []
    if not df_def.empty:
        df_def["issuer_code"] = df_def["symbol"].astype(str).str.extract(r"^([A-Z]+)")[0]
        def_issuers = df_def.groupby("issuer_code")["payment_date"].min().reset_index()
        
        for _, row in def_issuers.iterrows():
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
                verdict_row = f"ACTIONABLE 1-3M ALARM ({metrics['lead_months']:.1f}m lead)"
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
            
    # No synthetic fallback row. The previous version invented a "STARK" issuer with
    # a made-up 364-day lead when nothing was found, which would be reported to the
    # user as a real result.
    df_leadtime = pd.DataFrame(leadtime_rows)

    if verbose:
        print("=== [4/5] Evaluating Lead-Time Metrics & ROC-AUC ===")
    auc_in = float(roc_auc_score(y, pd_3m)) if y.nunique() > 1 else float("nan")

    # Honest out-of-sample number. The old code reported auc_oos = auc_in and
    # labelled it "VALIDATED", which is an in-sample figure wearing an
    # out-of-sample label. Hold out one defaulted issuer at a time: its months are
    # never seen while fitting, so the score really is out-of-sample.
    auc_oos = float("nan")
    ev_codes = sorted(panel.loc[y == 1, "issuer_code"].dropna().unique())
    oy, op = [], []
    for held in ev_codes:
        tr = (panel["issuer_code"] != held).to_numpy()
        if y[tr].sum() < 2:
            continue
        sc_i = StandardScaler().fit(X_scaled[tr])
        m_i = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
        m_i.fit(sc_i.transform(X_scaled[tr]), y[tr])
        oy.append(y[~tr].to_numpy())
        op.append(m_i.predict_proba(sc_i.transform(X_scaled[~tr]))[:, 1])
    if oy:
        yy, pp = np.concatenate(oy), np.concatenate(op)
        if 0 < yy.sum() < len(yy):
            auc_oos = float(roc_auc_score(yy, pp))

    n_high_risk = int((panel["alert_level"] == "HIGH RISK").sum())
    n_iss = int(panel["issuer_code"].nunique())
    caught = int(df_leadtime["lead_days"].notna().sum()) if not df_leadtime.empty else 0
    med_lead = (float(df_leadtime["lead_days"].median())
                if caught else float("nan"))

    if np.isnan(auc_oos):
        verdict = "NOT VALIDATED - no out-of-sample estimate could be produced"
    elif auc_oos < 0.55:
        verdict = (f"NO PREDICTIVE SKILL - out-of-sample AUC {auc_oos:.3f} is at or "
                   f"below chance; in-sample {auc_in:.3f} is overfitting")
    elif auc_oos < 0.65 or (n_high_risk / max(len(panel), 1)) > 0.20:
        verdict = (f"WEAK - out-of-sample AUC {auc_oos:.3f}, "
                   f"{n_high_risk / max(len(panel), 1) * 100:.0f}% of rows flagged")
    else:
        verdict = f"USABLE - out-of-sample AUC {auc_oos:.3f}"

    summary_df = pd.DataFrame([{
        "n_issuers": n_iss,
        "n_issuer_months": int(len(panel)),
        "n_defaulted_issuers": int(len(df_leadtime)),
        "n_events": int(len(df_leadtime)),
        "n_positive_months": int(y.sum()),
        "auc_in": auc_in,
        "auc_oos": auc_oos,
        "verdict": verdict,
        "n_high_risk": n_high_risk,
        **lead_metrics.summarize_lead_table(df_leadtime),
    }])
    
    # Save results to SQLite
    # Write ONLY the *_33 tables. Writing the unsuffixed names clobbered
    # bond_ews.py's calibrated output, so the GUI showed this module's numbers
    # under the other engine's label.
    panel.to_sql("bond_ews_alert_33", conn, if_exists="replace", index=False)
    df_leadtime.to_sql("bond_ews_leadtime_33", conn, if_exists="replace", index=False)
    summary_df.to_sql("bond_ews_summary_33", conn, if_exists="replace", index=False)
    
    # Rebuild view
    import build_ibond_33features_latest as b33l
    b33l.build_ibond_33features_latest(db_path=db_path, dta_path=DTA_PATH, verbose=False)
    
    conn.close()
    if verbose:
        print(f"=== [5/5] Done. {len(df_leadtime)} defaulted issuers, actionable "
              f"1-3M caught {caught}, median {med_lead:.0f} days ===")
        print(f"    AUC in-sample {auc_in:.4f} | out-of-sample {auc_oos:.4f}")
        print(f"    {verdict}")
        if not df_leadtime.empty:
            cols = [c for c in ("issuer_code", "first_alarm_date", "default_date",
                                "lead_days", "lead_months", "PD_3M", "alert")
                    if c in df_leadtime.columns]
            print("\n" + df_leadtime.sort_values("lead_days")[cols]
                  .to_string(index=False))
    return panel, summary_df

if __name__ == "__main__":
    run_33features_ews(verbose=True)
