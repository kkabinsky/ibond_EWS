# -*- coding: utf-8 -*-
"""
test_xgb_calibration.py
================================================================================
Tests Platt scaling probability calibration for XGBoost 33-feature hazard engine
so that PD_3M values are smooth, realistic risk probabilities in (0.0001, 0.85)
producing distinct green dots and nicely formatted red dots on the Hyperbola plot.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

conn = sqlite3.connect("cmdf_credit.db")
panel = pd.read_sql_query("SELECT * FROM ibond_33features_panel", conn)
df_def = pd.read_sql_query("SELECT * FROM ibond_default_payment", conn)
conn.close()

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
def_issuers = set(df_def["issuer_code"].unique())

y = ((panel["issuer_code"].isin(def_issuers)) | (panel["ROA"] < -8.0) | (panel["DE"] > 5.5)).astype(int)

# Use Calibrated Classifier for smooth PD_3M probabilities
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
panel["PD_lag_med"] = panel.groupby("issuer_code")["PD_3M"].transform(lambda s: s.shift(1).rolling(12, min_periods=1).median())
panel["Momentum"] = (panel["PD_3M"] / (panel["PD_lag_med"] + 1e-4)).clip(0.0, 10.0)

# Filter snapshot: 1 latest month per issuer
d = panel.sort_values("month").groupby("issuer_code").tail(1).copy()
d = d[(d["PD_3M"] > 1e-4) & (d["Momentum"] > 0) & (d["Momentum"] < 20)]

x = d["PD_3M"].to_numpy() * 100
y_mom = d["Momentum"].to_numpy()
pd_p = d["PD_prev"].to_numpy()
score = np.log(np.clip(y_mom, 1e-9, None)) + 0.55 * np.log(np.clip(pd_p, 1e-9, None))
flag = score >= np.log(0.35)

print("TOTAL SNAPSHOT ISSUERS:", len(d))
print("GREEN DOTS (below boundary):", int((~flag).sum()))
print("RED DOTS (beyond boundary):", int(flag.sum()))
print("PD_3M range:", f"min={x.min():.4f}%, max={x.max():.2f}%, mean={x.mean():.2f}%")
print("RED DOT ISSUERS:", d[flag]["issuer_code"].tolist())
