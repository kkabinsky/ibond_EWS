# -*- coding: utf-8 -*-
"""
Synthetic ThaiBMA credit-risk dataset (33 features) -> Excel.

One row per account (cross-sectional snapshot) with a 3-month default target,
so ThaiBMA can replace this file with the real Paynext/SET export that has the
same 33 columns and re-run the app unchanged.

Feature groups (33 total):
  behavioural (6) : delinquency_trend, credit_score_pd, loan_tenor,
                    utilization, behavioral_score, spend_activity
  financial  (10) : roa, roe, debt_to_equity, current_ratio, quick_ratio,
                    interest_coverage, leverage_ratio, profit_margin,
                    ebitda_margin, cash_to_assets
  market      (5) : equity_vol, distance_to_default, market_leverage,
                    momentum_12m, beta
  macro       (6) : yield_slope, credit_spread, policy_rate, gdp_growth,
                    inflation, unemployment
  governance  (4) : board_size, institutional_ownership, esg_score, governance_score
  loan        (2) : loan_amount, months_on_book
Target: default_3m (1 = defaults within 3 months).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    # behavioural
    "delinquency_trend", "credit_score_pd", "loan_tenor", "utilization",
    "behavioral_score", "spend_activity",
    # financial
    "roa", "roe", "debt_to_equity", "current_ratio", "quick_ratio",
    "interest_coverage", "leverage_ratio", "profit_margin", "ebitda_margin",
    "cash_to_assets",
    # market
    "equity_vol", "distance_to_default", "market_leverage", "momentum_12m", "beta",
    # macro
    "yield_slope", "credit_spread", "policy_rate", "gdp_growth", "inflation",
    "unemployment",
    # governance / ESG
    "board_size", "institutional_ownership", "esg_score", "governance_score",
    # loan
    "loan_amount", "months_on_book",
]
assert len(FEATURES) == 33

# sign of each feature's effect on default risk (+ raises PD, - lowers PD)
_RISK_W = {
    "delinquency_trend": 1.6, "credit_score_pd": 1.3, "utilization": 0.9,
    "debt_to_equity": 0.8, "leverage_ratio": 0.8, "equity_vol": 0.9,
    "credit_spread": 0.7, "market_leverage": 0.6, "beta": 0.4, "loan_tenor": 0.3,
    "inflation": 0.3, "unemployment": 0.4, "policy_rate": 0.3,
    "roa": -0.9, "roe": -0.7, "distance_to_default": -1.2, "current_ratio": -0.6,
    "quick_ratio": -0.5, "interest_coverage": -0.6, "profit_margin": -0.6,
    "ebitda_margin": -0.5, "cash_to_assets": -0.5, "spend_activity": -0.5,
    "behavioral_score": -0.7, "esg_score": -0.4, "governance_score": -0.4,
    "institutional_ownership": -0.3, "gdp_growth": -0.4, "yield_slope": -0.2,
    "credit_score_pd_dup": 0.0,
}

# plausible display scales (mean, sd) per feature so the Excel looks realistic
_SCALE = {
    "roa": (0.05, 0.06), "roe": (0.10, 0.12), "debt_to_equity": (1.2, 0.9),
    "current_ratio": (1.6, 0.6), "quick_ratio": (1.1, 0.5),
    "interest_coverage": (5.0, 3.5), "leverage_ratio": (0.45, 0.18),
    "profit_margin": (0.08, 0.09), "ebitda_margin": (0.18, 0.1),
    "cash_to_assets": (0.12, 0.08), "equity_vol": (0.30, 0.14),
    "distance_to_default": (4.0, 2.0), "market_leverage": (0.45, 0.2),
    "momentum_12m": (0.05, 0.25), "beta": (1.0, 0.4),
    "yield_slope": (0.8, 0.6), "credit_spread": (1.8, 0.8),
    "policy_rate": (2.5, 1.0), "gdp_growth": (2.8, 1.5), "inflation": (2.0, 1.2),
    "unemployment": (1.5, 0.6), "board_size": (9, 3),
    "institutional_ownership": (0.45, 0.2), "esg_score": (55, 18),
    "governance_score": (60, 18), "loan_amount": (500000, 350000),
    "months_on_book": (24, 14), "utilization": (0.5, 0.25),
    "credit_score_pd": (0.06, 0.05), "delinquency_trend": (0.0, 1.0),
    "behavioral_score": (0.0, 1.0), "spend_activity": (0.0, 1.0),
    "loan_tenor": (36, 18),
}


def generate(n_accounts=5000, default_rate=0.13, seed=11):
    rng = np.random.default_rng(seed)
    # latent standardized factors -> both features and risk
    Z = {f: rng.normal(0, 1, n_accounts) for f in FEATURES}
    # correlate a fragility factor into the risky features
    frag = rng.normal(0, 1, n_accounts)
    for f, w in _RISK_W.items():
        if f in Z:
            Z[f] = 0.6 * Z[f] + 0.5 * np.sign(w) * frag

    risk = np.zeros(n_accounts)
    for f, w in _RISK_W.items():
        if f in Z:
            risk += w * Z[f]
    risk += rng.normal(0, 1.2, n_accounts)                 # idiosyncratic noise
    thr = np.quantile(risk, 1 - default_rate)
    y = (risk >= thr).astype(int)

    # map standardized factors to realistic display scales
    data = {"account_id": np.arange(1, n_accounts + 1)}
    for f in FEATURES:
        mu, sd = _SCALE.get(f, (0.0, 1.0))
        col = mu + sd * Z[f]
        if f in ("board_size", "months_on_book", "loan_tenor"):
            col = np.clip(np.round(col), 1, None)
        if f in ("utilization", "leverage_ratio", "market_leverage",
                 "institutional_ownership", "credit_score_pd"):
            col = np.clip(col, 0, 1)
        if f in ("esg_score", "governance_score"):
            col = np.clip(col, 0, 100)
        data[f] = np.round(col, 4)
    data["default_3m"] = y
    return pd.DataFrame(data)


if __name__ == "__main__":
    import os
    df = generate()
    out = os.path.join(os.path.dirname(__file__), "credit_dataset_33features.xlsx")
    df.to_excel(out, index=False)
    print(f"wrote {out}")
    print(f"rows {len(df):,}  features {len(FEATURES)}  "
          f"default rate {df['default_3m'].mean()*100:.1f}%")
    print(df.iloc[:3, :8].to_string(index=False))
