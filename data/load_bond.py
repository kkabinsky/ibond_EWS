# -*- coding: utf-8 -*-
"""
Load 33 real features from the ThaiBMA bond database (Rev01_Database_final.dta).

Challenging EWS setup: the Merton risk columns (dd_12m, pd_12m, a_12m, sa_12m)
are NOT used as features — instead the 12-month **distance-to-default (dd_12m)**
defines the target ("high credit risk" = the low-DD tail). The model must then
predict that market-based risk assessment purely from fundamentals, liquidity,
governance/ESG and macro variables.
"""
from __future__ import annotations
import os
import pandas as pd

BOND_DTA = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"

# 33 real features — NO Merton internals (dd/pd/a/sa are held out for the target)
BOND_FEATURES = [
    # liquidity (8)
    "amihud_monthly", "amihud_monthly_100", "adj_illiq_kz", "scaled_amihud",
    "ln_amihud", "percent_zero_days", "zero_days", "n_days",
    # financial ratios (14)
    "ROA", "ROE", "DE", "CurrentRatio", "QuickRatio", "CashRatio",
    "EBITtoTA", "REtoTA", "WorkingCapitaltoTA", "TDTA", "LTDtoTA", "STDtoTA",
    "cf_Interestcoverageratio", "acc_DebtServiceCoverageRatio",
    # size / age (2)
    "lnTotalAssets", "lnAge",
    # macro (3)
    "Policyrate", "GDPgrowth", "UnemploymentratemodeledILOe",
    # governance / ESG (6)
    "ESGScore", "GovernancePillarScore", "EnvironmentalPillarScore",
    "SocialPillarScore", "IndependentBoardMembers", "AverageBoardTenure",
]
assert len(BOND_FEATURES) == 33

BOND_ID = "firm_id"
HORIZON = 3                   # forward window (months) for the early-warning label

# Real credit-event columns (each is 1 on AND AFTER the event month):
#   d_Default_Payment  — payment default
#   d_Restructure      — debt restructuring
#   d_DP_RS            — either / both (default and/or restructure)  <-- the target source
EVENT_SOURCE = "d_DP_RS"

# Merton columns: NOT features — kept only so the trajectory panel can plot the
# real Distance-to-Default decay and PD alongside the credit-event marker.
DISPLAY_COLS = ["dd_12m", "pd_12m"]


def load_bond(path=BOND_DTA, horizon=HORIZON):
    """Target = ONSET of a real credit event (restructure and/or default).

    The raw flag is persistent (1 on and after the event), so we take the first
    month it turns 1 as the event onset, censor every month after it (the firm
    has already failed), and label the `horizon` months *before* onset as the
    early-warning positives. dd_12m is held out of the features to avoid leakage.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cols = list(dict.fromkeys(
        BOND_FEATURES + DISPLAY_COLS
        + [BOND_ID, "month_year", EVENT_SOURCE, "d_Default_Payment", "d_Restructure"]))
    df = pd.read_stata(path, columns=cols, convert_categoricals=False)
    df = df.dropna(subset=["dd_12m"]).copy()             # keep rows we can model
    df["firm_id"] = pd.to_numeric(df[BOND_ID], errors="coerce")
    df = df.sort_values(["firm_id", "month_year"]).reset_index(drop=True)

    # --- credit-event onset + censoring ------------------------------------
    flag = (pd.to_numeric(df[EVENT_SOURCE], errors="coerce") > 0).astype(int)
    cum = flag.groupby(df["firm_id"]).cumsum()
    df["event"] = ((cum == 1) & (flag == 1)).astype(int)  # 1 only at the onset month
    df = df[cum <= 1].reset_index(drop=True)              # drop months after onset (already failed)
    df["default_event"] = df["event"]                     # trajectory X-marker = onset month

    # --- forward early-warning label: onset within the next calendar months
    # Using the next N rows is wrong for sparse firms because three observations
    # can span years. The month ordinal keeps HORIZON tied to calendar time.
    dates = pd.to_datetime(df["month_year"], errors="coerce")
    month_ord = dates.dt.year * 12 + dates.dt.month
    event_ord = month_ord.where(df["event"] == 1).groupby(df["firm_id"]).transform("min")
    months_ahead = event_ord - month_ord
    df["default_3m"] = ((months_ahead >= 1) & (months_ahead <= horizon)).astype(int)

    df["account_id"] = df["firm_id"]
    df["month_index"] = pd.factorize(df["month_year"], sort=True)[0] + 1   # chronological
    keep = (["account_id", "firm_id", "month_year", "month_index"] + BOND_FEATURES
            + DISPLAY_COLS + ["default_3m", "event", "default_event"])
    return df[keep]


if __name__ == "__main__":
    df = load_bond()
    print(f"rows {len(df):,}  features {len(BOND_FEATURES)} (no Merton)  firms {df['firm_id'].nunique()}")
    print(f"credit-event onsets (restructure and/or default): {int(df['event'].sum())} firms")
    print(f"early-warning positives (event within next {HORIZON}m): {int(df['default_3m'].sum())} "
          f"({df['default_3m'].mean()*100:.2f}%)")
    print("target source:", EVENT_SOURCE, "| display-only (not features):", DISPLAY_COLS)
    print(df.iloc[:3, :7].to_string(index=False))
