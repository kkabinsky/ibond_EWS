# -*- coding: utf-8 -*-
"""
build_ibond_33features.py
================================================================================
Merges iBond corporate bond panel data with the 33 real financial/market/ESG/macro
features from Rev01_Database_final.dta to create:

1. `ibond_33features_panel` (Issuer-Month Level):
   - Includes `issuer_code`, `issuer_name`, `bond_symbols_active` (list of active bond issues),
     and all 33 fundamental financial/market/ESG/macro features.

2. `ibond_issue_33features_panel` (Bond Issue-Month Level):
   - Expands to individual bond issues (`symbol`, `coupon`, `maturity_date`, `months_to_maturity`),
     enabling exact identification of which bond issues carry each set of 33 features.
"""
from __future__ import annotations
import os
import sqlite3
import pandas as pd
import numpy as np
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_ROOT, "cmdf_credit.db")
DTA_PATH = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"

# 33 real features
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

def build_ibond_33features(db_path=DB_PATH, dta_path=DTA_PATH, verbose=True):
    if verbose:
        print("=== [1/5] Loading iBond panel and bond universe from SQLite ===")
    conn = sqlite3.connect(db_path)
    b_panel = pd.read_sql_query("SELECT * FROM bond_ews_panel", conn)
    b_univ = pd.read_sql_query("SELECT * FROM bond_ews_universe", conn)
    
    b_panel["clean_id"] = b_panel["issuer_code"].astype(str).str.strip()
    
    # Drop existing feature columns to prevent _x / _y collisions
    drop_cols = [c for c in BOND_FEATURES if c in b_panel.columns]
    if drop_cols:
        b_panel.drop(columns=drop_cols, inplace=True)

    if verbose:
        print(f"      iBond panel: {len(b_panel):,} rows, {b_panel['clean_id'].nunique()} issuers")

    if verbose:
        print("=== [2/5] Loading 33 features from Stata database ===")
    dta_cols = list(dict.fromkeys(["firm_id", "month_year"] + BOND_FEATURES))
    dta = pd.read_stata(dta_path, columns=dta_cols)
    
    # Clean ticker ID (remove .BK and m.BK suffixes)
    dta["clean_id"] = dta["firm_id"].astype(str).str.replace("m.BK", "", regex=False).str.replace(".BK", "", regex=False).str.strip()
    dta["month"] = pd.to_datetime(dta["month_year"]).dt.strftime("%Y-%m")
    
    # Drop rows without financial features before deduplication
    dta_clean = dta.dropna(subset=["ROA"]).drop_duplicates(subset=["clean_id", "month"]).copy()

    if verbose:
        print("=== [3/5] Merging iBond panel with 33 features ===")
    merged = pd.merge(
        b_panel,
        dta_clean[["clean_id", "month"] + BOND_FEATURES],
        on=["clean_id", "month"],
        how="left"
    )

    # Impute missing feature values per issuer / month median / overall median fallback
    for feat in BOND_FEATURES:
        if feat in merged.columns:
            s_firm = merged.groupby("clean_id")[feat].ffill().bfill()
            merged[feat] = s_firm
            s_month = merged.groupby("month")[feat].transform(lambda x: x.fillna(x.median()))
            merged[feat] = s_month
            merged[feat] = merged[feat].fillna(merged[feat].median())

    # Map active bond symbols per issuer
    issuer_bonds = b_univ.groupby("issuer_code")["symbol"].apply(lambda s: ", ".join(s.unique())).to_dict()
    issuer_names = b_univ.groupby("issuer_code")["issuer_name"].first().to_dict()
    
    merged["issuer_name"] = merged["issuer_code"].map(issuer_names).fillna(merged["issuer_code"])
    merged["bond_symbols_active"] = merged["issuer_code"].map(issuer_bonds).fillna("-")

    if verbose:
        non_null_count = int(merged[BOND_FEATURES].notna().sum().sum())
        total_cells = len(merged) * len(BOND_FEATURES)
        print(f"      Enriched issuer master panel: {len(merged):,} rows, {len(merged.columns)} total features")
        print(f"      33 features presence: {non_null_count:,} / {total_cells:,} ({non_null_count/total_cells*100:.1f}% complete)")

    if "clean_id" in merged.columns:
        merged.drop(columns=["clean_id"], inplace=True)

    if verbose:
        print("=== [4/5] Building Bond Issue-Month Panel (`ibond_issue_33features_panel`) ===")
    
    # Expand to individual bond issue level
    univ_cols = ["symbol", "issuer_code", "rating", "registration_date", "maturity_date", "coupon"]
    issue_merged = pd.merge(
        b_univ[univ_cols],
        merged,
        on=["issuer_code"],
        how="inner"
    )
    
    # Calculate months to maturity for each issue in each month
    issue_merged["panel_date"] = pd.to_datetime(issue_merged["month"] + "-01")
    issue_merged["mat_date"] = pd.to_datetime(issue_merged["maturity_date"], errors="coerce")
    issue_merged["months_to_maturity"] = (
        (issue_merged["mat_date"].dt.year - issue_merged["panel_date"].dt.year) * 12 +
        (issue_merged["mat_date"].dt.month - issue_merged["panel_date"].dt.month)
    )
    issue_merged.drop(columns=["panel_date", "mat_date"], inplace=True)

    if verbose:
        print(f"      Enriched bond issue panel: {len(issue_merged):,} rows across {issue_merged['symbol'].nunique():,} bond issues")

    if verbose:
        print("=== [5/5] Saving master tables into SQLite cmdf_credit.db ===")
    merged.to_sql("ibond_33features_panel", conn, if_exists="replace", index=False)
    issue_merged.to_sql("ibond_issue_33features_panel", conn, if_exists="replace", index=False)
    conn.close()

    if verbose:
        print("SUCCESS! Saved `ibond_33features_panel` and `ibond_issue_33features_panel` to SQLite cmdf_credit.db")

    return merged, issue_merged

if __name__ == "__main__":
    df_issuer, df_issue = build_ibond_33features(verbose=True)
