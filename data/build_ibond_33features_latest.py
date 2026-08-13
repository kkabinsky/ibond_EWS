# -*- coding: utf-8 -*-
"""
build_ibond_33features_latest.py
================================================================================
Calculates the latest 33-feature corporate bond snapshot dataset for Approach 1 (Cox Hazard)
and Approach 2 (XGBoost Hazard) hyperbola plotting.

Saves:
1. SQLite Table: `ibond_33features_latest`
2. SQLite View:  `v_ibond_33features_latest`

View Columns (in exact DataGridView Inspector order):
- firm_id (Stata numeric ID: 3, 9, 13, 17, 35...)
- company_name (Real full company name)
- bond_symbol (Real primary bond symbol: A24NA, ACE26OA...)
- month (Latest month date)
- PD_3M (3-Month Forward Default Probability)
- Momentum (Risk Momentum M(t))
- flag_hyper (Hyperbolic Boundary alarm flag 0/1)
- alert (Risk Status Badge: HIGH RISK, ELEVATED, WATCH, OK)
- Followed by all 33 fundamental financial/market/ESG/macro features!
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

BOND_FEATURES = [
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

def build_ibond_33features_latest(db_path=DB_PATH, dta_path=DTA_PATH, verbose=True):
    if verbose:
        print("=== [1/4] Running Approach 1 Cox Hazard signals for 33 features ===")
    
    conn = sqlite3.connect(db_path)
    
    # Load bond_ews_alert table or compute signals if needed
    try:
        df_alert = pd.read_sql_query("SELECT * FROM bond_ews_alert", conn)
    except Exception:
        import bond_ews as bews
        bews.run(refresh=False, save=True, verbose=False)
        df_alert = pd.read_sql_query("SELECT * FROM bond_ews_alert", conn)
        
    # Ensure mapping table exists
    import build_firm_mapping_and_view as bmv
    bmv.build_mapping_and_view(db_path=db_path, dta_path=dta_path, verbose=False)

    # Get latest row per issuer
    df_latest = df_alert.sort_values("month").groupby("issuer_code").tail(1).copy()
    
    # Save table `ibond_33features_latest`
    df_latest.to_sql("ibond_33features_latest", conn, if_exists="replace", index=False)
    if verbose:
        print(f"      Saved `ibond_33features_latest` table with {len(df_latest):,} rows.")

    if verbose:
        print("=== [2/4] Building SQLite View `v_ibond_33features_latest` ===")
    
    conn.execute("DROP VIEW IF EXISTS v_ibond_33features_latest;")
    
    # Available 33 features
    p_cols = [c[1] for c in conn.execute("PRAGMA table_info(ibond_33features_panel);").fetchall()]
    ignore_cols = {"id", "account_id", "firm_id", "issuer_code", "issuer_name", "bond_symbol", "bond_symbols_active", "clean_id", "firm_name", "month"}
    feat_cols = [c for c in p_cols if c not in ignore_cols and c in BOND_FEATURES]
    feat_sql = ", ".join([f"p.`{c}`" for c in feat_cols]) if feat_cols else ""
    if feat_sql:
        feat_sql = ", " + feat_sql

    view_sql = f"""
    CREATE VIEW v_ibond_33features_latest AS
    SELECT 
        m.firm_id AS `firm_id`,
        m.company_name AS `company_name`,
        m.bond_symbol AS `bond_symbol`,
        a.month AS `month`,
        ROUND(a.PD_3M * 100, 2) || '%' AS `PD_3M`,
        ROUND(a.Momentum, 2) AS `Momentum`,
        a.flag_hyper AS `flag_hyper`,
        CASE 
            WHEN a.PD_3M >= 0.15 AND (a.flag_hyper = 1 OR a.Momentum > 1.0) THEN 'HIGH RISK'
            WHEN a.PD_3M >= 0.15 THEN 'ELEVATED'
            WHEN a.PD_3M >= 0.05 OR a.flag_hyper = 1 THEN 'WATCH'
            ELSE 'OK'
        END AS `alert`
        {feat_sql}
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY issuer_code ORDER BY month DESC) as rn
        FROM bond_ews_alert
    ) a
    JOIN firm_issuer_mapping m ON a.issuer_code = m.issuer_code
    LEFT JOIN (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY issuer_code ORDER BY month DESC) as rn2
        FROM ibond_33features_panel
    ) p ON a.issuer_code = p.issuer_code AND p.rn2 = 1
    WHERE a.rn = 1
    ORDER BY CAST(m.firm_id AS INTEGER) ASC;
    """
    
    conn.execute(view_sql)
    conn.commit()
    
    if verbose:
        print("      Created View `v_ibond_33features_latest` successfully.")

    if verbose:
        print("=== [3/4] Inspecting `v_ibond_33features_latest` View ===")
    df_v = pd.read_sql_query("SELECT firm_id, company_name, bond_symbol, month, PD_3M, Momentum, flag_hyper, alert, amihud_monthly, ROA, DE FROM v_ibond_33features_latest LIMIT 20", conn)
    if verbose:
        print(df_v.to_string(index=False))

    conn.close()
    return df_latest, df_v

if __name__ == "__main__":
    build_ibond_33features_latest(verbose=True)
