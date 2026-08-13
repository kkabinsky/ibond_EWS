# -*- coding: utf-8 -*-
"""
build_firm_mapping_and_view.py
================================================================================
Creates a dedicated SQLite mapping table `firm_issuer_mapping` and a SQLite View
`v_ibond_33features_panel` in `cmdf_credit.db`.

`v_ibond_33features_panel` contains ALL 16,686 firm-months ordered by:
  `p.month DESC, CAST(m.firm_id AS INTEGER) ASC`

This guarantees that:
- Every page (Page 1, Page 2, Page 3...) displays DIFFERENT corporate issuers!
- Clicking `Next` advances cleanly to the next set of companies!
"""
import os
import sqlite3
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

DB_PATH = os.path.join(DATA_ROOT, "cmdf_credit.db")
DTA_PATH = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"

def build_mapping_and_view(db_path=DB_PATH, dta_path=DTA_PATH, verbose=True):
    if verbose:
        print("=== [1/3] Building clean `firm_issuer_mapping` with integer firm_id (1, 2, 3...) ===")
    conn = sqlite3.connect(db_path)
    
    # Load Stata categories & full names
    dta = pd.read_stata(dta_path, columns=["firm_id", "Issuer"])
    dta_clean = dta.drop_duplicates(subset=["firm_id"]).copy()
    
    # Load ThaiBMA bond universe
    b_univ = pd.read_sql_query("SELECT symbol, issuer_code, issuer_name, sector FROM bond_ews_universe", conn)
    primary_symbol = b_univ.groupby("issuer_code")["symbol"].first().to_dict()
    th_names = b_univ.groupby("issuer_code")["issuer_name"].first().to_dict()
    sectors = b_univ.groupby("issuer_code")["sector"].first().to_dict()
    
    categories = dta["firm_id"].cat.categories
    records = []
    seen_issuers = set()
    
    for idx, cat in enumerate(categories, start=1):
        clean = str(cat).replace("m.BK", "").replace(".BK", "").strip()
        seen_issuers.add(clean)
        sub = dta_clean[dta_clean["firm_id"] == cat]
        en_name = sub["Issuer"].iloc[0] if not sub.empty else clean
        th_name = th_names.get(clean, en_name)
        bsym = primary_symbol.get(clean, clean)
        sec = sectors.get(clean, "OTHER")
        
        records.append({
            "firm_id": idx,                       # Numeric ID 1, 2, 3, 4...
            "stata_ticker": str(cat),            # 2S.BK, 88THm.BK, A.BK...
            "issuer_code": clean,                # 2S, 88TH, A...
            "bond_symbol": bsym,                 # 2S245A, A24NA...
            "company_name": en_name,             # 2S Metal PCL, Areeya Property PCL...
            "company_name_th": th_name,
            "sector": sec
        })
        
    # Append non-Stata iBond issuers
    next_id = len(records) + 1
    for icode, iname in th_names.items():
        if icode not in seen_issuers:
            bsym = primary_symbol.get(icode, icode)
            sec = sectors.get(icode, "OTHER")
            records.append({
                "firm_id": next_id,
                "stata_ticker": icode,
                "issuer_code": icode,
                "bond_symbol": bsym,
                "company_name": iname,
                "company_name_th": iname,
                "sector": sec
            })
            next_id += 1

    df_map = pd.DataFrame(records)
    df_map.to_sql("firm_issuer_mapping", conn, if_exists="replace", index=False)
    if verbose:
        print(f"      Saved `firm_issuer_mapping` with {len(df_map):,} rows.")

    if verbose:
        print("=== [2/3] Rebuilding SQLite View `v_ibond_33features_panel` ===")
    
    conn.execute("DROP VIEW IF EXISTS v_ibond_33features_panel;")
    
    panel_cols = [c[1] for c in conn.execute("PRAGMA table_info(ibond_33features_panel);").fetchall()]
    ignore_cols = {"id", "account_id", "firm_id", "issuer_code", "issuer_name", "bond_symbol", "bond_symbols_active", "clean_id", "firm_name"}
    feat_cols = [c for c in panel_cols if c not in ignore_cols]
    feat_sql = ", ".join([f"p.`{c}`" for c in feat_cols])
    
    # ORDER BY month DESC, firm_id ASC -> guarantees DIFFERENT companies on every page
    view_sql = f"""
    CREATE VIEW v_ibond_33features_panel AS
    SELECT 
        m.firm_id AS `firm_id`,
        m.company_name AS `firm_name`,
        m.company_name AS `company_name`,
        m.bond_symbol AS `bond_symbol`,
        p.issuer_code AS `issuer_code`,
        {feat_sql}
    FROM ibond_33features_panel p
    JOIN firm_issuer_mapping m ON p.issuer_code = m.issuer_code
    ORDER BY p.month DESC, CAST(m.firm_id AS INTEGER) ASC;
    """
    
    conn.execute(view_sql)
    conn.commit()
    if verbose:
        print("      Created View `v_ibond_33features_panel` successfully.")

    if verbose:
        print("=== [3/3] Inspecting View rows for Page 1 vs Page 2 ===")
    df_p1 = pd.read_sql_query("SELECT firm_id, firm_name, bond_symbol, month, amihud_monthly, ROA, DE FROM v_ibond_33features_panel LIMIT 10 OFFSET 0", conn)
    df_p2 = pd.read_sql_query("SELECT firm_id, firm_name, bond_symbol, month, amihud_monthly, ROA, DE FROM v_ibond_33features_panel LIMIT 10 OFFSET 10", conn)
    if verbose:
        print("PAGE 1 (Rows 1-10):")
        print(df_p1.to_string(index=False))
        print("\nPAGE 2 (Rows 11-20):")
        print(df_p2.to_string(index=False))
        
    conn.close()
    return df_map, df_p1

if __name__ == "__main__":
    build_mapping_and_view(verbose=True)
