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

def _have(conn, name):
    return conn.execute(
        "SELECT count(1) FROM sqlite_master WHERE name=?", (name,)).fetchone()[0] > 0


def build_mapping_from_shipped_tables(conn, verbose=True):
    """Build `firm_issuer_mapping` from the tables this repository actually ships.

    WHY THIS PATH EXISTS
        The original builder needs two things that are not in the repository: a
        Stata file at a hard-coded D:\\ path, and the bond_ews_universe table. On
        any machine without both, `firm_issuer_mapping` was never created, so the
        v_ibond_33features_panel view was never created either. The GUI then lost
        its issuer catalogue and its lead-time tab, and three tests failed to
        collect -- all with "no such table", which reads like missing data rather
        than a missing build step.

        Everything the mapping needs is already in ibond_33features_panel and
        ibond_issuer, both of which ship as CSV in dataset/. Names come from the
        issuer register where it has them and from the panel otherwise; the bond
        symbol comes from the panel's own active-symbol column.

    The Stata path is still preferred when it is available, since it carries the
    firm_id ordering the earlier research used.
    """
    if verbose:
        print("      Stata source unavailable; building from ibond_33features_panel "
              "+ ibond_issuer")
    panel = pd.read_sql_query(
        "SELECT issuer_code, issuer_name, sector, bond_symbols_active "
        "FROM ibond_33features_panel", conn)
    panel["issuer_code"] = panel["issuer_code"].astype(str)

    first = (panel.sort_values("issuer_code")
             .groupby("issuer_code", as_index=False)
             .agg(issuer_name=("issuer_name", "first"),
                  sector=("sector", "first"),
                  bond_symbols_active=("bond_symbols_active", "first")))

    reg = pd.DataFrame(columns=["institution_code", "name_en", "name_th",
                                "sector_code"])
    if _have(conn, "ibond_issuer"):
        reg = pd.read_sql_query(
            "SELECT institution_code, name_en, name_th, sector_code "
            "FROM ibond_issuer", conn)
        reg = reg.drop_duplicates(subset=["institution_code"])

    m = first.merge(reg, how="left", left_on="issuer_code",
                    right_on="institution_code")

    def first_symbol(v, code):
        if isinstance(v, str) and v.strip():
            return v.split(",")[0].strip()
        return code

    records = []
    for i, r in enumerate(m.itertuples(), start=1):
        en = (r.name_en if isinstance(getattr(r, "name_en", None), str)
              and r.name_en.strip() else r.issuer_name)
        en = en if isinstance(en, str) and en.strip() else r.issuer_code
        th = (r.name_th if isinstance(getattr(r, "name_th", None), str)
              and r.name_th.strip() else en)
        sec = (r.sector_code if isinstance(getattr(r, "sector_code", None), str)
               and r.sector_code.strip() else
               (r.sector if isinstance(r.sector, str) and r.sector.strip()
                else "OTHER"))
        records.append({
            "firm_id": i,
            "stata_ticker": f"{r.issuer_code}.BK",
            "issuer_code": r.issuer_code,
            "bond_symbol": first_symbol(r.bond_symbols_active, r.issuer_code),
            "company_name": en,
            "company_name_th": th,
            "sector": sec,
        })
    df_map = pd.DataFrame(records)
    df_map.to_sql("firm_issuer_mapping", conn, if_exists="replace", index=False)
    if verbose:
        print(f"      Saved `firm_issuer_mapping` with {len(df_map):,} rows.")
    return df_map


def build_mapping_and_view(db_path=DB_PATH, dta_path=DTA_PATH, verbose=True):
    if verbose:
        print("=== [1/3] Building clean `firm_issuer_mapping` with integer firm_id (1, 2, 3...) ===")
    conn = sqlite3.connect(db_path)

    if not (dta_path and os.path.exists(dta_path)
            and _have(conn, "bond_ews_universe")):
        df_map = build_mapping_from_shipped_tables(conn, verbose=verbose)
        return _build_view(conn, df_map, verbose=verbose)

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

    return _build_view(conn, df_map, verbose=verbose)


def _build_view(conn, df_map, verbose=True):
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
