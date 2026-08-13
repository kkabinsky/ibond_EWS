# -*- coding: utf-8 -*-
"""
add_firm_name_to_tables.py
================================================================================
Adds the `firm_name` column (full company name in English/Thai) across all SQLite
tables in `cmdf_credit.db` by mapping from Stata `Rev01_Database_final.dta` and
`bond_ews_universe`.
"""
import sqlite3
import pandas as pd
import numpy as np
from thaibma_paths import DATA_ROOT  # data lives outside the repo

DTA_PATH = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"
DB_PATH = os.path.join(DATA_ROOT, "cmdf_credit.db")

print("=== [1/3] Building firm_name lookup map ===")
dta = pd.read_stata(DTA_PATH, columns=["firm_id", "Issuer"])
dta_clean = dta.drop_duplicates(subset=["firm_id"])

# Map numeric ID (1-indexed) -> Full Issuer Name
categories = dta["firm_id"].cat.categories
num_to_name = {}
ticker_to_name = {}

for idx, cat in enumerate(categories, start=1):
    sub = dta_clean[dta_clean["firm_id"] == cat]
    name = sub["Issuer"].iloc[0] if not sub.empty else cat
    num_to_name[idx] = name
    num_to_name[str(idx)] = name
    ticker_to_name[cat] = name
    clean = cat.replace("m.BK", "").replace(".BK", "").strip()
    ticker_to_name[clean] = name

conn = sqlite3.connect(DB_PATH)
try:
    b_univ = pd.read_sql_query("SELECT issuer_code, issuer_name FROM bond_ews_universe", conn)
    for _, r in b_univ.iterrows():
        if pd.notna(r["issuer_code"]) and pd.notna(r["issuer_name"]):
            ticker_to_name[str(r["issuer_code"]).strip()] = str(r["issuer_name"]).strip()
except Exception:
    pass

print(f"Loaded {len(ticker_to_name)} firm name mappings (e.g. 1 -> '{num_to_name.get(1)}', 'A' -> '{ticker_to_name.get('A')}')")

def get_name(val):
    if pd.isna(val):
        return "-"
    v_str = str(val).strip()
    if v_str in ticker_to_name:
        return ticker_to_name[v_str]
    v_clean = v_str.replace("m.BK", "").replace(".BK", "").strip()
    if v_clean in ticker_to_name:
        return ticker_to_name[v_clean]
    try:
        v_int = int(float(val))
        if v_int in num_to_name:
            return num_to_name[v_int]
    except Exception:
        pass
    return v_str

print("\n=== [2/3] Adding `firm_name` column to SQLite tables ===")
tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

updated_count = 0
for tname in tables:
    if tname.startswith("sqlite_"):
        continue
    df = pd.read_sql_query(f"SELECT * FROM {tname}", conn)
    if df.empty:
        continue
    
    id_col = next((c for c in ["firm_id", "account_id", "issuer_code", "symbol", "ticker"] if c in df.columns), None)
    if not id_col:
        continue
        
    names = df[id_col].map(get_name)
    
    # Insert firm_name column right after id_col
    if "firm_name" in df.columns:
        df["firm_name"] = names
    else:
        idx_pos = df.columns.get_loc(id_col) + 1
        df.insert(idx_pos, "firm_name", names)
        
    df.to_sql(tname, conn, if_exists="replace", index=False)
    print(f"  [SUCCESS] Updated table `{tname}` with column `firm_name`")
    updated_count += 1

conn.close()
print(f"\n=== [3/3] Completed adding `firm_name` to {updated_count} tables! ===")
