# -*- coding: utf-8 -*-
"""
inspect_and_fix_all_sqlite_tables.py
================================================================================
Scans all tables in SQLite `cmdf_credit.db`, replaces numeric firm IDs (1, 2, 3...)
with exact ticker string labels ('2S.BK', '88THm.BK', 'A.BK', etc.), and ensures
every table displays ticker codes and firm names cleanly.
"""
import sqlite3
import pandas as pd
import numpy as np
from thaibma_paths import DATA_ROOT  # data lives outside the repo

DTA_PATH = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"
DB_PATH = os.path.join(DATA_ROOT, "cmdf_credit.db")

# 1. Load Stata category dictionary
print("=== [1/3] Loading Stata category mapping ===")
dta = pd.read_stata(DTA_PATH, columns=["firm_id"])
categories = dta["firm_id"].cat.categories
num_to_ticker = {i+1: cat for i, cat in enumerate(categories)}

print(f"Loaded {len(num_to_ticker)} category label mappings (1 -> '{num_to_ticker[1]}', 3 -> '{num_to_ticker[3]}')")

def safe_map(x):
    if pd.isna(x):
        return x
    try:
        val_int = int(x)
        if val_int in num_to_ticker:
            return num_to_ticker[val_int]
    except (ValueError, TypeError):
        pass
    return str(x)

# 2. Inspect and fix all SQLite tables
print("\n=== [2/3] Inspecting and updating SQLite tables in cmdf_credit.db ===")
conn = sqlite3.connect(DB_PATH)
tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

updated_count = 0
for tname in tables:
    if tname.startswith("sqlite_"):
        continue
    df = pd.read_sql_query(f"SELECT * FROM {tname}", conn)
    if df.empty:
        continue
    
    modified = False
    
    # Check candidates for firm_id / account_id / ticker
    for col in ["firm_id", "account_id", "ticker", "issuer", "issuer_code"]:
        if col in df.columns:
            sample_vals = df[col].dropna().head(50).tolist()
            if not sample_vals:
                continue
            
            # Check if any sample value is numeric or numeric string matching Stata index
            has_num = False
            for v in sample_vals:
                try:
                    v_int = int(v)
                    if v_int in num_to_ticker:
                        has_num = True
                        break
                except (ValueError, TypeError):
                    continue
                
            if has_num:
                print(f"  [FIXING] Table `{tname}`, column `{col}` contains numeric IDs -> mapping to ticker labels...")
                mapped_series = df[col].map(safe_map)
                df[col] = mapped_series
                df["firm_id"] = mapped_series
                df["issuer_code"] = mapped_series.astype(str).str.replace("m.BK", "", regex=False).str.replace(".BK", "", regex=False).str.strip()
                modified = True
                updated_count += 1
                break

    if modified:
        df.to_sql(tname, conn, if_exists="replace", index=False)
        print(f"  [SUCCESS] Updated table `{tname}` with string tickers (e.g. 'A.BK', 'A')")

print(f"\n=== [3/3] Completed updating {updated_count} tables ===")
conn.close()
