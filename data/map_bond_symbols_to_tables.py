# -*- coding: utf-8 -*-
"""
map_bond_symbols_to_tables.py
================================================================================
Maps issuer codes to their real primary iBond symbols (e.g. 'A' -> 'A24NA',
'GRAND' -> 'GRAND245A', 'CPALL' -> 'CPALL256A') across all SQLite tables in
cmdf_credit.db and app views.
"""
import sqlite3
import pandas as pd
import numpy as np
from thaibma_paths import DATA_ROOT  # data lives outside the repo

DB_PATH = os.path.join(DATA_ROOT, "cmdf_credit.db")

print("=== [1/3] Loading real bond symbols mapping from bond_ews_universe ===")
conn = sqlite3.connect(DB_PATH)
b_univ = pd.read_sql_query("SELECT symbol, issuer_code, issuer_name FROM bond_ews_universe", conn)

# Get primary representative bond symbol per issuer
primary_symbol = b_univ.groupby("issuer_code")["symbol"].first().to_dict()
# Also list of all symbols per issuer
all_symbols = b_univ.groupby("issuer_code")["symbol"].apply(lambda s: ", ".join(s.unique())).to_dict()
issuer_names = b_univ.groupby("issuer_code")["issuer_name"].first().to_dict()

# Add mapping for clean tickers (without .BK)
dta = pd.read_stata(r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta", columns=["firm_id"])
categories = dta["firm_id"].cat.categories
num_to_code = {i+1: cat.replace("m.BK", "").replace(".BK", "").strip() for i, cat in enumerate(categories)}
num_to_symbol = {i+1: primary_symbol.get(cat.replace("m.BK", "").replace(".BK", "").strip(), cat) for i, cat in enumerate(categories)}

print(f"Mapped {len(primary_symbol)} iBond issuers to real bond symbols (e.g., A -> {primary_symbol.get('A')}, GRAND -> {primary_symbol.get('GRAND')})")

print("\n=== [2/3] Updating SQLite tables to display real bond symbols ===")
tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

def to_bond_symbol(val):
    if pd.isna(val):
        return val
    val_str = str(val).replace("m.BK", "").replace(".BK", "").strip()
    try:
        val_int = int(val_str)
        code = num_to_code.get(val_int, str(val_int))
        return primary_symbol.get(code, code)
    except (ValueError, TypeError):
        pass
    return primary_symbol.get(val_str, val_str)

for tname in tables:
    if tname.startswith("sqlite_"):
        continue
    df = pd.read_sql_query(f"SELECT * FROM {tname}", conn)
    if df.empty:
        continue
    
    modified = False
    
    # Update symbol / bond_symbol column
    if "issuer_code" in df.columns:
        df["bond_symbol"] = df["issuer_code"].map(lambda x: primary_symbol.get(str(x).strip(), str(x)))
        df["symbol"] = df["bond_symbol"]
        modified = True
        
    for col in ["firm_id", "account_id", "ticker", "issuer"]:
        if col in df.columns:
            df["bond_symbol"] = df[col].map(to_bond_symbol)
            df["symbol"] = df["bond_symbol"]
            modified = True

    if modified:
        df.to_sql(tname, conn, if_exists="replace", index=False)
        print(f"  [SUCCESS] Updated table `{tname}` with real bond symbols (e.g. 'A24NA', 'GRAND245A')")

conn.close()
print("\n=== [3/3] All tables updated successfully with real bond symbols! ===")
