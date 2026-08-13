import sqlite3
import pandas as pd

dta = pd.read_stata('D:/tadgan_gaf/dataset_bond/Rev01_Database_final.dta', columns=['firm_id'])
categories = dta['firm_id'].cat.categories
num_map = {i+1: cat for i, cat in enumerate(categories)}

conn = sqlite3.connect('cmdf_credit.db')
b_univ = pd.read_sql_query('SELECT symbol, issuer_code FROM bond_ews_universe', conn)
p_sym = b_univ.groupby('issuer_code')['symbol'].first().to_dict()

for nid in [33, 76, 144, 194, 293, 327, 419]:
    cat = num_map.get(nid, "Unknown")
    clean = cat.replace("m.BK", "").replace(".BK", "").strip()
    bsym = p_sym.get(clean, clean)
    print(f"Numeric ID {nid:3d} -> Stata Ticker: {cat:10s} -> Clean Issuer: {clean:8s} -> Bond Symbol: {bsym}")
