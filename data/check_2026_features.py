# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd

con = sqlite3.connect("cmdf_credit.db")

df_sample = pd.read_sql("SELECT * FROM ibond_33features_panel LIMIT 1", con)
all_cols = df_sample.columns.tolist()

meta_cols = ['firm_id', 'issuer_code', 'company_name', 'bond_symbol', 'month', 'month_year', 'month_index', 'event', 'default_3m', 'PD_3M', 'Momentum', 'flag_hyper', 'alert']
feat_cols = [c for c in all_cols if c not in meta_cols]

print(f"Total Columns: {len(all_cols)}")
print(f"Feature Columns Count: {len(feat_cols)}")
print("\n=== All 33 Features ===")
for i, col in enumerate(feat_cols, 1):
    print(f"{i:2d}. {col}")

print("\n=== 2026 Monthly Breakdown in SQLite ===")
q_2026 = """
SELECT 
    month, 
    COUNT(issuer_code) as total_firms,
    COUNT(ROA) as count_roa,
    COUNT(DE) as count_de,
    COUNT(amihud_monthly) as count_amihud,
    COUNT(ESGScore) as count_esg,
    COUNT(Policyrate) as count_policyrate
FROM ibond_33features_panel 
WHERE month LIKE '2026%' 
GROUP BY month
"""
df_2026 = pd.read_sql(q_2026, con)
print(df_2026.to_string(index=False))

print("\n=== 2026 Sample Values (Eikon Ratios & Market Liquidity) ===")
q_sample_2026 = """
SELECT 
    company_name, bond_symbol, month, ROA, ROE, DE, CurrentRatio, amihud_monthly, ESGScore, Policyrate
FROM v_ibond_33features_panel 
WHERE month LIKE '2026%' 
LIMIT 10
"""
df_s = pd.read_sql(q_sample_2026, con)
print(df_s.to_string(index=False))

con.close()
