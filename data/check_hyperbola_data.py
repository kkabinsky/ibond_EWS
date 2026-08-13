import sqlite3
import pandas as pd

conn = sqlite3.connect('cmdf_credit.db')
df_a1 = pd.read_sql_query("SELECT symbol, issuer_code, month, PD_3M, Momentum, flag_hyper, alarm FROM bond_ews_alert WHERE flag_hyper = 1 LIMIT 15", conn)
df_a2 = pd.read_sql_query("SELECT symbol, issuer_code, month, PD_3M, Momentum, flag_hyper, alarm FROM bond_ews_xgb_alert WHERE flag_hyper = 1 LIMIT 15", conn)

print("==========================================================================================")
print(" APPROACH 1 (COX HAZARD) — HYPERBOLA ALERTS WITH REAL BOND SYMBOLS")
print("==========================================================================================")
print(df_a1.to_string(index=False))

print("\n==========================================================================================")
print(" APPROACH 2 (XGBOOST HAZARD) — HYPERBOLA ALERTS WITH REAL BOND SYMBOLS")
print("==========================================================================================")
print(df_a2.to_string(index=False))
print("==========================================================================================")
conn.close()
