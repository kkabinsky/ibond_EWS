import sqlite3
import pandas as pd

conn = sqlite3.connect('cmdf_credit.db')

print("Exporting ibond_33features_panel.xlsx...")
df_issuer = pd.read_sql_query("SELECT * FROM ibond_33features_panel", conn)
df_issuer.to_excel("ibond_33features_panel.xlsx", index=False)
print("Saved ibond_33features_panel.xlsx (", len(df_issuer), "rows)")

print("Exporting ibond_issue_33features_panel.xlsx...")
df_issue = pd.read_sql_query("SELECT * FROM ibond_issue_33features_panel LIMIT 10000", conn)
df_issue.to_excel("ibond_issue_33features_panel.xlsx", index=False)
print("Saved ibond_issue_33features_panel.xlsx ( 10,000 sample rows)")

conn.close()
