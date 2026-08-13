import sqlite3
import pandas as pd

conn = sqlite3.connect('cmdf_credit.db')
df = pd.read_sql_query("""
    SELECT firm_id, company_name, bond_symbol, month, amihud_monthly, ROA, DE 
    FROM v_ibond_33features_panel 
    WHERE firm_id NOT IN ('MOF', 'TTB', 'EXIM', 'GHB')
    LIMIT 30
""", conn)
print(df.to_string(index=False))
conn.close()
