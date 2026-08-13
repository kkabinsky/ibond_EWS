import sqlite3
import pandas as pd

conn = sqlite3.connect('cmdf_credit.db')
df = pd.read_sql_query("""
    SELECT symbol, issuer_code, issuer_name, month, coupon, maturity_date, months_to_maturity, ROA, DE, amihud_monthly, ESGScore 
    FROM ibond_issue_33features_panel 
    WHERE issuer_code = 'A' 
    LIMIT 10
""", conn)
print("==========================================================================================")
print(" SAMPLE BOND ISSUE MAPPING (ibond_issue_33features_panel)")
print("==========================================================================================")
print(df.to_string(index=False))
print("==========================================================================================")
