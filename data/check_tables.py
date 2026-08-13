import sqlite3
import pandas as pd

conn = sqlite3.connect('cmdf_credit.db')
tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables in cmdf_credit.db:")
for t in tables:
    df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 1", conn)
    print(f"  - {t}: {len(df.columns)} columns")
    cols = df.columns.tolist()
    if 'ROA' in cols or 'amihud_monthly' in cols or 'DE' in cols:
        print(f"    *** CONTAINS 33 FEATURES *** -> {t}")
