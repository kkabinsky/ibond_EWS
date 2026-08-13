import sqlite3
conn = sqlite3.connect('cmdf_credit.db')
tables = [c[0] for c in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables in SQLite DB:")
for t in sorted(tables):
    print(" -", t)
conn.close()
