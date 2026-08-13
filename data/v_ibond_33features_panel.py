# -*- coding: utf-8 -*-
"""
v_ibond_33features_panel.py
================================================================================
CLI Executable script that queries and displays the SQLite Database View
`v_ibond_33features_panel` from `cmdf_credit.db`.
"""
import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmdf_credit.db")

def main():
    print("==========================================================================================")
    print(" SQLITE DATABASE VIEW: v_ibond_33features_panel (cmdf_credit.db)")
    print("==========================================================================================")
    
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file `{DB_PATH}` not found.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    try:
        total_rows = conn.execute("SELECT COUNT(*) FROM v_ibond_33features_panel").fetchone()[0]
        total_firms = conn.execute("SELECT COUNT(DISTINCT firm_id) FROM v_ibond_33features_panel").fetchone()[0]
        
        print(f"Total Records: {total_rows:,} firm-months across {total_firms:,} corporate bond issuers\n")
        
        print("--- [1] LATEST SNAPSHOT PER COMPANY (1 Row per Corporate Issuer) ---")
        df_latest = pd.read_sql_query("""
            SELECT firm_id, company_name, bond_symbol, month, amihud_monthly, ROA, DE, Policyrate, ESGScore 
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY firm_id ORDER BY month DESC) as rn
                FROM v_ibond_33features_panel
            ) 
            WHERE rn = 1
            ORDER BY firm_id ASC
            LIMIT 25
        """, conn)
        print(df_latest.to_string(index=False))
        
        print("\n--- [2] SAMPLE PANEL ROWS (Cross-Sectional Month 2024-06) ---")
        df_cross = pd.read_sql_query("""
            SELECT firm_id, company_name, bond_symbol, month, amihud_monthly, ROA, DE, Policyrate, ESGScore 
            FROM v_ibond_33features_panel
            WHERE month = '2024-06'
            ORDER BY firm_id ASC
            LIMIT 25
        """, conn)
        print(df_cross.to_string(index=False))

        print("\n==========================================================================================")
        print("Hint: Run `python app.py` to open the full interactive GUI application with DataGrid Inspector.")
        print("==========================================================================================")
    except Exception as ex:
        print(f"Notice: Rebuilding View `v_ibond_33features_panel` due to: {ex}")
        import fix_view_and_sorting as fvs
        fvs.rebuild_view(verbose=True)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
