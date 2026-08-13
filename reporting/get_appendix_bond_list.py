# -*- coding: utf-8 -*-
"""
get_appendix_bond_list.py
================================================================================
Extracts the latest corporate bond list from `v_ibond_33features_latest`
and formats a clean LaTeX Appendix table.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("cmdf_credit.db")
df = pd.read_sql_query("SELECT * FROM v_ibond_33features_latest", conn)
conn.close()

print(f"Total Rows in View: {len(df)}")
print(df.head(25).to_string(index=False))

# Generate LaTeX table rows
latex_rows = []
for idx, r in df.head(45).iterrows():
    fid = r["firm_id"]
    cname = str(r["company_name"]).replace("&", r"\&")
    bsym = str(r["bond_symbol"]).replace("_", r"\_")
    pd3m = r["PD_3M"]
    mom = r["Momentum"]
    alert = r["alert"]
    if alert == "HIGH RISK":
        status_tex = r"\textcolor{red}{\textbf{HIGH RISK}}"
    elif alert == "ELEVATED":
        status_tex = r"\textcolor{orange}{\textbf{ELEVATED}}"
    elif alert == "WATCH":
        status_tex = r"\textcolor{yellow!80!black}{\textbf{WATCH}}"
    else:
        status_tex = r"\textcolor{green!60!black}{\textbf{OK}}"
        
    latex_rows.append(f"{fid} & {cname} & \\texttt{{{bsym}}} & {pd3m} & {mom} & {status_tex} \\\\")

with open("appendix_bonds_latex.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(latex_rows))

print("\nSaved appendix_bonds_latex.txt successfully.")
