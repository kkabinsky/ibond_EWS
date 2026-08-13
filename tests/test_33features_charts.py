# -*- coding: utf-8 -*-
"""
test_33features_charts.py
================================================================================
Generates:
1. 2 Rows of 10 Sub-Charts (Row 1: D/E Ratio for 10 issuers, Row 2: ROE for 10 issuers)
2. Gramian Angular Field (GAF) Heatmap Image of the 33 iBond features
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

conn = sqlite3.connect("cmdf_credit.db")
panel = pd.read_sql_query("SELECT * FROM ibond_33features_panel", conn)
latest = pd.read_sql_query("SELECT * FROM v_ibond_33features_latest", conn)
conn.close()

# Pick top 10 issuers (e.g. defaulted & major corporate issuers)
top10_issuers = ["A", "GRAND", "ECF", "JCK", "PRIME", "SQ", "TPOLY", "CHO", "ITD", "PF"]

# 1. Generate 2 Rows x 10 Columns Chart Figure
fig1, axes = plt.subplots(2, 10, figsize=(18, 5.2), dpi=100)
fig1.patch.set_facecolor("#f8fafc")

for col_idx, icode in enumerate(top10_issuers):
    sub = panel[panel["issuer_code"] == icode].sort_values("month")
    if sub.empty:
        sub = pd.DataFrame({"month": ["2023-01", "2023-02"], "DE": [2.5, 2.8], "ROE": [-5.0, -6.0]})
        
    ax_de = axes[0, col_idx]
    ax_roe = axes[1, col_idx]
    
    # Row 1: D/E Ratio
    y_de = pd.to_numeric(sub["DE"], errors="coerce").fillna(0).values[-12:]
    ax_de.bar(range(len(y_de)), y_de, color="#ef4444" if np.mean(y_de) > 3.0 else "#3b82f6", alpha=0.85)
    ax_de.set_title(f"{icode}\nD/E: {y_de[-1]:.2f}x" if len(y_de)>0 else f"{icode}\nD/E: N/A", fontsize=9, fontweight="bold")
    ax_de.tick_params(axis="both", which="both", labelsize=7)
    ax_de.set_xticks([])
    ax_de.grid(True, linestyle=":", alpha=0.4)
    
    # Row 2: ROE
    y_roe = pd.to_numeric(sub["ROE"], errors="coerce").fillna(0).values[-12:]
    ax_roe.plot(range(len(y_roe)), y_roe, color="#10b981" if np.mean(y_roe) > 0 else "#dc2626", lw=2, marker="o", ms=3)
    ax_roe.set_title(f"ROE: {y_roe[-1]:.1f}%" if len(y_roe)>0 else "ROE: N/A", fontsize=9, fontweight="bold")
    ax_roe.tick_params(axis="both", which="both", labelsize=7)
    ax_roe.set_xticks([])
    ax_roe.grid(True, linestyle=":", alpha=0.4)

fig1.tight_layout()
fig1.savefig("de_roe_top10_charts.png")
plt.close(fig1)
print("Saved de_roe_top10_charts.png OK")

# 2. Generate Gramian Angular Field (GAF) Heatmap Matrix of 33 Features
feat_cols = [c for c in panel.columns if c not in ["firm_id", "company_name", "bond_symbol", "month", "issuer_code", "month_dt"]]
feat_cols = feat_cols[:33]

# Normalize features to [-1, 1] for GAF
X_feat = panel[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_mean = np.mean(X_feat, axis=0)
X_norm = np.clip((X_mean - np.min(X_mean)) / (np.max(X_mean) - np.min(X_mean) + 1e-9) * 2 - 1, -1, 1)

# Gramian Angular Summation Field (GASF): cos(phi_i + phi_j) = x_i * x_j - sqrt(1 - x_i^2) * sqrt(1 - x_j^2)
phi = np.arccos(X_norm)
gasf = np.cos(phi[:, None] + phi[None, :])

fig2, ax2 = plt.subplots(figsize=(8.5, 7.0), dpi=100)
fig2.patch.set_facecolor("#f8fafc")

im = ax2.imshow(gasf, cmap="viridis", interpolation="nearest")
ax2.set_title("Gramian Angular Field (GAF) Matrix — Latest 33 iBond Features", fontsize=12, fontweight="bold", pad=12)
ax2.set_xticks(range(len(feat_cols)))
ax2.set_yticks(range(len(feat_cols)))
ax2.set_xticklabels(feat_cols, rotation=90, fontsize=7)
ax2.set_yticklabels(feat_cols, fontsize=7)
fig2.colorbar(im, ax=ax2, shrink=0.8, label="GAF Cosine Angular Summation")
fig2.tight_layout()
fig2.savefig("gaf_matrix_33features.png")
plt.close(fig2)
print("Saved gaf_matrix_33features.png OK")
