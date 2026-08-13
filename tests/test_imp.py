import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('cmdf_credit.db')
b_panel = pd.read_sql_query('SELECT * FROM bond_ews_panel', conn)
b_panel['clean_id'] = b_panel['issuer_code'].astype(str).str.strip()

BOND_FEATURES = [
    "amihud_monthly", "amihud_monthly_100", "adj_illiq_kz", "scaled_amihud",
    "ln_amihud", "percent_zero_days", "zero_days", "n_days",
    "ROA", "ROE", "DE", "CurrentRatio", "QuickRatio", "CashRatio",
    "EBITtoTA", "REtoTA", "WorkingCapitaltoTA", "TDTA", "LTDtoTA", "STDtoTA",
    "cf_Interestcoverageratio", "acc_DebtServiceCoverageRatio",
    "lnTotalAssets", "lnAge",
    "Policyrate", "GDPgrowth", "UnemploymentratemodeledILOe",
    "ESGScore", "GovernancePillarScore", "EnvironmentalPillarScore",
    "SocialPillarScore", "IndependentBoardMembers", "AverageBoardTenure",
]

drop_cols = [c for c in BOND_FEATURES if c in b_panel.columns]
if drop_cols:
    b_panel.drop(columns=drop_cols, inplace=True)

dta = pd.read_stata('D:/tadgan_gaf/dataset_bond/Rev01_Database_final.dta', columns=["firm_id", "month_year"] + BOND_FEATURES)
dta['clean_id'] = dta['firm_id'].astype(str).str.replace('m.BK', '', regex=False).str.replace('.BK', '', regex=False).str.strip()
dta['month'] = pd.to_datetime(dta['month_year']).dt.strftime('%Y-%m')

dta_clean = dta.drop_duplicates(subset=['clean_id', 'month']).copy()

merged = pd.merge(b_panel, dta_clean[['clean_id', 'month'] + BOND_FEATURES], on=['clean_id', 'month'], how='left')

print("Direct matched non-null total:", merged[BOND_FEATURES].notna().sum().sum())

for feat in BOND_FEATURES:
    s = merged.groupby("clean_id")[feat].ffill().bfill()
    merged[feat] = s
    s_m = merged.groupby("month")[feat].transform(lambda x: x.fillna(x.median()))
    merged[feat] = s_m
    merged[feat] = merged[feat].fillna(merged[feat].median())

print("After fill non-null total:", merged[BOND_FEATURES].notna().sum().sum(), "out of", len(merged)*len(BOND_FEATURES))
