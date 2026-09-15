# -*- coding: utf-8 -*-
"""
build_dataset2_view.py -- put dataset 2 into the database the GUI already opens, so
both panels can be inspected from one window.

THE PROBLEM
    The GUI opens exactly one SQLite file, cmdf_credit.db, and lists whatever tables
    and views it finds there. Dataset 2 lives in a different file, lime_credit.db,
    which is 127 MB and holds a much wider table than the GUI needs. Pointing the
    GUI at that file instead would swap one panel for the other and lose every
    operational table the other tabs read, which is not "look at both".

WHAT THIS COPIES
    Only what the grid and the charts actually read: the issuer key, the month, the
    33 determinants, the names, and the event label. The 941-firm table carries
    around 90 columns, most of them intermediate fields from its own construction,
    and none of those are worth 187,007 rows of space in a file that is opened on
    every launch.

WHAT IT BUILDS
    ibond_33features_panel_941firm   the compact panel
    firm_issuer_mapping_ds2          firm_id / name / symbol per firm
    v_ibond_33features_panel_ds2     the view, same column shape as dataset 1's

    The view is deliberately given the same shape as v_ibond_33features_panel, so
    the grid, the sorting and the search behave identically whichever one is
    selected in the table dropdown.

RUN
    python run.py build_dataset2_view
    python run.py build_dataset2_view --force     rebuild even if already present
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd

import ibond_dataset as ds
from cmdf_tree_classify import BOND_33

PANEL = "ibond_33features_panel_941firm"
MAPPING = "firm_issuer_mapping_ds2"
VIEW = "v_ibond_33features_panel_ds2"

KEEP_META = ["issuer_code", "symbol", "month", "name", "issuer_name",
             "issuer_name_th", "market", "sector", "industry", "y_pre3m",
             "d_DP_RS", "default_month"]


def _have(con, name):
    return con.execute("SELECT count(1) FROM sqlite_master WHERE name=?",
                       (name,)).fetchone()[0] > 0


def build(force=False, verbose=True):
    ds.use(2)
    src = ds.READ_DB
    dst = ds.RESULT_DB
    if not os.path.exists(src):
        raise SystemExit(
            f"dataset 2 database not found. Unzip lime_credit.db.zip from the "
            f"iBond_LIME repository into a datasets_bond/ folder beside this one.")
    if os.path.abspath(src) == os.path.abspath(dst):
        if verbose:
            print("  dataset 2 already lives in the GUI's database; nothing to copy")
        return

    out = sqlite3.connect(dst)
    if _have(out, VIEW) and not force:
        n = out.execute(f"SELECT count(*) FROM {VIEW}").fetchone()[0]
        out.close()
        if verbose:
            print(f"  {VIEW} already present with {n:,} rows (use --force to rebuild)")
        return

    if verbose:
        print(f"=== [1/3] Copying the compact dataset-2 panel ===")
        print(f"      from {src}")
        print(f"      into {dst}")

    con = sqlite3.connect(src)
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({ds.TABLE})")]
    take = [c for c in KEEP_META + BOND_33 if c in cols]
    seen, order = set(), []
    for c in take:                                   # KEEP_META and BOND_33 overlap
        if c not in seen:
            seen.add(c)
            order.append(c)
    sel = ", ".join(f"`{c}`" for c in order)
    panel = pd.read_sql(f"SELECT {sel} FROM {ds.TABLE}", con)
    con.close()

    if "issuer_code" not in panel.columns or panel["issuer_code"].isna().all():
        panel["issuer_code"] = panel["symbol"].astype(str)
    panel["issuer_code"] = panel["issuer_code"].astype(str)
    panel.to_sql(PANEL, out, if_exists="replace", index=False)
    if verbose:
        print(f"      {PANEL}: {len(panel):,} rows x {len(panel.columns)} columns")

    if verbose:
        print("=== [2/3] Building the firm mapping ===")
    name_col = next((c for c in ("name", "issuer_name", "issuer_name_th")
                     if c in panel.columns), None)
    g = (panel.sort_values("issuer_code")
         .groupby("issuer_code", as_index=False)
         .agg(**{
             "company_name": (name_col, "first") if name_col else
                             ("issuer_code", "first"),
             "sector": ("sector", "first") if "sector" in panel.columns else
                       ("issuer_code", "first"),
             "bond_symbol": ("symbol", "first") if "symbol" in panel.columns else
                            ("issuer_code", "first"),
         }))
    g.insert(0, "firm_id", range(1, len(g) + 1))
    g["company_name"] = g["company_name"].fillna(g["issuer_code"]).astype(str)
    g["company_name_th"] = g["company_name"]
    g["stata_ticker"] = g["issuer_code"].astype(str) + ".BK"
    g.to_sql(MAPPING, out, if_exists="replace", index=False)
    if verbose:
        print(f"      {MAPPING}: {len(g):,} firms")

    if verbose:
        print("=== [3/3] Creating the view ===")
    out.execute(f"DROP VIEW IF EXISTS {VIEW};")
    ignore = {"id", "firm_id", "issuer_code", "issuer_name", "bond_symbol",
              "symbol", "firm_name", "clean_id"}
    feat = [c for c in panel.columns if c not in ignore]
    feat_sql = ", ".join(f"p.`{c}`" for c in feat)
    out.execute(f"""
        CREATE VIEW {VIEW} AS
        SELECT
            m.firm_id       AS `firm_id`,
            m.company_name  AS `firm_name`,
            m.company_name  AS `company_name`,
            m.bond_symbol   AS `bond_symbol`,
            p.issuer_code   AS `issuer_code`,
            {feat_sql}
        FROM {PANEL} p
        JOIN {MAPPING} m ON p.issuer_code = m.issuer_code
        ORDER BY p.month DESC, CAST(m.firm_id AS INTEGER) ASC;
    """)
    out.commit()

    n = out.execute(f"SELECT count(*) FROM {VIEW}").fetchone()[0]
    sample = pd.read_sql(
        f"SELECT firm_id, firm_name, bond_symbol, month, ROA, DE FROM {VIEW} "
        f"LIMIT 6", out)
    size = os.path.getsize(dst) / 1048576
    out.close()
    ds.use(1)
    if verbose:
        print(f"      {VIEW}: {n:,} rows")
        print(f"      {os.path.basename(dst)} is now {size:.0f} MB")
        print()
        print(sample.to_string(index=False))
        print()
        print(f"  Both panels are now selectable in the GUI's table dropdown:")
        print(f"    v_ibond_33features_panel      dataset 1, 293 bond issuers")
        print(f"    {VIEW}  dataset 2, 941 listed firms")


if __name__ == "__main__":
    build(force="--force" in sys.argv, verbose=True)
