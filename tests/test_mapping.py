import sqlite3
import pandas as pd

import os
import unittest

from thaibma_paths import DB as _DB   # resolve the database from anywhere, not the cwd


def _require_table(name, hint):
    """Skip rather than error when an operational table was never produced.

    These tables come from the live iBond pipeline, which needs ThaiBMA
    credentials. They are deliberately not shipped, so their absence is an unmet
    prerequisite and not a failure; erroring on import made the whole suite look
    broken on a clean checkout.
    """
    import sqlite3
    if not os.path.exists(_DB):
        raise unittest.SkipTest(f"{os.path.basename(_DB)} not present")
    con = sqlite3.connect(_DB)
    try:
        found = con.execute(
            "SELECT count(1) FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
    finally:
        con.close()
    if not found:
        raise unittest.SkipTest(f"{name} not built - {hint}")


_DTA = os.environ.get("THAIBMA_STATA",
                          "D:/tadgan_gaf/dataset_bond/Rev01_Database_final.dta")
if not os.path.exists(_DTA):
    raise unittest.SkipTest(
        f"Stata source not found at {_DTA}; set THAIBMA_STATA to point at it")
_require_table("bond_ews_universe",
                "run the iBond Approach-1 pipeline (needs THAIBMA_USER/PASS)")

dta = pd.read_stata(_DTA, columns=['firm_id'])
categories = dta['firm_id'].cat.categories
num_map = {i+1: cat for i, cat in enumerate(categories)}

conn = sqlite3.connect(_DB)
b_univ = pd.read_sql_query('SELECT symbol, issuer_code FROM bond_ews_universe', conn)
p_sym = b_univ.groupby('issuer_code')['symbol'].first().to_dict()

for nid in [33, 76, 144, 194, 293, 327, 419]:
    cat = num_map.get(nid, "Unknown")
    clean = cat.replace("m.BK", "").replace(".BK", "").strip()
    bsym = p_sym.get(clean, clean)
    print(f"Numeric ID {nid:3d} -> Stata Ticker: {cat:10s} -> Clean Issuer: {clean:8s} -> Bond Symbol: {bsym}")
