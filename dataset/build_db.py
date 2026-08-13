# -*- coding: utf-8 -*-
"""
build_db.py -- rebuild a working cmdf_credit.db from the CSV extracts in this folder.

WHY THE DATABASE IS NOT IN THE REPOSITORY
    The full working database is 375 MB and holds 155 tables, most of which are
    operational: e-mail delivery logs, alert queues, intermediate model output. None of
    that belongs in a public repository, and GitHub rejects any single file above
    100 MB in any case.

    What the published research actually reads is three tables. They are shipped here
    as gzipped CSV, 1.4 MB in total, and this script turns them back into the SQLite
    file the code expects.

WHAT YOU GET
    ibond_33features_panel   16,986 issuer-months, 293 issuers, 2007-11 to 2026-08
    ibond_issuer                677 issuer records
    ibond_default_payment        50 recorded non-payment events

    That is enough to reproduce every figure and table in the report: the out-of-fold
    PD path, the review-capacity threshold, the shock ladder, the pairwise and triple
    shock decompositions, the PD surfaces, and the hyperbolic boundary.

WHAT YOU DO NOT GET
    The GUI tabs that read the operational tables (e-mail scheduling, delivery history,
    the older Approach-1 alert tables) will show nothing, because those tables are not
    part of the extract. The analysis scripts do not touch them.

LICENCE OF THE DATA
    The panel is derived from ThaiBMA iBond material. Check your own licence terms
    before redistributing it further.

RUN
    python build_db.py
    python build_db.py --out /somewhere/else/cmdf_credit.db
"""
from __future__ import annotations

import argparse
import gzip
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = {
    "ibond_33features_panel": "ibond_33features_panel.csv.gz",
    "ibond_issuer": "ibond_issuer.csv.gz",
    "ibond_default_payment": "ibond_default_payment.csv.gz",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE),
                                                  "cmdf_credit.db"),
                    help="where to write the database (default: repository root)")
    a = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install -r requirements.txt", file=sys.stderr)
        return 1

    missing = [f for f in TABLES.values() if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        print(f"missing extract files: {missing}", file=sys.stderr)
        return 1

    con = sqlite3.connect(a.out)
    for table, fname in TABLES.items():
        with gzip.open(os.path.join(HERE, fname), "rt", encoding="utf-8") as fh:
            # low_memory=False: event_date and event_month mix blanks with dates,
            # and chunked inference would type them differently per chunk
            df = pd.read_csv(fh, low_memory=False)
        df.to_sql(table, con, if_exists="replace", index=False)
        print(f"  {table:26s} {len(df):>7,} rows, {len(df.columns):>3} columns")
    con.commit()
    con.close()

    size = os.path.getsize(a.out) / 1048576
    print(f"\nwrote {a.out}  ({size:.1f} MB)")
    print("the code finds it automatically; nothing needs configuring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
