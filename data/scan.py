"""
scan.py - one-shot headless credit scan, for schedulers (openclaw, Windows Task
Scheduler, cron). This is the command an automation should call on a timer.

It runs the same pipeline as the GUI, without opening the GUI:

    load data -> train models -> compute PD alerts -> save `alerts` table
              -> notify.py sends email/Telegram ONLY for accounts that just
                 escalated INTO HIGH RISK

Usage
    python scan.py                 # scan whatever is already in SQLite
    python scan.py --real          # refresh REAL SET/US data first, then scan
    python scan.py --dry-run       # run everything but send nothing
    python scan.py --quiet         # print only the final summary line

Exit codes
    0  scan completed (whether or not a notification was sent)
    1  scan failed - the scheduler can alert on a non-zero exit
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
import traceback

import app as credit_app          # safe: app.py keeps its GUI behind __main__
import notify


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_scan(use_real: bool = False, dry_run: bool = False, quiet: bool = False) -> dict:
    """Execute one full scan and return a summary dict."""
    def say(msg: str) -> None:
        if not quiet:
            print("[%s] %s" % (_stamp(), msg))

    if use_real:
        say("refreshing REAL SET/US data ...")
        n_real = credit_app.import_real_to_sqlite()
        say("real data loaded (%s rows)" % n_real)
    else:
        try:
            probe = credit_app.load_df(limit=1)
            if probe is None or probe.empty:
                raise ValueError("empty table")
        except Exception:
            say("no data in SQLite yet - importing the Excel dataset ...")
            credit_app.import_to_sqlite()

    df = credit_app.load_df()
    say("scoring %d accounts ..." % len(df))

    res, best, feats = credit_app.train_models(df)
    alerts = credit_app.compute_alerts(df, res[best]["model"], feats)

    con = sqlite3.connect(credit_app.DB)
    try:
        alerts.to_sql("alerts", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()

    bands = alerts["alert"].value_counts().to_dict()
    say("best model: %s (AUC %.3f) | bands: %s" % (best, res[best]["auc"], bands))

    result = notify.notify_alerts(alerts, credit_app.DB, dry_run=dry_run)
    summary = notify.summary_line(result)
    print("[%s] %s" % (_stamp(), summary))

    return {"accounts": len(df), "best": best, "auc": res[best]["auc"],
            "bands": bands, "notify": result, "summary": summary}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Headless CMDF credit scan + alerting")
    parser.add_argument("--real", action="store_true",
                        help="refresh real SET/US data (yfinance + FRED) before scanning")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute escalations but do not send anything")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the final summary line")
    args = parser.parse_args(argv)

    try:
        run_scan(use_real=args.real, dry_run=args.dry_run, quiet=args.quiet)
        return 0
    except Exception:
        print("[%s] SCAN FAILED" % _stamp())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
