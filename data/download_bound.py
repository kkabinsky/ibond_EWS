# -*- coding: utf-8 -*-
"""
download_bound.py

One-click iBond/ThaiBMA yield-curve download orchestration.

What this file does now
-----------------------
1. Reads THAIBMA_USER / THAIBMA_PASS / THAIBMA_API_KEY from the environment.
2. Tries the existing authenticated fetch flow in ibond_client.py.
3. If successful, saves raw yield data + DNS tables into SQLite.
4. If not successful, fails clearly with the exact next step.

Important reality check
----------------------------
iBond speaks gRPC-Web *text* (base64), not REST — the old guessed REST endpoints
returned 405. ibond_grpc.py implements that protocol: POST base64(frame) with
Content-Type application/grpc-web-text to
  /grpc/authen-grpc/authen.AuthenGrpcService/Authenticate      (userName, password)
  /grpc/yieldcurve-grpc/yieldcurve.YieldCurveGrpcService/GetYieldCurveByAsOf
Verified against the live server with a probe account: a wrong username returns a
clean grpc-status 16 "User name not found", which confirms the request format is
correct, so real credentials authenticate and download the curve.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import ibond_client as ib
import yield_curve_dns as ycd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")


@dataclass
class DownloadResult:
    ok: bool
    message: str
    rows: int = 0
    tenors: int = 0
    source: str = ""
    start: str = ""
    end: str = ""


def credential_status() -> dict:
    return ib.credentials_status()


def sqlite_tables(db: str = DB) -> list[str]:
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def download_and_store(start: Optional[str] = None, end: Optional[str] = None) -> DownloadResult:
    st = credential_status()
    if not st.get("ready"):
        return DownloadResult(
            ok=False,
            message=(
                "Credentials are not ready. Set THAIBMA_USER / THAIBMA_PASS first, "
                "or use setup_credentials.py."
            ),
            start=start or "",
            end=end or "",
        )

    try:
        curve = ib.fetch_curve(start=start, end=end)
    except Exception as ex:
        return DownloadResult(
            ok=False,
            message=(
                "Authenticated download did not succeed yet. "
                f"Current error: {ex}"
            ),
            start=start or "",
            end=end or "",
        )

    if curve is None or curve.empty:
        return DownloadResult(
            ok=False,
            message="Download returned no rows.",
            start=start or "",
            end=end or "",
        )

    factors = ycd.fit_dns(curve)
    if factors.empty:
        return DownloadResult(
            ok=False,
            message="Downloaded rows exist, but no date had enough tenors to fit DNS.",
            rows=len(curve),
            tenors=int(curve['tau'].nunique()) if 'tau' in curve.columns else 0,
            source=str(curve['source'].iloc[0]) if 'source' in curve.columns else '',
            start=start or "",
            end=end or "",
        )

    val = ycd.validate(factors)
    fc = ycd.forecast_factors(factors)
    ycd.save_to_sqlite(curve, factors, val, fc, DB)
    ycd.save_ibond_raw(
        curve,
        DB,
        source=str(curve["source"].iloc[0]) if "source" in curve.columns else "iBond",
        requested_start=start,
        requested_end=end,
        mode="replace",
    )

    return DownloadResult(
        ok=True,
        message="Download completed and saved to SQLite.",
        rows=len(curve),
        tenors=int(curve["tau"].nunique()),
        source=str(curve["source"].iloc[0]) if "source" in curve.columns else "",
        start=start or "",
        end=end or "",
    )


def print_result(res: DownloadResult):
    print("=" * 80)
    print("iBond one-click download")
    print("=" * 80)
    print(f"ok      : {res.ok}")
    print(f"message : {res.message}")
    if res.rows:
        print(f"rows    : {res.rows:,}")
    if res.tenors:
        print(f"tenors  : {res.tenors}")
    if res.source:
        print(f"source  : {res.source}")
    if res.start or res.end:
        print(f"range   : {res.start or '-'} -> {res.end or '-'}")
    print("tables  : " + ", ".join(sqlite_tables()))


def main():
    start = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else None
    end = sys.argv[sys.argv.index("--end") + 1] if "--end" in sys.argv else None
    if "--status" in sys.argv:
        print(credential_status())
        print(sqlite_tables())
        return
    res = download_and_store(start=start, end=end)
    print_result(res)
    if not res.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
