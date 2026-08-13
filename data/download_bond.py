# -*- coding: utf-8 -*-
"""
download_bond.py -- download Thai CORPORATE BOND data from iBond.

This complements download_bound.py (which pulls the government yield curve).
Here we pull the issuer master, the outstanding corporate-bond list, and the
payment-default register -- the three tables the credit early-warning model
actually needs.

ENDPOINTS  (read out of iBond's own public JavaScript bundle, gRPC-Web text)

  issuers    /grpc/issuer-grpc/issuer.IssuerGrpcService/GetIssuerList
             req  IssuerRequest        1=searchdata  2=businessSectorCode
             rep  IssuerReply          1=institutionId 2=code 3=nameEn 4=nameTh
                                       5=website 6=sectorCode 7=sectorName
                                       8=ratingDate(ts) 9=corporateType 10=instRating

  bonds      /grpc/registeredbond-grpc/registeredbond.RegisteredBondGrpcService/
             GetBondOutstandingListCorp
             req  BondOutstandingCorpListRequest  1=issuerTypeCode  2=sector
             rep  BondOutstandingCorpListReply    1=issueId 3=symbol
                                       4=tris 5=fitch 6=moody 7=sp 8=fitchInter
                                       9=ri 10=couponType 11=interestRate(double)
                                       12=maturityDate(ts) 13=issuedDate(ts)
                                       14=ttm 15=outstanding(double) 16=currency
                                       17=issuerType 18=symbolOrder

  defaults   /grpc/bondfeature-grpc/bondfeature.BondFeatureGrpcService/GetDefaultPayment
             req  DefaultPaymentRequest  1=issueId  2=asof(ts)
             rep  DefaultPaymentReply    1=rowNo 2=issueId 3=legacyId 4=symbol
                                       5=paymentDate(ts) 6=remark 7=defaultTypeCode
                                       8=defaultTypeNameEn 9=defaultTypeNameTh

SECURITY
    Credentials are read from THAIBMA_USER / THAIBMA_PASS in your own environment
    (set them with setup_credentials.py). No password is stored in this file.

RUN
    python download_bond.py                  # issuers + outstanding corporate bonds
    python download_bond.py --defaults       # also scan issues for payment defaults
    python download_bond.py --defaults --limit 300
    python download_bond.py --no-save
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd

import ibond_grpc as ig
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")

ISSUER_SVC = "/grpc/issuer-grpc/issuer.IssuerGrpcService"
REGBOND_SVC = "/grpc/registeredbond-grpc/registeredbond.RegisteredBondGrpcService"
FEATURE_SVC = "/grpc/bondfeature-grpc/bondfeature.BondFeatureGrpcService"
BOND_SVC = "/grpc/bond-grpc/bond.BondGrpcService"

T_ISSUER = "ibond_issuer"
T_BOND = "ibond_corp_bond"
T_SUMMARY = "ibond_outstanding_summary"
T_DEFAULT = "ibond_default_payment"
T_LOG = "ibond_bond_log"


# ------------------------------------------------------------- decoding ------
def _s(fields, n):
    """String field n, or ''."""
    v = fields.get(n, [b""])[0]
    return v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else ""


def _i(fields, n):
    """Varint field n, or None."""
    v = fields.get(n, [None])[0]
    return int(v) if isinstance(v, int) else None


def _ts(fields, n):
    """google.protobuf.Timestamp field n -> date, or None."""
    v = fields.get(n, [None])[0]
    if not isinstance(v, (bytes, bytearray)):
        return None
    d = ig._timestamp_value(v)
    return d.date() if d else None


def _dbl(fields, n):
    """google.protobuf.DoubleValue / Int32Value wrapper field n -> float."""
    v = fields.get(n, [None])[0]
    if not isinstance(v, (bytes, bytearray)):
        return None
    got = ig._double_value(v)
    if got is not None:
        return got
    inner = ig.pb_parse(v)                      # Int32Value stores a varint
    iv = inner.get(1, [None])[0]
    return float(iv) if isinstance(iv, int) else None


def _items(reply: bytes):
    """IEnumerable_*Reply wraps repeated items in field 1."""
    return ig.pb_parse(reply).get(1, [])


# ------------------------------------------------------------- fetching ------
def fetch_issuers(c: ig.IBondGrpc, search: str = "", sector: str = "") -> pd.DataFrame:
    body = b""
    if search:
        body += ig.pb_string(1, search)
    if sector:
        body += ig.pb_string(2, sector)
    reply = c.call(ISSUER_SVC, "GetIssuerList", body)
    rows = []
    for it in _items(reply):
        f = ig.pb_parse(it)
        rows.append({
            "institution_id": _s(f, 1), "institution_code": _s(f, 2),
            "name_en": _s(f, 3), "name_th": _s(f, 4), "website": _s(f, 5),
            "sector_code": _s(f, 6), "sector_name": _s(f, 7),
            "rating_date": _ts(f, 8), "corporate_type": _s(f, 9),
            "issuer_rating": _s(f, 10),
        })
    return pd.DataFrame(rows)


def fetch_all_bonds(c: ig.IBondGrpc) -> pd.DataFrame:
    """bond.BondGrpcService/GetAllBond -> every registered issue (id + symbol).

    This is the endpoint that actually works for the full universe.
    GetBondOutstandingListCorp returns 0 rows for every parameter combination we
    tried, and BondSearch/GetSearchResult needs the whole 41-field filter object
    (and still returned nothing), so GetAllBond is the reliable entry point.
    """
    reply = c.call(BOND_SVC, "GetAllBond", b"", timeout=180)
    rows = []
    for it in _items(reply):
        f = ig.pb_parse(it)
        rows.append({"issue_id": _s(f, 1), "symbol": _s(f, 2)})
    return pd.DataFrame(rows)


def fetch_outstanding_summary(c: ig.IBondGrpc) -> pd.DataFrame:
    """Outstanding value summarised by corporate bond type."""
    reply = c.call(REGBOND_SVC, "GetOutstandingSummaryCorp", b"", timeout=90)
    rows = []
    for it in _items(reply):
        f = ig.pb_parse(it)
        rows.append({"bond_type": _s(f, 1), "value_1": _dbl(f, 2),
                     "value_2": _dbl(f, 3), "order": _i(f, 4),
                     "code": _s(f, 5), "n": _i(f, 6),
                     "name": _s(f, 8)})
    return pd.DataFrame(rows)


def fetch_default_payments(c: ig.IBondGrpc, issue_ids, verbose=True) -> pd.DataFrame:
    """GetDefaultPayment is per-issue, so this walks the issue list."""
    rows, hits, errs = [], 0, 0
    for k, iid in enumerate(issue_ids, 1):
        if not iid:
            continue
        try:
            reply = c.call(FEATURE_SVC, "GetDefaultPayment", ig.pb_string(1, iid), timeout=45)
        except Exception:
            errs += 1
            continue
        for it in _items(reply):
            f = ig.pb_parse(it)
            rows.append({
                "issue_id": _s(f, 2) or iid, "symbol": _s(f, 4),
                "payment_date": _ts(f, 5), "remark": _s(f, 6),
                "default_type_code": _s(f, 7),
                "default_type_en": _s(f, 8), "default_type_th": _s(f, 9),
            })
            hits += 1
        if verbose and k % 100 == 0:
            print(f"    scanned {k}/{len(issue_ids)} issues — {hits} default records, "
                  f"{errs} errors")
    if verbose:
        print(f"    scanned {len(issue_ids)} issues — {hits} default records, {errs} errors")
    return pd.DataFrame(rows)


# -------------------------------------------------------------- storage ------
def save_to_sqlite(issuers, bonds, defaults, summary=None, db=DB):
    con = sqlite3.connect(db)
    try:
        saved = {}
        for df, t in ((issuers, T_ISSUER), (bonds, T_BOND), (defaults, T_DEFAULT),
                      (summary, T_SUMMARY)):
            if df is not None and not df.empty:
                d = df.copy()
                for col in d.columns:
                    if d[col].map(lambda v: hasattr(v, "isoformat")).any():
                        d[col] = d[col].astype(str)
                d.to_sql(t, con, if_exists="replace", index=False)
                saved[t] = len(d)
        pd.DataFrame([{
            "downloaded_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_issuers": 0 if issuers is None else len(issuers),
            "n_bonds": 0 if bonds is None else len(bonds),
            "n_defaults": 0 if defaults is None else len(defaults),
            "source": "ibond-grpc",
        }]).to_sql(T_LOG, con, if_exists="append", index=False)
        con.commit()
        return saved
    finally:
        con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    out = []
    try:
        for t in (T_ISSUER, T_BOND, T_DEFAULT, T_LOG):
            try:
                out.append(pd.read_sql(f"select * from {t}", con))
            except Exception:
                out.append(pd.DataFrame())
    finally:
        con.close()
    return tuple(out)


# ------------------------------------------------------------------ run ------
def run(with_defaults=False, limit=None, save=True, verbose=True):
    if verbose:
        print("=" * 84)
        print("iBond corporate bond download")
        print("=" * 84)
    c = ig.IBondGrpc()
    who = c.login()
    if verbose:
        print(f"logged in as {who.get('user_name') or who.get('user_id')} "
              f"(token {who['token_len']} chars)\n")

    if verbose:
        print("[1/3] issuer master ...")
    issuers = fetch_issuers(c)
    if verbose:
        print(f"      {len(issuers):,} issuers")

    if verbose:
        print("[2/3] registered bond universe ...")
    bonds = fetch_all_bonds(c)
    summary = fetch_outstanding_summary(c)
    if verbose:
        print(f"      {len(bonds):,} issues, {len(summary)} outstanding-summary rows")

    defaults = pd.DataFrame()
    if with_defaults and not bonds.empty:
        ids = bonds["issue_id"].dropna().unique().tolist()
        if limit:
            ids = ids[:limit]
        if verbose:
            print(f"[3/3] payment defaults — scanning {len(ids):,} issues "
                  f"(this is one request per issue) ...")
        defaults = fetch_default_payments(c, ids, verbose=verbose)
    elif verbose:
        print("[3/3] payment defaults — skipped (use --defaults to scan)")

    saved = save_to_sqlite(issuers, bonds, defaults, summary, DB) if save else {}
    if verbose and saved:
        print("\nsaved to SQLite:")
        for t, n in saved.items():
            print(f"  {t:26s} {n:,} rows")
    return issuers, bonds, defaults, summary


def main():
    with_defaults = "--defaults" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    issuers, bonds, defaults, summary = run(with_defaults=with_defaults, limit=limit,
                                            save="--no-save" not in sys.argv)
    if not issuers.empty:
        print("\nISSUERS (first 8)")
        print(issuers[["institution_code", "name_en", "sector_name",
                       "issuer_rating"]].head(8).to_string(index=False))
    if not bonds.empty:
        print(f"\nBOND UNIVERSE: {len(bonds):,} issues (first 8)")
        print(bonds.head(8).to_string(index=False))
    if summary is not None and not summary.empty:
        print("\nOUTSTANDING SUMMARY BY BOND TYPE")
        print(summary[["code", "name", "n"]].to_string(index=False))
    if not defaults.empty:
        print("\nPAYMENT DEFAULTS (first 10)")
        print(defaults[["symbol", "payment_date", "default_type_en"]]
              .head(10).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
