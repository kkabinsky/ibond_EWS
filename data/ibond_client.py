# -*- coding: utf-8 -*-
"""
ibond_client.py -- authenticated download of the Thai government bond yield curve
                   from ThaiBMA / iBond.

SECURITY -- read this first
===========================
Your credentials stay on YOUR machine. This module never stores, prints or transmits
them anywhere except to thaibma.or.th over HTTPS, and there is deliberately no place
in the code to hard-code a password. Do NOT paste your password into a chat, a source
file, a notebook, or a screenshot.

Set them once in a terminal (they live only in your Windows user profile):

    setx THAIBMA_USER "your_username"
    setx THAIBMA_PASS "your_password"
    # optional, if you subscribe to the REST API instead of the web login:
    setx THAIBMA_API_KEY "your_api_key"

Open a NEW terminal afterwards so the variables are visible, then use the
"Yield Curve" menu in app.py (leave the file box empty) or:

    python ibond_client.py --fetch --start 2000-01 --end 2026-07

Access routes, tried in this order
----------------------------------
1. REST API      THAIBMA_API_KEY   -> apiportal.thaibma.or.th (subscription product)
2. Web login     THAIBMA_USER/PASS -> ibond.thaibma.or.th session, then its data calls
3. (nothing)     -> a clear error telling you to export the file manually

NOTE ON ROUTE 2: iBond is a JavaScript single-page app and ThaiBMA does not publish
its internal endpoints. The candidate login/data URLs below are best-effort guesses;
if ThaiBMA changes them this will fail with a clear diagnostic rather than silently
returning wrong data. Use `--probe` to see exactly what each candidate returns, and
adjust IBOND_LOGIN_URLS / IBOND_DATA_URLS to whatever your browser's DevTools
Network tab shows when you view the yield curve while logged in.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 45
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

API_BASE = os.environ.get("THAIBMA_API_URL", "https://apiportal.thaibma.or.th")
API_CANDIDATES = [
    "/api/YieldCurve/GovernmentBondYieldCurve",
    "/api/yieldcurve/government",
    "/api/YieldCurve/TTM",
]
IBOND_BASE = "https://www.ibond.thaibma.or.th"
IBOND_LOGIN_URLS = [f"{IBOND_BASE}/api/account/login",
                    f"{IBOND_BASE}/api/auth/login",
                    f"{IBOND_BASE}/ors/api/account/login"]
IBOND_DATA_URLS = [f"{IBOND_BASE}/api/yieldcurve/government",
                   f"{IBOND_BASE}/api/market/yieldcurve",
                   f"{IBOND_BASE}/ors/api/yieldcurve"]


def _from_user_registry(name: str):
    """Read a persisted USER environment variable (HKCU\\Environment).

    `setx` writes there, but a terminal that was already open keeps the environment
    it was launched with — so a freshly stored credential looks 'missing' until you
    reopen the shell. Reading the registry directly removes that trap. The value is
    used in-process only; it is never printed or written anywhere.
    """
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, name)
            val = (val or "").strip()
            return val or None
    except Exception:
        return None


def _creds():
    """Credentials from the process environment, falling back to the persisted
    user environment so a freshly-run `setx` works without reopening the terminal."""
    out = []
    for name in ("THAIBMA_USER", "THAIBMA_PASS", "THAIBMA_API_KEY"):
        v = os.environ.get(name) or _from_user_registry(name)
        if v:
            os.environ[name] = v          # make it visible to child modules too
        out.append(v)
    return tuple(out)


def credentials_status() -> dict:
    """Safe for display: reports only WHETHER each secret is set, never its value."""
    u, p, k = _creds()
    return {"user_set": bool(u), "pass_set": bool(p), "api_key_set": bool(k),
            "user_hint": (u[:2] + "***") if u else "",
            "ready": bool(k) or bool(u and p)}


# ------------------------------------------------------------------ parsing --
def _to_tidy(payload) -> pd.DataFrame:
    """Normalise whatever JSON came back into [date, tau, yield]."""
    rows = payload
    if isinstance(payload, dict):
        for key in ("data", "result", "items", "yieldCurve", "curve"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    df = pd.json_normalize(rows)
    if df.empty:
        raise ValueError("empty payload")
    low = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for lc, orig in low.items():
                if n in lc:
                    return orig
        return None

    dcol = pick("asof", "date", "period", "month")
    tcol = pick("ttm", "tenor", "maturity", "term")
    ycol = pick("yield", "rate", "value")
    if not (dcol and tcol and ycol):
        raise ValueError(f"unrecognised columns: {list(df.columns)[:15]}")
    out = df[[dcol, tcol, ycol]].copy()
    out.columns = ["date", "tau", "yield"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["tau"] = pd.to_numeric(out["tau"], errors="coerce")
    out["yield"] = pd.to_numeric(out["yield"], errors="coerce")
    return out.dropna().sort_values(["date", "tau"]).reset_index(drop=True)


# ---------------------------------------------------------------- route 1 ----
def fetch_via_api(start=None, end=None) -> pd.DataFrame:
    _u, _p, key = _creds()
    if not key:
        raise RuntimeError("THAIBMA_API_KEY not set")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json",
                      "User-Agent": UA})
    errs = []
    for path in API_CANDIDATES:
        url = API_BASE.rstrip("/") + path
        try:
            r = s.get(url, params={"startDate": start, "endDate": end}, timeout=TIMEOUT)
            if r.status_code == 200:
                df = _to_tidy(r.json())
                df["source"] = f"thaibma-api:{path}"
                return df
            errs.append(f"{path} -> HTTP {r.status_code}")
        except Exception as ex:
            errs.append(f"{path} -> {type(ex).__name__}: {str(ex)[:70]}")
    raise RuntimeError("API route failed:\n  " + "\n  ".join(errs))


# ---------------------------------------------------------------- route 2 ----
def fetch_via_login(start=None, end=None, probe=False) -> pd.DataFrame:
    u, p, _k = _creds()
    if not (u and p):
        raise RuntimeError("THAIBMA_USER / THAIBMA_PASS not set")
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                      "Content-Type": "application/json", "Origin": IBOND_BASE,
                      "Referer": IBOND_BASE + "/"})
    try:
        s.get(IBOND_BASE + "/", timeout=TIMEOUT)          # pick up cookies
    except Exception:
        pass

    logged, notes = False, []
    for url in IBOND_LOGIN_URLS:
        for body in ({"username": u, "password": p},
                     {"userName": u, "password": p},
                     {"user": u, "pass": p}):
            try:
                r = s.post(url, data=json.dumps(body), timeout=TIMEOUT)
                notes.append(f"POST {url} -> {r.status_code}")
                if r.status_code == 200:
                    try:
                        j = r.json()
                        tok = (j.get("token") or j.get("accessToken")
                               or j.get("data", {}).get("token") if isinstance(j, dict) else None)
                        if tok:
                            s.headers["Authorization"] = f"Bearer {tok}"
                    except Exception:
                        pass
                    logged = True
                    break
            except Exception as ex:
                notes.append(f"POST {url} -> {type(ex).__name__}")
        if logged:
            break
    if probe:
        print("login attempts:")
        for n in notes:
            print("   ", n)
    if not logged:
        raise RuntimeError("iBond login did not succeed. Check the credentials you set in "
                           "the environment, and confirm the login endpoint with your "
                           "browser's DevTools ▸ Network tab.\n  " + "\n  ".join(notes))

    errs = []
    for url in IBOND_DATA_URLS:
        try:
            r = s.get(url, params={"startDate": start, "endDate": end}, timeout=TIMEOUT)
            if probe:
                print(f"GET {url} -> {r.status_code} ({len(r.content)} bytes)")
            if r.status_code == 200 and r.content:
                df = _to_tidy(r.json())
                df["source"] = f"ibond-login:{url.rsplit('/', 1)[-1]}"
                return df
            errs.append(f"{url} -> HTTP {r.status_code}")
        except Exception as ex:
            errs.append(f"{url} -> {type(ex).__name__}: {str(ex)[:70]}")
    raise RuntimeError("logged in, but no data endpoint answered. Open the yield-curve page "
                       "in your browser with DevTools ▸ Network, copy the request URL it "
                       "calls, and add it to IBOND_DATA_URLS.\n  " + "\n  ".join(errs))


# ------------------------------------------------------------------ public ---
def fetch_via_grpc(start=None, end=None) -> pd.DataFrame:
    """iBond's real protocol: gRPC-Web (see ibond_grpc.py). This is the primary route."""
    import ibond_grpc as ig
    start = start or "2022-01"
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    return ig.fetch_curve_history(ig.month_ends(start, end))


def fetch_curve(start=None, end=None, probe=False) -> pd.DataFrame:
    """gRPC-Web first (what iBond actually speaks), then the REST API if you
    subscribe, then the legacy JSON-login guess. Raises with guidance if all fail."""
    st = credentials_status()
    if not st["ready"]:
        raise RuntimeError(
            "No credentials found. Run:  python setup_credentials.py\n"
            "  (it reads your password with hidden input and stores it in your own\n"
            "   Windows profile — never in this app's code and never in a chat.)")
    errors = []
    if st["user_set"] and st["pass_set"]:
        try:
            return fetch_via_grpc(start, end)
        except Exception as ex:
            errors.append(f"[gRPC-Web] {ex}")
    if st["api_key_set"]:
        try:
            return fetch_via_api(start, end)
        except Exception as ex:
            errors.append(f"[REST API] {ex}")
    if st["user_set"] and st["pass_set"]:
        try:
            return fetch_via_login(start, end, probe=probe)
        except Exception as ex:
            errors.append(f"[legacy JSON login] {ex}")
    raise RuntimeError("could not download the curve.\n\n" + "\n\n".join(errors) +
                       "\n\nFallback: export the curve manually from ThaiBMA ▸ Market ▸ "
                       "Yield Curve ▸ Government and load the file in the Yield Curve menu.")


def fetch_and_store(start=None, end=None, probe=False):
    """Download, run the DNS decomposition, and save everything to SQLite."""
    import yield_curve_dns as ycd
    curve = fetch_curve(start, end, probe=probe)
    factors = ycd.fit_dns(curve)
    if factors.empty:
        raise RuntimeError("downloaded data had fewer than 4 tenors per date")
    val = ycd.validate(factors)
    fc = ycd.forecast_factors(factors)
    ycd.save_to_sqlite(curve, factors, val, fc)
    return curve, factors, val


def main():
    if "--status" in sys.argv or len(sys.argv) == 1:
        st = credentials_status()
        print("credential status (values are never shown):")
        for k, v in st.items():
            print(f"  {k:12s}: {v}")
        if not st["ready"]:
            print('\nSet them with:  setx THAIBMA_USER "..."   and   setx THAIBMA_PASS "..."')
            print("Then open a NEW terminal.")
        return
    start = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else None
    end = sys.argv[sys.argv.index("--end") + 1] if "--end" in sys.argv else str(date.today())
    probe = "--probe" in sys.argv
    curve, factors, val = fetch_and_store(start, end, probe=probe)
    print(f"downloaded {len(curve):,} rows | {curve['tau'].nunique()} tenors | "
          f"{factors['date'].min():%Y-%m} .. {factors['date'].max():%Y-%m}")
    print(val.to_string(index=False))
    print("saved to SQLite (dns_curve / dns_factors / dns_summary).")


if __name__ == "__main__":
    main()
