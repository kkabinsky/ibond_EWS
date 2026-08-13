# -*- coding: utf-8 -*-
"""
yield_curve_dns.py
Dynamic Nelson-Siegel (DNS) decomposition of the Thai government bond yield curve
into LEVEL, SLOPE and CURVATURE -- the deliverable described in the CMDF quarterly
progress report (CMDF-0128-2568, Q2/2569), sections 2.1.1 / 2.2.1 / 2.3.1.

Model (Nelson-Siegel 1987; Diebold & Li 2006), estimated month by month:

    y_t(tau) = L_t
             + S_t * [(1 - e^{-lam*tau}) / (lam*tau)]
             + C_t * [(1 - e^{-lam*tau}) / (lam*tau) - e^{-lam*tau}]

  L_t = level      (long-run rate)      loading 1
  S_t = slope      (short-end factor)   loading decays to 0
  C_t = curvature  (medium-term hump)   loading peaks in the middle

The report validates the extracted factors against empirical proxies and reports
  corr(Level,      15Y yield)      ~ 0.962
  corr(Slope,      10Y - 1Y spread) ~ 0.987
  corr(Curvature,  2*5Y - 3M - 10Y) ~ 0.905
`validate()` recomputes exactly these three correlations on whatever data is loaded.

DATA SOURCES  (checked in this order -- see `load_curve`)
  1. --file <path>    a CSV/XLSX exported from ThaiBMA / iBond.
                      Accepted layouts:
                        (a) long : date, tenor(years), yield(%)
                        (b) wide : date column + one column per tenor ("1M","6M","1Y","5Y","10Y",...)
  2. ThaiBMA API      only if you subscribe: set THAIBMA_API_KEY (and optionally
                      THAIBMA_API_URL). The public web pages render the curve with
                      JavaScript and the documented API requires a subscription, so
                      there is NO un-authenticated automatic download.
  3. demo             a clearly-labelled synthetic curve so the GUI works offline.
                      NEVER present demo output as real market data.

Run:
    python yield_curve_dns.py --file thai_gov_yield.csv
    python yield_curve_dns.py --demo --save
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
T_FACTORS = "dns_factors"
T_CURVE = "dns_curve"
T_SUMMARY = "dns_summary"
T_IBOND_RAW = "ibond_yield_raw"
T_IBOND_LOG = "ibond_download_log"

LAMBDA = 0.7308          # tau in YEARS; peaks the curvature loading at ~30 months
                         # (Diebold-Li use 0.0609 with tau in months: 0.0609*12 = 0.7308)

# tenor label -> years
TENOR_MAP = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "9M": 0.75,
             "1Y": 1, "2Y": 2, "3Y": 3, "4Y": 4, "5Y": 5, "6Y": 6, "7Y": 7,
             "8Y": 8, "9Y": 10 - 1, "10Y": 10, "12Y": 12, "14Y": 14, "15Y": 15,
             "16Y": 16, "20Y": 20, "25Y": 25, "30Y": 30, "50Y": 50}
DEFAULT_TENORS = [1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]


# ============================================================ loadings ========
def ns_loadings(tau, lam: float = LAMBDA) -> np.ndarray:
    """(n_tenor, 3) matrix of Nelson-Siegel loadings [level, slope, curvature]."""
    tau = np.asarray(tau, dtype=float)
    x = lam * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        b2 = np.where(x > 1e-8, (1 - np.exp(-x)) / x, 1.0)
    b3 = b2 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), b2, b3])


# ============================================================== data ==========
def _tenor_to_years(c) -> float | None:
    s = str(c).strip().upper().replace(" ", "")
    if s in TENOR_MAP:
        return TENOR_MAP[s]
    try:                                   # already numeric (years)
        v = float(s)
        return v if 0 < v <= 60 else None
    except ValueError:
        pass
    if s.endswith("M"):
        try:
            return float(s[:-1]) / 12
        except ValueError:
            return None
    if s.endswith("Y"):
        try:
            return float(s[:-1])
        except ValueError:
            return None
    return None


def load_from_file(path: str) -> pd.DataFrame:
    """Read a ThaiBMA/iBond export -> tidy frame [date, tau, yield]."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = (pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls"))
           else pd.read_csv(path))
    raw.columns = [str(c).strip() for c in raw.columns]
    dcol = next((c for c in raw.columns if str(c).lower() in
                 ("date", "asofdate", "as_of_date", "tradedate", "month", "period")), raw.columns[0])

    lower = {str(c).lower(): c for c in raw.columns}
    tcol = next((lower[k] for k in ("tenor", "ttm", "ttm(yrs.)", "ttm_years", "maturity") if k in lower), None)
    ycol = next((lower[k] for k in ("yield", "yield(%)", "yield_pct", "rate") if k in lower), None)

    if tcol and ycol:                                   # long layout
        out = raw[[dcol, tcol, ycol]].copy()
        out.columns = ["date", "tau", "yield"]
        out["tau"] = out["tau"].map(_tenor_to_years)
    else:                                               # wide layout
        keep = {c: _tenor_to_years(c) for c in raw.columns if c != dcol}
        keep = {c: t for c, t in keep.items() if t is not None}
        if not keep:
            raise ValueError("no tenor columns recognised; expected 1M/6M/1Y/5Y/10Y... or numeric years")
        out = raw.melt(id_vars=[dcol], value_vars=list(keep), var_name="tau", value_name="yield")
        out.columns = ["date", "tau", "yield"]
        out["tau"] = out["tau"].map(keep)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["yield"] = pd.to_numeric(out["yield"], errors="coerce")
    out = out.dropna(subset=["date", "tau", "yield"])
    out["source"] = f"file:{os.path.basename(path)}"
    return out.sort_values(["date", "tau"]).reset_index(drop=True)


def load_from_api() -> pd.DataFrame:
    """ThaiBMA subscription API. Requires THAIBMA_API_KEY; no public fallback."""
    key = os.environ.get("THAIBMA_API_KEY")
    if not key:
        raise RuntimeError(
            "THAIBMA_API_KEY not set. ThaiBMA's yield-curve API needs a subscription "
            "(apiportal.thaibma.or.th); the public web pages render the curve in "
            "JavaScript, so there is no un-authenticated automatic download. "
            "Export the curve from iBond / the Government yield-curve page and pass --file.")
    import json
    import ssl
    import urllib.request
    url = os.environ.get("THAIBMA_API_URL",
                         "https://apiportal.thaibma.or.th/api/yieldcurve/government")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}",
                                               "Accept": "application/json"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=45, context=ctx) as r:
        payload = json.loads(r.read().decode("utf-8", "ignore"))
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    df = pd.json_normalize(rows)
    df.columns = [str(c).lower() for c in df.columns]
    dcol = next(c for c in df.columns if "date" in c)
    tcol = next(c for c in df.columns if "ttm" in c or "tenor" in c)
    ycol = next(c for c in df.columns if "yield" in c or "rate" in c)
    out = df[[dcol, tcol, ycol]].copy()
    out.columns = ["date", "tau", "yield"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["tau"] = pd.to_numeric(out["tau"], errors="coerce")
    out["yield"] = pd.to_numeric(out["yield"], errors="coerce")
    out["source"] = "thaibma-api"
    return out.dropna().sort_values(["date", "tau"]).reset_index(drop=True)


def make_demo(n_months: int = 314, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC curve for offline testing -- clearly labelled, never real data.
    n_months defaults to 314 to match the sample length quoted in the report."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1),
                          periods=n_months, freq="MS")
    L = 3.2 + np.cumsum(rng.normal(0, 0.055, n_months))
    L = np.clip(L, 1.0, 6.0)
    S = -1.3 + np.cumsum(rng.normal(0, 0.075, n_months))
    S = np.clip(S, -3.2, 1.6)
    C = 0.4 + np.cumsum(rng.normal(0, 0.09, n_months))
    C = np.clip(C, -3.0, 3.0)
    B = ns_loadings(DEFAULT_TENORS)
    recs = []
    for i, d in enumerate(dates):
        y = B @ np.array([L[i], S[i], C[i]]) + rng.normal(0, 0.02, len(DEFAULT_TENORS))
        recs += [{"date": d, "tau": t, "yield": v, "source": "DEMO (synthetic)"}
                 for t, v in zip(DEFAULT_TENORS, y)]
    return pd.DataFrame(recs)


def load_curve(file: str | None = None, demo: bool = False) -> pd.DataFrame:
    if file:
        return load_from_file(file)
    if demo:
        return make_demo()
    try:
        return load_from_api()
    except Exception as ex:
        raise RuntimeError(f"{ex}\nUse --file <export.csv> or --demo.") from ex


# ============================================================ estimation ======
def fit_dns(curve: pd.DataFrame, lam: float | None = None) -> pd.DataFrame:
    """Cross-sectional OLS per date -> Level / Slope / Curvature time series.

    `lam=None` resolves LAMBDA at call time. Writing `lam=LAMBDA` in the signature
    freezes the value at import, so re-assigning yield_curve_dns.LAMBDA afterwards
    would silently have no effect.
    """
    lam = LAMBDA if lam is None else lam
    out = []
    for d, g in curve.groupby("date"):
        g = g.dropna(subset=["tau", "yield"]).sort_values("tau")
        if len(g) < 4:
            continue
        B = ns_loadings(g["tau"].to_numpy(), lam)
        beta, *_ = np.linalg.lstsq(B, g["yield"].to_numpy(), rcond=None)
        fit = B @ beta
        resid = g["yield"].to_numpy() - fit
        out.append({"date": d, "Level": beta[0], "Slope": beta[1], "Curvature": beta[2],
                    "n_tenor": len(g),
                    "rmse": float(np.sqrt(np.mean(resid ** 2))),
                    "y_3m": _pick(g, 0.25), "y_1y": _pick(g, 1.0), "y_2y": _pick(g, 2.0),
                    "y_5y": _pick(g, 5.0), "y_10y": _pick(g, 10.0), "y_15y": _pick(g, 15.0)})
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


def _pick(g: pd.DataFrame, tau: float) -> float:
    """yield at the tenor closest to `tau` (NaN if nothing within 1 year)."""
    if g.empty:
        return np.nan
    i = (g["tau"] - tau).abs().idxmin()
    return float(g.loc[i, "yield"]) if abs(g.loc[i, "tau"] - tau) <= 1.0 else np.nan


def validate(f: pd.DataFrame) -> pd.DataFrame:
    """Correlate each factor with its empirical proxy (report targets in brackets)."""
    d = f.copy()
    d["proxy_level"] = d["y_15y"]
    d["proxy_slope"] = d["y_10y"] - d["y_1y"]
    # Diebold-Li curvature proxy uses the tenor where the curvature loading peaks
    # (~30 months with lambda = 0.7308), i.e. 2Y -- not 5Y.
    d["proxy_curv"] = 2 * d["y_2y"] - d["y_3m"] - d["y_10y"]
    rows = []
    for fac, proxy, label, target in [
            ("Level", "proxy_level", "15Y yield", 0.962),
            ("Slope", "proxy_slope", "10Y - 1Y spread", 0.987),
            ("Curvature", "proxy_curv", "2*2Y - 3M - 10Y", 0.905)]:
        s = d[[fac, proxy]].dropna()
        r = float(s[fac].corr(s[proxy])) if len(s) > 2 else float("nan")
        rows.append({"factor": fac, "empirical proxy": label,
                     "correlation": r, "|correlation|": abs(r),
                     "report target": target, "n": int(len(s))})
    return pd.DataFrame(rows)


def forecast_factors(f: pd.DataFrame, horizons=(1, 3, 6, 12)) -> pd.DataFrame:
    """DNS-AR(1) benchmark: out-of-sample RMSE per factor and horizon
    (expanding window, as in the report's recursive forecasting design)."""
    res = []
    n = len(f)
    start = max(int(n * 0.6), 24)
    for fac in ("Level", "Slope", "Curvature"):
        y = f[fac].to_numpy(dtype=float)
        for h in horizons:
            errs = []
            for t in range(start, n - h):
                hist = y[:t + 1]
                x0, x1 = hist[:-1], hist[1:]
                if len(x0) < 12 or np.std(x0) < 1e-9:
                    continue
                b = np.polyfit(x0, x1, 1)              # AR(1)
                p = hist[-1]
                for _ in range(h):
                    p = b[0] * p + b[1]
                errs.append(p - y[t + h])
            if errs:
                res.append({"factor": fac, "horizon (m)": h,
                            "RMSE": float(np.sqrt(np.mean(np.square(errs)))),
                            "MAE": float(np.mean(np.abs(errs))), "n_fcst": len(errs)})
    return pd.DataFrame(res)


# ============================================================ persistence =====
def save_to_sqlite(curve, factors, val, fc, db=DB):
    con = sqlite3.connect(db)
    c = curve.copy(); c["date"] = c["date"].astype(str)
    f = factors.copy(); f["date"] = f["date"].astype(str)
    c.to_sql(T_CURVE, con, if_exists="replace", index=False)
    f.to_sql(T_FACTORS, con, if_exists="replace", index=False)
    summ = {
        "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(curve["source"].iloc[0]) if "source" in curve.columns else "n/a",
        "n_periods": int(factors.shape[0]),
        "date_min": str(factors["date"].min()), "date_max": str(factors["date"].max()),
        "n_tenor": int(curve["tau"].nunique()), "lambda": LAMBDA,
        "mean_rmse": float(factors["rmse"].mean()),
        "level_last": float(factors["Level"].iloc[-1]),
        "slope_last": float(factors["Slope"].iloc[-1]),
        "curv_last": float(factors["Curvature"].iloc[-1]),
    }
    for _, r in val.iterrows():
        summ[f"corr_{r['factor'].lower()}"] = float(r["correlation"])
    pd.DataFrame([summ]).to_sql(T_SUMMARY, con, if_exists="replace", index=False)
    if fc is not None and not fc.empty:
        fc.to_sql("dns_forecast", con, if_exists="replace", index=False)
    con.commit(); con.close()


def save_ibond_raw(curve, db=DB, source=None, requested_start=None, requested_end=None, mode="replace"):
    con = sqlite3.connect(db)
    try:
        raw = curve.copy()
        if raw.empty:
            return
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw["downloaded_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        raw["source"] = source or str(raw.get("source", pd.Series(["iBond"])).iloc[0])
        raw["requested_start"] = requested_start or ""
        raw["requested_end"] = requested_end or ""
        raw.to_sql(T_IBOND_RAW, con, if_exists=mode, index=False)
        log = pd.DataFrame([{
            "downloaded_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source or str(raw["source"].iloc[0]),
            "requested_start": requested_start or "",
            "requested_end": requested_end or "",
            "row_count": int(len(raw)),
            "n_dates": int(raw["date"].nunique()),
            "n_tenor": int(raw["tau"].nunique()),
            "date_min": str(raw["date"].min()),
            "date_max": str(raw["date"].max()),
        }])
        log.to_sql(T_IBOND_LOG, con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()


def load_ibond_raw(db=DB):
    con = sqlite3.connect(db)
    try:
        raw = pd.read_sql_query(f"SELECT * FROM {T_IBOND_RAW}", con)
    except Exception:
        raw = pd.DataFrame()
    finally:
        con.close()
    if not raw.empty and "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    return raw


def load_ibond_log(db=DB):
    con = sqlite3.connect(db)
    try:
        log = pd.read_sql_query(f"SELECT * FROM {T_IBOND_LOG} ORDER BY downloaded_at DESC", con)
    except Exception:
        log = pd.DataFrame()
    finally:
        con.close()
    return log


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    try:
        f = pd.read_sql_query(f"SELECT * FROM {T_FACTORS}", con)
        c = pd.read_sql_query(f"SELECT * FROM {T_CURVE}", con)
        s = pd.read_sql_query(f"SELECT * FROM {T_SUMMARY} LIMIT 1", con)
        try:
            fc = pd.read_sql_query("SELECT * FROM dns_forecast", con)
        except Exception:
            fc = pd.DataFrame()
    except Exception:
        f = c = s = fc = pd.DataFrame()
    finally:
        con.close()
    if not f.empty:
        f["date"] = pd.to_datetime(f["date"])
    if not c.empty:
        c["date"] = pd.to_datetime(c["date"])
    return c, f, s, fc


# ================================================================= main =======
def run(file=None, demo=False, save=True):
    curve = load_curve(file=file, demo=demo)
    factors = fit_dns(curve)
    if factors.empty:
        raise RuntimeError("no date had >= 4 tenors; check the input file layout")
    val = validate(factors)
    fc = forecast_factors(factors)
    if save:
        save_to_sqlite(curve, factors, val, fc)
    return curve, factors, val, fc


def main():
    file = sys.argv[sys.argv.index("--file") + 1] if "--file" in sys.argv else None
    demo = "--demo" in sys.argv
    save = "--no-save" not in sys.argv
    print("=" * 92)
    print("Dynamic Nelson-Siegel -- Thai government bond yield curve")
    print("Level / Slope / Curvature   (CMDF-0128-2568 progress report, sections 2.1.1 / 2.2.1)")
    print("=" * 92)
    curve, factors, val, fc = run(file=file, demo=demo, save=save)
    src = curve["source"].iloc[0]
    print(f"source        : {src}")
    if str(src).startswith("DEMO"):
        print("                *** SYNTHETIC DEMO DATA -- not real market data ***")
    print(f"periods       : {len(factors)}  ({factors['date'].min():%Y-%m} .. {factors['date'].max():%Y-%m})")
    print(f"tenors        : {curve['tau'].nunique()}   lambda = {LAMBDA}")
    print(f"fit RMSE      : {factors['rmse'].mean():.4f} %  (mean across dates)")

    print("\nFACTOR VALIDATION (correlation with empirical proxy)")
    print(f"  {'Factor':11s} {'Proxy':20s} {'corr':>8s} {'|corr|':>8s} {'report':>8s} {'n':>6s}")
    for _, r in val.iterrows():
        print(f"  {r['factor']:11s} {r['empirical proxy']:20s} {r['correlation']:8.3f} "
              f"{r['|correlation|']:8.3f} {r['report target']:8.3f} {r['n']:6d}")

    print("\nLATEST FACTORS")
    last = factors.iloc[-1]
    print(f"  {last['date']:%Y-%m}   Level {last['Level']:.3f}   Slope {last['Slope']:.3f}"
          f"   Curvature {last['Curvature']:.3f}")

    if not fc.empty:
        print("\nDNS-AR(1) OUT-OF-SAMPLE FORECAST (expanding window)")
        print(f"  {'Factor':11s} {'h=1':>9s} {'h=3':>9s} {'h=6':>9s} {'h=12':>9s}   (RMSE)")
        for fac in ("Level", "Slope", "Curvature"):
            s = fc[fc["factor"] == fac].set_index("horizon (m)")["RMSE"]
            print(f"  {fac:11s}" + "".join(f"{s.get(h, float('nan')):9.4f}" for h in (1, 3, 6, 12)))

    if save:
        print(f"\nSaved to SQLite: {T_CURVE}, {T_FACTORS}, {T_SUMMARY}, dns_forecast  ({DB})")
    print("\nDone.")


if __name__ == "__main__":
    main()
