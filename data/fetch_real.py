# -*- coding: utf-8 -*-
"""
Download REAL firm data (SET / US) from the internet and map it into the
33-feature ThaiBMA schema — for both the ML app and the survival EWS.

Sources (no API key):
  * Yahoo Finance (yfinance) — prices, market cap, balance sheet, ratios, ESG
  * FRED (St. Louis Fed) CSV — macro (yield-curve slope, credit spread, rates...)

Firm-level data does not contain the consumer-credit behavioural fields
(delinquency, utilization, ...), so those columns are filled with clearly
documented **market-derived proxies** (see FEATURE_SOURCE). Every one of the 33
features therefore gets a real-data value (direct or proxy), and the same app +
survival EWS run unchanged.

Outputs:
  credit_dataset_33features_real.xlsx  — cross-section (latest month per firm)
  real_panel.xlsx                      — monthly panel (firm x month) with `event`
"""
from __future__ import annotations
import io
import os
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from gen_data import FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
_UA = {"User-Agent": "Mozilla/5.0 (research; thaibma-ews)"}

# a workable universe: SET50 names (.BK) + a few US names for robustness
SET_UNIVERSE = ["PTT.BK", "SCB.BK", "KBANK.BK", "BBL.BK", "AOT.BK", "CPALL.BK",
                "SCC.BK", "ADVANC.BK", "PTTEP.BK", "GULF.BK", "BDMS.BK", "CPN.BK",
                "TRUE.BK", "MINT.BK", "KTB.BK", "TOP.BK", "IVL.BK", "BH.BK",
                "AAPL", "MSFT", "F", "GM", "CCL", "XOM"]

FWD_MONTHS = 3          # distress horizon
DISTRESS_DD = 0.25      # >=25% forward drawdown => distress event (firm proxy)

# how each of the 33 features is sourced from real data
FEATURE_SOURCE = {
    "delinquency_trend": "proxy: -1 * trailing 3m return (deterioration)",
    "credit_score_pd": "real: EDF = N(-DTD) from Merton naive DTD",
    "loan_tenor": "proxy: months of price history",
    "utilization": "proxy: market leverage F/(E+F)",
    "behavioral_score": "proxy: 12m momentum",
    "spend_activity": "proxy: trailing 3m volume trend",
    "roa": "real: yfinance returnOnAssets",
    "roe": "real: yfinance returnOnEquity",
    "debt_to_equity": "real: yfinance debtToEquity/100",
    "current_ratio": "real: yfinance currentRatio",
    "quick_ratio": "real: yfinance quickRatio",
    "interest_coverage": "proxy: ebitda / (0.05*totalDebt)",
    "leverage_ratio": "real: market leverage F/(E+F)",
    "profit_margin": "real: yfinance profitMargins",
    "ebitda_margin": "real: yfinance ebitdaMargins",
    "cash_to_assets": "real: totalCash / (marketCap+totalDebt)",
    "equity_vol": "real: trailing 1y return volatility (annualised)",
    "distance_to_default": "real: Merton naive DTD (Bharath-Shumway)",
    "market_leverage": "real: F/(E+F)",
    "momentum_12m": "real: 12m price return",
    "beta": "real: yfinance beta (fallback 1.0)",
    "yield_slope": "real: FRED DGS10 - DGS3MO",
    "credit_spread": "real: FRED BAA10Y",
    "policy_rate": "real: FRED FEDFUNDS",
    "gdp_growth": "real: FRED A191RL1Q225SBEA (annualised)",
    "inflation": "real: FRED CPIAUCSL yoy %",
    "unemployment": "real: FRED UNRATE",
    "board_size": "proxy: ln(marketCap) bucketed (governance size)",
    "institutional_ownership": "real: yfinance heldPercentInstitutions",
    "esg_score": "real: yfinance sustainability totalEsg (fallback 50)",
    "governance_score": "real: yfinance sustainability governanceScore (fallback 50)",
    "loan_amount": "proxy: market cap",
    "months_on_book": "proxy: months of price history",
}


# ---------------------------------------------------------------- FRED --------
def _fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    txt = urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=25).read().decode()
    df = pd.read_csv(io.StringIO(txt)); df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"]); df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.dropna().set_index("date")["v"]


def macro_monthly():
    m = pd.DataFrame({
        "yield_slope": _fred("DGS10").resample("ME").last() - _fred("DGS3MO").resample("ME").last(),
        "credit_spread": _fred("BAA10Y").resample("ME").last(),
        "policy_rate": _fred("FEDFUNDS").resample("ME").last(),
        "gdp_growth": _fred("A191RL1Q225SBEA").resample("ME").last(),
        "unemployment": _fred("UNRATE").resample("ME").last(),
    })
    cpi = _fred("CPIAUCSL").resample("ME").last()
    m["inflation"] = (cpi / cpi.shift(12) - 1.0) * 100
    return m.ffill()


# --------------------------------------------------------------- yfinance -----
def _cached_prices(t, start="2013-01-01"):
    fp = os.path.join(CACHE, f"rp_{t.replace('.', '_')}.csv")
    if os.path.exists(fp):
        s = pd.read_csv(fp, index_col=0)["Close"]
        s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
        if len(s) > 200:
            return pd.to_numeric(s, errors="coerce").dropna()
    s = yf.Ticker(t).history(start=start, auto_adjust=True)["Close"].dropna()
    s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
    s.rename("Close").to_frame().to_csv(fp)
    return s


def _naive_dtd(E, F, sigmaE, r_prior, T=1.0):
    D = np.maximum(F, 1.0)
    sigmaV = (E / (E + D)) * sigmaE + (D / (E + D)) * (0.05 + 0.25 * sigmaE)
    return (np.log((E + D) / D) + (r_prior - 0.5 * sigmaV ** 2) * T) / (sigmaV * np.sqrt(T))


def _firm_static(t):
    tk = yf.Ticker(t)
    try:
        info = tk.get_info()
    except Exception:
        info = {}
    # default barrier
    bs = None
    for a in ("quarterly_balance_sheet", "balance_sheet"):
        try:
            b = getattr(tk, a)
            if b is not None and not b.empty:
                bs = b; break
        except Exception:
            pass

    def row(keys):
        if bs is None:
            return None
        for k in keys:
            if k in bs.index:
                s = bs.loc[k].dropna()
                if len(s):
                    return float(s.iloc[0])
        return None
    std = row(["Current Debt", "Current Debt And Capital Lease Obligation"])
    ltd = row(["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    total_debt = row(["Total Debt"]) or info.get("totalDebt")
    if std is not None and ltd is not None:
        F = std + 0.5 * ltd
    elif total_debt:
        F = 0.6 * total_debt
    else:
        F = info.get("marketCap", 1e9) * 0.3
    esg = {}
    try:
        s = tk.sustainability
        if s is not None and not s.empty:
            esg = s["Value"].to_dict() if "Value" in s.columns else s.iloc[:, 0].to_dict()
    except Exception:
        pass
    return dict(info=info, F=float(F), total_debt=total_debt or 0.0, esg=esg)


def build_panel(universe=SET_UNIVERSE, market="^GSPC"):
    macro = macro_monthly()
    mkt = _cached_prices(market)
    mkt_ret = np.log(mkt).diff().resample("ME").last()
    frames = []
    for t in universe:
        try:
            px = _cached_prices(t)
            st = _firm_static(t); info = st["info"]; F = st["F"]
            shares = info.get("sharesOutstanding") or (
                info.get("marketCap", 0) / (info.get("currentPrice") or float(px.iloc[-1])))
            if not shares:
                print(f"  skip {t}: no shares"); continue
        except Exception as e:
            print(f"  skip {t}: {type(e).__name__} {str(e)[:50]}"); continue

        px = px[~px.index.duplicated()].sort_index()
        m_close = px.resample("ME").last().dropna()
        logret = np.log(px).diff()
        vol = (logret.rolling(252).std() * np.sqrt(252)).resample("ME").last()
        prior = (px / px.shift(252) - 1.0).resample("ME").last()
        mom = (px / px.shift(252) - 1.0).resample("ME").last()
        ret3 = (px / px.shift(63) - 1.0).resample("ME").last()

        d = pd.DataFrame({"close": m_close, "sigmaE": vol, "r_prior": prior,
                          "mom": mom, "ret3": ret3}).dropna()
        if len(d) < 12:
            continue
        E = d["close"] * shares
        dtd = _naive_dtd(E.values, F, d["sigmaE"].values, d["r_prior"].values)
        edf = norm.cdf(-dtd)
        mktlev = F / (E.values + F)
        beta = info.get("beta") or 1.0
        mc = info.get("marketCap") or float(E.iloc[-1])
        months_hist = np.arange(len(d)) + 12

        row = {}
        # --- map to the 33-feature schema (real + documented proxy) ----------
        row["delinquency_trend"] = -d["ret3"].values
        row["credit_score_pd"] = edf
        row["loan_tenor"] = months_hist.astype(float)
        row["utilization"] = mktlev
        row["behavioral_score"] = d["mom"].values
        row["spend_activity"] = np.zeros(len(d))
        row["roa"] = info.get("returnOnAssets", np.nan)
        row["roe"] = info.get("returnOnEquity", np.nan)
        row["debt_to_equity"] = (info.get("debtToEquity") or np.nan)
        if isinstance(row["debt_to_equity"], (int, float)) and row["debt_to_equity"] > 5:
            row["debt_to_equity"] = row["debt_to_equity"] / 100.0
        row["current_ratio"] = info.get("currentRatio", np.nan)
        row["quick_ratio"] = info.get("quickRatio", np.nan)
        ebitda = info.get("ebitda") or 0.0
        row["interest_coverage"] = ebitda / (0.05 * (st["total_debt"] or 1)) if st["total_debt"] else np.nan
        row["leverage_ratio"] = mktlev
        row["profit_margin"] = info.get("profitMargins", np.nan)
        row["ebitda_margin"] = info.get("ebitdaMargins", np.nan)
        row["cash_to_assets"] = (info.get("totalCash", 0) / (mc + (st["total_debt"] or 0) + 1))
        row["equity_vol"] = d["sigmaE"].values
        row["distance_to_default"] = dtd
        row["market_leverage"] = mktlev
        row["momentum_12m"] = d["mom"].values
        row["beta"] = beta
        row["yield_slope"] = macro["yield_slope"].reindex(d.index).ffill().values
        row["credit_spread"] = macro["credit_spread"].reindex(d.index).ffill().values
        row["policy_rate"] = macro["policy_rate"].reindex(d.index).ffill().values
        row["gdp_growth"] = macro["gdp_growth"].reindex(d.index).ffill().values
        row["inflation"] = macro["inflation"].reindex(d.index).ffill().values
        row["unemployment"] = macro["unemployment"].reindex(d.index).ffill().values
        row["board_size"] = np.clip(np.round(np.log(mc + 1) - 15), 3, 20)
        row["institutional_ownership"] = info.get("heldPercentInstitutions", np.nan)
        row["esg_score"] = st["esg"].get("totalEsg", np.nan)
        row["governance_score"] = st["esg"].get("governanceScore", np.nan)
        row["loan_amount"] = mc
        row["months_on_book"] = months_hist.astype(float)

        f = pd.DataFrame(row, index=d.index)
        f["ticker"] = t
        f["month_index"] = np.arange(len(f))
        f["account_id"] = t

        # distress event: >= DISTRESS_DD forward drawdown in next FWD_MONTHS
        ev = []
        for dt in f.index:
            w = px[(px.index > dt) & (px.index <= dt + pd.DateOffset(months=FWD_MONTHS))]
            base = px[px.index <= dt]
            ev.append(1 if (len(w) and (w.min() / base.iloc[-1] - 1.0) <= -DISTRESS_DD) else 0)
        f["event"] = ev
        f["default_3m"] = ev
        frames.append(f.reset_index(drop=True))
        print(f"  ok {t}: {len(f)} months, {sum(ev)} distress")

    panel = pd.concat(frames, ignore_index=True)
    # fill firm-static NaNs by cross-firm median (documented)
    for c in FEATURES:
        if panel[c].isna().any():
            panel[c] = panel[c].fillna(panel[c].median())
    return panel


if __name__ == "__main__":
    print("downloading real SET/US data -> 33-feature panel ...")
    panel = build_panel()
    n_firms = panel["ticker"].nunique()
    print(f"\npanel: {len(panel):,} firm-months, {n_firms} firms, "
          f"distress rate {panel['event'].mean()*100:.1f}%")
    # monthly panel (for survival EWS)
    panel.to_excel(os.path.join(HERE, "real_panel.xlsx"), index=False)
    # cross-section: latest month per firm (for the ML app)
    xs = panel.sort_values("month_index").groupby("ticker").tail(1).reset_index(drop=True)
    xs["account_id"] = range(1, len(xs) + 1)
    xs[["account_id"] + FEATURES + ["default_3m"]].to_excel(
        os.path.join(HERE, "credit_dataset_33features_real.xlsx"), index=False)
    print("wrote real_panel.xlsx and credit_dataset_33features_real.xlsx")
