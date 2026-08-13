# -*- coding: utf-8 -*-
"""
bond_ews.py -- Approach 1 (discrete-time survival hazard) applied directly to the
iBond corporate-bond data. No SET / equity data is used anywhere in this module.

PIPELINE  (the same five steps as survivor2.py, but on bond-level inputs)

    1. enrich_universe()  every registered issue -> issuer, sector, rating,
                          maturity, coupon                (bond.Get + GetBondFeature)
    2. build_panel()      issuer x month panel with time-varying features
    3. fit_hazard()       pooled logistic  h(t|X) = sigmoid(b(t) + B.X)
    4. add_signals()      PD_3M = 1 - prod(1-h),  Momentum = PD(t)/PD(t-1),
                          hyperbolic alarm  M >= K / PD_prev^alpha
    5. lead_time()        actionable 1-3M lead time plus persistent alarm
                          duration against the real ThaiBMA default date

FEATURES (all derived from iBond only, all known at time t)
    n_outstanding      issues not yet matured
    mat_3m / 6m / 12m  issues maturing within the next 3 / 6 / 12 months
                       -> refinancing pressure, the main bond-level distress channel
    coupon_avg / max   higher coupon = the market priced the issuer as riskier
    coupon_spread      issuer coupon minus the same-month market median
    rating_num         numeric mapping of the latest issue rating (0 = unrated)
    months_since_issue how long since the issuer last came to market
    issue_age_avg      average age of outstanding issues

HONEST LIMITATIONS
    * iBond exposes no outstanding AMOUNT through GetAllBond, so refinancing
      pressure is a COUNT of issues, not a value. A large and a tiny issue count
      the same.
    * Only 10 issuers in the register have ever defaulted, so the fitted model is
      estimated on very few positive events. Treat every metric as indicative.
    * The default register covers issues currently in the system, not the full
      market history.

RUN
    python bond_ews.py                # use the cached enriched universe
    python bond_ews.py --refresh      # re-download and re-enrich (~3 min)
    python bond_ews.py --no-save
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

import lead_metrics
from thaibma_paths import DATA_ROOT  # data lives outside the repo

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")

T_UNIVERSE = "bond_ews_universe"
T_PANEL = "bond_ews_panel"
T_ALERT = "bond_ews_alert"
T_LEAD = "bond_ews_leadtime"
T_SUMMARY = "bond_ews_summary"

BOND_SVC = "/grpc/bond-grpc/bond.BondGrpcService"

ALARM_BUDGET = 0.10      # alarm on the riskiest 10% of issuer-months
P_STAR = None            # resolved at run time from ALARM_BUDGET (see add_signals)
K_BOUND, ALPHA_BOUND = 0.35, 0.55       # hyperbolic boundary M >= K / PD^alpha
DAYS_PER_MONTH = 30.4368
MIN_MONTHS = 6           # issuer must have this many months of history

RATING_MAP = {
    "AAA": 10, "AA+": 9.7, "AA": 9.3, "AA-": 9.0,
    "A+": 8.7, "A": 8.3, "A-": 8.0,
    "BBB+": 7.0, "BBB": 6.5, "BBB-": 6.0,
    "BB+": 5.0, "BB": 4.5, "BB-": 4.0,
    "B+": 3.0, "B": 2.5, "B-": 2.0,
    "CCC+": 1.5, "CCC": 1.2, "CC": 1.0, "C": 0.8, "D": 0.0,
}

BASE_FEATURES = ["n_outstanding", "mat_3m", "mat_6m", "mat_12m",
                 "coupon_avg", "coupon_max", "coupon_spread",
                 "rating_num", "months_since_issue", "issue_age_avg"]

# Year-on-year CHANGES are what actually carry out-of-sample signal. Levels alone
# score 0.48 leave-one-issuer-out (no better than chance); adding these lifts it to
# 0.70. Economically that is the deterioration path: an issuer in trouble has to pay
# a rising coupon and comes to market less often.
TREND_FEATURES = ["coupon_d12", "spread_d12", "n_out_d12"]

# Yield-curve state from yield_curve_dns.py (dns_factors), merged on month. These are
# macro: identical for every issuer in a given month. That is exactly why they were
# checked for being a disguised clock before being kept -- their correlation with the
# calendar index is at most 0.29, so they carry funding-condition information rather
# than "later month = more defaults". Adding them lifts leave-one-issuer-out AUC from
# 0.696 to 0.862.
CURVE_FEATURES = ["Level", "Slope", "Curvature",
                  "Level_d12", "Slope_d12", "Curv_d12"]

FEATURES = BASE_FEATURES + TREND_FEATURES + CURVE_FEATURES

C_REG = 0.1                 # strong regularisation: only 8 issuers ever defaulted
CLASS_WEIGHT = "balanced"


# =========================================================== enrichment =======
def _rating_num(s: str) -> float:
    if not s:
        return 0.0
    t = re.sub(r"\(.*?\)", "", str(s)).split("/")[0].strip().upper()
    return float(RATING_MAP.get(t, 0.0))


def _parse_schedule(cp: str):
    """couponPayment is a '|'-joined schedule. Return (maturity_date, first_rate)."""
    if not cp:
        return None, np.nan
    dates = re.findall(r"(\d{2})/(\d{2})/(\d{4})", cp)
    mat = None
    if dates:
        try:
            mat = max(pd.Timestamp(year=int(y), month=int(m), day=int(d))
                      for d, m, y in dates)
        except Exception:
            mat = None
    rates = re.findall(r"([\d.]+)\s*%", cp)
    rate = float(rates[0]) if rates else np.nan
    return mat, rate


def enrich_universe(verbose=True) -> pd.DataFrame:
    """Download every issue and attach issuer / sector / rating / maturity / coupon."""
    import ibond_grpc as ig
    c = ig.IBondGrpc()
    who = c.login()
    if verbose:
        print(f"  logged in as {who.get('user_name') or who.get('user_id')}")

    def s(f, n):
        v = f.get(n, [b""])[0]
        return v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else ""

    items = ig.pb_parse(c.call(BOND_SVC, "GetAllBond", b"", timeout=180)).get(1, [])
    ids = [(s(ig.pb_parse(i), 1), s(ig.pb_parse(i), 2)) for i in items]
    if verbose:
        print(f"  {len(ids):,} issues to enrich (about 3 minutes)")
    rows, errs = [], 0
    for k, (iid, sym) in enumerate(ids, 1):
        try:
            b = ig.pb_parse(c.call(BOND_SVC, "Get", ig.pb_string(1, iid), timeout=45))
            f = ig.pb_parse(c.call(BOND_SVC, "GetBondFeature", ig.pb_string(1, iid),
                                   timeout=45))
        except Exception:
            errs += 1
            continue
        mat, rate = _parse_schedule(s(f, 18))
        reg = None
        v = f.get(2, [None])[0]
        if isinstance(v, (bytes, bytearray)):
            d = ig._timestamp_value(v)
            reg = pd.Timestamp(d) if d else None
        sym = s(b, 3)
        icode = s(b, 5).strip()
        if not icode:
            m_code = re.match(r"^([A-Z]+)", sym)
            icode = m_code.group(1) if m_code else ""
        rows.append({
            "issue_id": iid, "symbol": sym,
            "issuer_code": icode, "issuer_name": s(b, 6), "sector": s(b, 7),
            "rating": s(f, 5), "rating_num": _rating_num(s(f, 5)),
            "registration_date": reg, "maturity_date": mat, "coupon": rate,
        })
        if verbose and k % 500 == 0:
            print(f"    {k}/{len(ids)} enriched, {errs} errors")
    if verbose:
        print(f"    {len(rows)} enriched, {errs} errors")
    return pd.DataFrame(rows)


def attach_curve(p: pd.DataFrame, db=DB, verbose=True) -> pd.DataFrame:
    """Merge the DNS yield-curve factors onto the issuer-month panel.

    Month t of the panel gets the curve observed in month t, which is public
    information at that point, so no look-ahead is introduced. If dns_factors is
    missing the columns are filled with zeros and the model simply runs without the
    macro block (the caller reports the reduced feature set)."""
    for c in CURVE_FEATURES:
        p[c] = 0.0
    try:
        con = sqlite3.connect(db)
        fac = pd.read_sql("select * from dns_factors", con)
        con.close()
    except Exception:
        if verbose:
            print("  yield-curve factors unavailable - running without the macro block")
        return p
    if fac.empty:
        return p
    fac["date"] = pd.to_datetime(fac["date"], errors="coerce")
    fac = fac.dropna(subset=["date"])
    fac["month"] = fac["date"].dt.to_period("M")
    cur = fac.groupby("month")[["Level", "Slope", "Curvature"]].last()
    cur["Level_d12"] = cur["Level"].diff(12)
    cur["Slope_d12"] = cur["Slope"].diff(12)
    cur["Curv_d12"] = cur["Curvature"].diff(12)
    p = p.drop(columns=CURVE_FEATURES).merge(
        cur, left_on="month", right_index=True, how="left")
    cov = float(p["Level"].notna().mean())
    p[CURVE_FEATURES] = p[CURVE_FEATURES].fillna(0.0)
    if verbose:
        print(f"  yield-curve factors merged on {cov:.0%} of issuer-months "
              f"({cur.index.min()} .. {cur.index.max()})")
    return p


# ================================================================ panel =======
def build_panel(uni: pd.DataFrame, defaults: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Issuer x month panel. Every feature uses only information available at t."""
    u = uni.copy()
    u["issuer_code"] = np.where(u["issuer_code"].isna() | (u["issuer_code"].astype(str).str.strip() == ""),
                                u["symbol"].astype(str).str.extract(r"^([A-Z]+)")[0], u["issuer_code"])
    u = u.dropna(subset=["issuer_code"])
    u = u[u["issuer_code"].astype(str).str.len() > 0]
    # iBond timestamps come back UTC-aware; month ends are naive. Strip the tz so
    # the two are comparable (the dates are calendar dates, not instants).
    for col in ("registration_date", "maturity_date"):
        u[col] = pd.to_datetime(u[col], errors="coerce", utc=True).dt.tz_localize(None)
    u = u.dropna(subset=["registration_date"])
    if u.empty:
        return pd.DataFrame()

    start = u["registration_date"].min().to_period("M")
    end = pd.Timestamp.today().to_period("M")
    months = pd.period_range(start, end, freq="M")

    # default events -> (issuer, month)
    ev = defaults.copy()
    if not ev.empty:
        ev["payment_date"] = pd.to_datetime(ev["payment_date"], errors="coerce")
        ev = ev.dropna(subset=["payment_date"])
        ev["issuer_code"] = ev["symbol"].astype(str).str.extract(r"^([A-Z]+)")[0]
        ev["month"] = ev["payment_date"].dt.to_period("M")
        first_ev = ev.groupby("issuer_code")["payment_date"].min()
    else:
        first_ev = pd.Series(dtype="datetime64[ns]")

    # market median coupon per month, for the spread feature
    rows = []
    for m in months:
        m_end = m.to_timestamp("M")
        live = u[(u["registration_date"] <= m_end)
                 & ((u["maturity_date"].isna()) | (u["maturity_date"] > m_end))]
        if live.empty:
            continue
        med_coupon = float(live["coupon"].median(skipna=True))
        for code, g in live.groupby("issuer_code"):
            mat = g["maturity_date"]
            rows.append({
                "issuer_code": code, "month": m,
                "n_outstanding": len(g),
                "mat_3m": int(((mat > m_end) & (mat <= m_end + pd.DateOffset(months=3))).sum()),
                "mat_6m": int(((mat > m_end) & (mat <= m_end + pd.DateOffset(months=6))).sum()),
                "mat_12m": int(((mat > m_end) & (mat <= m_end + pd.DateOffset(months=12))).sum()),
                "coupon_avg": float(g["coupon"].mean(skipna=True)),
                "coupon_max": float(g["coupon"].max(skipna=True)),
                "coupon_spread": float(g["coupon"].mean(skipna=True)) - med_coupon,
                "rating_num": float(g["rating_num"].max()),
                "months_since_issue": float(
                    (m_end - g["registration_date"].max()).days / DAYS_PER_MONTH),
                "issue_age_avg": float(
                    (m_end - g["registration_date"]).dt.days.mean() / DAYS_PER_MONTH),
                "sector": str(g["sector"].mode().iat[0]) if not g["sector"].mode().empty else "",
            })
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    p = p.sort_values(["issuer_code", "month"]).reset_index(drop=True)

    # event flag and the forward 3-month target
    p["event_date"] = p["issuer_code"].map(first_ev)
    p["event_month"] = p["event_date"].dt.to_period("M")
    p["event"] = (p["month"] == p["event_month"]).astype(int)
    # drop months after the first default (the issuer has left the risk set)
    p = p[(p["event_month"].isna()) | (p["month"] <= p["event_month"])]
    # target: does the event happen within the next 3 months?
    p["y_fwd"] = 0
    has_ev = p["event_month"].notna()
    delta = (p.loc[has_ev, "event_month"].astype("int64")
             - p.loc[has_ev, "month"].astype("int64"))
    p.loc[has_ev, "y_fwd"] = ((delta >= 0) & (delta <= 3)).astype(int)

    # year-on-year change features (computed per issuer, so no cross-issuer leakage)
    p = p.sort_values(["issuer_code", "month"])
    gb = p.groupby("issuer_code")
    p["coupon_d12"] = gb["coupon_avg"].diff(12)
    p["spread_d12"] = gb["coupon_spread"].diff(12)
    p["n_out_d12"] = gb["n_outstanding"].diff(12)
    p[TREND_FEATURES] = p[TREND_FEATURES].fillna(0.0)

    p = attach_curve(p, verbose=verbose)

    # issuers need a minimum history
    keep = p.groupby("issuer_code")["month"].transform("size") >= MIN_MONTHS
    p = p[keep].reset_index(drop=True)
    p[FEATURES] = p[FEATURES].replace([np.inf, -np.inf], np.nan)
    p[FEATURES] = p[FEATURES].fillna(p[FEATURES].median(numeric_only=True))
    if verbose:
        print(f"  panel: {len(p):,} issuer-months, {p['issuer_code'].nunique()} issuers, "
              f"{int(p['event'].sum())} default months, {int(p['y_fwd'].sum())} positive targets")
    return p


# =============================================================== model ========
def fit_hazard(panel: pd.DataFrame, verbose=True):
    """Pooled logistic hazard with a time baseline, walk-forward out-of-sample."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    d = panel.copy()
    d["t_idx"] = d["month"].astype("int64")
    d["t_idx"] = d["t_idx"] - d["t_idx"].min()
    # t_idx is deliberately NOT a feature. Every recorded default falls in 2024-2026,
    # so a calendar-time term separates the classes almost perfectly (odds ratio in
    # the hundreds of thousands) and drives PD to 1.0 for every issuer in the latest
    # month. That is a property of the sample window, not of credit risk.
    cols = list(FEATURES)
    X = d[cols].to_numpy(float)
    y = d["y_fwd"].to_numpy(int)
    if y.sum() < 5:
        raise RuntimeError(f"only {y.sum()} positive months - too few to fit a hazard model")

    # A temporal split cannot validate this sample: every recorded default falls in
    # the last two years, so any early cut leaves the training half with no events.
    # Leave-one-ISSUER-out is the honest alternative -- the held-out issuer's months
    # are never seen during fitting, so the score is genuinely out-of-sample.
    groups = d["issuer_code"].to_numpy()
    ev_issuers = sorted(d.loc[d["y_fwd"] == 1, "issuer_code"].unique())
    oos_y, oos_p = [], []
    for held in ev_issuers:
        m_tr = groups != held
        if y[m_tr].sum() < 2:
            continue
        sc_i = StandardScaler().fit(X[m_tr])
        mi = LogisticRegression(max_iter=3000, C=C_REG, class_weight=CLASS_WEIGHT)
        mi.fit(sc_i.transform(X[m_tr]), y[m_tr])
        te_i = ~m_tr
        oos_y.append(y[te_i])
        oos_p.append(mi.predict_proba(sc_i.transform(X[te_i]))[:, 1])
    auc_oos = np.nan
    if oos_y:
        yy, pp = np.concatenate(oos_y), np.concatenate(oos_p)
        if 0 < yy.sum() < len(yy):
            auc_oos = float(roc_auc_score(yy, pp))
    # final model on everything, used for the reported signals
    sc_all = StandardScaler().fit(X)
    m_all = LogisticRegression(max_iter=3000, C=C_REG, class_weight=CLASS_WEIGHT)
    m_all.fit(sc_all.transform(X), y)

    # PRIOR CORRECTION (King & Zeng 2001).
    # class_weight="balanced" reweights 32 positives out of 16,686 as if the classes
    # were 50/50, which is right for ranking but leaves the probabilities far too
    # high: PD_3M saturated at 1.0 for ~14% of issuers. Once PD(t-1) pins to 1.0,
    # Momentum = PD(t)/1.0 = PD(t), so every high-risk issuer fell exactly on the
    # line y = x in the (PD, Momentum) plane -- the straight line seen on the chart.
    # Shifting the intercept by log((1-tau)/tau) maps the balanced-sample
    # probabilities back onto the true base rate. It is a monotone transform, so the
    # AUC is unchanged; only the calibration is fixed.
    tau = float(y.mean())
    lin = sc_all.transform(X) @ m_all.coef_[0] + m_all.intercept_[0]
    prior_offset = float(np.log((1 - tau) / tau)) if 0 < tau < 1 else 0.0
    d["h"] = 1.0 / (1.0 + np.exp(-(lin - prior_offset)))
    auc_in = float(roc_auc_score(y, d["h"])) if y.sum() else np.nan
    coefs = pd.DataFrame({"feature": cols, "beta": m_all.coef_[0],
                          "odds_ratio": np.exp(m_all.coef_[0])}) \
        .sort_values("beta", key=np.abs, ascending=False)
    if verbose:
        print(f"  hazard fitted: AUC in-sample {auc_in:.3f}, out-of-sample "
              f"{auc_oos:.3f}" if not np.isnan(auc_oos) else
              f"  hazard fitted: AUC in-sample {auc_in:.3f}, out-of-sample n/a")
    return d, {"auc_in": auc_in, "auc_oos": auc_oos, "n_pos": int(y.sum()),
               "coefs": coefs, "base_rate": tau, "prior_offset": prior_offset}


def add_signals(d: pd.DataFrame) -> pd.DataFrame:
    """PD_3M from the hazard, then momentum and the hyperbolic alarm."""
    d = d.sort_values(["issuer_code", "month"]).copy()
    g = d.groupby("issuer_code")["h"]
    h1, h2 = g.shift(-1), g.shift(-2)
    d["PD_3M"] = 1 - (1 - d["h"]) * (1 - h1.fillna(d["h"])) * (1 - h2.fillna(d["h"]))
    d["PD_prev"] = d.groupby("issuer_code")["PD_3M"].shift(1)
    d["Momentum"] = d["PD_3M"] / d["PD_prev"].replace(0, np.nan)
    d["Momentum"] = d["Momentum"].fillna(1.0)
    ok = (d["PD_prev"] > 1e-9) & (d["Momentum"] > 0)
    score = np.where(ok, np.log(d["Momentum"].clip(lower=1e-9))
                     + ALPHA_BOUND * np.log(d["PD_prev"].clip(lower=1e-9)), -np.inf)
    d["flag_hyper"] = (score >= np.log(K_BOUND)).astype(int)
    # After the prior correction the calibrated PDs are small in absolute terms
    # (the true base rate is ~0.2% per month), so a fixed 0.50 cut would never fire.
    # Fix the ALARM BUDGET instead: alarm on the riskiest ALARM_BUDGET share of
    # issuer-months. The threshold is reported so the number is auditable.
    thr = float(d["PD_3M"].quantile(1 - ALARM_BUDGET))
    d["alarm"] = (d["PD_3M"] >= thr).astype(int)
    d.attrs["p_star"] = thr
    return d


def _levels(d: pd.DataFrame, thr: float):
    """Alert bands defined relative to the alarm threshold, not to absolute PD."""
    return [alert_level(r.PD_3M, r.Momentum, r.flag_hyper, thr) for r in d.itertuples()]


def _verdict(auc_oos, n_high, n_total) -> str:
    """One-line honest read of whether this model is usable. Called by the GUI so the
    screen cannot show a green dashboard for a model with no out-of-sample skill."""
    pct = (n_high / n_total * 100) if n_total else 0
    if np.isnan(auc_oos):
        return ("NOT VALIDATED - no out-of-sample estimate could be produced. "
                "Do not use for decisions.")
    if auc_oos < 0.55:
        return (f"NO PREDICTIVE SKILL - out-of-sample AUC {auc_oos:.3f} is at or below "
                f"chance (0.50). The in-sample fit is overfitting; {pct:.0f}% of issuers "
                f"are flagged. Do not use for decisions.")
    if auc_oos < 0.65 or pct > 20:
        return (f"WEAK - out-of-sample AUC {auc_oos:.3f}, {pct:.0f}% of issuers flagged. "
                f"Indicative only.")
    return f"USABLE - out-of-sample AUC {auc_oos:.3f}, {pct:.0f}% flagged."


def alert_level(pd3m, momentum, flag, thr):
    """Bands are relative to the alarm threshold `thr` (set by the alarm budget)."""
    if pd3m >= thr and (flag or momentum > 1.0):
        return "HIGH RISK"
    if pd3m >= thr:
        return "ELEVATED"
    if pd3m >= thr * 0.5 or flag:
        return "WATCH"
    return "OK"


def lead_time(d: pd.DataFrame, thr: float, verbose=True) -> pd.DataFrame:
    """Actionable 1-3M lead time plus final persistent alarm duration."""
    rows = []
    for code, g in d.groupby("issuer_code"):
        g = g.sort_values("month")
        ev = g["event_date"].dropna()
        if ev.empty:
            continue
        ev_date = ev.iloc[0]
        pre = g[g["month"].dt.to_timestamp() < ev_date]
        metrics = lead_metrics.compute_lead_metrics(
            g,
            event_date=ev_date,
            date_col="month",
            alarm_mask=g["alarm"] == 1,
        )
        status, kind = lead_metrics.status_and_kind(metrics, has_event=True)
        selected_idx = (
            metrics.get("actionable_alarm_index")
            if metrics.get("actionable_alarm_index") is not None
            else metrics.get("persistent_alarm_start_index")
        )
        selected = (
            g.loc[selected_idx]
            if selected_idx is not None and selected_idx in g.index
            else (pre.iloc[-1] if not pre.empty else g.iloc[0])
        )
        alert = (
            alert_level(
                selected["PD_3M"], selected["Momentum"],
                selected["flag_hyper"], thr,
            )
            if status == "detected"
            else ("EARLIER ALARM ONLY" if kind == "earlier-only" else "MISSED")
        )
        rows.append({
            "issuer_code": code,
            "PD_3M": float(selected["PD_3M"]),
            "Momentum": float(selected["Momentum"]),
            "alert": alert,
            "status": status,
            "kind": kind,
            **lead_metrics.strip_internal_fields(metrics),
        })
    lt = pd.DataFrame(rows)
    if not lt.empty:
        lt = lt.sort_values("lead_days", na_position="last")
    if verbose and not lt.empty:
        got = lt["lead_days"].notna().sum()
        print(f"  actionable lead (1-3M): caught {got}/{len(lt)} defaulted issuers, "
              f"median {lt['lead_days'].median():.0f} days"
              if got else f"  actionable lead (1-3M): caught 0/{len(lt)} defaulted issuers")
    return lt


# ============================================================== storage =======
def save_to_sqlite(uni, panel, alerts, lt, summary, db=DB):
    con = sqlite3.connect(db)
    try:
        for df, t in ((uni, T_UNIVERSE), (panel, T_PANEL), (alerts, T_ALERT), (lt, T_LEAD)):
            if df is not None and not df.empty:
                d = df.copy()
                for c in d.columns:
                    if str(d[c].dtype).startswith("period") or d[c].dtype == object:
                        try:
                            d[c] = d[c].astype(str)
                        except Exception:
                            pass
                    elif pd.api.types.is_datetime64_any_dtype(d[c]):
                        d[c] = d[c].astype(str)
                d.to_sql(t, con, if_exists="replace", index=False)
        pd.DataFrame([summary]).to_sql(T_SUMMARY, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    out = []
    try:
        for t in (T_UNIVERSE, T_PANEL, T_ALERT, T_LEAD, T_SUMMARY):
            try:
                out.append(pd.read_sql(f"select * from {t}", con))
            except Exception:
                out.append(pd.DataFrame())
    finally:
        con.close()
    return tuple(out)


# ================================================================== run =======
def run(refresh=False, save=True, verbose=True, log=None):
    def emit(m):
        if verbose:
            print(m)
        if log:
            log(m)

    import download_bond as dbnd
    emit("[1/5] loading the bond universe ...")
    uni, _b, defaults, _l = pd.DataFrame(), None, pd.DataFrame(), None
    if not refresh:
        con = sqlite3.connect(DB)
        try:
            uni = pd.read_sql(f"select * from {T_UNIVERSE}", con)
        except Exception:
            uni = pd.DataFrame()
        finally:
            con.close()
    if refresh or uni.empty:
        emit("      enriching every issue from iBond (about 3 minutes) ...")
        uni = enrich_universe(verbose=verbose)
    else:
        emit(f"      using the cached enriched universe ({len(uni):,} issues)")
    _i, _bo, defaults, _lg = dbnd.load_from_sqlite(DB)
    emit(f"      {len(uni):,} issues, {len(defaults):,} default records")

    emit("[2/5] building the issuer-month panel ...")
    panel = build_panel(uni, defaults, verbose=verbose)
    if panel.empty:
        raise RuntimeError("panel is empty - download the bond data first")
    emit(f"      {len(panel):,} issuer-months, {panel['issuer_code'].nunique()} issuers")

    emit("[3/5] fitting the discrete-time hazard ...")
    d, meta = fit_hazard(panel, verbose=verbose)
    emit(f"      AUC in-sample {meta['auc_in']:.3f} | out-of-sample "
         f"{meta['auc_oos']:.3f}" if not np.isnan(meta["auc_oos"])
         else f"      AUC in-sample {meta['auc_in']:.3f} | out-of-sample n/a")

    emit("[4/5] PD_3M, momentum and the hyperbolic alarm ...")
    d = add_signals(d)
    latest = (d.sort_values("month").groupby("issuer_code").tail(1)
              .sort_values("PD_3M", ascending=False).reset_index(drop=True))
    thr = float(d.attrs.get("p_star", np.nan))
    latest["alert"] = _levels(latest, thr)

    emit("[5/5] actionable 1-3M lead and persistent duration ...")
    lt = lead_time(d, thr, verbose=verbose)

    counts = latest["alert"].value_counts()
    summary = {
        "n_issues": int(len(uni)), "n_issuers": int(panel["issuer_code"].nunique()),
        "n_issuer_months": int(len(panel)),
        "n_defaulted_issuers": int(panel["event_date"].notna().groupby(
            panel["issuer_code"]).any().sum()),
        "n_positive_months": int(meta["n_pos"]),
        "auc_in": meta["auc_in"], "auc_oos": meta["auc_oos"],
        "p_star": thr, "alarm_budget": ALARM_BUDGET,
        "base_rate": meta.get("base_rate", np.nan),
        "prior_offset": meta.get("prior_offset", np.nan), "K": K_BOUND, "alpha": ALPHA_BOUND,
        "n_high": int(counts.get("HIGH RISK", 0)),
        "n_elevated": int(counts.get("ELEVATED", 0)),
        "n_watch": int(counts.get("WATCH", 0)),
        "n_ok": int(counts.get("OK", 0)),
        "n_events": int(len(lt)),
        "flagged_pct": float(counts.get("HIGH RISK", 0) / max(len(latest), 1) * 100),
        "verdict": _verdict(meta["auc_oos"], counts.get("HIGH RISK", 0), len(latest)),
        **lead_metrics.summarize_lead_table(lt),
        "as_of": str(latest["month"].max()) if not latest.empty else "",
    }
    if save:
        # `latest` (one scored row per issuer, carrying the alert band) is what the
        # alert table is meant to hold -- passing the full panel `d` here wrote
        # 16,686 rows without the `alert` column, so the GUI showed blank alerts.
        save_to_sqlite(uni, panel, latest, lt, summary, DB)
        emit(f"      saved: {T_UNIVERSE}, {T_PANEL}, {T_ALERT}, {T_LEAD}, {T_SUMMARY}")
    emit(f"DONE - {summary['n_high']} HIGH RISK, caught "
         f"{summary['n_caught']}/{summary['n_events']} defaulted issuers")
    return uni, panel, latest, lt, meta, summary


def main():
    uni, panel, latest, lt, meta, summ = run(
        refresh="--refresh" in sys.argv, save="--no-save" not in sys.argv)
    print("\n" + "=" * 92)
    print("APPROACH 1 ON iBOND CORPORATE BONDS")
    print("=" * 92)
    for k in ("n_issues", "n_issuers", "n_issuer_months", "n_defaulted_issuers",
              "n_positive_months", "auc_in", "auc_oos", "n_high", "n_elevated",
              "n_watch", "n_ok", "n_caught", "n_events", "median_lead_days"):
        v = summ[k]
        print(f"  {k:22} {v:.3f}" if isinstance(v, float) and not np.isnan(v)
              else f"  {k:22} {v}")
    print("\nTOP FEATURES (|beta|)")
    print(meta["coefs"].head(8).to_string(index=False,
                                          float_format=lambda v: f"{v:.4f}"))
    if not lt.empty:
        print("\nACTIONABLE 1-3M LEAD TIME PER DEFAULTED ISSUER")
        show = lt[["issuer_code", "first_alarm", "default_date", "lead_days",
                   "lead_months", "PD_3M", "alert"]].copy()
        print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nTOP 12 CURRENT RISK")
    print(latest[["issuer_code", "month", "PD_3M", "Momentum", "alert"]]
          .head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDone.")


if __name__ == "__main__":
    main()
