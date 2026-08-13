# -*- coding: utf-8 -*-
"""
survivor2.py  --  Approach 1 ONLY  (Dynamic Survival Hazard Early-Warning System)
================================================================================
Input : the 33 real bond features (Rev01_Database_final.dta), time-varying.
Output: Path-1 alert (OK / WATCH / ELEVATED / HIGH RISK) + LEAD TIME in DAYS,
        performance metrics, and the per-firm lead-time table.

WHY A SEPARATE ALARM STEP IS NEEDED
-----------------------------------
A Cox proportional-hazards model does NOT hand you a "lead time" directly. It is
NOT an anomaly detector that emits an alarm on some day which you then subtract
from the event day. Cox gives a *risk over time*:

    h(t | X)  = h0(t) * exp(beta' X)              instantaneous hazard
    H(t | X)  = H0(t) * exp(beta' X)              cumulative hazard
    S(t | X)  = exp(-H(t | X))                    survival probability
    F(t | X)  = 1 - S(t | X)                      P(event by time t)

To get a lead time you must FIRST define a warning rule (a threshold), THEN read
off the first day the rule fires. With time-varying covariates X(t) the hazard is
recomputed every period:

    h(t | X(t)) = h0(t) * exp(beta' X(t)).

PIPELINE (matches the paper, Stage 2-3)
    financial vars -> Cox hazard -> forward event prob -> THRESHOLD -> t_alarm -> Lead Time

WARNING RULE (operative, no peeking at the future):
    forward probability   P_H(t) = P(event within H months | survived to t, X(t))
                                 = 1 - prod_{k=1..H} ( 1 - h(t+k | X(t)) )
    alarm when            P_H(t) > p*            (default p* = 0.50, H = 3 months)
    t_alarm = first month where P_H crosses p*, taken BEFORE the real event.

LEAD TIME (empirical):
    ActionableLeadTime_i = DefaultDate_i - FirstAlarmDate_i   (in DAYS)
    - FirstAlarmDate = first alarm inside the 1-3 calendar-month window before
      the event. The backward-compatible lead_days / lead_months columns always
      use this definition.
    - Persistent Alarm Duration is reported separately from the start of the
      final continuous monthly alarm episode. It may legitimately span years.
    - firms that never default are CENSORED -> Lead Time = N/A
    Median Lead Time = median_i ( t_event,i - t_alarm,i )

NOTE: a hazard ratio (e.g. HR = 2.5) can NOT be converted to "lead time = N days"
      on its own -- it is a *relative* risk. Lead time needs the baseline hazard,
      the survival curve and the warning threshold defined above.

HYPERBOLIC BOUNDARY (secondary alarm layer)
    The paper's operative boundary lives in (PD_prev, Momentum) space:
        Momentum(t) >= K / PD_prev(t)^alpha      <=>   log M + alpha*log P >= log K
    Geometrically this is the Euclidean hyperbola  M * P^alpha = K.
    A risk/lead-time reading of the same idea keeps  hazard * remaining_time = c:
        h(t | X(t)) * (t_event - t) = c
    so a very high hazard warns even far from the event, while near the event a
    lower hazard already crosses. (A metric-true version would be a Poincare
    half-plane geodesic (r - r_c)^2 + tau^2 = R^2; not needed for the alarm.)
    Because (t_event - t) is unknown in real time, the OPERATIVE alarm used for
    the lead-time table is the probability threshold above; the hyperbolic
    boundary is reported alongside as a comparison.

Run:
    python survivor2.py                     # full run
    python survivor2.py --pstar 0.30        # change the warning threshold
    python survivor2.py --csv leadtime.csv  # also dump the per-firm table
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression

import lead_metrics
import survival
from load_bond import (BOND_DTA, BOND_FEATURES, DISPLAY_COLS, EVENT_SOURCE, HORIZON)

P_STAR_DEFAULT = 0.50        # warning threshold: P(event within H months) > p*
DAYS_PER_MONTH = 30.4368     # for reporting lead time in months
MIN_FEATURES = 17            # keep firm-months with >= 50% of the 33 features observed
                             # (a data-quality rule on the MODEL features -- NOT on DTD)
# extra columns that must never enter the hazard covariates (they encode the outcome)
HAZARD_DROP = ["event_dp", "event_rs", "duration_m", "date_DP", "date_RS", "default_3m"]
QUAL_LO, QUAL_HI = 1, 3      # qualifying lead-time window (calendar months before event)


# --------------------------------------------------------------- data ---------
def load_bond_dated(path=BOND_DTA, horizon=HORIZON):
    """33 features + REAL dates + credit-event onset (restructure and/or default),
    censored after the first event. Keeps month_year so lead time is in real days."""
    cols = list(dict.fromkeys(
        BOND_FEATURES + DISPLAY_COLS
        + ["firm_id", "month_year", EVENT_SOURCE, "d_Default_Payment", "d_Restructure",
           "date_DP", "date_RS"]))
    df = pd.read_stata(path, columns=cols, convert_categoricals=False)
    # Inclusion rule = completeness of the MODEL FEATURES themselves (>= 50% observed),
    # NOT the presence of dd_12m. Filtering on DTD (a market variable that is not a model
    # feature) would select toward market-observed firms and drop otherwise-usable firm-
    # months -- a selection bias. dd_12m/pd_12m are kept only for display and may be NaN.
    feat_present = df[BOND_FEATURES].notna().sum(axis=1)
    df = df[feat_present >= MIN_FEATURES].copy()
    df["firm_id"] = pd.to_numeric(df["firm_id"], errors="coerce")
    df["month_year"] = pd.to_datetime(df["month_year"])
    df = df.sort_values(["firm_id", "month_year"]).reset_index(drop=True)

    df["month_index"] = pd.factorize(df["month_year"], sort=True)[0] + 1   # global chronological
    dp = (pd.to_numeric(df["d_Default_Payment"], errors="coerce") > 0).astype(int)
    rs = (pd.to_numeric(df["d_Restructure"], errors="coerce") > 0).astype(int)
    flag = (pd.to_numeric(df[EVENT_SOURCE], errors="coerce") > 0).astype(int)   # composite DP or RS
    cum = flag.groupby(df["firm_id"]).cumsum()
    onset = (cum == 1) & (flag == 1)                          # first month of ANY credit event
    df["event"] = onset.astype(int)                          # composite onset
    # competing risks: same risk set (censor at first event of either cause), but the
    # exit is attributed to whichever cause is active at that month (both if simultaneous)
    df["event_dp"] = (onset & (dp == 1)).astype(int)         # exit by payment default
    df["event_rs"] = (onset & (rs == 1)).astype(int)         # exit by restructuring
    df = df[cum <= 1].reset_index(drop=True)                  # keep rows up to & including onset

    def _fwd(sub):                                            # event within the next 3 CALENDAR months
        ei = sub.loc[sub["event"] == 1, "month_index"].to_numpy()
        mi = sub["month_index"].to_numpy()
        out = np.zeros(len(mi), dtype=int)
        if ei.size:
            for k, m in enumerate(mi):
                out[k] = int(((ei > m) & (ei <= m + horizon)).any())
        return pd.Series(out, index=sub.index)
    # select the two needed columns before apply -> no "operated on grouping columns" warning
    df["default_3m"] = (df.groupby("firm_id", group_keys=False)[["event", "month_index"]]
                        .apply(_fwd).astype(int))
    df["account_id"] = df["firm_id"]
    df["date_DP"] = pd.to_datetime(df["date_DP"], errors="coerce")   # day-level event dates
    df["date_RS"] = pd.to_datetime(df["date_RS"], errors="coerce")
    # duration since the firm's first observation (PROXY for age since first bond issue;
    # the true first-issue date must be collected from iBond -- not in this .dta).
    df["duration_m"] = df["month_index"] - df.groupby("firm_id")["month_index"].transform("min")

    # keep ONLY features + ids/dates/targets; drop the raw event flags so they can
    # never leak into the hazard covariates (they equal the target).
    keep = (["account_id", "firm_id", "month_index", "month_year", "duration_m"]
            + BOND_FEATURES + DISPLAY_COLS
            + ["event", "event_dp", "event_rs", "date_DP", "date_RS", "default_3m"])
    return df[keep]


# --------------------------------------------------------------- alerts -------
def alert_level(pd_h, momentum, flag_rs):
    """Path-1 status from the forward probability + momentum + hyperbolic flag."""
    m = 1.0 if pd.isna(momentum) else float(momentum)
    if pd_h > 0.50 or (flag_rs and pd_h > 0.15):
        return "HIGH RISK"
    if pd_h > 0.15 or flag_rs:
        return "ELEVATED"
    if pd_h > 0.05 or m >= 1.15:
        return "WATCH"
    return "OK"


def _mom(v):
    return 1.0 if pd.isna(v) else float(v)


# --------------------------------------------------------------- lead time ----
def lead_time_table(df, p_star):
    """Per firm actionable 1-3M lead time plus persistent alarm duration."""
    df = df.copy()
    df["alert"] = [alert_level(p, m, f) for p, m, f in
                   zip(df["PD_3M"], df["Momentum"], df["flag_RS"])]
    recs = []
    for fid, g in df.groupby("firm_id"):
        g = g.sort_values("month_index").reset_index(drop=True)
        ev = g[g["event"] == 1]
        if len(ev):                                            # firm defaulted
            d_date = ev["month_year"].iloc[0]
            metrics = lead_metrics.compute_lead_metrics(
                g,
                event_date=d_date,
                date_col="month_year",
                alarm_mask=g["PD_3M"] > p_star,
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
                else g[g["month_year"] < d_date].tail(1).squeeze()
            )
            if not isinstance(selected, pd.Series) or selected.empty:
                selected = g.iloc[0]
            alert = selected["alert"] if status == "detected" else (
                "EARLIER ALARM ONLY" if kind == "earlier-only" else "MISSED"
            )
            recs.append({
                "firm_id": int(fid),
                "PD_3M": round(float(selected["PD_3M"]), 3),
                "Momentum": round(_mom(selected["Momentum"]), 2),
                "alert": alert,
                "status": status,
                "kind": kind,
                **lead_metrics.strip_internal_fields(metrics),
            })
        else:                                                  # censored (no event yet)
            last = g.iloc[-1]
            metrics = lead_metrics.compute_lead_metrics(
                g,
                event_date=pd.NaT,
                date_col="month_year",
                alarm_mask=g["PD_3M"] > p_star,
            )
            recs.append({
                "firm_id": int(fid),
                "PD_3M": round(float(last["PD_3M"]), 3),
                "Momentum": round(_mom(last["Momentum"]), 2),
                "alert": last["alert"],
                "status": "censored",
                "kind": "N/A",
                **lead_metrics.strip_internal_fields(metrics),
            })
    return pd.DataFrame(recs)


# --------------------------------------------------------------- report -------
def _fmt(v, dash="-"):
    return dash if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def print_table(tbl, title, limit=None):
    rows = tbl if limit is None else tbl.head(limit)
    head = ("Firm ID", "Actionable", "Default Date", "Lead(days)", "Persist(d)", "PD_3M", "Momentum", "Alert")
    w = (8, 12, 12, 11, 11, 7, 9, 18)
    line = "  ".join(h.ljust(wi) for h, wi in zip(head, w))
    print(f"\n{title}")
    print("-" * len(line)); print(line); print("-" * len(line))
    for _, r in rows.iterrows():
        lead = "N/A" if r["status"] == "censored" else ("MISSED" if r["status"] == "missed"
                                                         else str(r["lead_days"]))
        persistent = _fmt(r.get("persistent_alarm_days"))
        cells = (str(r["firm_id"]), _fmt(r["first_alarm"]), _fmt(r["default_date"]),
                 lead, persistent, f"{r['PD_3M']:.3f}", f"{r['Momentum']:.2f}", r["alert"])
        print("  ".join(c.ljust(wi) for c, wi in zip(cells, w)))


def odds_ratios(model, top=8):
    """exp(beta) per 1 SD of each covariate. NOTE: this is a DISCRETE-TIME LOGISTIC
    hazard (pooled logit), so exp(beta) is the odds ratio of the monthly event, not a
    Cox partial-likelihood hazard ratio (they coincide only when the hazard is small)."""
    clf, covs = model["clf"], model["covs"]
    coefs = clf.coef_[0][survival._DEG:]                       # drop the 3 baseline-time columns
    return pd.Series(np.exp(coefs), index=covs).sort_values(ascending=False)


# ============================================================================
#  --full : items 16.5 (2,4,5,6,7)
#    2  competing risks DP / RS / composite + day-level dates
#    4  walk-forward (expanding-window) OOS -- every alarm is out-of-sample
#    5  warning threshold frozen on a validation period before the test period
#    6  actionable lead time in a 1-3 month window; persistent duration separate
#    7  Brier score + calibration bins + lead-time IQR + false alarms / firm-year
# ============================================================================
BASELINE_COVS = BOND_FEATURES     # only the 33 features enter the hazard


def _standardize(sub, model):
    cols = []
    for c in model["covs"]:
        vals = pd.to_numeric(sub[c], errors="coerce")
        mu, sd = model["stats"][c]
        cols.append(np.nan_to_num((vals.fillna(mu) - mu) / sd, nan=0.0))
    return np.column_stack(cols)


def _pd3m(model, sub, horizon=3):
    """h(t) and forward PD_3M for the rows in `sub`, using a fitted hazard model."""
    Xstd = _standardize(sub, model)
    mi = sub["month_index"].to_numpy(float)
    h = survival._haz(model, mi, Xstd)
    S = np.ones(len(sub))
    for k in range(1, horizon + 1):
        S *= (1 - np.clip(survival._haz(model, mi + k, Xstd), 1e-9, 0.999))
    return h, 1.0 - S


def _fwd_label(panel, event_col, horizon=3):
    """1 if the event occurs in the next `horizon` calendar months (per firm)."""
    def f(sub):
        ei = sub.loc[sub[event_col] == 1, "month_index"].to_numpy()
        mi = sub["month_index"].to_numpy()
        if not ei.size:
            return pd.Series(np.zeros(len(mi), int), index=sub.index)
        return pd.Series([int(((ei > m) & (ei <= m + horizon)).any()) for m in mi], index=sub.index)
    return (panel.groupby("firm_id", group_keys=False)[[event_col, "month_index"]]
            .apply(f).astype(int).to_numpy())


def walk_forward(panel, event_col, n_folds=8, min_train_frac=0.35):
    """Expanding-window OOS (item 4): each firm-month's h / PD_3M is predicted by a
    model trained ONLY on strictly-earlier months, so every alarm is out-of-sample."""
    p = panel.copy()
    p["event"] = p[event_col].astype(int)
    p["y3"] = _fwd_label(p, "event")
    months = np.sort(p["month_index"].unique())
    start = months[int(len(months) * min_train_frac)]
    cuts = np.unique(np.linspace(start, months[-1] + 1, n_folds + 1).astype(int))
    parts = []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        train = p[p["month_index"] < c0]
        if train["event"].sum() < 2 or train["event"].nunique() < 2:
            continue
        model = survival.fit_hazard(train, covs=BASELINE_COVS)
        seg = p[(p["month_index"] >= c0) & (p["month_index"] < c1)].copy()
        if not len(seg):
            continue
        seg["h"], seg["PD_3M"] = _pd3m(model, seg)
        parts.append(seg)
    if not parts:
        return None
    oos = pd.concat(parts).sort_values(["firm_id", "month_index"]).reset_index(drop=True)
    oos["PD_prev"] = oos.groupby("firm_id")["PD_3M"].shift(1)
    oos["Momentum"] = oos["PD_3M"] / oos["PD_prev"]
    return oos


def _brier_bins(y, p, bins=5):
    y = np.asarray(y, float); p = np.asarray(p, float)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        hi = edges[i + 1]
        m = (p >= edges[i]) & ((p < hi) if i < bins - 1 else (p <= hi))
        if m.sum():
            rows.append((edges[i], hi, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return brier, rows


def _event_date(panel, fid, cause):
    g = panel[panel["firm_id"] == fid]
    dp = g["date_DP"].dropna(); rs = g["date_RS"].dropna()
    if cause == "DP":
        return dp.min() if len(dp) else None
    if cause == "RS":
        return rs.min() if len(rs) else None
    cand = [d for d in (dp.min() if len(dp) else None, rs.min() if len(rs) else None) if d is not None]
    return min(cand) if cand else None


def qualifying_leadtime(oos, panel, event_col, cause, p_star, lo=QUAL_LO, hi=QUAL_HI):
    """Item 6: main lead time from the earliest alarm inside the [lo,hi]-month window
    before the event, with persistent alarm duration reported separately."""
    em = panel[panel[event_col] == 1].groupby("firm_id")["month_index"].min().to_dict()
    edate = {fid: _event_date(panel, fid, cause) for fid in em}
    recs = []
    for fid, g in oos.groupby("firm_id"):
        if fid not in em:
            metrics = lead_metrics.compute_lead_metrics(
                g, event_date=pd.NaT, date_col="month_year",
                alarm_mask=g["PD_3M"] > p_star,
            )
            recs.append({
                "firm_id": int(fid), "status": "censored", "kind": "N/A",
                **lead_metrics.strip_internal_fields(metrics),
            })
            continue
        g = g.sort_values("month_index")
        ed = edate.get(fid)
        metrics = lead_metrics.compute_lead_metrics(
            g,
            event_date=ed if ed is not None else pd.NaT,
            event_month_ordinal=None if ed is not None else em[fid],
            date_col="month_year",
            alarm_mask=g["PD_3M"] > p_star,
            window_min_months=lo,
            window_max_months=hi,
        )
        status, kind = lead_metrics.status_and_kind(metrics, has_event=True)
        recs.append({
            "firm_id": int(fid),
            "status": status,
            "kind": kind,
            **lead_metrics.strip_internal_fields(metrics),
        })
    return pd.DataFrame(recs)


def false_alarms_per_firm_year(oos, panel, event_col, p_star, hi=QUAL_HI):
    d = oos.dropna(subset=["PD_3M"]).copy()
    d["flag"] = (d["PD_3M"] > p_star).astype(int)
    ev = panel[panel[event_col] == 1][["firm_id", "month_index"]].rename(columns={"month_index": "em"})
    d = d.merge(ev, on="firm_id", how="left")
    d["in_window"] = (d["em"] - d["month_index"]).between(1, hi).fillna(False)
    fp = int(((d["flag"] == 1) & (~d["in_window"])).sum())
    firm_years = len(d) / 12.0
    return (fp / firm_years if firm_years else float("nan")), fp


def run_cause(panel, event_col, cause, label):
    oos = walk_forward(panel, event_col)
    n_ev = int(panel[event_col].sum())
    if oos is None:
        print(f"\n[{label}]  event firms = {n_ev}: insufficient events for walk-forward."); return
    d = oos.dropna(subset=["PD_3M", "y3"])
    auc = survival.roc_auc_score(d["y3"], d["PD_3M"]) if d["y3"].nunique() > 1 else float("nan")
    brier, bins = _brier_bins(d["y3"], d["PD_3M"])

    # item 5: freeze the threshold on the first half of the OOS months (validation)
    oms = np.sort(oos["month_index"].unique())
    vcut = oms[len(oms) // 2]
    val = oos[oos["month_index"] <= vcut].dropna(subset=["PD_3M", "y3"])
    best_p, best_mcc = 0.50, -1.0
    for ps in np.linspace(0.05, 0.95, 19):
        f = (val["PD_3M"] > ps).astype(int)
        if f.nunique() < 2:
            continue
        mcc = survival.matthews_corrcoef(val["y3"].astype(int), f)
        if mcc > best_mcc:
            best_mcc, best_p = mcc, float(ps)

    # item 7: recalibrate probabilities on validation (isotonic), score the test period
    te = oos[oos["month_index"] > vcut].dropna(subset=["PD_3M", "y3"])
    brier_raw_te = brier_cal_te = float("nan")
    if len(val) and len(te) and val["y3"].nunique() > 1:
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            val["PD_3M"].to_numpy(), val["y3"].astype(int).to_numpy())
        pc = iso.predict(te["PD_3M"].to_numpy())
        yte = te["y3"].astype(int).to_numpy()
        brier_raw_te = float(np.mean((te["PD_3M"].to_numpy() - yte) ** 2))
        brier_cal_te = float(np.mean((pc - yte) ** 2))

    lt = qualifying_leadtime(oos, panel, event_col, cause, best_p)
    q = lt[lt["kind"] == "qualifying"]; lh = lt[lt["kind"] == "earlier-only"]
    miss = lt[lt["status"] == "missed"]
    leads = q["lead_days"].astype(float).values
    far, fp = false_alarms_per_firm_year(oos, panel, event_col, best_p)

    print(f"\n{'='*72}")
    print(f"[{label}]  event firms = {n_ev}   (walk-forward OOS; threshold frozen on validation)")
    print(f"  OOS ROC-AUC {auc:.3f}   Brier {brier:.4f}   frozen p* = {best_p:.2f}")
    print(f"  qualifying detection ({QUAL_LO}-{QUAL_HI} mo window): {len(q)}/{n_ev}"
          f"   | persistent-only earlier alarms: {len(lh)}   | missed: {len(miss)}")
    if len(leads):
        print(f"  qualifying lead time: median {np.median(leads):.0f} d "
              f"[IQR {np.percentile(leads,25):.0f}-{np.percentile(leads,75):.0f}]"
              f"  ({np.median(leads)/DAYS_PER_MONTH:.1f} mo)")
    print(f"  false alarms / firm-year: {far:.3f}  (flagged non-window firm-months = {fp})")
    print(f"  Brier (test period): raw {brier_raw_te:.4f} -> isotonic-recalibrated {brier_cal_te:.4f}")
    print("  raw calibration (predicted PD -> observed 3m-event frequency):")
    for lo, hi, n, pm, ym in bins:
        print(f"      PD [{lo:.1f},{hi:.1f})  n={n:7d}  mean pred {pm:.3f}  obs {ym:.3f}")


def run_full(panel):
    print("=" * 78)
    print("APPROACH 1  --  FULL OUT-OF-SAMPLE EVALUATION  (report items 16.5: 2,4,5,6,7)")
    print("=" * 78)
    print(f"  sample: {len(panel):,} firm-months, {panel['firm_id'].nunique()} firms, "
          f"{panel['month_year'].min().date()}..{panel['month_year'].max().date()}")
    print(f"  onsets:  composite {int(panel['event'].sum())}  |  "
          f"DP-only {int(panel['event_dp'].sum())}  |  RS-only {int(panel['event_rs'].sum())}")
    print("  time origin = calendar month (duration since first bond issue needs iBond data;")
    print("                duration_m is a within-sample proxy only).")
    for col, cause, label in [("event", "ANY", "Composite (DP and/or RS)"),
                              ("event_dp", "DP", "Default-Payment only"),
                              ("event_rs", "RS", "Restructure only")]:
        run_cause(panel, col, cause, label)
    print("\nDone (full evaluation).")


def main():
    p_star = P_STAR_DEFAULT
    csv_out = None
    if "--pstar" in sys.argv:
        p_star = float(sys.argv[sys.argv.index("--pstar") + 1])
    if "--csv" in sys.argv:
        csv_out = sys.argv[sys.argv.index("--csv") + 1]

    if "--full" in sys.argv:                 # items 16.5 (2,4,5,6,7): OOS + competing risks
        print("[Stage 1] Loading 33 features + dates + competing-risk onsets ...")
        run_full(load_bond_dated())
        return

    print("=" * 78)
    print("APPROACH 1  --  Dynamic Survival Hazard EWS   (33 bond features)")
    print("=" * 78)

    # ---- Stage 1: data -----------------------------------------------------
    print("\n[Stage 1] Loading 33 time-varying bond features + real dates + event onsets ...")
    panel = load_bond_dated()
    n_firms = panel["firm_id"].nunique()
    n_events = int(panel["event"].sum())
    span = f"{panel['month_year'].min().date()} .. {panel['month_year'].max().date()}"
    print(f"          {len(panel):,} firm-months | {n_firms} firms | period {span}")
    print(f"          real credit-event onsets (restructure and/or default): {n_events} firms")

    # ---- Stage 2-3: discrete-time (Cox-style) hazard -> forward probability -
    print("\n[Stage 2] Fitting time-varying DISCRETE-TIME logistic hazard (Cox-style,")
    print("          fit by pooled logit -- NOT Cox partial likelihood):")
    print("            h(t|X(t)) = sigmoid( baseline(t) + beta' X(t) )")
    print("[Stage 3] Forward probability  P_H(t) = 1 - prod_{k=1..3}(1 - h(t+k|X_t))")
    # drop the extra numeric columns so the auto-covariate selector cannot use them
    df, meta = survival.run(panel.drop(columns=HAZARD_DROP, errors="ignore"))

    hr = odds_ratios(meta["model"])
    print("\n          Top odds ratios exp(beta) per +1 SD  (discrete-time logit; NOT a Cox HR):")
    for name, v in list(hr.items())[:6]:
        print(f"            {name:34s} OR {v:8.2f}")
    print("          NOTE: this is a RELATIVE (odds) risk -- it can NOT be read as lead-time days.")
    print(f"          NOTE: with only {n_events} events these coefficients are UNSTABLE (rare-event")
    print("                overfit); use them for ranking, not literal magnitudes.")

    # ---- Stage 4-5: threshold -> alarm -> lead time -----------------------
    print(f"\n[Stage 4] Warning threshold: alarm when  P(event within 3m | X_t) > {p_star:.2f}")
    print("[Stage 5] Actionable lead = first alarm inside the 1-3M pre-event window")
    print("          Persistent duration = start of the final continuous monthly alarm episode")
    tbl = lead_time_table(df, p_star)

    det = tbl[tbl["status"] == "detected"]
    missed = tbl[tbl["status"] == "missed"]
    censored = tbl[tbl["status"] == "censored"]
    leads = det["lead_days"].astype(float).values

    # ---- Performance -------------------------------------------------------
    def f3(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "n/a"
    print("\n" + "=" * 78)
    print("PERFORMANCE")
    print("=" * 78)
    print("Forward-probability model (PD_3M):")
    print(f"    ROC-AUC   in-sample {f3(meta['pd_auc'])}   out-of-sample {f3(meta['pd_auc_oos'])}"
          f"   persistence-baseline {f3(meta['persistence_auc'])}")
    if meta.get("oos_pd"):
        e = meta["oos_pd"]
        print(f"    PD signal (out-of-sample): MCC {e['MCC']:.3f}  precision {e['precision']:.2f}"
              f"  recall {e['recall']:.2f}  flagged {e['volume']*100:.1f}%")
    print("\nActionable early-warning lead time (1-3M window before real event):")
    print(f"    defaulted firms          : {n_events}")
    print(f"    detected before default  : {len(det)}  ({len(det)/max(n_events,1)*100:.0f}%)")
    print(f"    missed (no alarm before) : {len(missed)}")
    if len(leads):
        print(f"    actionable lead (days) -> median {np.median(leads):.0f}   mean {leads.mean():.0f}"
              f"   min {leads.min():.0f}   max {leads.max():.0f}")
        print(f"    actionable lead (months) -> median {np.median(leads)/DAYS_PER_MONTH:.1f}"
              f"   mean {leads.mean()/DAYS_PER_MONTH:.1f}")
    persistent = pd.to_numeric(tbl["persistent_alarm_days"], errors="coerce").dropna()
    if len(persistent):
        print(f"    persistent duration (days) -> median {persistent.median():.0f}"
              f"   max {persistent.max():.0f}")

    # threshold sensitivity: how detection & lead time trade off with p*
    print("\nThreshold sensitivity (alarm when PD_3M > p*):")
    print("    p*      detected     median actionable (days)   median actionable (months)")
    for ps in [0.50, 0.30, 0.20, 0.10, 0.05]:
        t2 = lead_time_table(df, ps)
        L2 = t2.loc[t2["status"] == "detected", "lead_days"].astype(float).values
        med = f"{np.median(L2):.0f}" if len(L2) else "-"
        medm = f"{np.median(L2)/DAYS_PER_MONTH:.1f}" if len(L2) else "-"
        print(f"    {ps:.2f}    {len(L2):3d}/{n_events:<3d}      {med:>10}           {medm:>8}")
    print("    (lead remains restricted to the 1-3M actionable window at every p*)")

    # latest month per firm = current standing (used for boundary count + alert mix)
    latest = df.sort_values("month_index").groupby("firm_id").tail(1)

    # hyperbolic-boundary alarm, for comparison (LATEST month only, not ever-crossed)
    bnd = meta["boundary"]
    print(f"\nHyperbolic boundary (secondary):  Momentum >= K / PD_prev^alpha"
          f"   [K={bnd['K']:.2f}, alpha={bnd['alpha']:.2f}]")
    print(f"    firms beyond the boundary in their LATEST month: {int((latest['flag_RS'] == 1).sum())}")

    print("\nCurrent alert distribution (latest month per firm):")
    la = [alert_level(p, m, f) for p, m, f in zip(latest["PD_3M"], latest["Momentum"], latest["flag_RS"])]
    vc = pd.Series(la).value_counts()
    for lvl in ["HIGH RISK", "ELEVATED", "WATCH", "OK"]:
        print(f"    {lvl:11s}: {int(vc.get(lvl, 0)):4d}")

    # ---- Tables ------------------------------------------------------------
    det_sorted = det.sort_values("default_date")
    print_table(pd.concat([det_sorted, missed]), "LEAD-TIME TABLE -- defaulted firms")
    top_censored = censored[censored["alert"].isin(["HIGH RISK", "ELEVATED"])] \
        .sort_values("PD_3M", ascending=False)
    if len(top_censored):
        print_table(top_censored, "WATCHLIST -- not yet defaulted but currently ELEVATED / HIGH RISK "
                    "(Lead Time = N/A, censored)", limit=15)

    if csv_out:
        tbl.to_csv(csv_out, index=False)
        print(f"\nFull per-firm table ({len(tbl)} firms) written to {csv_out}")

    print("\n" + "=" * 78)
    print("CAVEATS (read before using these numbers)")
    print("=" * 78)
    print(f"1. Precision is very low: events are {n_events} in {len(df):,} firm-months "
          f"({n_events/len(df)*100:.2f}%). To reach")
    print("   high recall the alarm flags a large share of firm-months -> many false")
    print("   alarms. This is inherent to such a rare-event target.")
    print("2. Actionable lead time is restricted to alarms in the 1-3 calendar-month")
    print("   pre-event window. Persistent Alarm Duration is the separate metric that may")
    print("   span years; missing calendar months break a continuous episode.")
    print("3. Baseline time here is CALENDAR month. The paper's duration since the firm's")
    print("   first bond issuance (issue date from iBond, minus date_DP / date_RS) is not")
    print("   in the .dta yet; add it to get a true time-to-default hazard baseline.")

    print("\nDone.")


if __name__ == "__main__":
    main()
