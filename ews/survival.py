# -*- coding: utf-8 -*-
"""
Survival EWS on the firm panel — the PD_3M / momentum / hyperbolic-boundary
machinery (Cox-style discrete-time hazard) applied to the real (or synthetic)
monthly panel, so the credit app and the survival framework share one dataset.

Each firm is treated as an "account" observed monthly; the model gives:
  h(t|X)      monthly distress hazard (pooled logistic)
  PD_3M(t)    = 1 - prod_{k=1..3}(1 - h(t+k | X(t)))
  Momentum(t) = PD_3M(t) / PD_3M(t-1)
  RS flag     : Momentum(t) >= K / PD_3M(t-1)^alpha   (hyperbolic boundary)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import lead_metrics
try:
    from load_bond import BOND_FEATURES as BOND_33_COVS
except Exception:
    BOND_33_COVS = []

# compact, real, *time-varying* covariates for the hazard (from the 33 features)
HAZ_COVS = ["distance_to_default", "equity_vol", "delinquency_trend",
            "momentum_12m", "credit_score_pd", "market_leverage",
            "credit_spread", "yield_slope"]
HORIZON = 3
_DEG = 3
_ACCT_CANDIDATES = ("firm_id", "ticker", "account_id")
_EVENT_CANDIDATES = ("default_event", "observed_default", "event",
                     "d_Default_Payment", "d_DP_RS", "d_Restructure")
_DATE_CANDIDATES = ("month_year", "month", "as_of_date", "date")


def _acct_col(df):
    return next((c for c in _ACCT_CANDIDATES if c in df.columns), "account_id")


def _event_col(df):
    return next((c for c in _EVENT_CANDIDATES if c in df.columns), None)


def _date_col(df):
    return next((c for c in _DATE_CANDIDATES if c in df.columns), None)


def prepare_panel(panel):
    """Normalize the observed default onset used by Path 1.

    `default_3m` is a forward early-warning label, not an observed default.
    When a real onset column is available, keep `event` pinned to that onset so
    censored/pre-default rows are not accidentally treated as default rows.
    """
    d = panel.copy()
    if "firm_id" not in d.columns and "issuer_code" in d.columns:
        d["firm_id"] = d["issuer_code"].astype(str)
    acct = _acct_col(d)
    if acct not in d.columns:
        d["account_id"] = np.arange(len(d), dtype=int)
        acct = "account_id"

    if "default_event" in d.columns:
        de = pd.to_numeric(d["default_event"], errors="coerce").fillna(0).astype(int)
        if de.sum() > 0:
            d["event"] = de
    if "event" not in d.columns:
        src = next((c for c in _EVENT_CANDIDATES if c in d.columns and c != "event"), None)
        if src:
            flag = (pd.to_numeric(d[src], errors="coerce").fillna(0) > 0).astype(int)
            d["event"] = ((flag == 1) & (flag.groupby(d[acct]).cumsum() == 1)).astype(int)
        else:
            d["event"] = 0

    if "month_index" in d.columns:
        d["month_index"] = pd.to_numeric(d["month_index"], errors="coerce")
        if d["month_index"].isna().all():
            dc = _date_col(d)
            if dc:
                d["month_index"] = pd.factorize(pd.to_datetime(d[dc], errors="coerce"), sort=True)[0] + 1
            else:
                d["month_index"] = np.arange(len(d)) + 1
    else:
        dc = _date_col(d)
        if dc:
            d["month_index"] = pd.factorize(pd.to_datetime(d[dc], errors="coerce"), sort=True)[0] + 1
        else:
            d["month_index"] = np.arange(len(d)) + 1

    d = d.dropna(subset=["month_index"]).copy()
    d = d.sort_values([acct, "month_index"]).reset_index(drop=True)
    raw_event = (
        pd.to_numeric(d["event"], errors="coerce").fillna(0).gt(0)
    )
    first_event = raw_event & (
        raw_event.astype(int).groupby(d[acct]).cumsum() == 1
    )
    d["event"] = first_event.astype(int)

    # A survival risk set ends at the first observed event. Rows after that
    # onset must not re-enter training as if the issuer were still at risk.
    event_seen = d["event"].groupby(d[acct]).cumsum()
    d = d[(event_seen == 0) | d["event"].eq(1)].reset_index(drop=True)
    return d


def _num(row, col, default=0.0):
    v = row.get(col, default)
    if pd.isna(v):
        return default
    try:
        return float(v)
    except Exception:
        return default


def _flag(row, col):
    return int(_num(row, col, 0.0) > 0)


def alarm_source(row):
    src = []
    if _flag(row, "flag_PD"):
        src.append("PD")
    if _flag(row, "flag_RS"):
        src.append("RS")
    return "+".join(src) if src else "None"


def alert_level(row):
    pd3m = _num(row, "PD_3M", 0.0)
    mom = _num(row, "Momentum", 1.0)
    if _flag(row, "flag_PD") and _flag(row, "flag_RS"):
        return "HIGH RISK"
    if _flag(row, "flag_RS") or pd3m >= 0.15:
        return "ELEVATED"
    if mom >= 1.15 or pd3m >= 0.05:
        return "WATCH"
    return "OK"


def _dt(row, dc):
    if not dc:
        return pd.NaT
    return pd.to_datetime(row.get(dc), errors="coerce")


def _date_label(row, dc):
    dt = _dt(row, dc)
    if pd.notna(dt):
        return dt.date().isoformat()
    mi = row.get("month_index")
    if pd.notna(mi):
        try:
            return f"month {int(float(mi))}"
        except Exception:
            return str(mi)
    return None


def _numeric_series(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def _month_ordinal(values):
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.year * 12 + dates.dt.month


def _forward_event_target(df, acct, horizon=HORIZON):
    """Observed default in the next `horizon` calendar months.

    A row-based shift is incorrect for sparse panels because three observations
    can span years. Prefer calendar months and fall back to the global month
    index only when the panel has no usable date column.
    """
    dc = _date_col(df)
    if dc:
        current = _month_ordinal(df[dc])
    else:
        current = pd.to_numeric(df.get("month_index"), errors="coerce")
    event_time = current.where(_numeric_series(df, "event", 0) > 0)
    event_time = event_time.groupby(df[acct]).transform("min")
    months_ahead = event_time - current
    return ((months_ahead >= 1) & (months_ahead <= horizon)).astype(float)


def _months_before_default(rows, default_row, dc):
    if dc:
        default_dt = _dt(default_row, dc)
        if pd.notna(default_dt):
            default_ord = default_dt.year * 12 + default_dt.month
            return default_ord - _month_ordinal(rows[dc])
    default_ord = pd.to_numeric(
        pd.Series([default_row.get("month_index")]), errors="coerce"
    ).iloc[0]
    current = pd.to_numeric(rows.get("month_index"), errors="coerce")
    return default_ord - current


def _lead_days(alarm_row, default_row, dc):
    adt, ddt = _dt(alarm_row, dc), _dt(default_row, dc)
    if pd.notna(adt) and pd.notna(ddt):
        return int((ddt - adt).days)
    ami = pd.to_numeric(pd.Series([alarm_row.get("month_index")]), errors="coerce").iloc[0]
    dmi = pd.to_numeric(pd.Series([default_row.get("month_index")]), errors="coerce").iloc[0]
    if pd.notna(ami) and pd.notna(dmi):
        return int(round((float(dmi) - float(ami)) * 30.4375))
    return pd.NA


def compute_lead_time(df):
    """One row per firm/account with first qualifying EWS alarm lead time.

    Path 1 predicts observed default over a three-calendar-month horizon, so a
    qualifying lead-time alarm must fall inside that same pre-default target
    window. Earlier threshold crossings remain false alarms for MCC/flag-volume
    evaluation and are not reported as multi-year lead time. Non-defaulted firms
    remain in the table with N/A lead time because they are censored.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    d = prepare_panel(df)
    acct = _acct_col(d)
    ev_col = _event_col(d) or "event"
    dc = _date_col(d)
    d["_event_obs"] = (_numeric_series(d, ev_col, 0) > 0).astype(int)
    d["_time_ord"] = _numeric_series(d, "month_index", np.nan)
    if d["_time_ord"].isna().all() and dc:
        d["_time_ord"] = pd.to_datetime(d[dc], errors="coerce").astype("int64")
    d["_alarm"] = ((_numeric_series(d, "flag_PD", 0) > 0)
                   | (_numeric_series(d, "flag_RS", 0) > 0))
    d = d.sort_values([acct, "_time_ord"]).reset_index(drop=True)

    records = []
    for acct_val, g in d.groupby(acct, dropna=False):
        g = g.sort_values("_time_ord")
        latest = g.iloc[-1]
        defaults = g[g["_event_obs"] == 1]
        default_row = defaults.iloc[0] if not defaults.empty else None
        has_default = default_row is not None
        false_alarms_before_window = 0
        metrics = lead_metrics.compute_lead_metrics(
            g,
            event_date=default_row.get(dc) if has_default and dc else pd.NaT,
            event_month_ordinal=(
                None if dc or not has_default else default_row["_time_ord"]
            ),
            date_col=dc,
            alarm_mask=g["_alarm"],
        )
        first_alarm = None
        alarm_idx = metrics.get("actionable_alarm_index")
        if alarm_idx is not None and alarm_idx in g.index:
            first_alarm = g.loc[alarm_idx]

        if has_default:
            before_default = g["_time_ord"] < default_row["_time_ord"]
            if dc and pd.notna(_dt(default_row, dc)):
                old_cutoff = _dt(default_row, dc) - pd.DateOffset(
                    months=lead_metrics.LEAD_WINDOW_MAX_MONTHS
                )
                row_dates = pd.to_datetime(g[dc], errors="coerce")
                false_alarms_before_window = int(
                    (before_default & (row_dates < old_cutoff) & g["_alarm"]).sum()
                )
            else:
                months_before = _months_before_default(g, default_row, dc)
                false_alarms_before_window = int(
                    (
                        before_default
                        & (months_before > lead_metrics.LEAD_WINDOW_MAX_MONTHS)
                        & g["_alarm"]
                    ).sum()
                )

        rec = {
            "firm_id": acct_val if acct == "firm_id" else latest.get("firm_id", pd.NA),
            "account_id": latest.get("account_id", acct_val),
            "ticker": latest.get("ticker", pd.NA),
            "default_observed": bool(has_default),
            "default_date": _date_label(default_row, dc) if has_default else None,
            "lead_time_window_months": HORIZON,
            "false_alarms_before_window": false_alarms_before_window,
            "alarm_source": alarm_source(first_alarm) if first_alarm is not None else "None",
            "latest_PD_3M": latest.get("PD_3M", pd.NA),
            "latest_Momentum": latest.get("Momentum", pd.NA),
            "alert_level": alert_level(latest),
            "first_alarm_month_index": first_alarm.get("month_index", pd.NA) if first_alarm is not None else pd.NA,
            "default_month_index": default_row.get("month_index", pd.NA) if has_default else pd.NA,
            **lead_metrics.strip_internal_fields(metrics),
        }
        if has_default and first_alarm is None:
            rec["alarm_source"] = "No alarm in 3M pre-default window"
        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(["default_observed", "lead_time_days", "latest_PD_3M"],
                           ascending=[False, False, False], na_position="last").reset_index(drop=True)


def _get_covs(df, covs=None):
    if covs is not None:
        avail = [c for c in covs if c in df.columns]
        if avail: return avail
    avail = [c for c in HAZ_COVS if c in df.columns]
    if len(avail) >= 3:
        return avail
    bond_avail = [c for c in BOND_33_COVS if c in df.columns]
    if len(bond_avail) >= 17:
        return bond_avail
    skip = {"id", "account_id", "firm_id", "ticker", "month_index", "month_year",
            "default_3m", "event", "default_event", "dd_12m", "pd_12m",
            "h", "PD_3M", "PD_prev", "Momentum", "y_fwd", "flag_PD", "flag_RS"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def _base(mi, tref):
    m = (np.asarray(mi, float) - tref["mu"]) / tref["sd"]
    return np.column_stack([m ** d for d in range(1, _DEG + 1)])


def fit_hazard(df, covs=None):
    covs = _get_covs(df, covs)
    d = df.copy()
    for c in covs:                                   # standardize covariates
        vals = pd.to_numeric(d[c], errors="coerce")
        med = vals.median()
        d[c] = (vals.fillna(med).fillna(0.0) - vals.mean()) / (vals.std() + 1e-9)
    tref = {"mu": d["month_index"].mean(), "sd": d["month_index"].std() + 1e-9}
    X = np.column_stack([_base(d["month_index"].values, tref), d[covs].values])
    X = np.nan_to_num(X, nan=0.0)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, d["event"].astype(int))
    stats = {c: (df[c].mean(), df[c].std() + 1e-9) for c in covs}
    return {"clf": clf, "tref": tref, "covs": covs, "stats": stats}


def _haz(model, mi, Xstd):
    X = np.column_stack([_base(mi, model["tref"]), Xstd])
    X = np.nan_to_num(X, nan=0.0)
    return model["clf"].predict_proba(X)[:, 1]


def add_signals(df, model, horizon=HORIZON):
    acct = "firm_id" if "firm_id" in df.columns else ("ticker" if "ticker" in df.columns else "account_id")
    d = df.sort_values([acct, "month_index"]).reset_index(drop=True)
    covs = model["covs"]
    Xstd_cols = []
    for c in covs:
        vals = pd.to_numeric(d[c], errors="coerce")
        std_v = (vals.fillna(model["stats"][c][0]) - model["stats"][c][0]) / model["stats"][c][1]
        Xstd_cols.append(np.nan_to_num(std_v.values, nan=0.0))
    Xstd = np.column_stack(Xstd_cols)
    mi = d["month_index"].values.astype(float)
    d["h"] = _haz(model, mi, Xstd)
    surv = np.ones(len(d))
    for k in range(1, horizon + 1):
        surv *= (1 - np.clip(_haz(model, mi + k, Xstd), 1e-9, 0.999))
    d["PD_3M"] = 1 - surv
    d["PD_prev"] = d.groupby(acct)["PD_3M"].shift(1)
    d["Momentum"] = d["PD_3M"] / d["PD_prev"]

    d["y_fwd"] = _forward_event_target(d, acct, horizon)
    return d


def tune_boundary(df, alphas=np.linspace(0.2, 2.5, 20)):
    d = df.dropna(subset=["PD_prev", "Momentum", "y_fwd"])
    d = d[(d["PD_prev"] > 0) & (d["Momentum"] > 0)]
    lM, lP, y = np.log(d["Momentum"]), np.log(d["PD_prev"]), d["y_fwd"].astype(int).values
    best = None
    for a in alphas:
        s = lM + a * lP
        for q in np.linspace(0.5, 0.99, 30):
            tau = np.quantile(s, q); f = (s >= tau).astype(int)
            if f.sum() == 0 or len(set(f)) < 2:
                continue
            mcc = matthews_corrcoef(y, f)
            if best is None or mcc > best["MCC"]:
                best = dict(alpha=float(a), logK=float(tau), K=float(np.exp(tau)), MCC=float(mcc))
    return best


def apply_signals(df, alpha, logK, tau_pd):
    df = df.copy()
    df["flag_PD"] = (df["PD_3M"] >= tau_pd).astype(int)
    m = df["PD_prev"].notna() & (df["PD_prev"] > 0) & (df["Momentum"] > 0)
    df["flag_RS"] = 0
    s = np.log(df.loc[m, "Momentum"]) + alpha * np.log(df.loc[m, "PD_prev"])
    df.loc[m, "flag_RS"] = (s >= logK).astype(int)
    return df


def evaluate(df, col):
    d = df.dropna(subset=["y_fwd"])
    y, f = d["y_fwd"].astype(int).values, d[col].astype(int).values
    mcc = matthews_corrcoef(y, f) if len(set(f)) > 1 else 0.0
    tp = int(((f == 1) & (y == 1)).sum()); fp = int(((f == 1) & (y == 0)).sum()); fn = int(((f == 0) & (y == 1)).sum())
    return dict(MCC=mcc, precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0, volume=float(f.mean()))


def _auc(y, s):
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def run(panel, train_frac=0.7):
    """Fit the survival EWS and report BOTH in-sample and honest out-of-sample
    numbers, plus a persistence baseline (predict the forward event from the
    current state alone). The returned `df` carries the in-sample signals so the
    plots / trajectory panel stay descriptive; the reported metrics come from the
    time-split hold-out.
    """
    panel = prepare_panel(panel)
    # --- in-sample model on the full panel (drives plots + boundary shape) ----
    model = fit_hazard(panel)
    df = add_signals(panel, model)
    bnd = tune_boundary(df)
    tau_pd = float(np.nanquantile(df["PD_3M"], 0.85))
    df = apply_signals(df, bnd["alpha"], bnd["logK"], tau_pd)
    lead_time = compute_lead_time(df)
    d_in = df.dropna(subset=["y_fwd"])
    auc_in = _auc(d_in["y_fwd"], d_in["PD_3M"])

    # --- out-of-sample: fit on the earlier months, evaluate on the later ones -
    months = np.sort(panel["month_index"].unique())
    meta = dict(model=model, boundary=bnd, tau_pd=tau_pd, pd_auc=auc_in,
                pd_auc_oos=float("nan"), persistence_auc=float("nan"),
                cut_month=None, n_test=0, oos_pd=None, oos_rs=None, boundary_oos=bnd,
                lead_time=lead_time)

    # split on EVENT timing when events are rare, so ~train_frac of the onsets
    # land in the training window (a plain time-quantile can leave train event-free)
    onsets = np.sort(panel.loc[panel["event"] == 1, "month_index"].unique())
    if len(onsets) >= 4:
        cut = int(onsets[max(1, int(len(onsets) * train_frac)) - 1])
    elif len(months) >= 8:
        cut = int(months[int(len(months) * train_frac)])
    else:
        cut = None

    if cut is not None:
        tr_panel = panel[panel["month_index"] <= cut]
        if tr_panel["event"].sum() >= 2 and tr_panel["event"].nunique() > 1:
            try:
                m_tr = fit_hazard(tr_panel)
                full = add_signals(panel, m_tr)
                tr = full[full["month_index"] <= cut]
                bnd_o = tune_boundary(tr) or bnd
                tau_o = float(np.nanquantile(tr["PD_3M"], 0.85))
                te = full[full["month_index"] > cut].dropna(subset=["y_fwd"]).copy()
                if len(te) and te["y_fwd"].nunique() > 1:
                    te = apply_signals(te, bnd_o["alpha"], bnd_o["logK"], tau_o)
                    meta.update(pd_auc_oos=_auc(te["y_fwd"], te["PD_3M"]),
                                persistence_auc=_auc(te["y_fwd"], te["event"]),
                                cut_month=cut, n_test=int(len(te)),
                                oos_pd=evaluate(te, "flag_PD"), oos_rs=evaluate(te, "flag_RS"),
                                boundary_oos=bnd_o)
            except Exception:
                pass
    return df, meta


if __name__ == "__main__":
    import os
    panel = pd.read_excel(os.path.join(os.path.dirname(__file__), "real_panel.xlsx"))
    df, meta = run(panel)
    print(f"firms {panel['account_id'].nunique()}  firm-months {len(panel):,}")
    print(f"PD_3M AUC  in-sample {meta['pd_auc']:.3f}   out-of-sample {meta['pd_auc_oos']:.3f}   "
          f"persistence-baseline {meta['persistence_auc']:.3f}  (test rows {meta['n_test']:,})")
    print(f"boundary: K={meta['boundary']['K']:.3f}, alpha={meta['boundary']['alpha']:.2f}")
    for name, e in [("PD signal", meta["oos_pd"]), ("RS signal", meta["oos_rs"])]:
        if e:
            print(f"  [OOS] {name:10s} MCC {e['MCC']:.3f}  precision {e['precision']:.2f}  "
                  f"recall {e['recall']:.2f}  volume {e['volume']*100:.1f}%")
