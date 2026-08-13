# -*- coding: utf-8 -*-
"""
realtime_ews.py -- live early-warning scoring and LEAD TIME for Thai bond issuers.

What it does
------------
1. TRAIN   the discrete-time survival hazard on the historical ThaiBMA panel
           (survivor2 pipeline: hazard h(t|X) -> PD_3M -> momentum -> alarm).
2. REFRESH the issuer snapshot. Preferred source is your own authenticated iBond
           session (see the security note); otherwise it falls back to the most
           recent month already in the local database, clearly labelled.
3. SCORE   every issuer now: PD_3M, momentum, alert level, and the EXPECTED LEAD
           TIME -- how many days of warning this alert historically buys, taken
           from the realised lead-time distribution of firms that actually
           defaulted at the same alert level.
4. STORE   the result in SQLite so the GUI can show it without re-running.

SECURITY
--------
This module never asks for, stores or transmits your password. It calls
`ibond_client`, which reads THAIBMA_USER / THAIBMA_PASS (or THAIBMA_API_KEY)
from your own environment. Set them yourself:

    setx THAIBMA_USER "your_username"
    setx THAIBMA_PASS "your_password"

Never paste a password into a chat, a source file or a screenshot. If one has
been exposed, change it first.

Run:
    python realtime_ews.py                # local snapshot (offline, always works)
    python realtime_ews.py --live         # try iBond first, fall back to local
    python realtime_ews.py --top 30
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import lead_metrics
import survival
import survivor2 as s2
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
T_ALERTS = "realtime_alerts"
T_SUMMARY = "realtime_summary"
T_LEADREF = "realtime_leadtime_ref"

P_STAR = 0.50
BANDS = ["HIGH RISK", "ELEVATED", "WATCH", "OK"]
EXPECTED_LEAD_METHOD = "risk_matched_historical_actionable_1_3m_knn_v1"
LEAD_MATCH_FEATURES = ("PD_3M", "Momentum", "h")
LEAD_NEIGHBORS = 15
MIN_SAME_ALERT_FIRMS = 8
FIRM_NAME_SOURCE = "firm_issuer_mapping.company_name_v1"


def _firm_id_text(value) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(number) and float(number).is_integer():
        return str(int(number))
    return str(value).strip()


def _clean_stata_ticker(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    for suffix in ("m.BK", ".BK"):
        if text.endswith(suffix):
            return text[:-len(suffix)]
    return text


def load_firm_name_mapping(db: str = DB) -> pd.DataFrame:
    """Load the durable numeric firm-id to company-name mapping from SQLite."""
    columns = ["firm_id", "firm_name"]
    try:
        with sqlite3.connect(db) as con:
            mapping = pd.read_sql_query(
                """
                SELECT firm_id, company_name, company_name_th, issuer_code,
                       stata_ticker
                FROM firm_issuer_mapping
                """,
                con,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame(columns=columns)

    if mapping.empty:
        return pd.DataFrame(columns=columns)

    mapping["firm_id"] = pd.to_numeric(mapping["firm_id"], errors="coerce")
    company_name = mapping["company_name"].fillna("").astype(str).str.strip()
    company_name_th = (
        mapping["company_name_th"].fillna("").astype(str).str.strip()
    )
    issuer_code = mapping["issuer_code"].fillna("").astype(str).str.strip()
    stata_ticker = mapping["stata_ticker"].map(_clean_stata_ticker)
    mapping["firm_name"] = (
        company_name.mask(company_name.eq(""), company_name_th)
        .mask(lambda s: s.eq(""), issuer_code)
        .mask(lambda s: s.eq(""), stata_ticker)
    )
    mapping = mapping.dropna(subset=["firm_id"])
    mapping = mapping[mapping["firm_name"].ne("")]
    return mapping[columns].drop_duplicates("firm_id", keep="first")


def attach_firm_names(
    frame: pd.DataFrame,
    mapping: pd.DataFrame | None = None,
    db: str = DB,
) -> pd.DataFrame:
    """Attach a non-empty display name while preserving the numeric firm id."""
    out = frame.copy()
    if "firm_id" not in out.columns:
        out["firm_name"] = ""
        return out

    if mapping is None:
        mapping = load_firm_name_mapping(db)
    name_by_id = {}
    if mapping is not None and not mapping.empty:
        names = mapping.copy()
        names["firm_id"] = pd.to_numeric(names["firm_id"], errors="coerce")
        names["firm_name"] = names["firm_name"].fillna("").astype(str).str.strip()
        names = names.dropna(subset=["firm_id"])
        names = names[names["firm_name"].ne("")]
        name_by_id = (
            names.drop_duplicates("firm_id")
            .set_index("firm_id")["firm_name"]
            .to_dict()
        )

    firm_keys = pd.to_numeric(out["firm_id"], errors="coerce")
    firm_names = firm_keys.map(name_by_id)
    fallback = out["firm_id"].map(
        lambda value: f"Firm ID {_firm_id_text(value)}"
    )
    firm_names = firm_names.fillna(fallback)

    if "firm_name" in out.columns:
        out = out.drop(columns=["firm_name"])
    insert_at = out.columns.get_loc("firm_id") + 1
    out.insert(insert_at, "firm_name", firm_names)
    return out


# ============================================================ alert logic =====
def alert_of(pd3m: float, momentum: float, flag_rs: int) -> str:
    m = 1.0 if momentum is None or momentum != momentum else float(momentum)
    if flag_rs == 1 and pd3m >= P_STAR:
        return "HIGH RISK"
    if pd3m >= 0.50:
        return "HIGH RISK"
    if flag_rs == 1 or pd3m >= 0.15:
        return "ELEVATED"
    if m >= 1.15 or pd3m >= 0.05:
        return "WATCH"
    return "OK"


# ======================================================= lead-time reference ==
def historical_actionable_samples(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Historical alarm states inside the shared 1-3 calendar-month window."""
    columns = [
        "firm_id", "event_date", "observation_date", "alert", "lead_days",
        *LEAD_MATCH_FEATURES,
    ]
    if df_hist is None or df_hist.empty:
        return pd.DataFrame(columns=columns)

    d = df_hist.copy()
    d["month_year"] = pd.to_datetime(d["month_year"], errors="coerce")
    rows = []
    for fid, group in d.groupby("firm_id"):
        group = group.sort_values(["month_year", "month_index"])
        events = group[
            pd.to_numeric(group["event"], errors="coerce").fillna(0) == 1
        ]
        if events.empty:
            continue

        event_date = events["month_year"].iloc[0]
        window_start = event_date - pd.DateOffset(
            months=lead_metrics.LEAD_WINDOW_MAX_MONTHS
        )
        window_end = event_date - pd.DateOffset(
            months=lead_metrics.LEAD_WINDOW_MIN_MONTHS
        )
        window = group[
            group["month_year"].between(
                window_start, window_end, inclusive="both"
            )
        ].copy()

        for _, row in window.iterrows():
            flag_rs = row.get("flag_RS", 0)
            flag_rs = 0 if pd.isna(flag_rs) else int(flag_rs)
            band = alert_of(row["PD_3M"], row.get("Momentum"), flag_rs)
            if band == "OK":
                continue
            lead_days = float((event_date - row["month_year"]).days)
            rows.append({
                "firm_id": fid,
                "event_date": event_date,
                "observation_date": row["month_year"],
                "alert": band,
                "lead_days": lead_days,
                "PD_3M": row.get("PD_3M"),
                "Momentum": row.get("Momentum"),
                "h": row.get("h"),
            })

    samples = pd.DataFrame(rows, columns=columns)
    if samples.empty:
        return samples
    samples["lead_metric_version"] = lead_metrics.LEAD_METRIC_VERSION
    samples["lead_window_min_months"] = (
        lead_metrics.LEAD_WINDOW_MIN_MONTHS
    )
    samples["lead_window_max_months"] = (
        lead_metrics.LEAD_WINDOW_MAX_MONTHS
    )
    return samples


def leadtime_reference(
    df_hist: pd.DataFrame,
    samples: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Actionable 1-3M warning distribution by alert level."""
    samples = (
        historical_actionable_samples(df_hist)
        if samples is None
        else samples.copy()
    )
    base_columns = [
        "alert", "n", "median_days", "p25_days", "p75_days",
        "lead_metric_version", "lead_definition",
        "lead_window_min_months", "lead_window_max_months",
    ]
    if samples.empty:
        return pd.DataFrame(columns=base_columns)

    # One earliest qualifying observation per event firm and alert level.
    first_by_band = (
        samples.sort_values("lead_days", ascending=False)
        .drop_duplicates(["firm_id", "alert"])
    )
    rows = []
    for band in ("WATCH", "ELEVATED", "HIGH RISK"):
        values = first_by_band.loc[
            first_by_band["alert"] == band, "lead_days"
        ].dropna()
        if values.empty:
            continue
        rows.append({
            "alert": band,
            "n": int(len(values)),
            "median_days": float(values.median()),
            "p25_days": float(values.quantile(0.25)),
            "p75_days": float(values.quantile(0.75)),
        })

    first_any = samples.groupby("firm_id")["lead_days"].max().dropna()
    if not first_any.empty:
        rows.append({
            "alert": "ANY ACTIONABLE",
            "n": int(len(first_any)),
            "median_days": float(first_any.median()),
            "p25_days": float(first_any.quantile(0.25)),
            "p75_days": float(first_any.quantile(0.75)),
        })

    out = pd.DataFrame(rows)
    out["lead_metric_version"] = lead_metrics.LEAD_METRIC_VERSION
    out["lead_definition"] = lead_metrics.ACTIONABLE_LEAD_DEFINITION
    out["lead_window_min_months"] = lead_metrics.LEAD_WINDOW_MIN_MONTHS
    out["lead_window_max_months"] = lead_metrics.LEAD_WINDOW_MAX_MONTHS
    return out[base_columns]


def _risk_matrix(frame: pd.DataFrame) -> np.ndarray:
    pd3m = np.clip(
        pd.to_numeric(frame["PD_3M"], errors="coerce").to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )
    momentum = np.clip(
        pd.to_numeric(frame["Momentum"], errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=float),
        1e-6,
        None,
    )
    hazard = np.clip(
        pd.to_numeric(frame["h"], errors="coerce").to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )
    return np.column_stack([
        np.log(pd3m / (1 - pd3m)),
        np.log(momentum),
        np.log(hazard / (1 - hazard)),
    ])


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if len(values) == 0 or cumulative[-1] <= 0:
        return float("nan")
    return float(np.interp(quantile * cumulative[-1], cumulative, values))


def estimate_expected_actionable_lead(
    latest: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Risk-matched historical estimate for each currently alerted issuer.

    This is not an observed future default date. It is the weighted 1-3M lead
    distribution of the nearest historical event firms in PD, momentum and
    hazard space. Each historical firm contributes at most one neighbour.
    """
    out = latest.copy()
    numeric_columns = [
        "expected_lead_days", "expected_lead_months",
        "expected_lead_p25_days", "expected_lead_p75_days",
        "expected_lead_reference_n",
    ]
    for column in numeric_columns:
        out[column] = np.nan
    out["expected_lead_method"] = "not_applicable_no_active_alarm"
    out["expected_lead_pool"] = "none"
    out["lead_metric_version"] = lead_metrics.LEAD_METRIC_VERSION
    out["lead_definition"] = lead_metrics.ACTIONABLE_LEAD_DEFINITION
    out["lead_window_min_months"] = lead_metrics.LEAD_WINDOW_MIN_MONTHS
    out["lead_window_max_months"] = lead_metrics.LEAD_WINDOW_MAX_MONTHS

    if samples is None or samples.empty:
        active = out["alert"] != "OK"
        out.loc[active, "expected_lead_method"] = (
            "unavailable_no_actionable_history"
        )
        return out

    hist_matrix = _risk_matrix(samples)
    centre = np.nanmedian(hist_matrix, axis=0)
    hist_matrix = np.where(np.isfinite(hist_matrix), hist_matrix, centre)
    scale = (
        np.nanpercentile(hist_matrix, 75, axis=0)
        - np.nanpercentile(hist_matrix, 25, axis=0)
    )
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    hist_scaled = (hist_matrix - centre) / scale

    for index, row in out.iterrows():
        band = str(row.get("alert", "OK"))
        if band == "OK":
            continue

        same_alert = samples["alert"].astype(str) == band
        same_firms = samples.loc[same_alert, "firm_id"].nunique()
        if same_firms >= MIN_SAME_ALERT_FIRMS:
            candidate_positions = np.flatnonzero(same_alert.to_numpy())
            pool = "same_alert"
        else:
            candidate_positions = np.arange(len(samples))
            pool = "all_active_alerts"

        current = _risk_matrix(pd.DataFrame([row]))[0]
        current = np.where(np.isfinite(current), current, centre)
        current = (current - centre) / scale
        distances = np.sqrt(
            np.sum(
                (hist_scaled[candidate_positions] - current) ** 2,
                axis=1,
            )
        )
        candidates = samples.iloc[candidate_positions].copy()
        candidates["_position"] = candidate_positions
        candidates["_distance"] = distances
        neighbours = (
            candidates.sort_values("_distance")
            .drop_duplicates("firm_id")
            .head(LEAD_NEIGHBORS)
        )
        if neighbours.empty:
            out.at[index, "expected_lead_method"] = (
                "unavailable_no_actionable_history"
            )
            continue

        lead_days = pd.to_numeric(
            neighbours["lead_days"], errors="coerce"
        ).to_numpy(dtype=float)
        distances = neighbours["_distance"].to_numpy(dtype=float)
        valid = np.isfinite(lead_days) & np.isfinite(distances)
        lead_days = lead_days[valid]
        distances = distances[valid]
        if len(lead_days) == 0:
            continue

        weights = 1.0 / (distances + 0.25)
        expected = float(np.average(lead_days, weights=weights))
        p25 = _weighted_quantile(lead_days, weights, 0.25)
        p75 = _weighted_quantile(lead_days, weights, 0.75)
        out.at[index, "expected_lead_days"] = expected
        out.at[index, "expected_lead_months"] = (
            expected / lead_metrics.DAYS_PER_MONTH
        )
        out.at[index, "expected_lead_p25_days"] = p25
        out.at[index, "expected_lead_p75_days"] = p75
        out.at[index, "expected_lead_reference_n"] = int(len(lead_days))
        out.at[index, "expected_lead_method"] = EXPECTED_LEAD_METHOD
        out.at[index, "expected_lead_pool"] = pool

    return out


# =============================================================== live data ====
def fetch_live_snapshot():
    """Latest issuer features from your authenticated iBond session.

    Returns (DataFrame|None, note). Never raises: if the session or the endpoint is
    unavailable we say so and the caller falls back to the local snapshot.
    """
    try:
        import ibond_client as ib
    except Exception as ex:
        return None, f"ibond_client unavailable ({ex})"
    st = ib.credentials_status()
    if not st["ready"]:
        return None, ("no THAIBMA_USER / THAIBMA_PASS (or THAIBMA_API_KEY) in the "
                      "environment — set them yourself, then retry")
    try:
        df = ib.fetch_curve()          # proves the session works end to end
        return df, f"iBond session OK ({len(df):,} rows)"
    except Exception as ex:
        return None, f"iBond fetch failed: {str(ex)[:200]}"


# ================================================================= scoring ====
def run(live: bool = False, verbose: bool = True):
    if verbose:
        print("[1/4] loading historical panel ...")
    panel = s2.load_bond_dated()
    clean = panel.drop(columns=s2.HAZARD_DROP, errors="ignore")

    if verbose:
        print("[2/4] fitting hazard and building signals ...")
    df, meta = survival.run(clean)
    if "month_year" not in df.columns:          # only merge if the pipeline dropped it
        df = df.merge(panel[["firm_id", "month_index", "month_year"]],
                      on=["firm_id", "month_index"], how="left")
    df["month_year"] = pd.to_datetime(df["month_year"])

    live_note = "local snapshot (offline)"
    if live:
        _snap, live_note = fetch_live_snapshot()
        if verbose:
            print(f"      live source: {live_note}")

    if verbose:
        print("[3/4] computing lead-time reference ...")
    lead_samples = historical_actionable_samples(df)
    ref = leadtime_reference(df, samples=lead_samples)

    if verbose:
        print("[4/4] scoring the latest month per issuer ...")
    latest = df.sort_values("month_index").groupby("firm_id").tail(1).copy()
    # firms whose panel ends at their own credit event have ALREADY defaulted --
    # they are history, not a live watchlist item.
    defaulted = set(df.loc[df["event"] == 1, "firm_id"].unique())
    latest["already_defaulted"] = latest["firm_id"].isin(defaulted).astype(int)
    latest = latest[latest["already_defaulted"] == 0].copy()
    latest = attach_firm_names(latest)
    latest["alert"] = [alert_of(p, m, int(f)) for p, m, f in
                       zip(latest["PD_3M"], latest["Momentum"], latest["flag_RS"].fillna(0))]
    latest = estimate_expected_actionable_lead(latest, lead_samples)
    latest["as_of"] = latest["month_year"].dt.strftime("%Y-%m")

    keep = ["firm_id", "firm_name", "as_of", "PD_3M", "Momentum", "h", "flag_PD", "flag_RS",
            "alert", "expected_lead_days", "expected_lead_months",
            "expected_lead_p25_days", "expected_lead_p75_days",
            "expected_lead_reference_n", "expected_lead_method",
            "expected_lead_pool", "lead_metric_version", "lead_definition",
            "lead_window_min_months", "lead_window_max_months"]
    if "dd_12m" in latest.columns:
        keep.insert(5, "dd_12m")
    alerts = (latest[keep].sort_values(["alert", "PD_3M"],
                                       key=lambda s: s.map({a: i for i, a in enumerate(BANDS)})
                                       if s.name == "alert" else s,
                                       ascending=[True, False])
              .reset_index(drop=True))

    vc = alerts["alert"].value_counts()
    any_reference = ref[ref["alert"] == "ANY ACTIONABLE"]
    median_lead_all = (
        float(any_reference["median_days"].iloc[0])
        if not any_reference.empty
        else float("nan")
    )
    summary = {
        **lead_metrics.summary_metadata(),
        "data_source": live_note,
        "is_live": int(live_note.startswith("iBond session OK")),
        "as_of": str(alerts["as_of"].max()),
        "n_firms": int(len(alerts)),
        "n_high": int(vc.get("HIGH RISK", 0)), "n_elevated": int(vc.get("ELEVATED", 0)),
        "n_watch": int(vc.get("WATCH", 0)), "n_ok": int(vc.get("OK", 0)),
        "pd_auc_oos": float(meta.get("pd_auc_oos") or float("nan")),
        "median_lead_days_all": median_lead_all,
        "expected_lead_method": EXPECTED_LEAD_METHOD,
        "firm_name_source": FIRM_NAME_SOURCE,
        "n_firm_names": int(
            alerts["firm_name"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        "n_actionable_reference_samples": int(len(lead_samples)),
        "n_actionable_reference_firms": int(
            lead_samples["firm_id"].nunique()
            if not lead_samples.empty else 0
        ),
        "n_events_hist": int(panel["event"].sum()),
        "n_excluded_defaulted": int(len(defaulted)),
        "n_firm_months": int(len(panel)),
    }
    return alerts, ref, summary


def save_to_sqlite(alerts, ref, summary, db=DB):
    con = sqlite3.connect(db)
    alerts.to_sql(T_ALERTS, con, if_exists="replace", index=False)
    ref.to_sql(T_LEADREF, con, if_exists="replace", index=False)
    pd.DataFrame([summary]).to_sql(T_SUMMARY, con, if_exists="replace", index=False)
    con.commit(); con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    try:
        a = pd.read_sql_query(f"SELECT * FROM {T_ALERTS}", con)
        r = pd.read_sql_query(f"SELECT * FROM {T_LEADREF}", con)
        s = pd.read_sql_query(f"SELECT * FROM {T_SUMMARY} LIMIT 1", con)
    except Exception:
        a = r = s = pd.DataFrame()
    finally:
        con.close()
    return a, r, s


def main():
    live = "--live" in sys.argv
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    print("=" * 96)
    print("REAL-TIME EARLY WARNING -- current alert and expected lead time per issuer")
    print("=" * 96)
    alerts, ref, summary = run(live=live)
    if not summary["is_live"]:
        print(f"\n  data source: {summary['data_source']}")
        print("  (scores come from the newest month in the local database; supply your own")
        print("   iBond credentials via THAIBMA_USER / THAIBMA_PASS for a live refresh)")

    print(f"\nas of {summary['as_of']} | {summary['n_firms']} issuers | "
          f"hazard OOS AUC {summary['pd_auc_oos']:.3f}")
    print(f"  HIGH RISK {summary['n_high']}   ELEVATED {summary['n_elevated']}   "
          f"WATCH {summary['n_watch']}   OK {summary['n_ok']}")

    print("\nEXPECTED LEAD TIME BY ALERT LEVEL (from firms that actually defaulted)")
    print(f"  {'alert':11s} {'n':>4s} {'median d':>9s} {'p25':>7s} {'p75':>7s}")
    for _, r in ref.iterrows():
        print(f"  {r['alert']:11s} {int(r['n']):4d} {r['median_days']:9.0f} "
              f"{r['p25_days']:7.0f} {r['p75_days']:7.0f}")

    print(f"\nTOP {top} ISSUERS BY PD_3M")
    print(f"  {'firm':>6s} {'as_of':>8s} {'PD_3M':>8s} {'Mom':>7s} {'h(t)':>8s} "
          f"{'alert':>11s} {'exp lead':>10s}")
    for _, r in alerts.head(top).iterrows():
        el = "-" if pd.isna(r["expected_lead_days"]) else f"{r['expected_lead_days']:.0f}d"
        print(f"  {int(r['firm_id']):6d} {r['as_of']:>8s} {r['PD_3M']*100:7.1f}% "
              f"{(r['Momentum'] if pd.notna(r['Momentum']) else 1.0):7.2f} "
              f"{r['h']*100:7.2f}% {r['alert']:>11s} {el:>10s}")

    save_to_sqlite(alerts, ref, summary)
    print(f"\nSaved: {T_ALERTS}, {T_LEADREF}, {T_SUMMARY}  ({DB})")
    print("Done.")


if __name__ == "__main__":
    main()
