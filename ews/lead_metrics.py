# -*- coding: utf-8 -*-
"""Shared lead-time definitions for the CMDF credit app.

The project reports two separate alarm timing metrics:

* Actionable Lead Time: the first alarm inside the 1-3 calendar month window
  before an observed event. This remains the backward-compatible lead_days /
  lead_months value.
* Persistent Alarm Duration: the start of the final continuous monthly alarm
  episode before the event. This can be much longer and must not be labelled as
  the 3M actionable lead.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


LEAD_METRIC_VERSION = "lead_metrics_actionable_1_3m_persistent_v1"
LEAD_WINDOW_MIN_MONTHS = 1
LEAD_WINDOW_MAX_MONTHS = 3
DAYS_PER_MONTH = 30.4375

ACTIONABLE_LEAD_DEFINITION = (
    "Actionable Lead Time is the first alarm inside the 1-3 calendar-month "
    "window before an observed event; stored in lead_days/lead_months for "
    "backward compatibility."
)
PERSISTENT_DEFINITION = (
    "Persistent Alarm Duration starts at the first month of the final continuous "
    "monthly alarm episode before the observed event; continuity is checked by "
    "actual calendar months, so missing months break the episode."
)

INTERNAL_FIELDS = {
    "actionable_alarm_index",
    "persistent_alarm_start_index",
    "persistent_alarm_end_index",
}


def run_timestamp() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def summary_metadata() -> dict[str, Any]:
    return {
        "run_at": run_timestamp(),
        "lead_metric_version": LEAD_METRIC_VERSION,
        "lead_definition": ACTIONABLE_LEAD_DEFINITION,
        "lead_window_min_months": LEAD_WINDOW_MIN_MONTHS,
        "lead_window_max_months": LEAD_WINDOW_MAX_MONTHS,
        "persistent_definition": PERSISTENT_DEFINITION,
    }


def strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in INTERNAL_FIELDS}


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def to_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    if isinstance(value, pd.Period):
        return value.to_timestamp()
    try:
        if pd.isna(value):
            return pd.NaT
    except Exception:
        pass
    return pd.to_datetime(value, errors="coerce")


def month_ordinal(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, pd.Period):
        return float(value.year * 12 + value.month)
    ts = to_timestamp(value)
    if pd.notna(ts):
        return float(ts.year * 12 + ts.month)
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def date_label(value: Any) -> str | None:
    ts = to_timestamp(value)
    if pd.notna(ts):
        return ts.date().isoformat()
    if value is None or _is_missing(value):
        return None
    return str(value)


def _coerce_alarm_mask(rows: pd.DataFrame, alarm_mask: Any) -> pd.Series:
    if alarm_mask is None:
        if "alarm" in rows.columns:
            return pd.to_numeric(rows["alarm"], errors="coerce").fillna(0).astype(float) > 0
        raise ValueError("alarm_mask is required when rows has no 'alarm' column")
    if isinstance(alarm_mask, str):
        return pd.to_numeric(rows[alarm_mask], errors="coerce").fillna(0).astype(float) > 0
    if isinstance(alarm_mask, pd.Series):
        return alarm_mask.reindex(rows.index).fillna(False).astype(bool)
    values = list(alarm_mask) if isinstance(alarm_mask, Iterable) else alarm_mask
    return pd.Series(values, index=rows.index).fillna(False).astype(bool)


def _date_values(rows: pd.DataFrame, date_col: str | None) -> pd.Series:
    if date_col and date_col in rows.columns:
        values = rows[date_col]
    else:
        values = pd.Series(pd.NaT, index=rows.index)
    return values


def _prepare_rows(
    rows: pd.DataFrame,
    *,
    date_col: str | None,
    alarm_mask: Any,
) -> pd.DataFrame:
    d = rows.copy()
    d["_source_index"] = d.index
    d["_alarm"] = _coerce_alarm_mask(d, alarm_mask)
    values = _date_values(d, date_col)
    d["_date_value"] = values
    d["_alarm_ts"] = values.map(to_timestamp)
    d["_month_ord"] = values.map(month_ordinal)
    if d["_month_ord"].isna().all() and "month_index" in d.columns:
        d["_month_ord"] = pd.to_numeric(d["month_index"], errors="coerce")
    return d.sort_values(["_month_ord", "_alarm_ts"], na_position="last")


def _days_between(start_value: Any, event_value: Any, start_ord: Any, event_ord: Any) -> float:
    start_ts = to_timestamp(start_value)
    event_ts = to_timestamp(event_value)
    if pd.notna(start_ts) and pd.notna(event_ts):
        return float((event_ts - start_ts).days)
    try:
        so, eo = float(start_ord), float(event_ord)
        if np.isfinite(so) and np.isfinite(eo):
            return float(round((eo - so) * DAYS_PER_MONTH))
    except Exception:
        pass
    return np.nan


def _first_alarm_per_month(pre: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate rows within a calendar month without hiding gaps."""
    rows = []
    for _, g in pre.groupby("_month_ord", sort=True, dropna=True):
        alarm_rows = g[g["_alarm"]]
        if not alarm_rows.empty:
            row = alarm_rows.iloc[0].copy()
            row["_alarm"] = True
        else:
            row = g.iloc[-1].copy()
            row["_alarm"] = False
        rows.append(row)
    if not rows:
        return pre.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def compute_lead_metrics(
    rows: pd.DataFrame,
    *,
    event_date: Any,
    date_col: str | None,
    alarm_mask: Any,
    event_month_ordinal: Any = None,
    window_min_months: int = LEAD_WINDOW_MIN_MONTHS,
    window_max_months: int = LEAD_WINDOW_MAX_MONTHS,
) -> dict[str, Any]:
    """Return actionable and persistent alarm timing for one entity.

    The returned dict includes internal index fields so callers can copy PD,
    Momentum, or alert labels from the selected rows. Drop them before writing
    to SQLite with strip_internal_fields().
    """
    event_ts = to_timestamp(event_date)
    event_ord = (
        month_ordinal(event_date)
        if event_month_ordinal is None
        else float(event_month_ordinal)
    )
    base = {
        "first_alarm": None,
        "first_alarm_date": None,
        "default_date": date_label(event_date),
        "lead_days": np.nan,
        "lead_months": np.nan,
        "lead_time_days": np.nan,
        "lead_time_months": np.nan,
        "actionable_alarm_found": False,
        "qualifying_alarm_found": False,
        "actionable_months_before_default": np.nan,
        "lead_metric_version": LEAD_METRIC_VERSION,
        "lead_definition": ACTIONABLE_LEAD_DEFINITION,
        "lead_window_min_months": int(window_min_months),
        "lead_window_max_months": int(window_max_months),
        "persistent_alarm_start": None,
        "persistent_alarm_end": None,
        "persistent_alarm_days": np.nan,
        "persistent_alarm_months": np.nan,
        "persistent_months_before_default": np.nan,
        "persistent_definition": PERSISTENT_DEFINITION,
        "actionable_alarm_index": None,
        "persistent_alarm_start_index": None,
        "persistent_alarm_end_index": None,
    }
    if rows is None or rows.empty or not np.isfinite(event_ord):
        return base

    d = _prepare_rows(rows, date_col=date_col, alarm_mask=alarm_mask)
    if pd.notna(event_ts):
        before_event = d["_alarm_ts"].notna() & (d["_alarm_ts"] < event_ts)
        before_event |= d["_alarm_ts"].isna() & (d["_month_ord"] < event_ord)
    else:
        before_event = d["_month_ord"] < event_ord
    pre = d[before_event].copy()
    if pre.empty:
        return base

    pre["_months_before"] = event_ord - pre["_month_ord"]
    alarms = pre[pre["_alarm"]].copy()
    if not alarms.empty:
        if pd.notna(event_ts) and alarms["_alarm_ts"].notna().any():
            window_start = event_ts - pd.DateOffset(months=window_max_months)
            window_end = event_ts - pd.DateOffset(months=window_min_months)
            actionable = alarms[
                alarms["_alarm_ts"].between(window_start, window_end, inclusive="both")
            ].sort_values(["_alarm_ts", "_month_ord"])
        else:
            actionable = alarms[
                (alarms["_months_before"] >= window_min_months)
                & (alarms["_months_before"] <= window_max_months)
            ].sort_values(["_month_ord", "_alarm_ts"])
        if not actionable.empty:
            a = actionable.iloc[0]
            lead_days = _days_between(a["_date_value"], event_date, a["_month_ord"], event_ord)
            base.update({
                "first_alarm": date_label(a["_date_value"]),
                "first_alarm_date": date_label(a["_date_value"]),
                "lead_days": lead_days,
                "lead_months": lead_days / DAYS_PER_MONTH if np.isfinite(lead_days) else np.nan,
                "lead_time_days": lead_days,
                "lead_time_months": lead_days / DAYS_PER_MONTH if np.isfinite(lead_days) else np.nan,
                "actionable_alarm_found": True,
                "qualifying_alarm_found": True,
                "actionable_months_before_default": float(a["_months_before"]),
                "actionable_alarm_index": a["_source_index"],
            })

    compact = _first_alarm_per_month(pre)
    alarm_pos = np.flatnonzero(compact["_alarm"].to_numpy(dtype=bool))
    if len(alarm_pos):
        end_pos = int(alarm_pos[-1])
        start_pos = end_pos
        while start_pos > 0:
            curr = compact.iloc[start_pos]
            prev = compact.iloc[start_pos - 1]
            if not bool(prev["_alarm"]):
                break
            if int(curr["_month_ord"]) - int(prev["_month_ord"]) != 1:
                break
            start_pos -= 1
        start = compact.iloc[start_pos]
        end = compact.iloc[end_pos]
        p_days = _days_between(start["_date_value"], event_date, start["_month_ord"], event_ord)
        base.update({
            "persistent_alarm_start": date_label(start["_date_value"]),
            "persistent_alarm_end": date_label(end["_date_value"]),
            "persistent_alarm_days": p_days,
            "persistent_alarm_months": p_days / DAYS_PER_MONTH if np.isfinite(p_days) else np.nan,
            "persistent_months_before_default": float(event_ord - start["_month_ord"]),
            "persistent_alarm_start_index": start["_source_index"],
            "persistent_alarm_end_index": end["_source_index"],
        })

    return base


def status_and_kind(metrics: dict[str, Any], *, has_event: bool) -> tuple[str, str]:
    if not has_event:
        return "censored", "N/A"
    if bool(metrics.get("actionable_alarm_found")):
        return "detected", "qualifying"
    if metrics.get("persistent_alarm_start"):
        return "missed", "earlier-only"
    return "missed", "missed"


def summarize_lead_table(
    table: pd.DataFrame,
    *,
    lead_col: str = "lead_days",
    persistent_col: str = "persistent_alarm_days",
) -> dict[str, Any]:
    meta = summary_metadata()
    if table is None or table.empty:
        meta.update({
            "n_caught": 0,
            "median_lead_days": np.nan,
            "mean_lead_days": np.nan,
            "median_lead_months": np.nan,
            "mean_lead_months": np.nan,
            "median_persistent_alarm_days": np.nan,
            "median_persistent_alarm_months": np.nan,
        })
        return meta

    leads = pd.to_numeric(table.get(lead_col, pd.Series(dtype=float)), errors="coerce").dropna()
    pers = pd.to_numeric(table.get(persistent_col, pd.Series(dtype=float)), errors="coerce").dropna()
    meta.update({
        "n_caught": int(len(leads)),
        "median_lead_days": float(leads.median()) if len(leads) else np.nan,
        "mean_lead_days": float(leads.mean()) if len(leads) else np.nan,
        "median_lead_months": float(leads.median() / DAYS_PER_MONTH) if len(leads) else np.nan,
        "mean_lead_months": float(leads.mean() / DAYS_PER_MONTH) if len(leads) else np.nan,
        "median_persistent_alarm_days": float(pers.median()) if len(pers) else np.nan,
        "median_persistent_alarm_months": float(pers.median() / DAYS_PER_MONTH) if len(pers) else np.nan,
    })
    return meta


def require_metric_version(
    df: pd.DataFrame,
    *,
    table_name: str,
    allow_empty: bool = False,
) -> None:
    if df is None or df.empty:
        if allow_empty:
            return
        raise ValueError(f"{table_name} is empty; regenerate it with the current lead metric code")
    if "lead_metric_version" not in df.columns:
        raise ValueError(f"{table_name} has no lead_metric_version; regenerate legacy results")
    versions = set(df["lead_metric_version"].dropna().astype(str))
    if versions != {LEAD_METRIC_VERSION}:
        raise ValueError(
            f"{table_name} lead_metric_version is {sorted(versions)}; "
            f"expected {LEAD_METRIC_VERSION}"
        )
