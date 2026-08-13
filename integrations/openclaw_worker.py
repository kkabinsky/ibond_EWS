"""Deterministic jobs run by the local OpenClaw cron scheduler."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd

import app as credit_app
import openclaw_connector as oc
import scan
import survival


def _stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _safe_float(value):
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _safe_int(value):
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else int(round(float(parsed)))


def _load_panel() -> pd.DataFrame:
    con = sqlite3.connect(credit_app.DB)
    try:
        return pd.read_sql_query("SELECT * FROM panel", con)
    finally:
        con.close()


def _queue_row(con: sqlite3.Connection, row: pd.Series, mode: str) -> int:
    default_observed = bool(row.get("default_observed", False))
    signal_date = (_safe_text(row.get("first_alarm_date"))
                   or _safe_text(row.get("latest_signal_date")))
    lead_days = _safe_int(row.get("lead_time_days")) if default_observed else None
    lead_window = None if default_observed else 90
    payload = {
        "source": "ThaiBMA Credit EWS",
        "firm_id": _safe_text(row.get("firm_id")),
        "account_id": _safe_text(row.get("account_id")),
        "alert_mode": mode,
        "signal_date": signal_date,
        "first_alarm_date": _safe_text(row.get("first_alarm_date")) or None,
        "default_date": _safe_text(row.get("default_date")) or None,
        "lead_time_days": lead_days,
        "lead_window_days": lead_window,
        "alarm_source": _safe_text(row.get("alarm_source")),
        "latest_pd_3m": _safe_float(row.get("latest_PD_3M")),
        "latest_momentum": _safe_float(row.get("latest_Momentum")),
        "alert_level": _safe_text(row.get("alert_level")),
        "interpretation": (
            "Observed historical lead time" if default_observed else
            "Prospective EWS warning window; this is not an observed default date"
        ),
    }
    raw_key = "|".join([
        payload["firm_id"], payload["account_id"], mode, signal_date,
        payload["alert_level"], str(payload["lead_time_days"]),
    ])
    dedupe_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    now = _stamp()
    cur = con.execute(
        """
        INSERT OR IGNORE INTO lead_alert_queue
        (firm_id, account_id, alert_mode, signal_date, first_alarm_date,
         default_date, lead_time_days, lead_window_days, alarm_source,
         latest_pd_3m, latest_momentum, alert_level, payload_json,
         dedupe_key, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        """,
        (
            payload["firm_id"], payload["account_id"], mode, signal_date,
            payload["first_alarm_date"], payload["default_date"],
            payload["lead_time_days"], payload["lead_window_days"],
            payload["alarm_source"], payload["latest_pd_3m"],
            payload["latest_momentum"], payload["alert_level"],
            json.dumps(payload, ensure_ascii=False), dedupe_key, now, now,
        ),
    )
    return int(cur.lastrowid) if cur.rowcount > 0 else 0


def run_lead_time_alerts(dry_run: bool = False) -> dict:
    oc.ensure_schema(credit_app.DB)
    cfg = oc.get_connection(credit_app.DB)
    if not bool(cfg.get("sharing_enabled")) and not dry_run:
        return {"ok": False, "queued": 0,
                "detail": "OpenClaw sharing is disabled."}

    panel = _load_panel()
    result_df, meta = survival.run(panel)
    lead = meta.get("lead_time")
    if not isinstance(lead, pd.DataFrame):
        lead = survival.compute_lead_time(result_df)
    credit_app.save_lead_time(lead)

    entity_col = (
        "firm_id" if "firm_id" in result_df.columns
        else "account_id" if "account_id" in result_df.columns
        else "ticker"
    )
    latest = (
        result_df.sort_values([entity_col, "month_index"])
        .groupby(entity_col, as_index=False)
        .tail(1)
        .copy()
    )
    if "firm_id" not in latest.columns:
        latest["firm_id"] = latest[entity_col]
    if "account_id" not in latest.columns:
        latest["account_id"] = latest[entity_col]
    latest["alert_level"] = latest.apply(survival.alert_level, axis=1)
    latest["alarm_source"] = latest.apply(survival.alarm_source, axis=1)
    date_col = "month_year" if "month_year" in latest.columns else None
    latest["latest_signal_date"] = (
        latest[date_col].astype(str) if date_col else
        latest["month_index"].astype(str)
    )
    prospective = latest[
        latest["alert_level"].isin(["ELEVATED", "HIGH RISK"])
    ].copy()
    prospective["latest_PD_3M"] = prospective.get("PD_3M")
    prospective["latest_Momentum"] = prospective.get("Momentum")
    prospective["default_observed"] = False

    observed = lead[
        lead.get("qualifying_alarm_found", pd.Series(False, index=lead.index))
        .fillna(False).astype(bool)
    ].copy()

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "prospective": len(prospective),
            "observed": len(observed),
            "queued": 0,
        }

    inserted = 0
    shared = []
    shared_ids = []
    con = sqlite3.connect(credit_app.DB, timeout=30)
    try:
        for _, row in prospective.iterrows():
            added = _queue_row(con, row, "prospective")
            inserted += int(bool(added))
            if added:
                shared_ids.append(added)
            if added and len(shared) < 20:
                shared.append({
                    "firm_id": _safe_text(row.get("firm_id")),
                    "alert": _safe_text(row.get("alert_level")),
                    "pd_3m": _safe_float(row.get("latest_PD_3M")),
                    "momentum": _safe_float(row.get("latest_Momentum")),
                    "lead_window_days": 90,
                })
        for _, row in observed.iterrows():
            added = _queue_row(con, row, "retrospective")
            inserted += int(bool(added))
            if added:
                shared_ids.append(added)
            if added and len(shared) < 20:
                shared.append({
                    "firm_id": _safe_text(row.get("firm_id")),
                    "alert": _safe_text(row.get("alert_level")),
                    "lead_time_days": _safe_int(row.get("lead_time_days")),
                    "default_date": _safe_text(row.get("default_date")),
                })
        if inserted:
            con.execute(
                """
                UPDATE lead_alert_queue
                SET status='shared_to_openclaw', last_attempt_at=?, updated_at=?
                WHERE status='pending'
                """,
                (_stamp(), _stamp()),
            )
            con.executemany(
                """
                INSERT INTO openclaw_delivery_log
                (queue_id, event_type, status, detail, created_at)
                VALUES (?, 'alert_share', 'shared_to_openclaw',
                        'risk_summary_only', ?)
                """,
                [(queue_id, _stamp()) for queue_id in shared_ids],
            )
        con.commit()
    finally:
        con.close()
    return {
        "ok": True,
        "queued": inserted,
        "prospective": len(prospective),
        "observed": len(observed),
        "alerts": shared,
        "note": (
            "Prospective lead_window_days=90 is an EWS horizon, not an observed "
            "default date. Exact lead_time_days is retrospective only."
        ),
    }


def run_task(task: str, dry_run: bool = False) -> dict:
    if task == "lead_time_alerts":
        return run_lead_time_alerts(dry_run=dry_run)
    if task == "full_credit_scan":
        return scan.run_scan(dry_run=dry_run, quiet=True)
    if task == "refresh_and_alert":
        from load_bond import load_bond
        if dry_run:
            return {"ok": True, "dry_run": True,
                    "detail": "Would refresh bond data and run lead-time alerts."}
        df = load_bond()
        con = sqlite3.connect(credit_app.DB, timeout=30)
        try:
            df.to_sql(credit_app.TABLE, con, if_exists="replace", index_label="id")
            df.to_sql("panel", con, if_exists="replace", index_label="id")
            con.commit()
        finally:
            con.close()
        return run_lead_time_alerts(dry_run=False)
    raise ValueError(f"Unsupported task: {task}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ThaiBMA EWS worker for OpenClaw cron")
    parser.add_argument("--task", choices=sorted(oc.TASKS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_task(args.task, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 0 if result.get("ok") else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
