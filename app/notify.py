"""
notify.py - email + Telegram alerting for the CMDF Credit EWS.

A notification is sent **only when an account escalates INTO the trigger band**
(default ``HIGH RISK``): the account is new, or its previously stored band was
lower. Accounts that were already HIGH RISK stay silent, so re-running
"Compute alerts" does not re-send the same names.

State lives in the same SQLite DB (table ``alert_state``), next to the existing
``credit`` / ``alerts`` tables.

Configuration comes from environment variables, with a ``.env`` file in this
folder as a fallback (tiny parser below - no external dependency).

    # --- email (SMTP) ---
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@gmail.com
    SMTP_PASS=your-app-password
    SMTP_FROM=you@gmail.com        # optional, defaults to SMTP_USER
    SMTP_TO=risk@bank.com,me@x.com # comma separated
    SMTP_SECURITY=starttls         # starttls | ssl | none

    # --- Telegram ---
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=-1001234567890

    # --- behaviour (all optional) ---
    NOTIFY_LEVEL=HIGH RISK         # band that triggers a notification
    NOTIFY_MAX_ROWS=20             # accounts listed in the message body
    NOTIFY_ENABLED=1               # 0 = compute escalations but never send
    NOTIFY_SEED_SEND=0             # 1 = also notify on the very first run

A channel that is not configured is simply skipped (never raises), so the app
keeps working before any credentials are set.

Standalone use:
    python notify.py --check      # which channels are configured
    python notify.py --test       # send a test message to configured channels
    python notify.py --dry-run    # read the `alerts` table, print what WOULD be sent
"""

from __future__ import annotations

import datetime
import json
import os
import smtplib
import sqlite3
import ssl
import sys
from email.message import EmailMessage
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DEFAULT = os.path.join(DATA_ROOT, "cmdf_credit.db")
ENV_PATH = os.path.join(HERE, ".env")
STATE_TABLE = "alert_state"


# ------------------------------------------------------------------ config ---
def _load_env(path: str = ENV_PATH) -> None:
    """Minimal .env reader (KEY=VALUE, # comments). Real env vars always win."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except Exception:
        pass  # a broken .env must never stop the app


_load_env()


def _cfg(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_cfg(key) or default)
    except ValueError:
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    raw = _cfg(key).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


TRIGGER_LEVEL = _cfg("NOTIFY_LEVEL", "HIGH RISK")
MAX_ROWS = _cfg_int("NOTIFY_MAX_ROWS", 20)


def email_configured():
    missing = [k for k in ("SMTP_HOST", "SMTP_TO") if not _cfg(k)]
    return (not missing), missing


def telegram_configured():
    missing = [k for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not _cfg(k)]
    return (not missing), missing


def config_status() -> dict:
    ok_e, miss_e = email_configured()
    ok_t, miss_t = telegram_configured()
    return {
        "email": {"configured": ok_e, "missing": miss_e, "to": _cfg("SMTP_TO")},
        "telegram": {"configured": ok_t, "missing": miss_t, "chat_id": _cfg("TELEGRAM_CHAT_ID")},
        "trigger_level": TRIGGER_LEVEL,
        "enabled": _cfg_bool("NOTIFY_ENABLED", True),
        "seed_send": _cfg_bool("NOTIFY_SEED_SEND", False),
        "env_file": ENV_PATH if os.path.exists(ENV_PATH) else "(none)",
    }


# ---------------------------------------------------------------- channels ---
def send_email(subject: str, body: str):
    ok, missing = email_configured()
    if not ok:
        return False, "email skipped (missing: %s)" % ", ".join(missing)

    host = _cfg("SMTP_HOST")
    user, password = _cfg("SMTP_USER"), _cfg("SMTP_PASS")
    sender = _cfg("SMTP_FROM") or user or "cmdf-ews@localhost"
    recipients = [a.strip() for a in _cfg("SMTP_TO").split(",") if a.strip()]
    security = _cfg("SMTP_SECURITY", "starttls").lower()
    port = _cfg_int("SMTP_PORT", 465 if security == "ssl" else 587)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    try:
        if security == "ssl":
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as srv:
                if user:
                    srv.login(user, password)
                srv.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as srv:
                if security == "starttls":
                    srv.starttls(context=ssl.create_default_context())
                if user:
                    srv.login(user, password)
                srv.send_message(msg)
        return True, "email sent to %d recipient(s)" % len(recipients)
    except Exception as exc:
        return False, "email FAILED: %s" % exc


def _post_json(url: str, payload: dict):
    """POST JSON using requests when available, else stdlib urllib."""
    try:
        import requests  # bundled with yfinance, so normally present
        resp = requests.post(url, json=payload, timeout=30)
        return resp.status_code, resp.text
    except ImportError:
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")


def send_telegram(text: str):
    ok, missing = telegram_configured()
    if not ok:
        return False, "telegram skipped (missing: %s)" % ", ".join(missing)

    url = "https://api.telegram.org/bot%s/sendMessage" % _cfg("TELEGRAM_BOT_TOKEN")
    payload = {
        "chat_id": _cfg("TELEGRAM_CHAT_ID"),
        "text": text[:4000],           # Telegram caps a message at 4096 characters
        "disable_web_page_preview": True,
    }
    try:
        code, body = _post_json(url, payload)
        if code == 200:
            return True, "telegram sent"
        return False, "telegram FAILED: HTTP %s %s" % (code, body[:200])
    except Exception as exc:
        return False, "telegram FAILED: %s" % exc


# ------------------------------------------------------------------- state ---
def _ensure_state(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS %s ("
        "account_id TEXT PRIMARY KEY, alert TEXT, pd REAL, updated_at TEXT)" % STATE_TABLE)


def _load_state(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        _ensure_state(con)
        rows = con.execute("SELECT account_id, alert FROM %s" % STATE_TABLE).fetchall()
    finally:
        con.close()
    return {str(a): b for a, b in rows}


def _save_state(db_path: str, alerts) -> None:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    payload = [(str(r["account_id"]), str(r["alert"]), float(r["PD"]), stamp)
               for _, r in alerts.iterrows()]
    con = sqlite3.connect(db_path)
    try:
        _ensure_state(con)
        con.executemany(
            "INSERT INTO %s(account_id, alert, pd, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(account_id) DO UPDATE SET "
            "alert=excluded.alert, pd=excluded.pd, updated_at=excluded.updated_at" % STATE_TABLE,
            payload)
        con.commit()
    finally:
        con.close()


def escalations(alerts, db_path: str | None = None):
    """Rows that just moved INTO ``TRIGGER_LEVEL``; also returns how many
    accounts were already known (0 => this is the very first run)."""
    db_path = db_path or DB_DEFAULT
    previous = _load_state(db_path)
    current = alerts[alerts["alert"].astype(str) == TRIGGER_LEVEL]
    if current.empty:
        return current, len(previous)
    mask = current["account_id"].astype(str).map(
        lambda acc: previous.get(acc) != TRIGGER_LEVEL)
    return current[mask].sort_values("PD", ascending=False), len(previous)


# ----------------------------------------------------------------- message ---
def _format_message(escalated, total_in_band: int):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    count = len(escalated)
    subject = "[CMDF Credit EWS] %d account(s) escalated to %s" % (count, TRIGGER_LEVEL)

    lines = [
        subject,
        "time              : %s" % stamp,
        "newly escalated   : %d" % count,
        "total in %-9s: %d" % (TRIGGER_LEVEL, total_in_band),
        "",
        "%-20s %8s" % ("ACCOUNT", "PD"),
        "-" * 29,
    ]
    for _, row in escalated.head(MAX_ROWS).iterrows():
        lines.append("%-20s %7.1f%%" % (str(row["account_id"])[:20], float(row["PD"]) * 100))
    if count > MAX_ROWS:
        lines.append("... and %d more" % (count - MAX_ROWS))
    return subject, "\n".join(lines)


# -------------------------------------------------------------- public API ---
def notify_alerts(alerts, db_path: str | None = None, dry_run: bool = False) -> dict:
    """Compare ``alerts`` against the stored state and notify on escalations.

    ``alerts`` is the DataFrame returned by ``app.compute_alerts`` - it needs the
    columns ``account_id``, ``PD`` and ``alert``. Returns a summary dict and
    never raises, so a notification problem cannot break the app.
    """
    db_path = db_path or DB_DEFAULT
    try:
        if alerts is None or len(alerts) == 0:
            return {"sent": False, "escalated": 0, "detail": ["no alerts to check"]}

        escalated, known = escalations(alerts, db_path)
        total_in_band = int((alerts["alert"].astype(str) == TRIGGER_LEVEL).sum())

        # First ever run: store a baseline instead of blasting every existing name.
        if known == 0 and not _cfg_bool("NOTIFY_SEED_SEND", False):
            if not dry_run:
                _save_state(db_path, alerts)
            return {"sent": False, "escalated": 0, "seeded": True,
                    "detail": ["first run: baseline of %d accounts stored, nothing sent "
                               "(set NOTIFY_SEED_SEND=1 to notify on the first run)" % len(alerts)]}

        if escalated.empty:
            if not dry_run:
                _save_state(db_path, alerts)
            return {"sent": False, "escalated": 0,
                    "detail": ["no new escalation to %s (%d already in band)"
                               % (TRIGGER_LEVEL, total_in_band)]}

        subject, body = _format_message(escalated, total_in_band)

        if dry_run:
            return {"sent": False, "escalated": len(escalated), "dry_run": True,
                    "detail": ["dry-run: nothing sent"], "body": body}

        if not _cfg_bool("NOTIFY_ENABLED", True):
            _save_state(db_path, alerts)
            return {"sent": False, "escalated": len(escalated),
                    "detail": ["NOTIFY_ENABLED=0 - sending disabled"], "body": body}

        detail = []
        ok_email, msg_email = send_email(subject, body)
        detail.append(msg_email)
        ok_tg, msg_tg = send_telegram(body)
        detail.append(msg_tg)

        _save_state(db_path, alerts)
        return {"sent": bool(ok_email or ok_tg), "escalated": len(escalated),
                "detail": detail, "body": body}

    except Exception as exc:                                   # never break the caller
        return {"sent": False, "escalated": 0, "detail": ["notify error: %s" % exc]}


def summary_line(result: dict) -> str:
    """One-line status suitable for the GUI status bar."""
    if result.get("seeded"):
        return "Alerts baseline stored (no notification on first run)."
    if result.get("escalated", 0) == 0:
        return "No new %s escalation - nothing sent." % TRIGGER_LEVEL
    what = "sent" if result.get("sent") else "NOT sent"
    return "%d escalated to %s - notification %s (%s)" % (
        result["escalated"], TRIGGER_LEVEL, what, "; ".join(result.get("detail", [])))


# ----------------------------------------------------------------- CLI ------
def _cli_check() -> None:
    status = config_status()
    print("notify.py configuration")
    print("  env file      :", status["env_file"])
    print("  trigger level :", status["trigger_level"])
    print("  sending       :", "enabled" if status["enabled"] else "DISABLED (NOTIFY_ENABLED=0)")
    print("  first-run send:", "yes" if status["seed_send"] else "no (baseline only)")
    e, t = status["email"], status["telegram"]
    print("  email         :", "configured -> %s" % e["to"] if e["configured"]
          else "not configured (missing: %s)" % ", ".join(e["missing"]))
    print("  telegram      :", "configured -> chat %s" % t["chat_id"] if t["configured"]
          else "not configured (missing: %s)" % ", ".join(t["missing"]))


def _cli_test() -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = "[CMDF Credit EWS] test message"
    body = "CMDF Credit EWS test message\ntime: %s\n\nIf you can read this, the channel works." % stamp
    ok_e, msg_e = send_email(subject, body)
    ok_t, msg_t = send_telegram(body)
    print(" email    :", msg_e)
    print(" telegram :", msg_t)
    if not (ok_e or ok_t):
        print("\nNothing was sent. Fill in .env (see the header of this file) and retry.")


def _cli_dry_run() -> None:
    import pandas as pd
    if not os.path.exists(DB_DEFAULT):
        print("no database at %s - run the app and press 'Compute alerts' first." % DB_DEFAULT)
        return
    con = sqlite3.connect(DB_DEFAULT)
    try:
        alerts = pd.read_sql_query("SELECT * FROM alerts", con)
    except Exception as exc:
        print("cannot read the `alerts` table (%s) - press 'Compute alerts' in the app first." % exc)
        return
    finally:
        con.close()
    result = notify_alerts(alerts, dry_run=True)
    print(summary_line(result))
    if result.get("body"):
        print("\n--- message that would be sent ---")
        print(result["body"])


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--check" in args:
        _cli_check()
    elif "--test" in args:
        _cli_test()
    elif "--dry-run" in args:
        _cli_dry_run()
    else:
        print(__doc__.strip().splitlines()[0])
        print("\nusage: python notify.py [--check | --test | --dry-run]")
        _cli_check()
