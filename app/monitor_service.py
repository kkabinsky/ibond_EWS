# -*- coding: utf-8 -*-
"""
monitor_service.py -- scheduled iBond monitoring with email alerts.

WHAT ONE CYCLE DOES
    1. download   pull the corporate-bond universe and the payment-default register
                  from iBond (download_bond.py)
    2. score      re-run the Approach-1 hazard engine (bond_ews.py)
    3. diff       compare the new alert set against the previous cycle and list
                  issuers that CHANGED band -- that is what a monitor is for
    4. notify     email the summary, but only when something changed (or when the
                  schedule says "always")
    5. log        write the cycle to `monitor_runs` so the GUI can show history

SCHEDULING
    Two independent mechanisms, both optional:
      * in-app thread  -- runs while app.py is open (start_background / stop_background)
      * Windows Task Scheduler -- keeps running when the app is closed
        (install_scheduled_task / remove_scheduled_task, via schtasks)
    Intervals offered: hourly, every 4 hours, daily.

CREDENTIALS
    iBond : THAIBMA_USER / THAIBMA_PASS from the environment (setup_credentials.py).
    SMTP  : SMTP_USER / SMTP_PASS from the environment by preference. A password
            stored in the database is used only as a fallback, and `config_status`
            reports when that is the case so it can be moved out.

RUN
    python monitor_service.py --once            # one cycle now
    python monitor_service.py --once --no-download
    python monitor_service.py --install hourly  # register the Windows task
    python monitor_service.py --install 4h
    python monitor_service.py --install daily
    python monitor_service.py --remove
    python monitor_service.py --status
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")

T_CONFIG = "monitor_config"
T_RUNS = "monitor_runs"
T_STATE = "monitor_alert_state"

TASK_NAME = "ThaiBMA_iBond_EWS_Monitor"

INTERVALS = {
    "hourly": {"label": "Every hour", "minutes": 60,
               "schtasks": ["/SC", "HOURLY", "/MO", "1"]},
    "4h":     {"label": "Every 4 hours", "minutes": 240,
               "schtasks": ["/SC", "HOURLY", "/MO", "4"]},
    "daily":  {"label": "Once a day", "minutes": 1440,
               "schtasks": ["/SC", "DAILY"]},
}

ALERT_ORDER = {"OK": 0, "WATCH": 1, "ELEVATED": 2, "HIGH RISK": 3}


# =============================================================== config =======
def init_db(db_path=DB):
    con = sqlite3.connect(db_path)
    con.execute(f"""CREATE TABLE IF NOT EXISTS {T_CONFIG} (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        interval_key TEXT DEFAULT 'daily',
        enabled INTEGER DEFAULT 0,
        do_download INTEGER DEFAULT 1,
        notify_mode TEXT DEFAULT 'on_change',
        recipient TEXT DEFAULT '',
        min_band TEXT DEFAULT 'HIGH RISK',
        updated_at TEXT)""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {T_RUNS} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT, finished_at TEXT, seconds REAL,
        trigger TEXT, downloaded INTEGER, scored INTEGER,
        n_issuers INTEGER, n_high INTEGER, n_elevated INTEGER,
        n_changed INTEGER, n_new_high INTEGER,
        emailed INTEGER, status TEXT, detail TEXT)""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {T_STATE} (
        issuer_code TEXT PRIMARY KEY, alert TEXT, pd_3m REAL, seen_at TEXT)""")
    con.commit()
    con.close()


def load_config(db_path=DB) -> dict:
    init_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(f"SELECT interval_key, enabled, do_download, notify_mode, "
                          f"recipient, min_band, updated_at FROM {T_CONFIG} "
                          f"WHERE id = 1").fetchone()
    finally:
        con.close()
    if not row:
        return {"interval_key": "daily", "enabled": 0, "do_download": 1,
                "notify_mode": "on_change", "recipient": "", "min_band": "HIGH RISK",
                "updated_at": ""}
    keys = ["interval_key", "enabled", "do_download", "notify_mode",
            "recipient", "min_band", "updated_at"]
    return dict(zip(keys, row))


def save_config(interval_key="daily", enabled=0, do_download=1,
                notify_mode="on_change", recipient="", min_band="HIGH RISK",
                db_path=DB):
    init_db(db_path)
    con = sqlite3.connect(db_path)
    con.execute(f"""INSERT INTO {T_CONFIG}
        (id, interval_key, enabled, do_download, notify_mode, recipient, min_band, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            interval_key=excluded.interval_key, enabled=excluded.enabled,
            do_download=excluded.do_download, notify_mode=excluded.notify_mode,
            recipient=excluded.recipient, min_band=excluded.min_band,
            updated_at=excluded.updated_at""",
                (interval_key, int(enabled), int(do_download), notify_mode,
                 recipient, min_band, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    con.close()


def config_status(db_path=DB) -> dict:
    """Report readiness WITHOUT revealing any secret value."""
    cfg = load_config(db_path)
    try:
        import ibond_client as ic
        ib_ready = bool(ic.credentials_status().get("ready"))
    except Exception:
        ib_ready = False
    env_user, env_pass = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    db_user = db_pass = None
    try:
        con = sqlite3.connect(db_path)
        r = con.execute("SELECT smtp_user, smtp_pass FROM email_alert_config "
                        "WHERE is_enabled = 1 LIMIT 1").fetchone()
        con.close()
        if r:
            db_user, db_pass = r
    except Exception:
        pass
    return {
        "interval": cfg["interval_key"],
        "interval_label": INTERVALS.get(cfg["interval_key"], {}).get("label", "?"),
        "enabled": bool(cfg["enabled"]),
        "ibond_ready": ib_ready,
        "smtp_from_env": bool(env_user and env_pass),
        "smtp_from_db": bool(db_pass) and not (env_user and env_pass),
        "smtp_ready": bool((env_user and env_pass) or db_pass),
        "recipient": cfg["recipient"],
        "notify_mode": cfg["notify_mode"],
        "min_band": cfg["min_band"],
        "task_installed": task_installed(),
    }


def _smtp_creds(db_path=DB):
    """Environment first; the database only as a fallback."""
    u, p = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if u and p:
        return u, p, "env"
    try:
        con = sqlite3.connect(db_path)
        r = con.execute("SELECT smtp_user, smtp_pass FROM email_alert_config "
                        "WHERE is_enabled = 1 LIMIT 1").fetchone()
        con.close()
        if r and r[0] and r[1]:
            return r[0], r[1], "db"
    except Exception:
        pass
    return None, None, "none"


# ============================================================== one cycle ====
def _current_alerts(db_path=DB) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        d = pd.read_sql("SELECT issuer_code, PD_3M, alert FROM bond_ews_alert", con)
    except Exception:
        d = pd.DataFrame(columns=["issuer_code", "PD_3M", "alert"])
    finally:
        con.close()
    return d


def _previous_state(db_path=DB) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        d = pd.read_sql(f"SELECT issuer_code, alert, pd_3m FROM {T_STATE}", con)
    except Exception:
        d = pd.DataFrame(columns=["issuer_code", "alert", "pd_3m"])
    finally:
        con.close()
    return d


def _store_state(alerts: pd.DataFrame, db_path=DB):
    if alerts.empty:
        return
    d = alerts[["issuer_code", "alert", "PD_3M"]].copy()
    d.columns = ["issuer_code", "alert", "pd_3m"]
    d["seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(db_path)
    d.to_sql(T_STATE, con, if_exists="replace", index=False)
    con.commit()
    con.close()


def diff_alerts(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Issuers whose band changed since the previous cycle."""
    if new.empty:
        return pd.DataFrame()
    m = new.merge(old.rename(columns={"alert": "prev_alert", "pd_3m": "prev_pd"}),
                  on="issuer_code", how="left")
    m["prev_alert"] = m["prev_alert"].fillna("")
    ch = m[m["alert"] != m["prev_alert"]].copy()
    if ch.empty:
        return ch
    ch["from_rank"] = ch["prev_alert"].map(ALERT_ORDER).fillna(-1)
    ch["to_rank"] = ch["alert"].map(ALERT_ORDER).fillna(-1)
    ch["direction"] = ch.apply(
        lambda r: "WORSE" if r["to_rank"] > r["from_rank"] else "BETTER", axis=1)
    return ch.sort_values(["to_rank", "PD_3M"], ascending=[False, False])


def run_cycle(do_download=True, send_email=True, trigger="manual", db_path=DB,
              log=None, verbose=True):
    """One monitoring pass. Never raises: failures are recorded and returned."""
    def emit(m):
        # The Windows console here is cp874; a stray non-ASCII character in a print
        # raised UnicodeEncodeError mid-cycle and aborted the run before the alert
        # state was stored, so every cycle reported all 289 issuers as "changed".
        if verbose:
            try:
                print(m)
            except Exception:
                print(str(m).encode("ascii", "replace").decode("ascii"))
        if log:
            log(m)

    init_db(db_path)
    t0 = datetime.now()
    rec = {"started_at": t0.strftime("%Y-%m-%d %H:%M:%S"), "trigger": trigger,
           "downloaded": 0, "scored": 0, "n_issuers": 0, "n_high": 0,
           "n_elevated": 0, "n_changed": 0, "n_new_high": 0, "emailed": 0,
           "status": "OK", "detail": ""}
    changed = pd.DataFrame()
    try:
        prev = _previous_state(db_path)

        if do_download:
            emit("[1/4] downloading from iBond ...")
            try:
                import download_bond as dbnd
                dbnd.run(with_defaults=True, save=True, verbose=False)
                rec["downloaded"] = 1
                emit("      download OK")
            except Exception as ex:
                rec["status"] = "PARTIAL"
                rec["detail"] += f"download failed: {ex}; "
                emit(f"      download FAILED: {ex}")
        else:
            emit("[1/4] download skipped")

        emit("[2/4] scoring with Approach 1 ...")
        try:
            import bond_ews as bews
            bews.run(refresh=False, save=True, verbose=False)
            rec["scored"] = 1
            emit("      scoring OK")
        except Exception as ex:
            rec["status"] = "PARTIAL" if rec["status"] == "OK" else rec["status"]
            rec["detail"] += f"scoring failed: {ex}; "
            emit(f"      scoring FAILED: {ex}")

        emit("[3/4] comparing against the previous cycle ...")
        cur = _current_alerts(db_path)
        rec["n_issuers"] = int(len(cur))
        rec["n_high"] = int((cur["alert"] == "HIGH RISK").sum()) if not cur.empty else 0
        rec["n_elevated"] = int((cur["alert"] == "ELEVATED").sum()) if not cur.empty else 0
        changed = diff_alerts(cur, prev)
        rec["n_changed"] = int(len(changed))
        rec["n_new_high"] = int((changed["alert"] == "HIGH RISK").sum()) \
            if not changed.empty else 0
        emit(f"      {rec['n_issuers']} issuers | {rec['n_high']} HIGH RISK | "
             f"{rec['n_changed']} changed | {rec['n_new_high']} newly HIGH RISK")
        _store_state(cur, db_path)

        cfg = load_config(db_path)
        should = (send_email and cfg.get("recipient")
                  and (cfg.get("notify_mode") == "always" or rec["n_changed"] > 0))
        if should:
            emit("[4/4] sending the email alert ...")
            try:
                import email_alert_engine as eae
                u, p, src = _smtp_creds(db_path)
                eae.send_daily_email_alert(cfg["recipient"], smtp_user=u, smtp_pass=p,
                                           db_path=db_path)
                rec["emailed"] = 1
                emit(f"      email sent to {cfg['recipient']} (credentials from {src})"
                     if u else "      email simulated (no SMTP credentials set)")
            except Exception as ex:
                rec["status"] = "PARTIAL" if rec["status"] == "OK" else rec["status"]
                rec["detail"] += f"email failed: {ex}; "
                emit(f"      email FAILED: {ex}")
        else:
            emit("[4/4] no email (nothing changed, or no recipient configured)")
    except Exception as ex:
        rec["status"] = "ERROR"
        rec["detail"] += str(ex)
        emit(f"CYCLE ERROR: {ex}")
        if verbose:
            traceback.print_exc()

    t1 = datetime.now()
    rec["finished_at"] = t1.strftime("%Y-%m-%d %H:%M:%S")
    rec["seconds"] = round((t1 - t0).total_seconds(), 1)
    con = sqlite3.connect(db_path)
    pd.DataFrame([rec]).to_sql(T_RUNS, con, if_exists="append", index=False)
    con.commit()
    con.close()
    emit(f"DONE in {rec['seconds']}s - {rec['status']}")
    return rec, changed


def get_runs(limit=40, db_path=DB) -> pd.DataFrame:
    init_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        d = pd.read_sql(f"SELECT * FROM {T_RUNS} ORDER BY id DESC LIMIT {int(limit)}", con)
    except Exception:
        d = pd.DataFrame()
    finally:
        con.close()
    return d


# ====================================================== in-app scheduler =====
_thread = None
_stop = threading.Event()


def start_background(db_path=DB, log=None):
    """Run cycles on the configured interval while the app is open."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return False
    _stop.clear()

    def loop():
        while not _stop.is_set():
            cfg = load_config(db_path)
            if not cfg.get("enabled"):
                break
            minutes = INTERVALS.get(cfg["interval_key"], INTERVALS["daily"])["minutes"]
            try:
                run_cycle(do_download=bool(cfg.get("do_download", 1)),
                          send_email=True, trigger="in-app schedule",
                          db_path=db_path, log=log, verbose=False)
            except Exception:
                pass
            # wake up every 15s so a stop request is honoured promptly
            for _ in range(int(minutes * 60 / 15)):
                if _stop.is_set():
                    return
                time.sleep(15)

    _thread = threading.Thread(target=loop, daemon=True)
    _thread.start()
    return True


def stop_background():
    _stop.set()
    return True


def background_running() -> bool:
    return _thread is not None and _thread.is_alive()


# ================================================ Windows Task Scheduler =====
def task_installed() -> bool:
    if os.name != "nt":
        return False
    try:
        r = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME],
                           capture_output=True, text=True, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def install_scheduled_task(interval_key="daily", start_time="08:00"):
    """Register a Windows scheduled task so monitoring continues with the app closed.

    Uses the current interpreter and this file, so no wrapper script is needed.
    """
    if os.name != "nt":
        return False, "Windows Task Scheduler is only available on Windows."
    if interval_key not in INTERVALS:
        return False, f"unknown interval '{interval_key}'"
    cmd = ["schtasks", "/Create", "/TN", TASK_NAME, "/F",
           "/TR", f'"{sys.executable}" "{os.path.join(HERE, "monitor_service.py")}" --once',
           *INTERVALS[interval_key]["schtasks"], "/ST", start_time]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        if r.returncode == 0:
            return True, (f"Scheduled task '{TASK_NAME}' installed "
                          f"({INTERVALS[interval_key]['label']}, from {start_time}).")
        return False, (r.stderr or r.stdout or "schtasks failed").strip()
    except Exception as ex:
        return False, str(ex)


def remove_scheduled_task():
    if os.name != "nt":
        return False, "Windows only."
    try:
        r = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                           capture_output=True, text=True, shell=False)
        if r.returncode == 0:
            return True, f"Scheduled task '{TASK_NAME}' removed."
        return False, (r.stderr or r.stdout or "schtasks failed").strip()
    except Exception as ex:
        return False, str(ex)


# ==================================================================== cli ====
def main():
    a = sys.argv
    if "--status" in a:
        st = config_status()
        print("iBond EWS monitor status")
        for k, v in st.items():
            print(f"  {k:16} {v}")
        runs = get_runs(5)
        if not runs.empty:
            print("\nlast runs:")
            print(runs[["started_at", "trigger", "n_high", "n_changed",
                        "emailed", "status"]].to_string(index=False))
        return
    if "--install" in a:
        i = a.index("--install")
        key = a[i + 1] if len(a) > i + 1 else "daily"
        ok, msg = install_scheduled_task(key)
        print(("OK: " if ok else "FAILED: ") + msg)
        return
    if "--remove" in a:
        ok, msg = remove_scheduled_task()
        print(("OK: " if ok else "FAILED: ") + msg)
        return
    rec, changed = run_cycle(do_download="--no-download" not in a,
                             send_email="--no-email" not in a,
                             trigger="cli")
    print("\nCYCLE RESULT")
    for k in ("started_at", "seconds", "downloaded", "scored", "n_issuers",
              "n_high", "n_elevated", "n_changed", "n_new_high", "emailed", "status"):
        print(f"  {k:12} {rec[k]}")
    if not changed.empty:
        print("\nBAND CHANGES")
        print(changed[["issuer_code", "prev_alert", "alert", "direction", "PD_3M"]]
              .head(20).to_string(index=False))


if __name__ == "__main__":
    main()
