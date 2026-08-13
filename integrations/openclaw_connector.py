"""Local OpenClaw integration and persistence for the ThaiBMA EWS.

The application never stores an OpenClaw token. Authentication remains owned by
the local OpenClaw CLI/config. Sharing is disabled by default and only summary
risk fields are queued.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DEFAULT = os.path.join(DATA_ROOT, "cmdf_credit.db")
GATEWAY_CMD = os.path.expanduser(r"~\.openclaw\gateway.cmd")
DEFAULT_GATEWAY_URL = "http://127.0.0.1:18789"
JOB_PREFIX = "ThaiBMA EWS - "

TASKS = {
    "lead_time_alerts": "Lead Time Alert Scan",
    "full_credit_scan": "Full Credit Model Scan",
    "refresh_and_alert": "Refresh Bond Data + Alert Scan",
}
_CLI_FEATURE_CACHE: dict[tuple[str, str], bool] = {}


def _stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _connect(db_path: str = DB_DEFAULT) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def discover_cli() -> tuple[str, str]:
    """Return Node executable and OpenClaw JS entry without reading secrets."""
    node = os.getenv("OPENCLAW_NODE", "").strip() or shutil.which("node") or ""
    entry = os.getenv("OPENCLAW_CLI_ENTRY", "").strip()

    if os.path.isfile(GATEWAY_CMD):
        try:
            text = Path(GATEWAY_CMD).read_text(encoding="utf-8", errors="ignore")
            match = re.search(
                r'"([^"]*node\.exe)"\s+("?)([^"\r\n]+?openclaw\\dist\\index\.js)\2\s+gateway',
                text,
                flags=re.IGNORECASE,
            )
            if match:
                node = node or match.group(1).strip()
                entry = entry or match.group(3).strip()
        except OSError:
            pass

    if not node:
        candidate = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                 "nodejs", "node.exe")
        if os.path.isfile(candidate):
            node = candidate
    if not entry:
        appdata = os.getenv("APPDATA", os.path.expanduser(r"~\AppData\Roaming"))
        candidate = os.path.join(
            appdata, "npm", "node_modules", "openclaw", "dist", "index.js")
        if os.path.isfile(candidate):
            entry = candidate
    return node, entry


def cli_available(node: str = "", entry: str = "") -> bool:
    found_node, found_entry = discover_cli()
    node = node or found_node
    entry = entry or found_entry
    return bool(node and entry and os.path.isfile(node) and os.path.isfile(entry))


def ensure_schema(db_path: str = DB_DEFAULT) -> None:
    con = _connect(db_path)
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS openclaw_connection (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                gateway_url TEXT NOT NULL,
                sharing_enabled INTEGER NOT NULL DEFAULT 0,
                share_scope TEXT NOT NULL DEFAULT 'risk_summary_only',
                timezone TEXT NOT NULL DEFAULT 'Asia/Bangkok',
                delivery_enabled INTEGER NOT NULL DEFAULT 0,
                delivery_channel TEXT,
                delivery_target TEXT,
                cli_node_path TEXT,
                cli_entry_path TEXT,
                last_status TEXT NOT NULL DEFAULT 'not_checked',
                last_checked_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS openclaw_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                task_type TEXT NOT NULL,
                cron_expression TEXT NOT NULL,
                timezone TEXT NOT NULL DEFAULT 'Asia/Bangkok',
                enabled INTEGER NOT NULL DEFAULT 0,
                timeout_seconds INTEGER NOT NULL DEFAULT 900,
                command_json TEXT NOT NULL DEFAULT '{}',
                openclaw_job_id TEXT,
                sync_status TEXT NOT NULL DEFAULT 'local_only',
                last_synced_at TEXT,
                last_run_at TEXT,
                last_result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lead_alert_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                firm_id TEXT,
                account_id TEXT,
                alert_mode TEXT NOT NULL,
                signal_date TEXT,
                first_alarm_date TEXT,
                default_date TEXT,
                lead_time_days INTEGER,
                lead_window_days INTEGER,
                alarm_source TEXT,
                latest_pd_3m REAL,
                latest_momentum REAL,
                alert_level TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                dedupe_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TEXT,
                delivered_at TEXT,
                openclaw_job_id TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS openclaw_delivery_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id INTEGER,
                job_id INTEGER,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(queue_id) REFERENCES lead_alert_queue(id),
                FOREIGN KEY(job_id) REFERENCES openclaw_jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_openclaw_jobs_status
                ON openclaw_jobs(enabled, sync_status);
            CREATE INDEX IF NOT EXISTS idx_lead_alert_queue_status
                ON lead_alert_queue(status, alert_level, signal_date);
            """
        )
        existing = {
            row[1] for row in con.execute(
                "PRAGMA table_info(openclaw_connection)").fetchall()
        }
        connection_migrations = {
            "delivery_enabled": "INTEGER NOT NULL DEFAULT 0",
            "delivery_channel": "TEXT",
            "delivery_target": "TEXT",
        }
        for column, declaration in connection_migrations.items():
            if column not in existing:
                con.execute(
                    f"ALTER TABLE openclaw_connection "
                    f"ADD COLUMN {column} {declaration}")
        now = _stamp()
        node, entry = discover_cli()
        con.execute(
            """
            INSERT OR IGNORE INTO openclaw_connection
            (id, gateway_url, sharing_enabled, share_scope, timezone,
             delivery_enabled, delivery_channel, delivery_target,
             cli_node_path, cli_entry_path, created_at, updated_at)
            VALUES (1, ?, 1, 'risk_summary_only', 'Asia/Bangkok',
                    0, NULL, NULL, ?, ?, ?, ?)
            """,
            (DEFAULT_GATEWAY_URL, node, entry, now, now),
        )
        seeds = [
            ("Lead Time Alert Scan", "lead_time_alerts", "0 8 * * 1-5", 900),
            ("Full Credit Model Scan", "full_credit_scan", "0 18 * * 1-5", 1800),
        ]
        for name, task, cron, timeout in seeds:
            con.execute(
                """
                INSERT OR IGNORE INTO openclaw_jobs
                (name, task_type, cron_expression, timezone, enabled,
                 timeout_seconds, command_json, sync_status, created_at, updated_at)
                VALUES (?, ?, ?, 'Asia/Bangkok', 0, ?, '{}', 'local_only', ?, ?)
                """,
                (name, task, cron, timeout, now, now),
            )
        con.commit()
    finally:
        con.close()


def get_connection(db_path: str = DB_DEFAULT) -> dict:
    ensure_schema(db_path)
    con = _connect(db_path)
    try:
        row = con.execute("SELECT * FROM openclaw_connection WHERE id=1").fetchone()
        return dict(row) if row else {}
    finally:
        con.close()


def get_connection_cached(db_path: str = DB_DEFAULT) -> dict:
    con = _connect(db_path)
    try:
        row = con.execute("SELECT * FROM openclaw_connection WHERE id=1").fetchone()
        return dict(row) if row else {}
    finally:
        con.close()


def save_connection(gateway_url: str, sharing_enabled: bool,
                    timezone: str = "Asia/Bangkok",
                    delivery_enabled: bool = False,
                    delivery_channel: str = "",
                    delivery_target: str = "",
                    db_path: str = DB_DEFAULT) -> dict:
    ensure_schema(db_path)
    gateway_url = (gateway_url or DEFAULT_GATEWAY_URL).strip().rstrip("/")
    if not gateway_url.startswith(("http://127.0.0.1", "http://localhost",
                                   "https://127.0.0.1", "https://localhost")):
        raise ValueError("Only a local OpenClaw Gateway URL is allowed.")
    delivery_channel = (delivery_channel or "").strip().lower()
    delivery_target = (delivery_target or "").strip()
    if delivery_enabled and not delivery_channel:
        raise ValueError("Select an OpenClaw delivery channel.")
    if delivery_enabled and not delivery_target:
        raise ValueError("Enter the OpenClaw channel/chat target.")
    node, entry = discover_cli()
    con = _connect(db_path)
    try:
        con.execute(
            """
            UPDATE openclaw_connection
            SET gateway_url=?, sharing_enabled=?, timezone=?,
                delivery_enabled=?, delivery_channel=?, delivery_target=?,
                cli_node_path=?, cli_entry_path=?, updated_at=?
            WHERE id=1
            """,
            (gateway_url, int(bool(sharing_enabled)), timezone or "Asia/Bangkok",
             int(bool(delivery_enabled)), delivery_channel or None,
             delivery_target or None, node, entry, _stamp()),
        )
        con.commit()
    finally:
        con.close()
    return get_connection(db_path)


def enable_default_sharing(db_path: str = DB_DEFAULT) -> dict:
    ensure_schema(db_path)
    con = _connect(db_path)
    try:
        con.execute(
            """
            UPDATE openclaw_connection
            SET sharing_enabled=1, updated_at=?
            WHERE id=1 AND (sharing_enabled IS NULL OR sharing_enabled=0)
            """,
            (_stamp(),),
        )
        con.commit()
    finally:
        con.close()
    return get_connection(db_path)


def _extract_json(text: str):
    text = (text or "").strip()
    for start in [text.find("{"), text.find("[")]:
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                pass
    return None


def _safe_excerpt(text: str, limit: int = 600) -> str:
    value = (text or "").strip()
    value = re.sub(
        r'(?i)(token|password|secret|api[_-]?key)(["\s:=]+)([^,\s"}]+)',
        r"\1\2[REDACTED]",
        value,
    )
    return value[:limit]


def run_cli(args: list[str], timeout: int = 30,
            db_path: str = DB_DEFAULT) -> dict:
    cfg = get_connection_cached(db_path)
    node = cfg.get("cli_node_path") or ""
    entry = cfg.get("cli_entry_path") or ""
    if not cli_available(node, entry):
        node, entry = discover_cli()
    if not cli_available(node, entry):
        return {
            "ok": False,
            "returncode": 127,
            "error": "OpenClaw CLI was not found. Reinstall/repair OpenClaw or set "
                     "OPENCLAW_NODE and OPENCLAW_CLI_ENTRY.",
        }
    try:
        proc = subprocess.run(
            [node, entry, *args],
            cwd=HERE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": -1, "error": str(exc)}
    output = proc.stdout or ""
    error = proc.stderr or ""
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "data": _extract_json(output),
        "output": _safe_excerpt(output),
        "error": _safe_excerpt(error),
    }


def cli_supports(subcommand: str, flag: str,
                 db_path: str = DB_DEFAULT) -> bool:
    """Check an installed CLI flag without depending on a specific version."""
    cache_key = (subcommand, flag)
    if cache_key in _CLI_FEATURE_CACHE:
        return _CLI_FEATURE_CACHE[cache_key]
    cfg = get_connection(db_path)
    node = cfg.get("cli_node_path") or ""
    entry = cfg.get("cli_entry_path") or ""
    if not cli_available(node, entry):
        node, entry = discover_cli()
    if not cli_available(node, entry):
        _CLI_FEATURE_CACHE[cache_key] = False
        return False
    try:
        proc = subprocess.run(
            [node, entry, "cron", subcommand, "--help"],
            cwd=HERE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        supported = flag in ((proc.stdout or "") + (proc.stderr or ""))
    except (OSError, subprocess.TimeoutExpired):
        supported = False
    _CLI_FEATURE_CACHE[cache_key] = supported
    return supported


def test_connection(db_path: str = DB_DEFAULT) -> dict:
    """Read-only connection test. CLI status is preferred; HTTP is fallback."""
    ensure_schema(db_path)
    cfg = get_connection(db_path)
    result = run_cli(["gateway", "status", "--json"], timeout=20, db_path=db_path)
    detail = "Gateway status returned by local OpenClaw CLI."
    ok = bool(result.get("ok"))

    if not ok:
        base = str(cfg.get("gateway_url") or DEFAULT_GATEWAY_URL).rstrip("/")
        for endpoint in ("/healthz", "/health"):
            try:
                with urllib.request.urlopen(base + endpoint, timeout=3) as response:
                    if response.status == 200:
                        ok = True
                        detail = f"Gateway HTTP health check passed at {endpoint}."
                        break
            except (OSError, urllib.error.URLError):
                continue

    status = "connected" if ok else "unavailable"
    err = "" if ok else (result.get("error") or result.get("output")
                         or "OpenClaw Gateway did not respond.")
    con = _connect(db_path)
    try:
        con.execute(
            """
            UPDATE openclaw_connection
            SET last_status=?, last_checked_at=?, last_error=?, updated_at=?
            WHERE id=1
            """,
            (status, _stamp(), _safe_excerpt(err), _stamp()),
        )
        con.commit()
    finally:
        con.close()
    out = {"ok": ok, "status": status, "detail": detail if ok else err}
    if ok and bool(cfg.get("delivery_enabled")) and str(cfg.get("delivery_channel") or "").lower() == "telegram" and str(cfg.get("delivery_target") or "").strip():
        note = send_connection_announcement(db_path=db_path)
        if note:
            out["announcement"] = note
    return out


def send_connection_announcement(db_path: str = DB_DEFAULT) -> str:
    cfg = get_connection(db_path)
    channel = str(cfg.get("delivery_channel") or "").strip().lower()
    target = str(cfg.get("delivery_target") or "").strip()
    if channel != "telegram" or not target:
        return ""
    message_text = "ThaiBMA EWS connected to OpenClaw เรียบร้อยแล้วครับ"
    result = run_cli([
        "message", "send",
        "--channel", channel,
        "--target", target,
        "--message", message_text,
        "--json",
    ], timeout=30, db_path=db_path)
    if result.get("ok"):
        return "Telegram connection announcement sent."
    return "Telegram announcement failed: " + (result.get("error") or result.get("output") or "unknown error")


def list_jobs(db_path: str = DB_DEFAULT) -> list[dict]:
    ensure_schema(db_path)
    con = _connect(db_path)
    try:
        return [dict(r) for r in con.execute(
            "SELECT * FROM openclaw_jobs ORDER BY id").fetchall()]
    finally:
        con.close()


def get_job(job_id: int, db_path: str = DB_DEFAULT) -> dict | None:
    ensure_schema(db_path)
    con = _connect(db_path)
    try:
        row = con.execute(
            "SELECT * FROM openclaw_jobs WHERE id=?", (int(job_id),)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def upsert_job(name: str, task_type: str, cron_expression: str,
               timezone: str = "Asia/Bangkok", enabled: bool = False,
               timeout_seconds: int = 900, job_id: int | None = None,
               db_path: str = DB_DEFAULT) -> int:
    ensure_schema(db_path)
    name = (name or "").strip()
    cron_expression = (cron_expression or "").strip()
    if not name:
        raise ValueError("Job name is required.")
    if task_type not in TASKS:
        raise ValueError("Unsupported task type.")
    if len(cron_expression.split()) not in (5, 6):
        raise ValueError("Cron expression must contain 5 or 6 fields.")
    timeout_seconds = max(30, min(int(timeout_seconds), 7200))
    now = _stamp()
    con = _connect(db_path)
    try:
        if job_id:
            con.execute(
                """
                UPDATE openclaw_jobs
                SET name=?, task_type=?, cron_expression=?, timezone=?, enabled=?,
                    timeout_seconds=?,
                    sync_status=CASE WHEN openclaw_job_id IS NULL
                                     THEN 'local_only' ELSE 'needs_resync' END,
                    updated_at=?
                WHERE id=?
                """,
                (name, task_type, cron_expression, timezone, int(bool(enabled)),
                 timeout_seconds, now, int(job_id)),
            )
            result_id = int(job_id)
        else:
            cur = con.execute(
                """
                INSERT INTO openclaw_jobs
                (name, task_type, cron_expression, timezone, enabled,
                 timeout_seconds, command_json, sync_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, '{}', 'local_only', ?, ?)
                """,
                (name, task_type, cron_expression, timezone, int(bool(enabled)),
                 timeout_seconds, now, now),
            )
            result_id = int(cur.lastrowid)
        con.commit()
        return result_id
    finally:
        con.close()


def _worker_argv(task_type: str) -> list[str]:
    return [sys.executable, os.path.join(HERE, "openclaw_worker.py"),
            "--task", task_type]


def _find_job_id(data) -> str:
    if isinstance(data, dict):
        for key in ("id", "jobId", "job_id"):
            if data.get(key):
                return str(data[key])
        for value in data.values():
            found = _find_job_id(value)
            if found:
                return found
    if isinstance(data, list):
        for value in data:
            found = _find_job_id(value)
            if found:
                return found
    return ""


def sync_job(job_id: int, db_path: str = DB_DEFAULT) -> dict:
    cfg = get_connection(db_path)
    if not bool(cfg.get("sharing_enabled")):
        return {"ok": False, "error": "Enable OpenClaw sharing before syncing jobs."}
    job = get_job(job_id, db_path)
    if not job:
        return {"ok": False, "error": "Job was not found."}

    argv_json = json.dumps(_worker_argv(job["task_type"]), ensure_ascii=True)
    display_name = JOB_PREFIX + job["name"]
    existing_id = str(job.get("openclaw_job_id") or "").strip()
    command_mode = cli_supports("create", "--command-argv", db_path)
    delivery_args = ["--no-deliver"]
    if bool(cfg.get("delivery_enabled")):
        delivery_args = [
            "--announce",
            "--channel", str(cfg.get("delivery_channel") or ""),
            "--to", str(cfg.get("delivery_target") or ""),
        ]

    if command_mode and existing_id:
        args = [
            "cron", "edit", existing_id,
            "--name", display_name,
            "--cron", job["cron_expression"],
            "--tz", job["timezone"],
            "--command-argv", argv_json,
            "--command-cwd", HERE,
            "--timeout-seconds", str(job["timeout_seconds"]),
            *delivery_args,
        ]
    elif command_mode:
        args = [
            "cron", "create",
            "--cron", job["cron_expression"],
            "--name", display_name,
            "--tz", job["timezone"],
            "--command-argv", argv_json,
            "--command-cwd", HERE,
            "--timeout-seconds", str(job["timeout_seconds"]),
            *delivery_args,
            "--json",
        ]
    else:
        prompt = (
            "Run this approved ThaiBMA EWS scheduled task with the local execution "
            f"tool using this exact argv JSON: {argv_json}. "
            f"Use working directory {json.dumps(HERE)}. "
            "Do not install packages, change configuration, edit source files, or "
            "run any other command. Return the worker stdout JSON as the final "
            "response. Prospective lead_window_days is an EWS horizon and must not "
            "be described as an observed default date."
        )
        if existing_id:
            args = [
                "cron", "edit", existing_id,
                "--name", display_name,
                "--cron", job["cron_expression"],
                "--tz", job["timezone"],
                "--message", prompt,
                "--session", "isolated",
                "--tools", "exec,read,write,edit,apply_patch",
                "--timeout-seconds", str(job["timeout_seconds"]),
                *delivery_args,
            ]
        else:
            args = [
                "cron", "create",
                "--cron", job["cron_expression"],
                "--name", display_name,
                "--tz", job["timezone"],
                "--message", prompt,
                "--session", "isolated",
                "--tools", "exec,read,write,edit,apply_patch",
                "--timeout-seconds", str(job["timeout_seconds"]),
                *delivery_args,
                "--json",
            ]
    result = run_cli(args, timeout=45, db_path=db_path)
    now = _stamp()
    remote_id = existing_id
    if result.get("ok"):
        remote_id = existing_id or _find_job_id(result.get("data"))
        if not remote_id:
            remote_listing = run_cli(["cron", "list", "--all", "--json"], timeout=30, db_path=db_path)
            data = remote_listing.get("data")
            jobs = data if isinstance(data, list) else (data.get("jobs", []) if isinstance(data, dict) else [])
            for item in jobs:
                if str(item.get("name") or "") == display_name:
                    remote_id = str(item.get("jobId") or item.get("id") or "")
                    break
        if not remote_id:
            result = dict(result)
            result["ok"] = False
            result["error"] = "OpenClaw synced the job but no remote job ID could be resolved."
    toggle_result = None
    if result.get("ok"):
        toggle = "enable" if bool(job.get("enabled")) else "disable"
        toggle_result = run_cli(["cron", toggle, remote_id], timeout=30, db_path=db_path)
        if not toggle_result.get("ok"):
            result["warning"] = toggle_result.get("error") or toggle_result.get("output")

    con = _connect(db_path)
    try:
        if result.get("ok"):
            con.execute(
                """
                UPDATE openclaw_jobs
                SET openclaw_job_id=?, command_json=?, sync_status='synced',
                    last_synced_at=?, last_result=?, updated_at=?
                WHERE id=?
                """,
                (remote_id, argv_json, now,
                 "Synced as deterministic command job" if command_mode
                 else "Synced as isolated agent job (CLI has no --command-argv)",
                 now, int(job_id)),
            )
            con.execute(
                "UPDATE openclaw_jobs SET sync_status=? WHERE id=?",
                ("synced_command" if command_mode else "synced_agent", int(job_id)),
            )
        else:
            con.execute(
                """
                UPDATE openclaw_jobs
                SET sync_status='error', last_result=?, updated_at=? WHERE id=?
                """,
                (_safe_excerpt(result.get("error") or result.get("output", "")),
                 now, int(job_id)),
            )
        con.execute(
            """
            INSERT INTO openclaw_delivery_log
            (job_id, event_type, status, detail, created_at)
            VALUES (?, 'job_sync', ?, ?, ?)
            """,
            (int(job_id), "ok" if result.get("ok") else "error",
             _safe_excerpt(result.get("error") or result.get("output", "")), now),
        )
        con.commit()
    finally:
        con.close()
    return result


def run_job_now(job_id: int, db_path: str = DB_DEFAULT) -> dict:
    job = get_job(job_id, db_path)
    if not job:
        return {"ok": False, "error": "Job was not found."}
    remote_id = str(job.get("openclaw_job_id") or "").strip()
    if not remote_id:
        return {"ok": False, "error": "Sync this job to OpenClaw first."}
    result = run_cli(["cron", "run", remote_id], timeout=45,
                     db_path=db_path)
    now = _stamp()
    con = _connect(db_path)
    try:
        con.execute(
            """
            UPDATE openclaw_jobs SET last_run_at=?, last_result=?, updated_at=?
            WHERE id=?
            """,
            (now, "Queued in OpenClaw" if result.get("ok") else
             _safe_excerpt(result.get("error") or result.get("output", "")),
             now, int(job_id)),
        )
        con.execute(
            """
            INSERT INTO openclaw_delivery_log
            (job_id, event_type, status, detail, created_at)
            VALUES (?, 'manual_run', ?, ?, ?)
            """,
            (int(job_id), "queued" if result.get("ok") else "error",
             _safe_excerpt(result.get("error") or result.get("output", "")), now),
        )
        con.commit()
    finally:
        con.close()
    return result


def list_alerts(search: str = "", status: str = "all", limit: int = 500,
                db_path: str = DB_DEFAULT) -> list[dict]:
    ensure_schema(db_path)
    clauses, params = [], []
    if search.strip():
        clauses.append(
            "(firm_id LIKE ? OR account_id LIKE ? OR alert_level LIKE ?)")
        term = f"%{search.strip()}%"
        params.extend([term, term, term])
    if status and status != "all":
        clauses.append("status=?")
        params.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT * FROM lead_alert_queue" + where
            + " ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def list_logs(limit: int = 100, db_path: str = DB_DEFAULT) -> list[dict]:
    ensure_schema(db_path)
    con = _connect(db_path)
    try:
        return [dict(r) for r in con.execute(
            "SELECT * FROM openclaw_delivery_log ORDER BY id DESC LIMIT ?",
            (int(limit),)).fetchall()]
    finally:
        con.close()


def schema_summary(db_path: str = DB_DEFAULT) -> dict:
    ensure_schema(db_path)
    con = _connect(db_path)
    try:
        result = {}
        for table in (
            "openclaw_connection", "openclaw_jobs",
            "lead_alert_queue", "openclaw_delivery_log",
        ):
            result[table] = con.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return result
    finally:
        con.close()
