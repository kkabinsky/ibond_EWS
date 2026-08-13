from pathlib import Path

p = Path(r'D:\tadgan_gaf\cmdf_credit_app\openclaw_connector.py')
s = p.read_text(encoding='utf-8')
old = '''def sync_job(job_id: int, db_path: str = DB_DEFAULT) -> dict:
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
            "cron", "create", job["cron_expression"],
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
                *delivery_args,
            ]
        else:
            args = [
                "cron", "create", job["cron_expression"], prompt,
                "--name", display_name,
                "--tz", job["timezone"],
                "--session", "isolated",
                *delivery_args,
                "--json",
            ]
    result = run_cli(args, timeout=45, db_path=db_path)
    now = _stamp()
    if result.get("ok"):
        remote_id = existing_id or _find_job_id(result.get("data"))
        if not remote_id:
            remote_id = _find_job_id(_extract_json(result.get("output", "")))
        if not remote_id:
            result = dict(result)
            result["ok"] = False
            result["error"] = "OpenClaw created the job but did not return a job ID."
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
            toggle = "enable" if bool(job.get("enabled")) else "disable"
            toggle_result = run_cli(
                ["cron", toggle, remote_id], timeout=30, db_path=db_path)
            if not toggle_result.get("ok"):
                result["warning"] = toggle_result.get("error") or toggle_result.get("output")
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
'''
new = '''def sync_job(job_id: int, db_path: str = DB_DEFAULT) -> dict:
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
            "cron", "create", job["cron_expression"],
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
                "cron", "create", job["cron_expression"],
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
            toggle = "enable" if bool(job.get("enabled")) else "disable"
            toggle_result = run_cli(
                ["cron", toggle, remote_id], timeout=30, db_path=db_path)
            if not toggle_result.get("ok"):
                result["warning"] = toggle_result.get("error") or toggle_result.get("output")
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
'''
if old not in s:
    raise SystemExit('old block not found')
p.write_text(s.replace(old, new), encoding='utf-8')
print('patched')
