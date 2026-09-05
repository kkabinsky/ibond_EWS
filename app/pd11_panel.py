# -*- coding: utf-8 -*-
"""ข้อมูลและกราฟ PD ของ 11 วิธี สำหรับสองเมนูใหม่ใน app.py

แยกออกมาเป็นโมดูลของตัวเอง เพื่อให้ app.py เพิ่มแค่ปุ่มกับ view
ตรรกะการอ่านฐานข้อมูล การวาดกราฟ และการสั่งรัน อยู่ที่นี่ทั้งหมด

สิบเอ็ดวิธีมาจากสองแหล่ง
    เก้าวิธี   dataset2/standalone_leadtime_runs/standalone_*/standalone_leadtime_results.db
               XGBoost CatBoost LightGBM RandomForest Logistic Probit
               HistGradientBoosting GradientBoosting ExtraTrees
    สองแนวทาง  dataset2/pd_curves.db
               Approach 1 pooled logit และ Approach 2 gradient boosting

ทั้งสองแหล่งใช้ StratifiedGroupKFold แบ่งตาม issuer เหมือนกัน แต่ใช้พาเนลต่างกัน
เก้าวิธีรันบนพาเนลเต็ม ส่วนสองแนวทางรันบนชุดความเสี่ยงที่ตัดเดือนหลังเหตุการณ์แรก
ตารางผลจึงมีคอลัมน์ panel บอกไว้ ไม่ควรเทียบตัวเลขข้ามพาเนลโดยไม่ดูคอลัมน์นี้

ใช้งานตรงจาก command line ได้ด้วย
    python pd11_panel.py performance
    python pd11_panel.py issuer "THAI AIRWAYS"
    python pd11_panel.py run
"""
from __future__ import annotations

import argparse
import io
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ในคลังนี้โค้ดอยู่ในโฟลเดอร์ย่อย ส่วนข้อมูลเป็นของ ThaiBMA จึงไม่ได้ commit
# thaibma_paths จึงหาตำแหน่งข้อมูลให้เอง ถ้าไม่มีก็ถอยไปใช้โฟลเดอร์ของไฟล์นี้
try:
    from thaibma_paths import DATA_ROOT
    ROOT = Path(DATA_ROOT)
except Exception:
    ROOT = HERE

APP_DB = ROOT / "cmdf_credit.db"
DATASET2 = ROOT / "dataset2"
PD_CURVES_DB = DATASET2 / "pd_curves.db"
RUN_ROOT = DATASET2 / "standalone_leadtime_runs"

NINE_METHODS = ("XGBoost", "CatBoost", "LightGBM", "RandomForest", "Logistic",
                "Probit", "HistGradientBoosting", "GradientBoosting", "ExtraTrees")
TWO_APPROACHES = ("Approach 1", "Approach 2")
ALL_METHODS = NINE_METHODS + TWO_APPROACHES

# สีคงที่ต่อวิธี เพื่อให้กราฟทุกใบอ่านเทียบกันได้
METHOD_COLOR = {
    "XGBoost": "#1f77b4",
    "CatBoost": "#2ca02c",
    "LightGBM": "#9467bd",
    "RandomForest": "#d62728",
    "Logistic": "#8c564b",
    "Probit": "#e377c2",
    "HistGradientBoosting": "#7f7f7f",
    "GradientBoosting": "#bcbd22",
    "ExtraTrees": "#17becf",
    "Approach 1": "#1d4ed8",
    "Approach 2": "#c2410c",
}
PERFORMANCE_TABLE = "pd11_model_performance"
SCORES_TABLE = "pd11_oof_scores"
METADATA_TABLE = "pd11_run_metadata"


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def _month_key(values: pd.Series) -> pd.Series:
    """คีย์เดือนแบบ YYYY-MM

    สองแหล่งเก็บเดือนต่างรูปกัน เก้าวิธีเก็บเป็น 2009-09
    ส่วนสองแนวทางเก็บเป็น 2009-09-01 ถ้าไม่ทำให้ตรงกันก่อน merge
    จะได้ตารางที่ยาวเป็นสองเท่าและทุกช่องเป็นค่าว่าง
    """
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m")


def latest_run_db() -> Path | None:
    """ฐานผลของการรันเก้าวิธีครั้งล่าสุดที่มีตารางที่ต้องใช้ครบ"""
    if not RUN_ROOT.is_dir():
        return None
    for directory in sorted(RUN_ROOT.iterdir(),
                            key=lambda p: p.stat().st_mtime, reverse=True):
        candidate = directory / "standalone_leadtime_results.db"
        if not candidate.is_file():
            continue
        with closing(_readonly(candidate)) as connection:
            names = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        if {"standalone_oof_scores", "standalone_model_summary"} <= names:
            return candidate
    return None


def sources() -> dict[str, Path | None]:
    run_db = latest_run_db()
    return {
        "nine_methods": run_db,
        "two_approaches": PD_CURVES_DB if PD_CURVES_DB.is_file() else None,
    }


def missing_sources() -> list[str]:
    found = sources()
    notes = []
    if found["nine_methods"] is None:
        notes.append("ยังไม่มีผลเก้าวิธี ให้กดปุ่ม Run PD หรือรัน "
                     "dataset2/leadtime_allmethods.py run")
    if found["two_approaches"] is None:
        notes.append("ยังไม่มีผลสองแนวทาง ให้กดปุ่ม Run PD หรือรัน "
                     "dataset2/pd_curves.py build")
    return notes


# ------------------------------------------------------------------ ผลรวม
def load_performance() -> pd.DataFrame:
    """หนึ่งแถวต่อหนึ่งวิธี ครบทั้งสิบเอ็ดวิธี อ่านจาก sqlite ทั้งหมด"""
    found = sources()
    records: list[dict[str, Any]] = []

    if found["nine_methods"] is not None:
        with closing(_readonly(found["nine_methods"])) as connection:
            summary = pd.read_sql_query(
                "SELECT * FROM standalone_model_summary", connection)
        for row in summary.itertuples(index=False):
            records.append({
                "method": str(row.model),
                "family": "nine_methods",
                "panel": "full",
                "n_rows": int(row.n_rows),
                "auc_oof": float(row.auc_oof),
                "average_precision_oof": float(row.average_precision_oof),
                "recall_at_workload": float(row.recall_at_workload),
                "precision_at_workload": float(row.precision_at_workload),
                "detected_events": int(row.all_detected),
                "evaluable_events": int(row.all_evaluable_events),
                "lead_median_days": float(row.all_lead_median_days)
                if pd.notna(row.all_lead_median_days) else np.nan,
                "threshold": float(row.threshold_latest_cross_section),
                "threshold_rule": "latest_cross_section_quantile",
                "runtime_seconds": float(row.runtime_seconds),
                "source": str(found["nine_methods"]),
            })

    if found["two_approaches"] is not None:
        with closing(_readonly(found["two_approaches"])) as connection:
            metrics = pd.read_sql_query(
                "SELECT * FROM pd3m_model_metrics", connection)
            events = pd.read_sql_query(
                "SELECT * FROM pd3m_event_issuers", connection)
            metadata = pd.read_sql_query(
                "SELECT * FROM pd3m_run_metadata", connection).iloc[0]
        for row in metrics.itertuples(index=False):
            key = str(row.approach)
            hit = events[f"{key}_detected"].astype(bool)
            days = pd.to_numeric(events.loc[hit, f"{key}_lead_days"],
                                 errors="coerce").dropna()
            records.append({
                "method": "Approach 1" if key == "approach1" else "Approach 2",
                "family": "two_approaches",
                "panel": "risk_set",
                "n_rows": int(metadata.rows),
                "auc_oof": float(row.auc_oof),
                "average_precision_oof": float(row.average_precision_oof),
                "recall_at_workload": float(row.recall_at_workload),
                "precision_at_workload": float(row.precision_at_workload),
                "detected_events": int(hit.sum()),
                "evaluable_events": int(len(events)),
                "lead_median_days": float(days.median()) if len(days) else np.nan,
                "threshold": float(row.rank_threshold),
                "threshold_rule": "historical_top_5pct_rank",
                "runtime_seconds": float(row.runtime_seconds),
                "source": str(found["two_approaches"]),
            })

    if not records:
        return pd.DataFrame(columns=[
            "method", "family", "panel", "n_rows", "auc_oof",
            "average_precision_oof", "recall_at_workload",
            "precision_at_workload", "detected_events", "evaluable_events",
            "lead_median_days", "threshold", "threshold_rule",
            "runtime_seconds", "source"])
    frame = pd.DataFrame(records)
    order = {name: index for index, name in enumerate(ALL_METHODS)}
    frame["_o"] = frame["method"].map(order).fillna(99)
    return frame.sort_values(["_o"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------- PD รายบริษัท
def list_issuers() -> pd.DataFrame:
    """รายชื่อบริษัทที่มี PD ให้วาด รวมกับสถานะว่าเกิดเหตุการณ์หรือไม่"""
    found = sources()
    frames = []
    if found["nine_methods"] is not None:
        with closing(_readonly(found["nine_methods"])) as connection:
            frames.append(pd.read_sql_query(
                "SELECT DISTINCT issuer_code FROM standalone_oof_scores", connection))
    if found["two_approaches"] is not None:
        with closing(_readonly(found["two_approaches"])) as connection:
            frames.append(pd.read_sql_query(
                "SELECT DISTINCT issuer_code, firm_name FROM pd3m_oof_scores",
                connection))
    if not frames:
        return pd.DataFrame(columns=["issuer_code", "firm_name", "has_event"])
    names = frames[-1]
    if "firm_name" not in names.columns:
        names["firm_name"] = names["issuer_code"]
    merged = (pd.concat(frames, ignore_index=True)
              .drop_duplicates("issuer_code")
              [["issuer_code"]].merge(names, on="issuer_code", how="left"))
    merged["firm_name"] = merged["firm_name"].fillna(merged["issuer_code"])
    events = set()
    if found["two_approaches"] is not None:
        with closing(_readonly(found["two_approaches"])) as connection:
            events = set(pd.read_sql_query(
                "SELECT issuer_code FROM pd3m_event_issuers",
                connection)["issuer_code"].astype(str))
    merged["has_event"] = merged["issuer_code"].astype(str).isin(events)
    return merged.sort_values("issuer_code").reset_index(drop=True)


def resolve_issuer(query: str) -> tuple[str, str]:
    """รับ issuer_code หรือชื่อบางส่วน คืนรหัสและชื่อ"""
    names = list_issuers()
    if names.empty:
        raise RuntimeError("ยังไม่มีผล PD ให้ค้นหา")
    text = str(query).strip()
    exact = names[names["issuer_code"].astype(str).str.lower().eq(text.lower())]
    if len(exact) == 1:
        return str(exact.iloc[0]["issuer_code"]), str(exact.iloc[0]["firm_name"])
    hit = names[names["firm_name"].astype(str).str.contains(text, case=False, regex=False)
                | names["issuer_code"].astype(str).str.contains(text, case=False, regex=False)]
    if hit.empty:
        raise RuntimeError(f"ไม่พบบริษัทที่ตรงกับ {query!r}")
    return str(hit.iloc[0]["issuer_code"]), str(hit.iloc[0]["firm_name"])


def load_issuer_pd(issuer_code: str) -> pd.DataFrame:
    """หนึ่งแถวต่อหนึ่งเดือน คอลัมน์เป็น PD ของแต่ละวิธี"""
    found = sources()
    pieces = []
    if found["nine_methods"] is not None:
        with closing(_readonly(found["nine_methods"])) as connection:
            columns = [d[1] for d in connection.execute(
                "PRAGMA table_info(standalone_oof_scores)")]
            wanted = ["month", "target_pre3m"] + [
                c for c in columns if c.startswith("oof_score_")]
            frame = pd.read_sql_query(
                "SELECT " + ", ".join(f'"{c}"' for c in wanted)
                + " FROM standalone_oof_scores WHERE issuer_code = ?",
                connection, params=[issuer_code])
        rename = {}
        for method in NINE_METHODS:
            column = f"oof_score_{method.lower()}"
            if column in frame.columns:
                rename[column] = method
        frame = frame.rename(columns=rename)
        frame["month"] = _month_key(frame["month"])
        pieces.append(frame[["month", "target_pre3m"] + list(rename.values())])

    if found["two_approaches"] is not None:
        with closing(_readonly(found["two_approaches"])) as connection:
            frame = pd.read_sql_query(
                "SELECT month, y_pre3m, pd_approach1, pd_approach2, "
                "alarm_approach1, alarm_approach2 FROM pd3m_oof_scores "
                "WHERE issuer_code = ?", connection, params=[issuer_code])
        frame = frame.rename(columns={
            "y_pre3m": "target_pre3m",
            "pd_approach1": "Approach 1",
            "pd_approach2": "Approach 2"})
        frame["month"] = _month_key(frame["month"])
        pieces.append(frame)

    if not pieces:
        return pd.DataFrame(columns=["month"])
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="month", how="outer",
                              suffixes=("", "_dup"))
    for column in [c for c in merged.columns if c.endswith("_dup")]:
        base = column[:-4]
        merged[base] = merged[base].fillna(merged[column])
        merged = merged.drop(columns=column)
    merged["month_dt"] = pd.to_datetime(merged["month"], errors="coerce")
    return merged.sort_values("month_dt").reset_index(drop=True)


def event_month_for(issuer_code: str) -> pd.Timestamp | None:
    found = sources()
    if found["two_approaches"] is None:
        return None
    with closing(_readonly(found["two_approaches"])) as connection:
        frame = pd.read_sql_query(
            "SELECT event_month FROM pd3m_event_issuers WHERE issuer_code = ?",
            connection, params=[issuer_code])
    if frame.empty:
        return None
    return pd.Timestamp(frame.iloc[0]["event_month"])


def figure_issuer(issuer_code: str, firm_name: str = "", methods: Iterable[str] | None = None):
    """กราฟ PD ของบริษัทหนึ่ง วาดทุกวิธีในแกนเดียว"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = load_issuer_pd(issuer_code)
    if rows.empty:
        raise RuntimeError(f"ไม่มี PD ของ {issuer_code}")
    chosen = [m for m in (methods or ALL_METHODS) if m in rows.columns]
    event_month = event_month_for(issuer_code)

    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    for method in chosen:
        ax.plot(rows["month_dt"], rows[method], linewidth=1.3,
                color=METHOD_COLOR.get(method), label=method)
    positives = rows[pd.to_numeric(rows["target_pre3m"], errors="coerce").fillna(0) > 0]
    if not positives.empty:
        ax.plot(positives["month_dt"],
                np.full(len(positives), 1.01), linestyle="none",
                marker="|", markersize=9, color="#111827",
                label="months labelled y_pre3m = 1")
    if event_month is not None:
        ax.axvline(event_month, color="#111827", linewidth=1.2)
        ax.axvspan(event_month - pd.DateOffset(months=3),
                   event_month - pd.DateOffset(months=1),
                   color="#111827", alpha=0.08)
    ax.set_ylim(-0.03, 1.06)
    ax.set_ylabel("PD 3-month", fontsize=10)
    ax.set_xlabel("month", fontsize=10)
    subtitle = (f"event month {event_month.date().isoformat()[:7]}"
                if event_month is not None else "no recorded event")
    ax.set_title(f"{firm_name or issuer_code} ({issuer_code})  |  "
                 f"{len(chosen)} methods  |  {subtitle}", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.5)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(fontsize=7.5, ncol=4, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def figure_performance(performance: pd.DataFrame | None = None):
    """แท่งเทียบผลของทุกวิธี สามแผง AUC, average precision, เหตุที่ตรวจพบ"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = performance if performance is not None else load_performance()
    if frame.empty:
        raise RuntimeError("ยังไม่มีผลให้วาด")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    colors = [METHOD_COLOR.get(m, "#64748b") for m in frame["method"]]
    panels = (
        ("auc_oof", "AUC out-of-fold", None),
        ("average_precision_oof", "Average precision", None),
        ("detected_events", "Events caught", "evaluable_events"),
    )
    for ax, (column, title, denominator) in zip(axes, panels):
        values = pd.to_numeric(frame[column], errors="coerce")
        ax.barh(frame["method"], values, color=colors)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, axis="x", linewidth=0.5)
        ax.tick_params(labelsize=8)
        if denominator is not None:
            for index, (value, total) in enumerate(
                    zip(values, frame[denominator])):
                ax.text(value, index, f"  {int(value)}/{int(total)}",
                        va="center", fontsize=7.5)
    fig.suptitle("PD models on Dataset2, read from sqlite", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------- เขียนผลรวมลงฐานของแอป
def build_tables(app_db: Path = APP_DB) -> dict[str, int]:
    """รวมผลของทั้งสิบเอ็ดวิธีเป็นตาราง pd11_ ในฐานข้อมูลของแอป"""
    performance = load_performance()
    if performance.empty:
        raise RuntimeError("ยังไม่มีผลให้รวม")

    found = sources()
    pieces = []
    if found["nine_methods"] is not None:
        with closing(_readonly(found["nine_methods"])) as connection:
            columns = [d[1] for d in connection.execute(
                "PRAGMA table_info(standalone_oof_scores)")]
            wanted = ["issuer_code", "month", "target_pre3m"] + [
                c for c in columns if c.startswith("oof_score_")]
            frame = pd.read_sql_query(
                "SELECT " + ", ".join(f'"{c}"' for c in wanted)
                + " FROM standalone_oof_scores", connection)
        rename = {f"oof_score_{m.lower()}": m for m in NINE_METHODS
                  if f"oof_score_{m.lower()}" in frame.columns}
        frame = frame.rename(columns=rename)
        frame["month"] = _month_key(frame["month"])
        pieces.append(frame)
    if found["two_approaches"] is not None:
        with closing(_readonly(found["two_approaches"])) as connection:
            frame = pd.read_sql_query(
                "SELECT issuer_code, firm_name, month, y_pre3m AS target_pre3m, "
                "pd_approach1 AS 'Approach 1', pd_approach2 AS 'Approach 2' "
                "FROM pd3m_oof_scores", connection)
        frame["month"] = _month_key(frame["month"])
        pieces.append(frame)

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=["issuer_code", "month"], how="outer",
                              suffixes=("", "_dup"))
    for column in [c for c in merged.columns if c.endswith("_dup")]:
        base = column[:-4]
        merged[base] = merged[base].fillna(merged[column])
        merged = merged.drop(columns=column)

    metadata = pd.DataFrame([{
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methods": len(performance),
        "rows": int(len(merged)),
        "issuers": int(merged["issuer_code"].nunique()),
        "nine_method_source": str(found["nine_methods"] or ""),
        "two_approach_source": str(found["two_approaches"] or ""),
        "note": "คอลัมน์ panel ในตารางผลบอกว่าแต่ละวิธีรันบนพาเนลใด "
                "full คือพาเนลเต็ม risk_set คือตัดเดือนหลังเหตุการณ์แรก",
    }])

    with closing(sqlite3.connect(Path(app_db))) as connection:
        performance.to_sql(PERFORMANCE_TABLE, connection,
                           if_exists="replace", index=False)
        merged.to_sql(SCORES_TABLE, connection, if_exists="replace", index=False)
        metadata.to_sql(METADATA_TABLE, connection, if_exists="replace", index=False)
        connection.execute(f"CREATE INDEX IF NOT EXISTS pd11_scores_issuer "
                           f"ON {SCORES_TABLE}(issuer_code, month)")
        connection.commit()
    return {PERFORMANCE_TABLE: len(performance), SCORES_TABLE: len(merged),
            METADATA_TABLE: 1}


# ------------------------------------------------------------------ สั่งรัน
def _stream(command: list[str], cwd: Path,
            progress: Callable[[str], None] | None) -> int:
    """รันคำสั่งลูก พิมพ์ทุกบรรทัดออก command line และส่งต่อให้ GUI ด้วย"""
    print(f"\n$ {' '.join(command)}", flush=True)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        command, cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1, env=environment)
    for line in process.stdout:
        line = line.rstrip()
        print(line, flush=True)
        if progress is not None and line:
            progress(line)
    return process.wait()


def run_pd(progress: Callable[[str], None] | None = None,
           app_db: Path = APP_DB, skip_nine: bool = False) -> dict[str, Any]:
    """รัน PD ครบทุกวิธี แล้วเขียนผลรวมลงฐานข้อมูลของแอป

    สองแนวทางใช้เวลาราวหนึ่งนาที เก้าวิธีใช้เวลาราวสิบนาที
    ทุกบรรทัดของโปรแกรมลูกถูกพิมพ์ออก command line ตามที่สั่ง
    """
    started = time.perf_counter()
    steps: list[dict[str, Any]] = []

    def say(message: str) -> None:
        print(message, flush=True)
        if progress is not None:
            progress(message)

    say("เริ่มรัน PD ทุกวิธีบน Dataset2")
    plan = [("สองแนวทาง Approach 1 และ Approach 2",
             [sys.executable, "pd_curves.py", "build"])]
    if not skip_nine:
        plan.append(("เก้าวิธี", [sys.executable, "leadtime_allmethods.py", "run"]))

    for label, command in plan:
        say(f"--- {label}")
        code = _stream(command, DATASET2, progress)
        steps.append({"step": label, "command": " ".join(command),
                      "exit_code": code})
        if code != 0:
            say(f"{label} จบด้วยรหัส {code} จึงหยุด")
            return {"ok": False, "steps": steps,
                    "runtime_seconds": time.perf_counter() - started}

    say("รวมผลลงฐานข้อมูลของแอป")
    written = build_tables(app_db)
    for name, count in written.items():
        say(f"  {name}  {count:,} แถว")
    performance = load_performance()
    say(format_performance(performance))
    say(f"เสร็จใน {time.perf_counter() - started:,.1f} วินาที")
    return {"ok": True, "steps": steps, "tables": written,
            "performance": performance,
            "runtime_seconds": time.perf_counter() - started}


def format_performance(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "ยังไม่มีผล"
    show = frame[["method", "panel", "auc_oof", "average_precision_oof",
                  "recall_at_workload", "detected_events", "evaluable_events",
                  "lead_median_days"]].copy()
    show.columns = ["method", "panel", "AUC", "AP", "recall",
                    "detected", "evaluable", "lead_med_days"]
    return "\nPD model performance\n\n" + show.to_string(index=False)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="PD ของ 11 วิธี")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("performance", help="พิมพ์ผลของทุกวิธี")
    sub.add_parser("sources", help="บอกว่าอ่านผลจากไฟล์ไหน")
    p_issuer = sub.add_parser("issuer", help="วาดกราฟ PD ของบริษัทหนึ่ง")
    p_issuer.add_argument("query")
    p_issuer.add_argument("--out", type=Path, default=None)
    p_run = sub.add_parser("run", help="รัน PD ทุกวิธีแล้วเขียนลงฐานข้อมูล")
    p_run.add_argument("--skip-nine", action="store_true",
                       help="รันแต่สองแนวทาง ข้ามเก้าวิธีที่ใช้เวลานาน")
    sub.add_parser("tables", help="รวมผลที่มีอยู่ลงฐานข้อมูลของแอป")
    args = parser.parse_args()

    if args.command == "sources":
        for name, path in sources().items():
            print(f"  {name:16s} {path or 'ยังไม่มี'}")
        for note in missing_sources():
            print(f"  ! {note}")
        return
    if args.command == "performance":
        print(format_performance(load_performance()))
        return
    if args.command == "issuer":
        code, name = resolve_issuer(args.query)
        fig = figure_issuer(code, name)
        out = args.out or (DATASET2 / "pd_figures" / f"pd11_issuer_{code}.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"เขียนกราฟ {out}")
        return
    if args.command == "run":
        result = run_pd(skip_nine=args.skip_nine)
        raise SystemExit(0 if result["ok"] else 1)
    if args.command == "tables":
        written = build_tables()
        for name, count in written.items():
            print(f"  {name}  {count:,} แถว")
        return


if __name__ == "__main__":
    main()
