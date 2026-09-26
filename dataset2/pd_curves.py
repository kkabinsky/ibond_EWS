# -*- coding: utf-8 -*-
"""กราฟ PD 3 เดือน เทียบ Approach 1 กับ Approach 2 บน Dataset2

สามคำสั่งหลัก

    python pd_curves.py build
        คำนวณ PD ของทั้งสอง approach ด้วย protocol เดียวกัน แล้วเก็บลง sqlite
        ตารางใหม่ทั้งหมดขึ้นต้นด้วย pd3m_ อยู่ในไฟล์ pd_curves.db
        ฐานข้อมูลต้นทาง cmdf_credit.db ถูกเปิดแบบอ่านเท่านั้น ไม่มีการเขียนทับ

    python pd_curves.py events
        กราฟ PD ของ 32 บริษัทที่เกิด default เทียบสอง approach ในภาพเดียว
        ได้สองภาพ คือแบบตาราง 32 ช่อง และแบบวางเส้นทับกันโดยจัดให้เดือนเหตุ
        การณ์อยู่ตรงกันทุกบริษัท

    python pd_curves.py issuer "<ชื่อบริษัท หรือ issuer_code>"
        กราฟ PD ของบริษัทใดก็ได้ ระบุเป็น issuer_code หรือชื่อบางส่วนก็ได้
        ถ้าชื่อตรงหลายบริษัทจะพิมพ์รายการให้เลือก

คำสั่งช่วย

    python pd_curves.py list --contains ptt      ค้นหาชื่อบริษัท
    python pd_curves.py tables                   ดูตารางที่สร้างไว้
    python pd_curves.py self-test                ตรวจว่าโปรแกรมทำงานครบวง

นิยาม PD และค่าพารามิเตอร์ทั้งหมดคัดมาจาก benchmark_dataset1_dataset2.py
บรรทัด 169--221 ซึ่งเป็นที่มาของตัวเลข Approach 1 และ Approach 2 ที่รายงานไว้
ทั้งสอง approach ใช้สูตร PD เดียวกัน คือ sigma ของ F(x) ต่างกันแค่รูปของ F(x)
Approach 1 เป็นเชิงเส้น Approach 2 เป็นผลรวมต้นไม้
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import data_layer as DL

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

if hasattr(sys.stdout, "reconfigure"):  # ให้พิมพ์ภาษาไทยบน console ได้
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DB = HERE / "pd_curves.db"
FIGURE_DIR = HERE / "pd_figures"

# seed ของ benchmark_dataset1_dataset2.py ไม่ใช่ 42 ถ้าเปลี่ยนค่านี้ fold จะขยับ
# และ PD จะไม่ตรงกับตัวเลข Approach 1 และ 2 ที่รายงานไว้
SEED = 20260903
WORKLOAD = 0.05
N_SPLITS = 5
SCHEMA_VERSION = 1

APPROACHES = ("approach1", "approach2")
# ชุดเส้นที่วาดได้ xgboost เป็นตัวเสริม มาจากผลรันเก้าวิธี ไม่ได้คำนวณที่นี่
XGBOOST = "xgboost"
APPROACH_LABEL = {
    "approach1": "Approach 1 (pooled logit)",
    "approach2": "Approach 2 (gradient boosting)",
    XGBOOST: "XGBoost (nine-method run)",
}
APPROACH_COLOR = {"approach1": "#1d4ed8", "approach2": "#c2410c",
                  XGBOOST: "#0f766e"}
TABLES = ("pd3m_run_metadata", "pd3m_model_metrics", "pd3m_thresholds",
          "pd3m_oof_scores", "pd3m_event_issuers", "pd3m_figures")
RUN_ROOT = HERE / "standalone_leadtime_runs"


# --------------------------------------------------------------- แบบจำลอง
def make_model(approach: str, positive_weight: float) -> Any:
    """คัดค่าพารามิเตอร์มาจาก benchmark_dataset1_dataset2.py บรรทัด 169--191"""
    if approach == "approach1":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000, C=0.1, class_weight="balanced", random_state=SEED,
            )),
        ])
    if approach == "approach2":
        if xgb is None:
            raise RuntimeError("ต้องติดตั้ง xgboost ก่อน")
        return xgb.XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2.0,
            reg_lambda=3.0,
            scale_pos_weight=positive_weight,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=DL._workers(),
            random_state=SEED,
            verbosity=0,
        )
    raise ValueError(f"ไม่รู้จัก approach: {approach}")


def fit_approach(data: dict[str, Any], approach: str,
                 workload: float = WORKLOAD) -> dict[str, Any]:
    """PD นอกกลุ่มฝึก โดยแบ่ง fold ตาม issuer ทุกเดือนของบริษัทหนึ่งอยู่ fold เดียว

    เกณฑ์แจ้งเตือนคือการเรียงอันดับ PD ทั้งประวัติแล้วตัดที่ 5 เปอร์เซ็นต์แรก
    ซึ่งเป็นเกณฑ์เดียวกับที่ Approach 1 และ 2 ใช้รายงานผล
    """
    values = data["X"].to_numpy(dtype=float)
    y = np.asarray(data["y"])
    groups = data["panel"]["issuer_code"].to_numpy(dtype=str)
    positive_groups = int(pd.Series(groups[y == 1]).nunique())
    n_splits = min(N_SPLITS, positive_groups)
    if n_splits < 2:
        raise RuntimeError("บริษัทที่มีเหตุการณ์น้อยเกินกว่าจะแบ่ง fold ได้")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_of = np.full(len(y), -1, dtype=int)
    started = time.perf_counter()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(values, y, groups), start=1):
        positives = int(y[train_idx].sum())
        if positives == 0:
            raise RuntimeError(f"{approach} fold {fold} ไม่มีแถวคลาสบวกในชุดฝึก")
        print(f"  {APPROACH_LABEL[approach]}: fold {fold}/{n_splits} "
              f"({len(train_idx):,} train, {len(test_idx):,} held out)", flush=True)
        weight = float((len(train_idx) - positives) / positives)
        model = make_model(approach, weight)
        model.fit(values[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(values[test_idx])[:, 1]
        fold_of[test_idx] = fold
    if not np.isfinite(oof).all() or (fold_of < 0).any():
        raise RuntimeError(f"{approach} ให้คะแนนไม่ครบทุกแถว")

    n_flagged = max(1, int(round(workload * len(y))))
    selected = np.argsort(-oof, kind="stable")[:n_flagged]
    alarm = np.zeros(len(y), dtype=bool)
    alarm[selected] = True
    rank_threshold = float(oof[selected].min())
    latest_threshold = float(np.quantile(oof[data["latest_rows"]], 1.0 - workload))

    return {
        "approach": approach,
        "oof": oof,
        "fold_of": fold_of,
        "alarm": alarm,
        "rank_threshold": rank_threshold,
        "latest_threshold": latest_threshold,
        "n_flagged": int(n_flagged),
        "n_splits": int(n_splits),
        "metrics": {
            "approach": approach,
            "label": APPROACH_LABEL[approach],
            "auc_oof": float(roc_auc_score(y, oof)),
            "average_precision_oof": float(average_precision_score(y, oof)),
            "precision_at_workload": float(precision_score(y, alarm, zero_division=0)),
            "recall_at_workload": float(recall_score(y, alarm, zero_division=0)),
            "pd_min": float(oof.min()),
            "pd_median": float(np.median(oof)),
            "pd_max": float(oof.max()),
            "rank_threshold": rank_threshold,
            "latest_threshold": latest_threshold,
            "n_flagged": int(n_flagged),
            "n_splits": int(n_splits),
            "runtime_seconds": float(time.perf_counter() - started),
        },
    }


# ------------------------------------------------------------------ ตาราง
def _event_rows(data: dict[str, Any], runs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """หนึ่งแถวต่อหนึ่งบริษัทที่เกิดเหตุการณ์ พร้อมผลของทั้งสอง approach"""
    panel = data["panel"]
    catalog = data["all_events"]
    records = []
    for event in catalog.itertuples(index=False):
        rows = panel.index[panel["issuer_code"].astype(str).eq(event.issuer_code)]
        month = pd.to_datetime(panel.loc[rows, "month_dt"])
        event_month = pd.Timestamp(event.event_month).to_period("M").to_timestamp()
        record = {
            "issuer_code": event.issuer_code,
            "firm_name": event.firm_name,
            "event_month": event_month.date().isoformat(),
            "event_type": event.event_type,
            "event_source": event.event_source,
            "evaluable": bool(event.evaluable),
            "panel_months": int(len(rows)),
            "first_panel_month": (pd.Timestamp(event.first_panel_month)
                                  .date().isoformat()),
        }
        for approach in APPROACHES:
            run = runs[approach]
            score = pd.Series(run["oof"][rows], index=rows)
            frame = pd.DataFrame({
                "month_dt": month.to_numpy(),
                "alarm": run["alarm"][rows],
            })
            lead = DL.compute_lead_for_issuer(frame, event_month)
            at_event = score[month.dt.to_period("M").eq(event_month.to_period("M")).to_numpy()]
            before = score[month.lt(event_month).to_numpy()]
            record.update({
                f"{approach}_pd_max": float(score.max()) if len(score) else np.nan,
                f"{approach}_pd_at_event": float(at_event.iloc[0]) if len(at_event) else np.nan,
                f"{approach}_pd_max_before_event": float(before.max()) if len(before) else np.nan,
                f"{approach}_threshold": run["rank_threshold"],
                f"{approach}_alarm_months": int(run["alarm"][rows].sum()),
                f"{approach}_detected": bool(lead["actionable_alarm_found"]),
                f"{approach}_alarm_month": lead["actionable_alarm_month"],
                f"{approach}_lead_days": lead["actionable_lead_days"],
                f"{approach}_persistent_days": lead["persistent_alarm_days"],
            })
        records.append(record)
    return pd.DataFrame(records).sort_values("event_month").reset_index(drop=True)


def _score_rows(data: dict[str, Any], runs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    panel = data["panel"]
    frame = pd.DataFrame({
        "issuer_code": panel["issuer_code"].astype(str),
        "firm_name": panel["issuer_code"].astype(str).map(
            _name_lookup(data)).fillna(panel["issuer_code"].astype(str)),
        "month": pd.to_datetime(panel["month_dt"]).dt.date.astype(str),
        "y_pre3m": np.asarray(data["y"], dtype=int),
    })
    for approach in APPROACHES:
        run = runs[approach]
        frame[f"pd_{approach}"] = run["oof"]
        frame[f"fold_{approach}"] = run["fold_of"]
        frame[f"alarm_{approach}"] = run["alarm"].astype(int)
    return frame


def _name_lookup(data: dict[str, Any]) -> dict[str, str]:
    panel = data["panel"]
    if "firm_name" in panel.columns:
        pairs = (panel[["issuer_code", "firm_name"]].astype(str)
                 .drop_duplicates("issuer_code"))
        return dict(zip(pairs["issuer_code"], pairs["firm_name"]))
    catalog = pd.concat([data["all_events"], data["app_events"]], ignore_index=True)
    return dict(zip(catalog["issuer_code"].astype(str),
                    catalog["firm_name"].astype(str)))


def build(out_db: Path = DEFAULT_OUT_DB, workload: float = WORKLOAD) -> dict[str, Any]:
    """คำนวณ PD ของทั้งสอง approach แล้วเขียนตาราง pd3m_ ทั้งหมด"""
    print(f"Loading Dataset2 from {DL.DEFAULT_SOURCE_DB.resolve()}", flush=True)
    # risk_set=True คือกฎเดียวกับ Approach 1 และ 2 คือทิ้งเดือนหลังเหตุการณ์แรก
    data = DL.load_model_data(risk_set=True)
    print(DL.banner(data), flush=True)
    print(f"Risk set: {data['excluded_post_event_rows']:,} months after a first "
          f"event removed, {len(data['panel']):,} months kept", flush=True)

    runs = {}
    for approach in APPROACHES:
        print(f"\nFitting {APPROACH_LABEL[approach]}", flush=True)
        runs[approach] = fit_approach(data, approach, workload)

    scores = _score_rows(data, runs)
    events = _event_rows(data, runs)
    metrics = pd.DataFrame([runs[a]["metrics"] for a in APPROACHES])
    thresholds = pd.DataFrame([
        {"approach": a, "label": APPROACH_LABEL[a], "rule": rule,
         "threshold": runs[a][key], "workload": workload,
         "n_flagged": runs[a]["n_flagged"]}
        for a in APPROACHES
        for rule, key in (("historical_top_5pct_rank", "rank_threshold"),
                          ("latest_cross_section_quantile", "latest_threshold"))
    ])
    metadata = pd.DataFrame([{
        "schema_version": SCHEMA_VERSION,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_database": str(DL.DEFAULT_SOURCE_DB.resolve()),
        "source_table": DL.DEFAULT_TABLE,
        "risk_set": 1,
        "rows": int(len(scores)),
        "issuers": int(scores["issuer_code"].nunique()),
        "positive_months": int(scores["y_pre3m"].sum()),
        "event_issuers": int(len(events)),
        "seed": SEED,
        "workload": workload,
        "n_splits": runs["approach1"]["n_splits"],
        "approach1_source": "benchmark_dataset1_dataset2.py lines 170-176",
        "approach2_source": "benchmark_dataset1_dataset2.py lines 177-191",
        "pd_definition": "PD_3M(x) = 1 / (1 + exp(-F(x)))",
        "alarm_rule": "historical top 5% of PD by rank",
        "lead_metric_version": DL.LEAD_METRIC_VERSION,
        "features": json.dumps(data["features"]),
    }])

    out_db = Path(out_db)
    with closing(sqlite3.connect(out_db)) as connection:
        metadata.to_sql("pd3m_run_metadata", connection, if_exists="replace", index=False)
        metrics.to_sql("pd3m_model_metrics", connection, if_exists="replace", index=False)
        thresholds.to_sql("pd3m_thresholds", connection, if_exists="replace", index=False)
        scores.to_sql("pd3m_oof_scores", connection, if_exists="replace", index=False)
        events.to_sql("pd3m_event_issuers", connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS pd3m_figures ("
            "created_at TEXT, kind TEXT, issuer_code TEXT, firm_name TEXT, "
            "path TEXT, scale TEXT, n_issuers INTEGER)")
        connection.execute("CREATE INDEX IF NOT EXISTS pd3m_scores_issuer "
                           "ON pd3m_oof_scores(issuer_code, month)")
        connection.commit()

    print(f"\nWritten to {out_db}")
    for name in TABLES:
        print(f"  {name}")
    print()
    print(format_metrics(metrics))
    print()
    print(format_detection(events))
    return {"data": data, "runs": runs, "scores": scores, "events": events,
            "metrics": metrics}


# ------------------------------------------------------------------- อ่านผล
def read_scores(out_db: Path = DEFAULT_OUT_DB,
                issuer_codes: list[str] | None = None) -> pd.DataFrame:
    out_db = Path(out_db)
    if not out_db.exists():
        raise SystemExit(f"ยังไม่มี {out_db.name} ให้รัน python pd_curves.py build ก่อน")
    query = "SELECT * FROM pd3m_oof_scores"
    params: list[Any] = []
    if issuer_codes:
        query += " WHERE issuer_code IN (" + ",".join("?" * len(issuer_codes)) + ")"
        params = list(issuer_codes)
    query += " ORDER BY issuer_code, month"
    with closing(sqlite3.connect(out_db)) as connection:
        frame = pd.read_sql_query(query, connection, params=params)
    frame["month_dt"] = pd.to_datetime(frame["month"])
    return frame


def read_table(name: str, out_db: Path = DEFAULT_OUT_DB) -> pd.DataFrame:
    out_db = Path(out_db)
    if not out_db.exists():
        raise SystemExit(f"ยังไม่มี {out_db.name} ให้รัน python pd_curves.py build ก่อน")
    with closing(sqlite3.connect(out_db)) as connection:
        return pd.read_sql_query(f"SELECT * FROM {name}", connection)


def log_figure(path: Path, kind: str, out_db: Path = DEFAULT_OUT_DB,
               issuer_code: str = "", firm_name: str = "",
               scale: str = "pd", n_issuers: int = 0) -> None:
    with closing(sqlite3.connect(out_db)) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS pd3m_figures ("
            "created_at TEXT, kind TEXT, issuer_code TEXT, firm_name TEXT, "
            "path TEXT, scale TEXT, n_issuers INTEGER)")
        connection.execute(
            "INSERT INTO pd3m_figures VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind,
             issuer_code, firm_name, str(path), scale, int(n_issuers)))
        connection.commit()


# -------------------------------------------------------------------- กราฟ
def _to_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """เปอร์เซ็นไทล์ของ PD ในแต่ละ approach

    PD ของสอง approach อยู่คนละช่วง Approach 1 เกาะใกล้หนึ่งเพราะถ่วงคลาส
    ส่วน Approach 2 เกาะใกล้ศูนย์ ถ้าอยากเทียบ ``รูปทรง'' ของเส้นให้ใช้อันดับ
    """
    out = frame.copy()
    for approach in APPROACHES:
        out[f"pd_{approach}"] = out[f"pd_{approach}"].rank(pct=True)
    return out


def series_present(rows: pd.DataFrame) -> tuple[str, ...]:
    """ชื่อเส้นที่มีคอลัมน์อยู่จริงในตาราง เรียงตามลำดับที่อยากให้วาด"""
    return tuple(name for name in (*APPROACHES, XGBOOST)
                 if f"pd_{name}" in rows.columns)


def _draw_issuer(ax, rows: pd.DataFrame, thresholds: dict[str, float],
                 event_month: pd.Timestamp | None, scale: str,
                 show_threshold: bool = True) -> None:
    # วาดทุกเส้นที่มีอยู่ในตาราง ถ้าต่อ XGBoost มาด้วยก็จะได้สามเส้น
    for approach in series_present(rows):
        ax.plot(rows["month_dt"], rows[f"pd_{approach}"],
                color=APPROACH_COLOR[approach], linewidth=1.4,
                label=APPROACH_LABEL[approach])
        if show_threshold and scale == "pd" and approach in thresholds:
            ax.axhline(thresholds[approach], color=APPROACH_COLOR[approach],
                       linestyle=":", linewidth=0.9, alpha=0.75)
    if event_month is not None:
        ax.axvline(event_month, color="#111827", linewidth=1.1)
        # ช่วงที่นับว่าเตือนทัน คือ 1 ถึง 3 เดือนก่อนเดือนเหตุการณ์
        ax.axvspan(event_month - pd.DateOffset(months=3),
                   event_month - pd.DateOffset(months=1),
                   color="#111827", alpha=0.07)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)


def figure_events_grid(out_db: Path = DEFAULT_OUT_DB, scale: str = "pd") -> Path:
    events = read_table("pd3m_event_issuers", out_db)
    thresholds = _rank_thresholds(out_db)
    scores = read_scores(out_db, events["issuer_code"].astype(str).tolist())
    if scale == "rank":
        scores = _to_rank(read_scores(out_db)).loc[
            lambda f: f["issuer_code"].isin(events["issuer_code"].astype(str))]

    n = len(events)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(17, 2.35 * rows_n),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for index, event in enumerate(events.itertuples(index=False)):
        ax = axes[index]
        issuer_rows = scores[scores["issuer_code"].eq(str(event.issuer_code))]
        event_month = pd.Timestamp(event.event_month)
        _draw_issuer(ax, issuer_rows, thresholds, event_month, scale)
        marks = []
        for approach, tag in (("approach1", "A1"), ("approach2", "A2")):
            hit = bool(getattr(event, f"{approach}_detected"))
            days = getattr(event, f"{approach}_lead_days")
            marks.append(f"{tag} {int(days)}d" if hit and np.isfinite(days)
                         else f"{tag} miss")
        ax.set_title(f"{str(event.firm_name)[:24]}  ({event.event_type})\n"
                     f"{event_month.date().isoformat()[:7]}  |  "
                     + "  ".join(marks), fontsize=7.5)
        # ปล่อยให้ locator เลือกจำนวนขีดเอง ไม่งั้นบริษัทที่มีประวัติยาว
        # จะได้ปีติดกันจนอ่านไม่ออก
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in axes[n:]:
        ax.axis("off")

    ylabel = ("PD 3-month" if scale == "pd"
              else "PD percentile within approach")
    for index in range(0, len(axes), cols):
        axes[index].set_ylabel(ylabel, fontsize=8)

    handles = [Line2D([], [], color=APPROACH_COLOR[a], linewidth=1.6,
                      label=APPROACH_LABEL[a]) for a in APPROACHES]
    if scale == "pd":
        handles.append(Line2D([], [], color="#6b7280", linestyle=":",
                              linewidth=1.2,
                              label="alarm threshold, historical top 5%"))
    handles += [
        Line2D([], [], color="#111827", linewidth=1.2, label="event month"),
        matplotlib.patches.Patch(color="#111827", alpha=0.12,
                                 label="actionable window, 1-3 months before"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Dataset2 PD 3-month for the {n} issuers with a default or "
                 f"restructuring event: Approach 1 vs Approach 2",
                 fontsize=12, y=1.022)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    FIGURE_DIR.mkdir(exist_ok=True)
    suffix = "" if scale == "pd" else f"_{scale}"
    path = FIGURE_DIR / f"pd3m_32_events_grid{suffix}.jpg"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log_figure(path, "events_grid", out_db, scale=scale, n_issuers=n)
    return path


def figure_events_aligned(out_db: Path = DEFAULT_OUT_DB, months: int = 36,
                          scale: str = "pd") -> Path:
    """เส้นทุกบริษัทในแกนเดียว โดยเลื่อนให้เดือนเหตุการณ์อยู่ที่ศูนย์"""
    events = read_table("pd3m_event_issuers", out_db)
    thresholds = _rank_thresholds(out_db)
    scores = read_scores(out_db) if scale == "rank" else read_scores(
        out_db, events["issuer_code"].astype(str).tolist())
    if scale == "rank":
        scores = _to_rank(scores)
        scores = scores[scores["issuer_code"].isin(events["issuer_code"].astype(str))]

    pieces = []
    for event in events.itertuples(index=False):
        rows = scores[scores["issuer_code"].eq(str(event.issuer_code))].copy()
        if rows.empty:
            continue
        event_month = pd.Timestamp(event.event_month)
        offset = ((rows["month_dt"].dt.year - event_month.year) * 12
                  + rows["month_dt"].dt.month - event_month.month)
        rows["offset"] = offset.astype(int)
        pieces.append(rows[(rows["offset"] >= -months) & (rows["offset"] <= 0)])
    aligned = pd.concat(pieces, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for ax, (approach, title) in zip(axes, APPROACH_LABEL.items()):
        for issuer, rows in aligned.groupby("issuer_code", sort=True):
            ax.plot(rows["offset"], rows[f"pd_{approach}"],
                    color=APPROACH_COLOR[approach], linewidth=0.8, alpha=0.28)
        median = aligned.groupby("offset")[f"pd_{approach}"].median()
        ax.plot(median.index, median.to_numpy(), color=APPROACH_COLOR[approach],
                linewidth=2.6, label="median across issuers")
        if scale == "pd":
            ax.axhline(thresholds[approach], color="#111827", linestyle=":",
                       linewidth=1.2, label="alarm threshold, historical top 5%")
        ax.axvspan(-3, -1, color="#111827", alpha=0.08,
                   label="actionable window, 1-3 months before")
        ax.axvline(0, color="#111827", linewidth=1.1, label="event month")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("months relative to the event month", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[0].set_ylabel("PD 3-month" if scale == "pd"
                       else "PD percentile within approach", fontsize=9)
    axes[0].set_ylim(-0.02, 1.02)
    fig.suptitle(f"PD 3-month of the {len(events)} event issuers, aligned on the "
                 f"event month, each thin line one issuer", fontsize=12)
    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    suffix = "" if scale == "pd" else f"_{scale}"
    path = FIGURE_DIR / f"pd3m_32_events_aligned{suffix}.jpg"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log_figure(path, "events_aligned", out_db, scale=scale, n_issuers=len(events))
    return path


def source_issuers() -> pd.DataFrame:
    """รหัสและชื่อของทุกบริษัทในตารางต้นทาง ทั้ง 941 ราย"""
    with closing(sqlite3.connect(
            f"file:{DL.DEFAULT_SOURCE_DB.resolve().as_posix()}?mode=ro",
            uri=True)) as connection:
        columns = [d[1] for d in connection.execute(
            f'PRAGMA table_info("{DL.DEFAULT_TABLE}")')]
        name_column = next((c for c in ("issuer_name", "name", "issuer_name_th")
                            if c in columns), None)
        select = "issuer_code" + (f', "{name_column}"' if name_column else "")
        frame = pd.read_sql_query(
            f'SELECT DISTINCT {select} FROM "{DL.DEFAULT_TABLE}"', connection)
    frame["issuer_code"] = frame["issuer_code"].astype(str).str.strip()
    if name_column:
        frame = frame.rename(columns={name_column: "firm_name"})
        frame["firm_name"] = frame["firm_name"].astype(str).str.strip()
    else:
        frame["firm_name"] = frame["issuer_code"]
    frame = (frame.sort_values(["issuer_code", "firm_name"])
             .drop_duplicates("issuer_code", keep="first"))
    frame.loc[frame["firm_name"].isin(("", "None", "nan")), "firm_name"] = \
        frame["issuer_code"]
    return frame.sort_values("issuer_code").reset_index(drop=True)


def issuer_catalog(out_db: Path = DEFAULT_OUT_DB) -> pd.DataFrame:
    """หนึ่งแถวต่อหนึ่งบริษัทในฐาน พร้อมสถานะ default และสถิติ PD ของสอง approach

    ตารางนี้เป็นสารบัญของภาพชุดใหญ่ ใช้หาว่าบริษัทรหัสนี้อยู่หน้าไหน
    """
    scores = read_scores(out_db)
    events = read_table("pd3m_event_issuers", out_db)
    event_map = dict(zip(events["issuer_code"].astype(str),
                         events["event_month"].astype(str)))
    type_map = dict(zip(events["issuer_code"].astype(str),
                        events["event_type"].astype(str)))
    # รายชื่อตั้งต้นมาจากตารางต้นทาง ไม่ใช่จากตาราง PD เพราะชุดความเสี่ยงตัด
    # บริษัทที่ทุกเดือนอยู่หลังเหตุการณ์แรกออกไปหมด บริษัทแบบนั้นต้องยังอยู่
    # ในสารบัญ โดยระบุว่าไม่มีเดือนให้เตือน ไม่ใช่หายไปเงียบ ๆ
    all_issuers = source_issuers()
    grouped = {str(code): frame for code, frame in scores.groupby("issuer_code")}

    records = []
    for base in all_issuers.itertuples(index=False):
        issuer = str(base.issuer_code)
        rows = grouped.get(issuer)
        name = str(base.firm_name)
        if rows is not None and not rows.empty:
            name = str(rows["firm_name"].iloc[0]) or name
        record = {
            "issuer_code": issuer,
            "firm_name": name,
            "panel_months": int(len(rows)) if rows is not None else 0,
            "first_month": str(rows["month"].min()) if rows is not None and len(rows) else "",
            "last_month": str(rows["month"].max()) if rows is not None and len(rows) else "",
            "positive_months": int(pd.to_numeric(rows["y_pre3m"], errors="coerce")
                                   .fillna(0).sum()) if rows is not None else 0,
            "is_default": issuer in event_map,
            "event_month": event_map.get(issuer, ""),
            "event_type": type_map.get(issuer, ""),
            "in_risk_set": rows is not None and len(rows) > 0,
        }
        for approach in APPROACHES:
            if rows is None or rows.empty:
                record.update({f"{approach}_pd_median": np.nan,
                               f"{approach}_pd_max": np.nan,
                               f"{approach}_alarm_months": 0})
                continue
            series = pd.to_numeric(rows[f"pd_{approach}"], errors="coerce")
            record.update({
                f"{approach}_pd_median": float(series.median()),
                f"{approach}_pd_max": float(series.max()),
                f"{approach}_alarm_months": int(rows[f"alarm_{approach}"].sum()),
            })
        records.append(record)

    catalog = pd.DataFrame(records).sort_values("issuer_code").reset_index(drop=True)
    with closing(sqlite3.connect(Path(out_db))) as connection:
        catalog.to_sql("pd3m_issuer_catalog", connection,
                       if_exists="replace", index=False)
        connection.commit()
    return catalog


def figure_all_issuers(out_db: Path = DEFAULT_OUT_DB, cols: int = 4,
                       rows_per_page: int = 5, scale: str = "pd",
                       with_xgboost: bool = True, progress=None) -> list[Path]:
    """วาด PD ของทุกบริษัทในฐาน แบ่งเป็นหน้าละ cols x rows_per_page ช่อง

    ช่องของบริษัทที่เกิด default ใช้กรอบและหัวข้อสีแดง พร้อมเส้นเดือนเหตุการณ์
    ช่องของบริษัทที่ไม่เคยเกิดเหตุการณ์ใช้กรอบเทา รหัสบริษัทอยู่หน้าชื่อทุกช่อง
    """
    catalog = issuer_catalog(out_db)
    thresholds = _rank_thresholds(out_db, with_xgboost=with_xgboost)
    scores = read_scores(out_db)
    if with_xgboost:
        scores = attach_xgboost(scores)
    if scale == "rank":
        scores = _to_rank(scores)
    grouped = {str(code): frame for code, frame in scores.groupby("issuer_code")}
    drawn = series_present(scores)

    per_page = cols * rows_per_page
    pages = int(np.ceil(len(catalog) / per_page))
    FIGURE_DIR.mkdir(exist_ok=True)
    suffix = "" if scale == "pd" else f"_{scale}"
    written: list[Path] = []

    default_color = "#b91c1c"
    plain_color = "#475569"

    for page in range(pages):
        chunk = catalog.iloc[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(rows_per_page, cols,
                                 figsize=(17, 2.5 * rows_per_page), sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for index, firm in enumerate(chunk.itertuples(index=False)):
            ax = axes[index]
            rows = grouped.get(str(firm.issuer_code))
            is_default = bool(firm.is_default)
            event_month = (pd.Timestamp(firm.event_month)
                           if is_default and firm.event_month else None)
            accent = default_color if is_default else plain_color
            if rows is None or rows.empty:
                ax.text(0.5, 0.5, "no months in the risk set",
                        ha="center", va="center", fontsize=7,
                        color=plain_color, transform=ax.transAxes)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(alpha=0.25, linewidth=0.5)
                ax.tick_params(labelsize=7)
            else:
                _draw_issuer(ax, rows, thresholds, event_month, scale)
                locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            tag = (f"DEFAULT {str(firm.event_month)[:7]} ({firm.event_type})"
                   if is_default else "no event")
            ax.set_title(f"{firm.issuer_code}  |  {tag}\n"
                         f"{str(firm.firm_name)[:26]}",
                         fontsize=7.2, color=accent,
                         fontweight="bold" if is_default else "normal")
            for spine in ax.spines.values():
                spine.set_color(accent)
                spine.set_linewidth(1.6 if is_default else 0.8)
        for ax in axes[len(chunk):]:
            ax.axis("off")

        ylabel = ("PD 3-month" if scale == "pd"
                  else "PD percentile within approach")
        for index in range(0, len(axes), cols):
            axes[index].set_ylabel(ylabel, fontsize=8)

        handles = [Line2D([], [], color=APPROACH_COLOR[a], linewidth=1.6,
                          label=APPROACH_LABEL[a]) for a in drawn]
        if scale == "pd":
            handles.append(Line2D([], [], color="#6b7280", linestyle=":",
                                  linewidth=1.2,
                                  label="alarm threshold, historical top 5%"))
        handles += [
            Line2D([], [], color="#111827", linewidth=1.2, label="event month"),
            matplotlib.patches.Patch(color="#111827", alpha=0.12,
                                     label="actionable window, 1-3 months before"),
            matplotlib.patches.Patch(facecolor="white", edgecolor=default_color,
                                     linewidth=1.6,
                                     label="red frame and title: issuer defaulted "
                                           "or restructured"),
            matplotlib.patches.Patch(facecolor="white", edgecolor=plain_color,
                                     linewidth=0.8,
                                     label="grey frame: no event on record"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        first = str(chunk.iloc[0]["issuer_code"])
        last = str(chunk.iloc[-1]["issuer_code"])
        fig.suptitle(f"PD 3-month, every issuer in Dataset2, page {page + 1} of "
                     f"{pages}: {first} to {last}", fontsize=12, y=1.028)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        path = FIGURE_DIR / f"pd3m_all_p{page + 1:02d}{suffix}.jpg"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)
        message = (f"  หน้า {page + 1}/{pages}  {first} ถึง {last}  "
                   f"{path.name}")
        print(message, flush=True)
        if progress is not None:
            progress(message)

    for path in written:
        log_figure(path, "all_issuers_page", out_db, scale=scale,
                   n_issuers=per_page)
    return written


def pick_non_default(out_db: Path = DEFAULT_OUT_DB, count: int = 10,
                     min_months: int = 60) -> pd.DataFrame:
    """เลือกบริษัทที่ไม่เคยเกิดเหตุการณ์ ไว้เป็นกลุ่มเทียบ

    คัดเฉพาะบริษัทที่มีประวัติยาวพอ แล้วสุ่มด้วย seed คงที่ เพื่อให้ได้ชุดเดิม
    ทุกครั้ง ไม่ใช่ชุดที่เลือกมาให้ภาพดูดี
    """
    events = set(read_table("pd3m_event_issuers", out_db)["issuer_code"].astype(str))
    with closing(sqlite3.connect(Path(out_db))) as connection:
        names = pd.read_sql_query(
            "SELECT issuer_code, firm_name, COUNT(*) AS months, "
            "SUM(y_pre3m) AS positives "
            "FROM pd3m_oof_scores GROUP BY issuer_code, firm_name", connection)
    clean = names[~names["issuer_code"].astype(str).isin(events)
                  & names["positives"].eq(0)
                  & names["months"].ge(min_months)]
    if clean.empty:
        raise SystemExit("ไม่พบบริษัทที่ไม่เคยเกิดเหตุการณ์และมีประวัติยาวพอ")
    return (clean.sort_values("issuer_code")
            .sample(n=min(count, len(clean)), random_state=SEED)
            .sort_values("issuer_code").reset_index(drop=True))


def figure_non_default(out_db: Path = DEFAULT_OUT_DB, count: int = 10,
                       scale: str = "pd") -> Path:
    """กลุ่มเทียบ บริษัทที่ไม่เคย default วาดด้วยสเกลและเกณฑ์เดียวกัน"""
    chosen = pick_non_default(out_db, count)
    thresholds = _rank_thresholds(out_db)
    scores = read_scores(out_db, chosen["issuer_code"].astype(str).tolist())
    if scale == "rank":
        allrows = _to_rank(read_scores(out_db))
        scores = allrows[allrows["issuer_code"].isin(chosen["issuer_code"].astype(str))]

    n = len(chosen)
    cols = 5
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(17, 2.6 * rows_n), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for index, firm in enumerate(chosen.itertuples(index=False)):
        ax = axes[index]
        rows = scores[scores["issuer_code"].eq(str(firm.issuer_code))]
        _draw_issuer(ax, rows, thresholds, None, scale)
        flags = [int(rows[f"alarm_{a}"].sum()) for a in APPROACHES]
        ax.set_title(f"{str(firm.firm_name)[:24]}\nno event  |  "
                     f"A1 {flags[0]} alarms  A2 {flags[1]} alarms", fontsize=7.5)
        # ปล่อยให้ locator เลือกจำนวนขีดเอง ไม่งั้นบริษัทที่มีประวัติยาว
        # จะได้ปีติดกันจนอ่านไม่ออก
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in axes[n:]:
        ax.axis("off")
    ylabel = "PD 3-month" if scale == "pd" else "PD percentile within approach"
    for index in range(0, len(axes), cols):
        axes[index].set_ylabel(ylabel, fontsize=8)

    handles = [Line2D([], [], color=APPROACH_COLOR[a], linewidth=1.6,
                      label=APPROACH_LABEL[a]) for a in APPROACHES]
    if scale == "pd":
        handles.append(Line2D([], [], color="#6b7280", linestyle=":",
                              linewidth=1.2,
                              label="alarm threshold, historical top 5%"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Control group: {n} issuers with no default or restructuring, "
                 f"same thresholds and scale", fontsize=12, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIGURE_DIR.mkdir(exist_ok=True)
    suffix = "" if scale == "pd" else f"_{scale}"
    path = FIGURE_DIR / f"pd3m_non_default_{n}{suffix}.jpg"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log_figure(path, "non_default_grid", out_db, scale=scale, n_issuers=n)

    # เก็บรายชื่อกลุ่มเทียบไว้ด้วย เพื่อให้ภาพในรายงานย้อนกลับมาตรวจได้
    summary = []
    for firm in chosen.itertuples(index=False):
        rows = scores[scores["issuer_code"].eq(str(firm.issuer_code))]
        record = {"issuer_code": str(firm.issuer_code),
                  "firm_name": str(firm.firm_name),
                  "panel_months": int(len(rows))}
        for approach in APPROACHES:
            series = rows[f"pd_{approach}"]
            record.update({
                f"{approach}_pd_median": float(series.median()),
                f"{approach}_pd_max": float(series.max()),
                f"{approach}_alarm_months": int(rows[f"alarm_{approach}"].sum()),
                f"{approach}_threshold": thresholds[approach],
            })
        summary.append(record)
    with closing(sqlite3.connect(Path(out_db))) as connection:
        pd.DataFrame(summary).to_sql("pd3m_non_default_control", connection,
                                     if_exists="replace", index=False)
        connection.commit()
    return path


def figure_issuer(query: str, out_db: Path = DEFAULT_OUT_DB,
                  scale: str = "pd") -> Path:
    issuer_code, firm_name = resolve_issuer(query, out_db)
    thresholds = _rank_thresholds(out_db)
    rows = read_scores(out_db, [issuer_code])
    if scale == "rank":
        rows = _to_rank(read_scores(out_db))
        rows = rows[rows["issuer_code"].eq(issuer_code)]
    events = read_table("pd3m_event_issuers", out_db)
    event = events[events["issuer_code"].astype(str).eq(issuer_code)]
    event_month = (pd.Timestamp(event.iloc[0]["event_month"])
                   if not event.empty else None)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    _draw_issuer(ax, rows, thresholds, event_month, scale)
    positives = rows[rows["y_pre3m"].eq(1)]
    if not positives.empty:
        ax.plot(positives["month_dt"], positives[f"pd_{APPROACHES[0]}"],
                linestyle="none", marker="o", markersize=3.4,
                markerfacecolor="none", color="#111827",
                label="months labelled y_pre3m = 1")
    for approach in APPROACHES:
        flagged = rows[rows[f"alarm_{approach}"].eq(1)]
        if not flagged.empty:
            ax.plot(flagged["month_dt"], flagged[f"pd_{approach}"],
                    linestyle="none", marker="^", markersize=4.6,
                    color=APPROACH_COLOR[approach],
                    label=f"alarm months, {approach}")
    ax.set_ylabel("PD 3-month" if scale == "pd"
                  else "PD percentile within approach", fontsize=10)
    ax.set_xlabel("month", fontsize=10)
    subtitle = (f"event {event.iloc[0]['event_type']} in "
                f"{str(event.iloc[0]['event_month'])[:7]}"
                if not event.empty else "no recorded default or restructuring")
    ax.set_title(f"{firm_name}  ({issuer_code})  |  {subtitle}", fontsize=12)
    ax.legend(fontsize=8, loc="best", frameon=False, ncol=2)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in issuer_code)
    suffix = "" if scale == "pd" else f"_{scale}"
    path = FIGURE_DIR / f"pd3m_issuer_{safe}{suffix}.jpg"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log_figure(path, "issuer", out_db, issuer_code=issuer_code,
               firm_name=firm_name, scale=scale, n_issuers=1)
    print(format_issuer(rows, issuer_code, firm_name, thresholds, event))
    return path


def _rank_thresholds(out_db: Path = DEFAULT_OUT_DB,
                     with_xgboost: bool = False) -> dict[str, float]:
    frame = read_table("pd3m_thresholds", out_db)
    frame = frame[frame["rule"].eq("historical_top_5pct_rank")]
    thresholds = dict(zip(frame["approach"], frame["threshold"].astype(float)))
    if with_xgboost:
        loaded = load_xgboost()
        if loaded is not None:
            thresholds[XGBOOST] = loaded["threshold"]
    return thresholds


# ------------------------------------------------- เส้น XGBoost จากผลรันเก้าวิธี
def _latest_run_db() -> Path | None:
    """ฐานผลของการรันเก้าวิธีครั้งล่าสุดที่มีตาราง OOF ครบ"""
    if not RUN_ROOT.is_dir():
        return None
    for directory in sorted(RUN_ROOT.iterdir(),
                            key=lambda p: p.stat().st_mtime, reverse=True):
        candidate = directory / "standalone_leadtime_results.db"
        if not candidate.is_file():
            continue
        with closing(sqlite3.connect(
                f"file:{candidate.as_posix()}?mode=ro", uri=True)) as connection:
            names = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        if "standalone_oof_scores" in names:
            return candidate
    return None


def load_xgboost(workload: float = WORKLOAD) -> dict[str, Any] | None:
    """PD ของ XGBoost รายเดือน จากผลรันเก้าวิธี

    เกณฑ์แจ้งเตือนคำนวณใหม่ด้วยกฎเดียวกับสอง approach คือเรียงทั้งประวัติแล้วตัดที่
    5 เปอร์เซ็นต์แรก ไม่ใช้ค่าที่เก็บไว้ในตารางนั้นซึ่งมาจากกฎภาพตัดขวางล่าสุด
    ถ้าใช้ต่างกฎกัน เส้นประในกราฟจะเทียบกันไม่ได้
    """
    run_db = _latest_run_db()
    if run_db is None:
        return None
    with closing(sqlite3.connect(
            f"file:{run_db.as_posix()}?mode=ro", uri=True)) as connection:
        columns = [d[1] for d in connection.execute(
            "PRAGMA table_info(standalone_oof_scores)")]
        if "oof_score_xgboost" not in columns:
            return None
        frame = pd.read_sql_query(
            "SELECT issuer_code, month, oof_score_xgboost FROM "
            "standalone_oof_scores", connection)
    frame = frame.rename(columns={"oof_score_xgboost": f"pd_{XGBOOST}"})
    # เดือนของสองแหล่งเก็บต่างรูปกัน 2009-09 กับ 2009-09-01 ต้องทำให้ตรงก่อน join
    frame["month_key"] = pd.to_datetime(frame["month"],
                                        errors="coerce").dt.strftime("%Y-%m")
    frame["issuer_code"] = frame["issuer_code"].astype(str)
    scores = pd.to_numeric(frame[f"pd_{XGBOOST}"], errors="coerce").to_numpy()
    n_flagged = max(1, int(round(workload * len(scores))))
    selected = np.argsort(-scores, kind="stable")[:n_flagged]
    threshold = float(scores[selected].min())
    frame[f"alarm_{XGBOOST}"] = (scores >= threshold).astype(int)
    return {"frame": frame[["issuer_code", "month_key", f"pd_{XGBOOST}",
                            f"alarm_{XGBOOST}"]],
            "threshold": threshold, "source": run_db, "n_flagged": n_flagged}


def attach_xgboost(scores: pd.DataFrame) -> pd.DataFrame:
    """ต่อคอลัมน์ PD ของ XGBoost เข้ากับตาราง PD ของสอง approach"""
    loaded = load_xgboost()
    if loaded is None:
        print("  ไม่พบผลรันเก้าวิธี จึงวาดแต่สอง approach", flush=True)
        return scores
    out = scores.copy()
    out["month_key"] = pd.to_datetime(out["month"],
                                      errors="coerce").dt.strftime("%Y-%m")
    out["issuer_code"] = out["issuer_code"].astype(str)
    merged = out.merge(loaded["frame"], on=["issuer_code", "month_key"],
                       how="left")
    covered = merged[f"pd_{XGBOOST}"].notna().mean()
    print(f"  ต่อเส้น XGBoost จาก {loaded['source'].parent.name} "
          f"ครอบคลุม {covered * 100:.1f}% ของแถว  เกณฑ์แจ้งเตือน "
          f"{loaded['threshold']:.4f}", flush=True)
    return merged.drop(columns="month_key")


# ------------------------------------------------------------- ค้นหาบริษัท
def resolve_issuer(query: str, out_db: Path = DEFAULT_OUT_DB) -> tuple[str, str]:
    """รับ issuer_code หรือชื่อบางส่วน คืนรหัสและชื่อ"""
    with closing(sqlite3.connect(Path(out_db))) as connection:
        names = pd.read_sql_query(
            "SELECT DISTINCT issuer_code, firm_name FROM pd3m_oof_scores "
            "ORDER BY issuer_code", connection)
    text = str(query).strip()
    exact = names[names["issuer_code"].astype(str).str.lower().eq(text.lower())]
    if len(exact) == 1:
        return str(exact.iloc[0]["issuer_code"]), str(exact.iloc[0]["firm_name"])
    hit = names[names["firm_name"].astype(str).str.contains(text, case=False, regex=False)
                | names["issuer_code"].astype(str).str.contains(text, case=False, regex=False)]
    if hit.empty:
        raise SystemExit(f"ไม่พบบริษัทที่ตรงกับ {query!r}  "
                         f"ลอง python pd_curves.py list --contains <คำ>")
    if len(hit) > 1:
        lines = [f"  {r.issuer_code:12s} {r.firm_name}" for r in hit.itertuples(index=False)]
        raise SystemExit(f"{query!r} ตรงกับ {len(hit)} บริษัท ระบุให้ชัดกว่านี้\n"
                         + "\n".join(lines[:40]))
    return str(hit.iloc[0]["issuer_code"]), str(hit.iloc[0]["firm_name"])


def list_issuers(contains: str = "", out_db: Path = DEFAULT_OUT_DB) -> pd.DataFrame:
    with closing(sqlite3.connect(Path(out_db))) as connection:
        names = pd.read_sql_query(
            "SELECT issuer_code, firm_name, COUNT(*) AS months, "
            "MAX(alarm_approach1) AS ever_alarm_a1, "
            "MAX(alarm_approach2) AS ever_alarm_a2 "
            "FROM pd3m_oof_scores GROUP BY issuer_code, firm_name "
            "ORDER BY issuer_code", connection)
    if contains:
        names = names[names["firm_name"].str.contains(contains, case=False, regex=False)
                      | names["issuer_code"].str.contains(contains, case=False, regex=False)]
    return names.reset_index(drop=True)


# ------------------------------------------------------------- พิมพ์สรุป
def format_metrics(metrics: pd.DataFrame) -> str:
    lines = ["PD summary", ""]
    show = metrics[["label", "auc_oof", "average_precision_oof",
                    "precision_at_workload", "recall_at_workload",
                    "pd_median", "pd_max", "rank_threshold"]]
    lines.append(show.to_string(index=False))
    return "\n".join(lines)


def format_detection(events: pd.DataFrame) -> str:
    lines = ["Detection of the event issuers, actionable window 1-3 months", ""]
    for approach in APPROACHES:
        hit = events[f"{approach}_detected"].astype(bool)
        days = pd.to_numeric(events.loc[hit, f"{approach}_lead_days"],
                             errors="coerce").dropna()
        lines.append(
            f"  {APPROACH_LABEL[approach]:38s} caught {int(hit.sum())}/{len(events)}"
            f"   lead median {days.median():.0f} days" if len(days) else
            f"  {APPROACH_LABEL[approach]:38s} caught {int(hit.sum())}/{len(events)}")
    both = (events["approach1_detected"].astype(bool)
            & events["approach2_detected"].astype(bool)).sum()
    neither = (~events["approach1_detected"].astype(bool)
               & ~events["approach2_detected"].astype(bool)).sum()
    lines += ["", f"  caught by both {int(both)}   caught by neither {int(neither)}"]
    return "\n".join(lines)


def format_issuer(rows: pd.DataFrame, issuer_code: str, firm_name: str,
                  thresholds: dict[str, float], event: pd.DataFrame) -> str:
    lines = [f"\n{firm_name} ({issuer_code})",
             f"  months {len(rows)}   "
             f"{rows['month'].min()} to {rows['month'].max()}"]
    for approach in APPROACHES:
        series = rows[f"pd_{approach}"]
        alarms = int(rows[f"alarm_{approach}"].sum())
        lines.append(
            f"  {APPROACH_LABEL[approach]:38s} PD median {series.median():.4f}   "
            f"max {series.max():.4f}   alarm months {alarms}   "
            f"threshold {thresholds[approach]:.4f}")
    if not event.empty:
        row = event.iloc[0]
        lines.append(f"  event {row['event_type']} in {str(row['event_month'])[:7]}")
        for approach in APPROACHES:
            hit = bool(row[f"{approach}_detected"])
            days = row[f"{approach}_lead_days"]
            when = row[f"{approach}_alarm_month"]
            lines.append(
                f"    {approach}: " + (
                    f"detected, alarm {str(when)[:7]}, lead {float(days):.0f} days"
                    if hit else "no alarm in the 1-3 month window"))
    return "\n".join(lines)


# ---------------------------------------------------------------- self-test
def self_test(out_db: Path = DEFAULT_OUT_DB) -> None:
    print("self-test: ตรวจว่ามีตารางครบและกราฟสร้างได้")
    if not Path(out_db).exists():
        raise SystemExit("ยังไม่มีฐานผล ให้รัน python pd_curves.py build ก่อน")
    for name in TABLES:
        frame = read_table(name, out_db)
        print(f"  {name:24s} {len(frame):>7,} แถว")
    events = read_table("pd3m_event_issuers", out_db)
    assert len(events) >= 1, "ไม่มีบริษัทที่เกิดเหตุการณ์"
    scores = read_scores(out_db, [str(events.iloc[0]["issuer_code"])])
    assert len(scores) > 0, "อ่าน PD รายเดือนไม่ได้"
    for approach in APPROACHES:
        series = scores[f"pd_{approach}"]
        assert series.between(0.0, 1.0).all(), f"PD ของ {approach} ออกนอกช่วง 0 ถึง 1"
    path = figure_issuer(str(events.iloc[0]["issuer_code"]), out_db)
    assert path.exists() and path.stat().st_size > 5_000, "ไฟล์กราฟเล็กผิดปกติ"
    print(f"  ทดลองวาดกราฟหนึ่งบริษัทได้ {path.name}")
    print("self-test ผ่าน")


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="กราฟ PD 3 เดือน เทียบ Approach 1 กับ Approach 2")
    parser.add_argument("--out-db", type=Path, default=DEFAULT_OUT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="คำนวณ PD แล้วเก็บลง sqlite")
    p_build.add_argument("--workload", type=float, default=WORKLOAD)

    p_events = sub.add_parser("events", help="กราฟ 32 บริษัทที่เกิด default")
    p_events.add_argument("--scale", choices=("pd", "rank"), default="pd")
    p_events.add_argument("--months", type=int, default=36,
                          help="ความยาวย้อนหลังของภาพแบบจัดเรียงเดือนเหตุการณ์")
    p_events.add_argument("--refit", action="store_true",
                          help="คำนวณ PD ใหม่ก่อนวาด")

    p_issuer = sub.add_parser("issuer", help="กราฟบริษัทใดก็ได้ ระบุชื่อหรือรหัส")
    p_issuer.add_argument("query")
    p_issuer.add_argument("--scale", choices=("pd", "rank"), default="pd")

    p_all = sub.add_parser("all", help="กราฟ PD ของทุกบริษัทในฐาน แบ่งเป็นหน้า")
    p_all.add_argument("--cols", type=int, default=4)
    p_all.add_argument("--rows", type=int, default=5,
                       help="จำนวนแถวต่อหน้า ช่องต่อหน้าคือ cols คูณ rows")
    p_all.add_argument("--scale", choices=("pd", "rank"), default="pd")
    p_all.add_argument("--no-xgboost", action="store_true",
                       help="วาดแต่สอง approach ไม่ต่อเส้น XGBoost")

    p_control = sub.add_parser("nondefault",
                               help="กลุ่มเทียบ บริษัทที่ไม่เคย default")
    p_control.add_argument("--count", type=int, default=10)
    p_control.add_argument("--scale", choices=("pd", "rank"), default="pd")

    p_list = sub.add_parser("list", help="รายชื่อบริษัท")
    p_list.add_argument("--contains", default="")

    sub.add_parser("tables", help="ดูตารางที่สร้างไว้")
    sub.add_parser("self-test", help="ตรวจว่าทำงานครบวง")

    args = parser.parse_args()

    if args.command == "build":
        build(args.out_db, args.workload)
        return
    if args.command == "events":
        if args.refit or not Path(args.out_db).exists():
            build(args.out_db)
        grid = figure_events_grid(args.out_db, args.scale)
        aligned = figure_events_aligned(args.out_db, args.months, args.scale)
        events = read_table("pd3m_event_issuers", args.out_db)
        print(format_detection(events))
        print(f"\nเขียนกราฟแล้ว\n  {grid}\n  {aligned}")
        return
    if args.command == "issuer":
        path = figure_issuer(args.query, args.out_db, args.scale)
        print(f"\nเขียนกราฟแล้ว\n  {path}")
        return
    if args.command == "all":
        paths = figure_all_issuers(args.out_db, args.cols, args.rows,
                                   args.scale, not args.no_xgboost)
        catalog = read_table("pd3m_issuer_catalog", args.out_db)
        print(f"\nวาดครบ {len(catalog):,} บริษัท เป็น {len(paths)} หน้า")
        print(f"  default {int(catalog['is_default'].astype(bool).sum())} ราย  "
              f"ไม่ default {int((~catalog['is_default'].astype(bool)).sum())} ราย")
        print(f"  ตาราง pd3m_issuer_catalog {len(catalog):,} แถว")
        return
    if args.command == "nondefault":
        path = figure_non_default(args.out_db, args.count, args.scale)
        control = read_table("pd3m_non_default_control", args.out_db)
        print(control.to_string(index=False))
        print(f"\nเขียนกราฟแล้ว\n  {path}")
        return
    if args.command == "list":
        frame = list_issuers(args.contains, args.out_db)
        print(frame.to_string(index=False))
        print(f"\n{len(frame):,} บริษัท")
        return
    if args.command == "tables":
        with closing(sqlite3.connect(Path(args.out_db))) as connection:
            for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name LIKE 'pd3m_%' ORDER BY name"):
                count = connection.execute(
                    f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                print(f"  {name:24s} {count:>8,} แถว")
        return
    if args.command == "self-test":
        self_test(args.out_db)
        return


if __name__ == "__main__":
    main()
