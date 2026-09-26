# -*- coding: utf-8 -*-
"""Standalone Dataset2 database and lead-time pipeline.

This program never writes to app.py, app_dataset2.py, data_adapter.py, or the
source database. It creates versioned Dataset2 snapshots and versioned result
folders, then compares linear, classical ensemble, and modern gradient-boosting
models with grouped OOF validation and the shared 1-3 month lead definition.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
import platform
import re
import shutil
import sqlite3
import sys
import time
import uuid
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

VERSION = "1.2.1"
RESULT_SCHEMA_VERSION = "dataset2_standalone_leadtime_v3"
LEAD_METRIC_VERSION = "lead_metrics_actionable_1_3m_persistent_v1"
ROOT = Path(__file__).resolve().parent
SOURCE_DB_NAMES = ("cmdf_credit.db", "lime_credit.db")


def _find_source_db() -> Path:
    """Dataset2 database beside this file, in the working folder, or up to two
    levels above this file.

    lime_credit.db.zip (the compact database shipped with iBond_LIME and
    leadning_time) is unpacked on first use when no database is found.
    """
    bases = list(dict.fromkeys(
        path.resolve() for path in (
            ROOT, Path.cwd(), ROOT.parent, Path.cwd().parent, ROOT.parent.parent,
        )
    ))
    for base in bases:
        for name in SOURCE_DB_NAMES:
            candidate = base / name
            if candidate.is_file() and candidate.stat().st_size > 1024:
                return candidate
    for base in bases:
        archive = base / "lime_credit.db.zip"
        if archive.is_file():
            print(f"Extracting {archive} ...")
            with zipfile.ZipFile(archive) as bundle:
                bundle.extract("lime_credit.db", base)
            return base / "lime_credit.db"
    return ROOT / SOURCE_DB_NAMES[0]


DEFAULT_SOURCE_DB = _find_source_db()
DEFAULT_TABLE = "ibond_33features_panel_941firm"
DEFAULT_WORKLOAD = 0.05
WORKLOAD_SWEEP_PCT = tuple(range(1, 11))
SEED = 42
MODEL_KEY_TO_NAME = {
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "random_forest": "RandomForest",
    "logistic": "Logistic",
    "probit": "Probit",
    "hist_gradient_boosting": "HistGradientBoosting",
    "gradient_boosting": "GradientBoosting",
    "extra_trees": "ExtraTrees",
}
# ค่าเริ่มต้นรันครบทุกวิธี ถ้าอยากรันเฉพาะบางตัวใช้ --models
DEFAULT_MODEL_KEYS = tuple(MODEL_KEY_TO_NAME)
DEFAULT_MODEL_NAMES = tuple(MODEL_KEY_TO_NAME[key] for key in DEFAULT_MODEL_KEYS)
MODEL_COLORS = {
    "XGBoost": "#d97706",
    "CatBoost": "#15803d",
    "LightGBM": "#2563eb",
    "RandomForest": "#dc2626",
    "Logistic": "#7c3aed",
    "Probit": "#0891b2",
    "HistGradientBoosting": "#475569",
    "GradientBoosting": "#be123c",
    "ExtraTrees": "#0f766e",
}

FEATURES = [
    "amihud_monthly", "adj_illiq_kz", "percent_zero_days", "zero_days", "n_days",
    "ROA", "ROE", "DE", "CurrentRatio", "QuickRatio", "CashRatio",
    "EBITtoTA", "REtoTA", "WorkingCapitaltoTA", "TDTA", "LTDtoTA", "STDtoTA",
    "cf_Interestcoverageratio", "acc_DebtServiceCoverageRatio",
    "lnTotalAssets", "lnAge", "Policyrate", "GDPgrowth",
    "UnemploymentratemodeledILOe", "ESGScore", "GovernancePillarScore",
    "EnvironmentalPillarScore", "SocialPillarScore", "IndependentBoardMembers",
    "AverageBoardTenure",
]
REQUIRED_META = [
    "issuer_code", "month", "y_pre3m", "ev_DP_this_month", "ev_RS_this_month",
]
OPTIONAL_META = [
    "name", "issuer_name", "issuer_name_th", "symbol", "market", "sector",
    "ibond_matched", "d_Default_Payment", "d_Restructure", "d_DP_RS",
    "default_month", "restructure_month", "date_DP", "date_RS",
    "event_date_available",
]
RAW_IBOND_TABLES = [
    "ibond_issuer", "ibond_corp_bond", "ibond_default_payment",
    "ibond_outstanding_summary", "ibond_bond_log", "bond_ews_universe",
]


def _now_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _workers() -> int:
    return max(1, min(4, os.cpu_count() or 4))


def _qident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return '"' + name + '"'


def _readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({_qident(table)})")]


def _check_required(columns: Iterable[str]) -> None:
    missing = sorted(set(REQUIRED_META + FEATURES) - set(columns))
    if missing:
        raise ValueError("Dataset2 is missing required columns: " + ", ".join(missing))


def inspect_database(db_path: Path, table: str = DEFAULT_TABLE) -> dict[str, Any]:
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    with closing(_readonly(db_path)) as connection:
        if table not in _table_names(connection):
            raise RuntimeError(f"Table {table!r} was not found in {db_path}")
        cols = _columns(connection, table)
        _check_required(cols)
        qtable = _qident(table)
        row = connection.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT issuer_code), MIN(month), MAX(month), "
            f"SUM(CASE WHEN COALESCE(y_pre3m,0)>0 THEN 1 ELSE 0 END), "
            f"COUNT(DISTINCT CASE WHEN COALESCE(y_pre3m,0)>0 THEN issuer_code END), "
            f"SUM(CASE WHEN COALESCE(ev_DP_this_month,0)>0 OR "
            f"COALESCE(ev_RS_this_month,0)>0 THEN 1 ELSE 0 END), "
            f"COUNT(DISTINCT CASE WHEN COALESCE(ev_DP_this_month,0)>0 OR "
            f"COALESCE(ev_RS_this_month,0)>0 THEN issuer_code END) FROM {qtable}"
        ).fetchone()
        duplicate_groups = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT issuer_code, month, COUNT(*) n FROM {qtable} "
            f"GROUP BY issuer_code, month HAVING n>1)"
        ).fetchone()[0]
    return {
        "database": str(db_path),
        "table": table,
        "bytes": db_path.stat().st_size,
        "rows": int(row[0]),
        "issuers": int(row[1]),
        "start_month": row[2],
        "end_month": row[3],
        "positive_months": int(row[4] or 0),
        "positive_issuers": int(row[5] or 0),
        "event_onset_rows": int(row[6] or 0),
        "event_onset_issuers": int(row[7] or 0),
        "columns": len(cols),
        "model_features": len(FEATURES),
        "duplicate_issuer_month_groups": int(duplicate_groups),
    }


def _print_inspection(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _default_snapshot_path() -> Path:
    return ROOT / "standalone_dataset2" / f"dataset2_941_snapshot_{_now_id()}.db"


def _copy_sqlite_table(source: Path, destination: Path, table: str) -> None:
    with closing(sqlite3.connect(destination)) as out:
        out.execute("ATTACH DATABASE ? AS source_db", (str(source.resolve()),))
        source_tables = {
            row[0]
            for row in out.execute(
                "SELECT name FROM source_db.sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if table not in source_tables:
            raise RuntimeError(f"Table {table!r} was not found in {source}")
        source_cols = [
            row[1]
            for row in out.execute(f"PRAGMA source_db.table_info({_qident(table)})")
        ]
        _check_required(source_cols)
        out.execute(
            f"CREATE TABLE {_qident(table)} AS SELECT * FROM source_db.{_qident(table)}"
        )
        for raw_table in RAW_IBOND_TABLES:
            if raw_table in source_tables:
                out.execute(
                    f"CREATE TABLE {_qident(raw_table)} AS "
                    f"SELECT * FROM source_db.{_qident(raw_table)}"
                )
        out.commit()
        out.execute("DETACH DATABASE source_db")


def _copy_tabular_file(source: Path, destination: Path, table: str) -> None:
    lower = source.name.lower()
    connection = sqlite3.connect(destination)
    try:
        first = True
        if lower.endswith(".csv") or lower.endswith(".csv.gz"):
            iterator = pd.read_csv(source, chunksize=20000, low_memory=False)
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            iterator = [pd.read_excel(source)]
        else:
            raise ValueError("Panel source must be SQLite, CSV/CSV.GZ, XLSX, or XLS")
        for chunk in iterator:
            if first:
                _check_required(chunk.columns)
            chunk.to_sql(table, connection, if_exists="replace" if first else "append", index=False)
            first = False
        if first:
            raise ValueError("Panel source was empty")
        connection.commit()
    finally:
        connection.close()


def build_snapshot(source: Path, output: Path, table: str = DEFAULT_TABLE) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}. Choose a new versioned --output path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".building-{uuid.uuid4().hex}")
    try:
        if source.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            _copy_sqlite_table(source, temporary, table)
        else:
            _copy_tabular_file(source, temporary, table)
        with closing(sqlite3.connect(temporary)) as connection:
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_standalone_issuer_month "
                f"ON {_qident(table)}(issuer_code, month)"
            )
            pd.DataFrame([{
                "schema_version": RESULT_SCHEMA_VERSION,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "source": str(source),
                "source_table": table,
                "operation": "build_snapshot",
                "immutable_source": 1,
            }]).to_sql("standalone_dataset_metadata", connection, if_exists="replace", index=False)
            connection.commit()
        summary = inspect_database(temporary, table)
        if summary["duplicate_issuer_month_groups"]:
            raise ValueError("Snapshot contains duplicate issuer-month rows")
        os.replace(temporary, output)
        summary["database"] = str(output)
        return summary
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _read_raw_ibond(source_db: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with closing(_readonly(source_db)) as connection:
        available = _table_names(connection)
        for table in RAW_IBOND_TABLES:
            if table in available:
                out[table] = pd.read_sql_query(f"SELECT * FROM {_qident(table)}", connection)
    return out


def _download_raw_ibond() -> dict[str, pd.DataFrame]:
    import download_bond
    issuers, bonds, defaults, summary = download_bond.run(
        with_defaults=True, save=False, verbose=True
    )
    return {
        "ibond_issuer": issuers,
        "ibond_corp_bond": bonds,
        "ibond_default_payment": defaults,
        "ibond_outstanding_summary": summary,
    }


def _map_default_issuers(
    defaults: pd.DataFrame,
    universe: pd.DataFrame,
    panel_codes: set[str],
) -> pd.DataFrame:
    d = defaults.copy()
    if d.empty or "symbol" not in d.columns or "payment_date" not in d.columns:
        return pd.DataFrame(columns=["issuer_code", "payment_date", "symbol", "mapping"])
    symbol_map: dict[str, str] = {}
    if not universe.empty and {"symbol", "issuer_code"}.issubset(universe.columns):
        clean = universe[["symbol", "issuer_code"]].dropna().drop_duplicates("symbol")
        symbol_map = dict(zip(clean["symbol"].astype(str), clean["issuer_code"].astype(str)))
    rows = []
    for row in d.itertuples(index=False):
        symbol = str(getattr(row, "symbol", "")).strip().upper()
        event_date = pd.to_datetime(getattr(row, "payment_date", None), errors="coerce")
        if not symbol or pd.isna(event_date):
            continue
        issuer = str(symbol_map.get(symbol, "")).strip().upper()
        mapping = "bond_ews_universe"
        if not issuer:
            match = re.match(r"^([A-Z]+)", symbol)
            prefix = match.group(1) if match else ""
            issuer = prefix if prefix in panel_codes else ""
            mapping = "exact_symbol_prefix" if issuer else "unmatched_symbol_prefix"
        rows.append({
            "issuer_code": issuer,
            "payment_date": pd.Timestamp(event_date),
            "symbol": symbol,
            "mapping": mapping if issuer else "unmatched",
        })
    mapped = pd.DataFrame(rows)
    if mapped.empty:
        return mapped
    return (
        mapped.sort_values(["issuer_code", "payment_date", "symbol"])
        .groupby("issuer_code", as_index=False, dropna=False)
        .first()
    )


def update_ibond_snapshot(
    base_db: Path,
    output_db: Path,
    *,
    raw_source_db: Path | None = None,
    download: bool = False,
    table: str = DEFAULT_TABLE,
) -> dict[str, Any]:
    base_db = base_db.resolve()
    output_db = output_db.resolve()
    if output_db.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output_db}. Choose a new versioned --output path."
        )
    if not base_db.is_file():
        raise FileNotFoundError(base_db)
    if download == (raw_source_db is not None):
        raise ValueError("Choose exactly one of --download or --from-db")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_db.with_name(output_db.name + f".updating-{uuid.uuid4().hex}")
    shutil.copy2(base_db, temporary)
    try:
        raw = _download_raw_ibond() if download else _read_raw_ibond(raw_source_db.resolve())
        defaults = raw.get("ibond_default_payment", pd.DataFrame())
        if defaults.empty:
            raise RuntimeError("No payment-default rows were available from iBond")
        with closing(sqlite3.connect(temporary)) as connection:
            available = _table_names(connection)
            if table not in available:
                raise RuntimeError(f"Table {table!r} was not found in snapshot")
            cols = set(_columns(connection, table))
            needed_update = {
                "d_Default_Payment", "d_Restructure", "d_DP_RS", "default_month",
                "restructure_month", "date_DP", "ev_DP_this_month", "ev_RS_this_month",
                "y_pre3m",
            }
            missing_update = sorted(needed_update - cols)
            if missing_update:
                raise ValueError("Snapshot cannot receive iBond event update; missing: " + ", ".join(missing_update))
            panel_codes = {
                str(row[0]).strip().upper()
                for row in connection.execute(
                    f"SELECT DISTINCT issuer_code FROM {_qident(table)}"
                ).fetchall()
            }
            universe = raw.get("bond_ews_universe", pd.DataFrame())
            if universe.empty and "bond_ews_universe" in available:
                universe = pd.read_sql_query("SELECT * FROM bond_ews_universe", connection)
            mapped = _map_default_issuers(defaults, universe, panel_codes)
            for raw_table, frame in raw.items():
                if frame is not None and not frame.empty:
                    frame.to_sql(raw_table, connection, if_exists="replace", index=False)
            applied = []
            unmatched = []
            qtable = _qident(table)
            for event in mapped.itertuples(index=False):
                code = str(event.issuer_code).strip().upper()
                if not code or code not in panel_codes:
                    unmatched.append({
                        "symbol": event.symbol, "issuer_code": code,
                        "payment_date": str(pd.Timestamp(event.payment_date).date()),
                        "reason": "issuer_not_in_panel",
                    })
                    continue
                event_date = pd.Timestamp(event.payment_date)
                event_month = event_date.strftime("%Y-%m")
                event_date_text = event_date.date().isoformat()
                connection.execute(
                    f"UPDATE {qtable} SET default_month = CASE "
                    f"WHEN default_month IS NULL OR default_month='' OR substr(default_month,1,7)>? "
                    f"THEN ? ELSE default_month END, date_DP = CASE "
                    f"WHEN date_DP IS NULL OR date_DP='' OR substr(date_DP,1,10)>? "
                    f"THEN ? ELSE date_DP END WHERE upper(trim(issuer_code))=?",
                    (event_month, event_month + "-01", event_date_text, event_date_text, code),
                )
                connection.execute(
                    f"UPDATE {qtable} SET d_Default_Payment=1 "
                    f"WHERE upper(trim(issuer_code))=? AND substr(month,1,7)>=?",
                    (code, event_month),
                )
                onset_rows = connection.execute(
                    f"UPDATE {qtable} SET ev_DP_this_month=1 "
                    f"WHERE upper(trim(issuer_code))=? AND substr(month,1,7)=?",
                    (code, event_month),
                ).rowcount
                applied.append({
                    "issuer_code": code, "symbol": event.symbol,
                    "payment_date": event_date_text, "event_month": event_month,
                    "mapping": event.mapping, "onset_row_present": int(onset_rows > 0),
                })
            connection.execute(
                f"UPDATE {qtable} SET d_DP_RS = CASE WHEN "
                f"COALESCE(d_Default_Payment,0)>0 OR COALESCE(d_Restructure,0)>0 "
                f"THEN 1 ELSE 0 END"
            )
            event_dates = pd.read_sql_query(
                f"SELECT issuer_code, MIN(CASE WHEN default_month IS NOT NULL AND "
                f"default_month<>'' THEN substr(default_month,1,7) END) AS dp_month, "
                f"MIN(CASE WHEN restructure_month IS NOT NULL AND restructure_month<>'' "
                f"THEN substr(restructure_month,1,7) END) AS rs_month "
                f"FROM {qtable} GROUP BY issuer_code",
                connection,
            )
            for event in event_dates.itertuples(index=False):
                candidates = [value for value in (event.dp_month, event.rs_month) if isinstance(value, str) and value]
                if not candidates:
                    continue
                first_month = min(candidates)
                year, month = map(int, first_month.split("-"))
                event_ordinal = year * 12 + month
                code = str(event.issuer_code)
                connection.execute(
                    f"UPDATE {qtable} SET y_pre3m=0 WHERE issuer_code=?", (code,)
                )
                connection.execute(
                    f"UPDATE {qtable} SET y_pre3m=1 WHERE issuer_code=? AND "
                    f"(? - (CAST(substr(month,1,4) AS INTEGER)*12 + "
                    f"CAST(substr(month,6,2) AS INTEGER))) BETWEEN 0 AND 3",
                    (code, event_ordinal),
                )
            update_log = pd.DataFrame(applied + unmatched)
            update_log["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            update_log["source"] = "live_ibond_no_save" if download else str(raw_source_db.resolve())
            update_log.to_sql("standalone_ibond_event_update", connection, if_exists="replace", index=False)
            pd.DataFrame([{
                "schema_version": RESULT_SCHEMA_VERSION,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "source": str(base_db),
                "operation": "update_ibond_snapshot",
                "raw_ibond_source": "live_no_save" if download else str(raw_source_db.resolve()),
                "mapped_defaults": len(applied),
                "unmatched_defaults": len(unmatched),
            }]).to_sql("standalone_update_metadata", connection, if_exists="replace", index=False)
            connection.commit()
        summary = inspect_database(temporary, table)
        os.replace(temporary, output_db)
        summary.update({
            "database": str(output_db),
            "mapped_default_issuers": len(applied),
            "unmatched_default_issuers": len(unmatched),
        })
        return summary
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

def _to_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _firm_name(group: pd.DataFrame) -> str:
    for column in ("issuer_name", "name", "issuer_name_th", "symbol"):
        if column in group.columns:
            values = group[column].dropna().astype(str).str.strip()
            values = values[values.ne("") & values.ne("nan")]
            if not values.empty:
                return values.iloc[-1]
    return str(group["issuer_code"].iloc[0])


def _event_catalog(panel: pd.DataFrame, explicit_only: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for issuer, group in panel.groupby("issuer_code", sort=True):
        g = group.sort_values("month_dt", kind="stable")
        dp_onset = _to_number(g.get("ev_DP_this_month", pd.Series(0, index=g.index))).fillna(0).gt(0)
        rs_onset = _to_number(g.get("ev_RS_this_month", pd.Series(0, index=g.index))).fillna(0).gt(0)
        onset = dp_onset | rs_onset
        status_present = False
        for status_column in ("d_DP_RS", "d_Default_Payment", "d_Restructure"):
            if status_column in g.columns and _to_number(g[status_column]).fillna(0).gt(0).any():
                status_present = True
                break
        if explicit_only and not onset.any():
            continue
        if not explicit_only and not (onset.any() or status_present):
            continue
        candidates: list[tuple[pd.Timestamp, str, str]] = []
        if onset.any():
            first_onset = pd.Timestamp(g.loc[onset, "month_dt"].min()).to_period("M").to_timestamp()
            at_first = g["month_dt"].dt.to_period("M").eq(first_onset.to_period("M"))
            if (dp_onset & at_first).any():
                candidates.append((first_onset, "DP", "explicit_onset"))
            if (rs_onset & at_first).any():
                candidates.append((first_onset, "RS", "explicit_onset"))
        if not explicit_only:
            for column, kind in (
                ("default_month", "DP"), ("date_DP", "DP"),
                ("restructure_month", "RS"), ("date_RS", "RS"),
            ):
                if column not in g.columns:
                    continue
                dates = pd.to_datetime(g[column], errors="coerce").dropna()
                if not dates.empty:
                    candidates.append((pd.Timestamp(dates.min()).to_period("M").to_timestamp(), kind, column))
            for column, kind in (("d_Default_Payment", "DP"), ("d_Restructure", "RS")):
                if column in g.columns:
                    status = _to_number(g[column]).fillna(0).gt(0)
                    if status.any():
                        candidates.append((pd.Timestamp(g.loc[status, "month_dt"].min()).to_period("M").to_timestamp(), kind, column))
        if not candidates:
            continue
        event_month = min(item[0] for item in candidates)
        first = [item for item in candidates if item[0] == event_month]
        kinds = sorted({item[1] for item in first})
        sources = sorted({item[2] for item in first})
        records.append({
            "issuer_code": str(issuer),
            "firm_name": _firm_name(g),
            "event_month": event_month,
            "event_type": "+".join(kinds),
            "event_source": "+".join(sources),
            "first_panel_month": pd.Timestamp(g["month_dt"].min()),
            "evaluable": bool(pd.Timestamp(g["month_dt"].min()) < event_month),
        })
    if not records:
        return pd.DataFrame(columns=[
            "issuer_code", "firm_name", "event_month", "event_type",
            "event_source", "first_panel_month", "evaluable",
        ])
    return pd.DataFrame(records).sort_values(["event_month", "issuer_code"]).reset_index(drop=True)


def load_model_data(db_path: Path, table: str = DEFAULT_TABLE) -> dict[str, Any]:
    db_path = db_path.resolve()
    with closing(_readonly(db_path)) as connection:
        if table not in _table_names(connection):
            raise RuntimeError(f"Table {table!r} was not found in {db_path}")
        available = _columns(connection, table)
        _check_required(available)
        selected = list(dict.fromkeys(REQUIRED_META + [c for c in OPTIONAL_META if c in available] + FEATURES))
        frame = pd.read_sql_query(
            "SELECT " + ", ".join(_qident(column) for column in selected)
            + " FROM " + _qident(table),
            connection,
        )
    frame["issuer_code"] = frame["issuer_code"].astype(str).str.strip()
    frame["month_dt"] = pd.to_datetime(frame["month"], errors="coerce")
    if frame["issuer_code"].eq("").any() or frame["month_dt"].isna().any():
        raise ValueError("Blank issuer_code or invalid month found")
    duplicate = frame.duplicated(["issuer_code", "month_dt"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate issuer-month rows found:\n" + frame.loc[duplicate, ["issuer_code", "month"]].head().to_string(index=False))
    x_frame = frame[FEATURES].copy()
    for column in FEATURES:
        x_frame[column] = pd.to_numeric(x_frame[column], errors="coerce")
        median = x_frame[column].median()
        x_frame[column] = x_frame[column].fillna(0.0 if pd.isna(median) else median)
    target = pd.to_numeric(frame["y_pre3m"], errors="coerce").fillna(0).gt(0).astype(int)
    frame["_source_order"] = np.arange(len(frame), dtype=int)
    order = frame.sort_values(["issuer_code", "month_dt", "_source_order"], kind="stable").index.to_numpy()
    panel = frame.iloc[order].reset_index(drop=True)
    x_frame = x_frame.iloc[order].reset_index(drop=True)
    y = target.iloc[order].to_numpy(dtype=int)
    latest_rows = (
        panel.assign(_row=np.arange(len(panel), dtype=int))
        .groupby("issuer_code", sort=False)
        .tail(1)["_row"]
        .to_numpy(dtype=int)
    )
    return {
        "db_path": db_path,
        "table": table,
        "panel": panel,
        "X": x_frame,
        "y": y,
        "features": FEATURES.copy(),
        "latest_rows": latest_rows,
        "app_events": _event_catalog(panel, explicit_only=True),
        "all_events": _event_catalog(panel, explicit_only=False),
    }


class WeightedProbitClassifier:
    """L2-regularized weighted probit classifier with fold-local scaling."""

    def __init__(self, positive_weight: float, l2: float = 1e-3, max_iter: int = 120):
        self.positive_weight = float(positive_weight)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.params_: np.ndarray | None = None
        self.optimizer_success_: bool | None = None

    def fit(self, values: np.ndarray, target: np.ndarray) -> "WeightedProbitClassifier":
        values = np.asarray(values, dtype=float)
        target = np.asarray(target, dtype=int)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-12)] = 1.0
        scaled = (values - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(scaled), dtype=float), scaled])
        sample_weight = np.where(target == 1, self.positive_weight, 1.0)
        weight_sum = float(sample_weight.sum())
        log_sqrt_two_pi = 0.5 * np.log(2.0 * np.pi)

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            linear = design @ params
            log_pdf = -0.5 * linear * linear - log_sqrt_two_pi
            log_positive = log_ndtr(linear)
            log_negative = log_ndtr(-linear)
            log_likelihood = target * log_positive + (1 - target) * log_negative
            loss = -float(np.dot(sample_weight, log_likelihood)) / weight_sum
            loss += 0.5 * self.l2 * float(np.dot(params[1:], params[1:]))
            positive_ratio = np.exp(np.clip(log_pdf - log_positive, -50.0, 50.0))
            negative_ratio = np.exp(np.clip(log_pdf - log_negative, -50.0, 50.0))
            gradient_linear = sample_weight * np.where(
                target == 1, -positive_ratio, negative_ratio
            ) / weight_sum
            gradient = design.T @ gradient_linear
            gradient[1:] += self.l2 * params[1:]
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1], dtype=float),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-10, "maxls": 40},
        )
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise RuntimeError("Weighted probit optimization produced non-finite parameters")
        if not result.success:
            raise RuntimeError(f"Weighted probit did not converge: {result.message}")
        self.params_ = result.x
        self.optimizer_success_ = True
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.params_ is None:
            raise RuntimeError("WeightedProbitClassifier must be fitted before prediction")
        scaled = (np.asarray(values, dtype=float) - self.mean_) / self.scale_
        linear = self.params_[0] + scaled @ self.params_[1:]
        probability = np.clip(ndtr(linear), 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - probability, probability])


def _make_model(model_name: str, positive_weight: float) -> Any:
    if model_name == "XGBoost":
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
            n_jobs=_workers(),
            random_state=SEED,
            verbosity=0,
        )
    if model_name == "CatBoost":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost is not installed. Install catboost for Python 3.12.")
        return CatBoostClassifier(
            iterations=250,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=4.0,
            auto_class_weights="Balanced",
            random_seed=SEED,
            thread_count=_workers(),
            verbose=0,
            allow_writing_files=False,
        )
    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is not installed. Install lightgbm for Python 3.12.")
        return LGBMClassifier(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            min_child_samples=20,
            reg_lambda=3.0,
            scale_pos_weight=positive_weight,
            objective="binary",
            n_jobs=_workers(),
            random_state=SEED,
            verbosity=-1,
        )
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=_workers(),
            random_state=SEED,
        )
    if model_name == "Logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=SEED,
            ),
        )
    if model_name == "Probit":
        return WeightedProbitClassifier(positive_weight=positive_weight)

    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=3.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            random_state=SEED,
        )
    if model_name == "GradientBoosting":
        # this estimator has no class_weight, so the rare-event weight is
        # passed per row at fit time instead, see _fit_model below
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            random_state=SEED,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=_workers(),
            random_state=SEED,
        )
    raise ValueError(f"Unsupported model: {model_name}")


# estimators that carry no class-weight setting take a per-row weight at fit
SAMPLE_WEIGHT_MODELS = {"GradientBoosting"}


def _fit_model(model: Any, model_name: str, values: np.ndarray,
               target: np.ndarray, positive_weight: float) -> Any:
    """Fit, giving the rare-event weight to the estimators that need one.

    Without this a model with no class_weight setting would learn almost
    nothing, because 124 of 186,255 months carry the event.
    """
    if model_name not in SAMPLE_WEIGHT_MODELS:
        model.fit(values, target)
        return model
    weights = np.where(np.asarray(target) > 0, float(positive_weight), 1.0)
    model.fit(values, target, sample_weight=weights)
    return model


def fit_oof(data: dict[str, Any], model_name: str, workload: float) -> dict[str, Any]:
    values = data["X"].to_numpy(dtype=float)
    y = data["y"]
    groups = data["panel"]["issuer_code"].to_numpy(dtype=str)
    event_groups = int(pd.Series(groups[y == 1]).nunique())
    n_splits = min(5, event_groups)
    if n_splits < 2:
        raise RuntimeError("Too few positive issuers for grouped validation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_of = np.full(len(y), -1, dtype=int)
    started = time.perf_counter()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(values, y, groups), start=1):
        positives = int(y[train_idx].sum())
        if positives == 0:
            raise RuntimeError(f"Fold {fold} has no positive training row")
        print(f"  {model_name}: fold {fold}/{n_splits} ({len(train_idx):,} train, {len(test_idx):,} held out)", flush=True)
        weight = float((len(train_idx) - positives) / positives)
        model = _make_model(model_name, weight)
        _fit_model(model, model_name, values[train_idx], y[train_idx], weight)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            )
            oof[test_idx] = model.predict_proba(values[test_idx])[:, 1]
        fold_of[test_idx] = fold - 1
    if not np.isfinite(oof).all() or (fold_of < 0).any():
        raise RuntimeError("OOF scoring did not cover all rows")
    curve_precision, curve_recall, curve_thresholds = precision_recall_curve(y, oof)
    threshold = float(np.quantile(oof[data["latest_rows"]], 1.0 - workload))
    n_flagged = max(1, int(round(workload * len(oof))))
    ranked_alarm = np.zeros(len(oof), dtype=int)
    ranked_alarm[np.argsort(oof)[::-1][:n_flagged]] = 1
    metrics = {
        "model": model_name,
        "program_version": VERSION,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "auc_oof": float(roc_auc_score(y, oof)),
        "average_precision_oof": float(average_precision_score(y, oof)),
        "pr_auc_trapezoid_oof": float(auc(curve_recall, curve_precision)),
        "precision_at_workload": float(precision_score(y, ranked_alarm, zero_division=0)),
        "recall_at_workload": float(recall_score(y, ranked_alarm, zero_division=0)),
        "f1_at_workload": float(f1_score(y, ranked_alarm, zero_division=0)),
        "threshold_latest_cross_section": threshold,
        "workload": workload,
        "n_rows": int(len(y)),
        "n_issuers": int(data["panel"]["issuer_code"].nunique()),
        "n_features": len(FEATURES),
        "n_positive_rows": int(y.sum()),
        "n_positive_issuers": event_groups,
        "n_splits": n_splits,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return {
        "model": model_name,
        "oof": oof,
        "fold_of": fold_of,
        "ranked_alarm": ranked_alarm,
        "history_alarm": oof >= threshold,
        "pr_curve": {
            "precision": curve_precision,
            "recall": curve_recall,
            "thresholds": curve_thresholds,
        },
        "metrics": metrics,
    }


def precision_recall_workload_sweep(
    y: np.ndarray,
    scores: np.ndarray,
    latest_rows: np.ndarray,
    model_name: str,
    workload_percentages: Iterable[int] = WORKLOAD_SWEEP_PCT,
) -> pd.DataFrame:
    """Evaluate exact top-risk operating points from 1% through 10% workload."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    latest_rows = np.asarray(latest_rows, dtype=int)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("y and scores must be non-empty arrays of equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")

    prevalence = float(y.mean())
    ranked_indices = np.argsort(scores, kind="stable")[::-1]
    records: list[dict[str, Any]] = []
    for workload_pct in workload_percentages:
        workload_pct = int(workload_pct)
        if not 1 <= workload_pct <= 100:
            raise ValueError("workload percentages must be between 1 and 100")
        workload = workload_pct / 100.0
        n_flagged = max(1, int(round(workload * len(scores))))
        alarm = np.zeros(len(scores), dtype=int)
        alarm[ranked_indices[:n_flagged]] = 1
        true_positive = int(np.sum((alarm == 1) & (y == 1)))
        false_positive = int(np.sum((alarm == 1) & (y == 0)))
        false_negative = int(np.sum((alarm == 0) & (y == 1)))
        precision = float(precision_score(y, alarm, zero_division=0))
        recall = float(recall_score(y, alarm, zero_division=0))
        records.append({
            "model": model_name,
            "review_workload_pct": workload_pct,
            "actual_review_workload_pct": float(100.0 * n_flagged / len(scores)),
            "n_flagged_months": n_flagged,
            "rank_score_threshold": float(scores[ranked_indices[n_flagged - 1]]),
            "latest_cross_section_threshold": float(
                np.quantile(scores[latest_rows], 1.0 - workload)
            ),
            "true_positive_months": true_positive,
            "false_positive_months": false_positive,
            "false_negative_months": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": float(f1_score(y, alarm, zero_division=0)),
            "positive_prevalence": prevalence,
            "precision_lift_over_prevalence": (
                float(precision / prevalence) if prevalence > 0 else np.nan
            ),
        })
    return pd.DataFrame(records)


def _month_ordinal(value: pd.Timestamp) -> int:
    return int(value.year * 12 + value.month)


def compute_lead_for_issuer(rows: pd.DataFrame, event_month: pd.Timestamp) -> dict[str, Any]:
    event_month = pd.Timestamp(event_month).to_period("M").to_timestamp()
    base = {
        "actionable_alarm_month": None,
        "actionable_lead_days": np.nan,
        "actionable_lead_months": np.nan,
        "persistent_alarm_start": None,
        "persistent_alarm_end": None,
        "persistent_alarm_days": np.nan,
        "persistent_alarm_months": np.nan,
        "actionable_alarm_found": False,
        "lead_metric_version": LEAD_METRIC_VERSION,
    }
    if rows.empty:
        return base
    d = rows.copy().sort_values("month_dt", kind="stable")
    d = d.loc[d["month_dt"].lt(event_month)].copy()
    if d.empty:
        return base
    d["alarm"] = d["alarm"].fillna(False).astype(bool)
    alarms = d[d["alarm"]]
    if not alarms.empty:
        start = event_month - pd.DateOffset(months=3)
        end = event_month - pd.DateOffset(months=1)
        actionable = alarms[alarms["month_dt"].between(start, end, inclusive="both")]
        if not actionable.empty:
            first = actionable.iloc[0]
            alarm_month = pd.Timestamp(first["month_dt"])
            days = float((event_month - alarm_month).days)
            base.update({
                "actionable_alarm_month": alarm_month.date().isoformat(),
                "actionable_lead_days": days,
                "actionable_lead_months": float(_month_ordinal(event_month) - _month_ordinal(alarm_month)),
                "actionable_alarm_found": True,
            })
        compact = d.groupby(d["month_dt"].dt.to_period("M"), sort=True, as_index=False).last()
        alarm_positions = np.flatnonzero(compact["alarm"].to_numpy(dtype=bool))
        if len(alarm_positions):
            end_pos = int(alarm_positions[-1])
            start_pos = end_pos
            while start_pos > 0:
                current = pd.Timestamp(compact.iloc[start_pos]["month_dt"])
                previous = pd.Timestamp(compact.iloc[start_pos - 1]["month_dt"])
                if not bool(compact.iloc[start_pos - 1]["alarm"]):
                    break
                if _month_ordinal(current) - _month_ordinal(previous) != 1:
                    break
                start_pos -= 1
            first = pd.Timestamp(compact.iloc[start_pos]["month_dt"])
            last = pd.Timestamp(compact.iloc[end_pos]["month_dt"])
            days = float((event_month - first).days)
            base.update({
                "persistent_alarm_start": first.date().isoformat(),
                "persistent_alarm_end": last.date().isoformat(),
                "persistent_alarm_days": days,
                "persistent_alarm_months": float(days / 30.4375),
            })
    return base


def lead_table(data: dict[str, Any], run: dict[str, Any], catalog: pd.DataFrame) -> pd.DataFrame:
    scored = data["panel"].copy()
    scored["alarm"] = run["history_alarm"]
    records = []
    for event in catalog.itertuples(index=False):
        rows = scored.loc[scored["issuer_code"].eq(event.issuer_code)].copy()
        pre_count = int(rows["month_dt"].lt(event.event_month).sum())
        lead = compute_lead_for_issuer(rows, event.event_month)
        if pre_count == 0:
            status = "no_pre_event_data"
        elif lead["actionable_alarm_found"]:
            status = "detected"
        elif lead["persistent_alarm_start"]:
            status = "earlier_only"
        else:
            status = "missed"
        records.append({
            "model": run["model"],
            "issuer_code": event.issuer_code,
            "firm_name": event.firm_name,
            "event_month": pd.Timestamp(event.event_month).strftime("%Y-%m"),
            "event_type": event.event_type,
            "event_source": event.event_source,
            "first_panel_month": pd.Timestamp(event.first_panel_month).strftime("%Y-%m"),
            "evaluable": bool(pre_count > 0),
            "n_pre_event_months": pre_count,
            "status": status,
            **lead,
        })
    return pd.DataFrame(records)


def _lead_summary(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    evaluable = frame.loc[frame["evaluable"]].copy()
    detected = evaluable.loc[evaluable["status"].eq("detected")]
    leads = pd.to_numeric(detected["actionable_lead_days"], errors="coerce").dropna()
    persistent = pd.to_numeric(evaluable["persistent_alarm_days"], errors="coerce").dropna()
    return {
        f"{prefix}_event_catalog": int(len(frame)),
        f"{prefix}_evaluable_events": int(len(evaluable)),
        f"{prefix}_detected": int(len(detected)),
        f"{prefix}_detection_rate": float(len(detected) / max(len(evaluable), 1)),
        f"{prefix}_lead_min_days": float(leads.min()) if len(leads) else np.nan,
        f"{prefix}_lead_median_days": float(leads.median()) if len(leads) else np.nan,
        f"{prefix}_lead_max_days": float(leads.max()) if len(leads) else np.nan,
        f"{prefix}_persistent_median_days": float(persistent.median()) if len(persistent) else np.nan,
    }


def _plot_summary(summary: pd.DataFrame, output: Path) -> None:
    panels = [
        ("auc_oof", "OOF ROC-AUC", (0, 1)),
        ("average_precision_oof", "OOF average precision", (0, None)),
        ("recall_at_workload", "Positive-month recall at 5%", (0, 1)),
        ("app_detection_rate", "Actionable event detection (1-3M)", (0, 1)),
    ]
    present_models = set(summary["model"])
    model_order = [name for name in DEFAULT_MODEL_NAMES if name in present_models]
    model_order.extend(name for name in summary["model"] if name not in model_order)
    ordered = summary.set_index("model").loc[model_order]
    fig_height = max(7.4, 2.4 + 0.58 * len(model_order))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, fig_height), constrained_layout=True)
    for ax, (column, title, limits) in zip(axes.ravel(), panels):
        values = ordered[column]
        positions = np.arange(len(values))
        bars = ax.barh(
            positions,
            values.to_numpy(dtype=float),
            color=[MODEL_COLORS.get(name, "#64748b") for name in values.index],
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values.values],
            fontsize=7,
            padding=3,
        )
        ax.set_yticks(positions, labels=values.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, weight="bold")
        upper = limits[1] if limits[1] is not None else max(1e-6, float(values.max()) * 1.25)
        ax.set_xlim(limits[0], upper)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Standalone Dataset2 multi-model grouped OOF comparison", weight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_precision_recall(
    runs: dict[str, dict[str, Any]],
    sweep: pd.DataFrame,
    prevalence: float,
    output: Path,
) -> None:
    colors = MODEL_COLORS
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), constrained_layout=True)

    for model_name, run in runs.items():
        color = colors.get(model_name, "#2563eb")
        precision = np.asarray(run["pr_curve"]["precision"], dtype=float)
        recall = np.asarray(run["pr_curve"]["recall"], dtype=float)
        if len(precision) > 5000:
            sample = np.unique(np.linspace(0, len(precision) - 1, 5000).astype(int))
            precision = precision[sample]
            recall = recall[sample]
        axes[0].plot(
            recall * 100.0,
            precision * 100.0,
            color=color,
            linewidth=1.5,
            label=(
                f"{model_name}: AP={run['metrics']['average_precision_oof']:.4f}, "
                f"PR-AUC={run['metrics']['pr_auc_trapezoid_oof']:.4f}"
            ),
        )
        points = sweep.loc[sweep["model"].eq(model_name)].sort_values("review_workload_pct")
        axes[0].scatter(
            points["recall"] * 100.0,
            points["precision"] * 100.0,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=25,
            zorder=3,
        )

        axes[1].plot(
            points["review_workload_pct"],
            points["precision"] * 100.0,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=color,
            label=model_name,
        )
        axes[2].plot(
            points["review_workload_pct"],
            points["recall"] * 100.0,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=color,
            label=model_name,
        )

    axes[0].axhline(
        prevalence * 100.0,
        color="#64748b",
        linestyle="--",
        linewidth=1.2,
        label=f"Random baseline={prevalence * 100.0:.4f}%",
    )
    axes[0].set_title("Full out-of-fold precision-recall curve", weight="bold")
    axes[0].set_xlabel("Recall (%)")
    axes[0].set_ylabel("Precision (%)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=5.5, loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].axhline(prevalence * 100.0, color="#64748b", linestyle="--", linewidth=1.2)
    axes[1].set_title("Precision by review workload", weight="bold")
    axes[1].set_xlabel("Top-risk issuer-months reviewed (%)")
    axes[1].set_ylabel("Precision (%)")
    axes[1].set_xticks(WORKLOAD_SWEEP_PCT)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Recall by review workload", weight="bold")
    axes[2].set_xlabel("Top-risk issuer-months reviewed (%)")
    axes[2].set_ylabel("Recall (%)")
    axes[2].set_xticks(WORKLOAD_SWEEP_PCT)
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=6.5, ncol=2)
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        "Dataset2 OOF PR-AUC and operating thresholds (1%-10% review workload)",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _lead_matrix_table(lead: pd.DataFrame) -> pd.DataFrame:
    """หนึ่งแถวต่อหนึ่งบริษัท คอลัมน์เป็นวิธี ค่าคือ actionable lead time หน่วยวัน

    ตารางนี้อ่านง่ายกว่ารูปแบบยาว เพราะเทียบวิธีข้างกันได้ในบรรทัดเดียว
    """
    key = ["issuer_code", "firm_name", "event_type", "event_month"]
    # รายชื่อเหตุการณ์ทั้งหมด ตั้งเป็นโครงไว้ก่อน บริษัทที่ไม่มีวิธีไหนเตือนเลย
    # จะได้ไม่หายไปตอน pivot ซึ่งทิ้งแถวที่ว่างทุกช่อง
    events = lead[key].drop_duplicates().reset_index(drop=True)
    wide = lead.pivot_table(
        index=key, columns="model",
        values="actionable_lead_days", aggfunc="first").reset_index()
    status = lead.pivot_table(
        index=key, columns="model",
        values="status", aggfunc="first").reset_index()
    status = status.rename(columns={
        c: f"{c}__status" for c in status.columns if c not in key})
    merged = events.merge(wide, on=key, how="left").merge(
        status, on=key, how="left")
    merged.columns = [str(c) for c in merged.columns]
    return merged.sort_values("event_month").reset_index(drop=True)


def _lead_model_summary_table(lead: pd.DataFrame) -> pd.DataFrame:
    """หนึ่งแถวต่อหนึ่งวิธี ตรวจได้กี่ราย จากกี่รายที่วัดได้ และ lead กี่วัน"""
    records = []
    for model_name, group in lead.groupby("model", sort=True):
        evaluable = group.loc[group["evaluable"]]
        detected = evaluable.loc[evaluable["status"].eq("detected")]
        days = pd.to_numeric(detected["actionable_lead_days"], errors="coerce").dropna()
        persist = pd.to_numeric(evaluable["persistent_alarm_days"], errors="coerce").dropna()
        records.append({
            "model": model_name,
            "event_catalog": int(len(group)),
            "evaluable_events": int(len(evaluable)),
            "detected": int(len(detected)),
            "earlier_only": int((evaluable["status"] == "earlier_only").sum()),
            "missed": int((evaluable["status"] == "missed").sum()),
            "detection_rate": float(len(detected) / max(len(evaluable), 1)),
            "lead_min_days": float(days.min()) if len(days) else np.nan,
            "lead_median_days": float(days.median()) if len(days) else np.nan,
            "lead_max_days": float(days.max()) if len(days) else np.nan,
            "persistent_median_days": float(persist.median()) if len(persist) else np.nan,
            "lead_metric_version": LEAD_METRIC_VERSION,
        })
    return (pd.DataFrame(records)
            .sort_values("detected", ascending=False).reset_index(drop=True))


def _plot_lead_matrix(all_lead: pd.DataFrame, output: Path) -> None:
    present = set(all_lead["model"])
    models = [name for name in DEFAULT_MODEL_NAMES if name in present]
    models.extend(name for name in all_lead["model"].drop_duplicates() if name not in models)
    events = (
        all_lead[["issuer_code", "event_month"]]
        .drop_duplicates()
        .sort_values(["event_month", "issuer_code"])
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(max(9.0, 1.35 * len(models)), max(8.5, len(events) * 0.34 + 1.8)))
    ax.set_xlim(0, len(models))
    ax.set_ylim(0, len(events) + 1)
    ax.invert_yaxis()
    ax.axis("off")
    short_names = {"RandomForest": "Random Forest", "HistGradientBoosting": "HistGB"}
    for column, model in enumerate(models):
        ax.text(
            column + 0.5,
            0.35,
            short_names.get(model, model),
            ha="center",
            va="center",
            weight="bold",
            fontsize=8,
        )
    for row_number, event in events.iterrows():
        y = row_number + 1
        ax.text(-0.04, y + 0.5, f"{event.issuer_code}  {event.event_month}", ha="right", va="center", fontsize=8)
        for column, model in enumerate(models):
            record = all_lead.loc[
                all_lead["issuer_code"].eq(event.issuer_code) & all_lead["model"].eq(model)
            ].iloc[0]
            if record["status"] == "no_pre_event_data":
                color, label, text_color = "#cbd5e1", "N/A", "#334155"
            elif record["status"] == "detected":
                color, label, text_color = "#86efac", f"{int(record['actionable_lead_days'])} d", "#14532d"
            else:
                color, label, text_color = "#fecaca", "miss", "#7f1d1d"
            ax.add_patch(Rectangle((column + 0.03, y + 0.08), 0.94, 0.84, facecolor=color, edgecolor="white"))
            ax.text(column + 0.5, y + 0.5, label, ha="center", va="center", fontsize=8, color=text_color, weight="bold")
    ax.legend(handles=[
        Patch(facecolor="#86efac", label="Detected inside actionable 1-3M window"),
        Patch(facecolor="#fecaca", label="No actionable-window alarm"),
        Patch(facecolor="#cbd5e1", label="No pre-event panel data"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.025), frameon=False, fontsize=8)
    ax.set_title("Standalone lead-time audit by first DP/RS event", weight="bold", pad=10)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def run_analysis(
    db_path: Path,
    *,
    table: str = DEFAULT_TABLE,
    model_names: list[str] | None = None,
    workload: float = DEFAULT_WORKLOAD,
    output_root: Path | None = None,
    run_label: str = "standalone",
) -> dict[str, Any]:
    if not 0.005 <= workload <= 0.30:
        raise ValueError("workload must be between 0.005 and 0.30")
    model_names = model_names or list(DEFAULT_MODEL_NAMES)
    output_root = (output_root or (ROOT / "standalone_leadtime_runs")).resolve()
    run_id = f"{run_label}_{_now_id()}_{uuid.uuid4().hex[:6]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Loading Dataset2 from {db_path.resolve()}")
    data = load_model_data(db_path, table)
    print(
        f"Rows {len(data['panel']):,} | issuers {data['panel']['issuer_code'].nunique():,} | "
        f"positive months {int(data['y'].sum()):,} | explicit event issuers {len(data['app_events'])} | "
        f"all dated/status event issuers {len(data['all_events'])}"
    )
    runs: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    lead_app_frames: list[pd.DataFrame] = []
    lead_all_frames: list[pd.DataFrame] = []
    workload_sweep_frames: list[pd.DataFrame] = []
    oof_output = data["panel"][["issuer_code", "month"]].copy()
    oof_output["target_pre3m"] = data["y"]
    latest_output = data["panel"].iloc[data["latest_rows"]][["issuer_code", "month"]].copy()
    for model_name in model_names:
        print(f"\nTraining {model_name} with common issuer-held-out grouped OOF protocol")
        fitted = fit_oof(data, model_name, workload)
        model_sweep = precision_recall_workload_sweep(
            data["y"], fitted["oof"], data["latest_rows"], model_name
        )
        fitted["workload_sweep"] = model_sweep
        workload_sweep_frames.append(model_sweep)
        app_lead = lead_table(data, fitted, data["app_events"])
        all_lead = lead_table(data, fitted, data["all_events"])
        app_summary = _lead_summary(app_lead, "app")
        all_summary = _lead_summary(all_lead, "all")
        fitted["metrics"].update(app_summary)
        fitted["metrics"].update(all_summary)
        summary_rows.append(fitted["metrics"])
        app_lead["catalog_scope"] = "explicit_onset_app_compatible"
        all_lead["catalog_scope"] = "all_dated_or_status_events"
        lead_app_frames.append(app_lead)
        lead_all_frames.append(all_lead)
        key = model_name.lower()
        oof_output[f"fold_{key}"] = fitted["fold_of"]
        oof_output[f"oof_score_{key}"] = fitted["oof"]
        oof_output[f"history_alarm_{key}"] = fitted["history_alarm"].astype(int)
        latest_output = latest_output.merge(
            pd.DataFrame({
                "issuer_code": data["panel"].iloc[data["latest_rows"]]["issuer_code"].to_numpy(),
                f"latest_score_{key}": fitted["oof"][data["latest_rows"]],
            }),
            on="issuer_code",
            how="left",
            validate="one_to_one",
        )
        runs[model_name] = {
            **fitted,
            "lead_app": app_lead,
            "lead_all": all_lead,
        }
        partial_summary = pd.DataFrame(summary_rows)
        partial_summary.to_csv(
            run_dir / "model_summary.partial.csv", index=False, encoding="utf-8-sig"
        )
        pd.concat(lead_all_frames, ignore_index=True).to_csv(
            run_dir / "lead_time_all_events.partial.csv", index=False, encoding="utf-8-sig"
        )
        (run_dir / "run_progress.json").write_text(
            json.dumps({
                "run_id": run_id,
                "status": "running",
                "completed_models": list(runs),
                "requested_models": model_names,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = pd.DataFrame(summary_rows)
    lead_app_all = pd.concat(lead_app_frames, ignore_index=True)
    lead_all = pd.concat(lead_all_frames, ignore_index=True)
    workload_sweep = pd.concat(workload_sweep_frames, ignore_index=True)
    action_values = pd.to_numeric(lead_all["actionable_lead_days"], errors="coerce").dropna()
    if len(action_values) and float(action_values.max()) > 92:
        raise AssertionError("Actionable lead time exceeds 92 days")
    summary.to_csv(run_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    workload_sweep.to_csv(
        run_dir / "precision_recall_workload_1_10.csv", index=False, encoding="utf-8-sig"
    )
    lead_app_all.to_csv(run_dir / "lead_time_app_events.csv", index=False, encoding="utf-8-sig")
    lead_all.to_csv(run_dir / "lead_time_all_events.csv", index=False, encoding="utf-8-sig")
    latest_output.to_csv(run_dir / "latest_scores.csv", index=False, encoding="utf-8-sig")
    oof_output.to_csv(run_dir / "oof_scores.csv.gz", index=False, compression="gzip")
    _plot_summary(summary, run_dir / "standalone_model_summary.jpg")
    _plot_precision_recall(
        runs,
        workload_sweep,
        float(np.mean(data["y"])),
        run_dir / "precision_recall_curve_1_10pct.jpg",
    )
    _plot_lead_matrix(lead_all, run_dir / "standalone_lead_matrix.jpg")
    result_db = run_dir / "standalone_leadtime_results.db"
    with closing(sqlite3.connect(result_db)) as connection:
        summary.to_sql("standalone_model_summary", connection, if_exists="replace", index=False)
        workload_sweep.to_sql(
            "standalone_precision_recall_workload", connection, if_exists="replace", index=False
        )
        lead_app_all.to_sql("standalone_lead_time_app_events", connection, if_exists="replace", index=False)
        lead_all.to_sql("standalone_lead_time_all_events", connection, if_exists="replace", index=False)
        _lead_matrix_table(lead_all).to_sql(
            "standalone_lead_time_matrix", connection, if_exists="replace", index=False)
        _lead_model_summary_table(lead_all).to_sql(
            "standalone_lead_time_model_summary", connection, if_exists="replace", index=False)
        latest_output.to_sql("standalone_latest_scores", connection, if_exists="replace", index=False)
        oof_output.to_sql("standalone_oof_scores", connection, if_exists="replace", index=False, chunksize=10000)
        pd.DataFrame([{
            "schema_version": RESULT_SCHEMA_VERSION,
            "program_version": VERSION,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "run_id": run_id,
            "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_database": str(db_path.resolve()),
            "source_table": table,
            "lead_metric_version": LEAD_METRIC_VERSION,
            "lead_definition": "first alarm inside 1-3 calendar months before first event",
            "persistent_definition": "start of final continuous monthly alarm episode before event",
            "lead_window_min_months": 1,
            "lead_window_max_months": 3,
            "workload": workload,
            "workload_sweep_percent": "1,2,3,4,5,6,7,8,9,10",
            "pr_auc_definition": "trapezoidal area under full out-of-fold precision-recall curve",
            "seed": SEED,
            "model_suite": ",".join(model_names),
            "protocol": "common full-panel issuer-held-out OOF; XGBoost/CatBoost remain app_dataset2 compatible",
        }]).to_sql("standalone_run_metadata", connection, if_exists="replace", index=False)
        connection.commit()
    metadata = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "program_version": VERSION,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "run_id": run_id,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": inspect_database(db_path, table),
        "models": model_names,
        "workload": workload,
        "workload_sweep_percent": list(WORKLOAD_SWEEP_PCT),
        "pr_auc_definition": "trapezoidal area under full out-of-fold precision-recall curve",
        "seed": SEED,
        "protocol": "common issuer-held-out grouped OOF; app-compatible XGBoost/CatBoost",
        "lead_metric_version": LEAD_METRIC_VERSION,
        "lead_window_months": [1, 3],
        "summary": summary.to_dict(orient="records"),
        "output_database": str(result_db),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(_json_ready(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "run_progress.json").write_text(
        json.dumps({
            "run_id": run_id,
            "status": "complete",
            "completed_models": list(runs),
            "requested_models": model_names,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nRun summary")
    print(summary[[
        "model", "auc_oof", "average_precision_oof", "pr_auc_trapezoid_oof", "precision_at_workload",
        "recall_at_workload", "app_detected", "app_evaluable_events",
        "app_lead_median_days", "app_persistent_median_days", "runtime_seconds",
    ]].to_string(index=False))
    print(f"\nResults: {run_dir}")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "result_db": result_db,
        "data": data,
        "runs": runs,
        "summary": summary,
        "lead_app": lead_app_all,
        "lead_all": lead_all,
        "workload_sweep": workload_sweep,
    }


def _numeric_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    difference = np.abs(a - b)
    difference[both_nan] = 0.0
    difference[np.isnan(difference)] = np.inf
    return pd.Series(difference, index=left.index)


def verify_against_app(
    db_path: Path,
    *,
    model_names: list[str],
    workload: float,
    output_root: Path | None,
) -> tuple[pd.DataFrame, Path]:
    standalone = run_analysis(
        db_path,
        model_names=model_names,
        workload=workload,
        output_root=output_root,
        run_label="verify_app",
    )
    import app_dataset2
    engine = app_dataset2.Dataset2Engine()
    engine.load()
    if Path(engine.db_path).resolve() != db_path.resolve():
        raise RuntimeError(
            "app_dataset2.py resolved a different database: " + engine.db_path
        )
    rows = []
    for model_name in model_names:
        print(f"\nIndependent comparison with app_dataset2.py: {model_name}")
        app_run = engine.train_model(model_name, workload=workload, progress=print)
        own_run = standalone["runs"][model_name]
        oof_max_abs_diff = float(np.max(np.abs(app_run.oof - own_run["oof"])))
        threshold_abs_diff = float(abs(app_run.threshold - own_run["metrics"]["threshold_latest_cross_section"]))
        metric_pairs = {
            "auc_oof": "auc_oof",
            "average_precision_oof": "average_precision_oof",
            "precision_at_workload": "precision_at_workload",
            "recall_at_workload": "recall_at_workload",
            "f1_at_workload": "f1_at_workload",
        }
        metric_diffs = [
            abs(float(app_run.metrics[app_key]) - float(own_run["metrics"][own_key]))
            for app_key, own_key in metric_pairs.items()
        ]
        app_lead, app_lead_summary = engine.lead_time_table(model_name)
        own_lead = own_run["lead_app"].copy()
        compare = app_lead[[
            "issuer_code", "first_alarm", "lead_time_days", "persistent_alarm_start",
            "persistent_alarm_days",
        ]].merge(
            own_lead[[
                "issuer_code", "actionable_alarm_month", "actionable_lead_days",
                "persistent_alarm_start", "persistent_alarm_days",
            ]],
            on="issuer_code",
            how="outer",
            suffixes=("_app", "_standalone"),
            indicator=True,
        )
        mismatch = compare["_merge"].ne("both")
        mismatch |= compare["first_alarm"].fillna("").astype(str).ne(
            compare["actionable_alarm_month"].fillna("").astype(str)
        )
        mismatch |= compare["persistent_alarm_start_app"].fillna("").astype(str).ne(
            compare["persistent_alarm_start_standalone"].fillna("").astype(str)
        )
        mismatch |= _numeric_difference(compare["lead_time_days"], compare["actionable_lead_days"]).gt(1e-12)
        mismatch |= _numeric_difference(compare["persistent_alarm_days_app"], compare["persistent_alarm_days_standalone"]).gt(1e-12)
        lead_mismatch_rows = int(mismatch.sum())
        max_metric_abs_diff = float(max(metric_diffs))
        matches = (
            oof_max_abs_diff <= 1e-12
            and threshold_abs_diff <= 1e-12
            and max_metric_abs_diff <= 1e-12
            and lead_mismatch_rows == 0
            and len(app_lead) == len(own_lead)
        )
        rows.append({
            "model": model_name,
            "matches_app_dataset2": bool(matches),
            "rows_compared": len(app_run.oof),
            "event_rows_compared": len(app_lead),
            "oof_max_abs_diff": oof_max_abs_diff,
            "threshold_abs_diff": threshold_abs_diff,
            "max_metric_abs_diff": max_metric_abs_diff,
            "lead_mismatch_rows": lead_mismatch_rows,
            "app_auc": app_run.metrics["auc_oof"],
            "standalone_auc": own_run["metrics"]["auc_oof"],
            "app_threshold": app_run.threshold,
            "standalone_threshold": own_run["metrics"]["threshold_latest_cross_section"],
            "app_detected": app_lead_summary["n_caught"],
            "standalone_detected": own_run["metrics"]["app_detected"],
        })
    verification = pd.DataFrame(rows)
    verification.to_csv(
        standalone["run_dir"] / "app_dataset2_compatibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with closing(sqlite3.connect(standalone["result_db"])) as connection:
        verification.to_sql("app_dataset2_compatibility", connection, if_exists="replace", index=False)
    print("\nCompatibility result")
    print(verification.to_string(index=False))
    if not verification["matches_app_dataset2"].all():
        raise AssertionError("Standalone output does not match app_dataset2.py")
    return verification, standalone["run_dir"]


def self_test() -> None:
    months = pd.to_datetime(["2024-01-01", "2024-03-01", "2024-04-01", "2024-05-01"])
    rows = pd.DataFrame({"month_dt": months, "alarm": [True, True, True, True]})
    metrics = compute_lead_for_issuer(rows, pd.Timestamp("2024-06-01"))
    assert metrics["actionable_alarm_month"] == "2024-03-01"
    assert metrics["actionable_lead_days"] == 92.0
    assert metrics["persistent_alarm_start"] == "2024-03-01"
    no_alarm = rows.assign(alarm=False)
    empty_metrics = compute_lead_for_issuer(no_alarm, pd.Timestamp("2024-06-01"))
    assert not empty_metrics["actionable_alarm_found"]
    assert pd.isna(empty_metrics["persistent_alarm_days"])

    test_y = np.zeros(100, dtype=int)
    test_y[[0, 5, 20]] = 1
    test_scores = np.linspace(1.0, 0.0, 100)
    sweep = precision_recall_workload_sweep(
        test_y, test_scores, np.arange(10), "TestModel"
    )
    assert sweep["review_workload_pct"].tolist() == list(WORKLOAD_SWEEP_PCT)
    assert sweep["n_flagged_months"].tolist() == list(WORKLOAD_SWEEP_PCT)
    assert sweep["recall"].is_monotonic_increasing

    rng = np.random.default_rng(SEED)
    probit_x = rng.normal(size=(120, 4))
    probit_y = (probit_x[:, 0] - 0.5 * probit_x[:, 1] > 1.0).astype(int)
    probit = WeightedProbitClassifier(
        positive_weight=float((len(probit_y) - probit_y.sum()) / probit_y.sum()),
        max_iter=60,
    ).fit(probit_x, probit_y)
    probit_probability = probit.predict_proba(probit_x)
    assert probit_probability.shape == (len(probit_y), 2)
    assert np.isfinite(probit_probability).all()
    assert np.allclose(probit_probability.sum(axis=1), 1.0)
    assert _model_names(list(DEFAULT_MODEL_KEYS)) == list(DEFAULT_MODEL_NAMES)
    print("PASS: lead-time, PR workload sweep, model mapping, and weighted probit checks")


def _model_names(values: list[str]) -> list[str]:
    return [MODEL_KEY_TO_NAME[value.lower()] for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Dataset2 snapshot, iBond update, and multi-model lead-time pipeline"
    )
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Validate a Dataset2 SQLite database")
    inspect_parser.add_argument("--db", type=Path, default=DEFAULT_SOURCE_DB)
    inspect_parser.add_argument("--table", default=DEFAULT_TABLE)

    build_parser = sub.add_parser("build-db", help="Create a new immutable Dataset2 snapshot")
    build_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DB)
    build_parser.add_argument("--output", type=Path)
    build_parser.add_argument("--table", default=DEFAULT_TABLE)

    update_parser = sub.add_parser("update-ibond", help="Create a new snapshot with refreshed iBond payment-default labels")
    update_parser.add_argument("--base-db", type=Path, required=True)
    update_parser.add_argument("--output", type=Path)
    source_group = update_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--from-db", type=Path, help="Read already downloaded iBond raw tables")
    source_group.add_argument("--download", action="store_true", help="Call download_bond in no-save mode")
    update_parser.add_argument("--table", default=DEFAULT_TABLE)

    run_parser = sub.add_parser("run", help="Run the standalone multi-model lead-time comparison")
    run_parser.add_argument("--db", type=Path, default=DEFAULT_SOURCE_DB)
    run_parser.add_argument("--table", default=DEFAULT_TABLE)
    run_parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_KEY_TO_NAME),
        default=list(DEFAULT_MODEL_KEYS),
        help=(
            "Models to run; omission runs all nine models"
        ),
    )
    run_parser.add_argument("--workload", type=float, default=DEFAULT_WORKLOAD)
    run_parser.add_argument("--output-root", type=Path)

    verify_parser = sub.add_parser("verify-app", help="Run both implementations and require exact app_dataset2 agreement")
    verify_parser.add_argument("--db", type=Path, default=DEFAULT_SOURCE_DB)
    verify_parser.add_argument("--models", nargs="+", choices=["xgboost", "catboost"], default=["xgboost", "catboost"])
    verify_parser.add_argument("--workload", type=float, default=DEFAULT_WORKLOAD)
    verify_parser.add_argument("--output-root", type=Path)

    sub.add_parser("self-test", help="Run deterministic lead-time unit checks")
    args = parser.parse_args()

    if args.command == "inspect":
        _print_inspection(inspect_database(args.db, args.table))
    elif args.command == "build-db":
        output = args.output or _default_snapshot_path()
        _print_inspection(build_snapshot(args.source, output, args.table))
    elif args.command == "update-ibond":
        output = args.output or (
            ROOT / "standalone_dataset2" / f"dataset2_941_ibond_{_now_id()}.db"
        )
        _print_inspection(update_ibond_snapshot(
            args.base_db,
            output,
            raw_source_db=args.from_db,
            download=args.download,
            table=args.table,
        ))
    elif args.command == "run":
        run_analysis(
            args.db,
            table=args.table,
            model_names=_model_names(args.models),
            workload=args.workload,
            output_root=args.output_root,
        )
    elif args.command == "verify-app":
        verify_against_app(
            args.db,
            model_names=_model_names(args.models),
            workload=args.workload,
            output_root=args.output_root,
        )
    elif args.command == "self-test":
        self_test()


if __name__ == "__main__":
    main()
