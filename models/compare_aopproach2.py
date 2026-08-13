# -*- coding: utf-8 -*-
"""Read-only command-line comparison of Approach 1 and Approach 2 results.

The filename intentionally follows the requested spelling. This script does not
fit models or write SQLite tables. It validates and reports the latest saved
base-feature and 33-feature pipeline results.

Run:
    python compare_aopproach2.py
    python compare_aopproach2.py --scope base
    python compare_aopproach2.py --scope 33
    python compare_aopproach2.py --db cmdf_credit.db
"""
from __future__ import annotations

import argparse
import ast
import math
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "cmdf_credit.db"
CONTRACT_FILE = HERE / "lead_metrics.py"
MAX_ACTIONABLE_DAYS = 92.0

PAIRS = {
    "base": {
        "title": "Corporate bond base-feature pipelines",
        "approach_1": ("bond_ews_summary", "bond_ews_leadtime"),
        "approach_2": ("bond_ews_xgb_summary", "bond_ews_xgb_leadtime"),
    },
    "33": {
        "title": "Corporate bond 33-feature pipelines",
        "approach_1": ("bond_ews_summary_33", "bond_ews_leadtime_33"),
        "approach_2": (
            "bond_ews_xgb_summary_33",
            "bond_ews_xgb_leadtime_33",
        ),
    },
}


class ComparisonError(RuntimeError):
    pass


@dataclass
class ModelResult:
    summary_table: str
    lead_table: str
    run_at: str
    auc_oos: float
    n_events: int
    n_caught: int
    actionable: list[float]
    persistent: list[float]
    actionable_over_limit: int


def _metric_contract(path: Path) -> dict[str, Any]:
    wanted = {
        "LEAD_METRIC_VERSION",
        "LEAD_WINDOW_MIN_MONTHS",
        "LEAD_WINDOW_MAX_MONTHS",
    }
    values: dict[str, Any] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)
    missing = wanted.difference(values)
    if missing:
        raise ComparisonError(
            f"Cannot read metric contract fields: {sorted(missing)}"
        )
    return values


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
    }


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _integer(value: Any, default: int = 0) -> int:
    number = _finite_float(value)
    return int(number) if math.isfinite(number) else default


def _numeric_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
) -> list[float]:
    rows = conn.execute(
        f"SELECT {_quote(column)} FROM {_quote(table)} "
        f"WHERE {_quote(column)} IS NOT NULL"
    )
    values = [_finite_float(row[0]) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _distinct_text(
    conn: sqlite3.Connection,
    table: str,
    column: str,
) -> set[str]:
    rows = conn.execute(
        f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
        f"WHERE {_quote(column)} IS NOT NULL"
    )
    return {str(row[0]) for row in rows}


def _load_model_result(
    conn: sqlite3.Connection,
    summary_table: str,
    lead_table: str,
    contract: dict[str, Any],
    max_actionable_days: float,
) -> ModelResult:
    for table in (summary_table, lead_table):
        if not _table_exists(conn, table):
            raise ComparisonError(f"Required table is missing: {table}")

    required_summary = {
        "run_at",
        "lead_metric_version",
        "lead_window_min_months",
        "lead_window_max_months",
        "median_persistent_alarm_days",
    }
    required_lead = {
        "lead_days",
        "persistent_alarm_days",
        "lead_metric_version",
        "lead_window_min_months",
        "lead_window_max_months",
    }
    summary_missing = required_summary.difference(
        _columns(conn, summary_table)
    )
    lead_missing = required_lead.difference(_columns(conn, lead_table))
    if summary_missing:
        raise ComparisonError(
            f"{summary_table} missing columns: {sorted(summary_missing)}"
        )
    if lead_missing:
        raise ComparisonError(
            f"{lead_table} missing columns: {sorted(lead_missing)}"
        )

    conn.row_factory = sqlite3.Row
    summary = conn.execute(
        f"SELECT * FROM {_quote(summary_table)} LIMIT 1"
    ).fetchone()
    if summary is None:
        raise ComparisonError(f"{summary_table} is empty")

    expected_version = str(contract["LEAD_METRIC_VERSION"])
    summary_version = str(summary["lead_metric_version"])
    lead_versions = _distinct_text(
        conn, lead_table, "lead_metric_version"
    )
    if summary_version != expected_version:
        raise ComparisonError(
            f"{summary_table} version {summary_version!r}; "
            f"expected {expected_version!r}"
        )
    if lead_versions != {expected_version}:
        raise ComparisonError(
            f"{lead_table} versions {sorted(lead_versions)!r}; "
            f"expected only {expected_version!r}"
        )

    expected_window = (
        int(contract["LEAD_WINDOW_MIN_MONTHS"]),
        int(contract["LEAD_WINDOW_MAX_MONTHS"]),
    )
    summary_window = (
        _integer(summary["lead_window_min_months"]),
        _integer(summary["lead_window_max_months"]),
    )
    lead_windows = {
        (_integer(row[0]), _integer(row[1]))
        for row in conn.execute(
            f"SELECT DISTINCT lead_window_min_months, "
            f"lead_window_max_months FROM {_quote(lead_table)}"
        )
    }
    if summary_window != expected_window or lead_windows != {expected_window}:
        raise ComparisonError(
            f"{lead_table} window mismatch: summary={summary_window}, "
            f"rows={sorted(lead_windows)}, expected={expected_window}"
        )

    actionable = _numeric_column(conn, lead_table, "lead_days")
    persistent = _numeric_column(
        conn, lead_table, "persistent_alarm_days"
    )
    n_events = _integer(
        summary["n_events"]
        if "n_events" in summary.keys()
        else summary["n_defaulted_issuers"]
    )
    n_caught = _integer(summary["n_caught"])
    return ModelResult(
        summary_table=summary_table,
        lead_table=lead_table,
        run_at=str(summary["run_at"]),
        auc_oos=_finite_float(summary["auc_oos"]),
        n_events=n_events,
        n_caught=n_caught,
        actionable=actionable,
        persistent=persistent,
        actionable_over_limit=sum(
            value > max_actionable_days for value in actionable
        ),
    )


def _triplet(values: list[float]) -> str:
    if not values:
        return "N/A"
    return (
        f"{min(values):.1f} / {statistics.median(values):.1f} / "
        f"{max(values):.1f}"
    )


def _median_months(values: list[float]) -> str:
    if not values:
        return "N/A"
    return f"{statistics.median(values) / 30.4375:.4f}"


def _auc(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "N/A"


def _row(label: str, left: str, right: str) -> None:
    print(f"{label:<34}{left:>23}{right:>23}")


def _print_pair(
    title: str,
    approach_1: ModelResult,
    approach_2: ModelResult,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    _row("Metric", "Approach 1", "Approach 2")
    print("-" * 80)
    _row("Run at", approach_1.run_at, approach_2.run_at)
    _row("OOS AUC", _auc(approach_1.auc_oos), _auc(approach_2.auc_oos))
    _row(
        "Actionable caught / events",
        f"{approach_1.n_caught} / {approach_1.n_events}",
        f"{approach_2.n_caught} / {approach_2.n_events}",
    )
    _row(
        "Actionable n",
        str(len(approach_1.actionable)),
        str(len(approach_2.actionable)),
    )
    _row(
        "Actionable days min/med/max",
        _triplet(approach_1.actionable),
        _triplet(approach_2.actionable),
    )
    _row(
        "Persistent n",
        str(len(approach_1.persistent)),
        str(len(approach_2.persistent)),
    )
    _row(
        "Persistent days min/med/max",
        _triplet(approach_1.persistent),
        _triplet(approach_2.persistent),
    )
    _row(
        "Persistent median months",
        _median_months(approach_1.persistent),
        _median_months(approach_2.persistent),
    )
    _row(
        f"Actionable > {MAX_ACTIONABLE_DAYS:.0f} days",
        str(approach_1.actionable_over_limit),
        str(approach_2.actionable_over_limit),
    )
    if math.isfinite(approach_1.auc_oos) and math.isfinite(
        approach_2.auc_oos
    ):
        delta = approach_2.auc_oos - approach_1.auc_oos
        winner = (
            "Approach 2"
            if delta > 0
            else "Approach 1"
            if delta < 0
            else "Tie"
        )
        print(
            f"\nOOS AUC comparison: {winner}; "
            f"Approach 2 - Approach 1 = {delta:+.4f}"
        )


def _print_stored_33_comparison(conn: sqlite3.Connection) -> tuple[int, int]:
    table = "ibond_model_compare_33features"
    if not _table_exists(conn, table):
        print(f"\nStored 33-feature comparison: {table} is missing")
        return 0, 0
    required = {
        "model_approach",
        "mean_lead_time_months",
        "median_lead_time_months",
        "median_persistent_alarm_months",
        "lead_metric_version",
    }
    missing = required.difference(_columns(conn, table))
    if missing:
        raise ComparisonError(
            f"{table} missing columns: {sorted(missing)}"
        )

    rows = conn.execute(
        f"SELECT model_approach, mean_lead_time_months, "
        f"median_lead_time_months, median_persistent_alarm_months "
        f"FROM {_quote(table)}"
    ).fetchall()
    print()
    print("Stored 33-feature comparison table")
    print("-" * 80)
    for row in rows:
        print(str(row[0]))
        print(
            f"  actionable mean/median months: "
            f"{_finite_float(row[1]):.4f} / {_finite_float(row[2]):.4f}"
        )
        print(
            f"  persistent median months: "
            f"{_finite_float(row[3]):.4f}"
        )
    db_legacy = sum(
        math.isclose(
            _finite_float(row[0]),
            48.1,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for row in conn.execute(
            f"SELECT mean_lead_time_months FROM {_quote(table)}"
        )
    )
    source_file = HERE / "compare_ibond_33features_models.py"
    source_legacy = (
        source_file.read_text(encoding="utf-8", errors="replace").count("48.1")
        if source_file.exists()
        else -1
    )
    print(f"Legacy 48.1 rows in SQLite: {db_legacy}")
    print(f"Literal 48.1 in active comparison source: {source_legacy}")
    return db_legacy, source_legacy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only comparison of current Approach 1 and Approach 2 "
            "lead metric results."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="SQLite database path (default: cmdf_credit.db beside this script)",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "base", "33"),
        default="all",
        help="Result set to print (default: all)",
    )
    parser.add_argument(
        "--max-actionable-days",
        type=float,
        default=MAX_ACTIONABLE_DAYS,
        help="Validation ceiling for actionable lead days (default: 92)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    db_path = args.db.expanduser().resolve()
    if not db_path.is_file():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        return 2
    if not CONTRACT_FILE.is_file():
        print(
            f"ERROR: metric contract not found: {CONTRACT_FILE}",
            file=sys.stderr,
        )
        return 2

    try:
        contract = _metric_contract(CONTRACT_FILE)
        uri = db_path.as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        print("CMDF Approach 1 vs Approach 2 - saved result comparison")
        print(f"Database: {db_path}")
        print(f"Open mode: read-only")
        print(f"Lead metric version: {contract['LEAD_METRIC_VERSION']}")
        print(
            "Actionable window: "
            f"{contract['LEAD_WINDOW_MIN_MONTHS']}-"
            f"{contract['LEAD_WINDOW_MAX_MONTHS']} calendar months"
        )

        selected = PAIRS if args.scope == "all" else {
            args.scope: PAIRS[args.scope]
        }
        over_limit_total = 0
        for config in selected.values():
            a1 = _load_model_result(
                conn,
                *config["approach_1"],
                contract,
                args.max_actionable_days,
            )
            a2 = _load_model_result(
                conn,
                *config["approach_2"],
                contract,
                args.max_actionable_days,
            )
            over_limit_total += (
                a1.actionable_over_limit + a2.actionable_over_limit
            )
            _print_pair(config["title"], a1, a2)

        db_legacy, source_legacy = _print_stored_33_comparison(conn)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()

        print()
        print("=" * 80)
        print("VALIDATION")
        print("=" * 80)
        print(f"SQLite integrity_check: {integrity}")
        print(f"Actionable values over {args.max_actionable_days:.0f} days: {over_limit_total}")
        print(f"Legacy 48.1 rows in comparison table: {db_legacy}")
        print(f"Literal 48.1 in active source: {source_legacy}")
        passed = (
            integrity == "ok"
            and over_limit_total == 0
            and db_legacy == 0
            and source_legacy == 0
        )
        print(f"Overall validation: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    except (ComparisonError, sqlite3.Error, OSError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
