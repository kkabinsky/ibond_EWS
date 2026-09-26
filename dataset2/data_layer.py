# -*- coding: utf-8 -*-
"""Dataset2 data layer for app_dataset2.py.

Self-contained apart from thaibma_paths.py, which only locates the database;
nothing outside ./dataset2 is changed at run time.

Source     cmdf_credit.db found by thaibma_paths (repository root after
           `python dataset/build_db.py`, or a folder above it), read-only
Table      ibond_33features_panel_941firm
Target     y_pre3m, the 1-3 month pre-event window
Features   30 financial, market, macro and governance columns

Run
    python data_layer.py            print the load banner and exit
    python data_layer.py --selftest check the table, the columns and the models
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# LightGBM is fitted on a plain array, so sklearn's feature-name check has
# nothing to compare against and prints a warning that means nothing here.
import warnings
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning)

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

VERSION = "1.0.0"
LEAD_METRIC_VERSION = "lead_metrics_actionable_1_3m_persistent_v1"
HERE = Path(__file__).resolve().parent
# ibond_EWS: thaibma_paths finds cmdf_credit.db (the repository root after
# `python dataset/build_db.py`, or a folder above it)
sys.path.insert(0, str(HERE.parent))
try:
    from thaibma_paths import DB as _THAIBMA_DB
    DEFAULT_SOURCE_DB = Path(_THAIBMA_DB)
except ImportError:
    DEFAULT_SOURCE_DB = HERE.parent / "cmdf_credit.db"
DEFAULT_TABLE = "ibond_33features_panel_941firm"
DEFAULT_WORKLOAD = 0.05
SEED = 42

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

MODEL_NAMES = [
    # gradient boosting
    "XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting",
    "GradientBoosting",
    # bagging
    "RandomForest", "ExtraTrees",
    # linear, the two classic credit-scoring links
    "Logistic", "Probit",
]


class ProbitClassifier(BaseEstimator, ClassifierMixin):
    """Probit link through statsmodels, with the sklearn fit/predict_proba shape.

    The rare-event weight is applied as a frequency weight so the coefficients
    are comparable with the weighted logistic fit rather than dominated by the
    non-event months.
    """

    def __init__(self, positive_weight: float = 1.0, maxiter: int = 60):
        self.positive_weight = positive_weight
        self.maxiter = maxiter

    def fit(self, X, y):
        if sm is None:
            raise RuntimeError("statsmodels is not installed")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = sm.add_constant((X - self.mean_) / self.scale_, has_constant="add")
        weights = np.where(y > 0, float(self.positive_weight), 1.0)
        model = sm.GLM(y, Z, family=sm.families.Binomial(sm.families.links.Probit()),
                       freq_weights=weights)
        self.result_ = model.fit(maxiter=self.maxiter, tol=1e-6)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Z = sm.add_constant((X - self.mean_) / self.scale_, has_constant="add")
        p = np.clip(np.asarray(self.result_.predict(Z), dtype=float), 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------- SQLite access
def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _qident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return '"' + name + '"'


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def table_names(db_path: Path = DEFAULT_SOURCE_DB) -> list[str]:
    with closing(_readonly(db_path)) as con:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [str(r[0]) for r in rows]


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(r[1]) for r in connection.execute(f"PRAGMA table_info({_qident(table)})")]


def _check_required(columns: Iterable[str]) -> None:
    missing = sorted(set(REQUIRED_META + FEATURES) - set(columns))
    if missing:
        raise ValueError("Dataset2 is missing required columns: " + ", ".join(missing))


def read_table(table: str, limit: int | None = None,
               db_path: Path = DEFAULT_SOURCE_DB) -> pd.DataFrame:
    """Any table in the database, for the Data Inspector panel."""
    q = f"SELECT * FROM {_qident(table)}"
    if limit:
        q += f" LIMIT {int(limit)}"
    with closing(_readonly(db_path)) as con:
        return pd.read_sql_query(q, con)


def row_count(table: str, db_path: Path = DEFAULT_SOURCE_DB) -> int:
    with closing(_readonly(db_path)) as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {_qident(table)}").fetchone()[0])


# ------------------------------------------------------------- event catalogue
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


def event_catalog(panel: pd.DataFrame, explicit_only: bool) -> pd.DataFrame:
    """One row per issuer that has an event.

    explicit_only=True  keeps issuers with a dated onset flag, the set the app
                        evaluates lead time on.
    explicit_only=False also keeps issuers known only by a status column or a
                        dated default/restructure field.
    """
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
            for column, kind in (("default_month", "DP"), ("date_DP", "DP"),
                                 ("restructure_month", "RS"), ("date_RS", "RS")):
                if column not in g.columns:
                    continue
                dates = pd.to_datetime(g[column], errors="coerce").dropna()
                if not dates.empty:
                    candidates.append((pd.Timestamp(dates.min()).to_period("M").to_timestamp(),
                                       kind, column))
            for column, kind in (("d_Default_Payment", "DP"), ("d_Restructure", "RS")):
                if column in g.columns:
                    status = _to_number(g[column]).fillna(0).gt(0)
                    if status.any():
                        candidates.append((pd.Timestamp(g.loc[status, "month_dt"].min())
                                           .to_period("M").to_timestamp(), kind, column))
        if not candidates:
            continue
        event_month = min(item[0] for item in candidates)
        first = [item for item in candidates if item[0] == event_month]
        records.append({
            "issuer_code": str(issuer),
            "firm_name": _firm_name(g),
            "event_month": event_month,
            "event_type": "+".join(sorted({i[1] for i in first})),
            "event_source": "+".join(sorted({i[2] for i in first})),
            "first_panel_month": pd.Timestamp(g["month_dt"].min()),
            "evaluable": bool(pd.Timestamp(g["month_dt"].min()) < event_month),
        })
    cols = ["issuer_code", "firm_name", "event_month", "event_type",
            "event_source", "first_panel_month", "evaluable"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records).sort_values(["event_month", "issuer_code"]).reset_index(drop=True)


# ------------------------------------------------------------------ panel load
def load_model_data(db_path: Path = DEFAULT_SOURCE_DB,
                    table: str = DEFAULT_TABLE,
                    risk_set: bool = False) -> dict[str, Any]:
    """Load the panel.

    risk_set=True drops every month at or after an issuer's first event, which
    is the convention benchmark_dataset1_dataset2.py uses for Approach 1 and 2.
    A firm that has already defaulted is no longer at risk of defaulting, so
    keeping those months would let the model score rows that can never be
    warned about. Use it when comparing against those published numbers.
    """
    db_path = Path(db_path).resolve()
    with closing(_readonly(db_path)) as connection:
        names = {str(r[0]) for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if table not in names:
            raise RuntimeError(f"Table {table!r} was not found in {db_path}")
        available = _columns(connection, table)
        _check_required(available)
        selected = list(dict.fromkeys(
            REQUIRED_META + [c for c in OPTIONAL_META if c in available] + FEATURES))
        frame = pd.read_sql_query(
            "SELECT " + ", ".join(_qident(c) for c in selected)
            + " FROM " + _qident(table), connection)

    frame["issuer_code"] = frame["issuer_code"].astype(str).str.strip()
    frame["month_dt"] = pd.to_datetime(frame["month"], errors="coerce")
    if frame["issuer_code"].eq("").any() or frame["month_dt"].isna().any():
        raise ValueError("Blank issuer_code or invalid month found")
    duplicate = frame.duplicated(["issuer_code", "month_dt"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate issuer-month rows found:\n"
                         + frame.loc[duplicate, ["issuer_code", "month"]].head().to_string(index=False))

    x_frame = frame[FEATURES].copy()
    for column in FEATURES:
        x_frame[column] = pd.to_numeric(x_frame[column], errors="coerce")
        median = x_frame[column].median()
        x_frame[column] = x_frame[column].fillna(0.0 if pd.isna(median) else median)
    target = pd.to_numeric(frame["y_pre3m"], errors="coerce").fillna(0).gt(0).astype(int)

    frame["_source_order"] = np.arange(len(frame), dtype=int)
    order = frame.sort_values(["issuer_code", "month_dt", "_source_order"],
                              kind="stable").index.to_numpy()
    panel = frame.iloc[order].reset_index(drop=True)
    x_frame = x_frame.iloc[order].reset_index(drop=True)
    y = target.iloc[order].to_numpy(dtype=int)

    excluded_post_event = 0
    if risk_set:
        events = event_catalog(panel, explicit_only=False)
        first_event = dict(zip(events["issuer_code"].astype(str),
                               pd.to_datetime(events["event_month"])))
        cut = panel["issuer_code"].astype(str).map(first_event)
        # strictly after the event month: the event month itself stays in the
        # risk set, which is what reproduces the published row count
        keep = ~(cut.notna() & panel["month_dt"].gt(cut))
        excluded_post_event = int((~keep).sum())
        panel = panel.loc[keep].reset_index(drop=True)
        x_frame = x_frame.loc[keep.to_numpy()].reset_index(drop=True)
        y = y[keep.to_numpy()]
    latest_rows = (panel.assign(_row=np.arange(len(panel), dtype=int))
                   .groupby("issuer_code", sort=False).tail(1)["_row"].to_numpy(dtype=int))

    # the event catalogue is built before the risk-set cut, so an issuer whose
    # every month is dropped is still counted as an event that could not be
    # warned about rather than quietly disappearing from the denominator
    catalog_source = frame.iloc[order].reset_index(drop=True)
    return {
        "db_path": db_path,
        "table": table,
        "panel": panel,
        "X": x_frame,
        "y": y,
        "features": FEATURES.copy(),
        "latest_rows": latest_rows,
        "risk_set": bool(risk_set),
        "excluded_post_event_rows": excluded_post_event,
        "app_events": event_catalog(catalog_source, explicit_only=True),
        "all_events": event_catalog(catalog_source, explicit_only=False),
    }


def banner(data: dict[str, Any]) -> str:
    """The one-line load summary, in the same shape the standalone run prints."""
    panel = data["panel"]
    return (
        f"Rows {len(panel):,} | issuers {panel['issuer_code'].nunique():,} | "
        f"positive months {int(np.asarray(data['y']).sum()):,} | "
        f"explicit event issuers {len(data['app_events']):,} | "
        f"all dated/status event issuers {len(data['all_events']):,}"
    )


def load_with_banner(db_path: Path = DEFAULT_SOURCE_DB,
                     table: str = DEFAULT_TABLE) -> tuple[dict[str, Any], list[str]]:
    lines = [f"Loading Dataset2 from {Path(db_path).resolve()}"]
    data = load_model_data(db_path, table)
    lines.append(banner(data))
    for line in lines:
        print(line, flush=True)
    return data, lines


# ---------------------------------------------------------------------- models
def make_model(model_name: str, positive_weight: float) -> Any:
    if model_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        return xgb.XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2.0,
            reg_lambda=3.0, scale_pos_weight=positive_weight,
            tree_method="hist", eval_metric="logloss",
            n_jobs=_workers(), random_state=SEED, verbosity=0,
        )
    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        return LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
            scale_pos_weight=positive_weight, n_jobs=_workers(),
            random_state=SEED, verbose=-1,
        )
    if model_name == "CatBoost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        return CatBoostClassifier(
            iterations=250, depth=4, learning_rate=0.05, l2_leaf_reg=4.0,
            auto_class_weights="Balanced", random_seed=SEED,
            thread_count=_workers(), verbose=0, allow_writing_files=False,
        )
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=250, max_depth=4, learning_rate=0.06,
            l2_regularization=2.0, class_weight="balanced",
            random_state=SEED,
        )
    if model_name == "GradientBoosting":
        # no class_weight in this estimator, so the weight is passed at fit time
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.06,
            subsample=0.85, random_state=SEED,
        )
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=_workers(),
            random_state=SEED,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=_workers(),
            random_state=SEED,
        )
    if model_name == "Logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED),
        )
    if model_name == "Probit":
        if sm is None:
            raise RuntimeError("statsmodels is not installed")
        return ProbitClassifier(positive_weight=positive_weight)
    raise ValueError(f"Unsupported model: {model_name}")


# estimators that carry no class_weight setting take a per-row weight at fit time
SAMPLE_WEIGHT_MODELS = {"GradientBoosting"}


def _fit(model, model_name: str, X, y, positive_weight: float):
    """Fit, passing a sample weight to the estimators that need one.

    MLP sits inside a pipeline, so its weight has to be addressed to the
    final step by name.
    """
    if model_name not in SAMPLE_WEIGHT_MODELS:
        model.fit(X, y)
        return model
    weights = np.where(np.asarray(y) > 0, float(positive_weight), 1.0)
    model.fit(X, y, sample_weight=weights)
    return model


def fit_oof(data: dict[str, Any], model_name: str,
            workload: float = DEFAULT_WORKLOAD,
            progress=None) -> dict[str, Any]:
    """Out-of-fold scores with issuer-grouped stratified folds.

    Grouping by issuer keeps every month of one firm inside one fold, so a
    firm the model has already seen cannot inflate the held-out score.
    """
    values = data["X"].to_numpy(dtype=float)
    y = np.asarray(data["y"])
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
        msg = (f"{model_name}: fold {fold}/{n_splits} "
               f"({len(train_idx):,} train, {len(test_idx):,} held out)")
        print("  " + msg, flush=True)
        if progress is not None:
            progress(msg)
        weight = float((len(train_idx) - positives) / positives)
        model = make_model(model_name, weight)
        _fit(model, model_name, values[train_idx], y[train_idx], weight)
        oof[test_idx] = model.predict_proba(values[test_idx])[:, 1]
        fold_of[test_idx] = fold - 1
    if not np.isfinite(oof).all() or (fold_of < 0).any():
        raise RuntimeError("OOF scoring did not cover all rows")

    threshold = float(np.quantile(oof[data["latest_rows"]], 1.0 - workload))
    n_flagged = max(1, int(round(workload * len(oof))))
    ranked_alarm = np.zeros(len(oof), dtype=int)
    ranked_alarm[np.argsort(oof)[::-1][:n_flagged]] = 1

    # a full-sample refit, used only for feature importance and the live view
    full_weight = float((len(y) - int(y.sum())) / max(int(y.sum()), 1))
    full_model = make_model(model_name, full_weight)
    _fit(full_model, model_name, values, y, full_weight)

    curve_precision, curve_recall, _ = precision_recall_curve(y, oof)
    metrics = {
        "model": model_name,
        "auc_oof": float(roc_auc_score(y, oof)),
        "average_precision_oof": float(average_precision_score(y, oof)),
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
        "threshold": threshold,
        "full_model": full_model,
        "pr_curve": {"precision": curve_precision, "recall": curve_recall},
        "metrics": metrics,
    }


def feature_importance(run: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """Whatever the fitted model exposes, normalised to a share of the total."""
    model = run.get("full_model")
    values = None
    if model is not None:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "named_steps"):
            last = list(model.named_steps.values())[-1]
            if hasattr(last, "coef_"):
                values = np.abs(np.asarray(last.coef_, dtype=float)).ravel()
        elif hasattr(model, "get_feature_importance"):
            values = np.asarray(model.get_feature_importance(), dtype=float)
    if values is None or len(values) != len(features):
        values = np.zeros(len(features), dtype=float)
    total = values.sum()
    share = values / total if total > 0 else values
    return (pd.DataFrame({"feature": features, "importance": values, "share": share})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# ------------------------------------------------------------------- lead time
def _month_ordinal(value: pd.Timestamp) -> int:
    return int(value.year * 12 + value.month)


def compute_lead_for_issuer(rows: pd.DataFrame, event_month: pd.Timestamp) -> dict[str, Any]:
    """Actionable lead time is an alarm 1 to 3 months before the event month.

    An alarm in the event month itself is not a warning, and an alarm more
    than three months out is reported separately as a persistent run rather
    than counted as a detection.
    """
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
                "actionable_lead_months": float(_month_ordinal(event_month)
                                                - _month_ordinal(alarm_month)),
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


def lead_table(data: dict[str, Any], run: dict[str, Any],
               catalog: pd.DataFrame) -> pd.DataFrame:
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


def lead_summary(frame: pd.DataFrame, prefix: str = "app") -> dict[str, Any]:
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



WORKLOAD_SWEEP_PCT = tuple(range(1, 11))


def precision_recall_workload_sweep(y, scores, latest_rows, model_name,
                                    workload_percentages=WORKLOAD_SWEEP_PCT):
    """Precision and recall at each review workload from 1 to 10 per cent.

    The workload is the share of the highest-risk issuer-months a team would
    actually open. Two thresholds are reported for each level: the rank cut
    over the whole history, and the quantile of the latest month of every
    issuer, which is the one an operating desk would use.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    latest_rows = np.asarray(latest_rows, dtype=int)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("y and scores must be non-empty arrays of equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")

    ranked = np.argsort(scores, kind="stable")[::-1]
    records = []
    for pct in workload_percentages:
        pct = int(pct)
        if not 1 <= pct <= 100:
            raise ValueError("workload percentages must be between 1 and 100")
        workload = pct / 100.0
        n_flagged = max(1, int(round(workload * len(scores))))
        alarm = np.zeros(len(scores), dtype=int)
        alarm[ranked[:n_flagged]] = 1
        records.append({
            "model": model_name,
            "review_workload_pct": pct,
            "actual_review_workload_pct": float(100.0 * n_flagged / len(scores)),
            "n_flagged_months": n_flagged,
            "rank_score_threshold": float(scores[ranked[n_flagged - 1]]),
            "latest_cross_section_threshold": float(
                np.quantile(scores[latest_rows], 1.0 - workload)),
            "true_positive_months": int(np.sum((alarm == 1) & (y == 1))),
            "false_positive_months": int(np.sum((alarm == 1) & (y == 0))),
            "false_negative_months": int(np.sum((alarm == 0) & (y == 1))),
            "precision": float(precision_score(y, alarm, zero_division=0)),
            "recall": float(recall_score(y, alarm, zero_division=0)),
            "prevalence": float(y.mean()),
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------- run every model
SUMMARY_COLUMNS = [
    "model", "auc_oof", "average_precision_oof",
    "precision_at_workload", "recall_at_workload",
    "app_detected", "app_evaluable_events",
    "all_detected", "all_evaluable_events", "all_event_catalog",
    "app_lead_median_days", "app_persistent_median_days", "runtime_seconds",
]


def summarise_run(data: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """One row of the run summary: ranking quality plus how many events were caught.

    app_*  the issuers with a dated onset flag, the set lead time is defined on
    all_*  every issuer known by a dated field or a status column
    """
    row = dict(run["metrics"])
    for catalog_key, prefix in (("app_events", "app"), ("all_events", "all")):
        frame = lead_table(data, run, data[catalog_key])
        row.update(lead_summary(frame, prefix))
    return row


def run_all_models(data: dict[str, Any],
                   models: list[str] | None = None,
                   workload: float = DEFAULT_WORKLOAD,
                   progress=None) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Fit every model on the same folds and return the summary table.

    A model whose library is not installed is skipped with a printed note
    rather than stopping the run.
    """
    models = list(models or MODEL_NAMES)
    rows, runs = [], {}
    for name in models:
        header = f"\nTraining {name} with app-compatible grouped OOF protocol"
        print(header, flush=True)
        if progress is not None:
            progress(f"training {name}")
        try:
            run = fit_oof(data, name, workload, progress=progress)
        except Exception as exc:
            print(f"  {name} skipped: {exc}", flush=True)
            continue
        runs[name] = run
        rows.append(summarise_run(data, run))
    if not rows:
        raise RuntimeError("No model finished")
    summary = pd.DataFrame(rows)
    return summary[[c for c in SUMMARY_COLUMNS if c in summary.columns]], runs


def format_run_summary(summary: pd.DataFrame) -> str:
    """The printed block, in the same shape the standalone run produces."""
    catalog = int(summary["all_event_catalog"].iloc[0])
    evaluable = int(summary["all_evaluable_events"].iloc[0])
    lines = ["", "Run summary", summary.to_string(index=False), ""]
    lines.append(
        f"Defaults caught, out of the {catalog} issuers with an event. "
        f"{evaluable} of them can be evaluated: the rest have no month of panel "
        f"data before the event, so no warning was ever possible."
    )
    for row in summary.itertuples(index=False):
        app_rate = row.app_detected / max(row.app_evaluable_events, 1)
        all_rate = row.all_detected / max(row.all_evaluable_events, 1)
        lines.append(
            f"  {row.model:<14s} caught {int(row.all_detected):>3d} / "
            f"{int(row.all_evaluable_events):<3d} evaluable "
            f"({all_rate:5.1%})   of {catalog} in the catalogue   "
            f"explicit onset {int(row.app_detected):>3d} / "
            f"{int(row.app_evaluable_events):<3d} ({app_rate:5.1%})   "
            f"recall on issuer-months {row.recall_at_workload:6.4f}"
        )
    return "\n".join(lines)


def print_run_summary(summary: pd.DataFrame) -> str:
    text = format_run_summary(summary)
    print(text, flush=True)
    return text


# --------------------------------------------------- Approach 1 risk mechanics
RISK_BANDS = [
    ("HIGH RISK", "#dc2626"),
    ("ELEVATED", "#ea580c"),
    ("WATCH", "#f59e0b"),
    ("OK", "#16a34a"),
]


def risk_band(pd_value: float, thresholds=(0.20, 0.10, 0.05)) -> tuple[str, str]:
    high, elevated, watch = thresholds
    if pd_value >= high:
        return RISK_BANDS[0]
    if pd_value >= elevated:
        return RISK_BANDS[1]
    if pd_value >= watch:
        return RISK_BANDS[2]
    return RISK_BANDS[3]


def momentum(series: pd.Series, window: int = 3) -> pd.Series:
    """Risk momentum M(t): the change in PD over the trailing window."""
    s = pd.to_numeric(series, errors="coerce")
    return s.diff(window)


def issuer_frame(data: dict[str, Any], run: dict[str, Any] | None,
                 issuer_code: str) -> pd.DataFrame:
    """One issuer's monthly history with PD, momentum, band and alarm."""
    panel = data["panel"]
    mask = panel["issuer_code"].eq(str(issuer_code))
    out = panel.loc[mask, ["issuer_code", "month", "month_dt", "y_pre3m"]].copy()
    if run is not None:
        out["pd_3m"] = np.asarray(run["oof"])[mask.to_numpy()]
        out["alarm"] = np.asarray(run["history_alarm"])[mask.to_numpy()]
    else:
        out["pd_3m"] = np.nan
        out["alarm"] = False
    out["momentum_3m"] = momentum(out["pd_3m"])
    out["band"] = [risk_band(float(v))[0] if np.isfinite(v) else "OK"
                   for v in out["pd_3m"].to_numpy()]
    for column in ("issuer_name", "name", "symbol", "sector", "ROE", "ROA"):
        if column in panel.columns:
            out[column] = panel.loc[mask, column].to_numpy()
    return out.reset_index(drop=True)


def latest_cross_section(data: dict[str, Any], run: dict[str, Any] | None) -> pd.DataFrame:
    """The most recent month of every issuer, ranked by PD."""
    panel = data["panel"]
    rows = np.asarray(data["latest_rows"])
    keep = [c for c in ("issuer_code", "firm_name", "issuer_name", "name", "symbol",
                        "sector", "month", "ROE", "ROA", "TDTA", "y_pre3m")
            if c in panel.columns]
    out = panel.iloc[rows][keep].copy()
    if run is not None:
        out["pd_3m"] = np.asarray(run["oof"])[rows]
        out["alarm"] = np.asarray(run["history_alarm"])[rows]
    else:
        out["pd_3m"] = np.nan
        out["alarm"] = False
    out["band"] = [risk_band(float(v))[0] if np.isfinite(v) else "OK"
                   for v in out["pd_3m"].to_numpy()]
    return out.sort_values("pd_3m", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------- self test
def self_test() -> None:
    print(f"data_layer {VERSION}")
    print(f"database   {DEFAULT_SOURCE_DB}")
    if not DEFAULT_SOURCE_DB.exists():
        raise SystemExit(f"Database not found: {DEFAULT_SOURCE_DB}")
    data, _ = load_with_banner()
    panel = data["panel"]
    assert len(panel) == len(data["X"]) == len(data["y"]), "panel and X are out of step"
    assert panel["issuer_code"].nunique() == len(data["latest_rows"]), "latest row per issuer is wrong"
    print(f"features   {len(data['features'])}")
    print(f"months     {panel['month_dt'].min().date()} to {panel['month_dt'].max().date()}")
    print(f"tables     {len(table_names())} in the database")
    available = [m for m in MODEL_NAMES
                 if not (m == 'XGBoost' and xgb is None)
                 and not (m == 'LightGBM' and LGBMClassifier is None)
                 and not (m == 'CatBoost' and CatBoostClassifier is None)]
    print("models     " + ", ".join(available))
    print("self test passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        self_test()
    else:
        load_with_banner()
