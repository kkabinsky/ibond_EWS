# -*- coding: utf-8 -*-
"""Firm-level XAI for the CatBoost model used by ``firm_shock_panel``.

The module deliberately reuses the grouped out-of-fold CatBoost models, scalers,
probabilities, and review-capacity threshold from ``firm_shock_panel``.  It does
not fit a replacement prediction model.

Three diagnostics are persisted in ``cmdf_credit.db``:

* repeated-seed LIME explanations for an issuer's latest observation;
* out-of-fold CatBoost SHAP importance across the full issuer-month panel; and
* one-standard-deviation shocks for every model feature and all issuers.

When multiple Amihud transformations are present, they form one interpretable
LIME group.  A perturbation of ``amihud_monthly`` moves the other available
transformations together using their nearest empirical training row.  A schema
that already retains only ``amihud_monthly`` uses it directly.  CatBoost, SHAP,
and shocks always retain every feature in the loaded model.

Run from the command line:

    py -3.12 lime_panel.py
    py -3.12 lime_panel.py --all-lime --seeds 10 --samples 1000
    py -3.12 lime_panel.py --issuer PRIME A
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # The GUI can still open and show previously saved results.
    LimeTabularExplainer = None

import firm_shock_panel as firm_shock


HERE = os.path.dirname(os.path.abspath(__file__))
from thaibma_paths import DATA_ROOT  # data lives outside the repo
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
XAI_VERSION = "firm_catboost_oof_lime_shap_shock_v2_schema_aware"
DEFAULT_SEEDS = (17, 29, 43, 61, 83)
DEFAULT_SAMPLES = 750
DEFAULT_LIME_LIMIT = 25

LIQUIDITY_GROUP = (
    "amihud_monthly",
    "amihud_monthly_100",
    "scaled_amihud",
    "ln_amihud",
)
LIQUIDITY_REPRESENTATIVE = LIQUIDITY_GROUP[0]
LIQUIDITY_DROPS = LIQUIDITY_GROUP[1:]

T_RUN = "firm_xai_run"
T_ISSUER = "firm_xai_issuer"
T_LIME = "firm_xai_lime"
T_SHAP = "firm_xai_shap_global"
T_SHOCK = "firm_xai_shock_33"


def _available_liquidity_group(columns):
    """Return linked Amihud columns that exist in the loaded model schema."""
    available = tuple(name for name in LIQUIDITY_GROUP if name in set(columns))
    if LIQUIDITY_REPRESENTATIVE not in available:
        return ()
    return available


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sigmoid(value):
    value = np.asarray(value, dtype=float)
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)),
                    np.exp(value) / (1.0 + np.exp(value)))


def _create_schema(con):
    """Create append-only XAI result tables on an existing SQLite connection."""
    con.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {T_RUN} (
            run_id TEXT PRIMARY KEY,
            run_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            xai_version TEXT NOT NULL,
            source_table TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_scope TEXT NOT NULL,
            target_name TEXT NOT NULL,
            workload REAL NOT NULL,
            threshold REAL,
            seed INTEGER NOT NULL,
            lime_seeds TEXT NOT NULL,
            lime_n_runs INTEGER NOT NULL,
            lime_samples INTEGER NOT NULL,
            lime_limit INTEGER NOT NULL,
            n_lime_issuers INTEGER DEFAULT 0,
            shock_sd REAL NOT NULL,
            n_rows INTEGER,
            n_issuers INTEGER,
            n_positive_rows INTEGER,
            n_positive_issuers INTEGER,
            n_model_features INTEGER,
            n_lime_features INTEGER,
            oof_auc REAL,
            shap_additivity_max_error REAL,
            shap_pd_max_error REAL,
            liquidity_group_definition TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS {T_ISSUER} (
            run_id TEXT NOT NULL,
            issuer_code TEXT NOT NULL,
            issuer_name TEXT,
            observation_month TEXT,
            source_row INTEGER,
            fold_id INTEGER,
            model_pd REAL,
            threshold REAL,
            margin REAL,
            percentile REAL,
            status TEXT,
            event_observed INTEGER,
            target_at_observation INTEGER,
            lime_available INTEGER DEFAULT 0,
            lime_fidelity_mean REAL,
            lime_fidelity_sd REAL,
            lime_surrogate_pd_mean REAL,
            lime_surrogate_pd_sd REAL,
            top_lime_risk_feature TEXT,
            top_lime_risk_weight REAL,
            top_lime_protective_feature TEXT,
            top_lime_protective_weight REAL,
            PRIMARY KEY (run_id, issuer_code),
            FOREIGN KEY (run_id) REFERENCES {T_RUN}(run_id)
        );

        CREATE TABLE IF NOT EXISTS {T_LIME} (
            run_id TEXT NOT NULL,
            issuer_code TEXT NOT NULL,
            issuer_name TEXT,
            observation_month TEXT,
            feature_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            grouped_members TEXT,
            feature_value REAL,
            weight_mean REAL,
            weight_sd REAL,
            weight_p05 REAL,
            weight_p95 REAL,
            weight_min REAL,
            weight_max REAL,
            positive_share REAL,
            sign_stability REAL,
            unstable_sign INTEGER,
            lime_rank INTEGER,
            n_seed_runs INTEGER,
            model_pd REAL,
            threshold REAL,
            fidelity_mean REAL,
            fidelity_sd REAL,
            fidelity_min REAL,
            fidelity_max REAL,
            surrogate_pd_mean REAL,
            surrogate_pd_sd REAL,
            PRIMARY KEY (run_id, issuer_code, feature_name),
            FOREIGN KEY (run_id, issuer_code)
                REFERENCES {T_ISSUER}(run_id, issuer_code)
        );

        CREATE TABLE IF NOT EXISTS {T_SHAP} (
            run_id TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            shap_rank INTEGER NOT NULL,
            mean_abs_shap_log_odds REAL,
            mean_shap_log_odds REAL,
            sd_shap_log_odds REAL,
            positive_share REAL,
            n_oof_rows INTEGER,
            PRIMARY KEY (run_id, feature_name),
            FOREIGN KEY (run_id) REFERENCES {T_RUN}(run_id)
        );

        CREATE TABLE IF NOT EXISTS {T_SHOCK} (
            run_id TEXT NOT NULL,
            issuer_code TEXT NOT NULL,
            issuer_name TEXT,
            observation_month TEXT,
            feature_name TEXT NOT NULL,
            feature_value REAL,
            feature_sd REAL,
            shock_size_sd REAL,
            lower_value REAL,
            upper_value REAL,
            baseline_pd REAL,
            lower_pd REAL,
            upper_pd REAL,
            adverse_direction TEXT,
            adverse_value REAL,
            adverse_pd REAL,
            adverse_delta_pd REAL,
            adverse_delta_pp REAL,
            protective_direction TEXT,
            protective_pd REAL,
            protective_delta_pd REAL,
            raises_risk INTEGER,
            shock_rank INTEGER,
            PRIMARY KEY (run_id, issuer_code, feature_name),
            FOREIGN KEY (run_id, issuer_code)
                REFERENCES {T_ISSUER}(run_id, issuer_code)
        );

        CREATE INDEX IF NOT EXISTS idx_firm_xai_run_status
            ON {T_RUN}(status, completed_at);
        CREATE INDEX IF NOT EXISTS idx_firm_xai_issuer_pd
            ON {T_ISSUER}(run_id, model_pd DESC);
        CREATE INDEX IF NOT EXISTS idx_firm_xai_lime_rank
            ON {T_LIME}(run_id, issuer_code, lime_rank);
        CREATE INDEX IF NOT EXISTS idx_firm_xai_shock_rank
            ON {T_SHOCK}(run_id, issuer_code, shock_rank);
        """
    )


def ensure_schema(db=DB):
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA foreign_keys=ON")
        _create_schema(con)


def _issuer_anchors(state):
    panel = state["panel"]
    rows = np.asarray(state["last_rows"], dtype=int)
    name_col = next((c for c in ("issuer_name", "firm_name", "company_name")
                     if c in panel.columns), None)
    names = (panel.loc[rows, name_col].fillna("").astype(str).to_numpy()
             if name_col else np.repeat("", len(rows)))
    issuers = panel.loc[rows, "issuer_code"].astype(str).to_numpy()
    pd_now = np.asarray(state["oof"], dtype=float)[rows]
    threshold = float(state["thr"])
    cross = np.asarray(state["cross"], dtype=float)
    percentiles = np.array([(cross < value).mean() * 100.0 for value in pd_now])
    workload = float(state["workload"])
    status = np.where(pd_now >= threshold, "HIGH RISK",
                      np.where(percentiles >= 100 * (1 - workload) - 2,
                               "WATCH", "OK"))

    event_by_issuer = (panel.assign(_y=np.asarray(state["y"], dtype=int))
                       .groupby("issuer_code")["_y"].max())
    out = pd.DataFrame({
        "issuer_code": issuers,
        "issuer_name": names,
        "observation_month": panel.loc[rows, "month"].astype(str).to_numpy(),
        "source_row": rows,
        "fold_id": np.asarray(state["fold_of"], dtype=int)[rows],
        "model_pd": pd_now,
        "threshold": threshold,
        "margin": threshold - pd_now,
        "percentile": percentiles,
        "status": status,
        "event_observed": [int(event_by_issuer.get(code, 0)) for code in issuers],
        "target_at_observation": np.asarray(state["y"], dtype=int)[rows],
        "lime_available": 0,
        "lime_fidelity_mean": np.nan,
        "lime_fidelity_sd": np.nan,
        "lime_surrogate_pd_mean": np.nan,
        "lime_surrogate_pd_sd": np.nan,
        "top_lime_risk_feature": None,
        "top_lime_risk_weight": np.nan,
        "top_lime_protective_feature": None,
        "top_lime_protective_weight": np.nan,
    })
    return out.sort_values("model_pd", ascending=False).reset_index(drop=True)


def compute_oof_shap(state, verbose=True):
    """Aggregate native CatBoost SHAP values from each issuer-held-out fold."""
    from catboost import Pool

    cols = list(state["cols"])
    p = len(cols)
    total = np.zeros(p)
    total_abs = np.zeros(p)
    total_sq = np.zeros(p)
    positive = np.zeros(p)
    n = 0
    max_additivity_error = 0.0
    max_probability_error = 0.0

    for fold, (scaler, model) in sorted(state["fold_models"].items()):
        mask = np.asarray(state["fold_of"]) == fold
        if not mask.any():
            continue
        transformed = scaler.transform(state["A"][mask])
        pool = Pool(transformed)
        values = np.asarray(model.get_feature_importance(pool, type="ShapValues"),
                            dtype=float)
        shap = values[:, :-1]
        base = values[:, -1]
        raw = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"),
                         dtype=float).reshape(-1)
        proba = np.asarray(model.predict_proba(transformed), dtype=float)[:, 1]
        max_additivity_error = max(
            max_additivity_error,
            float(np.max(np.abs(base + shap.sum(axis=1) - raw))),
        )
        max_probability_error = max(
            max_probability_error,
            float(np.max(np.abs(_sigmoid(base + shap.sum(axis=1)) - proba))),
        )
        total += shap.sum(axis=0)
        total_abs += np.abs(shap).sum(axis=0)
        total_sq += np.square(shap).sum(axis=0)
        positive += (shap > 0).sum(axis=0)
        n += len(shap)
        if verbose:
            print(f"  SHAP fold {fold}: {len(shap):,} held-out rows")

    if n == 0:
        raise RuntimeError("No held-out rows were available for SHAP.")
    mean = total / n
    variance = np.maximum(total_sq / n - np.square(mean), 0.0)
    out = pd.DataFrame({
        "feature_name": cols,
        "mean_abs_shap_log_odds": total_abs / n,
        "mean_shap_log_odds": mean,
        "sd_shap_log_odds": np.sqrt(variance),
        "positive_share": positive / n,
        "n_oof_rows": n,
    })
    out = out.sort_values("mean_abs_shap_log_odds", ascending=False).reset_index(drop=True)
    out["shap_rank"] = np.arange(1, len(out) + 1)
    return out, max_additivity_error, max_probability_error


def compute_feature_shocks(state, anchors, shock_sd=1.0, verbose=True):
    """Score plus/minus ``shock_sd`` for every feature on every latest issuer row."""
    cols = list(state["cols"])
    p = len(cols)
    records = []

    for fold, (scaler, model) in sorted(state["fold_models"].items()):
        current = anchors[anchors["fold_id"] == fold]
        if current.empty:
            continue
        rows = current["source_row"].astype(int).to_numpy()
        x0 = np.asarray(state["A"], dtype=float)[rows]
        train_mask = ((np.asarray(state["fold_of"]) >= 0)
                      & (np.asarray(state["fold_of"]) != fold))
        train = np.asarray(state["A"], dtype=float)[train_mask]
        sd = np.nanstd(train, axis=0, ddof=1)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.0)
        q01 = np.nanquantile(train, 0.01, axis=0)
        q99 = np.nanquantile(train, 0.99, axis=0)

        m = len(x0)
        feature_index = np.tile(np.arange(p), m)
        flat_index = np.arange(m * p)
        lower = np.repeat(x0, p, axis=0)
        upper = lower.copy()
        lower[flat_index, feature_index] = np.maximum(
            lower[flat_index, feature_index] - shock_sd * sd[feature_index],
            q01[feature_index],
        )
        upper[flat_index, feature_index] = np.minimum(
            upper[flat_index, feature_index] + shock_sd * sd[feature_index],
            q99[feature_index],
        )
        baseline = np.asarray(model.predict_proba(scaler.transform(x0)))[:, 1]
        lower_pd = np.asarray(model.predict_proba(scaler.transform(lower)))[:, 1]
        upper_pd = np.asarray(model.predict_proba(scaler.transform(upper)))[:, 1]

        for issuer_pos, (_, issuer) in enumerate(current.reset_index(drop=True).iterrows()):
            start = issuer_pos * p
            for j, feature in enumerate(cols):
                lo_pd = float(lower_pd[start + j])
                hi_pd = float(upper_pd[start + j])
                base_pd = float(baseline[issuer_pos])
                upper_is_adverse = hi_pd >= lo_pd
                adverse_pd = hi_pd if upper_is_adverse else lo_pd
                protective_pd = lo_pd if upper_is_adverse else hi_pd
                records.append({
                    "issuer_code": str(issuer["issuer_code"]),
                    "issuer_name": str(issuer.get("issuer_name", "") or ""),
                    "observation_month": str(issuer["observation_month"]),
                    "feature_name": feature,
                    "feature_value": float(x0[issuer_pos, j]),
                    "feature_sd": float(sd[j]),
                    "shock_size_sd": float(shock_sd),
                    "lower_value": float(lower[start + j, j]),
                    "upper_value": float(upper[start + j, j]),
                    "baseline_pd": base_pd,
                    "lower_pd": lo_pd,
                    "upper_pd": hi_pd,
                    "adverse_direction": "+SD" if upper_is_adverse else "-SD",
                    "adverse_value": float(upper[start + j, j] if upper_is_adverse
                                             else lower[start + j, j]),
                    "adverse_pd": adverse_pd,
                    "adverse_delta_pd": adverse_pd - base_pd,
                    "adverse_delta_pp": (adverse_pd - base_pd) * 100.0,
                    "protective_direction": "-SD" if upper_is_adverse else "+SD",
                    "protective_pd": protective_pd,
                    "protective_delta_pd": protective_pd - base_pd,
                    "raises_risk": int(adverse_pd > base_pd + 1e-12),
                })
        if verbose:
            print(f"  shocks fold {fold}: {len(current):,} issuers x {p} features")

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["shock_rank"] = (out.groupby("issuer_code")["adverse_delta_pd"]
                         .rank(method="first", ascending=False).astype(int))
    return out


class _ReducedFoldPredictor:
    """Schema-aware LIME view wrapped around the unchanged CatBoost model."""

    def __init__(self, state, fold):
        self.state = state
        self.fold = int(fold)
        self.cols = list(state["cols"])
        self.group_members = _available_liquidity_group(self.cols)
        if not self.group_members:
            raise ValueError(
                "The loaded model schema does not contain amihud_monthly, "
                "which is required as the LIME liquidity representative.")
        self.group_drops = self.group_members[1:]
        self.kept = [c for c in self.cols if c not in self.group_drops]
        self.kept_index = np.array([self.cols.index(c) for c in self.kept], dtype=int)
        self.rep_reduced_index = self.kept.index(LIQUIDITY_REPRESENTATIVE)
        self.rep_model_index = self.cols.index(LIQUIDITY_REPRESENTATIVE)
        self.group_index = np.array([self.cols.index(c) for c in self.group_members],
                                    dtype=int)
        fold_of = np.asarray(state["fold_of"])
        self.train_mask = (fold_of >= 0) & (fold_of != self.fold)
        self.train = np.asarray(state["A"], dtype=float)[self.train_mask]
        self.training_reduced = self.train[:, self.kept_index]
        self.low = np.nanquantile(self.training_reduced, 0.005, axis=0)
        self.high = np.nanquantile(self.training_reduced, 0.995, axis=0)
        rep = self.train[:, self.rep_model_index]
        self.rep_order = np.argsort(rep)
        self.rep_sorted = rep[self.rep_order]
        self.group_train_sorted = self.train[self.rep_order][:, self.group_index]
        self.scaler, self.model = state["fold_models"][self.fold]
        self.anchor_reduced = None
        self.anchor_group = None

    def reduced_row(self, model_row):
        return np.asarray(model_row, dtype=float)[self.kept_index]

    def set_anchor(self, model_row):
        model_row = np.asarray(model_row, dtype=float)
        self.anchor_reduced = model_row[self.kept_index].copy()
        self.anchor_group = model_row[self.group_index].copy()

    def _nearest_group(self, values):
        pos = np.searchsorted(self.rep_sorted, values, side="left")
        right = np.clip(pos, 0, len(self.rep_sorted) - 1)
        left = np.clip(pos - 1, 0, len(self.rep_sorted) - 1)
        use_right = np.abs(self.rep_sorted[right] - values) < np.abs(
            self.rep_sorted[left] - values)
        chosen = np.where(use_right, right, left)
        return self.group_train_sorted[chosen]

    def reconstruct(self, reduced):
        reduced = np.atleast_2d(np.asarray(reduced, dtype=float))
        if self.anchor_reduced is not None:
            is_anchor = np.all(np.isclose(
                reduced, self.anchor_reduced, rtol=1e-11, atol=1e-12), axis=1)
        else:
            is_anchor = np.zeros(len(reduced), dtype=bool)
        clipped = np.clip(reduced, self.low, self.high)
        if is_anchor.any():
            clipped[is_anchor] = self.anchor_reduced
        model_matrix = np.empty((len(clipped), len(self.cols)), dtype=float)
        model_matrix[:, self.kept_index] = clipped
        rep_values = clipped[:, self.rep_reduced_index]
        model_matrix[:, self.group_index] = self._nearest_group(rep_values)
        if self.anchor_reduced is not None:
            anchor_rows = np.flatnonzero(is_anchor)
            if len(anchor_rows):
                model_matrix[np.ix_(anchor_rows, self.group_index)] = self.anchor_group
        model_matrix[:, self.rep_model_index] = rep_values
        return model_matrix

    def predict_proba(self, reduced):
        model_matrix = self.reconstruct(reduced)
        return np.asarray(self.model.predict_proba(
            self.scaler.transform(model_matrix)), dtype=float)


def _summarize_lime_runs(raw, anchors):
    if raw.empty:
        return raw, pd.DataFrame()
    keys = ["issuer_code", "feature_name"]
    grouped = raw.groupby(keys, sort=False)
    out = grouped.agg(
        issuer_name=("issuer_name", "first"),
        observation_month=("observation_month", "first"),
        display_name=("display_name", "first"),
        grouped_members=("grouped_members", "first"),
        feature_value=("feature_value", "first"),
        weight_mean=("weight", "mean"),
        weight_sd=("weight", "std"),
        weight_min=("weight", "min"),
        weight_max=("weight", "max"),
        positive_share=("weight", lambda x: float((x > 0).mean())),
        n_seed_runs=("seed", "nunique"),
        model_pd=("model_pd", "first"),
        threshold=("threshold", "first"),
        fidelity_mean=("fidelity", "mean"),
        fidelity_sd=("fidelity", "std"),
        fidelity_min=("fidelity", "min"),
        fidelity_max=("fidelity", "max"),
        surrogate_pd_mean=("surrogate_pd", "mean"),
        surrogate_pd_sd=("surrogate_pd", "std"),
    ).reset_index()
    quantiles = grouped["weight"].quantile([0.05, 0.95]).unstack()
    quantiles.columns = ["weight_p05", "weight_p95"]
    out = out.merge(quantiles.reset_index(), on=keys, how="left")
    out["weight_sd"] = out["weight_sd"].fillna(0.0)
    out["fidelity_sd"] = out["fidelity_sd"].fillna(0.0)
    out["surrogate_pd_sd"] = out["surrogate_pd_sd"].fillna(0.0)
    out["sign_stability"] = np.maximum(out["positive_share"],
                                        1.0 - out["positive_share"])
    out["unstable_sign"] = (out["sign_stability"] < 0.80).astype(int)
    out["lime_rank"] = (out.assign(_abs=out["weight_mean"].abs())
                        .groupby("issuer_code")["_abs"]
                        .rank(method="first", ascending=False).astype(int))

    summary_rows = []
    for issuer_code, group in out.groupby("issuer_code", sort=False):
        risk = group[group["weight_mean"] > 0].sort_values("weight_mean", ascending=False)
        protective = group[group["weight_mean"] < 0].sort_values("weight_mean")
        first = group.iloc[0]
        summary_rows.append({
            "issuer_code": issuer_code,
            "lime_available": 1,
            "lime_fidelity_mean": float(first["fidelity_mean"]),
            "lime_fidelity_sd": float(first["fidelity_sd"]),
            "lime_surrogate_pd_mean": float(first["surrogate_pd_mean"]),
            "lime_surrogate_pd_sd": float(first["surrogate_pd_sd"]),
            "top_lime_risk_feature": (str(risk.iloc[0]["display_name"])
                                      if not risk.empty else None),
            "top_lime_risk_weight": (float(risk.iloc[0]["weight_mean"])
                                     if not risk.empty else np.nan),
            "top_lime_protective_feature": (str(protective.iloc[0]["display_name"])
                                            if not protective.empty else None),
            "top_lime_protective_weight": (float(protective.iloc[0]["weight_mean"])
                                           if not protective.empty else np.nan),
        })
    return out, pd.DataFrame(summary_rows)


def compute_repeated_lime(state, anchors, issuer_codes=None,
                          seeds=DEFAULT_SEEDS, samples=DEFAULT_SAMPLES,
                          verbose=True):
    """Run official LIME repeatedly and retain variation across random seeds."""
    if LimeTabularExplainer is None:
        raise RuntimeError(
            "LIME is not installed. Run: py -3.12 -m pip install lime==0.2.0.1")
    seeds = tuple(int(value) for value in seeds)
    if len(seeds) < 2:
        raise ValueError("At least two LIME seeds are required to measure variability.")
    chosen = anchors.copy()
    if issuer_codes is not None:
        wanted = {str(value) for value in issuer_codes}
        chosen = chosen[chosen["issuer_code"].astype(str).isin(wanted)]
    if chosen.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw_records = []
    for fold in sorted(chosen["fold_id"].dropna().astype(int).unique()):
        fold_rows = chosen[chosen["fold_id"] == fold]
        predictor = _ReducedFoldPredictor(state, fold)
        display_names = list(predictor.kept)
        if len(predictor.group_members) > 1:
            display_names[predictor.rep_reduced_index] = (
                f"amihud_monthly [{len(predictor.group_members)} linked "
                "liquidity measures]")
            grouped_members = ", ".join(predictor.group_members)
        else:
            grouped_members = ""

        for seed in seeds:
            explainer = LimeTabularExplainer(
                predictor.training_reduced,
                mode="classification",
                feature_names=display_names,
                class_names=["Non-default", "Default"],
                feature_selection="none",
                discretize_continuous=False,
                sample_around_instance=True,
                random_state=int(seed),
            )
            for _, issuer in fold_rows.iterrows():
                source_row = int(issuer["source_row"])
                predictor.set_anchor(state["A"][source_row])
                x_reduced = predictor.reduced_row(state["A"][source_row])
                model_pd = float(predictor.predict_proba(x_reduced)[0, 1])
                if abs(model_pd - float(issuer["model_pd"])) > 1e-8:
                    raise AssertionError(
                        f"LIME predictor left the OOF probability scale for "
                        f"{issuer['issuer_code']}.")
                explanation = explainer.explain_instance(
                    x_reduced,
                    predictor.predict_proba,
                    labels=(1,),
                    num_features=len(predictor.kept),
                    num_samples=int(samples),
                )
                weights = dict(explanation.local_exp.get(1, []))
                local_pred = np.asarray(explanation.local_pred).reshape(-1)
                surrogate_pd = float(local_pred[0]) if len(local_pred) else np.nan
                for j, feature in enumerate(predictor.kept):
                    is_group = (feature == LIQUIDITY_REPRESENTATIVE
                                and len(predictor.group_members) > 1)
                    raw_records.append({
                        "issuer_code": str(issuer["issuer_code"]),
                        "issuer_name": str(issuer.get("issuer_name", "") or ""),
                        "observation_month": str(issuer["observation_month"]),
                        "feature_name": feature,
                        "display_name": display_names[j],
                        "grouped_members": grouped_members if is_group else "",
                        "feature_value": float(x_reduced[j]),
                        "seed": int(seed),
                        "weight": float(weights.get(j, 0.0)),
                        "model_pd": model_pd,
                        "threshold": float(state["thr"]),
                        "fidelity": float(explanation.score),
                        "surrogate_pd": surrogate_pd,
                    })
        if verbose:
            print(f"  LIME fold {fold}: {len(fold_rows):,} issuers x "
                  f"{len(seeds)} seeds x {int(samples):,} samples")

    raw = pd.DataFrame(raw_records)
    return _summarize_lime_runs(raw, anchors)


def _merge_lime_summary(anchors, summary):
    if summary.empty:
        return anchors
    fields = [c for c in summary.columns if c != "issuer_code"]
    merged = anchors.merge(summary, on="issuer_code", how="left", suffixes=("", "_new"))
    for field in fields:
        new = f"{field}_new"
        if new in merged.columns:
            merged[field] = merged[new].combine_first(merged[field])
            merged.drop(columns=new, inplace=True)
    merged["lime_available"] = merged["lime_available"].fillna(0).astype(int)
    return merged


def _insert_started_run(db, row):
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA foreign_keys=ON")
        _create_schema(con)
        pd.DataFrame([row]).to_sql(T_RUN, con, if_exists="append", index=False)


def _mark_failed(db, run_id, message):
    with sqlite3.connect(db) as con:
        con.execute(
            f"UPDATE {T_RUN} SET status='failed', completed_at=?, notes=? "
            "WHERE run_id=?",
            (_now_iso(), str(message)[:2000], run_id),
        )


def run_pipeline(db=DB, workload=firm_shock.DEFAULT_WORKLOAD,
                 seeds=DEFAULT_SEEDS, samples=DEFAULT_SAMPLES,
                 lime_limit=DEFAULT_LIME_LIMIT, lime_issuers=None,
                 shock_sd=1.0, force_model=False, verbose=True):
    """Build SHAP/shock tables for all issuers and repeated LIME for a subset."""
    started = time.perf_counter()
    seeds = tuple(int(value) for value in seeds)
    if len(seeds) < 2:
        raise ValueError("Use at least two seeds for repeated LIME.")
    state = firm_shock.load_state(float(workload), force=bool(force_model))
    anchors = _issuer_anchors(state)
    if lime_issuers:
        selected = [str(value) for value in lime_issuers]
    elif int(lime_limit) <= 0:
        selected = anchors["issuer_code"].astype(str).tolist()
    else:
        selected = anchors.head(int(lime_limit))["issuer_code"].astype(str).tolist()

    run_id = (datetime.now().strftime("firm_xai_%Y%m%dT%H%M%S_")
              + uuid.uuid4().hex[:8])
    y = np.asarray(state["y"], dtype=int)
    groups = state["panel"]["issuer_code"].astype(str).to_numpy()
    valid = np.isfinite(np.asarray(state["oof"], dtype=float))
    oof_auc = (float(roc_auc_score(y[valid], np.asarray(state["oof"])[valid]))
               if valid.any() and 0 < y[valid].sum() < valid.sum() else np.nan)
    liquidity_members = _available_liquidity_group(state["cols"])
    if not liquidity_members:
        raise ValueError(
            "The loaded model schema does not contain amihud_monthly.")
    liquidity_drops = liquidity_members[1:]
    if liquidity_drops:
        liquidity_note = (
            f"LIME links {len(liquidity_members)} available Amihud columns. ")
    else:
        liquidity_note = (
            "Only amihud_monthly is present, so LIME uses it directly. ")
    run_row = {
        "run_id": run_id,
        "run_at": _now_iso(),
        "completed_at": None,
        "status": "running",
        "xai_version": XAI_VERSION,
        "source_table": "ibond_33features_panel",
        "model_name": "CatBoostClassifier",
        "model_scope": "5-fold StratifiedGroupKFold; issuer-held-out scoring",
        "target_name": "real payment default within the stored 3-month event window",
        "workload": float(workload),
        "threshold": float(state["thr"]),
        "seed": int(firm_shock.SEED),
        "lime_seeds": json.dumps(list(seeds)),
        "lime_n_runs": len(seeds),
        "lime_samples": int(samples),
        "lime_limit": int(lime_limit),
        "n_lime_issuers": 0,
        "shock_sd": float(shock_sd),
        "n_rows": int(len(y)),
        "n_issuers": int(pd.Series(groups).nunique()),
        "n_positive_rows": int(y.sum()),
        "n_positive_issuers": int(pd.Series(groups[y == 1]).nunique()),
        "n_model_features": int(len(state["cols"])),
        "n_lime_features": int(len(state["cols"]) - len(liquidity_drops)),
        "oof_auc": oof_auc,
        "shap_additivity_max_error": None,
        "shap_pd_max_error": None,
        "liquidity_group_definition": ", ".join(liquidity_members),
        "notes": ("LIME uses repeated seeds and empirical 0.5%-99.5% clipping. "
                  + liquidity_note + "SHAP values are "
                  "native CatBoost log-odds contributions. Shocks are one feature "
                  "at a time and are not causal effects."),
    }
    _insert_started_run(db, run_row)

    try:
        if verbose:
            print(f"Firm XAI run {run_id}")
            print(f"  source: {len(y):,} issuer-months, "
                  f"{anchors['issuer_code'].nunique():,} issuers, "
                  f"{int(y.sum())} positive months / "
                  f"{pd.Series(groups[y == 1]).nunique()} issuers")
            print(f"  OOF AUC {oof_auc:.6f} | threshold {state['thr']:.8f}")

        shap, shap_error, shap_pd_error = compute_oof_shap(state, verbose=verbose)
        shocks = compute_feature_shocks(state, anchors, shock_sd=shock_sd,
                                         verbose=verbose)
        lime, lime_summary = compute_repeated_lime(
            state, anchors, issuer_codes=selected, seeds=seeds,
            samples=samples, verbose=verbose)
        anchors = _merge_lime_summary(anchors, lime_summary)

        shap.insert(0, "run_id", run_id)
        shocks.insert(0, "run_id", run_id)
        if not lime.empty:
            lime.insert(0, "run_id", run_id)
        anchors.insert(0, "run_id", run_id)

        with sqlite3.connect(db) as con:
            con.execute("PRAGMA foreign_keys=ON")
            anchors.to_sql(T_ISSUER, con, if_exists="append", index=False)
            shap.to_sql(T_SHAP, con, if_exists="append", index=False)
            shocks.to_sql(T_SHOCK, con, if_exists="append", index=False)
            if not lime.empty:
                lime.to_sql(T_LIME, con, if_exists="append", index=False)
            con.execute(
                f"UPDATE {T_RUN} SET status='completed', completed_at=?, "
                "n_lime_issuers=?, shap_additivity_max_error=?, "
                "shap_pd_max_error=? WHERE run_id=?",
                (_now_iso(), int(lime["issuer_code"].nunique()) if not lime.empty else 0,
                 float(shap_error), float(shap_pd_error), run_id),
            )
        elapsed = time.perf_counter() - started
        if verbose:
            print(f"  saved {len(shap):,} SHAP rows, {len(shocks):,} shock rows, "
                  f"{len(lime):,} LIME rows")
            print(f"  SHAP additivity max error {shap_error:.3e}; "
                  f"probability max error {shap_pd_error:.3e}")
            print(f"  completed in {elapsed:.1f} seconds")
        return {
            "run_id": run_id,
            "elapsed_seconds": elapsed,
            "oof_auc": oof_auc,
            "threshold": float(state["thr"]),
            "n_issuers": int(len(anchors)),
            "n_lime_issuers": int(lime["issuer_code"].nunique()) if not lime.empty else 0,
            "n_lime_rows": int(len(lime)),
            "n_shap_rows": int(len(shap)),
            "n_shock_rows": int(len(shocks)),
            "shap_additivity_max_error": float(shap_error),
            "shap_pd_max_error": float(shap_pd_error),
        }
    except Exception as exc:
        _mark_failed(db, run_id, exc)
        raise


def latest_run(db=DB):
    ensure_schema(db)
    with sqlite3.connect(db) as con:
        row = pd.read_sql_query(
            f"SELECT * FROM {T_RUN} WHERE status='completed' "
            "ORDER BY completed_at DESC LIMIT 1",
            con,
        )
    return None if row.empty else row.iloc[0].to_dict()


def load_issuer_index(db=DB, run_id=None):
    run = latest_run(db) if run_id is None else {"run_id": run_id}
    if not run:
        return pd.DataFrame()
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(
            f"SELECT * FROM {T_ISSUER} WHERE run_id=? "
            "ORDER BY model_pd DESC, issuer_code",
            con, params=(run["run_id"],),
        )


def load_issuer_explanation(db=DB, issuer_code=None, run_id=None):
    run = latest_run(db) if run_id is None else {"run_id": run_id}
    if not run or not issuer_code:
        return pd.DataFrame(), None
    with sqlite3.connect(db) as con:
        issuer = pd.read_sql_query(
            f"SELECT * FROM {T_ISSUER} WHERE run_id=? AND issuer_code=?",
            con, params=(run["run_id"], str(issuer_code)),
        )
        rows = pd.read_sql_query(
            f"""
            SELECT s.feature_name, s.feature_value, s.feature_sd,
                   s.shock_size_sd, s.lower_value, s.upper_value,
                   s.baseline_pd, s.lower_pd, s.upper_pd,
                   s.adverse_direction, s.adverse_pd, s.adverse_delta_pp,
                   s.raises_risk, s.shock_rank,
                   g.shap_rank, g.mean_abs_shap_log_odds,
                   g.mean_shap_log_odds, g.sd_shap_log_odds,
                   g.positive_share AS shap_positive_share,
                   l.display_name, l.grouped_members,
                   l.weight_mean AS lime_weight_mean,
                   l.weight_sd AS lime_weight_sd,
                   l.weight_p05 AS lime_weight_p05,
                   l.weight_p95 AS lime_weight_p95,
                   l.positive_share AS lime_positive_share,
                   l.sign_stability AS lime_sign_stability,
                   l.unstable_sign AS lime_unstable_sign,
                   l.lime_rank, l.n_seed_runs,
                   l.fidelity_mean AS lime_fidelity_mean,
                   CASE WHEN s.feature_name IN (?, ?, ?)
                        THEN 'Grouped into amihud_monthly for LIME'
                        ELSE '' END AS lime_group_note
            FROM {T_SHOCK} s
            JOIN {T_SHAP} g
              ON g.run_id=s.run_id AND g.feature_name=s.feature_name
            LEFT JOIN {T_LIME} l
              ON l.run_id=s.run_id AND l.issuer_code=s.issuer_code
             AND l.feature_name=s.feature_name
            WHERE s.run_id=? AND s.issuer_code=?
            ORDER BY COALESCE(l.lime_rank, 999), s.shock_rank
            """,
            con,
            params=(*LIQUIDITY_DROPS, run["run_id"], str(issuer_code)),
        )
    summary = None if issuer.empty else issuer.iloc[0].to_dict()
    return rows, summary


def ensure_issuer_lime(db=DB, issuer_code=None, run_id=None, verbose=True):
    """Compute repeated LIME for one issuer missing from an existing XAI run."""
    run = latest_run(db) if run_id is None else None
    if run_id is not None:
        ensure_schema(db)
        with sqlite3.connect(db) as con:
            d = pd.read_sql_query(f"SELECT * FROM {T_RUN} WHERE run_id=?", con,
                                  params=(run_id,))
        run = None if d.empty else d.iloc[0].to_dict()
    if not run:
        raise RuntimeError("No completed Firm XAI run is available.")
    issuer_code = str(issuer_code or "")
    with sqlite3.connect(db) as con:
        count = con.execute(
            f"SELECT COUNT(*) FROM {T_LIME} WHERE run_id=? AND issuer_code=?",
            (run["run_id"], issuer_code),
        ).fetchone()[0]
    if count:
        return int(count)

    state = firm_shock.load_state(float(run["workload"]))
    stored_feature_count = int(run.get("n_model_features") or 0)
    current_feature_count = len(state["cols"])
    if stored_feature_count and stored_feature_count != current_feature_count:
        raise RuntimeError(
            f"Stored XAI run uses {stored_feature_count} model features, but the "
            f"current model uses {current_feature_count}. Run a new Firm XAI "
            "pipeline before computing on-demand LIME; results from different "
            "schemas must not be mixed.")
    anchors = _issuer_anchors(state)
    if issuer_code not in set(anchors["issuer_code"].astype(str)):
        raise KeyError(f"Issuer {issuer_code} is not in ibond_33features_panel.")
    seeds = tuple(int(value) for value in json.loads(run["lime_seeds"]))
    lime, summary = compute_repeated_lime(
        state, anchors, issuer_codes=[issuer_code], seeds=seeds,
        samples=int(run["lime_samples"]), verbose=verbose)
    if lime.empty:
        raise RuntimeError(f"LIME produced no rows for {issuer_code}.")
    lime.insert(0, "run_id", run["run_id"])
    values = summary.iloc[0].to_dict()
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA foreign_keys=ON")
        lime.to_sql(T_LIME, con, if_exists="append", index=False)
        con.execute(
            f"""UPDATE {T_ISSUER}
                SET lime_available=1, lime_fidelity_mean=?, lime_fidelity_sd=?,
                    lime_surrogate_pd_mean=?, lime_surrogate_pd_sd=?,
                    top_lime_risk_feature=?, top_lime_risk_weight=?,
                    top_lime_protective_feature=?, top_lime_protective_weight=?
                WHERE run_id=? AND issuer_code=?""",
            (values["lime_fidelity_mean"], values["lime_fidelity_sd"],
             values["lime_surrogate_pd_mean"], values["lime_surrogate_pd_sd"],
             values["top_lime_risk_feature"], values["top_lime_risk_weight"],
             values["top_lime_protective_feature"],
             values["top_lime_protective_weight"], run["run_id"], issuer_code),
        )
        con.execute(
            f"UPDATE {T_RUN} SET n_lime_issuers=(SELECT COUNT(*) FROM {T_ISSUER} "
            "WHERE run_id=? AND lime_available=1) WHERE run_id=?",
            (run["run_id"], run["run_id"]),
        )
    return int(len(lime))


def _seed_values(first_seed, count):
    return tuple(int(first_seed) + i * 13 for i in range(int(count)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Repeated-seed LIME, OOF SHAP, and 33-feature shocks for Firm Shock")
    parser.add_argument("--db", default=DB, help="SQLite database path")
    parser.add_argument("--workload", type=float, default=firm_shock.DEFAULT_WORKLOAD)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                        help="number of repeated LIME seeds (minimum 2)")
    parser.add_argument("--first-seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="LIME perturbations per seed and issuer")
    parser.add_argument("--lime-limit", type=int, default=DEFAULT_LIME_LIMIT,
                        help="precompute the highest-risk N issuers")
    parser.add_argument("--all-lime", action="store_true",
                        help="precompute repeated LIME for all issuers")
    parser.add_argument("--issuer", nargs="*",
                        help="precompute repeated LIME for these issuer codes")
    parser.add_argument("--shock-sd", type=float, default=1.0)
    parser.add_argument("--force-model", action="store_true")
    args = parser.parse_args(argv)
    result = run_pipeline(
        db=os.path.abspath(args.db), workload=args.workload,
        seeds=_seed_values(args.first_seed, args.seeds), samples=args.samples,
        lime_limit=0 if args.all_lime else args.lime_limit,
        lime_issuers=args.issuer, shock_sd=args.shock_sd,
        force_model=args.force_model, verbose=True)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


if __name__ == "__main__":
    main()
