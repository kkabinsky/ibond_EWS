import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# NOTE:
# This file used to contain a third, unrelated "Time Series + Koopman" mode that required
# close prices (cp/close) and deep learning dependencies (torch, ta).
# That mode has been intentionally removed. This script now supports only:
#   (2) Tabular feature importance (XGBoost only) for any target column
#   (3) Bond default / RS: predict probability + bucket by credit spread from feature_bond.xlsx

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    # Keep a graceful fallback for environments without xgboost installed.
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    XGB_AVAILABLE = False
    print("XGBoost not found. Falling back to sklearn GradientBoosting.")

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

PAPER_CMAP = LinearSegmentedColormap.from_list("paper", ["#2d8a6e", "#3a7ca5", "#5c4d99", "#4a1a6b"], N=256)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_path(path: Path) -> Path:
    """If path cannot be overwritten (e.g. open in Excel), return a timestamped alternative."""
    return path.with_name(f"{path.stem}_{_timestamp()}{path.suffix}")


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        alt = _safe_path(path)
        df.to_csv(alt, **kwargs)
        print(f"  Warning: could not write '{path.name}' (file may be open). Wrote '{alt.name}' instead.")
        return alt


def safe_savefig(fig, path: Path, **kwargs) -> Path:
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        alt = _safe_path(path)
        fig.savefig(alt, **kwargs)
        print(f"  Warning: could not write '{path.name}' (file may be open). Wrote '{alt.name}' instead.")
        return alt

# ---------------------------------------------------------------------------
# Leakage guards (bond default)
# ---------------------------------------------------------------------------

LEAKAGE_FEATURES = {
    # user-specified
    "d_DP_RS",
    "sum_DP_RS",
    "d_Restructure",
    "date_DP",
    "default_month",
    "restructure_month",
    # common variants
    "date_RS",
    "d_Default_Payment",
    # time/age proxies should not drive default interpretation or GAF anomaly decks
    "year",
    "AgeYear",
    "lnAge",
    "FY_NonFinancial",
    "DF_FY_NonFinancial",
    "foundation_date",
}


def is_leakage_feature(name: str, target_col: str | None = None) -> bool:
    n = str(name)
    if target_col and n == target_col:
        return True
    if n in LEAKAGE_FEATURES:
        return True
    # Catch likely label/derived-event columns by name
    nlow = n.lower()
    if nlow in {
        "year",
        "ageyear",
        "lnage",
        "fy_nonfinancial",
        "df_fy_nonfinancial",
        "foundation_date",
    }:
        return True
    return any(
        key in nlow
        for key in [
            "dp_rs",
            "default_month",
            "restructure_month",
            "date_dp",
            "date_rs",
            "restructure",
            "default",
        ]
    )


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def _read_table(filepath: str, sheet_name: str | None = None) -> pd.DataFrame:
    p = Path(filepath)
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        if sheet_name is None:
            return pd.read_excel(filepath)
        return pd.read_excel(filepath, sheet_name=sheet_name)
    if suf == ".csv":
        # Keep default dtype inference; callers later filter numeric cols.
        return pd.read_csv(filepath)
    if suf == ".dta":
        return pd.read_stata(filepath)
    raise ValueError(f"Unsupported file type: {suf}. Expected .xlsx/.csv/.dta")
# ---------------------------------------------------------------------------
# XGBoost helpers (XGBoost only)
# ---------------------------------------------------------------------------

def _is_binary_like(y: np.ndarray) -> bool:
    y = np.asarray(y)
    if y.ndim != 1:
        return False
    y = y[~np.isnan(y)]
    uniq = np.unique(y)
    if len(uniq) == 0:
        return False
    if len(uniq) <= 2 and set(uniq.tolist()).issubset({0, 1}):
        return True
    return False


def build_xgb_model(seed: int = 42):
    if XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0,
        )
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3,
        learning_rate=0.05, random_state=seed,
    )


def build_xgb_classifier(seed: int = 42):
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0,
            eval_metric="logloss",
        )
    return GradientBoostingClassifier(
        n_estimators=400, max_depth=3,
        learning_rate=0.05, random_state=seed,
    )


def build_lgbm_classifier(seed: int = 42):
    if not LGBM_AVAILABLE:
        raise ImportError("lightgbm is not installed but was requested.")
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
    )


def build_catboost_classifier(seed: int = 42):
    if not CAT_AVAILABLE:
        raise ImportError("catboost is not installed but was requested.")
    return CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def importance_frame(names, importances, extra_cols=None):
    frame = pd.DataFrame({"feature": names, "importance": importances})
    if extra_cols:
        for k, v in extra_cols.items():
            frame[k] = v
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_regressor(model, xtr, ytr, xte, yte):
    model.fit(xtr, ytr)
    preds = model.predict(xte)
    rmse = float(np.sqrt(mean_squared_error(yte, preds)))
    return model, rmse


# ---------------------------------------------------------------------------
# Koopman-style linear dynamics (finite-dimensional EDMD on raw features)
# ---------------------------------------------------------------------------

def fit_linear_koopman_K(X_now: np.ndarray, X_next: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Fit a linear Koopman approximation K such that X_next ≈ X_now @ K.T.

    Uses ridge-regularized least squares:
      argmin_K ||X_next - X_now K^T||_F^2 + ridge ||K||_F^2
    """
    X_now = np.asarray(X_now, dtype=np.float64)
    X_next = np.asarray(X_next, dtype=np.float64)
    if X_now.ndim != 2 or X_next.ndim != 2:
        raise ValueError("X_now and X_next must be 2D arrays.")
    if X_now.shape != X_next.shape:
        raise ValueError(f"Shape mismatch: X_now {X_now.shape} vs X_next {X_next.shape}")
    n, d = X_now.shape
    if n < 2:
        raise ValueError("Need at least 2 paired samples to fit Koopman K.")

    A = X_now.T @ X_now
    A = A + float(ridge) * np.eye(d, dtype=np.float64)
    B = X_now.T @ X_next
    K_T = np.linalg.solve(A, B)  # (d,d)
    return K_T.T.astype(np.float32, copy=False)


def koopman_forecast_features(X: np.ndarray, K: np.ndarray, feat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Create Koopman 1-step forecast features: Kx (and its norm)."""
    X = np.asarray(X, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    Z = X @ K.T
    z_cols = [f"koop1_{c}" for c in feat_cols]
    z_norm = np.linalg.norm(Z.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    Z2 = np.concatenate([Z, z_norm], axis=1)
    return Z2, z_cols + ["koop1_norm"]


# ---------------------------------------------------------------------------
# Tabular feature importance (XGBoost only; any target column)
# ---------------------------------------------------------------------------

def prepare_data_tabular(filepath: str, target_col: str):
    df = _read_table(filepath)
    # Some exports can contain duplicated column labels; keep first occurrence.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    # Keep numeric-only features; drop obvious identifiers if present.
    exclude = {
        target_col,
        "Unnamed: 0",
        "date",
        "q_date",
        "month_year",
        "temp_date",
        "Date_Financial",
    }

    # Prefer time & entity keys if present, for rolling missing-value fill.
    entity_col = "firm_id" if "firm_id" in df.columns else None
    time_col = None
    for cand in ["q_date", "month_year", "temp_date", "Date_Financial", "year"]:
        if cand in df.columns:
            time_col = cand
            break

    # Ensure we don't accidentally include keys as model features.
    if entity_col:
        exclude.add(entity_col)
    if time_col:
        exclude.add(time_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in exclude and not is_leakage_feature(c, target_col=target_col)]
    # Guard against duplicated column names (can happen after merges/exports).
    feat_cols = list(dict.fromkeys(feat_cols))
    if not feat_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    cols_to_keep = []
    if entity_col:
        cols_to_keep.append(entity_col)
    if time_col and time_col not in cols_to_keep:
        cols_to_keep.append(time_col)
    cols_to_keep += feat_cols + [target_col]
    cols_to_keep = list(dict.fromkeys([c for c in cols_to_keep if c is not None]))
    work = df[cols_to_keep].copy()

    # Drop features with too many missing values (keep consistent feature set).
    nan_frac = work[feat_cols].isnull().mean()
    # If a column name is duplicated, nan_frac[c] may return a Series; handle robustly.
    kept: list[str] = []
    for c in feat_cols:
        v = nan_frac[c]
        if isinstance(v, pd.Series):
            v = float(v.iloc[0])
        else:
            v = float(v)
        if v <= 0.8:
            kept.append(c)
    feat_cols = kept
    work = work[[c for c in cols_to_keep if c in ({entity_col, time_col} | set(feat_cols) | {target_col})]].copy()

    # Sort for rolling fill (uses only prior information).
    if time_col:
        # Try parsing time columns; if parsing fails keep as-is for stable sort.
        try:
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        except Exception:
            pass
    sort_cols = [c for c in [entity_col, time_col] if c]
    if sort_cols:
        work = work.sort_values(sort_cols).reset_index(drop=True)

    # Rolling-window fill missing values from previous observations.
    # This matches the "use previous data" intent and avoids dropping rare positives.
    fill_window = 4
    if entity_col:
        g = work.groupby(entity_col, sort=False)
        for c in feat_cols:
            s = work[c]
            prev_roll = g[c].transform(lambda x: x.rolling(fill_window, min_periods=1).mean().shift(1))
            work[c] = s.fillna(prev_roll).ffill()
    else:
        for c in feat_cols:
            prev_roll = work[c].rolling(fill_window, min_periods=1).mean().shift(1)
            work[c] = work[c].fillna(prev_roll).ffill()

    # Keep rows where target is present; leave remaining NaNs in X (XGBoost can handle).
    work = work.dropna(subset=[target_col]).reset_index(drop=True)

    x = work[feat_cols].to_numpy(dtype=np.float32, copy=True)
    y = work[target_col].to_numpy(dtype=np.float32, copy=True)
    return x, y, feat_cols, work, None


def run_tabular_importance(x, y, feat_cols, seed: int = 42, train_ratio: float = 0.8):
    # Auto-detect binary vs regression
    if _is_binary_like(y):
        # Stratified split to avoid train containing only one class.
        xtr, xte, ytr, yte = train_test_split(
            x, y, train_size=train_ratio, random_state=seed, shuffle=True, stratify=y
        )
        base = float(np.clip(np.mean(ytr), 1e-3, 1 - 1e-3))
        model = build_xgb_classifier(seed=seed)
        # XGBoost needs base_score in (0,1) for logistic loss; guard against 0/1.
        if hasattr(model, "set_params"):
            model.set_params(base_score=base)
        model.fit(xtr, ytr)
        # Some classifiers expose feature_importances_, keep consistent with regressor.
        imps = getattr(model, "feature_importances_", None)
        if imps is None:
            raise RuntimeError("Model does not expose feature_importances_.")
        results = {
            "task": "classification",
            "importance": importance_frame(feat_cols, imps),
        }
        return model, results

    sp = max(2, int(train_ratio * len(y)))
    xtr, xte = x[:sp], x[sp:]
    ytr, yte = y[:sp], y[sp:]
    model = build_xgb_model(seed=seed)
    model, rmse = evaluate_regressor(model, xtr, ytr, xte, yte)
    results = {
        "task": "regression",
        "rmse": rmse,
        "importance": importance_frame(feat_cols, model.feature_importances_),
    }
    return model, results


# ---------------------------------------------------------------------------
# Bond default probability (DTD + selected important features)
# ---------------------------------------------------------------------------

def _pd_threshold_from_spread_bps(aa_spread_bps: float, recovery_rate: float) -> float:
    # Simple mapping: spread ≈ PD * (1 - R)
    # spread in bps -> decimal annual spread
    s = float(aa_spread_bps) / 10000.0
    rr = float(recovery_rate)
    lgd = max(1e-6, 1.0 - rr)
    return float(np.clip(s / lgd, 0.0, 1.0))


def _resolve_rs_target(df: pd.DataFrame, preferred: str = "RS") -> str:
    if preferred in df.columns:
        return preferred
    for cand in ["target_RS", "rs", "RS_flag", "label_RS"]:
        if cand in df.columns:
            print(f"  Warning: target column '{preferred}' not found; using '{cand}' instead.")
            return cand
    raise ValueError("RS target column not found. Expected 'RS' (preferred) or 'target_RS'.")


def prepare_data_bond_rs_from_feature_bond(
    filepath: str,
    sheet_name: str = "Bond_RS_TimeSeries_34",
    target_preferred: str = "RS",
):
    df = _read_table(filepath, sheet_name=sheet_name)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    target_col = _resolve_rs_target(df, preferred=target_preferred)
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    # Exclude common keys and non-features.
    exclude = {
        target_col,
        "firm_id",
        "dt",
        "date",
        "q_date",
        "Unnamed: 0",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in exclude and not is_leakage_feature(c, target_col=target_col)]
    feat_cols = list(dict.fromkeys(feat_cols))
    if not feat_cols:
        raise ValueError("No numeric feature columns found in feature_bond.xlsx after exclusions.")

    work = df[[c for c in ["firm_id", "dt"] if c in df.columns] + feat_cols + [target_col]].copy()
    work = work.dropna(subset=[target_col]).reset_index(drop=True)

    # Enforce binary 0/1 for RS
    y = work[target_col].to_numpy(dtype=np.float32, copy=True)
    if not _is_binary_like(y):
        # Try coercion for boolean-ish fields
        y2 = pd.Series(work[target_col]).astype(str).str.strip().str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
        if y2.isna().any():
            raise ValueError(f"RS target must be binary 0/1. Found non-binary values in column '{target_col}'.")
        work[target_col] = y2.astype(np.int32)
        y = work[target_col].to_numpy(dtype=np.float32, copy=True)

    # Simple median fill for remaining NaNs in X (stable and reproducible)
    for c in feat_cols:
        if work[c].isna().any():
            med = float(work[c].median(skipna=True)) if work[c].notna().any() else 0.0
            work[c] = work[c].fillna(med)

    X = work[feat_cols].to_numpy(dtype=np.float32, copy=True)
    return X, y, feat_cols, work, target_col


def run_bond_rs_prediction_from_feature_bond(
    feature_bond_xlsx: str,
    output_dir: Path,
    sheet_name: str = "Bond_RS_TimeSeries_34",
    target_preferred: str = "RS",
    aa_spread_bps: float | None = None,
    recovery_rate: float = 0.4,
    seed: int = 42,
):
    X, y, feat_cols, work, target_col = prepare_data_bond_rs_from_feature_bond(
        filepath=feature_bond_xlsx, sheet_name=sheet_name, target_preferred=target_preferred
    )

    if not _is_binary_like(y):
        raise ValueError(f"Bond RS mode expects a binary target in {{0,1}}, got non-binary '{target_col}'.")

    # Stratified split (very imbalanced)
    n = int(len(y))
    idx_all = np.arange(n, dtype=np.int64)
    idx_tr, idx_te = train_test_split(
        idx_all, train_size=0.8, random_state=seed, shuffle=True, stratify=y
    )
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    pos = float(np.sum(ytr == 1))
    neg = float(np.sum(ytr == 0))
    scale_pos_weight = float(neg / max(pos, 1.0))
    base = float(np.clip(np.mean(ytr), 1e-3, 1 - 1e-3))

    def _configure_imbalance(model_, model_name: str):
        """Best-effort imbalance handling per library."""
        if hasattr(model_, "set_params"):
            if model_name == "xgboost":
                try:
                    model_.set_params(base_score=base)
                except Exception:
                    pass
                try:
                    model_.set_params(scale_pos_weight=scale_pos_weight)
                except Exception:
                    pass
            elif model_name == "lightgbm":
                try:
                    model_.set_params(class_weight={0: 1.0, 1: scale_pos_weight})
                except Exception:
                    pass
            elif model_name == "catboost":
                try:
                    model_.set_params(class_weights=[1.0, scale_pos_weight])
                except Exception:
                    pass
        return model_

    # Candidate models for comparison (if installed)
    model = _configure_imbalance(build_xgb_classifier(seed=seed), "xgboost")
    model_lgbm = None
    model_cat = None
    if LGBM_AVAILABLE:
        model_lgbm = _configure_imbalance(build_lgbm_classifier(seed=seed), "lightgbm")
    if CAT_AVAILABLE:
        model_cat = _configure_imbalance(build_catboost_classifier(seed=seed), "catboost")

    def _eval_clf(m, Xte_, yte_):
        yhat_ = m.predict(Xte_)
        yhat_proba_ = None
        try:
            yhat_proba_ = m.predict_proba(Xte_)[:, 1]
        except Exception:
            yhat_proba_ = None
        acc_ = float(accuracy_score(yte_, yhat_))
        prec_, rec_, f1_, _ = precision_recall_fscore_support(yte_, yhat_, average="binary", zero_division=0)
        cm_ = confusion_matrix(yte_, yhat_, labels=[0, 1])
        tn_, fp_, fn_, tp_ = (int(cm_[0, 0]), int(cm_[0, 1]), int(cm_[1, 0]), int(cm_[1, 1]))
        auc_ = None
        if yhat_proba_ is not None:
            try:
                auc_ = float(roc_auc_score(yte_, yhat_proba_))
            except Exception:
                auc_ = None
        return {
            "acc": acc_,
            "prec": float(prec_),
            "rec": float(rec_),
            "f1": float(f1_),
            "auc": auc_,
            "tn": tn_,
            "fp": fp_,
            "fn": fn_,
            "tp": tp_,
        }, yhat_proba_

    # -----------------------------
    # Multi-model comparison (raw features)
    # -----------------------------
    runs = []

    model.fit(Xtr, ytr)
    m_raw, _ = _eval_clf(model, Xte, yte)
    p_all_raw = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X).astype(float)
    runs.append({"name": "raw_xgboost", "model": model, "metrics": m_raw})

    m_lgbm = None
    p_all_lgbm = None
    if model_lgbm is not None:
        model_lgbm.fit(Xtr, ytr)
        m_lgbm, _ = _eval_clf(model_lgbm, Xte, yte)
        p_all_lgbm = model_lgbm.predict_proba(X)[:, 1] if hasattr(model_lgbm, "predict_proba") else model_lgbm.predict(X).astype(float)
        runs.append({"name": "raw_lightgbm", "model": model_lgbm, "metrics": m_lgbm})

    m_cat = None
    p_all_cat = None
    if model_cat is not None:
        model_cat.fit(Xtr, ytr)
        m_cat, _ = _eval_clf(model_cat, Xte, yte)
        p_all_cat = model_cat.predict_proba(X)[:, 1] if hasattr(model_cat, "predict_proba") else model_cat.predict(X).astype(float)
        runs.append({"name": "raw_catboost", "model": model_cat, "metrics": m_cat})

    output_dir.mkdir(parents=True, exist_ok=True)

    out = work.copy()
    out["pd_RS_raw_xgboost"] = p_all_raw
    if p_all_lgbm is not None:
        out["pd_RS_raw_lightgbm"] = p_all_lgbm
    if p_all_cat is not None:
        out["pd_RS_raw_catboost"] = p_all_cat

    # -----------------------------
    # Koopman-XGBoost: augment with Kx_t (fit K on TRAIN pairs only)
    # -----------------------------
    K = None
    try:
        if "firm_id" in out.columns and "dt" in out.columns:
            tmp = out[["firm_id", "dt"]].copy()
            tmp["dt"] = pd.to_datetime(tmp["dt"], errors="coerce")
            order = np.lexsort((tmp["dt"].fillna(pd.Timestamp.min).to_numpy(), tmp["firm_id"].to_numpy()))
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))

            firm_sorted = tmp["firm_id"].to_numpy()[order]
            same_firm = firm_sorted[:-1] == firm_sorted[1:]
            X_sorted = X[order]

            is_train = np.zeros(n, dtype=bool)
            is_train[idx_tr] = True
            is_train_sorted = is_train[order]
            pair_mask = same_firm & is_train_sorted[:-1] & is_train_sorted[1:]

            if int(np.sum(pair_mask)) >= 50:
                X_now = X_sorted[:-1][pair_mask]
                X_next = X_sorted[1:][pair_mask]
                K = fit_linear_koopman_K(X_now, X_next, ridge=1e-3)

                Z_all, z_cols = koopman_forecast_features(X, K, feat_cols=feat_cols)
                Xk = np.concatenate([X, Z_all], axis=1)
                feat_cols_k = feat_cols + z_cols

                Xtr_k, Xte_k = Xk[idx_tr], Xk[idx_te]

                model_k = _configure_imbalance(build_xgb_classifier(seed=seed + 7), "xgboost")
                model_k.fit(Xtr_k, ytr)
                m_koop, _ = _eval_clf(model_k, Xte_k, yte)

                p_all_k = model_k.predict_proba(Xk)[:, 1] if hasattr(model_k, "predict_proba") else model_k.predict(Xk).astype(float)
                out["pd_RS_koopman_xgboost"] = p_all_k
                runs.append({"name": "koopman_xgboost", "model": model_k, "metrics": m_koop})

                model_k_lgbm = None
                model_k_cat = None
                m_koop_lgbm = None
                m_koop_cat = None
                if LGBM_AVAILABLE:
                    model_k_lgbm = _configure_imbalance(build_lgbm_classifier(seed=seed + 7), "lightgbm")
                    model_k_lgbm.fit(Xtr_k, ytr)
                    m_koop_lgbm, _ = _eval_clf(model_k_lgbm, Xte_k, yte)
                    p_all_k_lgbm = model_k_lgbm.predict_proba(Xk)[:, 1] if hasattr(model_k_lgbm, "predict_proba") else model_k_lgbm.predict(Xk).astype(float)
                    out["pd_RS_koopman_lightgbm"] = p_all_k_lgbm
                    runs.append({"name": "koopman_lightgbm", "model": model_k_lgbm, "metrics": m_koop_lgbm})

                if CAT_AVAILABLE:
                    model_k_cat = _configure_imbalance(build_catboost_classifier(seed=seed + 7), "catboost")
                    model_k_cat.fit(Xtr_k, ytr)
                    m_koop_cat, _ = _eval_clf(model_k_cat, Xte_k, yte)
                    p_all_k_cat = model_k_cat.predict_proba(Xk)[:, 1] if hasattr(model_k_cat, "predict_proba") else model_k_cat.predict(Xk).astype(float)
                    out["pd_RS_koopman_catboost"] = p_all_k_cat
                    runs.append({"name": "koopman_catboost", "model": model_k_cat, "metrics": m_koop_cat})
            else:
                model_k = None
                m_koop = None
        else:
            model_k = None
            m_koop = None
    except Exception as e:
        print(f"  Warning: Koopman feature augmentation failed: {e}")
        model_k = None
        m_koop = None

    pd_thr = None
    if aa_spread_bps is not None:
        pd_thr = _pd_threshold_from_spread_bps(aa_spread_bps=aa_spread_bps, recovery_rate=recovery_rate)
    out["pd_bucket_AA"] = "unbucketed"
    if pd_thr is not None:
        out["pd_bucket_AA"] = np.where(out["pd_RS"] <= pd_thr, "AA_or_better", "below_AA")
        out["pd_threshold_AA"] = pd_thr
        out["aa_spread_bps"] = float(aa_spread_bps)
        out["recovery_rate"] = float(recovery_rate)

    # Pick best by ROC-AUC (higher is better), fallback to accuracy
    def _sort_key(rr):
        auc_ = rr["metrics"]["auc"]
        return (auc_ if auc_ is not None else -1.0, rr["metrics"]["acc"])

    best = sorted(runs, key=_sort_key, reverse=True)[0]
    out["pd_RS"] = out[f"pd_RS_{best['name']}"]
    out["best_model"] = best["name"]

    pred_path = safe_to_csv(out, output_dir / "bond_rs_predictions.csv", index=False, encoding="utf-8-sig")

    # Write comparison table
    comp_rows = []
    for rr in runs:
        comp_rows.append(
            {
                "model": rr["name"],
                "roc_auc": rr["metrics"]["auc"],
                "accuracy": rr["metrics"]["acc"],
                "precision": rr["metrics"]["prec"],
                "recall": rr["metrics"]["rec"],
                "f1": rr["metrics"]["f1"],
                "tn": rr["metrics"]["tn"],
                "fp": rr["metrics"]["fp"],
                "fn": rr["metrics"]["fn"],
                "tp": rr["metrics"]["tp"],
            }
        )
    safe_to_csv(
        pd.DataFrame(comp_rows).sort_values(["roc_auc", "accuracy"], ascending=False),
        output_dir / "bond_rs_model_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------------
    # Plots for pd_RS (all firms in one figure + optional per-firm)
    # -------------------------------------------------------------------
    try:
        plot_pd_rs_timeseries_by_firm(
            out_df=out,
            output_dir=output_dir,
            firm_col="firm_id",
            time_col="dt",
            score_col="pd_RS",
            label_col=target_col,
            generate_per_firm=False,
        )
    except Exception as e:
        print(f"  Warning: failed to generate per-firm time-series plots: {e}")

    if "pd_RS_koopman_xgboost" in out.columns:
        try:
            plot_pd_rs_timeseries_by_firm(
                out_df=out,
                output_dir=output_dir,
                firm_col="firm_id",
                time_col="dt",
                score_col="pd_RS_koopman_xgboost",
                label_col=target_col,
                generate_per_firm=False,
            )
        except Exception as e:
            print(f"  Warning: failed to generate Koopman time-series plots: {e}")

    # --- raw importance + plot (write also xgboost.jpg for backward compatibility)
    imps = getattr(model, "feature_importances_", None)
    if imps is None:
        imps = np.zeros(len(feat_cols), dtype=float)
    imp_df = importance_frame(feat_cols, imps)
    imp_path = safe_to_csv(imp_df, output_dir / "bond_rs_importance.csv", index=False)
    plot_feature_importance(
        imp_df,
        title="Top 20 Feature Importance Analysis\n(Bond RS: XGBoost)",
        subtitle=f"target={target_col}",
        save_path=output_dir / "bond_rs_feature_importance.png",
    )
    # Keep legacy name used in manuals
    plot_feature_importance(
        imp_df,
        title="Top 20 Feature Importance Analysis\n(Bond RS: XGBoost)",
        subtitle=f"target={target_col}",
        save_path=output_dir / "xgboost.jpg",
    )

    # --- koopman importance + plot (if available)
    imp_df_k = None
    if "model_k" in locals() and model_k is not None:
        imps_k = getattr(model_k, "feature_importances_", None)
        if imps_k is None:
            imps_k = np.zeros(len(feat_cols_k), dtype=float)
        imp_df_k = importance_frame(feat_cols_k, imps_k)
        safe_to_csv(imp_df_k, output_dir / "koopman_xgboost_importance.csv", index=False)
        plot_feature_importance(
            imp_df_k,
            title="Top 20 Feature Importance Analysis\n(Bond RS: Koopman-XGBoost)",
            subtitle=f"target={target_col}",
            save_path=output_dir / "koopman_xgboost_feature_importance.png",
        )

    metrics_rows = []
    for rr in runs:
        metrics_rows.append({
            "model": rr["name"],
            "n_total": int(len(y)),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "pos_train": int(np.sum(ytr == 1)),
            "pos_test": int(np.sum(yte == 1)),
            "accuracy": rr["metrics"]["acc"],
            "accuracy_pct": rr["metrics"]["acc"] * 100.0,
            "precision": rr["metrics"]["prec"],
            "recall": rr["metrics"]["rec"],
            "f1": rr["metrics"]["f1"],
            "roc_auc": rr["metrics"]["auc"],
            "tn": rr["metrics"]["tn"],
            "fp": rr["metrics"]["fp"],
            "fn": rr["metrics"]["fn"],
            "tp": rr["metrics"]["tp"],
            "sheet": sheet_name,
            "target_col": target_col,
            "koopman_used": bool(rr["name"].startswith("koopman_")),
            "is_best": bool(rr["name"] == best["name"]),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["roc_auc", "accuracy"], ascending=False)
    metrics_path = safe_to_csv(metrics_df, output_dir / "bond_rs_metrics.csv", index=False)

    print(f"Saved bond RS outputs to: {output_dir.resolve()}")
    print(f"  predictions : {pred_path}")
    print(f"  importance  : {imp_path}")
    print(f"  comparison  : {output_dir / 'bond_rs_model_comparison.csv'}")
    print(f"  metrics     : {metrics_path}")
    for _, r in metrics_df.iterrows():
        auc_v = r.get("roc_auc", None)
        auc_s = f"{float(auc_v):.4f}" if auc_v is not None and not pd.isna(auc_v) else "NA"
        print(f"  {r['model']}: F1={float(r['f1']):.4f}  AUC={auc_s}  Acc={float(r['accuracy'])*100.0:.2f}%  tp={int(r['tp'])} fp={int(r['fp'])} fn={int(r['fn'])} tn={int(r['tn'])}")


# ---------------------------------------------------------------------------
# Plotting – paper-style horizontal bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance(
    imp_df: pd.DataFrame,
    title: str,
    subtitle: str,
    save_path: Path,
    top_n: int = 20,
):
    """Horizontal bar chart matching the paper's feature-score style."""
    top = imp_df.head(top_n).copy()
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    norm_vals = top["importance"].values
    max_val = norm_vals.max() if norm_vals.max() > 0 else 1.0
    colors = [PAPER_CMAP(v / max_val) for v in norm_vals]

    bars = ax.barh(range(len(top)), top["importance"].values, color=colors, height=0.7)

    for bar, val in zip(bars, top["importance"].values):
        ax.text(
            bar.get_width() + max_val * 0.008, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="medium",
        )

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.set_xlabel(f"Importance Score    {subtitle}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Features", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max_val * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    safe_savefig(fig, save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {save_path}")


def plot_pd_rs_timeseries_by_firm(
    out_df: pd.DataFrame,
    output_dir: Path,
    firm_col: str = "firm_id",
    time_col: str = "dt",
    score_col: str = "pd_RS",
    label_col: str | None = None,
    max_firms: int | None = None,
    generate_per_firm: bool = False,
):
    """Save time-series plots of predicted scores per firm.

    Writes:
      - output_dir/pd_rs_scatter_all_firms.png (single combined scatter/heatmap)
      - (optional) output_dir/firm_timeseries/firm_<firm_id>_pd_rs.png (one per firm)
    """
    if firm_col not in out_df.columns or time_col not in out_df.columns:
        raise ValueError(f"Required columns not found for time series plotting: {firm_col}, {time_col}")
    if score_col not in out_df.columns:
        raise ValueError(f"Score column not found: {score_col}")

    print("  Generating combined scatter plot (all firms) ...", flush=True)
    df = out_df[[firm_col, time_col, score_col] + ([label_col] if label_col and label_col in out_df.columns else [])].copy()

    # Robust datetime parsing for dt
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid datetime values found in dt; cannot plot time series.")

    df = df.sort_values([firm_col, time_col]).reset_index(drop=True)
    firm_ids = df[firm_col].dropna().unique().tolist()
    if max_firms is not None:
        firm_ids = firm_ids[: int(max_firms)]
    print(f"  Firms to plot: {len(firm_ids)}", flush=True)

    # Map firm_id to an ordered integer index for plotting on y-axis.
    # This makes a single scatter plot readable (like a heatmap).
    firm_order = sorted(firm_ids)
    firm_to_idx = {fid: i for i, fid in enumerate(firm_order)}
    df["_firm_idx"] = df[firm_col].map(firm_to_idx).astype(int)

    # Convert datetime to numeric for 3D plots
    x_num = mdates.date2num(df[time_col].dt.to_pydatetime())

    fig, ax = plt.subplots(figsize=(14, 8))
    sc = ax.scatter(
        df[time_col].values,
        df["_firm_idx"].values,
        c=df[score_col].values,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=2.0,
        alpha=0.85,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(score_col)

    # Optional: overlay label=1 as small red markers
    if label_col and label_col in df.columns:
        ylbl = pd.to_numeric(df[label_col], errors="coerce")
        m1 = ylbl == 1
        if bool(np.any(m1)):
            ax.scatter(
                df.loc[m1, time_col].values,
                df.loc[m1, "_firm_idx"].values,
                s=6.0,
                c="#e74c3c",
                alpha=0.75,
                linewidths=0,
                label=f"{label_col}=1",
            )
            ax.legend(loc="upper right", fontsize=9)

    ax.set_title(f"All firms scatter: {score_col} over time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("firm_id (ordered index)")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    safe_savefig(fig, output_dir / "pd_rs_scatter_all_firms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_dir / 'pd_rs_scatter_all_firms.png'}", flush=True)

    # --- 3D scatter: x=time, y=firm_idx, z=pd_RS ---
    fig = plt.figure(figsize=(14, 9))
    ax3 = fig.add_subplot(111, projection="3d")
    # Use a light alpha; 150k+ points can be heavy in 3D
    p = ax3.scatter(
        x_num,
        df["_firm_idx"].values,
        df[score_col].values,
        c=df[score_col].values,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=1.2,
        alpha=0.55,
        linewidths=0,
    )
    cb = fig.colorbar(p, ax=ax3, fraction=0.03, pad=0.04)
    cb.set_label(score_col)

    ax3.set_title(f"3D scatter: {score_col} over time (all firms)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("firm_id (ordered index)")
    ax3.set_zlabel(score_col)
    ax3.set_zlim(0.0, 1.0)
    try:
        # Force DD/MM/YY on 3D x-axis
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
        for t in ax3.get_xticklabels():
            t.set_rotation(35)
            t.set_ha("right")
    except Exception:
        pass
    fig.tight_layout()
    safe_savefig(fig, output_dir / "pd_rs_scatter_all_firms_3d.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_dir / 'pd_rs_scatter_all_firms_3d.png'}", flush=True)

    # --- 3D label plot: z = default/RS label (0/1) ---
    if label_col and label_col in df.columns:
        ylbl = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
        z_lbl = (ylbl == 1).astype(int).values

        fig = plt.figure(figsize=(14, 8))
        ax3 = fig.add_subplot(111, projection="3d")
        # Plot all points at z=0 lightly
        m0 = z_lbl == 0
        ax3.scatter(
            x_num[m0],
            df.loc[m0, "_firm_idx"].values,
            np.zeros(int(np.sum(m0))),
            c=df.loc[m0, score_col].values,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=1.0,
            alpha=0.25,
            linewidths=0,
        )
        # Plot default points at z=1 in red
        m1 = z_lbl == 1
        if bool(np.any(m1)):
            ax3.scatter(
                x_num[m1],
                df.loc[m1, "_firm_idx"].values,
                np.ones(int(np.sum(m1))),
                c="#e74c3c",
                s=6.0,
                alpha=0.85,
                linewidths=0,
                label=f"{label_col}=1",
            )
            ax3.legend(loc="upper right", fontsize=9)

        ax3.set_title("3D default label view: z=1 indicates RS/default events", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("firm_id (ordered index)")
        ax3.set_zlabel("default label (0/1)")
        ax3.set_zticks([0, 1])
        ax3.set_zlim(-0.05, 1.05)
        try:
            ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
            for t in ax3.get_xticklabels():
                t.set_rotation(35)
                t.set_ha("right")
        except Exception:
            pass
        fig.tight_layout()
        safe_savefig(fig, output_dir / "pd_rs_timeseries_all_firms_3d.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {output_dir / 'pd_rs_timeseries_all_firms_3d.png'}", flush=True)

    if not generate_per_firm:
        return

    # Optional per-firm plots (slow for hundreds of firms)
    ts_dir = output_dir / "firm_timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    print("  Generating per-firm time series plots (slow) ...", flush=True)
    for i, fid in enumerate(firm_ids, start=1):
        s = df[df[firm_col] == fid]
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(s[time_col], s[score_col], linewidth=1.4, color="#2d8a6e", label=score_col)
        ax.set_title(f"firm_id={fid}  |  {score_col} time series", fontsize=11, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel(score_col)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        safe_savefig(fig, ts_dir / f"firm_{fid}_pd_rs.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        if i % 50 == 0:
            print(f"  Plotted {i}/{len(firm_ids)} firms ...", flush=True)


# ---------------------------------------------------------------------------
# Summary & output
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="XGBoost utilities: (2) Tabular feature importance, (3) Bond RS prediction from feature_bond.xlsx")
    p.add_argument(
        "--mode",
        choices=["tabular", "bond_rs"],
        default="tabular",
        help="tabular: XGBoost feature importance for any target; bond_rs: predict RS (0/1) from feature_bond.xlsx",
    )
    p.add_argument(
        "--data",
        default="Database_final_nostrings.csv",
        help="Input file for tabular mode (.csv/.xlsx/.dta). Ignored by bond_rs unless you override --feature-bond-xlsx.",
    )
    p.add_argument(
        "--target-col",
        default=None,
        help="Target column name (required for tabular mode). For bond_rs, preferred target is 'RS' (fallback: target_RS).",
    )
    p.add_argument(
        "--feature-bond-xlsx",
        default="feature_bond.xlsx",
        help="Input Excel for bond_rs mode (default: feature_bond.xlsx).",
    )
    p.add_argument(
        "--sheet",
        default="Bond_RS_TimeSeries_34",
        help="Excel sheet name for bond_rs mode (default: Bond_RS_TimeSeries_34).",
    )
    p.add_argument(
        "--aa-spread-bps",
        type=float,
        default=None,
        help="AA credit spread threshold (bps). If provided, bucket PD into AA_or_better vs below_AA using spread ~ PD*(1-R).",
    )
    p.add_argument("--recovery-rate", type=float, default=0.4, help="Recovery rate used for AA spread->PD mapping.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="koopman_outputs")
    return p.parse_args()


def main():
    # Quick-run (no arguments):
    # `python koopman.py` will run bond_rs using ONLY feature_bond.xlsx in this folder.
    if len(sys.argv) == 1:
        script_dir = Path(__file__).resolve().parent
        feature_bond = script_dir / "feature_bond.xlsx"
        out_root = script_dir / "koopman_outputs"
        out_dir = out_root / "bond_rs"

        if not feature_bond.exists():
            raise FileNotFoundError(f"feature_bond.xlsx not found at: {feature_bond}")

        set_seed(42)
        run_bond_rs_prediction_from_feature_bond(
            feature_bond_xlsx=str(feature_bond),
            output_dir=out_dir,
            sheet_name="Bond_RS_TimeSeries_34",
            target_preferred="RS",
            aa_spread_bps=None,
            recovery_rate=0.4,
            seed=42,
        )
        return

    args = parse_args()
    set_seed(args.seed)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "bond_rs":
        out_dir = out_root / "bond_rs"
        run_bond_rs_prediction_from_feature_bond(
            feature_bond_xlsx=str(Path(args.feature_bond_xlsx)),
            output_dir=out_dir,
            sheet_name=args.sheet,
            target_preferred=args.target_col or "RS",
            aa_spread_bps=args.aa_spread_bps,
            recovery_rate=args.recovery_rate,
            seed=args.seed,
        )
        return

    # tabular mode
    if not args.target_col:
        raise ValueError("--mode tabular requires --target-col")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    dataset_name = data_path.stem
    out_dir = out_root / dataset_name
    x, y, feat_cols, df, _ = prepare_data_tabular(str(data_path), target_col=args.target_col)
    print(f"  Dataset : {dataset_name}")
    print(f"  Rows    : {len(df)}")
    print(f"  Features: {len(feat_cols)}")
    _, tab = run_tabular_importance(x, y, feat_cols, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    tab["importance"].to_csv(out_dir / "tabular_importance.csv", index=False)
    plot_feature_importance(
        tab["importance"],
        title="Top 20 Feature Importance Analysis\n(Tabular XGBoost)",
        subtitle=f"{dataset_name} target={args.target_col}",
        save_path=out_dir / "tabular_feature_importance.png",
    )
    # Backward compatibility (manuals may reference these names)
    plot_feature_importance(
        tab["importance"],
        title="Top 20 Feature Importance Analysis\n(Tabular XGBoost)",
        subtitle=f"{dataset_name} target={args.target_col}",
        save_path=out_dir / "xgboost.jpg",
    )
    tab["importance"].to_csv(out_dir / "xgboost_importance.csv", index=False)
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
