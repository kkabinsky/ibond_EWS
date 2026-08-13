"""
koopman_gaf.py - Approach 2 extension for the CMDF bond project.

Two research components are brought into the application:

  1. **Koopman + XGBoost** - a linear Koopman operator ``K`` is fitted on the
     firm-level factor panel so that ``x_{t+1} ~ K x_t``. The one-step Koopman
     forecast ``Kx`` is appended to the raw factors and an XGBoost model is
     trained on the RS / default target. This turns the static 34-factor
     classifier into a *dynamic* one that sees where each factor is heading.

  2. **Gramian Angular Field (GAF) imaging** - every factor trajectory is
     min-max normalised, mapped to an angle with the exponential mapping
     ``phi = pi (e^x - 1)/(e - 1)`` and encoded as GASF ``cos(phi_i + phi_j)``
     and GADF ``sin(phi_i - phi_j)`` images.

The numerical formulas are ported verbatim from the research scripts
``koopman.py`` (``fit_linear_koopman_K``, ``koopman_forecast_features``) and
``run_gaf_features.py`` (``normalize_series`` .. ``gadf``) that live in this
folder; this module only adds panel handling, caching and matplotlib figures so
that the Flet application can embed the results.

Standalone check:
    python koopman_gaf.py --selftest
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BOND_XLSX = os.path.join(HERE, "feature_bond.xlsx")
TS_SHEET = "Bond_RS_TimeSeries_34"

ID_COLS = ("firm_id", "dt", "target_RS")
_PANEL_CACHE: dict[str, pd.DataFrame] = {}


# ----------------------------------------------------------------- data ------
PANEL_TABLE = "bond_panel"
PANEL_META = "bond_panel_meta"


def _xlsx_stamp(path: str) -> str:
    st = os.stat(path)
    return "%d:%d" % (st.st_size, int(st.st_mtime))


def _read_bond_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=TS_SHEET)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        df = df.sort_values(["firm_id", "dt"]).reset_index(drop=True)
    return df


def import_bond_panel(path: str | None = None, db_path: str | None = None,
                      force: bool = False) -> int:
    """Copy the 34-factor panel from the workbook into SQLite.

    Parsing the 25 MB workbook costs about twenty seconds, which is why it must
    not happen while the user is waiting for a menu. The SQLite copy is written
    once, is indexed by firm and date, and reads back in well under a second.
    Returns the number of rows now in the table.
    """
    path = path or BOND_XLSX
    db_path = db_path or DB_DEFAULT
    if not os.path.exists(path):
        raise FileNotFoundError("bond panel not found: %s" % path)
    if not force and _panel_is_current(path, db_path):
        con = sqlite3.connect(db_path)
        try:
            return int(con.execute("SELECT COUNT(*) FROM %s" % PANEL_TABLE).fetchone()[0])
        finally:
            con.close()
    df = _read_bond_xlsx(path)
    out = df.copy()
    if "dt" in out.columns:
        out["dt"] = out["dt"].astype(str)
    con = sqlite3.connect(db_path)
    try:
        out.to_sql(PANEL_TABLE, con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_%s_firm_dt ON %s (firm_id, dt)"
                    % (PANEL_TABLE, PANEL_TABLE))
        pd.DataFrame([{"source": os.path.basename(path), "stamp": _xlsx_stamp(path),
                       "rows": len(out), "cols": out.shape[1]}]).to_sql(
            PANEL_META, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    _PANEL_CACHE[path] = df
    return len(df)


def _panel_is_current(path: str, db_path: str) -> bool:
    """True when the SQLite copy was built from exactly this workbook."""
    if not os.path.exists(db_path):
        return False
    con = sqlite3.connect(db_path)
    try:
        meta = pd.read_sql_query("SELECT * FROM %s" % PANEL_META, con)
        return (not meta.empty) and str(meta["stamp"].iloc[0]) == _xlsx_stamp(path)
    except Exception:
        return False
    finally:
        con.close()


def load_bond_panel(path: str | None = None, force: bool = False,
                    db_path: str | None = None) -> pd.DataFrame:
    """Load (and cache) the 34-factor bond time-series panel.

    The SQLite copy is preferred; the workbook is parsed only when that copy is
    missing or was built from a different version of the file.
    """
    path = path or BOND_XLSX
    db_path = db_path or DB_DEFAULT
    if not force and path in _PANEL_CACHE:
        return _PANEL_CACHE[path]
    if not force and _panel_is_current(path, db_path):
        con = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM %s" % PANEL_TABLE, con)
        finally:
            con.close()
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            df = df.sort_values(["firm_id", "dt"]).reset_index(drop=True)
        _PANEL_CACHE[path] = df
        return df
    if not os.path.exists(path):
        raise FileNotFoundError("bond panel not found: %s" % path)
    df = _read_bond_xlsx(path)
    _PANEL_CACHE[path] = df
    try:                                   # keep the fast path warm for next time
        import_bond_panel(path, db_path, force=True)
    except Exception:
        pass
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric factor columns (identifiers and the target are excluded)."""
    return [c for c in df.columns
            if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


def list_factors(df: pd.DataFrame | None = None) -> list[str]:
    return feature_columns(df if df is not None else load_bond_panel())


# -------------------------------------------------------------- Koopman ------
def fit_linear_koopman_K(X_now: np.ndarray, X_next: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge least squares for K with X_next ~ X_now @ K.T (ported verbatim)."""
    X_now = np.asarray(X_now, dtype=np.float64)
    X_next = np.asarray(X_next, dtype=np.float64)
    if X_now.shape != X_next.shape:
        raise ValueError("shape mismatch: %s vs %s" % (X_now.shape, X_next.shape))
    n, d = X_now.shape
    if n < 2:
        raise ValueError("need at least 2 paired samples to fit Koopman K")
    A = X_now.T @ X_now + float(ridge) * np.eye(d)
    B = X_now.T @ X_next
    return np.linalg.solve(A, B).T


def build_koopman_pairs(df: pd.DataFrame, feats: list[str]):
    """Stack consecutive within-firm observations into (X_now, X_next, index)."""
    now_idx, next_idx = [], []
    for _, g in df.groupby("firm_id", sort=False):
        idx = g.index.to_numpy()
        if len(idx) >= 2:
            now_idx.append(idx[:-1])
            next_idx.append(idx[1:])
    if not now_idx:
        raise ValueError("no firm has two consecutive observations")
    now_idx = np.concatenate(now_idx)
    next_idx = np.concatenate(next_idx)
    X = df[feats].astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy()
    return X[now_idx], X[next_idx], now_idx


def run_koopman_xgb(df: pd.DataFrame | None = None, ridge: float = 1e-3,
                    target: str = "target_RS", test_size: float = 0.3,
                    seed: int = 42) -> dict:
    """Fit K, append the one-step Koopman forecast, train XGBoost on the target."""
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score

    df = load_bond_panel() if df is None else df
    feats = feature_columns(df)
    if target not in df.columns:
        raise ValueError("target column %r not in panel" % target)

    # standardise so that K is not dominated by scale differences
    raw = df[feats].astype(float)
    mu, sd = raw.mean(), raw.std().replace(0, 1.0)
    std = ((raw - mu) / sd).fillna(0.0)
    dfs = df.copy()
    dfs[feats] = std

    X_now, X_next, rows = build_koopman_pairs(dfs, feats)
    K = fit_linear_koopman_K(X_now, X_next, ridge=ridge)

    # design matrix: current factors + their one-step Koopman forecast
    Z = X_now @ K.T
    z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
    X_full = np.hstack([X_now, Z, z_norm])
    names = list(feats) + ["koop1_%s" % c for c in feats] + ["koop1_norm"]

    y = pd.to_numeric(df.loc[rows, target], errors="coerce").fillna(0).to_numpy()
    binary = set(np.unique(y)) <= {0, 1} and len(np.unique(y)) == 2

    # IMPORTANT: split by FIRM, never by row. The RS target is extremely rare and
    # clustered inside a few firms, so a random row split would place the same
    # firm on both sides and the model would simply memorise its factor
    # signature (that yields a meaningless AUC of ~1.0).
    groups = df.loc[rows, "firm_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(X_full, y, groups=groups))
    Xtr, Xte, ytr, yte = X_full[tr], X_full[te], y[tr], y[te]

    metrics: dict[str, float] = {}
    try:
        import xgboost as xgb
        if binary:
            model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                      subsample=0.9, colsample_bytree=0.9,
                                      eval_metric="logloss", random_state=seed)
        else:
            model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                                     random_state=seed)
    except ImportError:                                    # graceful fallback
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        model = (GradientBoostingClassifier(random_state=seed) if binary
                 else GradientBoostingRegressor(random_state=seed))

    model.fit(Xtr, ytr)
    if binary:
        p = model.predict_proba(Xte)[:, 1]
        base = float(yte.mean())
        if len(np.unique(yte)) < 2:          # no positive firm landed in the test split
            metrics["AUC"] = float("nan"); metrics["PR_AUC"] = float("nan")
            metrics["MCC"] = float("nan")
        else:
            metrics["AUC"] = float(roc_auc_score(yte, p))
            metrics["PR_AUC"] = float(average_precision_score(yte, p))
            metrics["MCC"] = float(matthews_corrcoef(yte, (p >= 0.5).astype(int)))
        metrics["base_rate"] = base
        metrics["n_test_pos"] = float(int(yte.sum()))
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        pred = model.predict(Xte)
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(yte, pred)))
        metrics["R2"] = float(r2_score(yte, pred))

    imp = getattr(model, "feature_importances_", np.zeros(len(names)))
    importance = (pd.DataFrame({"feature": names, "importance": imp})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    eig = np.linalg.eigvals(K)
    return {"K": K, "eigenvalues": eig, "features": feats, "names": names,
            "importance": importance, "metrics": metrics, "binary": binary,
            "n_pairs": int(X_now.shape[0]), "n_firms": int(df["firm_id"].nunique()),
            "spectral_radius": float(np.max(np.abs(eig)))}


# -------------------------------------------------------------- figures ------
def fig_koopman_importance(res: dict, top: int = 15):
    d = res["importance"].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(d) + 1.2))
    colors = ["#7c3aed" if str(f).startswith("koop1_") else "#2563eb" for f in d["feature"]]
    ax.barh(d["feature"].astype(str), d["importance"], color=colors)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Koopman + XGBoost: top %d drivers  (purple = Koopman forecast feature)" % top,
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def fig_koopman_spectrum(res: dict):
    """Eigenvalues of K in the complex plane: inside the unit circle = stable."""
    eig = res["eigenvalues"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#94a3b8", lw=1.0, ls="--", label="unit circle")
    ax.scatter(eig.real, eig.imag, s=34, color="#7c3aed", edgecolor="white", zorder=3,
               label="eigenvalues of $K$")
    ax.axhline(0, color="#e2e8f0", lw=0.8); ax.axvline(0, color="#e2e8f0", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.set_title("Koopman spectrum  (spectral radius = %.3f)" % res["spectral_radius"],
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


# ------------------------------------------------------- persistence --------
KOOPMAN_TABLE = "koopman_factors"


def factor_table(res: dict) -> pd.DataFrame:
    """One row per factor, combining the XGBoost importance of the raw factor
    and of its Koopman one-step forecast with the Koopman dynamics itself:

      k_self      diagonal of K - how strongly the factor predicts itself
      k_influence column norm of K - how strongly it drives the whole system
    """
    feats, K = res["features"], res["K"]
    imp = dict(zip(res["importance"]["feature"].astype(str),
                   res["importance"]["importance"].astype(float)))
    out = pd.DataFrame([{
        "factor": f,
        "imp_raw": round(float(imp.get(f, 0.0)), 6),
        "imp_koopman": round(float(imp.get("koop1_%s" % f, 0.0)), 6),
        "k_self": round(float(K[i, i]), 4),
        "k_influence": round(float(np.linalg.norm(K[:, i])), 4),
    } for i, f in enumerate(feats)])
    out["imp_total"] = (out["imp_raw"] + out["imp_koopman"]).round(6)
    out = out.sort_values("imp_total", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[["rank", "factor", "imp_total", "imp_raw", "imp_koopman",
                "k_self", "k_influence"]]


def save_to_sqlite(res: dict, db_path: str, table: str = KOOPMAN_TABLE) -> pd.DataFrame:
    """Persist the per-factor table (and a small metrics table) into SQLite."""
    import sqlite3
    tbl = factor_table(res)
    met = pd.DataFrame(
        [{"metric": k, "value": float(v)} for k, v in res["metrics"].items()]
        + [{"metric": "spectral_radius", "value": float(res["spectral_radius"])},
           {"metric": "n_pairs", "value": float(res["n_pairs"])},
           {"metric": "n_firms", "value": float(res["n_firms"])}])
    con = sqlite3.connect(db_path)
    try:
        tbl.to_sql(table, con, if_exists="replace", index=False)
        met.to_sql(table + "_metrics", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return tbl


def load_from_sqlite(db_path: str, table: str = KOOPMAN_TABLE) -> pd.DataFrame:
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM %s" % table, con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


# ------------------------------------------------------------------ GAF ------
def normalize_series(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    v = values.copy()
    v[~finite] = np.nanmedian(v[finite])
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def resample_series(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.size == target_len:
        return values
    src = np.linspace(0.0, 1.0, num=values.size)
    dst = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(dst, src, values)


def exponential_phi(x_tilde: np.ndarray) -> np.ndarray:
    return math.pi * (np.exp(x_tilde) - 1.0) / (math.e - 1.0)


def gasf(phi: np.ndarray) -> np.ndarray:
    return np.cos(phi[:, None] + phi[None, :])


def gadf(phi: np.ndarray) -> np.ndarray:
    return np.sin(phi[:, None] - phi[None, :])


def factor_series(df: pd.DataFrame, feature: str, firm: object | None = None):
    """Trajectory of one factor: a single firm, or the cross-sectional median."""
    if firm is not None:
        g = df[df["firm_id"].astype(str) == str(firm)].sort_values("dt")
        return g["dt"].to_numpy(), pd.to_numeric(g[feature], errors="coerce").to_numpy()
    g = (df.groupby("dt")[feature].median().sort_index())
    return g.index.to_numpy(), g.to_numpy(dtype=float)


def fig_gaf_factor(df: pd.DataFrame | None, feature: str, firm: object | None = None,
                   img_size: int = 64):
    """Four-panel GAF card: time series, polar embedding, GASF and GADF."""
    df = load_bond_panel() if df is None else df
    dates, values = factor_series(df, feature, firm)
    if values.size < 4:
        raise ValueError("factor %r has too few observations" % feature)

    x = normalize_series(np.asarray(values, dtype=float))
    xr = resample_series(x, min(img_size, max(8, x.size)))
    phi = exponential_phi(xr)
    A, D = gasf(phi), gadf(phi)

    who = "median across firms" if firm is None else "firm %s" % firm
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.5))
    ax_ts, ax_pol, ax_a, ax_d = axes

    ax_ts.plot(dates, values, color="#2563eb", lw=1.2)
    ax_ts.set_title("%s - time series (%s)" % (feature, who), fontsize=9)
    ax_ts.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax_ts.tick_params(axis="y", labelsize=7)

    ax_pol.remove()
    ax_pol = fig.add_subplot(1, 4, 2, projection="polar")
    ax_pol.plot(phi, np.linspace(0, 1, phi.size), color="#7c3aed", lw=1.2)
    ax_pol.set_title("polar embedding  $\\phi=\\pi(e^{\\tilde x}-1)/(e-1)$", fontsize=9)
    ax_pol.tick_params(labelsize=6)

    im_a = ax_a.imshow(A, cmap="viridis", origin="lower", vmin=-1, vmax=1)
    ax_a.set_title(r"GASF  $\cos(\phi_i+\phi_j)$", fontsize=9)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

    im_d = ax_d.imshow(D, cmap="coolwarm", origin="lower", vmin=-1, vmax=1)
    ax_d.set_title(r"GADF  $\sin(\phi_i-\phi_j)$", fontsize=9)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------- lead time --------
LEAD_TABLE = "koopman_lead_time"


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic link p = 1 / (1 + exp(-z))."""
    z = np.clip(np.asarray(z, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def koopman_lead_time(df: pd.DataFrame | None = None, threshold_q: float = 0.99,
                      n_splits: int = 5, ridge: float = 1e-3,
                      target: str = "target_RS", seed: int = 42,
                      horizon_days: float = 365.0) -> dict:
    """Early-warning lead time from the Koopman + XGBoost model.

    Same definition as the survival EWS -- ``L = t_event - t_first_alarm`` -- but
    the alarm comes from a machine-learning score passed through a **sigmoid**:

        m_it  = XGBoost margin on [factors | Koopman forecast]
        p_it  = sigmoid(m_it) = 1 / (1 + exp(-m_it))
        alarm = 1[ p_it >= tau ],   tau = quantile(p, threshold_q)

    Scores are produced **out-of-fold with GroupKFold on firm_id**, so a firm is
    never scored by a model that saw it in training; otherwise the lead time is
    measured on memorised firms and is meaningless.
    """
    from sklearn.model_selection import GroupKFold

    df = load_bond_panel() if df is None else df
    feats = feature_columns(df)
    if target not in df.columns:
        raise ValueError("target column %r not in panel" % target)

    raw = df[feats].astype(float)
    mu, sd = raw.mean(), raw.std().replace(0, 1.0)
    dfs = df.copy()
    dfs[feats] = ((raw - mu) / sd).fillna(0.0)

    X_now, X_next, rows = build_koopman_pairs(dfs, feats)
    K = fit_linear_koopman_K(X_now, X_next, ridge=ridge)
    Z = X_now @ K.T
    X_full = np.hstack([X_now, Z, np.linalg.norm(Z, axis=1, keepdims=True)])

    y = pd.to_numeric(df.loc[rows, target], errors="coerce").fillna(0).to_numpy().astype(int)
    firms = df.loc[rows, "firm_id"].astype(str).to_numpy()
    dates = pd.to_datetime(df.loc[rows, "dt"], errors="coerce").to_numpy()

    # ---- out-of-fold margins (GroupKFold by firm) --------------------------
    margins = np.full(len(y), np.nan)
    n_splits = int(max(2, min(n_splits, len(np.unique(firms)))))
    for tr, te in GroupKFold(n_splits=n_splits).split(X_full, y, groups=firms):
        if len(np.unique(y[tr])) < 2:
            continue
        try:
            import xgboost as xgb
            mdl = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                    subsample=0.9, colsample_bytree=0.9,
                                    eval_metric="logloss", random_state=seed)
            mdl.fit(X_full[tr], y[tr])
            margins[te] = mdl.predict(X_full[te], output_margin=True)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            mdl = GradientBoostingClassifier(random_state=seed).fit(X_full[tr], y[tr])
            margins[te] = mdl.decision_function(X_full[te])

    ok = ~np.isnan(margins)
    p = np.full(len(y), np.nan)
    p[ok] = sigmoid(margins[ok])                       # <-- sigmoid link
    tau = float(np.nanquantile(p, threshold_q))
    alarm = ok & (p >= tau)

    # ---- lead time per firm that actually experienced the event ------------
    scored = pd.DataFrame({"firm_id": firms, "dt": dates, "y": y,
                           "p": p, "alarm": alarm})
    out = []
    for firm, g in scored.groupby("firm_id", sort=False):
        g = g.sort_values("dt")
        ev = g[g["y"] == 1]
        if ev.empty:
            continue                                   # censored firm
        t_event = ev["dt"].iloc[0]
        # Only alarms inside the pre-event horizon count. Without this window an
        # unrelated crossing years earlier would be reported as a multi-year
        # "lead time", which is a false alarm rather than an early warning.
        window_start = t_event - np.timedelta64(int(horizon_days), "D")
        pre = g[(g["alarm"]) & (g["dt"] < t_event) & (g["dt"] >= window_start)]
        if pre.empty:
            out.append({"firm_id": firm, "event_date": t_event, "first_alarm": pd.NaT,
                        "lead_time_days": np.nan, "detected": 0,
                        "max_p_before_event": float(g[g["dt"] < t_event]["p"].max())
                        if (g["dt"] < t_event).any() else np.nan})
        else:
            t_first = pre["dt"].iloc[0]
            out.append({"firm_id": firm, "event_date": t_event, "first_alarm": t_first,
                        "lead_time_days": float((t_event - t_first) / np.timedelta64(1, "D")),
                        "detected": 1,
                        "max_p_before_event": float(pre["p"].max())})

    tbl = pd.DataFrame(out)
    if not tbl.empty:
        tbl = tbl.sort_values("lead_time_days", ascending=False,
                              na_position="last").reset_index(drop=True)
    lead = tbl["lead_time_days"].dropna() if not tbl.empty else pd.Series(dtype=float)
    summary = {
        "threshold_q": float(threshold_q),
        "horizon_days": float(horizon_days),
        "tau": tau,
        "event_firms": int(len(tbl)),
        "detected": int(tbl["detected"].sum()) if not tbl.empty else 0,
        "detection_rate": float(tbl["detected"].mean()) if not tbl.empty else 0.0,
        "median_lead_days": float(lead.median()) if len(lead) else float("nan"),
        "mean_lead_days": float(lead.mean()) if len(lead) else float("nan"),
        "max_lead_days": float(lead.max()) if len(lead) else float("nan"),
        "alarm_rate": float(np.nanmean(alarm.astype(float))),
    }
    return {"table": tbl, "summary": summary, "scored": scored, "tau": tau}


def fig_lead_time(res: dict):
    """Distribution of the detected lead times plus the headline numbers."""
    tbl, s = res["table"], res["summary"]
    lead = tbl["lead_time_days"].dropna() if not tbl.empty else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    if len(lead):
        ax.hist(lead / 30.44, bins=min(20, max(4, len(lead))),
                color="#2563eb", edgecolor="white")
        ax.axvline(lead.median() / 30.44, color="#dc2626", ls="--", lw=1.6,
                   label="median %.1f months" % (lead.median() / 30.44))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "no firm was detected before its event",
                ha="center", va="center", color="#dc2626")
    ax.set_xlabel("lead time before the RS event (months)")
    ax.set_ylabel("firms")
    ax.set_title("Koopman + XGBoost (sigmoid alarm): detected %d / %d event firms  "
                 "(tau = %.4f)" % (s["detected"], s["event_firms"], s["tau"]), fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_lead_time_sqlite(res: dict, db_path: str, table: str = LEAD_TABLE) -> pd.DataFrame:
    import sqlite3
    tbl = res["table"].copy()
    for c in ("event_date", "first_alarm"):
        if c in tbl.columns:
            tbl[c] = tbl[c].astype(str)
    con = sqlite3.connect(db_path)
    try:
        tbl.to_sql(table, con, if_exists="replace", index=False)
        pd.DataFrame([{"metric": k, "value": float(v)}
                      for k, v in res["summary"].items()]).to_sql(
            table + "_summary", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return tbl


def fig_factor_series(df: pd.DataFrame | None, feature: str, firm: object | None = None):
    """Plain time-series graph of one factor (for the factor-selection panel)."""
    df = load_bond_panel() if df is None else df
    dates, values = factor_series(df, feature, firm)
    who = "median across firms" if firm is None else "firm %s" % firm
    fig, ax = plt.subplots(figsize=(9.0, 3.1))
    ax.plot(dates, values, color="#2563eb", lw=1.4)
    ax.fill_between(dates, values, color="#2563eb", alpha=0.10)
    ax.set_title("%s  (%s)" % (feature, who), fontsize=11, weight="bold")
    ax.set_ylabel(feature, fontsize=8)
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_gaf_gallery(df: pd.DataFrame | None = None, factors: list[str] | None = None,
                    ncols: int = 6, img_size: int = 48):
    """One GASF thumbnail per factor - the whole 34-factor panel on a single card."""
    df = load_bond_panel() if df is None else df
    factors = factors or feature_columns(df)
    n = len(factors)
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, 2.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, factors):
        try:
            _, values = factor_series(df, name)
            x = normalize_series(np.asarray(values, dtype=float))
            xr = resample_series(x, min(img_size, max(8, x.size)))
            ax.imshow(gasf(exponential_phi(xr)), cmap="viridis", origin="lower",
                      vmin=-1, vmax=1)
        except Exception:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
        ax.set_title(str(name)[:18], fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Gramian Angular Summation Field - all %d factors" % n,
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def export_gaf_images(df: pd.DataFrame | None = None, outdir: str | None = None,
                      factors: list[str] | None = None, img_size: int = 64) -> list[str]:
    """Write one four-panel GAF card per factor; returns the written paths."""
    df = load_bond_panel() if df is None else df
    factors = factors or feature_columns(df)
    outdir = outdir or os.path.join(HERE, "gaf_outputs")
    os.makedirs(outdir, exist_ok=True)
    written = []
    for i, name in enumerate(factors, start=1):
        try:
            fig = fig_gaf_factor(df, name, img_size=img_size)
        except Exception:
            continue
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
        path = os.path.join(outdir, "gaf_%02d_%s.jpg" % (i, safe))
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


# ------------------------------------------------------------------ CLI ------
def _selftest() -> None:
    df = load_bond_panel()
    feats = feature_columns(df)
    print("panel      : %d rows, %d firms, %d factors"
          % (len(df), df["firm_id"].nunique(), len(feats)))

    res = run_koopman_xgb(df)
    print("koopman    : %d pairs, spectral radius %.3f"
          % (res["n_pairs"], res["spectral_radius"]))
    print("metrics    :", {k: round(v, 4) for k, v in res["metrics"].items()})
    print("top factors:", ", ".join(res["importance"].head(5)["feature"].astype(str)))

    for fn, kw in ((fig_koopman_importance, {"res": res}),
                   (fig_koopman_spectrum, {"res": res})):
        fig = fn(**kw); assert fig is not None; plt.close(fig)
    fig = fig_gaf_factor(df, feats[0]); assert fig is not None; plt.close(fig)
    print("figures    : importance + spectrum + GAF(%s) built OK" % feats[0])
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__.strip().splitlines()[0])
        print("\nusage: python koopman_gaf.py --selftest")
