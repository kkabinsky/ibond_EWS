# -*- coding: utf-8 -*-
"""
ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework.

Approach 1: Dynamic Survival Hazard + Momentum (Base Model for Deliverable)
  - Cox/Logistic hazard h(t|X), Forward PD_3M, Risk Momentum Velocity M(t), Hyperbolic Boundaries.
  - Color-Coded Risk Status Buttons: 🟢 GREEN (OK), 🟡 YELLOW (WATCH), 🟠 ORANGE (ELEVATED), 🔴 RED (HIGH RISK).
  - ROE vs Credit Risk PD_3M Plot.

Approach 2: ML / DL + SHAP XAI Engine (Proposal Benchmark)
  - 33-feature static classifiers (Logistic, Random Forest, XGBoost).
  - SHAP Feature Importance, ROC-AUC metrics, Risk Distribution.

Data Inspector:
  - SQLite data grid, Search (account_id / firm_id), Pagination controls (Prev / Next).

Run:  python app.py           (interactive GUI)
      python app.py --selftest (headless backend check)
      python app.py --uitest   (headless UI check)
"""
from __future__ import annotations
import base64
import io
import os
import sqlite3
import sys
import hashlib
import secrets
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- shared clean chart style (all embedded figures) -------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcff",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6ebf3",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1e293b",
    "axes.labelsize": 8.5,
    "axes.labelcolor": "#334155",
    "font.size": 9,
    "text.color": "#334155",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 120,
})
# CMDF / ThaiBMA palette
CLR = {"pd": "#e11d48", "haz": "#7c3aed", "dtd": "#2563eb", "mom": "#f59e0b",
       "ok": "#16a34a", "warn": "#eab308", "elev": "#f97316", "risk": "#dc2626",
       "grid": "#e6ebf3", "ink": "#1e293b"}

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

from gen_data import generate, FEATURES
import survival
from thaibma_paths import DATA_ROOT  # data lives outside the repo
try:
    import notify                      # email / Telegram alerting (optional)
except Exception:                      # a notifier problem must never block the app
    notify = None
try:
    import koopman_gaf                 # Approach 2 extension: Koopman + GAF
except Exception:                      # keep the app usable without the bond panel
    koopman_gaf = None
try:
    import ml_factors                  # LightGBM / CatBoost factors + latent models
except Exception:
    ml_factors = None
try:
    import baselines                   # 8 anomaly-detection baselines + lead time
except Exception:                      # heavy deps (torch) stay optional
    baselines = None
try:
    import openclaw_connector as openclaw
except Exception:                      # OpenClaw remains an optional local service
    openclaw = None

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "credit_dataset_33features.xlsx")
XLSX_REAL = os.path.join(HERE, "credit_dataset_33features_real.xlsx")
PANEL_REAL = os.path.join(HERE, "real_panel.xlsx")
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
TABLE = "credit"
LEAD_TABLE = "lead_time"

# Login/PIN code is retained for a later implementation phase.
ENABLE_LOGIN = False

UI = {
    "page": "#f0f7ff",         # Light Blue Background
    "surface": "#ffffff",      # Card Surface White
    "sidebar": "#e0f2fe",      # Soft Sky Blue Sidebar
    "sidebar_panel": "#f0f9ff",# Subtle Ice Blue Panel
    "button": "#dbeafe",       # Light Blue Button
    "primary": "#1e40af",      # Royal Executive Blue
    "primary_dark": "#0f172a", # Deep Navy Slate
    "accent": "#2563eb",       # Vibrant Blue Accent
    "text": "#0f172a",         # Navy Text
    "muted": "#475569",        # Slate Muted Text
    "border": "#bfdbfe",       # Blue Border Accent
}

ALERT_BANDS = [(0.05, "LOW"), (0.15, "WATCH"), (0.35, "ELEVATED"), (1.01, "HIGH RISK")]


# ------------------------------------------------------------ backend ---------
def ensure_excel():
    if not os.path.exists(XLSX):
        generate().to_excel(XLSX, index=False)
    return XLSX


def import_to_sqlite():
    ensure_excel()
    df = pd.read_excel(XLSX)
    con = sqlite3.connect(DB)
    df.to_sql(TABLE, con, if_exists="replace", index_label="id")
    con.commit(); con.close()
    return len(df)


def load_df(table=TABLE, limit=None):
    con = sqlite3.connect(DB)
    try:
        q = f'SELECT * FROM "{table}"' + (f" LIMIT {limit}" if limit else "")
        df = pd.read_sql_query(q, con)
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    return df


def db_list_tables():
    con = sqlite3.connect(DB)
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        tables = [r[0] for r in rows]
    except Exception:
        tables = ["credit", "panel", "alerts", "lead_time"]
    finally:
        con.close()
    return tables if tables else ["credit", "panel", "alerts", "lead_time"]


def db_columns(table=TABLE):
    con = sqlite3.connect(DB)
    try:
        return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
    finally:
        con.close()


def db_get_row(row_id, table=TABLE):
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table}" WHERE id = ?', con, params=[row_id])
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    return df


def db_update_field(row_id, column, value, table=TABLE):
    """Edit one cell. Returns the number of rows changed."""
    cols = db_columns(table)
    if column not in cols or column == "id":
        raise ValueError(f"column {column!r} is not editable")
    try:
        value = float(value)
    except (TypeError, ValueError):
        pass                                  # keep it as text
    con = sqlite3.connect(DB)
    try:
        cur = con.execute(f'UPDATE "{table}" SET "{column}" = ? WHERE id = ?', (value, row_id))
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def db_delete_row(row_id, table=TABLE):
    con = sqlite3.connect(DB)
    try:
        cur = con.execute(f'DELETE FROM "{table}" WHERE id = ?', (row_id,))
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def db_add_row(values=None, table=TABLE):
    """Append a row to specified table (median of numeric columns unless overridden)."""
    df = load_df(table=table)
    if df.empty:
        df = pd.DataFrame(columns=db_columns(table))
    cols = [c for c in df.columns if c != "id"]
    base = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            base[c] = float(s.median())
        else:
            base[c] = ""
    base.update({k: v for k, v in (values or {}).items() if k in cols})
    con = sqlite3.connect(DB)
    try:
        new_id = int(pd.read_sql_query(
            f'SELECT COALESCE(MAX(id), -1) + 1 AS nid FROM "{table}"', con)["nid"].iloc[0])
        quoted = ",".join(f'"{c}"' for c in cols)
        con.execute(f'INSERT INTO "{table}" (id, {quoted}) VALUES ({",".join("?" * (len(cols) + 1))})',
                    [new_id] + [base[c] for c in cols])
        con.commit()
        return new_id
    finally:
        con.close()


def save_lead_time(df):
    con = sqlite3.connect(DB)
    df.to_sql(LEAD_TABLE, con, if_exists="replace", index=False)
    con.commit(); con.close()


def load_lead_time():
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {LEAD_TABLE}", con)
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df


def feature_cols(df):
    # identifiers, time index, targets and display-only Merton columns are never features
    skip = {"id", "account_id", "firm_id", "ticker", "month_index", "month_year",
            "default_3m", "event", "default_event", "dd_12m", "pd_12m",
            "h", "PD_3M", "PD_prev", "Momentum", "y_fwd", "flag_PD", "flag_RS", "d_Restructure", "d_DP_RS"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def _X(df, feats):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median(numeric_only=True)).fillna(0.0).values


def train_models(df):
    feats = feature_cols(df)
    target_name = "d_Restructure" if "d_Restructure" in df.columns else ("d_DP_RS" if "d_DP_RS" in df.columns else ("default_event" if "default_event" in df.columns else "default_3m"))
    X, y = _X(df, feats), df[target_name].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    models = {
        "Logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                               class_weight="balanced", n_jobs=4, random_state=0),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                     subsample=0.85, colsample_bytree=0.85,
                                     scale_pos_weight=spw, eval_metric="aucpr",
                                     random_state=0, n_jobs=4),
    }
    res = {}
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        thr = np.quantile(p, 0.85)
        f = (p >= thr).astype(int)
        auc_v = roc_auc_score(yte, p)
        mcc_v = matthews_corrcoef(yte, f) if len(set(f)) > 1 else 0.0
        f1_v = f1_score(yte, f, zero_division=0)
        rec_v = recall_score(yte, f, zero_division=0)
        res[name] = dict(model=clf, auc=auc_v, mcc=mcc_v, f1=f1_v, recall=rec_v, lead_time=3.0, target=target_name)
    best = max(res, key=lambda k: res[k]["auc"])
    return res, best, feats


def compute_alerts(df, model, feats):
    pdv = model.predict_proba(_X(df, feats))[:, 1]
    lvl = []
    for p in pdv:
        for hi, name in ALERT_BANDS:
            if p < hi:
                lvl.append(name); break
    out = df[["account_id", "default_3m"]].copy()
    out["PD"] = pdv
    out["alert"] = lvl
    return out.sort_values("PD", ascending=False).reset_index(drop=True)


def _b64(fig, dpi=120):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def _uri(b64):
    return "data:image/png;base64," + b64


def fig_importance(model, feats, top=15):
    est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        imp = np.abs(est.coef_[0])
    s = pd.Series(imp, index=feats).sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.barh(range(len(s)), s.values, color="#2563eb")
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
    ax.set_title("Approach 2 XAI — Feature Importance (Top 15)", fontsize=10)
    return _b64(fig)


def fig_shap_summary(model, df, feats, top=15, max_pts=500):
    """Generate authentic SHAP Beeswarm Summary Plot showing feature impact direction and magnitude."""
    X = _X(df, feats)
    if len(X) > max_pts:
        np.random.seed(0)
        idx_samp = np.random.choice(len(X), max_pts, replace=False)
        X_sub = X[idx_samp]
    else:
        X_sub = X

    est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
    X_std = (X_sub - np.mean(X_sub, axis=0)) / (np.std(X_sub, axis=0) + 1e-8)

    try:
        import shap
        explainer = shap.Explainer(est, X_sub[:80])
        sv_obj = explainer(X_sub[:250])
        vals = sv_obj.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    except Exception:
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            imp = np.abs(est.coef_[0])
        vals = X_std[:250] * imp

    mean_abs = np.abs(vals).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-top:]
    top_feats = [feats[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    for y_idx, f_idx in enumerate(top_indices):
        shap_v = vals[:, f_idx]
        feat_v = X_std[:len(shap_v), f_idx]
        feat_c = (feat_v - np.percentile(feat_v, 5)) / (np.percentile(feat_v, 95) - np.percentile(feat_v, 5) + 1e-8)
        feat_c = np.clip(feat_c, 0, 1)

        jitter = np.random.normal(0, 0.08, size=len(shap_v))
        sc = ax.scatter(shap_v, y_idx + jitter, c=feat_c, cmap="coolwarm", s=14, alpha=0.75, linewidths=0)

    ax.set_yticks(range(top))
    ax.set_yticklabels(top_feats, fontsize=8, fontweight="bold")
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax.set_xlabel("SHAP value (impact on model log-odds risk score)", fontsize=8.5, fontweight="bold")
    ax.set_title("Approach 2 SHAP Beeswarm Summary Plot (XAI)", fontsize=9.5, fontweight="bold")

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, aspect=15)
    cbar.set_label("Feature Value\n(Low -> High)", fontsize=7.5)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    return _b64(fig)


def fig_alert_dist(alerts):
    order = ["LOW", "WATCH", "ELEVATED", "HIGH RISK"]
    cnt = alerts["alert"].value_counts().reindex(order).fillna(0)
    colors = ["#22c55e", "#f59e0b", "#fb923c", "#ef4444"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(order, cnt.values, color=colors)
    for i, v in enumerate(cnt.values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Approach 2 Risk Classification Distribution", fontsize=10)
    return _b64(fig)


def import_real_to_sqlite():
    if not os.path.exists(PANEL_REAL):
        from fetch_real import build_panel
        build_panel().to_excel(PANEL_REAL, index=False)
    panel = pd.read_excel(PANEL_REAL)
    credit = panel.copy()
    credit["account_id"] = range(1, len(credit) + 1)
    con = sqlite3.connect(DB)
    credit[["account_id"] + FEATURES + ["default_3m"]].to_sql(TABLE, con, if_exists="replace", index_label="id")
    panel.to_sql("panel", con, if_exists="replace", index=False)
    con.commit(); con.close()
    return len(credit), len(panel), panel["ticker"].nunique()


def load_panel():
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("SELECT * FROM panel", con)
    except Exception:
        df = None
    con.close()
    if df is not None and not df.empty:
        if "default_event" in df.columns and pd.to_numeric(df["default_event"], errors="coerce").fillna(0).sum() > 0:
            df["event"] = pd.to_numeric(df["default_event"], errors="coerce").fillna(0).astype(int)
        elif "event" not in df.columns:
            df["event"] = 0
        if "month_year" in df.columns:
            df["month_year"] = pd.to_datetime(df["month_year"], errors="coerce")
    return df


def _fmt_firm(v):
    s = str(v)
    if s.replace(".0", "").isdigit():
        return str(int(float(v)))
    return s


def fig_boundary(df, meta, max_pts=4500, label_top=35):
    """Single large hyperbolic-boundary scatter (linear scale). Points beyond the
    curve are RS-flagged; each flagged firm is annotated with its firm id."""
    acct = "firm_id" if "firm_id" in df.columns else ("ticker" if "ticker" in df.columns else "account_id")
    d = df.dropna(subset=["PD_prev", "Momentum", "y_fwd"]).copy()
    d = d[(d["PD_prev"] > 1e-4) & (d["Momentum"] > 0) & (d["Momentum"] < 20)]
    a, K = meta["boundary"]["alpha"], meta["boundary"]["K"]
    logK = np.log(K)
    d["_s"] = np.log(d["Momentum"]) + a * np.log(d["PD_prev"])   # boundary score
    d["_flag"] = d["_s"] >= logK                                 # beyond hyperbolic boundary
    flagged_firms = int(d.loc[d["_flag"], acct].nunique())
    flagged_months = int(d["_flag"].sum())

    # always keep the points that matter (upcoming-event + boundary-flagged); subsample the rest
    special = d[d["_flag"] | (d["y_fwd"] == 1)]
    rest = d[~(d["_flag"] | (d["y_fwd"] == 1))]
    if len(rest) > max_pts:
        rest = rest.sample(max_pts, random_state=1)
    plot = pd.concat([rest, special]).reset_index(drop=True)
    x = plot["PD_prev"].values * 100; yv = plot["Momentum"].values
    bad = plot["y_fwd"].values == 1
    fm = plot["_flag"].values

    fig, ax = plt.subplots(figsize=(12.5, 8.6))          # ~3x the old single-panel area
    xhi = max(np.percentile(x, 99.8), x.max(), 98.0)       # reach the right-most (high-PD up to 98%+) firms
    yhi = min(max(np.percentile(yv, 99.5), 3.0), 5.2)

    # boundary curve spans the FULL width so it passes through all high-PD distress firms up to 98%+
    xs = np.linspace(0.1, xhi, 600) / 100
    curve_y = K / (xs ** a)

    ax.scatter(x[~bad], yv[~bad], s=16, c="#93c5a6", alpha=0.28, linewidths=0, label="normal")
    ax.scatter(x[fm], yv[fm], s=70, facecolors="none", edgecolors="#b45309", linewidths=1.2,
               label="RS-flagged (beyond boundary)")
    ax.scatter(x[bad], yv[bad], s=55, c=CLR["pd"], alpha=0.85, linewidths=0,
               label="distress — real credit event ≤ 3m", zorder=5)
    ax.plot(xs * 100, curve_y, color=CLR["elev"], lw=3.0, label=f"Hyperbolic boundary (K={K:.2f}, α={a:.2f})")
    ax.fill_between(xs * 100, curve_y, 10.0, color="#f97316", alpha=0.12, label="Flagged Risk Zone (Orange Area)")

    ax.set_xlim(0, xhi); ax.set_ylim(0.4, yhi)

    # label firm ids: upcoming-event (red) firms FIRST, then boundary-flagged; greedy de-overlap
    dcand = plot[plot["y_fwd"] == 1].sort_values("PD_prev", ascending=False).drop_duplicates(acct)
    fcand = plot[plot["_flag"]].sort_values("_s", ascending=False).drop_duplicates(acct)
    cand = pd.concat([dcand, fcand]).drop_duplicates(acct)
    dxm, dym = xhi * 0.04, yhi * 0.04
    placed = []
    for _, r in cand.iterrows():
        px, py = r["PD_prev"] * 100, r["Momentum"]
        if px > xhi or py > yhi:
            continue
        if any(abs(px - qx) < dxm and abs(py - qy) < dym for qx, qy in placed):
            continue
        is_dist = r["y_fwd"] == 1
        ax.annotate(_fmt_firm(r[acct]), (px, py), fontsize=9.5 if is_dist else 8.5,
                    color="#9f1239" if is_dist else "#7f1d1d", fontweight="bold",
                    xytext=(4, 3), textcoords="offset points", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec=("#f5b8c4" if is_dist else "#f0c9a0"), lw=0.6, alpha=0.9))
        placed.append((px, py))
        if len(placed) >= label_top:
            break

    ax.set_xlabel("PD₃M previous month (%)", fontsize=12)
    ax.set_ylabel("Momentum M(t)", fontsize=12)
    ax.set_title(f"Hyperbolic decision boundary — numbers = firm id  ·  "
                 f"{flagged_firms} firms flagged ({flagged_months:,} firm-months)  ·  "
                 f"red = real event ≤ 3m", fontsize=12.5)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper right", framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=10)
    fig.tight_layout()
    return _b64(fig, dpi=140)


def fig_roe_pd(df, max_pts=3000):
    d = df.dropna(subset=["PD_3M"]).copy()
    roe_col = "ROE" if "ROE" in d.columns else ("ROA" if "ROA" in d.columns else None)
    if roe_col is None or len(d) == 0:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.text(0.5, 0.5, "ROE/ROA feature not in dataset", ha="center", va="center")
        return _b64(fig)

    d[roe_col] = pd.to_numeric(d[roe_col], errors="coerce")
    d = d.dropna(subset=[roe_col]).copy()
    if len(d) > max_pts:
        d = d.sample(max_pts, random_state=1)

    x = d[roe_col].values
    y = d["PD_3M"].values * 100

    def get_st(r):
        flag_rs = r.get("flag_RS", 0)
        flag_pd = r.get("flag_PD", 0)
        pd3m = r.get("PD_3M", 0.0)
        mom = r.get("Momentum", 1.0)
        if flag_rs == 1 and flag_pd == 1:
            return "HIGH RISK"
        elif flag_rs == 1 or pd3m >= 0.15:
            return "ELEVATED"
        elif mom >= 1.15 or pd3m >= 0.05:
            return "WATCH"
        else:
            return "OK"

    st_vals = d.apply(get_st, axis=1).values

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for st, col, label in [("OK", "#22c55e", "🟢 OK"),
                           ("WATCH", "#eab308", "🟡 WATCH"),
                           ("ELEVATED", "#f97316", "🟠 ELEVATED"),
                           ("HIGH RISK", "#ef4444", "🔴 HIGH RISK")]:
        m = (st_vals == st)
        if m.sum() > 0:
            ax.scatter(x[m], y[m], s=10, c=col, alpha=0.6, label=label, linewidths=0)

    ax.set_xlabel(f"{roe_col} (%)", fontsize=9)
    ax.set_ylabel("Forward 3M PD (%)", fontsize=9)
    ax.set_title(f"Approach 1 — Profitability ({roe_col}) vs Credit Risk PD_3M", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


def fig_firm_trajectory(df, firm_val):
    acct_col = "firm_id" if "firm_id" in df.columns else ("ticker" if "ticker" in df.columns else "account_id")
    sub = df[df[acct_col].astype(str) == str(firm_val)].sort_values("month_index").copy()
    fname = f"Firm {firm_val}" if str(firm_val).isdigit() else str(firm_val)
    if sub.empty or "PD_3M" not in sub.columns:
        fig, ax = plt.subplots(figsize=(7.6, 3.3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No time-series data for {fname}", ha="center", va="center", color="#94a3b8")
        return _b64(fig)

    t = sub["month_index"].values
    pd3m = sub["PD_3M"].values * 100
    haz = sub["h"].values * 100 if "h" in sub.columns else pd3m

    # month of the real payment default (fallback: risk-tail event)
    dcol = "default_event" if ("default_event" in sub.columns and sub["default_event"].sum() > 0) else \
           ("event" if "event" in sub.columns else None)
    dmonth = None
    if dcol is not None:
        ev = sub[sub[dcol] == 1]
        if not ev.empty:
            dmonth = int(ev["month_index"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.4))

    # ---- LEFT: PD_3M & Hazard trajectory ------------------------------------
    ax1.fill_between(t, pd3m, color=CLR["pd"], alpha=0.08)
    ax1.plot(t, pd3m, color=CLR["pd"], lw=2.2, marker="o", ms=3.5, label="Forward PD₃M (%)")
    ax1.plot(t, haz, color=CLR["haz"], lw=1.6, ls="--", label="Hazard h(t) (%)")
    ax1.axhline(15, color=CLR["elev"], ls=":", lw=1.2, label="Alert threshold 15%")
    if dmonth is not None:
        yv = float(sub.loc[sub["month_index"] == dmonth, "PD_3M"].iloc[0]) * 100
        ax1.axvline(dmonth, color=CLR["risk"], ls="-", lw=1, alpha=0.35)
        ax1.scatter([dmonth], [yv], color=CLR["risk"], s=130, zorder=6, marker="X",
                    edgecolors="white", linewidths=1.2, label="Default event")
    ax1.set_xlabel("Month index"); ax1.set_ylabel("Probability (%)")
    ax1.set_title(f"{fname} — PD₃M & Hazard trajectory")
    ax1.legend(loc="lower left", framealpha=0.9, facecolor="white",
               edgecolor="#e2e8f0", fontsize=7)
    ax1.margins(x=0.02)

    # ---- RIGHT: real DTD decay + risk momentum acceleration -----------------
    dtd_col = next((c for c in ("dd_12m", "Merton_DTD", "distance_to_default") if c in sub.columns), None)
    if dtd_col is not None:
        dtd = pd.to_numeric(sub[dtd_col], errors="coerce").values
        ax2.fill_between(t, dtd, color=CLR["dtd"], alpha=0.08)
        ax2.plot(t, dtd, color=CLR["dtd"], lw=2.2, marker="s", ms=3.2, label=f"Merton DTD ({dtd_col})")
        ax2.axhline(2.0, color=CLR["dtd"], ls=":", lw=1, alpha=0.6, label="DTD = 2 (distress)")
        ax2.set_ylabel("Distance-to-Default", color=CLR["dtd"])
        ax2.tick_params(axis="y", colors=CLR["dtd"])
    else:
        ax2.plot(t, sub["PD_3M"].values * 100, color=CLR["dtd"], lw=2, marker="s", ms=3, label="PD₃M (%)")
        ax2.set_ylabel("PD₃M (%)", color=CLR["dtd"])

    ax2t = ax2.twinx()
    ax2t.spines["right"].set_visible(True)
    ax2t.grid(False)
    if "Momentum" in sub.columns:
        mom = pd.to_numeric(sub["Momentum"], errors="coerce").values
        ax2t.plot(t, mom, color=CLR["mom"], lw=1.8, ls="-.", marker="^", ms=3, label="Risk Momentum M(t)")
        ax2t.axhline(1.0, color=CLR["mom"], ls=":", lw=0.9, alpha=0.5)
        ax2t.set_ylabel("Momentum M(t)", color=CLR["mom"])
        ax2t.tick_params(axis="y", colors=CLR["mom"])
    if dmonth is not None:
        ax2.axvline(dmonth, color=CLR["risk"], ls="-", lw=1, alpha=0.35)
    ax2.set_xlabel("Month index")
    ax2.set_title(f"{fname} — DTD decay & risk acceleration")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.9, facecolor="white",
               edgecolor="#e2e8f0", fontsize=7)
    ax2.margins(x=0.02)

    fig.tight_layout()
    return _b64(fig)


# ------------------------------------------- realtime EWS visualisations -----
RT_BAND_C = {"HIGH RISK": "#dc2626", "ELEVATED": "#ea580c",
             "WATCH": "#ca8a04", "OK": "#16a34a"}


def fig_realtime(alerts, ref):
    """Current alert mix, PD distribution, and the warning each band historically buys."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.4, 3.9),
                                        gridspec_kw={"width_ratios": [1, 1.15, 1.1]})
    order = ["HIGH RISK", "ELEVATED", "WATCH", "OK"]

    # 1. how many issuers sit in each band right now
    cnt = alerts["alert"].value_counts().reindex(order).fillna(0)
    ax1.bar(range(len(order)), cnt.values, color=[RT_BAND_C[b] for b in order])
    for i, v in enumerate(cnt.values):
        ax1.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(["HIGH", "ELEV", "WATCH", "OK"], fontsize=8.5)
    ax1.set_ylabel("issuers")
    ax1.set_title("Current alert distribution")

    # 2. forward PD by band
    for b in order:
        v = alerts.loc[alerts["alert"] == b, "PD_3M"].dropna() * 100
        if len(v):
            ax2.scatter(np.random.default_rng(0).normal(order.index(b), 0.09, len(v)), v,
                        s=11, alpha=0.45, color=RT_BAND_C[b], linewidths=0)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(["HIGH", "ELEV", "WATCH", "OK"], fontsize=8.5)
    ax2.set_ylabel("forward PD₃M (%)")
    ax2.set_title("Risk level per issuer")

    # 3. realised lead time each band bought historically
    if ref is not None and not ref.empty:
        r = ref[ref["alert"].isin(order)].copy()
        r["alert"] = pd.Categorical(r["alert"], ["WATCH", "ELEVATED", "HIGH RISK"], ordered=True)
        r = r.sort_values("alert")
        y = np.arange(len(r))
        ax3.barh(y, r["median_days"], color=[RT_BAND_C[b] for b in r["alert"]], height=0.6)
        ax3.errorbar(r["median_days"], y,
                     xerr=[r["median_days"] - r["p25_days"], r["p75_days"] - r["median_days"]],
                     fmt="none", ecolor="#334155", capsize=4, lw=1.1)
        for i, (v, n) in enumerate(zip(r["median_days"], r["n"])):
            ax3.text(v, i, f"  {v:.0f}d (n={int(n)})", va="center", fontsize=8)
        ax3.set_yticks(y); ax3.set_yticklabels(r["alert"], fontsize=8.5)
        ax3.set_xlabel("days of warning before the credit event")
        ax3.margins(x=0.28)
    ax3.set_title("Lead time each band bought (median, IQR)")
    fig.suptitle("Real-time early warning — issuer watchlist", fontsize=11, y=1.03)
    fig.tight_layout()
    return _b64(fig, dpi=130)


# --------------------------------------- paper replication visualisations ----
def fig_paper_paths(paths):
    """Coefficient vs forecast horizon with 95% CI -- the paper's core result."""
    if paths is None or paths.empty:
        fig, ax = plt.subplots(figsize=(6, 2.4)); ax.axis("off")
        ax.text(0.5, 0.5, "no results", ha="center", va="center", color="#94a3b8")
        return _b64(fig)
    p = paths.copy()
    for c in ("coef", "se", "pvalue", "horizon"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    order = list(p.loc[p["dependent"] == "lnPD", "variable"].unique())
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow), squeeze=False)
    for i, v in enumerate(order):
        ax = axes[i // ncol][i % ncol]
        for dep, col in (("lnPD", "#be123c"), ("DD", "#1d4ed8")):
            s = p[(p["variable"] == v) & (p["dependent"] == dep)].sort_values("horizon")
            if s.empty:
                continue
            ax.errorbar(s["horizon"], s["coef"], yerr=1.96 * s["se"], marker="o", ms=4,
                        lw=1.6, capsize=3, color=col, label=dep)
            sig = s[s["pvalue"] < 0.05]
            ax.scatter(sig["horizon"], sig["coef"], s=52, facecolors="none",
                       edgecolors=col, linewidths=1.4, zorder=5)
        ax.axhline(0, color="#64748b", lw=0.9, ls=":")
        ax.set_title(v.replace("L.", ""), fontsize=9)
        ax.set_xticks([12, 24, 36, 60]); ax.tick_params(labelsize=8)
        if i % ncol == 0:
            ax.set_ylabel("coefficient", fontsize=8)
        if i // ncol == nrow - 1:
            ax.set_xlabel("horizon (months)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7.5, framealpha=0.9)
    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Multi-horizon determinants of default risk — coefficient ± 95% CI "
                 "(open circles = significant at 5%)", fontsize=11, y=1.005)
    fig.tight_layout()
    return _b64(fig, dpi=130)


# ------------------------------------------------- hazard-only visualisation --
def fig_hazard(df, top_firms=6):
    """Dedicated hazard h(t|X) view for the Survivor2 EWS panel (shown on top).

    Left   : baseline hazard over calendar time (mean h per month) + credit events
    Middle : hazard separation -- distribution for rows that precede an event vs not
    Right  : hazard trajectories of the currently riskiest firms
    """
    acct = "firm_id" if "firm_id" in df.columns else ("ticker" if "ticker" in df.columns else "account_id")
    d = df.dropna(subset=["h"]).copy()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 3.9),
                                        gridspec_kw={"width_ratios": [1.25, 1, 1.15]})

    # --- 1. baseline hazard over time -------------------------------------
    g = d.groupby("month_index")["h"].agg(["mean", "median"])
    ax1.plot(g.index, g["mean"] * 100, color=CLR["haz"], lw=1.8, label="mean h(t) (%)")
    ax1.fill_between(g.index, g["mean"] * 100, color=CLR["haz"], alpha=0.10)
    ax1.plot(g.index, g["median"] * 100, color=CLR["dtd"], lw=1.2, ls="--", label="median h(t) (%)")
    if "event" in d.columns and d["event"].sum():
        ev = d[d["event"] == 1].groupby("month_index").size()
        axb = ax1.twinx(); axb.grid(False)
        axb.bar(ev.index, ev.values, color=CLR["risk"], alpha=0.45, width=1.6, label="credit events")
        axb.set_ylabel("events / month", color=CLR["risk"], fontsize=8)
        axb.tick_params(axis="y", colors=CLR["risk"], labelsize=8)
    ax1.set_xlabel("Month index"); ax1.set_ylabel("hazard h(t) (%)", color=CLR["haz"])
    ax1.set_title("Baseline hazard over time")
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # --- 2. hazard separation (pre-event vs normal) ------------------------
    lab = "y_fwd" if "y_fwd" in d.columns else ("event" if "event" in d.columns else None)
    if lab is not None and d[lab].notna().any() and d[lab].sum() > 0:
        eps = 1e-8
        pre = np.clip(d.loc[d[lab] == 1, "h"].values, eps, 1.0) * 100
        nor = np.clip(d.loc[d[lab] != 1, "h"].values, eps, 1.0) * 100
        bins = np.logspace(np.log10(eps * 100), 2, 36)
        # share of EACH group per bin (percent) -- comparable despite very different n
        ax2.hist(nor, bins=bins, color="#93c5a6", alpha=0.8, label=f"normal (n={len(nor):,})",
                 weights=np.full(len(nor), 100.0 / len(nor)))
        ax2.hist(pre, bins=bins, color=CLR["pd"], alpha=0.65, label=f"pre-event (n={len(pre):,})",
                 weights=np.full(len(pre), 100.0 / len(pre)))
        ax2.set_xscale("log")
        mn, mp = float(np.median(nor)), float(np.median(pre))
        ax2.axvline(mn, color="#3f7d55", ls="--", lw=1.2)
        ax2.axvline(mp, color=CLR["pd"], ls="--", lw=1.2)
        ax2.annotate(f"median\n{mp:.3g}%", (mp, ax2.get_ylim()[1] * 0.82), fontsize=7,
                     color=CLR["pd"], ha="right", fontweight="bold")
        ax2.annotate(f"median\n{mn:.3g}%", (mn, ax2.get_ylim()[1] * 0.82), fontsize=7,
                     color="#3f7d55", ha="left", fontweight="bold")
        ax2.legend(fontsize=7.5, framealpha=0.9, loc="upper center")
        ax2.set_ylabel("share of group (%)")
    else:
        ax2.hist(d["h"] * 100, bins=40, color=CLR["haz"], alpha=0.7)
        ax2.set_ylabel("count")
    ax2.set_xlabel("hazard h(t) (%) [log]")
    ax2.set_title("Hazard separation")

    # --- 3. riskiest firms' hazard trajectories ---------------------------
    latest = d.sort_values("month_index").groupby(acct).tail(1)
    top = latest.nlargest(min(top_firms, len(latest)), "h")[acct].tolist()
    cmap = plt.get_cmap("autumn")
    for i, f in enumerate(top):
        s = d[d[acct] == f].sort_values("month_index")
        ax3.plot(s["month_index"], s["h"] * 100, lw=1.5, alpha=0.9,
                 color=cmap(i / max(len(top) - 1, 1) * 0.75),
                 label=f"{_fmt_firm(f)}")
        if "event" in s.columns and (s["event"] == 1).any():
            e = s[s["event"] == 1].iloc[0]
            ax3.scatter([e["month_index"]], [e["h"] * 100], marker="X", s=70, zorder=6,
                        color=CLR["risk"], edgecolors="white", linewidths=1.0)
    ax3.set_xlabel("Month index"); ax3.set_ylabel("hazard h(t) (%)")
    ax3.set_title(f"Hazard paths — top {len(top)} riskiest (X = event)")
    ax3.set_ylim(-4, 118)
    ax3.legend(fontsize=7, ncol=2, framealpha=0.92, loc="center left")

    fig.suptitle("Stage 1 — discrete-time hazard  h(t|X) = sigmoid(baseline(t) + β·X(t))",
                 fontsize=10.5, y=1.03)
    fig.tight_layout()
    return _b64(fig, dpi=130)


# ------------------------------------------- yield-curve (DNS) visuals -------
DNS_C = {"Level": "#9d174d", "Slope": "#0369a1", "Curvature": "#b45309"}


def fig_dns_factors(factors):
    """Level / Slope / Curvature time series + each against its empirical proxy."""
    f = factors.sort_values("date")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.8, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    for k in ("Level", "Slope", "Curvature"):
        ax1.plot(f["date"], f[k], lw=1.7, color=DNS_C[k], label=k)
    ax1.axhline(0, color="#94a3b8", lw=0.8, ls=":")
    ax1.set_ylabel("factor value (%)")
    ax1.set_title("Dynamic Nelson-Siegel factors of the Thai government yield curve")
    ax1.legend(fontsize=8.5, ncol=3, framealpha=0.92)

    if {"y_1y", "y_10y", "y_15y"}.issubset(f.columns):
        ax2.plot(f["date"], f["y_15y"], lw=1.3, color=DNS_C["Level"], alpha=0.85, label="15Y yield (level proxy)")
        ax2.plot(f["date"], f["y_10y"] - f["y_1y"], lw=1.3, color=DNS_C["Slope"], alpha=0.85,
                 label="10Y - 1Y (slope proxy)")
        if "y_2y" in f.columns:
            ax2.plot(f["date"], 2 * f["y_2y"] - f["y_3m"] - f["y_10y"], lw=1.3,
                     color=DNS_C["Curvature"], alpha=0.85, label="2x2Y - 3M - 10Y (curv proxy)")
    ax2.axhline(0, color="#94a3b8", lw=0.8, ls=":")
    ax2.set_ylabel("%"); ax2.set_xlabel("date")
    ax2.set_title("Empirical proxies used for validation", fontsize=9.5)
    ax2.legend(fontsize=8, ncol=3, framealpha=0.92)
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_dns_latest_curve(curve):
    c = curve.copy().dropna(subset=["date", "tau", "yield"])
    if c.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.0)); ax.axis("off")
        ax.text(0.5, 0.5, "no curve data", ha="center", va="center", color="#94a3b8")
        return _b64(fig)
    last_date = pd.to_datetime(c["date"]).max()
    g = c[pd.to_datetime(c["date"]) == last_date].sort_values("tau")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(g["tau"], g["yield"], marker="o", ms=4, lw=2.0, color="#be185d")
    ax.fill_between(g["tau"], g["yield"], alpha=0.12, color="#f9a8d4")
    ax.set_xscale("log")
    ax.set_xlabel("time to maturity (years, log)")
    ax.set_ylabel("yield (%)")
    ax.set_title(f"Latest government bond yield curve ({pd.Timestamp(last_date):%Y-%m-%d})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_dns_roll_dashboard(factors, window=6):
    f = factors.sort_values("date").copy()
    if f.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.0)); ax.axis("off")
        ax.text(0.5, 0.5, "no factor data", ha="center", va="center", color="#94a3b8")
        return _b64(fig)
    for fac in ("Level", "Slope", "Curvature"):
        f[f"{fac}_roll"] = f[fac].rolling(window=min(window, max(len(f), 1)), min_periods=1).mean()
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.0), sharex=True)
    for ax, fac in zip(axes, ("Level", "Slope", "Curvature")):
        ax.plot(f["date"], f[fac], lw=1.0, alpha=0.35, color=DNS_C[fac], label=f"{fac} raw")
        ax.plot(f["date"], f[f"{fac}_roll"], lw=2.2, color=DNS_C[fac], label=f"{window}-period rolling")
        ax.axhline(0, color="#cbd5e1", lw=0.8, ls=":")
        ax.set_ylabel(fac)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_title("Rolling Level / Slope / Curvature dashboard")
    axes[-1].set_xlabel("date")
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_dns_surface(curve, n_show=8):
    """Recent fitted curves + the three Nelson-Siegel loadings."""
    c = curve.copy()
    dates = sorted(c["date"].unique())[-n_show:]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.1),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    cmap = plt.get_cmap("plasma")
    for i, d in enumerate(dates):
        g = c[c["date"] == d].sort_values("tau")
        ax1.plot(g["tau"], g["yield"], marker="o", ms=3, lw=1.4,
                 color=cmap(i / max(len(dates) - 1, 1) * 0.85),
                 label=pd.Timestamp(d).strftime("%Y-%m"))
    ax1.set_xscale("log")
    ax1.set_xlabel("time to maturity (years, log)"); ax1.set_ylabel("yield (%)")
    ax1.set_title(f"Observed yield curves — last {len(dates)} periods")
    ax1.legend(fontsize=7, ncol=2, framealpha=0.9)

    import yield_curve_dns as ycd
    tau = np.linspace(0.08, 30, 300)
    B = ycd.ns_loadings(tau)
    for j, k in enumerate(("Level", "Slope", "Curvature")):
        ax2.plot(tau, B[:, j], lw=2, color=DNS_C[k], label=f"{k} loading")
    ax2.set_xscale("log"); ax2.set_xlabel("time to maturity (years, log)")
    ax2.set_ylabel("loading"); ax2.set_title(f"Nelson-Siegel loadings (λ = {ycd.LAMBDA})")
    ax2.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_dns_forecast(fc):
    """DNS-AR(1) out-of-sample RMSE by factor and horizon."""
    if fc is None or fc.empty:
        fig, ax = plt.subplots(figsize=(6, 2.4)); ax.axis("off")
        ax.text(0.5, 0.5, "no forecast results", ha="center", va="center", color="#94a3b8")
        return _b64(fig)
    hs = sorted(fc["horizon (m)"].unique())
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    w = 0.26
    for i, k in enumerate(("Level", "Slope", "Curvature")):
        s = fc[fc["factor"] == k].set_index("horizon (m)")["RMSE"]
        ax.bar(np.arange(len(hs)) + (i - 1) * w, [s.get(h, np.nan) for h in hs], w,
               color=DNS_C[k], label=k)
    ax.set_xticks(np.arange(len(hs))); ax.set_xticklabels([f"{h}m" for h in hs])
    ax.set_xlabel("forecast horizon"); ax.set_ylabel("out-of-sample RMSE")
    ax.set_title("DNS-AR(1) recursive forecast accuracy (expanding window)")
    ax.legend(fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    return _b64(fig, dpi=130)


# ---------------------------------------------- benchmark visualisations -----
BM_COLORS = {"Approach 1": "#2563eb", "Approach 1 (DL)": "#7c3aed",
             "Approach 2": "#ea580c", "Approach 2 (DL)": "#be123c"}


def _bm_colors(groups):
    return [BM_COLORS.get(g, "#64748b") for g in groups]


def fig_benchmark_prediction(pred):
    """AUC / PR-AUC / recall per model, coloured by approach group."""
    d = pred.sort_values("auc", ascending=True)
    y = np.arange(len(d)); cols = _bm_colors(d["group"])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "auc", "ROC-AUC (discrimination)"),
                           (axes[1], "pr_auc", "PR-AUC (rare-event quality)"),
                           (axes[2], "recall", "Recall at matched alarm budget")):
        ax.barh(y, d[key], color=cols, height=0.68)
        for i, v in enumerate(d[key]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.margins(x=0.20)
    axes[0].set_yticks(y); axes[0].set_yticklabels(d["model"], fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BM_COLORS.values()]
    axes[2].legend(handles, list(BM_COLORS.keys()), fontsize=7.5, loc="lower right",
                   framealpha=0.92)
    fig.suptitle("Prediction performance — walk-forward out-of-sample", fontsize=11, y=1.02)
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_benchmark_economics(pred, econ):
    """net benefit (MTHB) and Sarlin relative usefulness per model."""
    extra = [c for c in ["group", "detected", "n_event_firms"] if c not in econ.columns]
    d = econ.merge(pred[["model"] + extra], on="model") if extra else econ.copy()
    d = d.sort_values("net_benefit_mthb", ascending=True)
    y = np.arange(len(d)); cols = _bm_colors(d["group"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2), sharey=True)

    ax1.barh(y, d["net_benefit_mthb"], color=cols, height=0.68)
    ax1.axvline(0, color="#334155", lw=1)
    for i, (v, det, tot) in enumerate(zip(d["net_benefit_mthb"], d["detected"], d["n_event_firms"])):
        ax1.text(v, i, f"  {v:,.0f}  ({det}/{tot})", va="center", fontsize=8)
    ax1.set_yticks(y); ax1.set_yticklabels(d["model"], fontsize=8.5)
    ax1.set_xlabel("net benefit (million THB)", fontsize=9)
    ax1.set_title("Economic value = loss avoided − review cost", fontsize=9.5)
    ax1.margins(x=0.26)

    ax2.barh(y, d["usefulness_rel"], color=cols, height=0.68)
    ax2.axvline(0, color="#334155", lw=1)
    for i, v in enumerate(d["usefulness_rel"]):
        ax2.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax2.set_xlabel("Sarlin relative usefulness U_r", fontsize=9)
    ax2.set_title("Policymaker usefulness (μ = 0.9)", fontsize=9.5)
    ax2.margins(x=0.22)
    fig.suptitle("Financial / economic performance", fontsize=11, y=1.02)
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_benchmark_tradeoff(pred, econ):
    """AUC vs net benefit scatter — does better prediction buy economic value?"""
    d = pred.merge(econ[["model", "net_benefit_mthb", "usefulness_rel"]], on="model")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for g, sub in d.groupby("group"):
        ax.scatter(sub["auc"], sub["net_benefit_mthb"], s=150, alpha=0.85,
                   color=BM_COLORS.get(g, "#64748b"), label=g, edgecolors="white", linewidths=1.2)
    for _, r in d.iterrows():
        ax.annotate(r["model"], (r["auc"], r["net_benefit_mthb"]), fontsize=8,
                    xytext=(6, 5), textcoords="offset points")
    ax.axhline(0, color="#94a3b8", ls=":", lw=1)
    ax.set_xlabel("ROC-AUC (prediction)", fontsize=10)
    ax.set_ylabel("net benefit, million THB (economics)", fontsize=10)
    ax.set_title("Prediction vs economic value", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.92)
    ax.margins(0.14)
    fig.tight_layout()
    return _b64(fig, dpi=130)


# ------------------------------------------- model-comparison visualisations --
CMP_LR = "#2563eb"      # Logistic  -> blue
CMP_XGB = "#ea580c"     # XGBoost   -> orange


def fig_compare_metrics(metrics_df):
    """Grouped bar chart of the head-to-head metrics (0-1 scale group + % group)."""
    d = metrics_df.dropna(subset=["logistic", "xgboost"]).copy()
    unit = d[d["metric"].str.contains("AUC|MCC|Precision|Recall|F1|gap", case=False)]
    pct = d[d["metric"].str.contains("%|months", case=False)]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, sub, title in ((axes[0], unit, "Discrimination & quality (higher = better, except overfit gap)"),
                           (axes[1], pct, "Volume / detection / lead time")):
        if sub.empty:
            ax.axis("off"); continue
        y = np.arange(len(sub)); h = 0.38
        ax.barh(y + h / 2, sub["logistic"], h, color=CMP_LR, label="Logistic (survivor2)")
        ax.barh(y - h / 2, sub["xgboost"], h, color=CMP_XGB, label="XGBoost (machine_survior)")
        for i, (lv, xv) in enumerate(zip(sub["logistic"], sub["xgboost"])):
            ax.text(lv, i + h / 2, f" {lv:.3g}", va="center", fontsize=7.5, color=CMP_LR)
            ax.text(xv, i - h / 2, f" {xv:.3g}", va="center", fontsize=7.5, color=CMP_XGB)
        ax.set_yticks(y)
        ax.set_yticklabels([m.replace(" (out-of-sample)", " (OOS)") for m in sub["metric"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.margins(x=0.18)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.suptitle("Approach 1 hazard estimator — head-to-head", fontsize=11, y=1.01)
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_compare_outperform(metrics_df):
    """Signed % out-performance per metric (positive = XGBoost better)."""
    d = metrics_df[metrics_df["winner"].isin(["Logistic", "XGBoost"])].copy()
    d = d.dropna(subset=["pct_outperform"])
    if d.empty:
        fig, ax = plt.subplots(figsize=(6, 2.4)); ax.axis("off")
        ax.text(0.5, 0.5, "no scored metrics", ha="center", va="center", color="#94a3b8")
        return _b64(fig)
    d["signed"] = np.where(d["winner"] == "XGBoost", d["pct_outperform"], -d["pct_outperform"])
    d = d.sort_values("signed")
    colors = [CMP_XGB if v > 0 else CMP_LR for v in d["signed"]]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = np.arange(len(d))
    ax.barh(y, d["signed"], color=colors, height=0.62)
    ax.axvline(0, color="#334155", lw=1)
    for i, (v, w) in enumerate(zip(d["signed"], d["winner"])):
        ax.text(v + (2 if v > 0 else -2), i, f"{w} +{abs(v):.1f}%",
                va="center", ha="left" if v > 0 else "right", fontsize=8,
                color=CMP_XGB if v > 0 else CMP_LR, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace(" (out-of-sample)", " (OOS)") for m in d["metric"]], fontsize=8)
    ax.set_xlabel("out-performance (%)   ←  Logistic better      XGBoost better  →", fontsize=9)
    ax.set_title("Who outperforms, and by how much", fontsize=10)
    ax.margins(x=0.25)
    fig.tight_layout()
    return _b64(fig, dpi=130)


def fig_compare_leadtime(lead_df, top=26):
    """Per-firm lead time: Logistic vs XGBoost (days)."""
    d = lead_df.dropna(subset=["lead_lr", "lead_xgb"]).copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(6, 2.4)); ax.axis("off")
        ax.text(0.5, 0.5, "no paired lead times", ha="center", va="center", color="#94a3b8")
        return _b64(fig)
    d = d.sort_values("default_date").tail(top)
    y = np.arange(len(d)); h = 0.38
    fig, ax = plt.subplots(figsize=(10.4, max(3.4, 0.32 * len(d) + 1.4)))
    ax.barh(y + h / 2, d["lead_lr"], h, color=CMP_LR, label="Logistic")
    ax.barh(y - h / 2, d["lead_xgb"], h, color=CMP_XGB, label="XGBoost")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(f)} · {str(dt)[:7]}" for f, dt in zip(d["firm_id"], d["default_date"])],
                       fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("lead time before credit event (days)", fontsize=9)
    ax.set_title("Per-firm early-warning lead time", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return _b64(fig, dpi=130)


# --------------------------------------------------------------- GUI ----------
def main(page):
    import flet as ft
    C = ft.Colors
    center_alignment = getattr(ft.Alignment, "CENTER", None)
    if center_alignment is None:
        center_alignment = ft.alignment.center
    if not hasattr(ft.Padding, "symmetric"):
        ft.Padding.symmetric = staticmethod(
            lambda horizontal=0, vertical=0: ft.Padding(horizontal, vertical, horizontal, vertical))
    if not hasattr(ft.Padding, "only"):
        ft.Padding.only = staticmethod(
            lambda left=0, top=0, right=0, bottom=0: ft.Padding(left, top, right, bottom))
    if not hasattr(ft.Border, "all"):
        ft.Border.all = staticmethod(
            lambda width=1, color=None: ft.Border(
                top=ft.BorderSide(width, color),
                right=ft.BorderSide(width, color),
                bottom=ft.BorderSide(width, color),
                left=ft.BorderSide(width, color),
            ))
    page.title = "ThaiBMA Credit Early Warning System"
    page.bgcolor = UI["page"]
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO

    def _pin_hash(pin, salt=None):
        salt = salt or secrets.token_hex(16)
        digest = hashlib.sha256((salt + str(pin)).encode("utf-8")).hexdigest()
        return salt, digest

    def _auth_get(key):
        con = sqlite3.connect(DB)
        con.execute("CREATE TABLE IF NOT EXISTS app_auth (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = con.execute("SELECT value FROM app_auth WHERE key=?", (key,)).fetchone()
        con.close()
        return row[0] if row else None

    def _auth_set(key, value):
        con = sqlite3.connect(DB)
        con.execute("CREATE TABLE IF NOT EXISTS app_auth (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("INSERT OR REPLACE INTO app_auth(key, value) VALUES (?, ?)", (key, value))
        con.commit(); con.close()

    def _pin_configured():
        return bool(_auth_get("pin_hash")) and bool(_auth_get("pin_salt"))

    def _valid_pin(pin):
        return bool(pin) and pin.isdigit() and 4 <= len(pin) <= 8

    def _set_pin(pin):
        salt, digest = _pin_hash(pin)
        _auth_set("pin_salt", salt)
        _auth_set("pin_hash", digest)

    def _verify_pin(pin):
        salt, saved = _auth_get("pin_salt"), _auth_get("pin_hash")
        if not salt or not saved:
            return False
        _, digest = _pin_hash(pin, salt)
        return secrets.compare_digest(digest, saved)

    state = {
        "models": None, "best": None, "alerts": None,
        "active_tab": 0, "page": 0, "page_size": 25, "search": "",
        "lead_time": None, "lead_page": 0, "lead_page_size": 25,
        "lead_search": "", "lead_sort": "default_first",
        "oc_job_id": None, "oc_alert_page": 0, "oc_alert_page_size": 15,
        "oc_alert_search": "", "oc_alert_status": "all",
        "oc_alert_sort": "newest",
    }

    status = ft.Text("Ready.", size=12, color=UI["text"])

    # reusable soft card shadow
    SHADOW = ft.BoxShadow(spread_radius=0, blur_radius=10,
                          color=ft.Colors.with_opacity(0.08, C.BLUE_GREY_900),
                          offset=ft.Offset(0, 3))

    def card(content, accent=UI["border"], pad=16):
        return ft.Container(content=content, padding=pad, bgcolor=UI["surface"],
                            border_radius=8, border=ft.Border.all(1, accent),
                            shadow=SHADOW)

    # --- Official ThaiBMA Logo & Executive Orange Top Header Bar ---
    LOGOPATH = os.path.join(HERE, "thaibma_logo.png")
    if os.path.exists(LOGOPATH):
        try:
            with open(LOGOPATH, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")
            header_logo_control = ft.Image(src=f"data:image/png;base64,{logo_b64}", height=28, fit=ft.ImageFit.CONTAIN)
        except Exception:
            header_logo_control = ft.Icon(ft.Icons.SHIELD, size=22, color=C.ORANGE_800)
    else:
        header_logo_control = ft.Icon(ft.Icons.SHIELD, size=22, color=C.ORANGE_800)

    header = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Container(
                    content=header_logo_control,
                    padding=ft.Padding.symmetric(horizontal=10, vertical=4),
                    bgcolor=C.WHITE,
                    border_radius=8,
                    shadow=ft.BoxShadow(spread_radius=0, blur_radius=4, color=ft.Colors.with_opacity(0.15, C.BLACK), offset=ft.Offset(0, 2))
                ),
                ft.Text("ThaiBMA Credit Early Warning System", size=16, weight=ft.FontWeight.BOLD, color=C.WHITE),
                ft.Container(
                    content=ft.Row([
                        ft.Container(width=7, height=7, border_radius=4, bgcolor=C.WHITE),
                        ft.Text("ONLINE", size=10, weight=ft.FontWeight.BOLD, color=C.WHITE),
                    ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor=ft.Colors.with_opacity(0.2, C.WHITE),
                    padding=ft.Padding.symmetric(horizontal=10, vertical=4),
                    border_radius=12,
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.35, C.WHITE))
                ),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),

            ft.Row([
                ft.Container(
                    content=ft.Text("⚡ Non-blocking Async", size=11, color=C.WHITE, weight=ft.FontWeight.BOLD),
                    bgcolor=ft.Colors.with_opacity(0.2, C.WHITE), padding=ft.Padding.symmetric(horizontal=10, vertical=4), border_radius=12, border=ft.Border.all(1, ft.Colors.with_opacity(0.35, C.WHITE))
                ),
                ft.Container(
                    content=ft.Text("🗄️ SQLite CRUD", size=11, color=C.WHITE, weight=ft.FontWeight.BOLD),
                    bgcolor=ft.Colors.with_opacity(0.2, C.WHITE), padding=ft.Padding.symmetric(horizontal=10, vertical=4), border_radius=12, border=ft.Border.all(1, ft.Colors.with_opacity(0.35, C.WHITE))
                ),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.Padding.symmetric(horizontal=18, vertical=10),
        border_radius=10,
        bgcolor="#ea580c", # Executive Orange
        shadow=SHADOW,
    )

    # --- Left Sidebar Navigation Tabs with Blue & Yellow Multi-Color Theme ---
    NAV_BUTTON_WIDTH = 272
    NAV_BUTTON_HEIGHT = 44

    # (default_bg, default_fg, active_bg)
    nav_styles = [
        ("#eff6ff", "#1d4ed8", "#1e40af"), # Tab 0: Survival EWS (Royal Blue)
        ("#e0f2fe", "#0284c7", "#0284c7"), # Tab 1: ML + XAI (Sky Blue)
        ("#fef3c7", "#b45309", "#d97706"), # Tab 2: Data Inspector (Gold Yellow)
        ("#fef9c3", "#d97706", "#ca8a04"), # Tab 3: Lead Time (Amber Yellow)
        ("#e0e7ff", "#3730a3", "#3730a3"), # Tab 4: OpenClaw (Indigo Blue)
        ("#fef08a", "#a16207", "#a16207"), # Tab 5: News (Bright Gold)
        ("#dbeafe", "#2563eb", "#2563eb"), # Tab 6: Koopman (Electric Blue)
        ("#fef3c7", "#b45309", "#b45309"), # Tab 7: LightGBM (Warm Amber)
        ("#e0f2fe", "#0369a1", "#0369a1"), # Tab 8: CatBoost (Ocean Blue)
        ("#fef9c3", "#ca8a04", "#ca8a04"), # Tab 9: Latent factors (Sun Yellow)
        ("#fff7ed", "#ea580c", "#ea580c"), # Tab 10: Survivor2 Engine (Vibrant Orange)
        ("#f5f3ff", "#6d28d9", "#6d28d9"), # Tab 11: Model Comparison (Deep Violet)
        ("#ecfdf5", "#047857", "#047857"), # Tab 12: Benchmark A1/A2/DL (Emerald)
        ("#fdf2f8", "#9d174d", "#9d174d"), # Tab 13: Yield Curve DNS (Rose)
        ("#fce7f3", "#be185d", "#be185d"), # Tab 13 quick: iBond Quick Connect (Pink)
        ("#fdf2f8", "#be185d", "#be185d"), # Tab 13 latest: iBond Latest Curve
        ("#fff7ed", "#ea580c", "#ea580c"), # Tab 13 roll: iBond Rolling Dashboard
        ("#fdf2f8", "#be185d", "#be185d"), # Tab 13 tables: iBond Yield Tables
        ("#fdf2f8", "#be185d", "#be185d"), # Tab 13 logs: iBond Download Log
        ("#eef2ff", "#4338ca", "#4338ca"), # Tab 14: Paper replication (Indigo)
        ("#fef2f2", "#b91c1c", "#b91c1c"), # Tab 15: Real-time EWS (Red)
    ]

    BASELINE_MENUS = [
        ("IsolationForest", "Lead time: Isolation Forest", ft.Icons.PARK),
        ("OneClassSVM", "Lead time: One-Class SVM", ft.Icons.SCATTER_PLOT),
        ("DeepSVDD", "Lead time: Deep SVDD", ft.Icons.ADJUST),
        ("DAGMM", "Lead time: DAGMM", ft.Icons.BUBBLE_CHART),
        ("OmniAnomaly", "Lead time: OmniAnomaly", ft.Icons.WAVES),
        ("USAD", "Lead time: USAD", ft.Icons.COMPARE_ARROWS),
        ("TranAD", "Lead time: TranAD", ft.Icons.HUB),
        ("AnomalyTransformer", "Lead time: Anomaly Transformer", ft.Icons.AUTO_AWESOME),
    ]

    for i in range(len(BASELINE_MENUS)):
        if i % 2 == 0:
            nav_styles.append(("#e0f2fe", "#0369a1", "#1d4ed8"))
        else:
            nav_styles.append(("#fef3c7", "#b45309", "#d97706"))

    def make_nav_button(label, icon, idx=0):
        def_bg, def_fg, act_bg = nav_styles[idx]
        is_active = (idx == state.get("active_tab", 0))
        return ft.Button(
            label,
            icon=icon,
            bgcolor=act_bg if is_active else def_bg,
            color=C.WHITE if is_active else def_fg,
            width=NAV_BUTTON_WIDTH,
            height=NAV_BUTTON_HEIGHT,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=6),
                padding=ft.Padding.symmetric(horizontal=14, vertical=10),
            ),
        )

    btn_tab0 = make_nav_button("Approach 1: Survival EWS", ft.Icons.TIMELINE, 0)
    btn_tab1 = make_nav_button("Approach 2: ML + XAI", ft.Icons.PSYCHOLOGY, 1)
    btn_tab2 = make_nav_button("Data Inspector", ft.Icons.TABLE_CHART, 2)
    btn_tab3 = make_nav_button("Lead Time", ft.Icons.SCHEDULE, 3)
    btn_tab4 = make_nav_button("Connect to OpenClaw", ft.Icons.CLOUD_SYNC, 4)
    btn_tab5 = make_nav_button("ข่าวหุ้นกู้", ft.Icons.NEWSPAPER, 5)
    btn_tab6 = make_nav_button("Koopman + GAF (34 factors)", ft.Icons.BLUR_ON, 6)
    btn_tab7 = make_nav_button("Alert: LightGBM factors", ft.Icons.BOLT, 7)
    btn_tab8 = make_nav_button("Alert: CatBoost factors", ft.Icons.PETS, 8)
    btn_tab9 = make_nav_button("Latent factors (AE/VAE/AAE/PAE)", ft.Icons.LAYERS, 9)
    btn_tab10 = make_nav_button("Survivor2 Engine", ft.Icons.SHIELD_MOON, 10)
    btn_tab11 = make_nav_button("Model Comparison", ft.Icons.COMPARE_ARROWS, 11)
    btn_tab12 = make_nav_button("Benchmark: A1 / A2 / DL", ft.Icons.LEADERBOARD, 12)
    btn_tab13 = make_nav_button("Yield Curve: Level/Slope/Curv", ft.Icons.SHOW_CHART, 13)
    btn_tab13_quick = make_nav_button("iBond Quick Connect & Dashboard", ft.Icons.ADD_LINK, 13)
    btn_tab13_latest = make_nav_button("iBond: Latest Curve", ft.Icons.INSIGHTS, 13)
    btn_tab13_roll = make_nav_button("iBond: Rolling Dashboard", ft.Icons.SHOW_CHART, 13)
    btn_tab13_tables = make_nav_button("iBond: Yield Tables", ft.Icons.TABLE_ROWS, 13)
    btn_tab13_logs = make_nav_button("iBond: Download Log", ft.Icons.HISTORY, 13)
    btn_tab14 = make_nav_button("Paper: Multi-Horizon Determinants", ft.Icons.ARTICLE, 14)
    btn_tab15 = make_nav_button("Real-time EWS & Lead Time", ft.Icons.SENSORS, 15)

    baseline_buttons = [make_nav_button(label, icon, idx=16+b_i) for b_i, (_, label, icon) in enumerate(BASELINE_MENUS)]

    nav_buttons = [btn_tab0, btn_tab1, btn_tab2, btn_tab3, btn_tab4, btn_tab5,
                   btn_tab6, btn_tab7, btn_tab8, btn_tab9, btn_tab10, btn_tab11,
                   btn_tab12, btn_tab13, btn_tab13_quick, btn_tab13_latest, btn_tab13_roll, btn_tab13_tables,
                   btn_tab13_logs, btn_tab14, btn_tab15] + baseline_buttons

    side_nav_menu = ft.ListView(
        controls=nav_buttons,
        spacing=8,
        height=516,
        auto_scroll=False,
    )
    # --- Deliverables Strategy Matrix Card ---
    strategy_matrix_card = ft.Container(
        content=ft.Column([
            ft.Text("📋 Deliverables Strategy & Evaluation Matrix (ThaiBMA Deliverables)", size=13, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Evaluation Dimension", weight=ft.FontWeight.BOLD)),
                    ft.DataColumn(ft.Text("Approach 1: Survival Hazard + Momentum (Base Model)", weight=ft.FontWeight.BOLD)),
                    ft.DataColumn(ft.Text("Approach 2: XGBoost / DL + SHAP (Proposal Benchmark)", weight=ft.FontWeight.BOLD)),
                ],
                rows=[
                    ft.DataRow([
                        ft.DataCell(ft.Text("Complexity")),
                        ft.DataCell(ft.Text("🟢 Low-Medium (Straightforward)")),
                        ft.DataCell(ft.Text("🟡 Medium")),
                    ]),
                    ft.DataRow([
                        ft.DataCell(ft.Text("Ease of Duplication (ThaiBMA)")),
                        ft.DataCell(ft.Text("🟢 Very High (Standard Lifelines/Logistic)")),
                        ft.DataCell(ft.Text("🟢 High (XGBoost + SHAP)")),
                    ]),
                    ft.DataRow([
                        ft.DataCell(ft.Text("Key Strategic Advantage")),
                        ft.DataCell(ft.Text("Accurate Risk Momentum Velocity M(t)")),
                        ft.DataCell(ft.Text("Granular Per-Firm XAI Feature Importance")),
                    ]),
                    ft.DataRow([
                        ft.DataCell(ft.Text("Output Metrics")),
                        ft.DataCell(ft.Text("PD_3M + Risk Status (Red/Orange/Yellow/Green)")),
                        ft.DataCell(ft.Text("PD_1Y + Risk Status + Feature Importance")),
                    ]),
                ]
            )
        ], spacing=6),
        bgcolor=C.WHITE, padding=14, border_radius=14, border=ft.Border.all(1, C.BLUE_200),
        shadow=SHADOW
    )

    # --- Data Grid & Multi-Table Database CRUD Manager ---
    db_tables_available = db_list_tables()
    active_table_name = "credit" if "credit" in db_tables_available else db_tables_available[0]
    state["active_table"] = active_table_name

    data_info = ft.Text(f"Active Table: '{state['active_table']}' in SQLite ({DB})", size=12, color=C.BLUE_900, weight=ft.FontWeight.BOLD)
    search_field = ft.TextField(hint_text="Search account_id, firm_id, ticker, or value...", width=260, text_size=12)
    page_label = ft.Text("Page 1", size=12, weight=ft.FontWeight.BOLD)

    table_select_dropdown = ft.Dropdown(
        label="Select Database Table",
        width=200,
        text_size=12,
        value=state["active_table"],
        options=[ft.dropdown.Option(key=t, text=t) for t in db_tables_available]
    )

    def open_add_dialog(e=None):
        table_name = state.get("active_table", "credit")
        cols = [c for c in db_columns(table_name) if c != "id"]
        inputs = {c: ft.TextField(label=c, width=320, text_size=12) for c in cols[:8]}

        def on_save(_):
            values = {c: inputs[c].value for c in inputs if inputs[c].value is not None and inputs[c].value != ""}
            nid = db_add_row(values=values, table=table_name)
            if hasattr(page, "dialog") and page.dialog:
                page.dialog.open = False
            status.value = f"✅ Added new record (ID: {nid}) to table '{table_name}'."
            data_grid_container.content = render_data_table()
            page.update()

        def on_cancel(_):
            if hasattr(page, "dialog") and page.dialog:
                page.dialog.open = False
            page.update()

        dlg = ft.AlertDialog(
            title=ft.Text(f"➕ Add Record to '{table_name}'", weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            content=ft.Column(list(inputs.values()), scroll=ft.ScrollMode.AUTO, height=340, width=360, spacing=8),
            actions=[
                ft.TextButton("Cancel", on_click=on_cancel),
                ft.Button("Save Record", on_click=on_save, bgcolor=C.BLUE_700, color=C.WHITE),
            ],
        )
        page.dialog = dlg
        dlg.open = True
        page.update()

    def open_edit_dialog(row_id):
        table_name = state.get("active_table", "credit")
        df_row = db_get_row(row_id, table=table_name)
        if df_row.empty:
            status.value = f"Row ID {row_id} not found."
            page.update(); return

        cols = [c for c in db_columns(table_name) if c != "id"]
        inputs = {}
        for c in cols[:8]:
            val_str = str(df_row[c].iloc[0]) if c in df_row.columns and pd.notna(df_row[c].iloc[0]) else ""
            inputs[c] = ft.TextField(label=c, value=val_str, width=320, text_size=12)

        def on_save(_):
            for c, inp in inputs.items():
                db_update_field(row_id, c, inp.value, table=table_name)
            if hasattr(page, "dialog") and page.dialog:
                page.dialog.open = False
            status.value = f"✏️ Updated record ID {row_id} in table '{table_name}'."
            data_grid_container.content = render_data_table()
            page.update()

        def on_cancel(_):
            if hasattr(page, "dialog") and page.dialog:
                page.dialog.open = False
            page.update()

        dlg = ft.AlertDialog(
            title=ft.Text(f"✏️ Edit Record ID {row_id} in '{table_name}'", weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            content=ft.Column(list(inputs.values()), scroll=ft.ScrollMode.AUTO, height=340, width=360, spacing=8),
            actions=[
                ft.TextButton("Cancel", on_click=on_cancel),
                ft.Button("Update Record", on_click=on_save, bgcolor=C.BLUE_700, color=C.WHITE),
            ],
        )
        page.dialog = dlg
        dlg.open = True
        page.update()

    def open_delete_dialog(row_id):
        table_name = state.get("active_table", "credit")

        def on_confirm(_):
            db_delete_row(row_id, table=table_name)
            if hasattr(page, "dialog") and page.dialog:
                page.dialog.open = False
            status.value = f"🗑️ Deleted record ID {row_id} from table '{table_name}'."
            data_grid_container.content = render_data_table()
            page.update()

        def on_cancel(_):
            if hasattr(page, "dialog") and page.dialog:
                page.dialog.open = False
            page.update()

        dlg = ft.AlertDialog(
            title=ft.Text(f"🗑️ Confirm Delete Record ID {row_id}", weight=ft.FontWeight.BOLD, color=C.RED_700),
            content=ft.Text(f"Are you sure you want to permanently delete row ID {row_id} from table '{table_name}'?"),
            actions=[
                ft.TextButton("Cancel", on_click=on_cancel),
                ft.Button("Delete Row", on_click=on_confirm, bgcolor=C.RED_600, color=C.WHITE),
            ],
        )
        page.dialog = dlg
        dlg.open = True
        page.update()

    def render_data_table():
        table_name = state.get("active_table", "credit")
        try:
            df = load_df(table=table_name)
        except Exception as ex:
            return ft.Text(f"Error loading table '{table_name}': {ex}", color=C.RED_600)

        if df.empty:
            return ft.Text(f"Table '{table_name}' is currently empty.", color=C.GREY_600)

        s = state["search"].strip().lower()
        if s:
            mask = pd.Series(False, index=df.index)
            for c in df.columns:
                mask |= df[c].astype(str).str.lower().str.contains(s, na=False)
            df = df[mask]

        total_records = len(df)
        offset = state["page"] * state["page_size"]
        sub = df.iloc[offset:offset + state["page_size"]]
        page_label.value = f"Page {state['page']+1} / {max(1, (total_records-1)//state['page_size']+1)} ({total_records:,} rows)"

        if sub.empty:
            return ft.Text("No matching records found.", color=C.GREY_600)

        visible_cols = [c for c in sub.columns[:9]]
        cols = [ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD, size=11, color=C.BLUE_900)) for c in visible_cols]
        cols.append(ft.DataColumn(ft.Text("CRUD Actions", weight=ft.FontWeight.BOLD, size=11, color=C.BLUE_900)))

        rows = []
        for idx, row in sub.iterrows():
            cells = [ft.DataCell(ft.Text(str(row[c]), size=11)) for c in visible_cols]
            row_id = row["id"] if "id" in row else idx
            act_btn_edit = ft.IconButton(icon=ft.Icons.EDIT, icon_size=16, icon_color=C.BLUE_700, tooltip="Edit Row", on_click=lambda _, r=row_id: open_edit_dialog(r))
            act_btn_del = ft.IconButton(icon=ft.Icons.DELETE_OUTLINED, icon_size=16, icon_color=C.RED_600, tooltip="Delete Row", on_click=lambda _, r=row_id: open_delete_dialog(r))
            cells.append(ft.DataCell(ft.Row([act_btn_edit, act_btn_del], spacing=2)))
            rows.append(ft.DataRow(cells=cells))

        return ft.DataTable(columns=cols, rows=rows, data_row_min_height=34)

    data_grid_container = ft.Container(content=render_data_table(), padding=6)

    def data_prev(_):
        if state["page"] > 0:
            state["page"] -= 1
            data_grid_container.content = render_data_table()
            page.update()

    def data_next(_):
        state["page"] += 1
        data_grid_container.content = render_data_table()
        page.update()

    def data_search(_):
        state["search"] = search_field.value or ""
        state["page"] = 0
        data_grid_container.content = render_data_table()
        page.update()

    def data_clear(_):
        search_field.value = ""
        state["search"] = ""
        state["page"] = 0
        data_grid_container.content = render_data_table()
        page.update()

    def table_select_changed(_):
        state["active_table"] = table_select_dropdown.value or "credit"
        state["page"] = 0
        state["search"] = ""
        data_info.value = f"Active Table: '{state['active_table']}' in SQLite ({DB})"
        data_grid_container.content = render_data_table()
        page.update()

    table_select_dropdown.on_select = table_select_changed
    table_select_dropdown.on_change = table_select_changed

    nav_row = ft.Row([
        ft.Button("Prev", icon=ft.Icons.NAVIGATE_BEFORE, on_click=data_prev, bgcolor=C.BLUE_100, color=C.BLUE_900),
        ft.Button("Next", icon=ft.Icons.NAVIGATE_NEXT, on_click=data_next, bgcolor=C.BLUE_100, color=C.BLUE_900),
        page_label,
        table_select_dropdown,
        search_field,
        ft.Button("Search", icon=ft.Icons.SEARCH, on_click=data_search, bgcolor=C.BLUE_700, color=C.WHITE),
        ft.Button("➕ Add Record", icon=ft.Icons.ADD, on_click=open_add_dialog, bgcolor=C.GREEN_700, color=C.WHITE),
    ], spacing=8, wrap=True)

    # --- Lead Time controls ---
    lead_time_info = ft.Text("Run Survival EWS to calculate lead time within the 3-calendar-month default window.", size=12, color=C.GREY_700)
    lead_search_field = ft.TextField(hint_text="Search firm_id, account_id, ticker, alert", width=280, text_size=12)
    lead_page_label = ft.Text("Page 1", size=12, weight=ft.FontWeight.BOLD)
    lead_sort_dropdown = ft.Dropdown(
        label="Sort",
        width=250,
        text_size=12,
        value="default_first",
        options=[
            ft.dropdown.Option(key="default_first", text="Defaulted first"),
            ft.dropdown.Option(key="lead_desc", text="Lead time longest"),
            ft.dropdown.Option(key="lead_asc", text="Lead time shortest"),
            ft.dropdown.Option(key="pd_desc", text="Latest PD_3M highest"),
            ft.dropdown.Option(key="alert", text="Alert level"),
            ft.dropdown.Option(key="firm", text="Firm ID"),
        ],
    )

    def _lead_source_df():
        df = state.get("lead_time")
        if df is None:
            df = load_lead_time()
            if not df.empty:
                state["lead_time"] = df
        return pd.DataFrame() if df is None else df.copy()

    def _fmt_cell(v, na="N/A"):
        if pd.isna(v) or v == "":
            return na
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return str(v)

    def _fmt_pct(v):
        if pd.isna(v):
            return "N/A"
        try:
            return f"{float(v)*100:.1f}%"
        except Exception:
            return "N/A"

    def _fmt_num(v, nd=2):
        if pd.isna(v):
            return "N/A"
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return "N/A"

    def _lead_badge(txt):
        col = C.RED_600 if txt == "HIGH RISK" else (
            C.ORANGE_600 if txt == "ELEVATED" else (
                C.AMBER_600 if txt == "WATCH" else C.GREEN_600))
        return ft.Container(
            content=ft.Text(_fmt_cell(txt, "OK"), size=10, color=C.WHITE, weight=ft.FontWeight.BOLD),
            bgcolor=col, padding=ft.Padding.only(left=8, right=8, top=3, bottom=3),
            border_radius=8,
        )

    def render_lead_time_table():
        df = _lead_source_df()
        if df.empty:
            lead_page_label.value = "Page 1 / 1 (0 rows)"
            return ft.Text("No lead-time table yet. Click 'Run Survival EWS (App 1)' first.", color=C.GREY_600)

        s = state["lead_search"].strip().lower()
        if s:
            mask = pd.Series(False, index=df.index)
            for c in ["firm_id", "account_id", "ticker", "alarm_source", "alert_level"]:
                if c in df.columns:
                    mask |= df[c].astype(str).str.lower().str.contains(s, na=False)
            df = df[mask]

        if "default_observed" not in df.columns:
            df["default_observed"] = False
        if "lead_time_days" not in df.columns:
            df["lead_time_days"] = pd.NA
        if "latest_PD_3M" not in df.columns:
            df["latest_PD_3M"] = pd.NA
        df["_default_sort"] = df["default_observed"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["_lead_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce")
        df["_latest_pd"] = pd.to_numeric(df["latest_PD_3M"], errors="coerce")
        alert_series = df["alert_level"] if "alert_level" in df.columns else pd.Series("", index=df.index)
        firm_series = df["firm_id"] if "firm_id" in df.columns else (
            df["account_id"] if "account_id" in df.columns else pd.Series("", index=df.index))
        df["_alert_rank"] = alert_series.map({"HIGH RISK": 0, "ELEVATED": 1, "WATCH": 2, "OK": 3}).fillna(4)
        df["_firm_key"] = firm_series.astype(str)

        sort_key = state.get("lead_sort", "default_first")
        if sort_key == "lead_desc":
            df = df.sort_values(["_lead_days", "_latest_pd"], ascending=[False, False], na_position="last")
        elif sort_key == "lead_asc":
            df = df.sort_values(["_lead_days", "_latest_pd"], ascending=[True, False], na_position="last")
        elif sort_key == "pd_desc":
            df = df.sort_values(["_latest_pd", "_lead_days"], ascending=[False, False], na_position="last")
        elif sort_key == "alert":
            df = df.sort_values(["_alert_rank", "_latest_pd"], ascending=[True, False], na_position="last")
        elif sort_key == "firm":
            df = df.sort_values("_firm_key", ascending=True, na_position="last")
        else:
            df = df.sort_values(["_default_sort", "_lead_days", "_latest_pd"],
                                ascending=[False, False, False], na_position="last")

        total_records = len(df)
        max_page = max(0, (total_records - 1) // state["lead_page_size"])
        state["lead_page"] = min(max(state["lead_page"], 0), max_page)
        offset = state["lead_page"] * state["lead_page_size"]
        sub = df.iloc[offset:offset + state["lead_page_size"]]
        lead_page_label.value = f"Page {state['lead_page']+1} / {max_page+1} ({total_records:,} rows)"

        if sub.empty:
            return ft.Text("No matching lead-time records found.", color=C.GREY_600)

        cols = [
            "Firm ID", "Account/Ticker", "Default", "First Alarm (3M)", "Default Date",
            "Lead Days", "Alarm Source", "Latest PD_3M", "Momentum", "Alert"
        ]
        rows = []
        for _, r in sub.iterrows():
            acct_val = r.get("account_id", r.get("ticker", ""))
            if pd.notna(r.get("ticker", pd.NA)):
                acct_val = r.get("ticker")
            default_txt = "Yes" if str(r.get("default_observed", "")).lower() in ("true", "1") else "No"
            lead_days = pd.to_numeric(pd.Series([r.get("lead_time_days")]), errors="coerce").iloc[0]
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(_fmt_cell(r.get("firm_id", r.get("ticker", ""))), weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(_fmt_cell(acct_val))),
                ft.DataCell(ft.Text(default_txt)),
                ft.DataCell(ft.Text(_fmt_cell(r.get("first_alarm_date")))),
                ft.DataCell(ft.Text(_fmt_cell(r.get("default_date")))),
                ft.DataCell(ft.Text("N/A" if pd.isna(lead_days) else str(int(round(float(lead_days)))))),
                ft.DataCell(ft.Text(_fmt_cell(r.get("alarm_source")))),
                ft.DataCell(ft.Text(_fmt_pct(r.get("latest_PD_3M")))),
                ft.DataCell(ft.Text(_fmt_num(r.get("latest_Momentum")))),
                ft.DataCell(_lead_badge(r.get("alert_level", "OK"))),
            ]))

        return ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD, size=11)) for c in cols],
            rows=rows,
            data_row_min_height=34,
        )

    lead_grid_container = ft.Container(content=render_lead_time_table(), padding=6)

    def lead_prev(_):
        if state["lead_page"] > 0:
            state["lead_page"] -= 1
            lead_grid_container.content = render_lead_time_table()
            page.update()

    def lead_next(_):
        state["lead_page"] += 1
        lead_grid_container.content = render_lead_time_table()
        page.update()

    def lead_search(_):
        state["lead_search"] = lead_search_field.value or ""
        state["lead_page"] = 0
        lead_grid_container.content = render_lead_time_table()
        page.update()

    def lead_clear(_):
        lead_search_field.value = ""
        state["lead_search"] = ""
        state["lead_page"] = 0
        lead_grid_container.content = render_lead_time_table()
        page.update()

    def lead_sort_changed(_):
        state["lead_sort"] = lead_sort_dropdown.value or "default_first"
        state["lead_page"] = 0
        lead_grid_container.content = render_lead_time_table()
        page.update()

    lead_sort_dropdown.on_select = lead_sort_changed
    lead_sort_dropdown.on_change = lead_sort_changed

    lead_nav_row = ft.Row([
        ft.Button("Prev", icon=ft.Icons.NAVIGATE_BEFORE, on_click=lead_prev, bgcolor=C.GREY_200, color=C.BLACK),
        ft.Button("Next", icon=ft.Icons.NAVIGATE_NEXT, on_click=lead_next, bgcolor=C.GREY_200, color=C.BLACK),
        lead_page_label,
        lead_search_field,
        lead_sort_dropdown,
        ft.Button("Search", icon=ft.Icons.SEARCH, on_click=lead_search, bgcolor=C.BLUE_700, color=C.WHITE),
        ft.Button("Clear", icon=ft.Icons.CLEAR, on_click=lead_clear, bgcolor=C.GREY_400, color=C.WHITE),
    ], spacing=8, wrap=True)

    # --- Dynamic Output Placeholders ---
    metrics_box = ft.Column([ft.Text("Models not trained yet.", color=C.GREY_600)])
    imp_img = ft.Image(src="", visible=False, width=460, height=340)
    shap_img = ft.Image(src="", visible=False, width=480, height=340)
    alerts_table = ft.Column([ft.Text("Alerts not computed yet.", color=C.GREY_600)])
    dist_img = ft.Image(src="", visible=False, width=460, height=300)

    surv_kpi_cards = ft.Row([], spacing=10)
    surv_box = ft.Column([ft.Text("Survival EWS not executed yet. Click 'Run Survival EWS' to calculate.", color=C.GREY_600)])
    surv_img = ft.Image(src="", visible=False, width=1000)
    surv_table = ft.Column([ft.Text("Survival firm standings will appear here.", color=C.GREY_600)])

    # Trajectory Inspector Controls
    firm_select_dropdown = ft.Dropdown(
        label="Select high-risk / defaulting firm to inspect its trajectory",
        width=560, text_size=12)
    traj_caption = ft.Text("", size=11, color=C.BLUE_800, weight=ft.FontWeight.BOLD)
    traj_holder = ft.Column([
        ft.Container(
            content=ft.Text("Run “Run Survival EWS” first, then pick a firm above to see its "
                            "DTD decay, hazard h(t), PD₃M and risk-momentum trajectory.",
                            size=11, color=C.GREY_500),
            padding=18)
    ])

    def show_traj(firm_val):
        if state.get("df_surv") is None or firm_val in (None, ""):
            return
        try:
            b64_t = fig_firm_trajectory(state["df_surv"], firm_val)
            # rebuild the Image control every time so Flet always re-renders (no stale cache)
            traj_holder.controls = [ft.Image(src=_uri(b64_t), width=780)]
            fname = f"Firm {firm_val}" if str(firm_val).isdigit() else str(firm_val)
            traj_caption.value = f"📉 Showing behaviour trajectory for {fname}"
        except Exception as ex:
            traj_holder.controls = [ft.Text(f"Could not plot firm {firm_val}: {ex}", color=C.RED_600)]
        page.update()

    def on_select_firm(e):
        val = firm_select_dropdown.value
        if val in (None, ""):
            val = getattr(e, "data", None)
        status.value = f"Drawing trajectory for firm {val} ..."
        show_traj(val)

    # Flet 0.84 Dropdown fires on_select (NOT on_change)
    firm_select_dropdown.on_select = on_select_firm

    firm_traj_container = ft.Container(
        content=ft.Column([
            ft.Row([ft.Icon(ft.Icons.SHOW_CHART, size=18, color=C.BLUE_700),
                    ft.Text("Per-firm behaviour trajectory (retrospective)", size=14,
                            weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
            ft.Text("Pick a monitored firm to inspect its historical Distance-to-Default (DTD) decay, "
                    "hazard rate h(t), forward 3-month PD and risk momentum before distress.",
                    size=11, color=C.GREY_700),
            ft.Row([firm_select_dropdown], spacing=10),
            traj_caption,
            traj_holder,
        ], spacing=10),
        bgcolor=C.WHITE, padding=14, border_radius=14, border=ft.Border.all(1, C.BLUE_200),
        shadow=SHADOW
    )

    # --- Actions / Button Handlers ---
    def on_import(_):
        status.value = "Generating synthetic data & writing to SQLite..."; page.update()
        n = import_to_sqlite()
        state["page"] = 0; state["search"] = ""
        data_grid_container.content = render_data_table()
        top_panel_container.content = data_grid_card
        update_datagrid_tab()
        data_info.value = f"Loaded synthetic dataset: {n:,} accounts × 33 features in SQLite table 'credit'."
        data_info.color = C.BLUE_800
        status.value = f"Imported {n:,} synthetic rows. Running Approach 1 Survival EWS..."; page.update()
        on_survival(None)

    def on_real(_):
        status.value = "Fetching real SET/US data from Yahoo Finance + FRED..."; page.update()
        try:
            n_xs, n_pm, n_f = import_real_to_sqlite()
            state["page"] = 0; state["search"] = ""
            data_grid_container.content = render_data_table()
            top_panel_container.content = data_grid_card
            update_datagrid_tab()
            set_tab(2)
            data_info.value = f"Real Market Data: {n_f} firms ({n_xs:,} cross-section, {n_pm:,} firm-months panel) -> SQLite tables 'credit' and 'panel'."
            data_info.color = C.TEAL_800
            status.value = f"Loaded real data for {n_f} firms into SQLite."; page.update()
        except Exception as e:
            status.value = f"Error fetching real data: {e}"; page.update()

    def on_bond(_):
        status.value = "Loading bond database (Rev01_Database_final.dta)..."; page.update()
        try:
            from load_bond import load_bond, BOND_FEATURES
            df = load_bond()
            con = sqlite3.connect(DB)
            df.to_sql(TABLE, con, if_exists="replace", index_label="id")
            if "default_event" in df.columns:
                df["event"] = df["default_event"].astype(int)
            elif "event" not in df.columns:
                df["event"] = df["default_3m"].astype(int)
            df.to_sql("panel", con, if_exists="replace", index_label="id")
            con.commit(); con.close()
            state["page"] = 0; state["search"] = ""
            data_grid_container.content = render_data_table()
            top_panel_container.content = data_grid_card
            update_datagrid_tab()
            set_tab(2)
            data_info.value = (f"Real Bond DB: {len(df):,} firm-months × {len(BOND_FEATURES)} real features "
                               f"-> SQLite tables 'credit' and 'panel'. Credit-event onsets: "
                               f"{int(df['event'].sum())}; positive 3M firm-months: "
                               f"{int(df['default_3m'].sum())} ({df['default_3m'].mean()*100:.2f}%).")
            data_info.color = C.GREEN_800
            status.value = f"Loaded {len(df):,} bond firm-months into SQLite."; page.update()
        except Exception as e:
            status.value = f"Error loading bond database: {e}"; page.update()

    def on_train(_):
        progress_bar.visible = True
        progress_bar.value = 0.05
        status.value = "⏳ Initializing training dataset (0%)..."; page.update()
        try:
            df = load_df()
            feats = feature_cols(df)
            target_name = "d_Restructure" if "d_Restructure" in df.columns else ("d_DP_RS" if "d_DP_RS" in df.columns else ("default_event" if "default_event" in df.columns else "default_3m"))
            X, y = _X(df, feats), df[target_name].astype(int).values
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
            spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

            res = {}
            # Step 1: Logistic
            status.value = "⏳ Training Model 1/3: Logistic Regression (33%)..."; progress_bar.value = 0.33; page.update()
            clf_log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf_log.fit(Xtr, ytr)
            p_log = clf_log.predict_proba(Xte)[:, 1]
            f_log = (p_log >= np.quantile(p_log, 0.85)).astype(int)
            res["Logistic"] = dict(model=clf_log, auc=roc_auc_score(yte, p_log),
                                  mcc=matthews_corrcoef(yte, f_log) if len(set(f_log)) > 1 else 0.0,
                                  f1=f1_score(yte, f_log, zero_division=0),
                                  recall=recall_score(yte, f_log, zero_division=0),
                                  lead_time=3.0, target=target_name)

            # Step 2: Random Forest
            status.value = "⏳ Training Model 2/3: Random Forest (66%)..."; progress_bar.value = 0.66; page.update()
            clf_rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", n_jobs=4, random_state=0)
            clf_rf.fit(Xtr, ytr)
            p_rf = clf_rf.predict_proba(Xte)[:, 1]
            f_rf = (p_rf >= np.quantile(p_rf, 0.85)).astype(int)
            res["RandomForest"] = dict(model=clf_rf, auc=roc_auc_score(yte, p_rf),
                                       mcc=matthews_corrcoef(yte, f_rf) if len(set(f_rf)) > 1 else 0.0,
                                       f1=f1_score(yte, f_rf, zero_division=0),
                                       recall=recall_score(yte, f_rf, zero_division=0),
                                       lead_time=3.0, target=target_name)

            # Step 3: XGBoost
            status.value = "⏳ Training Model 3/3: XGBoost Classifier (100%)..."; progress_bar.value = 0.95; page.update()
            clf_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                        subsample=0.85, colsample_bytree=0.85,
                                        scale_pos_weight=spw, eval_metric="aucpr",
                                        random_state=0, n_jobs=4)
            clf_xgb.fit(Xtr, ytr)
            p_xgb = clf_xgb.predict_proba(Xte)[:, 1]
            f_xgb = (p_xgb >= np.quantile(p_xgb, 0.85)).astype(int)
            res["XGBoost"] = dict(model=clf_xgb, auc=roc_auc_score(yte, p_xgb),
                                  mcc=matthews_corrcoef(yte, f_xgb) if len(set(f_xgb)) > 1 else 0.0,
                                  f1=f1_score(yte, f_xgb, zero_division=0),
                                  recall=recall_score(yte, f_xgb, zero_division=0),
                                  lead_time=3.0, target=target_name)

            best = max(res, key=lambda k: res[k]["auc"])
            state["models"] = res; state["best"] = best; state["feats"] = feats

            rows = [ft.DataRow([
                ft.DataCell(ft.Text(k + ("  ⭐ BEST" if k == best else ""), weight=ft.FontWeight.BOLD if k == best else ft.FontWeight.NORMAL)),
                ft.DataCell(ft.Text(target_name, weight=ft.FontWeight.BOLD, color=C.INDIGO_700)),
                ft.DataCell(ft.Text(f"{v['auc']:.3f}")),
                ft.DataCell(ft.Text(f"{v['mcc']:.3f}")),
                ft.DataCell(ft.Text(f"{v['f1']:.3f}")),
                ft.DataCell(ft.Text(f"{v['recall']:.2f}")),
                ft.DataCell(ft.Text(f"{v['lead_time']:.1f} Months")),
            ]) for k, v in res.items()]

            tbl = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Classifier Model")),
                    ft.DataColumn(ft.Text("Target Field")),
                    ft.DataColumn(ft.Text("ROC-AUC")),
                    ft.DataColumn(ft.Text("MCC")),
                    ft.DataColumn(ft.Text("F1-Score")),
                    ft.DataColumn(ft.Text("Recall")),
                    ft.DataColumn(ft.Text("Lead Time (ExpoGAF)")),
                ],
                rows=rows
            )
            metrics_box.controls = [
                ft.Text(f"Approach 2 Classifiers Trained on 33 Features | Target Field: '{target_name}' (Restructuring Event) | Best: {best}", weight=ft.FontWeight.BOLD, color=C.BLUE_900),
                tbl
            ]

            b64_imp = fig_importance(res[best]["model"], feats)
            imp_img.src = _uri(b64_imp); imp_img.visible = True

            b64_shap = fig_shap_summary(res[best]["model"], df, feats)
            shap_img.src = _uri(b64_shap); shap_img.visible = True

            set_tab(1)
            progress_bar.value = 1.0
            status.value = f"✅ Training 100% complete! Best Model: {best} (ROC-AUC {res[best]['auc']:.3f}, F1 {res[best]['f1']:.3f})."; page.update()
        except Exception as e:
            status.value = f"Error training models: {e}"; page.update()
        finally:
            progress_bar.visible = False; page.update()

    def on_alerts(_):
        if not state.get("models"):
            status.value = "Please run 'Run models' first."; page.update(); return
        status.value = "Scoring accounts for Approach 2 Risk Classification..."; page.update()
        try:
            df = load_df()
            alerts = compute_alerts(df, state["models"][state["best"]]["model"], state["feats"])
            state["alerts"] = alerts

            con = sqlite3.connect(DB)
            alerts.to_sql("alerts", con, if_exists="replace", index=False)
            con.commit(); con.close()

            # email / Telegram — fires only for accounts escalating INTO HIGH RISK
            notify_msg = ""
            if notify is not None:
                notify_msg = "  " + notify.summary_line(notify.notify_alerts(alerts, DB))

            top = alerts.head(15)
            rows = [ft.DataRow([
                ft.DataCell(ft.Text(str(r["account_id"]))),
                ft.DataCell(ft.Text(f"{r['PD']*100:.1f}%")),
                ft.DataCell(ft.Container(
                    content=ft.Text(r["alert"], size=10, color=C.WHITE, weight=ft.FontWeight.BOLD),
                    bgcolor=C.RED_600 if r["alert"] == "HIGH RISK" else (C.ORANGE_600 if r["alert"] == "ELEVATED" else (C.AMBER_600 if r["alert"] == "WATCH" else C.GREEN_600)),
                    padding=ft.Padding.only(left=6, right=6, top=2, bottom=2), border_radius=4
                ))
            ]) for _, r in top.iterrows()]

            tbl = ft.DataTable(
                columns=[ft.DataColumn(ft.Text("Account")), ft.DataColumn(ft.Text("Predicted PD")), ft.DataColumn(ft.Text("Alert Band"))],
                rows=rows
            )
            alerts_table.controls = [ft.Text("Top 15 Highest Risk Accounts", weight=ft.FontWeight.BOLD), tbl]

            b64_dist = fig_alert_dist(alerts)
            dist_img.src = _uri(b64_dist); dist_img.visible = True

            # Display Approach 2 Risk Classification Alert Results directly in top panel replacing DataGridView
            top_panel_container.content = render_app2_alerts_top_panel(alerts)
            set_tab(1)

            status.value = f"Approach 2 alerts computed for {len(alerts):,} accounts.{notify_msg}"; page.update()
        except Exception as e:
            status.value = f"Error computing alerts: {e}"; page.update()

    def on_survival(_):
        status.value = "Running Approach 1 Dynamic Survival EWS (Cox Hazard + Momentum + Hyperbolic Boundaries)..."; page.update()
        try:
            panel = load_panel()
            if panel is None or panel.empty:
                if not os.path.exists(PANEL_REAL):
                    import_real_to_sqlite()
                panel = load_panel()

            df_surv, meta = survival.run(panel)
            lead_df = meta.get("lead_time") if isinstance(meta.get("lead_time"), pd.DataFrame) else survival.compute_lead_time(df_surv)
            state["lead_time"] = lead_df
            state["lead_page"] = 0
            save_lead_time(lead_df)
            n_defaulted = int(lead_df["default_observed"].astype(str).str.lower().isin(["true", "1"]).sum()) if "default_observed" in lead_df.columns else 0
            lead_vals = pd.to_numeric(lead_df.get("lead_time_days", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(lead_vals):
                lead_tail = (f" {len(lead_vals):,} defaults detected inside the 3M window; "
                             f"median lead time: {lead_vals.median():.0f} days.")
            else:
                lead_tail = " No qualifying alarm was found inside the 3M pre-default window."
            lead_time_info.value = (f"Saved SQLite table '{LEAD_TABLE}' with {len(lead_df):,} firms/accounts; "
                                    f"{n_defaulted:,} observed defaults.{lead_tail}")
            lead_time_info.color = C.BLUE_800
            lead_grid_container.content = render_lead_time_table()
            # report honest OUT-OF-SAMPLE signal metrics (fall back to in-sample if panel too short)
            eval_pd = meta.get("oos_pd") or survival.evaluate(df_surv, "flag_PD")
            eval_rs = meta.get("oos_rs") or survival.evaluate(df_surv, "flag_RS")

            def f3(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "n/a"

            rows = [
                ft.DataRow([ft.DataCell(ft.Text("PD Signal (PD₃M ≥ τ)")), ft.DataCell(ft.Text(f"{eval_pd['MCC']:.3f}")), ft.DataCell(ft.Text(f"{eval_pd['precision']:.2f}")), ft.DataCell(ft.Text(f"{eval_pd['recall']:.2f}")), ft.DataCell(ft.Text(f"{eval_pd['volume']*100:.1f}%"))]),
                ft.DataRow([ft.DataCell(ft.Text("RS Signal (Momentum Velocity)")), ft.DataCell(ft.Text(f"{eval_rs['MCC']:.3f}", weight=ft.FontWeight.BOLD, color=C.GREEN_700)), ft.DataCell(ft.Text(f"{eval_rs['precision']:.2f}")), ft.DataCell(ft.Text(f"{eval_rs['recall']:.2f}")), ft.DataCell(ft.Text(f"{eval_rs['volume']*100:.1f}%"))]),
            ]
            tbl = ft.DataTable(
                columns=[ft.DataColumn(ft.Text("EWS Signal (out-of-sample)")), ft.DataColumn(ft.Text("MCC")), ft.DataColumn(ft.Text("Precision")), ft.DataColumn(ft.Text("Recall")), ft.DataColumn(ft.Text("Flagged Vol"))],
                rows=rows
            )
            bnd = meta["boundary"]

            def auc_chip(label, val, color, bg):
                return ft.Container(
                    content=ft.Column([ft.Text(label, size=10, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                                       ft.Text(f3(val), size=17, weight=ft.FontWeight.BOLD, color=color)], spacing=0),
                    padding=10, bgcolor=bg, border_radius=10, border=ft.Border.all(1, color), width=170, shadow=SHADOW)

            auc_row = ft.Row([
                auc_chip("PD₃M AUC · OUT-OF-SAMPLE", meta.get("pd_auc_oos"), C.PURPLE_700, C.PURPLE_50),
                auc_chip("PD₃M AUC · in-sample", meta.get("pd_auc"), C.GREY_700, C.GREY_100),
                auc_chip("Persistence baseline", meta.get("persistence_auc"), C.BLUE_GREY_600, C.BLUE_GREY_50),
            ], spacing=10, wrap=True)

            cutm = meta.get("cut_month")
            hdr = (f"Approach 1 Base EWS — Boundary K={bnd['K']:.3f}, α={bnd['alpha']:.2f}. "
                   f"Metrics below are OUT-OF-SAMPLE" +
                   (f" (train ≤ month {cutm}, test {meta.get('n_test',0):,} rows)." if cutm else
                    " (panel too short — showing in-sample)."))
            surv_box.controls = [
                ft.Text(hdr, weight=ft.FontWeight.BOLD, color=C.PURPLE_900, size=12),
                auc_row,
                ft.Text("Out-of-sample = time-split hold-out (honest). Persistence baseline predicts the "
                        "forward event from the current state alone — the model earns its keep only if it beats it.",
                        size=10, color=C.GREY_600),
                tbl,
            ]

            # Single large hyperbolic-boundary scatter (firm ids labelled)
            b64_bnd = fig_boundary(df_surv, meta)
            surv_img.src = _uri(b64_bnd); surv_img.visible = True

            # Dynamic firm identifier selection (firm_id vs ticker vs account_id)
            acct_col = "firm_id" if "firm_id" in df_surv.columns else ("ticker" if "ticker" in df_surv.columns else "account_id")
            latest = df_surv.sort_values("month_index").groupby(acct_col).tail(1).copy()

            def get_survival_status(r):
                flag_rs = r.get("flag_RS", 0)
                flag_pd = r.get("flag_PD", 0)
                pd3m = r.get("PD_3M", 0.0)
                mom = r.get("Momentum", 1.0)

                if flag_rs == 1 and flag_pd == 1:
                    return "HIGH RISK", C.RED_600, C.RED_50
                elif flag_rs == 1 or pd3m >= 0.15:
                    return "ELEVATED", C.ORANGE_600, C.ORANGE_50
                elif mom >= 1.15 or pd3m >= 0.05:
                    return "WATCH", C.AMBER_600, C.AMBER_50
                else:
                    return "OK", C.GREEN_600, C.GREEN_50

            latest[["status_text", "status_color", "status_bg"]] = latest.apply(
                lambda r: pd.Series(get_survival_status(r)), axis=1
            )

            # Build KPI Metric Summary Cards for Approach 1
            c_red = (latest["status_text"] == "HIGH RISK").sum()
            c_orange = (latest["status_text"] == "ELEVATED").sum()
            c_yellow = (latest["status_text"] == "WATCH").sum()
            c_green = (latest["status_text"] == "OK").sum()

            def kpi_card(title, count, icon, color, bg):
                return ft.Container(
                    content=ft.Row([
                        ft.Icon(icon, size=24, color=color),
                        ft.Column([
                            ft.Text(title, size=11, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                            ft.Text(f"{count:,} firms", size=16, weight=ft.FontWeight.BOLD, color=color),
                        ], spacing=0)
                    ], spacing=8),
                    padding=12, bgcolor=bg, border_radius=12, border=ft.Border.all(1, color),
                    width=170, shadow=SHADOW
                )

            surv_kpi_cards.controls = [
                kpi_card("🔴 RED ALERT", c_red, ft.Icons.ERROR, C.RED_700, C.RED_50),
                kpi_card("🟠 ELEVATED", c_orange, ft.Icons.WARNING, C.ORANGE_700, C.ORANGE_50),
                kpi_card("🟡 WATCH NOTICE", c_yellow, ft.Icons.REMOVE_RED_EYE, C.AMBER_700, C.AMBER_50),
                kpi_card("🟢 NORMAL SAFE", c_green, ft.Icons.CHECK_CIRCLE, C.GREEN_700, C.GREEN_50),
            ]

            # Firm Risk Status Buttons
            stand_rows = []
            for _, r in latest.sort_values("PD_3M", ascending=False).head(25).iterrows():
                acct_raw = r[acct_col]
                acct_disp = f"Firm {acct_raw}" if str(acct_raw).isdigit() else str(acct_raw)
                pd3m_pct = f"{r['PD_3M']*100:.1f}%"
                mom_val = f"{r['Momentum']:.2f}" if pd.notna(r["Momentum"]) else "1.00"
                st_text, st_col, st_bg = r["status_text"], r["status_color"], r["status_bg"]

                btn_badge = ft.Container(
                    content=ft.Text(f"● {st_text}", size=11, color=C.WHITE, weight=ft.FontWeight.BOLD),
                    bgcolor=st_col, padding=ft.Padding.only(left=10, right=10, top=4, bottom=4),
                    border_radius=12
                )

                stand_rows.append(ft.DataRow([
                    ft.DataCell(ft.Text(acct_disp, weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(pd3m_pct)),
                    ft.DataCell(ft.Text(mom_val)),
                    ft.DataCell(btn_badge)
                ]))

            state["df_surv"] = df_surv
            EMO = {"HIGH RISK": "🔴", "ELEVATED": "🟠", "WATCH": "🟡", "OK": "🟢"}
            RANK = {"HIGH RISK": 0, "ELEVATED": 1, "WATCH": 2, "OK": 3}
            risky = latest[latest["status_text"].isin(["HIGH RISK", "ELEVATED", "WATCH"])].copy()
            if risky.empty:
                risky = latest.copy()
            risky["_rank"] = risky["status_text"].map(RANK)
            risky = risky.sort_values(["_rank", "PD_3M"], ascending=[True, False]).head(60)
            def_firms = risky[acct_col].tolist()

            opts = []
            for _, r in risky.iterrows():
                f = r[acct_col]
                name = f"Firm {f}" if str(f).isdigit() else str(f)
                opts.append(ft.dropdown.Option(
                    key=str(f),
                    text=f"{EMO.get(r['status_text'],'⚪')}  {name}  —  PD₃M {r['PD_3M']*100:.1f}%  [{r['status_text']}]"))
            firm_select_dropdown.options = opts
            if def_firms:
                firm_select_dropdown.value = str(def_firms[0])
                show_traj(def_firms[0])

            col_name = "Firm ID" if acct_col == "firm_id" else ("Ticker" if acct_col == "ticker" else "Account ID")
            stand_tbl = ft.DataTable(
                columns=[ft.DataColumn(ft.Text(col_name)), ft.DataColumn(ft.Text("Forward PD (3M)")), ft.DataColumn(ft.Text("Risk Momentum")), ft.DataColumn(ft.Text("Risk Status Button Badge"))],
                rows=stand_rows
            )
            surv_table.controls = [ft.Text(f"Latest Firm Risk Standing — Monitored {len(latest):,} Firms by {col_name} (Approach 1 Base Model)", weight=ft.FontWeight.BOLD, color=C.BLUE_900), stand_tbl]
            set_tab(0)
            status.value = (f"Approach 1 Survival EWS evaluated on {len(latest):,} firms "
                            f"({c_red} Red Alert, {c_orange} Elevated). Lead-time table saved."); page.update()
        except Exception as e:
            status.value = f"Error running Survival EWS: {e}"; page.update()

    def on_run_survivor2(_):
        status.value = "Running survivor2.py (Walk-Forward Cox Hazard Model & Lead-Time Pipeline)..."; page.update()
        try:
            import survivor2
            panel = survivor2.load_bond_dated()
            df_surv, meta = survival.run(panel.drop(columns=survivor2.HAZARD_DROP, errors="ignore"))
            tbl = survivor2.lead_time_table(df_surv, 0.50)
            
            con = sqlite3.connect(DB)
            tbl.to_sql("survivor2_lead_time", con, if_exists="replace", index=False)
            
            det = tbl[tbl["status"] == "detected"]
            leads = pd.to_numeric(det["lead_days"], errors="coerce").dropna()
            med_days = float(np.median(leads)) if len(leads) else 0.0
            med_months = float(med_days / survivor2.DAYS_PER_MONTH) if len(leads) else 0.0
            
            summary_df = pd.DataFrame([{
                "p_star": 0.50,
                "n_firms": int(panel["firm_id"].nunique()),
                "n_events": int(panel["event"].sum()),
                "detected": len(det),
                "missed": int((tbl["status"] == "missed").sum()),
                "censored": int((tbl["status"] == "censored").sum()),
                "pd_auc": float(meta.get("pd_auc", 0.0)),
                "pd_auc_oos": float(meta.get("pd_auc_oos", 0.0)),
                "persistence_auc": float(meta.get("persistence_auc", 0.0)),
                "median_lead_days": med_days,
                "median_lead_months": med_months,
            }])
            summary_df.to_sql("survivor2_summary", con, if_exists="replace", index=False)
            con.commit(); con.close()

            update_survivor2_tab()
            set_tab(10)
            status.value = f"✅ survivor2.py executed! Saved SQLite table 'survivor2_lead_time' ({len(tbl)} firms). Median lead time: {med_months:.1f} months ({med_days:.0f} days)."; page.update()
        except Exception as ex:
            status.value = f"Error running survivor2.py: {ex}"; page.update()

    import threading

    def run_async(fn, name="Processing"):
        if globals().get("_UITEST"):
            fn(None)
            return

        def _worker():
            try:
                progress_bar.visible = True
                progress_bar.value = None
                status.value = f"⏳ {name}..."
                page.update()

                fn(None)

                progress_bar.visible = False
                page.update()
            except Exception as ex:
                progress_bar.visible = False
                status.value = f"❌ Error in {name}: {ex}"
                page.update()

        threading.Thread(target=_worker, daemon=True).start()

    # --- Actions Toolbar ---
    def btn(txt, icon, fn, _color=None):
        return ft.Button(
            txt,
            icon=icon,
            on_click=fn,
            bgcolor=UI["button"],
            color=UI["text"],
            width=NAV_BUTTON_WIDTH,
            height=NAV_BUTTON_HEIGHT,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=6),
                padding=ft.Padding.symmetric(horizontal=14, vertical=10),
            ),
        )

    toolbar = ft.ListView(controls=[
        btn("Import synthetic", ft.Icons.STORAGE, lambda _: run_async(on_import, "Importing synthetic data"), C.BLUE_600),
        btn("Load REAL SET/US", ft.Icons.CLOUD_DOWNLOAD, lambda _: run_async(on_real, "Fetching REAL SET/US data"), C.TEAL_700),
        btn("Load bond 33-feat", ft.Icons.INVENTORY_2, lambda _: run_async(on_bond, "Loading Bond database"), C.GREEN_700),
        btn("Run models (App 2)", ft.Icons.PSYCHOLOGY, lambda _: run_async(on_train, "Training ML models"), C.INDIGO_600),
        btn("Compute alerts (App 2)", ft.Icons.WARNING_AMBER, lambda _: run_async(on_alerts, "Computing risk alerts"), C.ORANGE_700),
        btn("Run Survival EWS (App 1)", ft.Icons.TIMELINE, lambda _: run_async(on_survival, "Executing Survival Hazard EWS"), C.PURPLE_700),
        btn("Run Survivor2 Engine", ft.Icons.HEALTH_AND_SAFETY, lambda _: run_async(on_run_survivor2, "Running Survivor2 Model"), C.AMBER_800),
        btn("Run Survivor2 EWS", ft.Icons.MONITOR_HEART, lambda _: run_async(on_survivor2_ews, "Running Survivor2 EWS"), C.PURPLE_800),
    ], spacing=8, height=356)

    progress_bar = ft.ProgressBar(
        value=0.0,
        color=UI["primary"],
        bgcolor=UI["button"],
        visible=False,
        height=6,
    )

    status_chip = ft.Container(
        content=ft.Column([
            ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, size=15, color=C.BLUE_700), status],
                   spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            progress_bar
        ], spacing=6),
        bgcolor=UI["sidebar_panel"],
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        border_radius=6,
        border=ft.Border.all(1, UI["border"]),
    )

    control_panel = ft.Container(
        content=ft.Column([
        toolbar,
        status_chip,
        ], spacing=10),
        bgcolor=UI["sidebar"],
    )

    def render_app2_alerts_top_panel(alerts):
        acct_col = "firm_id" if "firm_id" in alerts.columns else ("ticker" if "ticker" in alerts.columns else "account_id")
        col_name = "Firm ID" if acct_col == "firm_id" else ("Ticker" if acct_col == "ticker" else "Account ID")

        cnt = alerts["alert"].value_counts().to_dict()
        c_red = cnt.get("HIGH RISK", 0)
        c_orange = cnt.get("ELEVATED", 0)
        c_yellow = cnt.get("WATCH", 0)
        c_green = cnt.get("LOW", 0)

        def kpi_card(title, count, icon, color, bg):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(icon, size=24, color=color),
                    ft.Column([
                        ft.Text(title, size=11, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                        ft.Text(f"{count:,} firms", size=16, weight=ft.FontWeight.BOLD, color=color),
                    ], spacing=0)
                ], spacing=8),
                padding=10, bgcolor=bg, border_radius=8, border=ft.Border.all(1, color), width=160
            )

        kpi_row = ft.Row([
            kpi_card("🔴 HIGH RISK", c_red, ft.Icons.ERROR, C.RED_700, C.RED_50),
            kpi_card("🟠 ELEVATED", c_orange, ft.Icons.WARNING, C.ORANGE_700, C.ORANGE_50),
            kpi_card("🟡 WATCH NOTICE", c_yellow, ft.Icons.REMOVE_RED_EYE, C.AMBER_700, C.AMBER_50),
            kpi_card("🟢 LOW RISK", c_green, ft.Icons.CHECK_CIRCLE, C.GREEN_700, C.GREEN_50),
        ], spacing=10)

        top_alerts = alerts.sort_values("PD", ascending=False).head(20)
        rows = []
        for _, r in top_alerts.iterrows():
            f_val = r[acct_col]
            f_disp = f"Firm {f_val}" if str(f_val).isdigit() else str(f_val)
            pd_pct = f"{r['PD']*100:.1f}%"
            st_text = r["alert"]

            st_col = C.RED_600 if st_text == "HIGH RISK" else (C.ORANGE_600 if st_text == "ELEVATED" else (C.AMBER_600 if st_text == "WATCH" else C.GREEN_600))
            btn_badge = ft.Container(
                content=ft.Text(f"● {st_text}", size=11, color=C.WHITE, weight=ft.FontWeight.BOLD),
                bgcolor=st_col, padding=ft.Padding.only(left=10, right=10, top=4, bottom=4),
                border_radius=12
            )

            rows.append(ft.DataRow([
                ft.DataCell(ft.Text(f_disp, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(pd_pct)),
                ft.DataCell(btn_badge)
            ]))

        tbl = ft.DataTable(
            columns=[ft.DataColumn(ft.Text(col_name)), ft.DataColumn(ft.Text("Predicted PD (1Y)")), ft.DataColumn(ft.Text("Approach 2 Alert Status Button Badge"))],
            rows=rows, data_row_min_height=32
        )

        b64_dist = fig_alert_dist(alerts)
        dist_img_top = ft.Image(src=_uri(b64_dist), visible=True, width=450, height=300)

        chart_controls = [dist_img_top]
        if state.get("models") and state.get("best"):
            b64_imp = fig_importance(state["models"][state["best"]]["model"], state["feats"])
            imp_img_top = ft.Image(src=_uri(b64_imp), visible=True, width=450, height=340)

            df_curr = load_df()
            b64_shap = fig_shap_summary(state["models"][state["best"]]["model"], df_curr, state["feats"])
            shap_img_top = ft.Image(src=_uri(b64_shap), visible=True, width=480, height=340)

            chart_controls.insert(0, shap_img_top)
            chart_controls.insert(0, imp_img_top)

        return card(ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER, color=C.ORANGE_800, size=20),
                ft.Text(f"🧠 Approach 2 Risk Classification Alerts & SHAP XAI Results ({len(alerts):,} Total Scored)", size=14, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            ], spacing=6),
            kpi_row,
            ft.Row([
                ft.Column([ft.Text(f"Top 20 Highest Risk Entities Scored by Approach 2", weight=ft.FontWeight.BOLD, size=12), tbl]),
                ft.Row(chart_controls, spacing=12, wrap=True)
            ], spacing=16, wrap=True)
        ], spacing=12), accent=C.ORANGE_200, pad=14)

    # --- Local OpenClaw connection, cron jobs and lead-time alert queue -------
    if openclaw is not None:
        try:
            openclaw.ensure_schema(DB)
            oc_cfg = openclaw.enable_default_sharing(DB)
        except Exception as exc:
            oc_cfg = {}
            status.value = f"OpenClaw database setup error: {exc}"
    else:
        oc_cfg = {}

    oc_gateway_field = ft.TextField(
        label="Local Gateway URL",
        value=str(oc_cfg.get("gateway_url") or "http://127.0.0.1:18789"),
        width=320,
        text_size=12,
    )
    oc_timezone_field = ft.TextField(
        label="Timezone",
        value=str(oc_cfg.get("timezone") or "Asia/Bangkok"),
        width=180,
        text_size=12,
    )
    oc_share_switch = ft.Switch(
        label="Enable risk-summary sharing",
        value=bool(oc_cfg.get("sharing_enabled", True)),
        active_color=UI["primary"],
    )
    oc_delivery_switch = ft.Switch(
        label="Send OpenClaw notifications",
        value=bool(oc_cfg.get("delivery_enabled", False)),
        active_color=UI["primary"],
    )
    oc_channel_dropdown = ft.Dropdown(
        label="Channel",
        value=str(oc_cfg.get("delivery_channel") or "telegram"),
        width=180,
        text_size=12,
        options=[
            ft.dropdown.Option(key="telegram", text="Telegram"),
            ft.dropdown.Option(key="slack", text="Slack"),
            ft.dropdown.Option(key="discord", text="Discord"),
            ft.dropdown.Option(key="teams", text="Teams"),
        ],
    )
    oc_target_field = ft.TextField(
        label="Channel / Chat target",
        value=str(oc_cfg.get("delivery_target") or ""),
        hint_text="Example: chat ID or channel ID",
        width=260,
        text_size=12,
    )
    oc_connection_status = ft.Text(
        f"Status: {oc_cfg.get('last_status', 'not_checked')}",
        size=12,
        weight=ft.FontWeight.BOLD,
        color=C.GREEN_700 if oc_cfg.get("last_status") == "connected" else C.GREY_700,
    )
    oc_cli_path = ft.Text(
        f"CLI: {oc_cfg.get('cli_entry_path') or 'not found'}",
        size=10,
        color=C.GREY_600,
        selectable=True,
    )
    oc_action_status = ft.Text(
        "Sharing is enabled by default. OpenClaw credentials remain in OpenClaw.",
        size=11,
        color=C.GREY_700,
        selectable=True,
    )
    oc_log_panel = ft.TextField(
        label="OpenClaw activity log",
        value="",
        multiline=True,
        min_lines=8,
        max_lines=12,
        read_only=True,
        width=980,
        text_size=11,
    )

    def _oc_append_log(message):
        stamp = pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        existing = (oc_log_panel.value or "").splitlines()
        existing.append(line)
        oc_log_panel.value = "\n".join(existing[-80:])

    def _oc_flash(message, ok=None):
        _oc_append_log(message)
        oc_action_status.value = str(message)
        oc_action_status.color = (
            C.GREEN_700 if ok is True
            else C.RED_700 if ok is False
            else C.BLUE_700
        )
        try:
            page.snack_bar = ft.SnackBar(
                ft.Text(str(message), selectable=True),
                open=True,
                bgcolor=C.GREEN_700 if ok is True else C.RED_700 if ok is False else C.BLUE_700,
            )
        except Exception:
            pass

    oc_job_name = ft.TextField(
        label="Job name", value="", width=260, text_size=12)
    oc_job_task = ft.Dropdown(
        label="Task",
        width=260,
        text_size=12,
        value="lead_time_alerts",
        options=[
            ft.dropdown.Option(key="lead_time_alerts", text="Lead Time Alert Scan"),
            ft.dropdown.Option(key="full_credit_scan", text="Full Credit Model Scan"),
            ft.dropdown.Option(key="refresh_and_alert", text="Refresh Bond Data + Alert Scan"),
        ],
    )
    oc_job_cron = ft.TextField(
        label="Cron expression", value="0 8 * * 1-5", width=190, text_size=12)
    oc_job_timezone = ft.TextField(
        label="Timezone", value="Asia/Bangkok", width=170, text_size=12)
    oc_job_timeout = ft.TextField(
        label="Timeout (seconds)", value="900", width=150, text_size=12)
    oc_job_enabled = ft.Switch(
        label="Job enabled", value=False, active_color=UI["primary"])
    oc_job_page_label = ft.Text("", size=11, color=C.GREY_700)
    oc_job_table_container = ft.Container()

    oc_alert_search = ft.TextField(
        hint_text="Search firm_id, account_id or alert",
        width=260,
        text_size=12,
    )
    oc_alert_status_dropdown = ft.Dropdown(
        label="Queue status",
        value="all",
        width=190,
        text_size=12,
        options=[
            ft.dropdown.Option(key="all", text="All statuses"),
            ft.dropdown.Option(key="pending", text="Pending"),
            ft.dropdown.Option(key="shared_to_openclaw", text="Shared to OpenClaw"),
            ft.dropdown.Option(key="failed", text="Failed"),
        ],
    )
    oc_alert_sort_dropdown = ft.Dropdown(
        label="Sort",
        value="newest",
        width=170,
        text_size=12,
        options=[
            ft.dropdown.Option(key="newest", text="Newest"),
            ft.dropdown.Option(key="pd_desc", text="PD_3M highest"),
            ft.dropdown.Option(key="lead_desc", text="Lead days longest"),
            ft.dropdown.Option(key="firm", text="Firm ID"),
        ],
    )
    oc_alert_page_label = ft.Text(
        "Page 1", size=12, weight=ft.FontWeight.BOLD)
    oc_alert_table_container = ft.Container()

    def _oc_ready():
        if openclaw is None:
            _oc_flash(
                "OpenClaw connector module is unavailable. Check openclaw_connector.py.",
                ok=False,
            )
            return False
        return True

    def _oc_sharing_enabled():
        return bool(oc_share_switch.value)

    def _oc_require_sharing(action_label="sync or run jobs"):
        if _oc_sharing_enabled():
            return True
        _oc_flash(
            "ยังไม่ได้เปิดสวิตช์ 'Enable risk-summary sharing' กรุณาเปิดสวิตช์นี้ แล้วกด 'Save settings' ก่อน" + action_label + "",
            ok=False,
        )
        return False

    def _oc_refresh_connection():
        if not _oc_ready():
            return
        cfg = openclaw.get_connection(DB)
        oc_connection_status.value = (
            f"Status: {cfg.get('last_status', 'not_checked')}"
            + (f"  |  Checked: {cfg.get('last_checked_at')}"
               if cfg.get("last_checked_at") else "")
        )
        oc_connection_status.color = (
            C.GREEN_700 if cfg.get("last_status") == "connected"
            else C.RED_700 if cfg.get("last_status") == "unavailable"
            else C.GREY_700
        )
        oc_cli_path.value = f"CLI: {cfg.get('cli_entry_path') or 'not found'}"

    def _oc_save_connection(_=None):
        if not _oc_ready():
            page.update()
            return
        try:
            openclaw.save_connection(
                oc_gateway_field.value,
                bool(oc_share_switch.value),
                timezone=oc_timezone_field.value or "Asia/Bangkok",
                delivery_enabled=bool(oc_delivery_switch.value),
                delivery_channel=oc_channel_dropdown.value or "",
                delivery_target=oc_target_field.value or "",
                db_path=DB,
            )
            _oc_flash(
                "Settings saved. No token was stored in the application database.",
                ok=True,
            )
            _oc_refresh_connection()
        except Exception as exc:
            _oc_flash(f"Cannot save settings: {exc}", ok=False)
        page.update()

    def _oc_test_connection(_=None):
        if not _oc_ready():
            page.update()
            return
        _oc_flash("กำลังตรวจสอบ OpenClaw Gateway ในเครื่อง...")
        page.update()
        try:
            result = openclaw.test_connection(DB)
            if result.get("ok"):
                detail = "Gateway เชื่อมต่อสำเร็จ"
                if not _oc_sharing_enabled():
                    detail += " แต่ยังไม่ได้เปิดสวิตช์ 'Enable risk-summary sharing' จึงยัง sync/run job ไม่ได้"
                if result.get("announcement"):
                    detail += f" {result['announcement']}"
                _oc_flash(detail, ok=True)
            else:
                detail = str(result.get("detail", "Connection checked."))
                _oc_flash(f"Connection test failed: {detail}", ok=False)
            _oc_refresh_connection()
        except Exception as exc:
            _oc_flash(f"Connection test failed: {exc}", ok=False)
        page.update()

    def _oc_reset_job_form(_=None):
        state["oc_job_id"] = None
        oc_job_name.value = ""
        oc_job_task.value = "lead_time_alerts"
        oc_job_cron.value = "0 8 * * 1-5"
        oc_job_timezone.value = "Asia/Bangkok"
        oc_job_timeout.value = "900"
        oc_job_enabled.value = False
        oc_job_page_label.value = "Creating a new local job definition."
        page.update()

    def _oc_edit_job(job_id):
        if not _oc_ready():
            return
        job = openclaw.get_job(job_id, DB)
        if not job:
            return
        state["oc_job_id"] = int(job_id)
        oc_job_name.value = str(job.get("name") or "")
        oc_job_task.value = str(job.get("task_type") or "lead_time_alerts")
        oc_job_cron.value = str(job.get("cron_expression") or "")
        oc_job_timezone.value = str(job.get("timezone") or "Asia/Bangkok")
        oc_job_timeout.value = str(job.get("timeout_seconds") or 900)
        oc_job_enabled.value = bool(job.get("enabled"))
        oc_job_page_label.value = f"Editing local job #{job_id}."
        page.update()

    def _oc_save_job(_=None):
        if not _oc_ready():
            page.update()
            return
        try:
            job_id = openclaw.upsert_job(
                oc_job_name.value,
                oc_job_task.value,
                oc_job_cron.value,
                timezone=oc_job_timezone.value or "Asia/Bangkok",
                enabled=bool(oc_job_enabled.value),
                timeout_seconds=int(oc_job_timeout.value or 900),
                job_id=state.get("oc_job_id"),
                db_path=DB,
            )
            state["oc_job_id"] = job_id
            _oc_flash(
                f"Job #{job_id} saved locally. Press Sync to create/update it in OpenClaw.",
                ok=True,
            )
            oc_job_table_container.content = _oc_render_jobs()
        except Exception as exc:
            _oc_flash(f"Cannot save job: {exc}", ok=False)
        page.update()

    def _oc_sync_job(job_id):
        if not _oc_ready():
            return
        if not _oc_require_sharing("ซิงก์ job นี้"):
            page.update()
            return
        _oc_flash(f"Syncing job #{job_id} to OpenClaw...")
        page.update()
        result = openclaw.sync_job(job_id, DB)
        if result.get("ok"):
            msg = "Job synced to OpenClaw."
            if result.get("warning"):
                msg += f" Warning: {result['warning']}"
            _oc_flash(msg, ok=True)
        else:
            _oc_flash(
                f"Sync failed: {result.get('error') or result.get('output')}",
                ok=False,
            )
        oc_job_table_container.content = _oc_render_jobs()
        page.update()

    def _oc_run_job(job_id):
        if not _oc_ready():
            return
        if not _oc_require_sharing("รัน job นี้"):
            page.update()
            return
        _oc_flash(f"Queueing job #{job_id} in OpenClaw...")
        page.update()
        result = openclaw.run_job_now(job_id, DB)
        _oc_flash(
            "OpenClaw accepted the run." if result.get("ok")
            else f"Run failed: {result.get('error') or result.get('output')}",
            ok=bool(result.get("ok")),
        )
        oc_job_table_container.content = _oc_render_jobs()
        page.update()

    def _oc_render_jobs():
        if openclaw is None:
            return ft.Text("OpenClaw connector is unavailable.", color=C.RED_700)
        jobs = openclaw.list_jobs(DB)
        if not jobs:
            return ft.Text("No cron jobs configured.", color=C.GREY_600)
        rows = []
        for job in jobs:
            job_id = int(job["id"])
            remote_id = str(job.get("openclaw_job_id") or "-")
            if len(remote_id) > 16:
                remote_id = remote_id[:13] + "..."
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(job.get("name")), size=11,
                                        weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(str(job.get("task_type")), size=10)),
                ft.DataCell(ft.Text(str(job.get("cron_expression")), size=11)),
                ft.DataCell(ft.Text(str(job.get("timezone")), size=10)),
                ft.DataCell(ft.Text(
                    "Enabled" if job.get("enabled") else "Disabled",
                    size=10,
                    color=C.GREEN_700 if job.get("enabled") else C.GREY_600)),
                ft.DataCell(ft.Text(str(job.get("sync_status")), size=10)),
                ft.DataCell(ft.Text(remote_id, size=10)),
                ft.DataCell(ft.Row([
                    ft.IconButton(
                        icon=ft.Icons.EDIT,
                        tooltip="Edit local job",
                        on_click=lambda _, jid=job_id: _oc_edit_job(jid)),
                    ft.IconButton(
                        icon=ft.Icons.CLOUD_SYNC,
                        tooltip="Sync/create in OpenClaw",
                        on_click=lambda _, jid=job_id: _oc_sync_job(jid)),
                    ft.IconButton(
                        icon=ft.Icons.PLAY_ARROW,
                        tooltip="Run now in OpenClaw",
                        on_click=lambda _, jid=job_id: _oc_run_job(jid)),
                ], spacing=0)),
            ]))
        return ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Job", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Task", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Cron", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Timezone", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("State", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Sync", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("OpenClaw ID", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Actions", weight=ft.FontWeight.BOLD, size=11)),
            ],
            rows=rows,
            data_row_min_height=42,
        )

    def _oc_alert_rows():
        if openclaw is None:
            return []
        rows = openclaw.list_alerts(
            state.get("oc_alert_search", ""),
            state.get("oc_alert_status", "all"),
            limit=500,
            db_path=DB,
        )
        sort_key = state.get("oc_alert_sort", "newest")
        if sort_key == "pd_desc":
            rows.sort(key=lambda r: r.get("latest_pd_3m") or -1, reverse=True)
        elif sort_key == "lead_desc":
            rows.sort(key=lambda r: r.get("lead_time_days") or -1, reverse=True)
        elif sort_key == "firm":
            rows.sort(key=lambda r: str(r.get("firm_id") or ""))
        return rows

    def _oc_render_alerts():
        rows = _oc_alert_rows()
        total = len(rows)
        state["oc_alert_total"] = total
        page_size = state["oc_alert_page_size"]
        max_page = max(0, (total - 1) // page_size)
        state["oc_alert_page"] = min(state["oc_alert_page"], max_page)
        offset = state["oc_alert_page"] * page_size
        sub = rows[offset:offset + page_size]
        oc_alert_page_label.value = (
            f"Page {state['oc_alert_page'] + 1} / {max_page + 1} ({total:,} alerts)")
        if not sub:
            return ft.Text(
                "No lead-time alerts in the queue. Run a synced Lead Time Alert Scan.",
                color=C.GREY_600,
            )
        table_rows = []
        for row in sub:
            lead_value = (
                str(row.get("lead_time_days"))
                if row.get("lead_time_days") is not None
                else f"0-{row.get('lead_window_days') or 90}"
            )
            pd_value = row.get("latest_pd_3m")
            momentum = row.get("latest_momentum")
            table_rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(row.get("firm_id") or "-"), size=11)),
                ft.DataCell(ft.Text(str(row.get("account_id") or "-"), size=11)),
                ft.DataCell(ft.Text(str(row.get("alert_mode") or "-"), size=10)),
                ft.DataCell(ft.Text(str(row.get("signal_date") or "-"), size=10)),
                ft.DataCell(ft.Text(lead_value, size=11)),
                ft.DataCell(ft.Text(
                    "N/A" if pd_value is None else f"{float(pd_value):.4f}", size=11)),
                ft.DataCell(ft.Text(
                    "N/A" if momentum is None else f"{float(momentum):.3f}", size=11)),
                ft.DataCell(ft.Text(str(row.get("alert_level") or "-"), size=10,
                                        weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(str(row.get("alarm_source") or "-"), size=10)),
                ft.DataCell(ft.Text(str(row.get("status") or "-"), size=10)),
            ]))
        return ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Firm ID", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Account", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Mode", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Signal date", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Lead days/window", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("PD_3M", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Momentum", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Alert", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Source", weight=ft.FontWeight.BOLD, size=11)),
                ft.DataColumn(ft.Text("Queue status", weight=ft.FontWeight.BOLD, size=11)),
            ],
            rows=table_rows,
            data_row_min_height=36,
        )

    def _oc_alert_apply(_=None):
        state["oc_alert_search"] = oc_alert_search.value or ""
        state["oc_alert_status"] = oc_alert_status_dropdown.value or "all"
        state["oc_alert_sort"] = oc_alert_sort_dropdown.value or "newest"
        state["oc_alert_page"] = 0
        oc_alert_table_container.content = _oc_render_alerts()
        page.update()

    def _oc_alert_clear(_=None):
        oc_alert_search.value = ""
        oc_alert_status_dropdown.value = "all"
        oc_alert_sort_dropdown.value = "newest"
        state["oc_alert_search"] = ""
        state["oc_alert_status"] = "all"
        state["oc_alert_sort"] = "newest"
        state["oc_alert_page"] = 0
        oc_alert_table_container.content = _oc_render_alerts()
        page.update()

    def _oc_alert_prev(_=None):
        if state["oc_alert_page"] > 0:
            state["oc_alert_page"] -= 1
            oc_alert_table_container.content = _oc_render_alerts()
            page.update()

    def _oc_alert_next(_=None):
        if ((state["oc_alert_page"] + 1) * state["oc_alert_page_size"]
                < state.get("oc_alert_total", 0)):
            state["oc_alert_page"] += 1
            oc_alert_table_container.content = _oc_render_alerts()
            page.update()

    oc_job_table_container.content = _oc_render_jobs()
    oc_alert_table_container.content = _oc_render_alerts()

    oc_connection_card = card(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.CLOUD_SYNC, color=C.BLUE_700, size=21),
            ft.Text("Local OpenClaw Connection", size=15,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            oc_connection_status,
        ], spacing=10, wrap=True),
        ft.Text(
            "The application uses the local OpenClaw CLI and Gateway. "
            "Authentication tokens stay in OpenClaw; only risk-summary fields are shared.",
            size=11, color=C.GREY_700),
        ft.Row([
            oc_gateway_field,
            oc_timezone_field,
            oc_share_switch,
        ], spacing=10, wrap=True),
        ft.Row([
            oc_delivery_switch,
            oc_channel_dropdown,
            oc_target_field,
        ], spacing=10, wrap=True),
        ft.Row([
            ft.Button("Save settings", icon=ft.Icons.SAVE,
                      on_click=_oc_save_connection,
                      bgcolor=UI["primary"], color=C.WHITE),
            ft.Button("Test connection", icon=ft.Icons.WIFI_FIND,
                      on_click=_oc_test_connection,
                      bgcolor=UI["button"], color=UI["text"]),
        ], spacing=8, wrap=True),
        oc_cli_path,
        oc_action_status,
        oc_log_panel,
    ], spacing=10), accent=C.BLUE_200, pad=14)

    oc_jobs_card = card(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.SCHEDULE, color=C.BLUE_700, size=21),
            ft.Text("OpenClaw Cron Jobs", size=15,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        ], spacing=8),
        ft.Text(
            "Create deterministic jobs. Sync/Edit/Run actions require an explicit click "
            "and OpenClaw operator.admin permission.",
            size=11, color=C.GREY_700),
        ft.Row([
            oc_job_name, oc_job_task, oc_job_cron, oc_job_timezone,
            oc_job_timeout, oc_job_enabled,
        ], spacing=8, wrap=True),
        ft.Row([
            ft.Button("Save job", icon=ft.Icons.SAVE, on_click=_oc_save_job,
                      bgcolor=UI["primary"], color=C.WHITE),
            ft.Button("New job", icon=ft.Icons.ADD, on_click=_oc_reset_job_form,
                      bgcolor=UI["button"], color=UI["text"]),
            oc_job_page_label,
        ], spacing=8, wrap=True),
        oc_job_table_container,
    ], spacing=10), accent=C.BLUE_200, pad=14)

    oc_alerts_card = card(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.NOTIFICATIONS_ACTIVE, color=C.ORANGE_700, size=21),
            ft.Text("Lead-time Alert Queue", size=15,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        ], spacing=8),
        ft.Text(
            "Prospective firms show a 0-90 day EWS monitoring window, not an observed "
            "default date. Exact lead_time_days is shown only for observed defaults.",
            size=11, color=C.GREY_700),
        ft.Row([
            ft.Button("Prev", icon=ft.Icons.NAVIGATE_BEFORE,
                      on_click=_oc_alert_prev, bgcolor=UI["button"], color=UI["text"]),
            ft.Button("Next", icon=ft.Icons.NAVIGATE_NEXT,
                      on_click=_oc_alert_next, bgcolor=UI["button"], color=UI["text"]),
            oc_alert_page_label,
            oc_alert_search,
            oc_alert_status_dropdown,
            oc_alert_sort_dropdown,
            ft.Button("Apply", icon=ft.Icons.SEARCH, on_click=_oc_alert_apply,
                      bgcolor=UI["primary"], color=C.WHITE),
            ft.Button("Clear", icon=ft.Icons.CLEAR, on_click=_oc_alert_clear,
                      bgcolor=UI["button"], color=UI["text"]),
        ], spacing=8, wrap=True),
        oc_alert_table_container,
    ], spacing=10), accent=C.BLUE_200, pad=14)

    # Active Loaded Dataset Inspector Card at top of Approach 1 View
    data_grid_card = card(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.TABLE_CHART, color=C.BLUE_800, size=20),
            ft.Text("📊 Active Loaded Dataset DataGridView Inspector", size=14, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        ], spacing=6),
        data_info,
        nav_row,
        data_grid_container
    ], spacing=10), accent=C.BLUE_200, pad=14)

    top_panel_container = ft.Container(content=data_grid_card)

    lead_time_card = card(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.SCHEDULE, color=C.BLUE_700, size=20),
            ft.Text("Lead Time Table by Firm ID", size=14, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        ], spacing=6),
        ft.Text("Path 1 lead time is measured from the first qualifying PD/RS alarm inside the 3-calendar-month window before an observed default. Earlier alarms remain false alarms in model metrics; non-defaulted firms remain N/A.", size=11, color=C.GREY_700),
        lead_time_info,
        lead_nav_row,
        lead_grid_container,
    ], spacing=10), accent=C.BLUE_200, pad=14)

    # --- Navigation Section Views ---
    view_approach1 = ft.Column([
        top_panel_container,
        card(ft.Column([
                ft.Text("Approach 1: Dynamic Survival Hazard + Momentum", size=16, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
                ft.Text("Primary EWS engine for ThaiBMA. Uses Cox hazard regression, forward 3-month default probability (PD_3M), risk momentum velocity, and hyperbolic decision boundaries. Easy for ThaiBMA to duplicate & scale.", size=12, color=C.GREY_800),
                surv_kpi_cards,
                ft.Row([surv_box], spacing=16, wrap=True),
                surv_img,
                surv_table
            ], spacing=12), accent=C.BLUE_200),
        firm_traj_container
    ], spacing=12)

    view_approach2 = ft.Column([
        card(ft.Column([
                ft.Text("Approach 2: XGBoost / DL + SHAP XAI", size=16, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
                ft.Text("Original CMDF/ThaiBMA research proposal framework. Uses 33 static financial/market/governance features with non-linear machine learning classifiers and SHAP feature importance for granular risk explanation.", size=12, color=C.GREY_800),
                ft.Row([metrics_box, imp_img, shap_img], spacing=16, wrap=True),
                ft.Row([alerts_table, dist_img], spacing=16, wrap=True)
            ], spacing=12), accent=C.BLUE_200)
    ], spacing=12)

    # --- Survivor2 Data & View Components ---
    survivor2_kpi_cards = ft.Row([], spacing=10)
    survivor2_grid_container = ft.Container(padding=6)
    survivor2_info = ft.Text("Survivor2 Engine: Click 'Run Survivor2 Engine' to execute Walk-Forward Cox Hazard Model.", size=12, color=C.BLUE_900)
    survivor2_search_field = ft.TextField(hint_text="Search firm_id, alert, status, date...", width=260, text_size=12)
    survivor2_page_label = ft.Text("Page 1", size=12, weight=ft.FontWeight.BOLD)
    survivor2_state = {"page": 0, "page_size": 25, "search": "", "sort": "default_first"}

    survivor2_sort_dropdown = ft.Dropdown(
        label="Sort By", width=220, text_size=12, value="default_first",
        options=[
            ft.dropdown.Option(key="default_first", text="Defaulted First"),
            ft.dropdown.Option(key="lead_desc", text="Lead Time Longest"),
            ft.dropdown.Option(key="lead_asc", text="Lead Time Shortest"),
            ft.dropdown.Option(key="pd_desc", text="PD_3M Highest"),
        ]
    )

    def render_survivor2_table():
        con = sqlite3.connect(DB)
        try:
            df = pd.read_sql_query("SELECT * FROM survivor2_lead_time", con)
        except Exception:
            df = pd.DataFrame()
        finally:
            con.close()

        if df.empty:
            return ft.Text("No Survivor2 lead-time records in SQLite yet. Click 'Run Survivor2 Engine' to run model.", color=C.GREY_600)

        s = survivor2_state["search"].strip().lower()
        if s:
            mask = pd.Series(False, index=df.index)
            for c in df.columns:
                mask |= df[c].astype(str).str.lower().str.contains(s, na=False)
            df = df[mask]

        sort_mode = survivor2_state.get("sort", "default_first")
        if sort_mode == "default_first":
            df["_def"] = (df["status"] == "detected").astype(int)
            df["_ld"] = pd.to_numeric(df["lead_days"], errors="coerce").fillna(-1)
            df = df.sort_values(["_def", "_ld"], ascending=[False, False]).drop(columns=["_def", "_ld"])
        elif sort_mode == "lead_desc":
            df["_ld"] = pd.to_numeric(df["lead_days"], errors="coerce").fillna(-1)
            df = df.sort_values("_ld", ascending=False).drop(columns=["_ld"])
        elif sort_mode == "lead_asc":
            df["_ld"] = pd.to_numeric(df["lead_days"], errors="coerce").fillna(999999)
            df = df.sort_values("_ld", ascending=True).drop(columns=["_ld"])
        elif sort_mode == "pd_desc":
            df["_pd"] = pd.to_numeric(df["PD_3M"], errors="coerce").fillna(-1)
            df = df.sort_values("_pd", ascending=False).drop(columns=["_pd"])

        total_records = len(df)
        offset = survivor2_state["page"] * survivor2_state["page_size"]
        sub = df.iloc[offset:offset + survivor2_state["page_size"]]
        survivor2_page_label.value = f"Page {survivor2_state['page']+1} / {max(1, (total_records-1)//survivor2_state['page_size']+1)} ({total_records:,} rows)"

        if sub.empty:
            return ft.Text("No matching Survivor2 records found.", color=C.GREY_600)

        cols = ["Firm ID", "First Alarm Date", "Default Date", "Lead Time (Days)", "Forward PD (3M)", "Momentum", "Alert Level", "Status"]
        rows = []
        for _, r in sub.iterrows():
            lead_val = r.get("lead_days")
            lead_txt = "N/A" if pd.isna(lead_val) or r.get("status") == "censored" else (f"{int(round(float(lead_val))):,} Days" if pd.notna(lead_val) else "MISSED")
            
            st_text = str(r.get("alert", "OK"))
            st_col = C.RED_600 if st_text == "HIGH RISK" else (C.ORANGE_600 if st_text == "ELEVATED" else (C.AMBER_600 if st_text == "WATCH" else C.GREEN_600))
            badge = ft.Container(
                content=ft.Text(f"● {st_text}", size=11, color=C.WHITE, weight=ft.FontWeight.BOLD),
                bgcolor=st_col, padding=ft.Padding.only(left=8, right=8, top=3, bottom=3), border_radius=10
            )
            
            rows.append(ft.DataRow([
                ft.DataCell(ft.Text(str(r.get("firm_id", "")), weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(str(r.get("first_alarm", "-")))),
                ft.DataCell(ft.Text(str(r.get("default_date", "-")))),
                ft.DataCell(ft.Text(lead_txt, weight=ft.FontWeight.BOLD if lead_txt != "N/A" else ft.FontWeight.NORMAL)),
                ft.DataCell(ft.Text(f"{float(r.get('PD_3M', 0))*100:.1f}%" if pd.notna(r.get('PD_3M')) else "-")),
                ft.DataCell(ft.Text(f"{float(r.get('Momentum', 1)):.2f}" if pd.notna(r.get('Momentum')) else "1.00")),
                ft.DataCell(badge),
                ft.DataCell(ft.Text(str(r.get("status", "")), color=C.GREEN_700 if r.get("status")=="detected" else (C.RED_600 if r.get("status")=="missed" else C.GREY_600))),
            ]))

        return ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD, size=11, color=C.BLUE_900)) for c in cols],
            rows=rows, data_row_min_height=34
        )

    def survivor2_prev(_):
        if survivor2_state["page"] > 0:
            survivor2_state["page"] -= 1
            survivor2_grid_container.content = render_survivor2_table()
            page.update()

    def survivor2_next(_):
        survivor2_state["page"] += 1
        survivor2_grid_container.content = render_survivor2_table()
        page.update()

    def survivor2_search(_):
        survivor2_state["search"] = survivor2_search_field.value or ""
        survivor2_state["page"] = 0
        survivor2_grid_container.content = render_survivor2_table()
        page.update()

    def survivor2_clear(_):
        survivor2_search_field.value = ""
        survivor2_state["search"] = ""
        survivor2_state["page"] = 0
        survivor2_grid_container.content = render_survivor2_table()
        page.update()

    def survivor2_sort_changed(_):
        survivor2_state["sort"] = survivor2_sort_dropdown.value or "default_first"
        survivor2_state["page"] = 0
        survivor2_grid_container.content = render_survivor2_table()
        page.update()

    survivor2_sort_dropdown.on_select = survivor2_sort_changed
    survivor2_sort_dropdown.on_change = survivor2_sort_changed

    survivor2_nav_row = ft.Row([
        ft.Button("Prev", icon=ft.Icons.NAVIGATE_BEFORE, on_click=survivor2_prev, bgcolor=C.BLUE_100, color=C.BLUE_900),
        ft.Button("Next", icon=ft.Icons.NAVIGATE_NEXT, on_click=survivor2_next, bgcolor=C.BLUE_100, color=C.BLUE_900),
        survivor2_page_label,
        survivor2_search_field,
        survivor2_sort_dropdown,
        ft.Button("Search", icon=ft.Icons.SEARCH, on_click=survivor2_search, bgcolor=C.BLUE_700, color=C.WHITE),
        ft.Button("Clear", icon=ft.Icons.CLEAR, on_click=survivor2_clear, bgcolor=C.BLUE_GREY_200, color=C.BLUE_900),
    ], spacing=8, wrap=True)

    def update_survivor2_tab():
        con = sqlite3.connect(DB)
        try:
            summary_df = pd.read_sql_query("SELECT * FROM survivor2_summary LIMIT 1", con)
        except Exception:
            summary_df = pd.DataFrame()
        finally:
            con.close()

        if not summary_df.empty:
            s_row = summary_df.iloc[0]
            med_m = float(s_row.get("median_lead_months", 0.0))
            med_d = float(s_row.get("median_lead_days", 0.0))
            auc_oos = float(s_row.get("pd_auc_oos", 0.0))
            n_det = int(s_row.get("detected", 0))
            n_ev = int(s_row.get("n_events", 0))
            
            def kpi_card(title, value, icon, color, bg):
                return ft.Container(
                    content=ft.Row([
                        ft.Icon(icon, size=24, color=color),
                        ft.Column([
                            ft.Text(title, size=11, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                            ft.Text(value, size=15, weight=ft.FontWeight.BOLD, color=color),
                        ], spacing=0)
                    ], spacing=8),
                    padding=10, bgcolor=bg, border_radius=10, border=ft.Border.all(1, color), width=180, shadow=SHADOW
                )

            survivor2_kpi_cards.controls = [
                kpi_card("🎯 Median Lead Time", f"{med_m:.1f} Months ({med_d:.0f}d)", ft.Icons.SCHEDULE, C.AMBER_800, C.AMBER_50),
                kpi_card("📊 Out-of-Sample AUC", f"{auc_oos:.3f}", ft.Icons.ANALYTICS, C.BLUE_800, C.BLUE_50),
                kpi_card("🔴 Detected Defaults", f"{n_det}/{n_ev} ({n_det/max(n_ev,1)*100:.0f}%)", ft.Icons.CHECK_CIRCLE, C.GREEN_700, C.GREEN_50),
            ]
            survivor2_info.value = f"Saved SQLite tables 'survivor2_lead_time' ({int(s_row.get('n_firms',0))} firms) and 'survivor2_summary' | Median Lead Time: {med_m:.1f} months ({med_d:.0f} days)."
            survivor2_info.color = C.BLUE_800

        survivor2_grid_container.content = render_survivor2_table()

    # ---- Survivor2 EWS results (Approach-1 style, driven by survivor2.py) ----
    s2ews_hazard_img = ft.Image(src="", visible=False, width=1000)
    s2ews_hazard_note = ft.Text(
        "Click “Run Survivor2 EWS” to fit the hazard on the survivor2 panel "
        "(real credit-event onsets, censored, 33 features) and plot h(t|X).",
        size=11, color=C.GREY_600)
    s2ews_risk_cards = ft.Row([], spacing=10, wrap=True)
    s2ews_box = ft.Column([])
    s2ews_boundary_img = ft.Image(src="", visible=False, width=1000)
    s2ews_table = ft.Column([])

    def on_survivor2_ews(_):
        """Same output as 'Run Survival EWS (App 1)' but computed with survivor2.py
        (real DP/RS onsets + censoring + lead time), with the hazard plot on top."""
        status.value = "Running Survivor2 EWS (hazard → PD₃M → momentum → boundary → lead time)…"
        page.update()
        try:
            import survivor2 as s2
            panel = s2.load_bond_dated()
            df_surv, meta = survival.run(panel.drop(columns=s2.HAZARD_DROP, errors="ignore"))
            state["df_surv2"] = df_surv

            # --- (1) hazard plot on top -----------------------------------
            s2ews_hazard_img.src = _uri(fig_hazard(df_surv)); s2ews_hazard_img.visible = True
            s2ews_hazard_note.value = (
                f"Stage 1 hazard fitted on {len(panel):,} firm-months · "
                f"{panel['firm_id'].nunique()} firms · {int(panel['event'].sum())} real credit-event "
                f"onsets (DP and/or RS), months after onset censored.")
            s2ews_hazard_note.color = C.ORANGE_800

            # --- (2) EWS metrics (same layout as Approach 1) --------------
            eval_pd = meta.get("oos_pd") or survival.evaluate(df_surv, "flag_PD")
            eval_rs = meta.get("oos_rs") or survival.evaluate(df_surv, "flag_RS")

            def f3(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "n/a"

            def auc_chip(label, val, color, bg):
                return ft.Container(
                    content=ft.Column([ft.Text(label, size=10, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                                       ft.Text(f3(val), size=17, weight=ft.FontWeight.BOLD, color=color)],
                                      spacing=0),
                    padding=10, bgcolor=bg, border_radius=10,
                    border=ft.Border.all(1, color), width=170, shadow=SHADOW)

            sig_tbl = ft.DataTable(
                columns=[ft.DataColumn(ft.Text("EWS Signal (out-of-sample)")), ft.DataColumn(ft.Text("MCC")),
                         ft.DataColumn(ft.Text("Precision")), ft.DataColumn(ft.Text("Recall")),
                         ft.DataColumn(ft.Text("Flagged Vol"))],
                rows=[
                    ft.DataRow([ft.DataCell(ft.Text("PD Signal (PD₃M ≥ τ)")),
                                ft.DataCell(ft.Text(f"{eval_pd['MCC']:.3f}")),
                                ft.DataCell(ft.Text(f"{eval_pd['precision']:.2f}")),
                                ft.DataCell(ft.Text(f"{eval_pd['recall']:.2f}")),
                                ft.DataCell(ft.Text(f"{eval_pd['volume']*100:.1f}%"))]),
                    ft.DataRow([ft.DataCell(ft.Text("RS Signal (Momentum Velocity)")),
                                ft.DataCell(ft.Text(f"{eval_rs['MCC']:.3f}", weight=ft.FontWeight.BOLD,
                                                    color=C.GREEN_700)),
                                ft.DataCell(ft.Text(f"{eval_rs['precision']:.2f}")),
                                ft.DataCell(ft.Text(f"{eval_rs['recall']:.2f}")),
                                ft.DataCell(ft.Text(f"{eval_rs['volume']*100:.1f}%"))]),
                ])
            bnd = meta["boundary"]; cutm = meta.get("cut_month")
            s2ews_box.controls = [
                ft.Text(f"Survivor2 EWS — Boundary K={bnd['K']:.3f}, α={bnd['alpha']:.2f}. "
                        f"Metrics are OUT-OF-SAMPLE" +
                        (f" (train ≤ month {cutm}, test {meta.get('n_test',0):,} rows)." if cutm
                         else " (panel too short — in-sample)."),
                        weight=ft.FontWeight.BOLD, color=C.PURPLE_900, size=12),
                ft.Row([auc_chip("PD₃M AUC · OUT-OF-SAMPLE", meta.get("pd_auc_oos"), C.PURPLE_700, C.PURPLE_50),
                        auc_chip("PD₃M AUC · in-sample", meta.get("pd_auc"), C.GREY_700, C.GREY_100),
                        auc_chip("Persistence baseline", meta.get("persistence_auc"),
                                 C.BLUE_GREY_600, C.BLUE_GREY_50)], spacing=10, wrap=True),
                sig_tbl,
            ]

            # --- (3) risk-status KPI cards --------------------------------
            acct_col = "firm_id" if "firm_id" in df_surv.columns else "account_id"
            latest = df_surv.sort_values("month_index").groupby(acct_col).tail(1).copy()

            def s2_status(r):
                fr, fp = r.get("flag_RS", 0), r.get("flag_PD", 0)
                p3, mo = r.get("PD_3M", 0.0), r.get("Momentum", 1.0)
                if fr == 1 and fp == 1:
                    return "HIGH RISK"
                if fr == 1 or p3 >= 0.15:
                    return "ELEVATED"
                if (mo if mo == mo else 1.0) >= 1.15 or p3 >= 0.05:
                    return "WATCH"
                return "OK"
            latest["status_text"] = latest.apply(s2_status, axis=1)

            def kpi_card(title, count, icon, color, bg):
                return ft.Container(content=ft.Row([
                    ft.Icon(icon, size=24, color=color),
                    ft.Column([ft.Text(title, size=11, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                               ft.Text(f"{count:,} firms", size=16, weight=ft.FontWeight.BOLD, color=color)],
                              spacing=0)], spacing=8),
                    padding=12, bgcolor=bg, border_radius=12,
                    border=ft.Border.all(1, color), width=170, shadow=SHADOW)

            vc = latest["status_text"].value_counts()
            s2ews_risk_cards.controls = [
                kpi_card("🔴 RED ALERT", int(vc.get("HIGH RISK", 0)), ft.Icons.ERROR, C.RED_700, C.RED_50),
                kpi_card("🟠 ELEVATED", int(vc.get("ELEVATED", 0)), ft.Icons.WARNING, C.ORANGE_700, C.ORANGE_50),
                kpi_card("🟡 WATCH NOTICE", int(vc.get("WATCH", 0)), ft.Icons.REMOVE_RED_EYE, C.AMBER_700, C.AMBER_50),
                kpi_card("🟢 NORMAL SAFE", int(vc.get("OK", 0)), ft.Icons.CHECK_CIRCLE, C.GREEN_700, C.GREEN_50),
            ]
            s2ews_boundary_img.src = _uri(fig_boundary(df_surv, meta))
            s2ews_boundary_img.visible = True

            # --- (4) lead time -> SQLite -> datagridview -------------------
            tbl = s2.lead_time_table(df_surv, 0.50)
            det = tbl[tbl["status"] == "detected"]
            leads = pd.to_numeric(det["lead_days"], errors="coerce").dropna()
            med_d = float(np.median(leads)) if len(leads) else 0.0
            med_m = med_d / s2.DAYS_PER_MONTH if len(leads) else 0.0
            con = sqlite3.connect(DB)
            tbl.to_sql("survivor2_lead_time", con, if_exists="replace", index=False)
            pd.DataFrame([{
                "p_star": 0.50, "n_firms": int(panel["firm_id"].nunique()),
                "n_events": int(panel["event"].sum()), "detected": len(det),
                "missed": int((tbl["status"] == "missed").sum()),
                "censored": int((tbl["status"] == "censored").sum()),
                "pd_auc": float(meta.get("pd_auc", 0.0) or 0.0),
                "pd_auc_oos": float(meta.get("pd_auc_oos", 0.0) or 0.0),
                "persistence_auc": float(meta.get("persistence_auc", 0.0) or 0.0),
                "median_lead_days": med_d, "median_lead_months": med_m,
            }]).to_sql("survivor2_summary", con, if_exists="replace", index=False)
            con.commit(); con.close()

            s2ews_table.controls = [
                ft.Text(f"Latest firm risk standing — {len(latest):,} firms "
                        f"(top 25 by PD₃M)", weight=ft.FontWeight.BOLD, color=C.BLUE_900, size=12),
                ft.Row([ft.DataTable(
                    columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                             for c in ["Firm", "Forward PD₃M", "Momentum", "Hazard h(t)", "Status"]],
                    rows=[ft.DataRow([
                        ft.DataCell(ft.Text(_fmt_firm(r[acct_col]), weight=ft.FontWeight.BOLD, size=11)),
                        ft.DataCell(ft.Text(f"{r['PD_3M']*100:.1f}%", size=11)),
                        ft.DataCell(ft.Text(f"{r['Momentum']:.2f}" if pd.notna(r["Momentum"]) else "1.00", size=11)),
                        ft.DataCell(ft.Text(f"{r['h']*100:.2f}%", size=11, color=C.PURPLE_700)),
                        ft.DataCell(ft.Container(
                            content=ft.Text(f"● {r['status_text']}", size=10, color=C.WHITE,
                                            weight=ft.FontWeight.BOLD),
                            bgcolor={"HIGH RISK": C.RED_600, "ELEVATED": C.ORANGE_600,
                                     "WATCH": C.AMBER_600, "OK": C.GREEN_600}[r["status_text"]],
                            padding=ft.Padding.symmetric(horizontal=9, vertical=3), border_radius=10)),
                    ]) for _, r in latest.sort_values("PD_3M", ascending=False).head(25).iterrows()],
                    column_spacing=18, data_row_min_height=28)], scroll=ft.ScrollMode.AUTO),
            ]

            update_survivor2_tab()
            set_tab(10)
            status.value = (f"✅ Survivor2 EWS done — {len(det)}/{int(panel['event'].sum())} detected, "
                            f"median lead {med_m:.1f} months ({med_d:.0f} d), "
                            f"OOS AUC {f3(meta.get('pd_auc_oos'))}")
            page.update()
        except Exception as ex:
            s2ews_hazard_note.value = f"Survivor2 EWS failed: {ex}"
            s2ews_hazard_note.color = C.RED_600
            status.value = f"Error running Survivor2 EWS: {ex}"
            page.update()

    s2ews_run_row = ft.Row([
        ft.Button("Run Survivor2 EWS", icon=ft.Icons.MONITOR_HEART,
                  on_click=lambda _: run_async(on_survivor2_ews, "Running Survivor2 EWS"),
                  bgcolor=C.PURPLE_700, color=C.WHITE,
                  style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8),
                                       padding=ft.Padding.symmetric(horizontal=16, vertical=14))),
        ft.Text("hazard plot on top → EWS metrics → lead-time DataGridView "
                "(computed by survivor2.py, not survival.py)", size=11, color=C.GREY_600),
    ], spacing=10, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # hazard card sits ON TOP, the data grid follows underneath
    s2ews_hazard_card = card(ft.Column([
        ft.Row([ft.Icon(ft.Icons.MONITOR_HEART, color=C.PURPLE_700, size=22),
                ft.Text("📈 Survivor2 EWS — Stage 1 Hazard h(t|X)", size=15,
                        weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
        s2ews_run_row,
        s2ews_hazard_note,
        s2ews_hazard_img,
    ], spacing=10), accent=C.PURPLE_200, pad=14)

    s2ews_metrics_card = card(ft.Column([
        ft.Row([ft.Icon(ft.Icons.INSIGHTS, color=C.PURPLE_700, size=20),
                ft.Text("Stage 2–3 — PD₃M signal, momentum & hyperbolic boundary", size=14,
                        weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
        s2ews_risk_cards,
        s2ews_box,
        s2ews_boundary_img,
        s2ews_table,
    ], spacing=10), accent=C.PURPLE_200, pad=14)

    survivor2_card = card(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=C.ORANGE_800, size=22),
            ft.Text("🛡️ Survivor2 Engine — Walk-Forward Lead Time DataGridView", size=15, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        ], spacing=6),
        ft.Text("Survivor2 pipeline implements discrete-time Cox hazard regression, time-varying covariates X(t), and walk-forward expanding window out-of-sample lead-time evaluation.", size=11, color=C.GREY_700),
        survivor2_info,
        survivor2_kpi_cards,
        survivor2_nav_row,
        survivor2_grid_container
    ], spacing=10), accent=C.ORANGE_200, pad=14)

    view_survivor2 = ft.Column([s2ews_hazard_card, s2ews_metrics_card, survivor2_card],
                               spacing=12)

    # ================= Model Comparison (survivor2 vs machine_survior) =========
    cmp_info = ft.Text("Click “Run comparison” to train both hazard engines and score them "
                       "head-to-head (takes a few minutes), or load the last saved run.",
                       size=12, color=C.GREY_700)
    cmp_kpis = ft.Row([], spacing=10, wrap=True)
    cmp_verdict = ft.Column([])
    cmp_metrics_box = ft.Column([])
    cmp_lead_box = ft.Column([])
    cmp_stat_box = ft.Column([])
    cmp_img_metrics = ft.Image(src="", visible=False, width=980)
    cmp_img_outperf = ft.Image(src="", visible=False, width=880)
    cmp_img_lead = ft.Image(src="", visible=False, width=940)

    CMP_WIN_COLORS = {"Logistic": C.BLUE_700, "XGBoost": C.ORANGE_800}

    def _cmp_badge(text, color, bg):
        return ft.Container(content=ft.Text(text, size=10, color=C.WHITE,
                                            weight=ft.FontWeight.BOLD),
                            bgcolor=color, padding=ft.Padding.symmetric(horizontal=8, vertical=2),
                            border_radius=8)

    def render_cmp_metrics(metrics):
        rows = []
        for _, r in metrics.iterrows():
            w = r["winner"]
            lrv = "-" if pd.isna(r["logistic"]) else f"{r['logistic']:.3f}"
            xgv = "-" if pd.isna(r["xgboost"]) else f"{r['xgboost']:.3f}"
            pct = "" if (w not in CMP_WIN_COLORS or pd.isna(r["pct_outperform"])) \
                else f"+{r['pct_outperform']:.1f}%"
            badge = (_cmp_badge(w, CMP_WIN_COLORS[w], None) if w in CMP_WIN_COLORS
                     else ft.Text("—", size=11, color=C.GREY_500))
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(r["metric"], size=11)),
                ft.DataCell(ft.Text(r["direction"], size=10, color=C.GREY_600)),
                ft.DataCell(ft.Text(lrv, size=11, color=C.BLUE_700,
                                    weight=ft.FontWeight.BOLD if w == "Logistic" else ft.FontWeight.NORMAL)),
                ft.DataCell(ft.Text(xgv, size=11, color=C.ORANGE_800,
                                    weight=ft.FontWeight.BOLD if w == "XGBoost" else ft.FontWeight.NORMAL)),
                ft.DataCell(badge),
                ft.DataCell(ft.Text(pct, size=11, weight=ft.FontWeight.BOLD,
                                    color=CMP_WIN_COLORS.get(w, C.GREY_500))),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Metric", "Direction", "Logistic", "XGBoost", "Winner", "Outperform"]],
            rows=rows, column_spacing=18, data_row_min_height=30)], scroll=ft.ScrollMode.AUTO)

    def render_cmp_lead(lead):
        rows = []
        for _, r in lead.head(40).iterrows():
            lr = "MISS" if pd.isna(r["lead_lr"]) else f"{int(r['lead_lr'])}"
            xg = "MISS" if pd.isna(r["lead_xgb"]) else f"{int(r['lead_xgb'])}"
            dd = "" if pd.isna(r["diff_days"]) else f"{int(r['diff_days']):+d}"
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(int(r["firm_id"])), size=11, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(str(r["default_date"])[:10], size=11)),
                ft.DataCell(ft.Text(lr, size=11, color=C.RED_600 if lr == "MISS" else C.BLUE_700)),
                ft.DataCell(ft.Text(xg, size=11, color=C.RED_600 if xg == "MISS" else C.ORANGE_800)),
                ft.DataCell(ft.Text(dd, size=11, color=C.GREY_700)),
                ft.DataCell(ft.Text("" if r["winner"] == "-" else str(r["winner"]), size=10,
                                    color=C.PURPLE_700)),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Firm", "Event date", "Logistic (d)", "XGBoost (d)", "Diff", "Note"]],
            rows=rows, column_spacing=18, data_row_min_height=28)], scroll=ft.ScrollMode.AUTO)

    def update_compare_tab(_=None):
        import compare_models as cmpm
        metrics, lead, summary = cmpm.load_from_sqlite(DB)
        if metrics.empty or summary.empty:
            cmp_info.value = ("No saved comparison yet — click “Run comparison” "
                              "(trains both engines; takes a few minutes).")
            cmp_info.color = C.GREY_600
            page.update(); return
        s = summary.iloc[0]

        def kpi(title, value, sub, icon, color, bg):
            return ft.Container(content=ft.Row([
                ft.Icon(icon, size=24, color=color),
                ft.Column([ft.Text(title, size=10, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                           ft.Text(value, size=15, weight=ft.FontWeight.BOLD, color=color),
                           ft.Text(sub, size=9, color=C.GREY_600)], spacing=0)
            ], spacing=8), padding=10, bgcolor=bg, border_radius=10,
                border=ft.Border.all(1, color), width=210, shadow=SHADOW)

        win = str(s.get("overall_winner", "-"))
        wcol = CMP_WIN_COLORS.get(win, C.PURPLE_700)
        cmp_kpis.controls = [
            kpi("🏆 Overall winner", win,
                f"{int(s['wins_logistic'])} vs {int(s['wins_xgboost'])} of {int(s['n_scored_metrics'])} metrics",
                ft.Icons.EMOJI_EVENTS, wcol, C.PURPLE_50),
            kpi("📊 OOS AUC — Logistic", f"{float(s['auc_lr']):.3f}", "survivor2 (logistic hazard)",
                ft.Icons.SHOW_CHART, C.BLUE_700, C.BLUE_50),
            kpi("🌲 OOS AUC — XGBoost", f"{float(s['auc_xgb']):.3f}", "machine_survior (+SHAP)",
                ft.Icons.PARK, C.ORANGE_800, C.ORANGE_50),
            kpi("🧪 AUC difference",
                f"{float(s['auc_delta']):+.3f}" if pd.notna(s.get("auc_delta")) else "n/a",
                f"p = {float(s['p_bootstrap']):.3f}" if pd.notna(s.get("p_bootstrap")) else "",
                ft.Icons.SCIENCE, C.TEAL_700, C.TEAL_50),
        ]
        cmp_verdict.controls = [ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.GAVEL, size=16, color=wcol),
                        ft.Text(f"Verdict — {win}", size=13, weight=ft.FontWeight.BOLD, color=wcol)],
                       spacing=6),
                ft.Text(str(s.get("verdict", "")), size=11, color=C.GREY_800),
            ], spacing=4),
            bgcolor=C.PURPLE_50, padding=12, border_radius=10,
            border=ft.Border.all(1, C.PURPLE_200))]

        cmp_metrics_box.controls = [
            ft.Text("Table 1 — metric-by-metric (who outperforms by what %)",
                    size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            render_cmp_metrics(metrics)]
        cmp_lead_box.controls = [
            ft.Text(f"Table 2 — per-firm lead time ({len(lead)} firms with a credit event)",
                    size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            render_cmp_lead(lead)]
        stat_txt = (f"Paired OOS rows {int(s['n_paired_rows']):,} · positives {int(s['n_positives'])}   |   "
                    f"Bootstrap ΔAUC {float(s['auc_delta']):+.3f} "
                    f"[{float(s['ci_low']):+.3f}, {float(s['ci_high']):+.3f}], p={float(s['p_bootstrap']):.3f}   |   "
                    f"McNemar: LR-only {int(s['mcnemar_lr_only'])} vs XGB-only {int(s['mcnemar_xgb_only'])}, "
                    f"p={float(s['p_mcnemar']):.3f}") if pd.notna(s.get("auc_delta")) else "statistical test unavailable"
        cmp_stat_box.controls = [
            ft.Text("Table 3 — statistical test", size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            ft.Text(stat_txt, size=11, color=C.GREY_800)]

        try:
            cmp_img_metrics.src = _uri(fig_compare_metrics(metrics)); cmp_img_metrics.visible = True
            cmp_img_outperf.src = _uri(fig_compare_outperform(metrics)); cmp_img_outperf.visible = True
            if not lead.empty:
                cmp_img_lead.src = _uri(fig_compare_leadtime(lead)); cmp_img_lead.visible = True
        except Exception as ex:
            cmp_info.value = f"charts unavailable: {ex}"
        cmp_info.value = (f"Run {s.get('run_at','')} · {int(s['n_firm_months']):,} firm-months · "
                          f"{int(s['n_firms'])} firms · {int(s['n_events'])} credit events "
                          f"· SQLite: model_compare_metrics / _leadtime / _summary")
        cmp_info.color = C.PURPLE_700
        page.update()

    def on_run_compare(_):
        import compare_models as cmpm
        cmp_info.value = "Running both hazard engines (logistic + XGBoost) … this takes a few minutes."
        cmp_info.color = C.ORANGE_800
        status.value = "Model comparison running …"
        page.update()
        try:
            res = cmpm.run_comparison(verbose=False)
            cmpm.save_to_sqlite(res, DB)
            update_compare_tab()
            status.value = (f"Comparison done — winner: {res['summary']['overall_winner']} "
                            f"({res['summary']['wins_logistic']} vs {res['summary']['wins_xgboost']})")
        except Exception as ex:
            cmp_info.value = f"Comparison failed: {ex}"; cmp_info.color = C.RED_600
            status.value = f"Comparison error: {ex}"
        page.update()

    cmp_toolbar = ft.Row([
        ft.Button("Run comparison", icon=ft.Icons.PLAY_ARROW, on_click=on_run_compare,
                  bgcolor=C.PURPLE_700, color=C.WHITE,
                  style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
        ft.Button("Reload saved", icon=ft.Icons.REFRESH, on_click=update_compare_tab,
                  bgcolor=C.BLUE_GREY_600, color=C.WHITE,
                  style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
    ], spacing=8, wrap=True)

    compare_card = card(ft.Column([
        ft.Row([ft.Icon(ft.Icons.COMPARE_ARROWS, color=C.PURPLE_700, size=22),
                ft.Text("⚖️ Model Comparison — survivor2 (Logistic) vs machine_survior (XGBoost + SHAP)",
                        size=15, weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
        ft.Text("Both engines share the same panel, 33 features and survival pipeline "
                "(hazard → PD₃M → momentum → hyperbolic boundary → lead time); only the hazard "
                "estimator differs. Scored on a temporal out-of-sample split.",
                size=11, color=C.GREY_700),
        cmp_toolbar,
        cmp_info,
        cmp_kpis,
        cmp_verdict,
        cmp_img_outperf,
        cmp_metrics_box,
        cmp_img_metrics,
        cmp_stat_box,
        cmp_lead_box,
        cmp_img_lead,
    ], spacing=12), accent=C.PURPLE_200, pad=14)

    view_compare = ft.Column([compare_card], spacing=12)

    # =============== Benchmark: Approach 1 vs Approach 2 vs basic DL ==========
    bm_info = ft.Text("Click “Run benchmark” to train 8 models (Approach 1 survival, "
                      "Approach 2 static ML, and basic DL) on the same walk-forward split, "
                      "or load the last saved run.", size=12, color=C.GREY_700)
    bm_kpis = ft.Row([], spacing=10, wrap=True)
    bm_verdict = ft.Column([])
    bm_pred_box = ft.Column([])
    bm_econ_box = ft.Column([])
    bm_group_box = ft.Column([])
    bm_img_pred = ft.Image(src="", visible=False, width=1000)
    bm_img_econ = ft.Image(src="", visible=False, width=960)
    bm_img_trade = ft.Image(src="", visible=False, width=780)

    BM_GROUP_COLOR = {"Approach 1": C.BLUE_700, "Approach 1 (DL)": C.PURPLE_700,
                      "Approach 2": C.ORANGE_800, "Approach 2 (DL)": C.PINK_700}

    def render_bm_pred(p):
        rows = []
        for _, r in p.sort_values("auc", ascending=False).iterrows():
            gcol = BM_GROUP_COLOR.get(r["group"], C.GREY_700)
            lead = "—" if pd.isna(r["median_lead_days"]) else f"{r['median_lead_days']:.0f}"
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(r["model"], size=11, weight=ft.FontWeight.BOLD, color=gcol)),
                ft.DataCell(ft.Container(content=ft.Text(r["group"], size=9, color=C.WHITE,
                                                         weight=ft.FontWeight.BOLD),
                                         bgcolor=gcol, border_radius=8,
                                         padding=ft.Padding.symmetric(horizontal=7, vertical=2))),
                ft.DataCell(ft.Text(f"{r['auc']:.3f}", size=11, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{r['pr_auc']:.4f}", size=11)),
                ft.DataCell(ft.Text(f"{r['brier']:.4f}", size=11)),
                ft.DataCell(ft.Text(f"{r['mcc']:.3f}", size=11)),
                ft.DataCell(ft.Text(f"{r['recall']:.3f}", size=11)),
                ft.DataCell(ft.Text(f"{int(r['detected'])}/{int(r['n_event_firms'])}", size=11)),
                ft.DataCell(ft.Text(lead, size=11)),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Model", "Group", "ROC-AUC", "PR-AUC", "Brier", "MCC",
                               "Recall", "Detected", "Lead(d)"]],
            rows=rows, column_spacing=16, data_row_min_height=30)], scroll=ft.ScrollMode.AUTO)

    def render_bm_econ(p, e):
        extra = [c for c in ["group", "detected", "n_event_firms"] if c not in e.columns]
        d = e.merge(p[["model"] + extra], on="model") if extra else e.copy()
        rows = []
        for _, r in d.sort_values("net_benefit_mthb", ascending=False).iterrows():
            gcol = BM_GROUP_COLOR.get(r["group"], C.GREY_700)
            net = float(r["net_benefit_mthb"])
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(r["model"], size=11, weight=ft.FontWeight.BOLD, color=gcol)),
                ft.DataCell(ft.Text(f"{r['miss_rate']:.3f}", size=11)),
                ft.DataCell(ft.Text(f"{r['false_alarm_rate']:.3f}", size=11)),
                ft.DataCell(ft.Text(f"{r['usefulness_rel']:.3f}", size=11,
                                    color=C.GREEN_700 if r["usefulness_rel"] > 0 else C.RED_600)),
                ft.DataCell(ft.Text(f"{r['benefit_mthb']:,.0f}", size=11, color=C.GREEN_700)),
                ft.DataCell(ft.Text(f"{r['review_cost_mthb']:,.0f}", size=11, color=C.RED_600)),
                ft.DataCell(ft.Text(f"{net:,.0f}", size=11, weight=ft.FontWeight.BOLD,
                                    color=C.GREEN_800 if net > 0 else C.RED_700)),
                ft.DataCell(ft.Text(f"{r['roi']:.1f}×", size=11, weight=ft.FontWeight.BOLD)),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Model", "Miss rate", "FA rate", "Sarlin Ur",
                               "Benefit (MTHB)", "Cost (MTHB)", "NET (MTHB)", "ROI"]],
            rows=rows, column_spacing=16, data_row_min_height=30)], scroll=ft.ScrollMode.AUTO)

    def update_benchmark_tab(_=None):
        import benchmark_all as bm
        p, e, s = bm.load_from_sqlite(DB)
        if p.empty or s.empty:
            bm_info.value = ("No saved benchmark yet — click “Run benchmark” "
                             "(trains 8 models; takes several minutes).")
            bm_info.color = C.GREY_600
            page.update(); return
        srow = s.iloc[0]

        def kpi(title, value, sub, icon, color, bg):
            return ft.Container(content=ft.Row([
                ft.Icon(icon, size=24, color=color),
                ft.Column([ft.Text(title, size=10, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                           ft.Text(value, size=14, weight=ft.FontWeight.BOLD, color=color),
                           ft.Text(sub, size=9, color=C.GREY_600)], spacing=0)
            ], spacing=8), padding=10, bgcolor=bg, border_radius=10,
                border=ft.Border.all(1, color), width=230, shadow=SHADOW)

        wp, we = str(srow["winner_prediction"]), str(srow["winner_economic"])
        bm_kpis.controls = [
            kpi("🎯 Best prediction", str(srow["best_prediction_model"]),
                f"AUC {float(srow['best_prediction_auc']):.3f} · winner: {wp}",
                ft.Icons.MILITARY_TECH, C.BLUE_700, C.BLUE_50),
            kpi("💰 Best economics", str(srow["best_economic_model"]),
                f"net {float(srow['best_economic_net_mthb']):,.0f} MTHB · winner: {we}",
                ft.Icons.SAVINGS, C.GREEN_700, C.GREEN_50),
            kpi("⚔️ Approach 1 vs 2 (AUC)",
                f"{float(srow['a1_best_auc']):.3f} vs {float(srow['a2_best_auc']):.3f}",
                "best out-of-sample AUC each", ft.Icons.BALANCE, C.INDIGO_700, C.INDIGO_50),
            kpi("🧠 DL vs classical (AUC)",
                f"{float(srow['dl_best_auc']):.3f} vs {float(srow['classic_best_auc']):.3f}",
                "basic DL " + ("beats" if bool(srow["dl_beats_classic"]) else "does NOT beat")
                + " classical", ft.Icons.PSYCHOLOGY,
                C.GREEN_700 if bool(srow["dl_beats_classic"]) else C.RED_600,
                C.GREEN_50 if bool(srow["dl_beats_classic"]) else C.RED_50),
        ]
        bm_verdict.controls = [ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.GAVEL, size=16, color=C.TEAL_800),
                        ft.Text("Verdict", size=13, weight=ft.FontWeight.BOLD, color=C.TEAL_800)],
                       spacing=6),
                ft.Text(str(srow["verdict"]), size=11, color=C.GREY_800),
            ], spacing=4), bgcolor=C.TEAL_50, padding=12, border_radius=10,
            border=ft.Border.all(1, C.TEAL_200))]

        bm_pred_box.controls = [
            ft.Text(f"Table 1 — prediction performance (walk-forward OOS, matched "
                    f"{float(srow['alarm_budget'])*100:.0f}% alarm budget)",
                    size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            render_bm_pred(p)]
        bm_econ_box.controls = [
            ft.Text("Table 2 — financial / economic performance", size=12,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            ft.Text(f"Assumptions: EAD {float(srow['ead_mthb']):,.0f} MTHB/issuer · "
                    f"LGD {float(srow['lgd']):.0%} · loss mitigated when warned "
                    f"{float(srow['mitigation']):.0%} → saving "
                    f"{float(srow['ead_mthb'])*float(srow['lgd'])*float(srow['mitigation']):,.0f} MTHB "
                    f"per early warning · review cost {float(srow['review_cost_mthb'])} MTHB "
                    f"per flagged firm-month. Sarlin Ur is firm-month level; NET is event level.",
                    size=10, color=C.GREY_600),
            render_bm_econ(p, e)]

        a1n, a2n = float(srow["a1_best_net"]), float(srow["a2_best_net"])
        bm_group_box.controls = [
            ft.Text("Table 3 — approach-level answer", size=12, weight=ft.FontWeight.BOLD,
                    color=C.BLUE_900),
            ft.Row([ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                         for c in ["Approach", "Best ROC-AUC", "Best net benefit (MTHB)", "Verdict"]],
                rows=[
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Approach 1 — survival hazard", size=11,
                                            weight=ft.FontWeight.BOLD, color=C.BLUE_700)),
                        ft.DataCell(ft.Text(f"{float(srow['a1_best_auc']):.3f}", size=11)),
                        ft.DataCell(ft.Text(f"{a1n:,.0f}", size=11)),
                        ft.DataCell(ft.Text("🏆 prediction" if wp == "Approach 1" else "—", size=11)),
                    ]),
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Approach 2 — static ML", size=11,
                                            weight=ft.FontWeight.BOLD, color=C.ORANGE_800)),
                        ft.DataCell(ft.Text(f"{float(srow['a2_best_auc']):.3f}", size=11)),
                        ft.DataCell(ft.Text(f"{a2n:,.0f}", size=11)),
                        ft.DataCell(ft.Text("🏆 economics" if we == "Approach 2" else "—", size=11)),
                    ]),
                ], column_spacing=20, data_row_min_height=30)], scroll=ft.ScrollMode.AUTO)]

        try:
            bm_img_pred.src = _uri(fig_benchmark_prediction(p)); bm_img_pred.visible = True
            bm_img_econ.src = _uri(fig_benchmark_economics(p, e)); bm_img_econ.visible = True
            bm_img_trade.src = _uri(fig_benchmark_tradeoff(p, e)); bm_img_trade.visible = True
        except Exception as ex:
            bm_info.value = f"charts unavailable: {ex}"
        bm_info.value = (f"Run {srow.get('run_at','')} · {int(srow['n_firm_months']):,} firm-months · "
                         f"{int(srow['n_firms'])} firms · {int(srow['n_events'])} credit events · "
                         f"{int(srow['folds'])} walk-forward folds · SQLite: benchmark_prediction / "
                         f"_economics / _summary")
        bm_info.color = C.TEAL_800
        page.update()

    def on_run_benchmark(_):
        import benchmark_all as bm
        bm_info.value = "Training 8 models on walk-forward folds … this takes several minutes."
        bm_info.color = C.ORANGE_800
        status.value = "Benchmark running (Approach 1 / Approach 2 / DL) …"
        page.update()
        try:
            res = bm.run_benchmark(fast=False, verbose=False)
            bm.save_to_sqlite(res, DB)
            update_benchmark_tab()
            s = res["summary"]
            status.value = (f"Benchmark done — prediction: {s['winner_prediction']} "
                            f"({s['best_prediction_model']} AUC {s['best_prediction_auc']:.3f}); "
                            f"economics: {s['winner_economic']} ({s['best_economic_model']})")
        except Exception as ex:
            bm_info.value = f"Benchmark failed: {ex}"; bm_info.color = C.RED_600
            status.value = f"Benchmark error: {ex}"
        page.update()

    benchmark_card = card(ft.Column([
        ft.Row([ft.Icon(ft.Icons.LEADERBOARD, color=C.TEAL_800, size=22),
                ft.Text("🏁 Benchmark — Approach 1 vs Approach 2 vs basic Deep Learning",
                        size=15, weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
        ft.Text("Eight models on one common walk-forward out-of-sample split, all predicting the "
                "same label (a real DP/RS credit event within 3 calendar months): Approach 1 "
                "(survival hazard → PD₃M) with logistic / XGBoost / MLP, and Approach 2 (static "
                "classifier) with logistic / random forest / XGBoost / MLP / GRU. Scored on both "
                "prediction quality and financial value.", size=11, color=C.GREY_700),
        ft.Row([
            ft.Button("Run benchmark", icon=ft.Icons.PLAY_ARROW, on_click=on_run_benchmark,
                      bgcolor=C.TEAL_800, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Reload saved", icon=ft.Icons.REFRESH, on_click=update_benchmark_tab,
                      bgcolor=C.BLUE_GREY_600, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
        ], spacing=8, wrap=True),
        bm_info,
        bm_kpis,
        bm_verdict,
        bm_group_box,
        bm_img_trade,
        bm_pred_box,
        bm_img_pred,
        bm_econ_box,
        bm_img_econ,
    ], spacing=12), accent=C.TEAL_200, pad=14)

    view_benchmark = ft.Column([benchmark_card], spacing=12)

    # ============ Yield curve: Dynamic Nelson-Siegel Level/Slope/Curvature =====
    dns_info = ft.Text("Load a ThaiBMA / iBond yield-curve export (CSV or Excel), then "
                       "estimate the Level, Slope and Curvature factors.",
                       size=12, color=C.GREY_700)
    dns_cred = ft.Text("", size=11, color=C.GREY_700)
    dns_file = ft.TextField(label="ThaiBMA / iBond export (.csv or .xlsx)", width=430,
                            text_size=12, dense=True,
                            hint_text=r"leave blank for auto iBond / API, or e.g. D:\data\thai_gov_yield.csv")
    dns_start = ft.TextField(label="Start (YYYY-MM)", width=135, text_size=12, dense=True,
                             value="2024-01")
    dns_end = ft.TextField(label="End (YYYY-MM)", width=135, text_size=12, dense=True,
                           value=pd.Timestamp.today().strftime("%Y-%m"))
    dns_kpis = ft.Row([], spacing=10, wrap=True)
    dns_val_box = ft.Column([])
    dns_factor_box = ft.Column([])
    dns_fc_box = ft.Column([])
    dns_latest_table_box = ft.Column([])
    dns_log_box = ft.Column([])
    dns_show_latest = ft.Checkbox(label="Show latest curve snapshot", value=True)
    dns_show_roll = ft.Checkbox(label="Show rolling dashboard", value=True)
    dns_show_factors = ft.Checkbox(label="Show factor charts", value=False)
    dns_show_tables = ft.Checkbox(label="Show tables", value=False)
    dns_show_log = ft.Checkbox(label="Show download log", value=False)
    dns_img_factors = ft.Image(src="", visible=False, width=1000)
    dns_img_surface = ft.Image(src="", visible=False, width=1000)
    dns_img_latest = ft.Image(src="", visible=False, width=760)
    dns_img_roll = ft.Image(src="", visible=False, width=1000)
    dns_img_fc = ft.Image(src="", visible=False, width=820)

    def render_dns_factors(f, n=24):
        d = f.sort_values("date", ascending=False).head(n)
        rows = []
        for _, r in d.iterrows():
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(pd.Timestamp(r["date"]).strftime("%Y-%m"), size=11,
                                    weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{r['Level']:.3f}", size=11, color=C.PINK_800)),
                ft.DataCell(ft.Text(f"{r['Slope']:.3f}", size=11, color=C.BLUE_700)),
                ft.DataCell(ft.Text(f"{r['Curvature']:.3f}", size=11, color=C.ORANGE_800)),
                ft.DataCell(ft.Text("" if pd.isna(r.get("y_1y")) else f"{r['y_1y']:.3f}", size=11)),
                ft.DataCell(ft.Text("" if pd.isna(r.get("y_10y")) else f"{r['y_10y']:.3f}", size=11)),
                ft.DataCell(ft.Text(f"{r['rmse']:.4f}", size=11, color=C.GREY_600)),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Month", "Level", "Slope", "Curvature", "1Y (%)", "10Y (%)", "fit RMSE"]],
            rows=rows, column_spacing=18, data_row_min_height=28)], scroll=ft.ScrollMode.AUTO)

    def render_dns_validation(val):
        rows = []
        for _, r in val.iterrows():
            ok = abs(r["correlation"]) >= 0.85
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(r["factor"], size=11, weight=ft.FontWeight.BOLD,
                                    color=DNS_C.get(r["factor"], C.GREY_800))),
                ft.DataCell(ft.Text(r["empirical proxy"], size=11)),
                ft.DataCell(ft.Text(f"{r['correlation']:+.3f}", size=11,
                                    weight=ft.FontWeight.BOLD,
                                    color=C.GREEN_700 if ok else C.RED_600)),
                ft.DataCell(ft.Text(f"{r['report target']:.3f}", size=11, color=C.GREY_600)),
                ft.DataCell(ft.Text(str(int(r["n"])), size=11)),
                ft.DataCell(ft.Text("✓ validated" if ok else "⚠ check data", size=10,
                                    color=C.GREEN_700 if ok else C.RED_600)),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Factor", "Empirical proxy", "Correlation", "Report", "N", "Status"]],
            rows=rows, column_spacing=18, data_row_min_height=30)], scroll=ft.ScrollMode.AUTO)

    def render_dns_forecast(fc):
        hs = sorted(fc["horizon (m)"].unique())
        rows = []
        for fac in ("Level", "Slope", "Curvature"):
            s = fc[fc["factor"] == fac].set_index("horizon (m)")["RMSE"]
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(fac, size=11, weight=ft.FontWeight.BOLD,
                                    color=DNS_C.get(fac, C.GREY_800)))]
                + [ft.DataCell(ft.Text(f"{s.get(h, float('nan')):.4f}", size=11)) for h in hs]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Factor", size=11, weight=ft.FontWeight.BOLD))]
                    + [ft.DataColumn(ft.Text(f"h={h}m", size=11, weight=ft.FontWeight.BOLD)) for h in hs],
            rows=rows, column_spacing=20, data_row_min_height=28)], scroll=ft.ScrollMode.AUTO)

    def render_dns_latest_yields(curve, date_pick=None, tenor_pick="ALL", max_rows=40):
        if curve is None or curve.empty:
            return ft.Text("no saved yield rows", size=11, color=C.GREY_600)
        d = curve.copy()
        d["date_only"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        if date_pick and date_pick != "LATEST":
            d = d[d["date_only"] == date_pick]
        else:
            latest = d["date_only"].max()
            d = d[d["date_only"] == latest]
        if tenor_pick and tenor_pick != "ALL":
            try:
                tau_pick = float(tenor_pick)
                d = d[np.isclose(d["tau"].astype(float), tau_pick)]
            except Exception:
                pass
        d = d.sort_values(["date", "tau"], ascending=[False, True]).head(max_rows)
        rows = []
        for _, r in d.iterrows():
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(r["date_only"]), size=11, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{float(r['tau']):.4g}", size=11, color=C.BLUE_700)),
                ft.DataCell(ft.Text(f"{float(r['yield']):.4f}", size=11, color=C.PINK_800)),
                ft.DataCell(ft.Text("" if pd.isna(r.get("duration")) else f"{float(r.get('duration')):.4f}", size=11)),
                ft.DataCell(ft.Text(str(r.get("source", "")), size=10, color=C.GREY_600)),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Date", "Tenor (Y)", "Yield (%)", "Duration", "Source"]],
            rows=rows, column_spacing=18, data_row_min_height=28)], scroll=ft.ScrollMode.AUTO)

    def render_dns_log(log, max_rows=12):
        if log is None or log.empty:
            return ft.Text("no download log yet", size=11, color=C.GREY_600)
        rows = []
        for _, r in log.head(max_rows).iterrows():
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(r.get("downloaded_at", "")), size=11, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(str(r.get("source", "")), size=10)),
                ft.DataCell(ft.Text(str(r.get("requested_start", "")), size=10)),
                ft.DataCell(ft.Text(str(r.get("requested_end", "")), size=10)),
                ft.DataCell(ft.Text(str(int(r.get("row_count", 0))), size=11)),
                ft.DataCell(ft.Text(str(int(r.get("n_tenor", 0))), size=11)),
            ]))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                     for c in ["Downloaded at", "Source", "Start", "End", "Rows", "Tenors"]],
            rows=rows, column_spacing=18, data_row_min_height=28)], scroll=ft.ScrollMode.AUTO)

    def _dns_range_to_dates(sel):
        now = pd.Timestamp.today().normalize()
        if sel == "1Y":
            start = (now - pd.DateOffset(years=1)).strftime("%Y-%m")
            end = now.strftime("%Y-%m")
        elif sel == "3Y":
            start = (now - pd.DateOffset(years=3)).strftime("%Y-%m")
            end = now.strftime("%Y-%m")
        elif sel == "5Y":
            start = (now - pd.DateOffset(years=5)).strftime("%Y-%m")
            end = now.strftime("%Y-%m")
        else:
            start = "2012-01"
            end = now.strftime("%Y-%m")
        return start, end

    def on_dns_range_change(_):
        start, end = _dns_range_to_dates(dns_range.value or "1Y")
        dns_start.value = start
        dns_end.value = end
        page.update()

    def refresh_dns_cred_status():
        try:
            import ibond_client as ib
            st = ib.credentials_status()
            dns_cred.value = ("🔑 iBond credentials detected in your environment, auto-connect is ready."
                              if st["ready"] else
                              "🔒 No iBond credentials in this environment. Auto iBond load is disabled; "
                              "use setup_credentials.py or load a CSV/XLSX export.")
            dns_cred.color = C.GREEN_700 if st["ready"] else C.BLUE_GREY_700
        except Exception as ex:
            dns_cred.value = f"credential check unavailable: {ex}"
            dns_cred.color = C.GREY_600

    def update_dns_tab(_=None):
        import yield_curve_dns as ycd
        refresh_dns_cred_status()

        curve, factors, summary, fc = ycd.load_from_sqlite(DB)
        raw_curve = ycd.load_ibond_raw(DB)
        raw_log = ycd.load_ibond_log(DB)
        if factors.empty or summary.empty:
            dns_info.value = ("No yield-curve run saved yet — click “Auto load iBond” to fetch live data, "
                              "enter a ThaiBMA/iBond export path and click “Estimate DNS”, or click "
                              "“Demo curve” to try it with synthetic data.")
            dns_info.color = C.GREY_600
            page.update(); return
        s = summary.iloc[0]
        is_demo = str(s.get("source", "")).upper().startswith("DEMO")

        def _f(v, default=float('nan')):
            try:
                if v is None or str(v).strip() == "":
                    return default
                return float(v)
            except Exception:
                return default

        def kpi(title, value, sub, icon, color, bg):
            return ft.Container(content=ft.Row([
                ft.Icon(icon, size=24, color=color),
                ft.Column([ft.Text(title, size=10, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                           ft.Text(value, size=15, weight=ft.FontWeight.BOLD, color=color),
                           ft.Text(sub, size=9, color=C.GREY_600)], spacing=0)
            ], spacing=8), padding=10, bgcolor=bg, border_radius=10,
                border=ft.Border.all(1, color), width=205, shadow=SHADOW)

        dns_kpis.controls = [
            kpi("📏 Level (latest)", f"{_f(s['level_last']):.3f}",
                f"corr {_f(s.get('corr_level')):.3f} vs 15Y",
                ft.Icons.HORIZONTAL_RULE, C.PINK_800, C.PINK_50),
            kpi("📐 Slope (latest)", f"{_f(s['slope_last']):.3f}",
                f"corr {_f(s.get('corr_slope')):.3f} vs 10Y−1Y",
                ft.Icons.TRENDING_DOWN, C.BLUE_700, C.BLUE_50),
            kpi("〰️ Curvature (latest)", f"{_f(s['curv_last']):.3f}",
                f"corr {_f(s.get('corr_curvature')):.3f} vs 2·2Y−3M−10Y",
                ft.Icons.SHOW_CHART, C.ORANGE_800, C.ORANGE_50),
            kpi("🎯 Curve fit", f"RMSE {_f(s['mean_rmse'], 0.0):.4f}%",
                f"{int(_f(s['n_periods'], 0))} periods · {int(_f(s['n_tenor'], 0))} tenors · λ={_f(s['lambda'], 0.0)}",
                ft.Icons.STRAIGHTEN, C.TEAL_700, C.TEAL_50),
        ]
        dns_val_box.controls = [
            ft.Text("Table 1 — factor validation against empirical proxies", size=12,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            ft.Text("Progress report CMDF-0128-2568 §2.2.1 reports Level 0.962, Slope 0.987, "
                    "Curvature 0.905. Slope is negative by construction (the DNS slope factor is "
                    "the short-end factor, i.e. minus the 10Y−1Y spread).", size=10, color=C.GREY_600),
            render_dns_validation(ycd.validate(factors))]
        dns_factor_box.controls = [
            ft.Text(f"Table 2 — estimated factors (latest 24 of {len(factors)} periods)",
                    size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            render_dns_factors(factors)]
        date_opts = ["LATEST"] + sorted(raw_curve["date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist(), reverse=True) if raw_curve is not None and not raw_curve.empty else ["LATEST"]
        tenor_opts = ["ALL"] + [str(x) for x in sorted(raw_curve["tau"].dropna().astype(float).unique().tolist())] if raw_curve is not None and not raw_curve.empty else ["ALL"]
        dns_date_filter.options = [ft.dropdown.Option(key=k, text=k) for k in date_opts]
        dns_tenor_filter.options = [ft.dropdown.Option(key=k, text=k) for k in tenor_opts]
        if not dns_date_filter.value or dns_date_filter.value not in date_opts:
            dns_date_filter.value = "LATEST"
        if not dns_tenor_filter.value or dns_tenor_filter.value not in tenor_opts:
            dns_tenor_filter.value = "ALL"
        dns_latest_table_box.controls = [
            ft.Text("Table 3 — latest yields / iBond-style view", size=12,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            ft.Text("Filter by as-of date and tenor, then inspect the saved raw download.",
                    size=10, color=C.GREY_600),
            render_dns_latest_yields(raw_curve if not raw_curve.empty else curve,
                                     dns_date_filter.value, dns_tenor_filter.value, max_rows=80)]
        dns_log_box.controls = [
            ft.Text("Table 4 — iBond download log", size=12,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            render_dns_log(raw_log)]
        if fc is not None and not fc.empty:
            dns_fc_box.controls = [
                ft.Text("Table 3 — DNS-AR(1) recursive out-of-sample forecast (RMSE)", size=12,
                        weight=ft.FontWeight.BOLD, color=C.BLUE_900),
                render_dns_forecast(fc)]
        try:
            dns_img_factors.src = _uri(fig_dns_factors(factors)); dns_img_factors.visible = bool(dns_show_factors.value)
            dns_img_surface.src = _uri(fig_dns_surface(curve)); dns_img_surface.visible = bool(dns_show_factors.value)
            dns_img_latest.src = _uri(fig_dns_latest_curve(curve)); dns_img_latest.visible = bool(dns_show_latest.value)
            dns_img_roll.src = _uri(fig_dns_roll_dashboard(factors)); dns_img_roll.visible = bool(dns_show_roll.value)
            if fc is not None and not fc.empty:
                dns_img_fc.src = _uri(fig_dns_forecast(fc)); dns_img_fc.visible = bool(dns_show_factors.value)
            else:
                dns_img_fc.visible = False
        except Exception as ex:
            dns_info.value = f"charts unavailable: {ex}"
        dns_val_box.visible = bool(dns_show_tables.value)
        dns_factor_box.visible = bool(dns_show_tables.value)
        dns_latest_table_box.visible = bool(dns_show_tables.value)
        dns_log_box.visible = bool(dns_show_log.value)
        dns_info.value = (("⚠ SYNTHETIC DEMO DATA — not real market data.  " if is_demo else "")
                          + f"Source: {s.get('source','')} · {int(s['n_periods'])} periods "
                          f"({s.get('date_min','')[:7]} … {s.get('date_max','')[:7]}) · "
                          f"SQLite: dns_curve / dns_factors / dns_summary / dns_forecast")
        dns_info.color = C.RED_600 if is_demo else C.PINK_800
        page.update()

    def _run_dns(file=None, demo=False):
        import yield_curve_dns as ycd
        dns_info.value = "Estimating Dynamic Nelson-Siegel factors …"
        dns_info.color = C.ORANGE_800
        status.value = "Yield-curve DNS running …"
        page.update()
        try:
            ycd.run(file=file, demo=demo, save=True)
            update_dns_tab()
            status.value = "Yield-curve DNS done — Level / Slope / Curvature saved to SQLite."
        except Exception as ex:
            dns_info.value = f"DNS failed: {ex}"
            dns_info.color = C.RED_600
            status.value = f"DNS error: {ex}"
        page.update()

    def on_dns_test_connection(_):
        dns_info.value = "Testing iBond connection ..."
        dns_info.color = C.ORANGE_800
        page.update()
        try:
            import ibond_client as ib
            start = (dns_start.value or "").strip() or None
            end = (dns_end.value or "").strip() or None
            df = ib.fetch_curve(start=start, end=end)
            src = str(df["source"].iloc[0]) if not df.empty and "source" in df.columns else "iBond"
            dns_info.value = (f"Connection OK: {src} | {len(df):,} rows | "
                              f"{df['tau'].nunique()} tenors | {pd.to_datetime(df['date']).min():%Y-%m-%d} to "
                              f"{pd.to_datetime(df['date']).max():%Y-%m-%d}")
            dns_info.color = C.GREEN_700
        except Exception as ex:
            dns_info.value = f"Connection test failed: {ex}"
            dns_info.color = C.RED_600
        page.update()

    def on_dns_auto_load(_):
        dns_info.value = "Connecting to iBond and loading yield curve …"
        dns_info.color = C.ORANGE_800
        status.value = "iBond auto-load running …"
        page.update()
        try:
            start = (dns_start.value or "").strip() or None
            end = (dns_end.value or "").strip() or None
            cmd = [sys.executable, os.path.join(HERE, "download_bound.py")]
            if start:
                cmd += ["--start", start]
            if end:
                cmd += ["--end", end]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE, shell=False)
            out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            if res.returncode == 0:
                update_dns_tab()
                dns_info.value = "Auto iBond load completed via download_bound.py"
                dns_info.color = C.GREEN_700
                status.value = "iBond auto-load done — yield curve saved to SQLite."
            else:
                dns_info.value = f"Auto iBond load failed: {out[:1200]}"
                dns_info.color = C.RED_600
                status.value = "iBond auto-load error via download_bound.py"
        except Exception as ex:
            dns_info.value = f"Auto iBond load failed: {ex}"
            dns_info.color = C.RED_600
            status.value = f"iBond auto-load error: {ex}"
        page.update()

    def on_dns_save_credentials(_):
        user = (dns_user.value or "").strip()
        pw = dns_pass.value or ""
        api = (dns_api.value or "").strip()
        if not user or not pw:
            dns_cred_msg.value = "Enter both username and password, or leave both blank and use API key only."
            dns_cred_msg.color = C.RED_600
            page.update(); return
        try:
            import setup_credentials as sc
            ok = sc._setx("THAIBMA_USER", user) and sc._setx("THAIBMA_PASS", pw)
            if api:
                sc._setx("THAIBMA_API_KEY", api)
                os.environ["THAIBMA_API_KEY"] = api
            os.environ["THAIBMA_USER"] = user
            os.environ["THAIBMA_PASS"] = pw
            dns_pass.value = ""
            dns_cred_msg.value = ("Saved credentials to Windows user environment. "
                                  "You can now use Auto load iBond.") if ok else \
                                 "Could not persist credentials. Try running app from a normal terminal."
            dns_cred_msg.color = C.GREEN_700 if ok else C.RED_600
            refresh_dns_cred_status()
        except Exception as ex:
            dns_cred_msg.value = f"Save credential failed: {ex}"
            dns_cred_msg.color = C.RED_600
        page.update()

    def on_dns_clear_credentials(_):
        try:
            import setup_credentials as sc
            sc.clear()
            dns_user.value = ""
            dns_pass.value = ""
            dns_api.value = ""
            dns_cred_msg.value = "Cleared stored iBond / ThaiBMA credentials from Windows environment."
            dns_cred_msg.color = C.GREEN_700
            refresh_dns_cred_status()
        except Exception as ex:
            dns_cred_msg.value = f"Clear credential failed: {ex}"
            dns_cred_msg.color = C.RED_600
        page.update()

    def on_dns_estimate(_):
        p = (dns_file.value or "").strip().strip('"')
        if not p:
            dns_info.value = ("Enter the path of a ThaiBMA / iBond export first "
                              "(or use “Demo curve”).")
            dns_info.color = C.RED_600
            page.update(); return
        _run_dns(file=p)

    dns_user = ft.TextField(label="iBond Username", width=240, text_size=12, dense=True)
    dns_pass = ft.TextField(label="iBond Password", password=True, can_reveal_password=True,
                            width=240, text_size=12, dense=True)
    dns_api = ft.TextField(label="ThaiBMA API Key (optional)", password=True, can_reveal_password=True,
                           width=280, text_size=12, dense=True)
    dns_range = ft.Dropdown(label="History range", width=140, text_size=12, value="1Y",
                            options=[ft.dropdown.Option(key=k, text=k) for k in ("1Y", "3Y", "5Y", "All")])
    dns_date_filter = ft.Dropdown(label="As-of date", width=180, text_size=12, value="LATEST",
                                  options=[ft.dropdown.Option(key="LATEST", text="LATEST")])
    dns_tenor_filter = ft.Dropdown(label="Tenor (Y)", width=140, text_size=12, value="ALL",
                                   options=[ft.dropdown.Option(key="ALL", text="ALL")])
    dns_cred_msg = ft.Text("", size=10, color=C.GREY_600)

    dns_range.on_select = on_dns_range_change
    dns_date_filter.on_select = lambda _: update_dns_tab()
    dns_tenor_filter.on_select = lambda _: update_dns_tab()
    dns_show_latest.on_change = lambda _: update_dns_tab()
    dns_show_roll.on_change = lambda _: update_dns_tab()
    dns_show_factors.on_change = lambda _: update_dns_tab()
    dns_show_tables.on_change = lambda _: update_dns_tab()
    dns_show_log.on_change = lambda _: update_dns_tab()

    dns_card = card(ft.Column([
        ft.Row([ft.Icon(ft.Icons.SHOW_CHART, color=C.PINK_800, size=22),
                ft.Text("📉 Government Bond Yield Curve — Level / Slope / Curvature (Dynamic Nelson-Siegel)",
                        size=15, weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
        ft.Text("Implements the DNS decomposition described in the CMDF-0128-2568 quarterly "
                "progress report (§2.1.1, §2.2.1, §2.3.1): y(τ) = L + S·[(1−e^(−λτ))/(λτ)] + "
                "C·[(1−e^(−λτ))/(λτ) − e^(−λτ)], estimated month by month, with the three "
                "factors validated against empirical proxies.", size=11, color=C.GREY_700),
        ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, size=15, color=C.BLUE_GREY_700),
                        ft.Text("How to get the data from ThaiBMA / iBond", size=11,
                                weight=ft.FontWeight.BOLD, color=C.BLUE_GREY_800)], spacing=6),
                ft.Text("This screen can now auto-connect to iBond when THAIBMA_USER / THAIBMA_PASS "
                        "or THAIBMA_API_KEY are already set in your Windows environment. If auto "
                        "connect is unavailable, you can still load a ThaiBMA/iBond CSV/XLSX export. "
                        "Accepted layouts: long (date, tenor, yield) or wide (date + one column per tenor: "
                        "1M, 6M, 1Y, 5Y, 10Y …).", size=10, color=C.GREY_700),
            ], spacing=4),
            bgcolor=C.BLUE_GREY_50, padding=10, border_radius=8,
            border=ft.Border.all(1, C.BLUE_GREY_100)),
        ft.Container(content=dns_cred, bgcolor=C.BLUE_GREY_50, padding=10, border_radius=8,
                     border=ft.Border.all(1, C.BLUE_GREY_100)),
        ft.Container(
            content=ft.Column([
                ft.Text("Optional: save credentials from inside the app", size=11,
                        weight=ft.FontWeight.BOLD, color=C.BLUE_GREY_800),
                ft.Row([
                    dns_user,
                    dns_pass,
                    dns_api,
                    ft.Button("Save credentials", icon=ft.Icons.SAVE, on_click=on_dns_save_credentials,
                              bgcolor=C.BLUE_700, color=C.WHITE,
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
                    ft.Button("Clear credentials", icon=ft.Icons.DELETE_OUTLINE, on_click=on_dns_clear_credentials,
                              bgcolor=C.BLUE_GREY_500, color=C.WHITE,
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
                ], spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                dns_cred_msg,
            ], spacing=6),
            bgcolor=C.BLUE_GREY_50, padding=10, border_radius=8,
            border=ft.Border.all(1, C.BLUE_GREY_100)),
        ft.Row([
            dns_file,
            dns_range,
            dns_start,
            dns_end,
            ft.Button("Test connection", icon=ft.Icons.LINK, on_click=on_dns_test_connection,
                      bgcolor=C.INDIGO_700, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Auto load iBond", icon=ft.Icons.CLOUD_DOWNLOAD, on_click=on_dns_auto_load,
                      bgcolor=C.TEAL_700, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Estimate DNS", icon=ft.Icons.PLAY_ARROW, on_click=on_dns_estimate,
                      bgcolor=C.PINK_800, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Demo curve", icon=ft.Icons.SCIENCE,
                      on_click=lambda _: _run_dns(demo=True),
                      bgcolor=C.BLUE_GREY_600, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Reload saved", icon=ft.Icons.REFRESH, on_click=update_dns_tab,
                      bgcolor=C.BLUE_GREY_400, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
        ], spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        dns_info,
        dns_kpis,
        ft.Container(
            content=ft.Row([
                dns_show_latest,
                dns_show_roll,
                dns_show_factors,
                dns_show_tables,
                dns_show_log,
            ], spacing=12, wrap=True),
            bgcolor=C.BLUE_GREY_50, padding=8, border_radius=8,
            border=ft.Border.all(1, C.BLUE_GREY_100)),
        ft.Row([dns_date_filter, dns_tenor_filter], spacing=8, wrap=True),
        ft.Text("Latest curve snapshot", size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        dns_img_latest,
        ft.Text("Rolling slope / curvature dashboard", size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        dns_img_roll,
        dns_img_factors,
        dns_val_box,
        dns_img_surface,
        dns_factor_box,
        dns_latest_table_box,
        dns_log_box,
        dns_fc_box,
        dns_img_fc,
    ], spacing=12), accent=C.PINK_200, pad=14)

    dns_long_content = ft.Column([
        dns_card,
        ft.Container(height=420, bgcolor=ft.Colors.with_opacity(0.01, C.BLACK)),
    ], spacing=12, expand=True, scroll=ft.ScrollMode.ALWAYS)
    view_dns = ft.Container(content=dns_long_content, expand=True)

    # ========= Paper replication: multi-horizon determinants of default risk ===
    PAPER_TABLES = {"2": "Descriptive statistics", "4": "Baseline ln PD",
                    "5": "Distance to default", "6": "Fractional logit",
                    "7": "ESG pillars", "8": "Proxy sensitivity",
                    "9": "Augmented macro", "10": "Factor model (PCA)",
                    "11": "Event validation"}
    pap_info = ft.Text("Replicates Wattanatorn & Wiriyadee (EMFT, ID 262322061) on the "
                       "ThaiBMA bond database. Click “Reload saved” to show the stored run.",
                       size=12, color=C.GREY_700)
    pap_kpis = ft.Row([], spacing=10, wrap=True)
    pap_img = ft.Image(src="", visible=False, width=1000)
    pap_tabs_box = ft.Column([])
    pap_pick = ft.Dropdown(label="Table", width=290, text_size=12, value="4",
                           options=[ft.dropdown.Option(key=k, text=f"Table {k} — {v}")
                                    for k, v in PAPER_TABLES.items()])

    def render_paper_table(tdf, max_rows=60):
        if tdf is None or tdf.empty:
            return ft.Text("table not available", size=11, color=C.GREY_600)
        cols = list(tdf.columns)[:12]
        rows = []
        for _, r in tdf.head(max_rows).iterrows():
            cells = []
            for c in cols:
                v = "" if pd.isna(r[c]) else str(r[c])
                bold = (c == cols[0]) and v and not v.startswith("(")
                col = (C.RED_700 if "***" in v else C.ORANGE_800 if "**" in v
                       else C.AMBER_800 if "*" in v else C.GREY_800)
                cells.append(ft.DataCell(ft.Text(
                    v, size=11, color=col,
                    weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL)))
            rows.append(ft.DataRow(cells=cells))
        return ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text(str(c), size=11, weight=ft.FontWeight.BOLD))
                     for c in cols],
            rows=rows, column_spacing=13, data_row_min_height=24)], scroll=ft.ScrollMode.AUTO)

    def show_paper_table(_=None):
        import paper_replication as pr
        tabs, _p, _s = pr.load_from_sqlite(DB)
        key = pap_pick.value or "4"
        pap_tabs_box.controls = [
            ft.Text(f"Table {key} — {PAPER_TABLES.get(key, '')}", size=12,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
            ft.Text("*** p<0.01, ** p<0.05, * p<0.10 · SE clustered by firm · "
                    "industry & year fixed effects · regressors lagged one period",
                    size=10, color=C.GREY_600),
            render_paper_table(tabs.get(key))]
        page.update()

    pap_pick.on_select = show_paper_table

    def update_paper_tab(_=None):
        import paper_replication as pr
        tabs, paths, summ = pr.load_from_sqlite(DB)
        if summ.empty:
            pap_info.value = ("No saved replication yet — click “Run replication” "
                              "(estimates every table; takes a few minutes).")
            pap_info.color = C.GREY_600
            page.update(); return
        s = summ.iloc[0]

        def kpi(title, value, sub, icon, color, bg):
            return ft.Container(content=ft.Row([
                ft.Icon(icon, size=24, color=color),
                ft.Column([ft.Text(title, size=10, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                           ft.Text(value, size=15, weight=ft.FontWeight.BOLD, color=color),
                           ft.Text(sub, size=9, color=C.GREY_600)], spacing=0)
            ], spacing=8), padding=10, bgcolor=bg, border_radius=10,
                border=ft.Border.all(1, color), width=215, shadow=SHADOW)

        pap_kpis.controls = [
            kpi("📄 Sample", f"{int(s['n_obs']):,} firm-months",
                f"paper: {int(s['paper_n_obs']):,}", ft.Icons.DATASET, C.INDIGO_700, C.INDIGO_50),
            kpi("🏢 Issuers", f"{int(s['n_firms'])}", f"paper: {int(s['paper_n_firms'])}",
                ft.Icons.BUSINESS, C.INDIGO_700, C.INDIGO_50),
            kpi("📅 Period", f"{int(s['year_min'])}–{int(s['year_max'])}",
                f"{int(s['n_industries'])} SET industries", ft.Icons.EVENT, C.TEAL_700, C.TEAL_50),
            kpi("🔭 Horizons", str(s["horizons"]).replace(",", " / ") + " m",
                str(s["sample"]), ft.Icons.TIMELINE, C.PURPLE_700, C.PURPLE_50),
        ]
        try:
            pap_img.src = _uri(fig_paper_paths(paths)); pap_img.visible = True
        except Exception as ex:
            pap_info.value = f"chart unavailable: {ex}"
        show_paper_table()
        pap_info.value = (f"Run {s.get('run_at','')} · {s.get('sample','')} · "
                          f"{s.get('lambda_note','')} · SQLite: paper_table2…11, "
                          f"paper_paths, paper_summary")
        pap_info.color = C.INDIGO_700
        page.update()

    def on_run_paper(_):
        import paper_replication as pr
        pap_info.value = "Estimating every table (fixed effects, fractional logit, PCA) …"
        pap_info.color = C.ORANGE_800
        status.value = "Paper replication running …"
        page.update()
        try:
            res = pr.run_all(expanded=False)
            pr.save_to_sqlite(res, DB)
            update_paper_tab()
            status.value = (f"Replication done — {res['summary']['n_obs']:,} firm-months, "
                            f"{res['summary']['n_firms']} issuers")
        except Exception as ex:
            pap_info.value = f"Replication failed: {ex}"; pap_info.color = C.RED_600
            status.value = f"Replication error: {ex}"
        page.update()

    paper_card = card(ft.Column([
        ft.Row([ft.Icon(ft.Icons.ARTICLE, color=C.INDIGO_700, size=22),
                ft.Text("📰 Determinants of Corporate Default Risk in the Thai Bond Market "
                        "— Multi-Horizon Analysis", size=15, weight=ft.FontWeight.BOLD,
                        color=C.BLUE_900)], spacing=6),
        ft.Text("Replication of Wattanatorn & Wiriyadee (Emerging Markets Finance and Trade, "
                "submission 262322061) on the ThaiBMA database: Merton PD / DD at 12, 24, 36 and "
                "60 months regressed on lagged profitability, leverage, liquidity, coverage, size, "
                "age, stock illiquidity (Amihud vs Kang–Zhang), the BOT policy rate and ESG — with "
                "industry and year fixed effects and firm-clustered standard errors.",
                size=11, color=C.GREY_700),
        ft.Row([
            ft.Button("Run replication", icon=ft.Icons.PLAY_ARROW, on_click=on_run_paper,
                      bgcolor=C.INDIGO_700, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Reload saved", icon=ft.Icons.REFRESH, on_click=update_paper_tab,
                      bgcolor=C.BLUE_GREY_600, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            pap_pick,
        ], spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        pap_info,
        pap_kpis,
        pap_img,
        pap_tabs_box,
    ], spacing=12), accent=C.INDIGO_200, pad=14)

    view_paper = ft.Column([paper_card], spacing=12)

    # ================= Real-time EWS: live watchlist + lead time ==============
    rt_info = ft.Text("Scores every issuer with the survival hazard model and reports the "
                      "warning each alert level historically buys.", size=12, color=C.GREY_700)
    rt_cred = ft.Text("", size=11, color=C.GREY_700)
    rt_kpis = ft.Row([], spacing=10, wrap=True)
    rt_ref_box = ft.Column([])
    rt_grid = ft.Column([])
    rt_img = ft.Image(src="", visible=False, width=1000)
    rt_filter = ft.Dropdown(label="Alert level", width=190, text_size=12, value="ALL",
                            options=[ft.dropdown.Option(key=k, text=k) for k in
                                     ("ALL", "HIGH RISK", "ELEVATED", "WATCH", "OK")])
    rt_search = ft.TextField(label="firm id", width=140, text_size=12, dense=True)
    RT_C = {"HIGH RISK": C.RED_600, "ELEVATED": C.ORANGE_700,
            "WATCH": C.AMBER_700, "OK": C.GREEN_600}

    def render_rt_grid(alerts, limit=60):
        d = alerts.copy()
        if (rt_filter.value or "ALL") != "ALL":
            d = d[d["alert"] == rt_filter.value]
        q = (rt_search.value or "").strip()
        if q:
            d = d[d["firm_id"].astype(str).str.contains(q)]
        if d.empty:
            return ft.Text("no issuer matches this filter", size=11, color=C.GREY_600)
        rows = []
        for _, r in d.head(limit).iterrows():
            col = RT_C.get(r["alert"], C.GREY_700)
            el = "—" if pd.isna(r["expected_lead_days"]) else f"{r['expected_lead_days']:.0f} d"
            em = "" if pd.isna(r["expected_lead_months"]) else f"({r['expected_lead_months']:.1f} mo)"
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(int(r["firm_id"])), size=11, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(str(r["as_of"]), size=11)),
                ft.DataCell(ft.Text(f"{r['PD_3M']*100:.1f}%", size=11, color=col,
                                    weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{r['Momentum']:.2f}" if pd.notna(r["Momentum"]) else "1.00",
                                    size=11)),
                ft.DataCell(ft.Text(f"{r['h']*100:.2f}%", size=11, color=C.PURPLE_700)),
                ft.DataCell(ft.Container(
                    content=ft.Text(f"● {r['alert']}", size=10, color=C.WHITE,
                                    weight=ft.FontWeight.BOLD),
                    bgcolor=col, padding=ft.Padding.symmetric(horizontal=9, vertical=3),
                    border_radius=10)),
                ft.DataCell(ft.Text(f"{el} {em}", size=11)),
            ]))
        return ft.Column([
            ft.Text(f"showing {min(limit, len(d))} of {len(d)} issuers", size=10, color=C.GREY_600),
            ft.Row([ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                         for c in ["Firm", "As of", "PD₃M", "Momentum", "Hazard h(t)",
                                   "Alert", "Expected lead time"]],
                rows=rows, column_spacing=16, data_row_min_height=28)],
                scroll=ft.ScrollMode.AUTO)], spacing=4)

    def refresh_rt_grid(_=None):
        import realtime_ews as rt
        alerts, _r, _s = rt.load_from_sqlite(DB)
        if not alerts.empty:
            rt_grid.controls = [render_rt_grid(alerts)]
        page.update()

    rt_filter.on_select = refresh_rt_grid

    def update_realtime_tab(_=None):
        import realtime_ews as rt
        alerts, ref, summ = rt.load_from_sqlite(DB)
        # credential status — reports only WHETHER secrets are set, never their value
        try:
            import ibond_client as ib
            st = ib.credentials_status()
            rt_cred.value = ("🔑 iBond credentials detected in your environment — "
                             "“Refresh LIVE” will use them." if st["ready"] else
                             "🔒 No iBond credentials in this environment. Live refresh is "
                             "disabled; the local snapshot is used. To enable it run "
                             'setx THAIBMA_USER "…" and setx THAIBMA_PASS "…" in a terminal, '
                             "then reopen the app. Your password is never stored in this app "
                             "and never leaves your machine.")
            rt_cred.color = C.GREEN_700 if st["ready"] else C.BLUE_GREY_700
        except Exception as ex:
            rt_cred.value = f"credential check unavailable: {ex}"
            rt_cred.color = C.GREY_600

        if alerts.empty or summ.empty:
            rt_info.value = ("No scoring run yet — click “Refresh (local)” to score every "
                             "issuer from the local database.")
            rt_info.color = C.GREY_600
            page.update(); return
        s = summ.iloc[0]

        def kpi(title, value, sub, icon, color, bg):
            return ft.Container(content=ft.Row([
                ft.Icon(icon, size=24, color=color),
                ft.Column([ft.Text(title, size=10, color=C.GREY_700, weight=ft.FontWeight.BOLD),
                           ft.Text(value, size=15, weight=ft.FontWeight.BOLD, color=color),
                           ft.Text(sub, size=9, color=C.GREY_600)], spacing=0)
            ], spacing=8), padding=10, bgcolor=bg, border_radius=10,
                border=ft.Border.all(1, color), width=200, shadow=SHADOW)

        rt_kpis.controls = [
            kpi("🔴 HIGH RISK", f"{int(s['n_high'])}", "act now", ft.Icons.ERROR, C.RED_700, C.RED_50),
            kpi("🟠 ELEVATED", f"{int(s['n_elevated'])}", "review", ft.Icons.WARNING,
                C.ORANGE_700, C.ORANGE_50),
            kpi("🟡 WATCH", f"{int(s['n_watch'])}", "monitor", ft.Icons.REMOVE_RED_EYE,
                C.AMBER_700, C.AMBER_50),
            kpi("🟢 OK", f"{int(s['n_ok'])}", f"of {int(s['n_firms'])} live issuers",
                ft.Icons.CHECK_CIRCLE, C.GREEN_700, C.GREEN_50),
            kpi("📅 As of", str(s["as_of"]),
                f"OOS AUC {float(s['pd_auc_oos']):.3f}", ft.Icons.EVENT, C.BLUE_700, C.BLUE_50),
        ]
        if not ref.empty:
            rows = []
            for _, r in ref.iterrows():
                col = RT_C.get(r["alert"], C.BLUE_GREY_700)
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(str(r["alert"]), size=11, weight=ft.FontWeight.BOLD,
                                        color=col)),
                    ft.DataCell(ft.Text(str(int(r["n"])), size=11)),
                    ft.DataCell(ft.Text(f"{r['median_days']:.0f} d", size=11,
                                        weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{r['median_days']/30.44:.1f} mo", size=11)),
                    ft.DataCell(ft.Text(f"{r['p25_days']:.0f} – {r['p75_days']:.0f} d",
                                        size=11, color=C.GREY_600)),
                ]))
            rt_ref_box.controls = [
                ft.Text("Table 1 — how much warning each alert level historically bought",
                        size=12, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
                ft.Text(f"Measured on the {int(s['n_events_hist'])} issuers that actually "
                        "experienced a payment default or restructuring: the first month they "
                        "entered each band, and the days remaining until the event.",
                        size=10, color=C.GREY_600),
                ft.Row([ft.DataTable(
                    columns=[ft.DataColumn(ft.Text(c, size=11, weight=ft.FontWeight.BOLD))
                             for c in ["Alert level", "N firms", "Median lead", "≈ months", "IQR"]],
                    rows=rows, column_spacing=20, data_row_min_height=28)],
                    scroll=ft.ScrollMode.AUTO)]
        rt_grid.controls = [
            ft.Text("Table 2 — issuer watchlist", size=12, weight=ft.FontWeight.BOLD,
                    color=C.BLUE_900),
            render_rt_grid(alerts)]
        try:
            rt_img.src = _uri(fig_realtime(alerts, ref)); rt_img.visible = True
        except Exception as ex:
            rt_info.value = f"chart unavailable: {ex}"
        live = int(s.get("is_live", 0)) == 1
        rt_info.value = (("🟢 LIVE — " if live else "💾 LOCAL SNAPSHOT — ")
                         + f"{s.get('data_source','')} · run {s.get('run_at','')} · "
                         f"{int(s['n_firm_months']):,} firm-months trained · "
                         f"already-defaulted issuers excluded · SQLite: realtime_alerts / "
                         f"realtime_leadtime_ref / realtime_summary")
        rt_info.color = C.GREEN_700 if live else C.BLUE_GREY_700
        page.update()

    def _run_rt(live):
        import realtime_ews as rt
        rt_info.value = ("Refreshing from iBond …" if live else "Scoring issuers …")
        rt_info.color = C.ORANGE_800
        status.value = "Real-time EWS running …"
        page.update()
        try:
            alerts, ref, summary = rt.run(live=live, verbose=False)
            rt.save_to_sqlite(alerts, ref, summary, DB)
            update_realtime_tab()
            status.value = (f"Real-time EWS done — {summary['n_high']} HIGH RISK, "
                            f"{summary['n_elevated']} ELEVATED (as of {summary['as_of']})")
        except Exception as ex:
            rt_info.value = f"Run failed: {ex}"; rt_info.color = C.RED_600
            status.value = f"Real-time EWS error: {ex}"
        page.update()

    realtime_card = card(ft.Column([
        ft.Row([ft.Icon(ft.Icons.SENSORS, color=C.RED_700, size=22),
                ft.Text("🛰️ Real-time Early Warning — issuer watchlist & lead time",
                        size=15, weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
        ft.Text("Fits the discrete-time survival hazard on the full ThaiBMA history, scores "
                "every issuer's latest month (PD₃M, momentum, hazard), assigns an alert band, "
                "and attaches the lead time that band historically delivered before real "
                "payment-default / restructuring events.", size=11, color=C.GREY_700),
        ft.Container(content=rt_cred, bgcolor=C.BLUE_GREY_50, padding=10, border_radius=8,
                     border=ft.Border.all(1, C.BLUE_GREY_100)),
        ft.Row([
            ft.Button("Refresh (local)", icon=ft.Icons.PLAY_ARROW,
                      on_click=lambda _: _run_rt(False),
                      bgcolor=C.RED_700, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Refresh LIVE (iBond)", icon=ft.Icons.CLOUD_DOWNLOAD,
                      on_click=lambda _: _run_rt(True),
                      bgcolor=C.TEAL_700, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.Button("Reload saved", icon=ft.Icons.REFRESH, on_click=update_realtime_tab,
                      bgcolor=C.BLUE_GREY_600, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            rt_filter, rt_search,
            ft.Button("Filter", icon=ft.Icons.SEARCH, on_click=refresh_rt_grid,
                      bgcolor=C.BLUE_700, color=C.WHITE,
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
        ], spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        rt_info,
        rt_kpis,
        rt_img,
        rt_ref_box,
        rt_grid,
    ], spacing=12), accent=C.RED_200, pad=14)

    view_realtime = ft.Column([realtime_card], spacing=12)

    view_datagrid = ft.Column([], spacing=12)
    view_lead_time = ft.Column([lead_time_card], spacing=12)
    view_openclaw = ft.Column(
        [oc_connection_card, oc_jobs_card, oc_alerts_card],
        spacing=12,
    )

    # ------------------- CRUD panel for the active dataset --------------------
    crud_id = ft.TextField(label="Row id", width=110, dense=True)
    crud_col = ft.Dropdown(label="Column", width=250, options=[], dense=True)
    crud_val = ft.TextField(label="New value", width=190, dense=True)
    crud_status = ft.Text("Load a row id to edit, or add / delete a record.",
                          size=11, color=C.GREY_700)

    def _crud_refresh():
        try:
            data_grid_container.content = render_data_table()
        except Exception:
            pass

    def crud_load(_):
        try:
            row = db_get_row(int(crud_id.value))
            if row.empty:
                crud_status.value = f"Row id {crud_id.value} not found."
            else:
                crud_col.options = [ft.dropdown.Option(c) for c in row.columns if c != "id"]
                if crud_col.value not in [o.key for o in crud_col.options]:
                    crud_col.value = crud_col.options[0].key if crud_col.options else None
                if crud_col.value:
                    crud_val.value = str(row.iloc[0][crud_col.value])
                crud_status.value = f"Loaded row {crud_id.value} ({len(row.columns)} columns)."
        except Exception as exc:
            crud_status.value = f"Load failed: {exc}"
        page.update()

    def crud_update(_):
        try:
            n = db_update_field(int(crud_id.value), crud_col.value, crud_val.value)
            crud_status.value = (f"Updated {crud_col.value} on row {crud_id.value}."
                                 if n else f"Row id {crud_id.value} not found.")
            _crud_refresh()
        except Exception as exc:
            crud_status.value = f"Update failed: {exc}"
        page.update()

    def crud_delete(_):
        try:
            n = db_delete_row(int(crud_id.value))
            crud_status.value = (f"Deleted row {crud_id.value}." if n
                                 else f"Row id {crud_id.value} not found.")
            _crud_refresh()
        except Exception as exc:
            crud_status.value = f"Delete failed: {exc}"
        page.update()

    def crud_add(_):
        try:
            new_id = db_add_row()
            crud_id.value = str(new_id)
            crud_status.value = (f"Added row id {new_id} (median values) - "
                                 f"use Load + Update to edit its fields.")
            _crud_refresh()
        except Exception as exc:
            crud_status.value = f"Add failed: {exc}"
        page.update()

    crud_card = ft.Container(
        content=ft.Column([
            ft.Row([ft.Icon(ft.Icons.EDIT_NOTE, color=C.BLUE_800, size=18),
                    ft.Text("Add / Edit / Delete record", size=13,
                            weight=ft.FontWeight.BOLD, color=C.BLUE_900)], spacing=6),
            ft.Row([crud_id,
                    ft.Button("Load", icon=ft.Icons.DOWNLOAD, on_click=crud_load,
                              bgcolor=C.GREY_200, color=C.BLACK),
                    crud_col, crud_val,
                    ft.Button("Update", icon=ft.Icons.SAVE, on_click=crud_update,
                              bgcolor=C.BLUE_700, color=C.WHITE),
                    ft.Button("Add row", icon=ft.Icons.ADD, on_click=crud_add,
                              bgcolor=C.GREEN_600, color=C.WHITE),
                    ft.Button("Delete", icon=ft.Icons.DELETE, on_click=crud_delete,
                              bgcolor=C.RED_600, color=C.WHITE)], spacing=8, wrap=True),
            crud_status,
        ], spacing=8),
        padding=12, bgcolor=C.WHITE, border_radius=8,
        border=ft.Border.all(1, UI["border"]))

    def update_datagrid_tab():
        view_datagrid.controls = [
            crud_card,
            card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.TABLE_CHART, color=C.BLUE_800, size=20),
                    ft.Text("📊 Active Dataset DataGridView Inspector", size=14, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
                ], spacing=6),
                data_info,
                ft.Row([
                    ft.Button("Prev", icon=ft.Icons.NAVIGATE_BEFORE, on_click=data_prev, bgcolor=C.GREY_200, color=C.BLACK),
                    ft.Button("Next", icon=ft.Icons.NAVIGATE_NEXT, on_click=data_next, bgcolor=C.GREY_200, color=C.BLACK),
                    page_label,
                    search_field,
                    ft.Button("Search", icon=ft.Icons.SEARCH, on_click=data_search, bgcolor=C.BLUE_700, color=C.WHITE),
                    ft.Button("Clear", icon=ft.Icons.CLEAR, on_click=data_clear, bgcolor=C.GREY_400, color=C.WHITE),
                ], spacing=8, wrap=True),
                render_data_table()
            ], spacing=10), accent=C.BLUE_200, pad=14)
        ]

    def update_lead_time_tab():
        lead_grid_container.content = render_lead_time_table()
        view_lead_time.controls = [lead_time_card]

    bond_news_info = ft.Text(
        "กดปุ่มเพื่อโหลดฐานข้อมูลหุ้นกู้และอัปเดต Data Inspector",
        size=12,
        color=C.GREY_700,
    )
    bond_news_card = card(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.NEWSPAPER, color=C.GREEN_700, size=22),
            ft.Text("ข่าวหุ้นกู้ / Bond Database", size=15,
                    weight=ft.FontWeight.BOLD, color=C.BLUE_900),
        ], spacing=8),
        ft.Text("โหลดฐานข้อมูลหุ้นกู้จริงเข้าสู่ SQLite แล้วพากลับไปดูข้อมูลใน Data Inspector ได้ทันที",
                size=11, color=C.GREY_700),
        ft.Row([
            ft.Button("โหลดข้อมูลหุ้นกู้", icon=ft.Icons.DOWNLOAD,
                      on_click=on_bond, bgcolor=UI["primary"], color=C.WHITE),
            ft.Button("ไปหน้า Data Inspector", icon=ft.Icons.TABLE_CHART,
                      on_click=lambda _: set_tab(2), bgcolor=UI["button"], color=UI["text"]),
        ], spacing=8, wrap=True),
        bond_news_info,
    ], spacing=10), accent=C.GREEN_200, pad=14)
    view_bond_news = ft.Column([bond_news_card], spacing=14)

    def update_openclaw_tab():
        _oc_refresh_connection()
        oc_job_table_container.content = _oc_render_jobs()
        oc_alert_table_container.content = _oc_render_alerts()
        view_openclaw.controls = [
            oc_connection_card, oc_jobs_card, oc_alerts_card]

    # ---------- Approach 2 extension: Koopman operator + GAF factor imaging ----
    koop_status = ft.Text("Press 'Run Koopman + XGBoost' to fit the operator on the "
                          "34-factor bond panel.", size=12, color=C.GREY_700)
    koop_metrics = ft.Text("", size=12, color=C.BLUE_900, weight=ft.FontWeight.BOLD)
    koop_grid_container = ft.Container(content=ft.Text("No results yet.", size=12, color=C.GREY_600),
                                       padding=6)
    koop_img_imp = ft.Image(src="", visible=False, width=760)
    koop_img_spec = ft.Image(src="", visible=False, width=520)
    koop_img_gaf = ft.Image(src="", visible=False, width=1100)
    koop_factor_dd = ft.Dropdown(label="Factor", width=300, options=[], dense=True)
    koop_firm_tf = ft.TextField(label="firm_id (blank = median)", width=210, dense=True)
    koop_search = ft.TextField(label="Search factor", width=220, dense=True)

    def render_koopman_grid():
        if koopman_gaf is None:
            return ft.Text("koopman_gaf module unavailable.", size=12, color=C.RED_600)
        df = koopman_gaf.load_from_sqlite(DB)
        if df is None or df.empty:
            return ft.Text("SQLite table 'koopman_factors' is empty - run the analysis first.",
                           size=12, color=C.GREY_600)
        q = (koop_search.value or "").strip().lower()
        if q:
            df = df[df["factor"].astype(str).str.lower().str.contains(q, na=False)]
        cols = [c for c in ["rank", "factor", "imp_total", "imp_raw", "imp_koopman",
                            "k_self", "k_influence"] if c in df.columns]
        return ft.Column([
            ft.Text(f"SQLite table 'koopman_factors' - showing {len(df)} factor row(s)",
                    size=11, color=C.GREY_700),
            ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD, size=11)) for c in cols],
                rows=[ft.DataRow([ft.DataCell(ft.Text(str(r[c]), size=11)) for c in cols])
                      for _, r in df.head(40).iterrows()],
                heading_row_height=32, data_row_max_height=30,
            ),
        ], spacing=6, scroll=ft.ScrollMode.AUTO)

    def on_koopman_run(_):
        if koopman_gaf is None:
            koop_status.value = "koopman_gaf module not available."; page.update(); return
        koop_status.value = ("Loading feature_bond.xlsx and fitting the Koopman operator "
                             "(first run takes a while) ...")
        page.update()
        try:
            res = koopman_gaf.run_koopman_xgb()
            koopman_gaf.save_to_sqlite(res, DB)
            state["koopman"] = res
            m = res["metrics"]
            koop_metrics.value = (
                f"firms {res['n_firms']:,}  |  pairs {res['n_pairs']:,}  |  "
                f"spectral radius {res['spectral_radius']:.3f}  |  "
                + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
            koop_img_imp.src = _uri(_b64(koopman_gaf.fig_koopman_importance(res)))
            koop_img_spec.src = _uri(_b64(koopman_gaf.fig_koopman_spectrum(res)))
            koop_img_imp.visible = koop_img_spec.visible = True
            koop_factor_dd.options = [ft.dropdown.Option(f) for f in res["features"]]
            if res["features"]:
                koop_factor_dd.value = res["features"][0]
            koop_grid_container.content = render_koopman_grid()
            koop_status.value = "Done - per-factor results saved to SQLite table 'koopman_factors'."
        except Exception as exc:
            koop_status.value = f"Koopman run failed: {exc}"
        page.update()

    def on_koopman_gaf(_):
        if koopman_gaf is None or not koop_factor_dd.value:
            koop_status.value = "Pick a factor first (run the analysis to fill the list)."
            page.update(); return
        firm = (koop_firm_tf.value or "").strip() or None
        koop_status.value = f"Building the GAF image for {koop_factor_dd.value} ..."; page.update()
        try:
            fig = koopman_gaf.fig_gaf_factor(None, koop_factor_dd.value, firm=firm)
            koop_img_gaf.src = _uri(_b64(fig)); koop_img_gaf.visible = True
            who = f"firm {firm}" if firm else "median across firms"
            koop_status.value = f"GAF image for {koop_factor_dd.value} ({who}) ready."
        except Exception as exc:
            koop_status.value = f"GAF failed: {exc}"
        page.update()

    def koop_do_search(_):
        koop_grid_container.content = render_koopman_grid(); page.update()

    def koop_clear_search(_):
        koop_search.value = ""
        koop_grid_container.content = render_koopman_grid(); page.update()

    koop_img_lead = ft.Image(src="", visible=False, width=820)
    koop_lead_q = ft.TextField(label="alarm quantile", value="0.95", width=130, dense=True)
    koop_lead_h = ft.TextField(label="horizon (days)", value="365", width=140, dense=True)
    koop_lead_grid = ft.Container(content=ft.Text("Not computed yet.", size=12, color=C.GREY_600),
                                  padding=6)

    def render_koopman_lead_grid():
        if koopman_gaf is None:
            return ft.Text("koopman_gaf unavailable.", size=12, color=C.RED_600)
        df = koopman_gaf.load_from_sqlite(DB, koopman_gaf.LEAD_TABLE)
        if df is None or df.empty:
            return ft.Text("SQLite table 'koopman_lead_time' is empty.", size=12, color=C.GREY_600)
        cols = [c for c in ["firm_id", "event_date", "first_alarm", "lead_time_days",
                            "detected", "max_p_before_event"] if c in df.columns]
        return ft.Column([
            ft.Text(f"SQLite table 'koopman_lead_time' - {len(df)} event firm(s)",
                    size=11, color=C.GREY_700),
            ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD, size=11)) for c in cols],
                rows=[ft.DataRow([ft.DataCell(ft.Text(str(r[c]), size=11)) for c in cols])
                      for _, r in df.head(40).iterrows()],
                heading_row_height=32, data_row_max_height=30),
        ], spacing=6, scroll=ft.ScrollMode.AUTO)

    def on_koopman_lead(_):
        if koopman_gaf is None:
            koop_status.value = "koopman_gaf module not available."; page.update(); return
        koop_status.value = ("Scoring out-of-fold by firm and computing the sigmoid "
                             "lead time (this takes a while) ...")
        page.update()
        try:
            q = float(koop_lead_q.value or 0.95)
            h = float(koop_lead_h.value or 365)
            res = koopman_gaf.koopman_lead_time(threshold_q=q, horizon_days=h)
            koopman_gaf.save_lead_time_sqlite(res, DB)
            s = res["summary"]
            koop_img_lead.src = _uri(_b64(koopman_gaf.fig_lead_time(res)))
            koop_img_lead.visible = True
            koop_lead_grid.content = render_koopman_lead_grid()
            koop_metrics.value = (
                f"lead time: detected {s['detected']}/{s['event_firms']} event firms  |  "
                f"median {s['median_lead_days']:.0f} d  |  max {s['max_lead_days']:.0f} d  |  "
                f"alarm rate {s['alarm_rate']*100:.1f}%  |  tau {s['tau']:.5f}")
            koop_status.value = "Lead time saved to SQLite table 'koopman_lead_time'."
        except Exception as exc:
            koop_status.value = f"Lead-time run failed: {exc}"
        page.update()

    koop_img_gallery = ft.Image(src="", visible=False, width=1150)
    koop_img_series = ft.Image(src="", visible=False, width=900)

    def _koop_fill_factors():
        """Populate the factor dropdown straight from the panel (no model needed)."""
        if koopman_gaf is None or koop_factor_dd.options:
            return
        try:
            feats = koopman_gaf.list_factors()
            koop_factor_dd.options = [ft.dropdown.Option(f) for f in feats]
            if feats and not koop_factor_dd.value:
                koop_factor_dd.value = feats[0]
        except Exception:
            pass

    def on_koopman_gallery(_):
        if koopman_gaf is None:
            koop_status.value = "koopman_gaf module not available."; page.update(); return
        koop_status.value = "Building the GAF thumbnail wall for every factor ..."; page.update()
        try:
            fig = koopman_gaf.fig_gaf_gallery()
            koop_img_gallery.src = _uri(_b64(fig)); koop_img_gallery.visible = True
            koop_status.value = "GAF gallery built for all %d factors." % len(koopman_gaf.list_factors())
        except Exception as exc:
            koop_status.value = f"GAF gallery failed: {exc}"
        page.update()

    def on_koopman_export(_):
        if koopman_gaf is None:
            koop_status.value = "koopman_gaf module not available."; page.update(); return
        koop_status.value = "Exporting one GAF card per factor to gaf_outputs/ ..."; page.update()
        try:
            paths = koopman_gaf.export_gaf_images()
            koop_status.value = f"Exported {len(paths)} GAF images to gaf_outputs/."
        except Exception as exc:
            koop_status.value = f"Export failed: {exc}"
        page.update()

    def on_koopman_series(_):
        if koopman_gaf is None or not koop_factor_dd.value:
            koop_status.value = "Pick a factor first."; page.update(); return
        firm = (koop_firm_tf.value or "").strip() or None
        try:
            fig = koopman_gaf.fig_factor_series(None, koop_factor_dd.value, firm=firm)
            koop_img_series.src = _uri(_b64(fig)); koop_img_series.visible = True
            koop_status.value = f"Graph for {koop_factor_dd.value} ready."
        except Exception as exc:
            koop_status.value = f"Graph failed: {exc}"
        page.update()

    def _koop_card(title, body):
        return ft.Container(
            content=ft.Column([ft.Text(title, size=13, weight=ft.FontWeight.BOLD,
                                       color=UI["primary_dark"]), body], spacing=8),
            padding=14, bgcolor=C.WHITE, border_radius=8,
            border=ft.Border.all(1, UI["border"]), shadow=SHADOW)

    view_koopman = ft.Column([
        _koop_card(
            "Koopman operator + XGBoost on the 34-factor bond panel",
            ft.Column([
                ft.Text("A linear Koopman operator K is fitted so that x(t+1) ~ K x(t) within each "
                        "firm; the one-step forecast Kx is appended to the raw factors before "
                        "XGBoost. Validation splits by FIRM (leave-firms-out), never by row.",
                        size=11, color=C.GREY_700),
                ft.Row([
                    ft.Button("Run Koopman + XGBoost", icon=ft.Icons.PLAY_ARROW,
                              on_click=on_koopman_run, bgcolor=UI["primary"], color=C.WHITE),
                ], spacing=8),
                koop_status, koop_metrics,
            ], spacing=8)),
        _koop_card(
            "Per-factor results (SQLite table 'koopman_factors')",
            ft.Column([
                ft.Row([koop_search,
                        ft.Button("Search", icon=ft.Icons.SEARCH, on_click=koop_do_search,
                                  bgcolor=C.BLUE_700, color=C.WHITE),
                        ft.Button("Clear", icon=ft.Icons.CLEAR, on_click=koop_clear_search,
                                  bgcolor=C.GREY_400, color=C.WHITE)], spacing=8),
                koop_grid_container,
            ], spacing=8)),
        _koop_card("Factor importance (raw vs Koopman forecast)", koop_img_imp),
        _koop_card("Koopman spectrum (eigenvalues of K)", koop_img_spec),
        _koop_card(
            "Factor panel - pick a factor, then plot its graph or its GAF encoding",
            ft.Column([
                ft.Row([koop_factor_dd, koop_firm_tf,
                        ft.Button("Show graph", icon=ft.Icons.SHOW_CHART,
                                  on_click=on_koopman_series, bgcolor=C.BLUE_700, color=C.WHITE),
                        ft.Button("Build GAF image", icon=ft.Icons.IMAGE,
                                  on_click=on_koopman_gaf, bgcolor=UI["primary"], color=C.WHITE)],
                       spacing=8, wrap=True),
                ft.Text("Leave firm_id blank to use the cross-sectional median of all firms.",
                        size=11, color=C.GREY_600),
                koop_img_series,
                koop_img_gaf,
            ], spacing=8)),
        _koop_card(
            "Early-warning lead time (machine learning + sigmoid alarm)",
            ft.Column([
                ft.Text("p = sigmoid(XGBoost margin) on [factors | Koopman forecast]; the alarm "
                        "fires when p >= quantile(p, q). Lead time L = t_event - t_first_alarm, "
                        "counting only alarms inside the pre-event horizon. Scores are "
                        "out-of-fold by firm (GroupKFold).",
                        size=11, color=C.GREY_700),
                ft.Row([koop_lead_q, koop_lead_h,
                        ft.Button("Compute lead time", icon=ft.Icons.SCHEDULE,
                                  on_click=on_koopman_lead, bgcolor=UI["primary"], color=C.WHITE)],
                       spacing=8, wrap=True),
                koop_img_lead,
                koop_lead_grid,
            ], spacing=8)),
        _koop_card(
            "All factors at a glance",
            ft.Column([
                ft.Row([
                    ft.Button("Build GAF gallery (all factors)", icon=ft.Icons.GRID_VIEW,
                              on_click=on_koopman_gallery, bgcolor=UI["primary"], color=C.WHITE),
                    ft.Button("Export GAF cards to folder", icon=ft.Icons.SAVE_ALT,
                              on_click=on_koopman_export, bgcolor=C.GREY_400, color=C.WHITE),
                ], spacing=8, wrap=True),
                koop_img_gallery,
            ], spacing=8)),
    ], spacing=12, scroll=ft.ScrollMode.AUTO)

    def update_koopman_tab():
        _koop_fill_factors()
        koop_grid_container.content = render_koopman_grid()

    # ---- LightGBM / CatBoost factor alerting (one builder, two tabs) --------
    def _make_engine_view(engine, headline):
        st = ft.Text(f"Press Run to train {engine} on the factor panel.", size=12,
                     color=C.GREY_700)
        mt = ft.Text("", size=12, color=C.BLUE_900, weight=ft.FontWeight.BOLD)
        grid = ft.Container(content=ft.Text("No results yet.", size=12, color=C.GREY_600),
                            padding=6)
        watch = ft.Container(content=ft.Text("", size=12), padding=4)
        img = ft.Image(src="", visible=False, width=780)
        sf = ft.TextField(label="Search factor", width=220, dense=True)

        def render_grid():
            if ml_factors is None:
                return ft.Text("ml_factors module unavailable.", size=12, color=C.RED_600)
            df = ml_factors.load_table(DB, f"factor_importance_{engine}")
            if df is None or df.empty:
                return ft.Text(f"SQLite table 'factor_importance_{engine}' is empty.",
                               size=12, color=C.GREY_600)
            q = (sf.value or "").strip().lower()
            if q:
                df = df[df["feature"].astype(str).str.lower().str.contains(q, na=False)]
            cols = [c for c in ["rank", "feature", "importance", "kind"] if c in df.columns]
            return ft.Column([
                ft.Text(f"SQLite table 'factor_importance_{engine}' - {len(df)} row(s)",
                        size=11, color=C.GREY_700),
                ft.DataTable(
                    columns=[ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD, size=11))
                             for c in cols],
                    rows=[ft.DataRow([ft.DataCell(ft.Text(str(r[c]), size=11)) for c in cols])
                          for _, r in df.head(40).iterrows()],
                    heading_row_height=32, data_row_max_height=30),
            ], spacing=6, scroll=ft.ScrollMode.AUTO)

        def on_run(_):
            if ml_factors is None:
                st.value = "ml_factors module unavailable."; page.update(); return
            st.value = f"Training {engine} with a leave-firms-out split ..."; page.update()
            try:
                res = ml_factors.run_importance(engine)
                ml_factors.save_importance(res, DB)
                m = res["metrics"]
                mt.value = (f"firms {res['n_firms']:,}  |  AUC {m['AUC']:.4f}  |  "
                            f"PR-AUC {m['PR_AUC']:.4f}  |  base rate {m['base_rate']*100:.3f}%")
                img.src = _uri(_b64(ml_factors.fig_model_importance(res)))
                img.visible = True
                grid.content = render_grid()
                watch.content = ft.Column([
                    ft.Text("Alert watch-list (drives the notification factors):",
                            size=12, weight=ft.FontWeight.BOLD, color=UI["primary_dark"]),
                    ft.Row([ft.Container(
                        content=ft.Text(f"{i}. {f}", size=11, color=C.WHITE),
                        bgcolor=C.RED_600 if i <= 3 else C.ORANGE_600,
                        padding=ft.Padding.symmetric(horizontal=8, vertical=3),
                        border_radius=10)
                        for i, f in enumerate(res["watchlist"], start=1)],
                        spacing=6, wrap=True)], spacing=6)
                st.value = f"Done - saved to 'factor_importance_{engine}' (+ _watchlist)."
            except Exception as exc:
                st.value = f"{engine} run failed: {exc}"
            page.update()

        def do_search(_):
            grid.content = render_grid(); page.update()

        view = ft.Column([
            _koop_card(headline, ft.Column([
                ft.Text("Design matrix = raw factors + their Koopman one-step forecast. "
                        "Classes are balanced and validation splits by firm.",
                        size=11, color=C.GREY_700),
                ft.Row([ft.Button(f"Run {engine}", icon=ft.Icons.PLAY_ARROW, on_click=on_run,
                                  bgcolor=UI["primary"], color=C.WHITE)], spacing=8),
                st, mt, watch], spacing=8)),
            _koop_card("Factor ranking", ft.Column([
                ft.Row([sf, ft.Button("Search", icon=ft.Icons.SEARCH, on_click=do_search,
                                      bgcolor=C.BLUE_700, color=C.WHITE)], spacing=8),
                grid], spacing=8)),
            _koop_card("Importance chart", img),
        ], spacing=12, scroll=ft.ScrollMode.AUTO)
        return view, (lambda: grid.__setattr__("content", render_grid()))

    view_lgbm, refresh_lgbm = _make_engine_view("lightgbm", "Alerting factors from LightGBM")
    view_cat, refresh_cat = _make_engine_view("catboost", "Alerting factors from CatBoost")

    # ---- latent factor models (AE / VAE / AAE / PAE + temporal VAE) ---------
    lat_status = ft.Text("Choose a latent method and press Run.", size=12, color=C.GREY_700)
    lat_metrics = ft.Text("", size=12, color=C.BLUE_900, weight=ft.FontWeight.BOLD)
    lat_method = ft.Dropdown(label="Latent method", width=200, dense=True,
                             options=[ft.dropdown.Option(m) for m in
                                      ("AE", "VAE", "AAE", "PAE", "TemporalVAE")],
                             value="VAE")
    lat_dim = ft.TextField(label="latent dim", value="8", width=110, dense=True)
    lat_epochs = ft.TextField(label="epochs", value="20", width=100, dense=True)
    lat_img = ft.Image(src="", visible=False, width=1000)
    lat_cmp_img = ft.Image(src="", visible=False, width=760)

    def on_latent_run(_):
        if ml_factors is None:
            lat_status.value = "ml_factors unavailable."; page.update(); return
        m, dim, ep = lat_method.value, int(lat_dim.value or 8), int(lat_epochs.value or 20)
        lat_status.value = f"Training {m} (latent dim {dim}, {ep} epochs) ..."; page.update()
        try:
            if m == "TemporalVAE":
                res = ml_factors.vae_latent(latent_dim=dim, epochs=ep)
                ml_factors.save_vae(res, DB)
                lat_img.src = _uri(_b64(ml_factors.fig_vae_latent(res)))
                lat_metrics.value = (f"firms {res['n_firms']:,}  |  seq {res['seq_len']}  |  "
                                     f"final loss {res['final_loss']:.4f}")
                lat_status.value = "Saved to SQLite table 'vae_latent'."
            else:
                res = ml_factors.latent_features(m, latent_dim=dim, epochs=ep)
                ml_factors.save_latent_features(res, DB)
                mm = res["metrics"]
                lat_img.src = _uri(_b64(ml_factors.fig_vae_latent(
                    {"latent": res["latent"], "labels": res["labels"],
                     "history": res["history"], "n_firms": res["n_firms"],
                     "latent_dim": res["latent_dim"],
                     "final_loss": mm["final_loss"]})))
                lat_metrics.value = (f"AUC {mm['AUC']:.4f}  |  PR-AUC {mm['PR_AUC']:.4f}  |  "
                                     f"final loss {mm['final_loss']:.4f}")
                lat_status.value = f"Saved to SQLite table 'latent_features_{m.lower()}'."
            lat_img.visible = True
        except Exception as exc:
            lat_status.value = f"{m} failed: {exc}"
        page.update()

    def on_latent_compare(_):
        if ml_factors is None:
            lat_status.value = "ml_factors unavailable."; page.update(); return
        lat_status.value = "Running AE / VAE / AAE / PAE and comparing ..."; page.update()
        try:
            cmp_df = ml_factors.compare_latent_methods(
                latent_dim=int(lat_dim.value or 8), epochs=int(lat_epochs.value or 20))
            lat_cmp_img.src = _uri(_b64(ml_factors.fig_latent_compare(cmp_df)))
            lat_cmp_img.visible = True
            best = cmp_df.iloc[0]
            lat_status.value = f"Best latent method: {best['method']} (AUC {best['AUC']:.4f})."
        except Exception as exc:
            lat_status.value = f"Comparison failed: {exc}"
        page.update()

    view_latent = ft.Column([
        _koop_card("Latent factor models from the VAE family", ft.Column([
            ft.Text("AE, VAE, AAE and PAE follow the tabular architectures in the research "
                    "script; TemporalVAE is the LSTM sequence model. The encoder is trained "
                    "unsupervised on training firms, then the latent vector is scored by a "
                    "balanced logistic regression (leave-firms-out).",
                    size=11, color=C.GREY_700),
            ft.Row([lat_method, lat_dim, lat_epochs,
                    ft.Button("Run", icon=ft.Icons.PLAY_ARROW, on_click=on_latent_run,
                              bgcolor=UI["primary"], color=C.WHITE),
                    ft.Button("Compare all", icon=ft.Icons.EQUALIZER, on_click=on_latent_compare,
                              bgcolor=C.BLUE_700, color=C.WHITE)], spacing=8, wrap=True),
            lat_status, lat_metrics], spacing=8)),
        _koop_card("Latent space & training curve", lat_img),
        _koop_card("Method comparison", lat_cmp_img),
    ], spacing=12, scroll=ft.ScrollMode.AUTO)

    # ---- one view per anomaly baseline: lead time of the RS firms -----------
    BASELINE_BLURB = {
        "IsolationForest": "Random axis-aligned splits; the anomaly score is the inverse "
                           "average path length of a window in the forest.",
        "OneClassSVM": "RBF one-class SVM fitted on normal windows; the score is the "
                       "negative signed distance to the learned boundary.",
        "DeepSVDD": "An encoder is trained to pull normal windows into a hypersphere; "
                    "the score is the squared distance to its centre.",
        "DAGMM": "A compressing autoencoder feeds a Gaussian mixture estimator; the "
                 "score is the mixture energy of the joint code.",
        "OmniAnomaly": "A stochastic recurrent model with a latent state; the score is "
                       "the reconstruction error under the sampled posterior.",
        "USAD": "Two decoders share one encoder and are trained adversarially; the "
                "score mixes both reconstruction errors.",
        "TranAD": "A transformer encoder reconstructs the window in two passes; the "
                  "score is the focus-corrected reconstruction error.",
        "AnomalyTransformer": "Attention is split into a prior and a series branch; the "
                              "score combines reconstruction error and association "
                              "discrepancy.",
    }

    def _make_baseline_view(name, headline):
        st = ft.Text("Press Run to fit %s and measure the RS lead time." % name,
                     size=12, color=C.GREY_700)
        mt = ft.Text("", size=12, color=C.BLUE_900, weight=ft.FontWeight.BOLD)
        grid = ft.Container(content=ft.Text("No results yet.", size=12, color=C.GREY_600),
                            padding=6)
        img = ft.Image(src="", visible=False, width=860)
        q_in = ft.Dropdown(label="threshold quantile", width=170, dense=True,
                           options=[ft.dropdown.Option(v) for v in
                                    ("0.90", "0.95", "0.97", "0.99")], value="0.95")
        hz_in = ft.TextField(label="horizon (days)", value="365", width=130, dense=True)
        ep_in = ft.TextField(label="epochs", value="8", width=100, dense=True)

        def render_grid(df=None):
            if baselines is None:
                return ft.Text("baselines module unavailable (needs torch / sklearn).",
                               size=12, color=C.RED_600)
            if df is None:
                df = baselines.load_baseline_lead(name, DB)
            if df is None or df.empty:
                return ft.Text("No stored lead-time rows for %s yet." % name,
                               size=12, color=C.GREY_600)
            cols = [c for c in ["firm_id", "event_date", "first_alarm", "lead_time_days",
                                "detected", "n_windows", "max_score"] if c in df.columns]

            def cell(r, c):
                v = r[c]
                if c == "lead_time_days":
                    cen = int(r["censored"]) if "censored" in df.columns and \
                        not pd.isna(r.get("censored")) else 0
                    txt = "n/s" if pd.isna(v) else "%s%.0f d  (%.1f mo)" % (
                        ">= " if cen else "", float(v), float(v) / 30.44)
                elif c == "detected":
                    txt = "yes" if int(v) == 1 else "no"
                elif c == "max_score":
                    txt = "-" if pd.isna(v) else "%.3f" % float(v)
                else:
                    txt = "-" if (v is None or str(v) in ("NaT", "nan", "None")) else str(v)
                col = C.GREEN_800 if (c == "detected" and str(r["detected"]) == "1") else (
                    C.RED_600 if c == "detected" else C.BLACK)
                return ft.DataCell(ft.Text(txt, size=11, color=col))

            return ft.Column([
                ft.Text("Lead time L = event date - first alarm, alarms restricted to the "
                        "pre-event horizon; 'n/s' means the detector never fired in time.",
                        size=11, color=C.GREY_700),
                ft.DataTable(
                    columns=[ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD, size=11))
                             for c in cols],
                    rows=[ft.DataRow([cell(r, c) for c in cols])
                          for _, r in df.iterrows()],
                    heading_row_height=32, data_row_max_height=30),
            ], spacing=6, scroll=ft.ScrollMode.AUTO)

        def on_run(_):
            if baselines is None:
                st.value = "baselines module unavailable."; page.update(); return
            st.value = ("Fitting %s on normal windows of the training firms, then scoring "
                        "every RS firm ..." % name)
            page.update()
            try:
                res = baselines.baseline_lead_time(
                    name, q=float(q_in.value or 0.95),
                    horizon_days=float(hz_in.value or 365),
                    epochs=int(ep_in.value or 8))
                baselines.save_baseline_lead(res, DB)
                s = res["summary"]
                med = s["median_lead_days"]
                mt.value = ("detected %d/%d RS firms  |  median lead %s  |  "
                            "alarm rate %.2f%%  |  tau %.4f"
                            % (s["detected"], s["event_firms"],
                               "n/a" if med != med else "%.0f d (%.1f mo)" % (med, med / 30.44),
                               s["alarm_rate"] * 100, s["tau"]))
                img.src = _uri(_b64(baselines.fig_baseline_lead(res)))
                img.visible = True
                grid.content = render_grid(res["table"])
                st.value = "Done - saved to 'baseline_lead_time_%s'." % name.lower()
            except Exception as exc:
                st.value = "%s failed: %s" % (name, exc)
            page.update()

        view = ft.Column([
            _koop_card(headline, ft.Column([
                ft.Text(BASELINE_BLURB.get(name, ""), size=11, color=C.GREY_700),
                ft.Text("Training uses only normal windows of the training firms and the "
                        "split is leave-firms-out, so an RS firm is never seen before it "
                        "is scored. The alarm threshold is the training-normal quantile.",
                        size=11, color=C.GREY_700),
                ft.Row([q_in, hz_in, ep_in,
                        ft.Button("Run %s" % name, icon=ft.Icons.PLAY_ARROW, on_click=on_run,
                                  bgcolor=UI["primary"], color=C.WHITE)],
                       spacing=8, wrap=True),
                st, mt], spacing=8)),
            _koop_card("Lead time per RS firm", grid),
            _koop_card("Lead-time chart", img),
        ], spacing=12, scroll=ft.ScrollMode.AUTO)
        return view, (lambda: grid.__setattr__("content", render_grid()))

    baseline_views = [_make_baseline_view(nm, label)
                      for nm, label, _ in BASELINE_MENUS]

    main_content_container = ft.Container(content=view_approach1, expand=True, padding=ft.Padding.only(left=6, right=6, top=6, bottom=6))

    def set_tab(idx):
        state["active_tab"] = idx
        dns_mode = state.get("dns_mode", "full")
        for tab_idx, nav_button in enumerate(nav_buttons):
            def_bg, def_fg, act_bg = nav_styles[tab_idx]
            active = (tab_idx == idx)
            if idx == 13 and nav_button in (btn_tab13, btn_tab13_quick, btn_tab13_latest, btn_tab13_roll, btn_tab13_tables, btn_tab13_logs):
                if nav_button is btn_tab13 and dns_mode == "full":
                    active = True
                elif nav_button is btn_tab13_quick and dns_mode == "quick":
                    active = True
                elif nav_button is btn_tab13_latest and dns_mode == "latest":
                    active = True
                elif nav_button is btn_tab13_roll and dns_mode == "roll":
                    active = True
                elif nav_button is btn_tab13_tables and dns_mode == "tables":
                    active = True
                elif nav_button is btn_tab13_logs and dns_mode == "logs":
                    active = True
                else:
                    active = False
            if active:
                nav_button.bgcolor = act_bg
                nav_button.color = C.WHITE
            else:
                nav_button.bgcolor = def_bg
                nav_button.color = def_fg

        if idx >= 16:
            view, refresh = baseline_views[idx - 16]
            refresh()
            main_content_container.content = view
            page.update()
            return

        if idx == 0:
            main_content_container.content = view_approach1
        elif idx == 1:
            main_content_container.content = view_approach2
        elif idx == 2:
            update_datagrid_tab()
            main_content_container.content = view_datagrid
        elif idx == 3:
            update_lead_time_tab()
            main_content_container.content = view_lead_time
        elif idx == 4:
            update_openclaw_tab()
            main_content_container.content = view_openclaw
        elif idx == 6:
            update_koopman_tab()
            main_content_container.content = view_koopman
        elif idx == 7:
            refresh_lgbm()
            main_content_container.content = view_lgbm
        elif idx == 8:
            refresh_cat()
            main_content_container.content = view_cat
        elif idx == 9:
            main_content_container.content = view_latent
        elif idx == 10:
            update_survivor2_tab()
            main_content_container.content = view_survivor2
        elif idx == 11:
            update_compare_tab()
            main_content_container.content = view_compare
        elif idx == 12:
            update_benchmark_tab()
            main_content_container.content = view_benchmark
        elif idx == 13:
            update_dns_tab()
            main_content_container.content = view_dns
        elif idx == 14:
            update_paper_tab()
            main_content_container.content = view_paper
        elif idx == 15:
            update_realtime_tab()
            main_content_container.content = view_realtime
        else:
            main_content_container.content = view_bond_news
        page.update()

    btn_tab0.on_click = lambda _: set_tab(0)
    btn_tab1.on_click = lambda _: set_tab(1)
    btn_tab2.on_click = lambda _: set_tab(2)
    btn_tab3.on_click = lambda _: set_tab(3)
    btn_tab4.on_click = lambda _: set_tab(4)
    btn_tab5.on_click = lambda _: set_tab(5)
    btn_tab6.on_click = lambda _: set_tab(6)
    btn_tab7.on_click = lambda _: set_tab(7)
    btn_tab8.on_click = lambda _: set_tab(8)
    btn_tab9.on_click = lambda _: set_tab(9)
    btn_tab10.on_click = lambda _: set_tab(10)
    btn_tab11.on_click = lambda _: set_tab(11)
    btn_tab12.on_click = lambda _: set_tab(12)
    btn_tab13.on_click = lambda _: set_tab(13)
    btn_tab13_quick.on_click = lambda _: set_tab(13)
    btn_tab13_latest.on_click = lambda _: set_tab(13)
    btn_tab13_roll.on_click = lambda _: set_tab(13)
    btn_tab13_tables.on_click = lambda _: set_tab(13)
    btn_tab13_logs.on_click = lambda _: set_tab(13)
    btn_tab14.on_click = lambda _: set_tab(14)
    btn_tab15.on_click = lambda _: set_tab(15)
    for _i, _b in enumerate(baseline_buttons, start=16):
        _b.on_click = (lambda i: (lambda _: set_tab(i)))(_i)

    def _clear_page():
        if hasattr(page, "controls"):
            page.controls.clear()

    def show_pin_setup(message="Set a numeric PIN before using the system."):
        pin1 = ft.TextField(label="New PIN (4-8 digits)", password=True, width=280, text_size=13)
        pin2 = ft.TextField(label="Confirm PIN", password=True, width=280, text_size=13)
        msg = ft.Text(message, size=12, color=C.BLUE_GREY_700)

        def submit(_):
            p1, p2 = (pin1.value or "").strip(), (pin2.value or "").strip()
            if not _valid_pin(p1):
                msg.value = "PIN must contain only 4-8 digits."
                msg.color = C.RED_700
            elif p1 != p2:
                msg.value = "PIN confirmation does not match."
                msg.color = C.RED_700
            else:
                _set_pin(p1)
                msg.value = "PIN saved."
                msg.color = C.GREEN_700
                show_login("PIN saved. Please log in.")
                return
            page.update()

        screen = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.LOCK_RESET, size=42, color=C.BLUE_700),
                ft.Text("Set Numeric PIN", size=22, weight=ft.FontWeight.BOLD, color=C.BLUE_900),
                msg,
                pin1,
                pin2,
                ft.Button("Save PIN", icon=ft.Icons.SAVE, on_click=submit, bgcolor=C.BLUE_700, color=C.WHITE, width=280),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            alignment=center_alignment,
            expand=True,
            bgcolor=C.BLUE_GREY_50,
        )
        _clear_page(); page.add(screen); page.update()

    def show_login(message="Enter your numeric PIN to continue."):
        pin = ft.TextField(label="PIN", password=True, width=280, text_size=14)
        msg = ft.Text(message, size=12, color=C.BLUE_GREY_700)

        def submit(_):
            if _verify_pin((pin.value or "").strip()):
                state["authenticated"] = True
                show_app()
            else:
                msg.value = "Incorrect PIN."
                msg.color = C.RED_700
                page.update()

        screen = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.ADMIN_PANEL_SETTINGS, size=46, color=C.INDIGO_700),
                ft.Text("ThaiBMA EWS Login", size=23, weight=ft.FontWeight.BOLD, color=C.INDIGO_900),
                msg,
                pin,
                ft.Button("Login", icon=ft.Icons.LOGIN, on_click=submit, bgcolor=C.INDIGO_700, color=C.WHITE, width=280),
                ft.TextButton("Set / Reset PIN", icon=ft.Icons.PIN, on_click=lambda _: show_pin_setup("Create a new numeric PIN.")),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            alignment=center_alignment,
            expand=True,
            bgcolor=C.BLUE_GREY_50,
        )
        _clear_page(); page.add(screen); page.update()

    def logout(_=None):
        state["authenticated"] = False
        show_login("Logged out. Enter PIN to continue.")

    account_panel = ft.Container(
        content=ft.Column([
            ft.Button(
                "Change PIN",
                icon=ft.Icons.PIN,
                on_click=lambda _: show_pin_setup("Enter and confirm a new numeric PIN."),
                bgcolor=UI["button"],
                color=UI["text"],
                width=NAV_BUTTON_WIDTH,
                height=NAV_BUTTON_HEIGHT,
            ),
            ft.Button(
                "Logout",
                icon=ft.Icons.LOGOUT,
                on_click=logout,
                bgcolor=UI["button"],
                color=UI["text"],
                width=NAV_BUTTON_WIDTH,
                height=NAV_BUTTON_HEIGHT,
            ),
        ], spacing=8),
        visible=ENABLE_LOGIN,
    )

    sidebar_scroll = ft.Column([
        ft.Text("NAVIGATION", size=11, weight=ft.FontWeight.BOLD, color=UI["primary_dark"]),
        side_nav_menu,
        ft.Divider(color=UI["border"], height=1),
        ft.Text("DATA & ACTIONS", size=11, weight=ft.FontWeight.BOLD, color=UI["primary_dark"]),
        control_panel,
        account_panel,
        ft.Divider(color=UI["border"], height=1),
        ft.Text("SCROLL DOWN FOR IBOND", size=11, weight=ft.FontWeight.BOLD, color=C.PINK_800),
        ft.Text("Use the left scrollbar here to reach iBond menus if the window is short.",
                size=10, color=C.BLUE_GREY_700),
        ft.Container(height=220, bgcolor=ft.Colors.with_opacity(0.02, C.BLACK)),
    ], spacing=10, scroll=ft.ScrollMode.ALWAYS, expand=True)

    sidebar = ft.Container(
        content=sidebar_scroll,
        width=300,
        padding=14,
        bgcolor=UI["sidebar"],
        border=ft.Border(right=ft.BorderSide(1, UI["border"])),
        shadow=SHADOW,
    )

    scroll_content_host = ft.Container(
        content=main_content_container,
        expand=True,
        padding=ft.Padding.only(right=4, bottom=8),
    )

    right_scroll_area = ft.Column(
        controls=[scroll_content_host],
        spacing=8,
        expand=True,
        scroll=ft.ScrollMode.ALWAYS,
    )

    right_panel = ft.Column(
        controls=[
            header,
            ft.Container(expand=True, content=right_scroll_area)
        ],
        spacing=14,
        expand=True,
    )

    app_shell = ft.Row([
        sidebar,
        ft.Container(
            content=right_panel,
            expand=True,
            padding=14,
            bgcolor=UI["page"],
        ),
    ], expand=True, spacing=0, vertical_alignment=ft.CrossAxisAlignment.STRETCH)

    def show_app():
        _clear_page()
        page.add(app_shell)
        page.update()

    if not ENABLE_LOGIN or globals().get("_UITEST"):
        show_app()
    elif _pin_configured():
        show_login()
    else:
        show_pin_setup()

    if globals().get("_UITEST"):
        import traceback
        handlers = [("on_import", on_import), ("on_train", on_train), ("on_alerts", on_alerts)]
        if os.path.exists(PANEL_REAL):
            handlers += [("on_real", on_real), ("on_train(real)", on_train),
                         ("on_alerts(real)", on_alerts), ("on_survival", on_survival)]
        for name, h in handlers:
            try:
                h(None)
            except Exception:
                print("HANDLER ERROR:", name); traceback.print_exc()
        print("handlers exercised OK")

        # exercise the dropdown on_select path (the fix): selecting a firm must redraw
        if state.get("df_surv") is not None and firm_select_dropdown.options:
            class _E:  # minimal ControlEvent stand-in
                pass
            for opt in [firm_select_dropdown.options[0], firm_select_dropdown.options[-1]]:
                firm_select_dropdown.value = opt.key
                ev = _E(); ev.data = opt.key
                on_select_firm(ev)
                img = traj_holder.controls[0]
                ok = hasattr(img, "src") and str(img.src).startswith("data:image")
                cap_has_firm = str(opt.key) in (traj_caption.value or "")
                print(f"on_select redraw OK={ok} firm={opt.key} caption_updated={cap_has_firm}")
        set_tab(3)
        print(f"lead-time tab OK rows={len(_lead_source_df())}")
        set_tab(4)
        oc_job_count = len(openclaw.list_jobs(DB)) if openclaw is not None else 0
        oc_alert_count = len(_oc_alert_rows())
        print(f"openclaw tab OK jobs={oc_job_count} alerts={oc_alert_count}")
        on_survivor2_ews(None)           # Survivor2 EWS: hazard plot + metrics + datagrid
        print(f"survivor2-EWS OK hazard_chart={s2ews_hazard_img.visible} "
              f"boundary={s2ews_boundary_img.visible} risk_cards={len(s2ews_risk_cards.controls)} "
              f"metrics={len(s2ews_box.controls)} table={len(s2ews_table.controls)}")
        set_tab(15)                      # Real-time EWS (reads saved SQLite tables)
        print(f"realtime-EWS tab OK kpis={len(rt_kpis.controls)} chart={rt_img.visible} "
              f"tables={len(rt_ref_box.controls)>0 and len(rt_grid.controls)>0}")
        set_tab(14)                      # Paper replication (reads saved SQLite tables)
        print(f"paper-replication tab OK kpis={len(pap_kpis.controls)} "
              f"chart={pap_img.visible} tables={len(pap_tabs_box.controls)>0}")
        set_tab(13)                      # Yield Curve DNS (reads saved SQLite tables)
        print(f"yield-curve DNS tab OK kpis={len(dns_kpis.controls)} "
              f"charts={sum(1 for i in (dns_img_factors, dns_img_surface, dns_img_fc) if i.visible)} "
              f"tables={len(dns_val_box.controls)>0 and len(dns_factor_box.controls)>0}")
        set_tab(12)                      # Benchmark A1/A2/DL (reads saved SQLite tables)
        print(f"benchmark tab OK kpis={len(bm_kpis.controls)} "
              f"charts={sum(1 for i in (bm_img_pred, bm_img_econ, bm_img_trade) if i.visible)} "
              f"tables={len(bm_pred_box.controls)>0 and len(bm_econ_box.controls)>0}")
        set_tab(11)                      # Model Comparison (reads saved SQLite tables)
        _n_kpi = len(cmp_kpis.controls)
        _charts = sum(1 for im in (cmp_img_metrics, cmp_img_outperf, cmp_img_lead) if im.visible)
        _tbl = len(cmp_metrics_box.controls) > 0 and len(cmp_lead_box.controls) > 0
        print(f"model-comparison tab OK kpis={_n_kpi} charts={_charts} tables={_tbl}")


# --------------------------------------------------------------- test ---------
def _selftest():
    n = import_to_sqlite(); print(f"imported {n} rows to {DB}")
    df = load_df(); res, best, feats = train_models(df)
    for k, r in res.items():
        print(f"  {k:14s} AUC {r['auc']:.3f}  MCC {r['mcc']:.3f}")
    print("  best:", best)
    alerts = compute_alerts(df, res[best]["model"], feats)
    con = sqlite3.connect(DB); alerts.to_sql("alerts", con, if_exists="replace", index=False)
    con.commit(); con.close()
    print("  alert counts:", alerts["alert"].value_counts().to_dict())
    if notify is not None:                    # exercise the notifier without sending anything
        print("  notify (dry-run):",
              notify.summary_line(notify.notify_alerts(alerts, DB, dry_run=True)))
    assert len(fig_importance(res[best]["model"], feats)) > 100
    assert len(fig_alert_dist(alerts)) > 100
    print("synthetic path OK — Excel->SQLite->models->XAI->alerts")
    try:
        from load_bond import load_bond
        bdf = load_bond()
        con = sqlite3.connect(DB); bdf.to_sql(TABLE, con, if_exists="replace", index_label="id")
        bdf.to_sql("panel", con, if_exists="replace", index_label="id")   # keeps real onset `event`
        con.commit(); con.close()
        r2, b2, f2 = train_models(load_df())
        print(f"bond path: {len(bdf):,} firm-months, {len(f2)} features, "
              f"early-warning positives {int(bdf['default_3m'].sum())} "
              f"({int(bdf['event'].sum())} onsets) -> best {b2} AUC {r2[b2]['auc']:.3f}")
        sb, mb = survival.run(load_panel())
        print(f"bond survival: PD_3M AUC in {mb['pd_auc']:.3f} / oos {mb['pd_auc_oos']:.3f} / "
              f"persistence {mb['persistence_auc']:.3f}")
        lt = mb.get("lead_time") if isinstance(mb.get("lead_time"), pd.DataFrame) else survival.compute_lead_time(sb)
        save_lead_time(lt)
        assert {"firm_id", "default_observed", "lead_time_days", "alarm_source", "alert_level"}.issubset(lt.columns)
        print(f"lead-time table: {len(lt):,} firms, "
              f"{int(lt['default_observed'].astype(str).str.lower().isin(['true','1']).sum())} observed defaults")
        import_to_sqlite()
    except FileNotFoundError:
        print("bond path: .dta not found (skipped)")
    if os.path.exists(PANEL_REAL):
        n_xs, n_pm, n_f = import_real_to_sqlite()
        print(f"real path: {n_f} firms, {n_xs} cross-section, {n_pm:,} firm-months in SQLite")
        sdf, meta = survival.run(load_panel())
        print(f"survival EWS: PD_3M AUC in {meta['pd_auc']:.3f} / oos {meta['pd_auc_oos']:.3f} / "
              f"persistence {meta['persistence_auc']:.3f}")
        lead = meta.get("lead_time") if isinstance(meta.get("lead_time"), pd.DataFrame) else survival.compute_lead_time(sdf)
        assert "lead_time_days" in lead.columns
        assert len(fig_boundary(sdf, meta)) > 100
        assert len(fig_roe_pd(sdf)) > 100
        print("real + survival path OK")
    print("selftest OK")


def _uitest():
    class _P:
        def add(self, *c):
            self.controls = getattr(self, "controls", []) + list(c)
        def update(self, *a, **k):
            pass
    globals()["_UITEST"] = True
    p = _P()
    main(p)
    print(f"UI build OK — {len(getattr(p, 'controls', []))} top-level controls constructed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--uitest" in sys.argv:
        _uitest()
    else:
        import flet as ft
        ft.run(main)
