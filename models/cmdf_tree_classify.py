# -*- coding: utf-8 -*-
"""
cmdf_tree_classify.py -- Random Forest / XGBoost / CatBoost / LightGBM against the
logistic Approach-1 baseline, on the 33-feature iBond panel.

This is a DIFFERENT problem from cmdf_tree_models.py. That module regresses
ln(PD)_12m, a Merton-implied quantity, and reports R2 / RMSE. This one predicts the
REAL event -- does the issuer miss a bond payment within the next three months --
and reports AUC / F1 / precision / recall, which is what the Approach-1 logistic
engine is scored on.

DATA
    ibond_33features_panel  (issuer x month, 33 determinants)
    ibond_default_payment   (ThaiBMA payment-default register)
    Target: first default date falls within the next 3 months.

VALIDATION
    Leave-one-issuer-out. Every recorded default sits in the last two years, so a
    time split leaves the training half with no events at all; holding out one
    defaulted issuer at a time is the only estimate here that is genuinely
    out-of-sample. In-sample AUC is reported beside it purely to show the gap.

    Because so few issuers ever default, F1 / precision / recall are computed at a
    MATCHED ALARM BUDGET: each model flags the same number of issuer-months (the
    top k by predicted risk), so the comparison is not decided by where a model
    happens to put its 0.5 cut-off.

RUN
    python cmdf_tree_classify.py
    python cmdf_tree_classify.py --budget 0.02
    python cmdf_tree_classify.py --no-save
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)


BOND_33 = [
    "amihud_monthly", "amihud_monthly_100", "adj_illiq_kz", "scaled_amihud",
    "ln_amihud", "percent_zero_days", "zero_days", "n_days",
    "ROA", "ROE", "DE", "CurrentRatio", "QuickRatio", "CashRatio",
    "EBITtoTA", "REtoTA", "WorkingCapitaltoTA", "TDTA", "LTDtoTA", "STDtoTA",
    "cf_Interestcoverageratio", "acc_DebtServiceCoverageRatio",
    "lnTotalAssets", "lnAge",
    "Policyrate", "GDPgrowth", "UnemploymentratemodeledILOe",
    "ESGScore", "GovernancePillarScore", "EnvironmentalPillarScore",
    "SocialPillarScore", "IndependentBoardMembers", "AverageBoardTenure",
]

BASELINE = "Logistic (Approach 1)"
MODEL_ORDER = [BASELINE, "Random Forest", "XGBoost", "CatBoost", "LightGBM"]
MC = {BASELINE: "#6b7280", "Random Forest": "#2e7d4f", "XGBoost": "#1f3a5f",
      "CatBoost": "#a8501a", "LightGBM": "#e0a52e"}

ALARM_BUDGET = 0.02          # flag the riskiest 2% of issuer-months
T_METRICS = "cmdf_classify_metrics"
T_ROC = "cmdf_classify_roc"

plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})


# ================================================================== data =====
def load_panel(db=DB, verbose=True):
    con = sqlite3.connect(db)
    panel = pd.read_sql("SELECT * FROM ibond_33features_panel", con)
    dflt = pd.read_sql("SELECT * FROM ibond_default_payment", con)
    con.close()

    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    y = pd.Series(0, index=panel.index, dtype=int)
    if not dflt.empty:
        d = dflt.copy()
        d["issuer_code"] = d["symbol"].astype(str).str.extract(r"^([A-Z]+)")[0]
        d["payment_date"] = pd.to_datetime(d["payment_date"], errors="coerce")
        first = (d.dropna(subset=["payment_date"])
                 .groupby("issuer_code")["payment_date"].min())
        ev = panel["issuer_code"].map(first)
        gap = ((ev.dt.year - panel["month_dt"].dt.year) * 12
               + (ev.dt.month - panel["month_dt"].dt.month))
        y = ((gap >= 0) & (gap <= 3)).fillna(False).astype(int)
        panel["event_date"] = ev
    panel["y"] = y

    cols = [c for c in BOND_33 if c in panel.columns]
    X = panel[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if verbose:
        print(f"  panel {len(panel):,} issuer-months | {panel['issuer_code'].nunique()} "
              f"issuers | {len(cols)}/{len(BOND_33)} features")
        print(f"  positives {int(y.sum())} ({y.mean()*100:.2f}%) from "
              f"{panel.loc[y == 1, 'issuer_code'].nunique()} issuers")
    return panel, X, y, cols


# ================================================================ models =====
def classifiers(seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    import xgboost as xgb
    m = {
        BASELINE: lambda: LogisticRegression(max_iter=3000, C=0.1,
                                             class_weight="balanced"),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, n_jobs=-1,
            class_weight="balanced", random_state=seed),
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=2.0, n_jobs=-1, random_state=seed,
            eval_metric="logloss", verbosity=0),
        # importance_type="gain" matters here beyond reporting: cmdf_feature_select
        # ranks determinants by feature_importances_ to pick the top-k inside each
        # fold. On LightGBM's default "split" that ranking is a branch count, which
        # favours determinants with many distinct values over the ones that actually
        # reduce loss, so the selected subset was not the model's own best subset.
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=2.0, n_jobs=-1, random_state=seed,
            class_weight="balanced", importance_type="gain", verbose=-1),
    }
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda: CatBoostClassifier(
            iterations=300, depth=3, learning_rate=0.05, l2_leaf_reg=3.0,
            auto_class_weights="Balanced", random_seed=seed, verbose=0,
            allow_writing_files=False)
    except ImportError:
        print("  NOTE: catboost not installed — that row will be missing")
    return m


def _budget_metrics(y_true, score, budget=ALARM_BUDGET):
    """Precision / recall / F1 at a matched alarm budget (top-k by score).
    Using a fixed 0.5 cut-off would compare where each model happens to place its
    probabilities rather than how well it ranks."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    n = len(y_true)
    k = max(1, int(round(budget * n)))
    thr_idx = np.argsort(score)[::-1][:k]
    pred = np.zeros(n, dtype=int)
    pred[thr_idx] = 1
    return (float(precision_score(y_true, pred, zero_division=0)),
            float(recall_score(y_true, pred, zero_division=0)),
            float(f1_score(y_true, pred, zero_division=0)),
            int(k))


def evaluate(panel, X, y, budget=ALARM_BUDGET, verbose=True):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    Xv = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    ev_issuers = sorted(panel.loc[y == 1, "issuer_code"].dropna().unique())
    if verbose:
        print(f"\n  leave-one-issuer-out over {len(ev_issuers)} defaulted issuers")

    rows, roc_rows = [], []
    for name, ctor in classifiers().items():
        t0 = time.time()
        sc_all = StandardScaler().fit(Xv)
        m_all = ctor()
        m_all.fit(sc_all.transform(Xv), yv)
        p_in = m_all.predict_proba(sc_all.transform(Xv))[:, 1]
        auc_in = float(roc_auc_score(yv, p_in)) if yv.sum() else np.nan

        oy, op = [], []
        for held in ev_issuers:
            tr = groups != held
            if yv[tr].sum() < 2:
                continue
            try:
                sc = StandardScaler().fit(Xv[tr])
                mi = ctor()
                mi.fit(sc.transform(Xv[tr]), yv[tr])
                oy.append(yv[~tr])
                op.append(mi.predict_proba(sc.transform(Xv[~tr]))[:, 1])
            except Exception:
                continue
        if not oy:
            continue
        yy, pp = np.concatenate(oy), np.concatenate(op)
        auc_oos = (float(roc_auc_score(yy, pp))
                   if 0 < yy.sum() < len(yy) else np.nan)
        ap_oos = (float(average_precision_score(yy, pp))
                  if 0 < yy.sum() < len(yy) else np.nan)
        prec, rec, f1, k = _budget_metrics(yy, pp, budget)
        rows.append(dict(model=name, auc_in=auc_in, auc_oos=auc_oos,
                         avg_precision=ap_oos, precision=prec, recall=rec, f1=f1,
                         n_flagged=k, n_eval=int(len(yy)), n_pos=int(yy.sum()),
                         seconds=round(time.time() - t0, 1)))
        # ROC curve on the pooled out-of-sample predictions
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(yy, pp)
            step = max(1, len(fpr) // 400)
            for a, b in zip(fpr[::step], tpr[::step]):
                roc_rows.append({"model": name, "fpr": float(a), "tpr": float(b)})
        except Exception:
            pass
        if verbose:
            print(f"    {name:22s} AUC(oos)={auc_oos:.3f}  F1={f1:.3f}  "
                  f"recall={rec:.3f}  precision={prec:.3f}  ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    if not res.empty:
        base = res.loc[res["model"] == BASELINE]
        if not base.empty:
            b = base.iloc[0]
            for col, new in (("auc_oos", "auc_vs_base_pct"), ("f1", "f1_vs_base_pct")):
                res[new] = (res[col] - b[col]) / abs(b[col]) * 100 if b[col] else np.nan
            res["outperforms"] = res["auc_oos"] > b["auc_oos"]
        res["_o"] = res["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
        res = res.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return res, pd.DataFrame(roc_rows)


# ================================================================ output =====
def write_tex(res, path=None):
    import cmdf_tree_models as tm
    path = path or out("tab_classify_compare.tex")
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    fpc = lambda v: "--" if pd.isna(v) else (f"{v:+.1f}\\%" if abs(v) > 1e-9 else "--")
    d = res.copy()
    d["Model"] = d["model"]
    best = d.loc[d["auc_oos"].idxmax(), "model"] if not d.empty else None
    tm.write_tex_table(
        d, path,
        "Classification performance against the Approach-1 logistic baseline "
        "(33-feature iBond panel, leave-one-issuer-out)",
        "tab:classify",
        cols=["Model", "auc_in", "auc_oos", "f1", "recall", "precision",
              "auc_vs_base_pct"],
        fmt={"auc_in": f3, "auc_oos": f3, "f1": f3, "recall": f3,
             "precision": f3, "auc_vs_base_pct": fpc},
        bold_row=lambda r, b=best: r["model"] == b,
        note=(f"F1, recall and precision are measured at a matched alarm budget of "
              f"{ALARM_BUDGET*100:.0f}\\% of issuer-months, so every model flags the "
              f"same number of cases. The last column is the change in "
              f"out-of-sample AUC relative to the logistic baseline. In-sample AUC "
              f"is shown only to expose the overfitting gap; it is not a "
              f"performance claim."))
    return path


def draw(res, roc, budget=ALARM_BUDGET):
    if res.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

    ax = axes[0]
    d = res.dropna(subset=["auc_oos"])
    xs = np.arange(len(d))
    ax.bar(xs - 0.19, d["auc_in"], width=0.38, color="#cbd5e1",
           label="in-sample", edgecolor="white")
    ax.bar(xs + 0.19, d["auc_oos"], width=0.38,
           color=[MC.get(m, "#888") for m in d["model"]],
           label="out-of-sample", edgecolor="white")
    base = d.loc[d["model"] == BASELINE, "auc_oos"]
    if not base.empty:
        ax.axhline(float(base.iloc[0]), color="#dc2626", lw=1.3, ls="--")
        ax.text(len(d) - 0.5, float(base.iloc[0]) + 0.01, "logistic baseline",
                fontsize=7.5, color="#dc2626", ha="right")
    ax.axhline(0.5, color="#6b7280", lw=1.0, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace(" (Approach 1)", "\n(Approach 1)")
                        for m in d["model"]], fontsize=7.5)
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("ROC AUC: in-sample and out-of-sample",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    w = 0.26
    for i, (col, c, lb) in enumerate((("f1", "#1f3a5f", "F1"),
                                      ("recall", "#2e7d4f", "Recall"),
                                      ("precision", "#a8501a", "Precision"))):
        ax.bar(xs + (i - 1) * w, d[col], width=w, color=c, alpha=0.9, label=lb)
    ax.set_xticks(xs)
    ax.set_xticklabels([m.split(" (")[0] for m in d["model"]], fontsize=7.5,
                       rotation=15)
    ax.set_title(f"Matched alarm budget of {budget*100:.0f}% of issuer-months",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    if not roc.empty:
        for m in MODEL_ORDER:
            g = roc[roc["model"] == m].sort_values("fpr")
            if g.empty:
                continue
            ax.plot(g["fpr"], g["tpr"], lw=1.9, color=MC.get(m, "#888"),
                    ls="--" if m == BASELINE else "-", label=m.split(" (")[0])
        ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.0, ls=":")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("Out-of-sample ROC curves", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
    fig.suptitle("Classification of the payment-default event: tree ensembles and the "
                 "Approach-1 logistic baseline", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out("fig_classify_compare.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    fig_classify_compare.png")


def run(budget=ALARM_BUDGET, save=True, verbose=True):
    print("=" * 78)
    print("33-feature classification: tree ensembles vs Approach-1 logistic")
    print("=" * 78)
    panel, X, y, cols = load_panel(verbose=verbose)
    if y.sum() < 5:
        raise RuntimeError(f"only {int(y.sum())} positive months — run "
                           "download_bond.py --defaults first")
    res, roc = evaluate(panel, X, y, budget, verbose)
    if res.empty:
        raise RuntimeError("no model produced an out-of-sample estimate")
    write_tex(res)
    draw(res, roc, budget)
    res.to_csv(out("classify_comparison.csv"), index=False)
    if save:
        con = sqlite3.connect(DB)
        res.to_sql(T_METRICS, con, if_exists="replace", index=False)
        if not roc.empty:
            roc.to_sql(T_ROC, con, if_exists="replace", index=False)
        con.commit(); con.close()
    return res, roc


def main():
    a = sys.argv
    b = float(a[a.index("--budget") + 1]) if "--budget" in a else ALARM_BUDGET
    res, _ = run(budget=b, save="--no-save" not in a)
    print("\n" + "=" * 92)
    print("PERFORMANCE vs THE APPROACH-1 LOGISTIC BASELINE")
    print("=" * 92)
    show = res[["model", "auc_in", "auc_oos", "f1", "recall", "precision",
                "auc_vs_base_pct", "outperforms"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    base = res[res["model"] == BASELINE]
    if not base.empty:
        b0 = base.iloc[0]
        beat = res[(res["model"] != BASELINE) & (res["auc_oos"] > b0["auc_oos"])]
        print(f"\nBaseline out-of-sample AUC {b0['auc_oos']:.3f}, F1 {b0['f1']:.3f}")
        print(f"{len(beat)}/{len(res)-1} tree models beat it on AUC"
              + (": " + ", ".join(beat["model"]) if len(beat) else ""))
        best = res.loc[res["auc_oos"].idxmax()]
        print(f"Best overall: {best['model']} (AUC {best['auc_oos']:.3f}, "
              f"F1 {best['f1']:.3f})")
    print("\nArtefacts: tex_out/tab_classify_compare.tex, "
          "tex_out/fig_classify_compare.png")
    print("Done.")


if __name__ == "__main__":
    main()
