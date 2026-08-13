# -*- coding: utf-8 -*-
"""
cmdf_ensemble.py -- two more methods on top of the four tree ensembles:

    Soft-Vote   the plain average of the four base predictions. No extra fitting,
                so it cannot leak. Helps whenever base errors are less than
                perfectly correlated.

    Stacking    a linear meta-learner over the base predictions. Learns HOW MUCH to
                trust each base model instead of assuming equal weight, and usually
                beats the plain average -- but only if it is built without leakage.

LEAKAGE CONTROL FOR STACKING
    The meta-learner must never see a base prediction that was made on data the base
    model was trained on. Inside each expanding-window fold this module therefore:

        1. splits the TRAINING window again by time -> inner-train / inner-holdout
        2. fits the base models on inner-train, predicts inner-holdout
        3. fits the meta-learner on those inner-holdout predictions only
        4. refits the base models on the FULL training window
        5. applies the meta-learner to the test window

    Skipping step 1-3 and fitting the meta on in-sample base predictions is the
    classic stacking mistake: the meta-learner then learns to trust whichever base
    model overfits hardest, and the reported score is meaningless.

RUN
    python cmdf_ensemble.py                 # regression + classification
    python cmdf_ensemble.py --regression
    python cmdf_ensemble.py --classification
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

warnings.filterwarnings("ignore")

import cmdf_gbm_compare as base
import cmdf_tree_models as tm

HERE = tm.HERE
OUTDIR = tm.OUTDIR
DB = tm.DB
TARGET = base.TARGET
out = tm.out

TREE_MODELS = tm.TREE_MODELS
SOFT, STACK = "Soft-Vote (avg of 4)", "Stacking (linear meta)"
MC = dict(tm.MC)
MC[SOFT] = "#7c3aed"
MC[STACK] = "#be185d"
MC["Ridge (linear)"] = "#9ca3af"
MC["Logistic (Approach 1)"] = "#6b7280"


# ========================================================== regression =======
def _fit_predict(name, Xtr, ytr, Xte):
    from sklearn.preprocessing import StandardScaler
    ctor = base.models()[name]
    m = ctor()
    if name == "Ridge (linear)":
        sc = StandardScaler().fit(Xtr)
        m.fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def regression_ensembles(df_full, verbose=True, with_ensembles=True):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import spearmanr

    df, lag = base.build_sample(df_full, "sample_noESG", False)
    X = df[lag].reset_index(drop=True)
    y = df[TARGET].reset_index(drop=True)
    yr = df["year"].reset_index(drop=True).astype(int)
    cuts = [c for c in range(base.EXPAND_FIRST_CUT, int(yr.max()), base.EXPAND_STEP)]
    avail = [m for m in TREE_MODELS if m in base.models()]
    if verbose:
        print(f"\n  regression: {len(df):,} firm-months, base models {avail}")

    rows, folds, weights = [], [], []
    extra = [SOFT, STACK] if with_ensembles else []
    per_model = {m: {"r2": [], "rm": [], "ma": [], "sp": []}
                 for m in avail + extra}
    # per-observation test predictions, kept so the DM test and the ranking metrics
    # can be computed on exactly the same rows for every model
    obs_pred = {m: [] for m in avail + extra}
    obs_true = []
    for t in cuts:
        tr = (yr <= t).values
        te = ((yr > t) & (yr <= t + base.EXPAND_STEP)).values
        if te.sum() < base.MIN_TEST or tr.sum() < base.MIN_TRAIN:
            continue
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        # ---- base predictions on the test window
        P_te = {}
        for m in avail:
            P_te[m] = _fit_predict(m, Xtr, ytr, Xte)

        # ---- inner split of the TRAINING window, for the meta-learner only
        yr_tr = yr[tr].reset_index(drop=True)
        inner_cut = int(yr_tr.quantile(0.70))
        i_tr = (yr_tr <= inner_cut).values
        i_ho = ~i_tr
        w = None
        if i_tr.sum() >= 200 and i_ho.sum() >= 100:
            Xi_tr = Xtr.reset_index(drop=True)[i_tr]
            Xi_ho = Xtr.reset_index(drop=True)[i_ho]
            yi_tr = ytr.reset_index(drop=True)[i_tr]
            yi_ho = ytr.reset_index(drop=True)[i_ho]
            P_ho = np.column_stack([_fit_predict(m, Xi_tr, yi_tr, Xi_ho)
                                    for m in avail])
            meta = RidgeCV(alphas=np.logspace(-3, 3, 13))
            meta.fit(P_ho, yi_ho)
            w = dict(zip(avail, meta.coef_))
            w["intercept"] = float(meta.intercept_)
            weights.append({"cut_year": t, **{k: float(v) for k, v in w.items()}})
            p_stack = meta.predict(np.column_stack([P_te[m] for m in avail]))
        else:
            p_stack = None

        if with_ensembles:
            P_te[SOFT] = np.mean([P_te[m] for m in avail], axis=0)
            if p_stack is not None:
                P_te[STACK] = p_stack

        # only record observations for which every model produced a prediction,
        # otherwise the DM test would compare models on different samples
        if set(P_te) >= set(avail + extra):
            obs_true.append(np.asarray(yte, dtype=float))
            for m in avail + extra:
                obs_pred[m].append(np.asarray(P_te[m], dtype=float))

        for m, p in P_te.items():
            r2 = r2_score(yte, p)
            rm = float(np.sqrt(mean_squared_error(yte, p)))
            ma = mean_absolute_error(yte, p)
            sp = spearmanr(yte, p).correlation
            per_model[m]["r2"].append(r2)
            per_model[m]["rm"].append(rm)
            per_model[m]["ma"].append(ma)
            per_model[m]["sp"].append(sp)
            folds.append(dict(model=m, cut_year=t, R2=float(r2), RMSE=rm,
                              MAE=float(ma), Spearman=float(sp),
                              n_train=int(tr.sum()), n_test=int(te.sum())))

    for m, d in per_model.items():
        if not d["r2"]:
            continue
        rows.append(dict(model=m, R2=float(np.median(d["r2"])),
                         RMSE=float(np.mean(d["rm"])), MAE=float(np.mean(d["ma"])),
                         Spearman=float(np.nanmean(d["sp"])),
                         R2_min=float(np.min(d["r2"])), R2_max=float(np.max(d["r2"])),
                         n_folds=len(d["r2"])))
    res = pd.DataFrame(rows)
    order = avail + extra
    res["_o"] = res["model"].map({m: i for i, m in enumerate(order)}).fillna(99)
    res = res.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    if verbose:
        for _, r in res.iterrows():
            print(f"    {r['model']:24s} R2={r['R2']:6.3f}  RMSE={r['RMSE']:6.3f}  "
                  f"Spearman={r['Spearman']:6.3f}")
    obs = None
    if obs_true:
        obs = {"y": np.concatenate(obs_true),
               **{m: np.concatenate(v) for m, v in obs_pred.items() if v}}
    return res, pd.DataFrame(folds), pd.DataFrame(weights), avail, obs


# ============================================================== DM test =====
def dm_test(e1, e2, h=1, power=2):
    """Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction.

    Tests whether two forecasts have equal expected loss. Returns (stat, p, better).
    A positive statistic means model 1 has the LARGER loss, i.e. model 2 is better.
    Reporting a raw accuracy gap without this test invites reading noise as a result.
    """
    from scipy import stats
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan, ""
    dbar = d.mean()
    # long-run variance with h-1 autocovariances
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * gk
    if var <= 0:
        return np.nan, np.nan, ""
    stat = dbar / np.sqrt(var / n)
    # HLN correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = stat * corr
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    better = "model 2" if dbar > 0 else "model 1"
    return float(stat), float(p), better


def dm_matrix(obs, models, verbose=True):
    """Pairwise DM p-values on squared error, plus a champion-vs-all column."""
    if obs is None:
        return pd.DataFrame(), pd.DataFrame()
    y = obs["y"]
    err = {m: y - obs[m] for m in models if m in obs}
    names = [m for m in models if m in err]
    rmse = {m: float(np.sqrt(np.mean(err[m] ** 2))) for m in names}
    champ = min(rmse, key=rmse.get)

    grid = pd.DataFrame(index=names, columns=names, dtype=float)
    rows = []
    for i, a in enumerate(names):
        for b in names:
            if a == b:
                grid.loc[a, b] = np.nan
                continue
            stat, p, _ = dm_test(err[a], err[b])
            grid.loc[a, b] = p
        stat, p, _ = dm_test(err[a], err[champ])
        rows.append(dict(model=a, RMSE=rmse[a],
                         vs_champion=champ,
                         dm_stat=(np.nan if a == champ else stat),
                         p_value=(np.nan if a == champ else p),
                         verdict=("champion" if a == champ else
                                  ("worse (p<0.05)" if p is not None and p < 0.05
                                   else "not distinguishable"))))
    tab = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    if verbose:
        print(f"\n  DM test against the champion ({champ}), squared-error loss, "
              f"n={len(y):,}")
        for _, r in tab.iterrows():
            print(f"    {r['model']:24s} RMSE={r['RMSE']:.4f}  "
                  f"p={'--' if pd.isna(r['p_value']) else f'{r[chr(112)+chr(95)+chr(118)+chr(97)+chr(108)+chr(117)+chr(101)]:.4f}'}"
                  f"  {r['verdict']}")
    return tab, grid


# ================================================= ranking metrics (reg) ====
def ranking_metrics(obs, models, budget=0.10, verbose=True):
    """Turn the regression into the watchlist question it is actually used for:
    can the model pick out the riskiest firm-months?

    The positive class is the top `budget` share of ACTUAL ln(PD); each model then
    flags the same number of firm-months by predicted risk. Precision, recall and F1
    follow. RMSE alone cannot answer this: a model can have good average error and
    still miss the tail that matters for supervision."""
    if obs is None:
        return pd.DataFrame()
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, average_precision_score)
    y = obs["y"]
    k = max(1, int(round(budget * len(y))))
    thr = np.sort(y)[::-1][k - 1]
    y_bin = (y >= thr).astype(int)
    rows = []
    for m in models:
        if m not in obs:
            continue
        p = obs[m]
        idx = np.argsort(p)[::-1][:k]
        pred = np.zeros(len(y), dtype=int)
        pred[idx] = 1
        rows.append(dict(
            model=m,
            precision=float(precision_score(y_bin, pred, zero_division=0)),
            recall=float(recall_score(y_bin, pred, zero_division=0)),
            f1=float(f1_score(y_bin, pred, zero_division=0)),
            auc=float(roc_auc_score(y_bin, p)),
            avg_precision=float(average_precision_score(y_bin, p)),
            n_flagged=int(k), n_pos=int(y_bin.sum())))
    tab = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    if verbose and not tab.empty:
        print(f"\n  ranking metrics: positive = top {budget*100:.0f}% of actual "
              f"ln(PD) ({int(tab.iloc[0]['n_pos']):,} of {len(y):,} rows)")
        for _, r in tab.iterrows():
            print(f"    {r['model']:24s} F1={r['f1']:.3f}  recall={r['recall']:.3f}  "
                  f"precision={r['precision']:.3f}  AUC={r['auc']:.3f}")
    return tab


# ====================================================== classification ======
def _rankit(p):
    """Map scores to [0,1] by rank. Removes every difference in scale and
    calibration between models, keeping only the ordering."""
    from scipy.stats import rankdata
    r = rankdata(p, method="average")
    return (r - 1) / max(len(r) - 1, 1)


def soft_vote_variants(P, y, auc_w=None, budget=0.02):
    """Four ways to combine base probabilities under heavy class imbalance.

    Plain probability averaging has a specific weakness here: with 0.19% positives
    each model is trained with class weighting and ends up on its OWN probability
    scale. Averaging then lets whichever model emits the largest numbers dominate,
    regardless of how well it ranks. The alternatives below remove that effect.

        prob    mean of the raw probabilities                (the naive version)
        logit   mean of log-odds -- pulls apart the crowded region near zero
        rank    mean of within-model ranks                   (scale-free)
        auc-w   rank average weighted by each model's out-of-sample AUC
    """
    names = list(P.keys())
    M = np.column_stack([P[m] for m in names])
    variants = {}
    variants["Soft-Vote prob"] = M.mean(axis=1)
    eps = 1e-6
    Mc = np.clip(M, eps, 1 - eps)
    variants["Soft-Vote logit"] = np.log(Mc / (1 - Mc)).mean(axis=1)
    R = np.column_stack([_rankit(M[:, j]) for j in range(M.shape[1])])
    variants["Soft-Vote rank"] = R.mean(axis=1)
    if auc_w:
        w = np.array([max(auc_w.get(m, 0.5) - 0.5, 0.0) for m in names], dtype=float)
        if w.sum() > 0:
            w = w / w.sum()
            variants["Soft-Vote rank (AUC-wt)"] = R @ w
    return variants


def classification_ensembles(budget=0.02, verbose=True):
    import cmdf_tree_classify as cl
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    panel, Xdf, y, cols = cl.load_panel(verbose=verbose)
    Xv, yv = Xdf.to_numpy(float), y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    ev = sorted(panel.loc[y == 1, "issuer_code"].dropna().unique())
    clf = cl.classifiers()
    avail = [m for m in TREE_MODELS if m in clf]
    if verbose:
        print(f"\n  classification: leave-one-issuer-out over {len(ev)} issuers, "
              f"base models {avail}")

    def fit_pred(name, Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        m = clf[name]()
        m.fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]

    oy = []
    op = {m: [] for m in avail + [cl.BASELINE, SOFT, STACK]}
    for held in ev:
        tr = groups != held
        te = ~tr
        if yv[tr].sum() < 2:
            continue
        oy.append(yv[te])
        for m in avail + [cl.BASELINE]:
            op[m].append(fit_pred(m, Xv[tr], yv[tr], Xv[te]))

        # inner leave-one-issuer-out INSIDE the training set, for the meta only
        inner_ev = [e for e in ev if e != held]
        Pi, yi = [], []
        for e2 in inner_ev:
            i_tr = tr & (groups != e2)
            i_te = tr & (groups == e2)
            if yv[i_tr].sum() < 2 or i_te.sum() == 0:
                continue
            Pi.append(np.column_stack([fit_pred(m, Xv[i_tr], yv[i_tr], Xv[i_te])
                                       for m in avail]))
            yi.append(yv[i_te])
        base_te = np.column_stack([op[m][-1] for m in avail])
        op[SOFT].append(base_te.mean(axis=1))
        if Pi:
            meta = LogisticRegression(max_iter=3000, C=1.0,
                                      class_weight="balanced")
            meta.fit(np.vstack(Pi), np.concatenate(yi))
            op[STACK].append(meta.predict_proba(base_te)[:, 1])
        else:
            op[STACK].append(base_te.mean(axis=1))

    yy = np.concatenate(oy)

    # base-model AUCs first, so the AUC-weighted variant can use them
    from sklearn.metrics import roc_auc_score as _auc
    base_auc = {}
    for m in avail:
        if op[m]:
            pp = np.concatenate(op[m])
            if 0 < yy.sum() < len(yy):
                base_auc[m] = float(_auc(yy, pp))
    # every soft-vote flavour, computed from the SAME stored base predictions
    Pmat = {m: np.concatenate(op[m]) for m in avail if op[m]}
    for vname, vscore in soft_vote_variants(Pmat, yy, base_auc, budget).items():
        op[vname] = [vscore]

    rows = []
    sv_names = [k for k in op if k.startswith("Soft-Vote")]
    for m in [cl.BASELINE] + avail + sorted(sv_names) + [STACK]:
        if not op[m]:
            continue
        pp = np.concatenate(op[m])
        auc = float(roc_auc_score(yy, pp)) if 0 < yy.sum() < len(yy) else np.nan
        prec, rec, f1, k = cl._budget_metrics(yy, pp, budget)
        rows.append(dict(model=m, auc_oos=auc, f1=f1, recall=rec, precision=prec,
                         n_flagged=k, n_eval=int(len(yy)), n_pos=int(yy.sum())))
    res = pd.DataFrame(rows)
    res = res.drop_duplicates("model").reset_index(drop=True)
    b = res[res["model"] == cl.BASELINE]
    if not b.empty:
        b0 = float(b.iloc[0]["auc_oos"])
        res["auc_vs_base_pct"] = (res["auc_oos"] - b0) / abs(b0) * 100
    if verbose:
        for _, r in res.iterrows():
            print(f"    {r['model']:24s} AUC={r['auc_oos']:6.3f}  F1={r['f1']:6.3f}  "
                  f"recall={r['recall']:6.3f}")
    return res


# ================================================================ output ====
def write_outputs(reg, reg_folds, weights, cls, avail, dm=None, rank=None):
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    fpc = lambda v: "--" if pd.isna(v) else f"{v:+.1f}\\%"

    if reg is not None and not reg.empty:
        best = reg.loc[reg["R2"].idxmax(), "model"]
        tm.write_tex_table(
            reg, out("tab_ensemble_regression.tex"),
            "Adding a soft-vote and a stacked ensemble (regression on "
            "$\\ln(PD)_{12}$, Expanded sample)", "tab:ens-reg",
            cols=["model", "R2", "RMSE", "MAE", "Spearman", "R2_min"],
            fmt={"R2": f3, "RMSE": f3, "MAE": f3, "Spearman": f3, "R2_min": f3},
            bold_row=lambda r, b=best: r["model"] == b,
            note=("Soft-Vote is the unweighted mean of the four base predictions. "
                  "Stacking fits a ridge meta-learner on base predictions made on an "
                  "inner time-split holdout inside each training window, so the meta "
                  "never sees in-sample base predictions. $R^2$ is the median across "
                  "folds and R2\\_min the worst fold."))
        reg.to_csv(out("ensemble_regression.csv"), index=False)

    if weights is not None and not weights.empty:
        w = weights.copy()
        cols = ["cut_year"] + [c for c in avail if c in w.columns] + \
               (["intercept"] if "intercept" in w.columns else [])
        tm.write_tex_table(
            w, out("tab_stack_weights.tex"),
            "Ridge meta-learner weights per fold (regression stacking)",
            "tab:stack-w", cols=cols,
            fmt={c: f3 for c in cols if c != "cut_year"},
            note=("Weights are fitted on the inner holdout of each training window. "
                  "A negative weight is not a bug: the meta-learner is free to use a "
                  "base model as a correction term rather than as a predictor."))
        w.to_csv(out("stack_weights.csv"), index=False)

    if cls is not None and not cls.empty:
        bestc = cls.loc[cls["auc_oos"].idxmax(), "model"]
        tm.write_tex_table(
            cls, out("tab_ensemble_classify.tex"),
            "Adding a soft-vote and a stacked ensemble (classification of the real "
            "payment-default event, 33 features)", "tab:ens-cls",
            cols=["model", "auc_oos", "f1", "recall", "precision", "auc_vs_base_pct"],
            fmt={"auc_oos": f3, "f1": f3, "recall": f3, "precision": f3,
                 "auc_vs_base_pct": fpc},
            bold_row=lambda r, b=bestc: r["model"] == b,
            note=("Leave-one-issuer-out. The stacked meta-learner is trained on a "
                  "nested leave-one-issuer-out inside each training set. F1, recall "
                  "and precision use a matched alarm budget of 2\\% of issuer-months. "
                  "The last column compares out-of-sample AUC with the Approach-1 "
                  "logistic baseline."))
        cls.to_csv(out("ensemble_classify.csv"), index=False)

    if dm is not None and not dm.empty:
        tm.write_tex_table(
            dm, out("tab_dm_test.tex"),
            "Diebold-Mariano test against the most accurate model "
            "(squared-error loss, pooled out-of-time observations)", "tab:dm",
            cols=["model", "RMSE", "dm_stat", "p_value", "verdict"],
            fmt={"RMSE": lambda v: f"{v:.4f}",
                 "dm_stat": lambda v: "--" if pd.isna(v) else f"{v:.3f}",
                 "p_value": lambda v: "--" if pd.isna(v) else f"{v:.4f}"},
            bold_row=lambda r: r["verdict"] == "champion",
            note=("The test asks whether the difference in expected squared error is "
                  "distinguishable from zero, with the Harvey-Leybourne-Newbold "
                  "small-sample correction. \emph{not distinguishable} means the "
                  "accuracy gap to the champion is within noise, so preferring one "
                  "model over the other cannot be justified on this evidence alone."))
        dm.to_csv(out("dm_test.csv"), index=False)

    if rank is not None and not rank.empty:
        bestr = rank.loc[rank["f1"].idxmax(), "model"]
        tm.write_tex_table(
            rank, out("tab_ranking_metrics.tex"),
            "Watchlist metrics for the regression models: identifying the riskiest "
            "10\% of firm-months", "tab:rank",
            cols=["model", "precision", "recall", "f1", "auc", "avg_precision"],
            fmt={c: (lambda v: f"{v:.3f}") for c in
                 ("precision", "recall", "f1", "auc", "avg_precision")},
            bold_row=lambda r, b=bestr: r["model"] == b,
            note=("The positive class is the top decile of ACTUAL $\ln(PD)_{12}$; "
                  "every model flags the same number of firm-months by predicted "
                  "risk. This answers the supervisory question directly, which RMSE "
                  "cannot: a model can have a good average error and still miss the "
                  "tail that matters."))
        rank.to_csv(out("ranking_metrics.csv"), index=False)

    # figure: both panels side by side
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))
    if reg is not None and not reg.empty:
        ax = axes[0]
        d = reg
        xs = np.arange(len(d))
        ax.bar(xs, d["R2"], color=[MC.get(m, "#888") for m in d["model"]],
               alpha=0.92, edgecolor="white")
        for x, (_, r) in zip(xs, d.iterrows()):
            ax.text(x, r["R2"] + 0.006, f"{r['R2']:.3f}", ha="center", fontsize=7.5)
        bb = d[d["model"] == "XGBoost"]["R2"]
        if not bb.empty:
            ax.axhline(float(bb.iloc[0]), color="#dc2626", lw=1.2, ls="--")
            ax.text(len(d) - 0.5, float(bb.iloc[0]) + 0.012, "best single model",
                    fontsize=7.5, color="#dc2626", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels([m.split(" (")[0] for m in d["model"]], rotation=18,
                           fontsize=8)
        ax.set_ylabel("median out-of-time $R^2$")
        ax.set_title("Regression on $\\ln(PD)_{12}$", fontsize=10.5,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    if cls is not None and not cls.empty:
        ax = axes[1]
        d = cls
        xs = np.arange(len(d))
        w = 0.38
        ax.bar(xs - w / 2, d["auc_oos"], width=w,
               color=[MC.get(m, "#888") for m in d["model"]], alpha=0.92,
               label="AUC (oos)", edgecolor="white")
        ax.bar(xs + w / 2, d["f1"], width=w, color="#94a3b8", alpha=0.9,
               label="F1", edgecolor="white")
        b0 = d[d["model"] == "Logistic (Approach 1)"]["auc_oos"]
        if not b0.empty:
            ax.axhline(float(b0.iloc[0]), color="#dc2626", lw=1.2, ls="--")
            ax.text(len(d) - 0.5, float(b0.iloc[0]) + 0.012, "logistic baseline",
                    fontsize=7.5, color="#dc2626", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels([m.split(" (")[0] for m in d["model"]], rotation=18,
                           fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("Classification of the real default event", fontsize=10.5,
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Does combining the four tree models help?", fontsize=11.5,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    tm.save_fig(fig, "fig_ensemble_compare.png")


def run(do_reg=True, do_cls=True, verbose=True, with_ensembles=True):
    print("=" * 78)
    print("Soft-vote and stacked ensembles on top of the four tree models")
    print("=" * 78)
    reg = reg_folds = weights = cls = dm = rank = None
    avail = TREE_MODELS
    if do_reg:
        df_full = base.load_full(verbose=verbose)
        reg, reg_folds, weights, avail, obs = regression_ensembles(
            df_full, verbose, with_ensembles=with_ensembles)
        order = avail + ([SOFT, STACK] if with_ensembles else [])
        dm, _grid = dm_matrix(obs, order, verbose)
        rank = ranking_metrics(obs, order, verbose=verbose)
    if do_cls:
        cls = classification_ensembles(verbose=verbose)
    write_outputs(reg, reg_folds, weights, cls, avail, dm, rank)
    con = sqlite3.connect(DB)
    if reg is not None and not reg.empty:
        reg.to_sql("cmdf_ensemble_regression", con, if_exists="replace", index=False)
    if cls is not None and not cls.empty:
        cls.to_sql("cmdf_ensemble_classify", con, if_exists="replace", index=False)
    if dm is not None and not dm.empty:
        dm.to_sql("cmdf_dm_test", con, if_exists="replace", index=False)
    if rank is not None and not rank.empty:
        rank.to_sql("cmdf_ranking_metrics", con, if_exists="replace", index=False)
    con.commit(); con.close()
    return reg, cls, dm, rank


def main():
    a = sys.argv
    do_reg = "--classification" not in a
    do_cls = "--regression" not in a
    reg, cls, dm, rank = run(do_reg, do_cls,
                             with_ensembles="--no-ensemble" not in a)
    print("\n" + "=" * 92)
    print("DOES COMBINING HELP?")
    print("=" * 92)
    if reg is not None and not reg.empty:
        best_single = reg[~reg["model"].isin([SOFT, STACK])]["R2"].max()
        print("\nRegression (median out-of-time R2)")
        print(reg[["model", "R2", "RMSE", "Spearman", "R2_min"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        for m in (SOFT, STACK):
            r = reg[reg["model"] == m]
            if r.empty:
                continue
            v = float(r.iloc[0]["R2"])
            print(f"  {m}: R2 {v:.3f} vs best single {best_single:.3f} "
                  f"({'BETTER' if v > best_single else 'not better'})")
    if cls is not None and not cls.empty:
        best_single = cls[~cls["model"].isin([SOFT, STACK])]["auc_oos"].max()
        print("\nClassification (leave-one-issuer-out AUC)")
        print(cls[["model", "auc_oos", "f1", "recall", "precision"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        for m in (SOFT, STACK):
            r = cls[cls["model"] == m]
            if r.empty:
                continue
            v = float(r.iloc[0]["auc_oos"])
            print(f"  {m}: AUC {v:.3f} vs best single {best_single:.3f} "
                  f"({'BETTER' if v > best_single else 'not better'})")
    print("\nArtefacts: tex_out/tab_ensemble_regression.tex, "
          "tab_ensemble_classify.tex, tab_stack_weights.tex, "
          "fig_ensemble_compare.png")
    print("Done.")


if __name__ == "__main__":
    main()
