# -*- coding: utf-8 -*-
"""
cmdf_feature_select.py -- does a small, model-chosen feature set do as well as all 33?

Three feature sets are compared on IDENTICAL rows of the iBond panel:

    All 33          the full 33-determinant bond panel
    Top-5 (own)     for each model, the five determinants that model itself ranks
                    highest -- selected INSIDE each training fold, never on the
                    evaluation rows
    19 (curve set)  the bond_ews specification: 10 bond-level levels + 3 twelve-month
                    changes + 6 yield-curve factors (Level / Slope / Curvature and
                    their changes)

WHY SELECTION HAPPENS INSIDE THE FOLD
    Ranking the determinants on the whole sample and then scoring the same rows makes
    the small set look better than it is: the ranking has already seen the answers.
    Here the importance ranking is recomputed from the training issuers of each fold,
    so the chosen determinants never depend on the held-out issuer.

METRICS
    Same as the rest of the report: leave-one-issuer-out AUC, and F1 / recall /
    precision at a matched alarm budget of 2% of issuer-months.

RUN
    python cmdf_feature_select.py
    python cmdf_feature_select.py --top 8
    python cmdf_feature_select.py --no-save
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import bond_ews as be
import cmdf_tree_classify as cl
import cmdf_tree_models as tm

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out
TREE_MODELS = tm.TREE_MODELS
MC = dict(tm.MC)

SET_ALL, SET_TOP, SET_CURVE = "All 33", "Top-5 (own)", "19 (curve set)"
SC = {SET_ALL: "#1f3a5f", SET_TOP: "#a8501a", SET_CURVE: "#2e7d4f"}

TOP_K = 5
BUDGET = 0.02

T_RESULT = "cmdf_featsel_result"
T_PICKED = "cmdf_featsel_picked"
T_DM = "cmdf_featsel_dm"
T_PERM = "cmdf_featsel_perm"
T_LOFO = "cmdf_featsel_lofo"


# ============================================== significance on probabilities =
def dm_brier(p1, p2, y, h=1):
    """Diebold-Mariano on the Brier loss of two probability forecasts.

    The regression sections of this report use squared error on ln(PD). For a binary
    outcome the equivalent loss is the Brier score (y - p)^2, so the same test
    applies: it asks whether the difference in expected loss is distinguishable from
    zero rather than whether one AUC number happens to be larger.
    """
    from scipy import stats
    y = np.asarray(y, dtype=float)
    l1 = (y - np.asarray(p1, dtype=float)) ** 2
    l2 = (y - np.asarray(p2, dtype=float)) ** 2
    d = l1 - l2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    var = np.mean((d - dbar) ** 2)
    for k in range(1, h):
        var += 2 * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


# ================================================================== data =====
def load_joint(verbose=True):
    """One frame holding both feature sets on the same issuer-months."""
    panel, X33, y, cols33 = cl.load_panel(verbose=False)
    con = sqlite3.connect(DB)
    p19 = pd.read_sql("SELECT * FROM bond_ews_panel", con)
    con.close()

    key = ["issuer_code", "month"]
    left = panel[key].astype(str).copy()
    left["_i"] = np.arange(len(left))
    right = p19[key + [c for c in be.FEATURES if c in p19.columns]].copy()
    right[key] = right[key].astype(str)
    merged = left.merge(right, on=key, how="left")
    merged = merged.sort_values("_i")
    cols19 = [c for c in be.FEATURES if c in merged.columns]
    X19 = merged[cols19].apply(pd.to_numeric, errors="coerce")
    X19 = X19.fillna(X19.median(numeric_only=True)).fillna(0.0).reset_index(drop=True)

    if verbose:
        print(f"  rows {len(panel):,} | issuers {panel['issuer_code'].nunique()} | "
              f"positives {int(y.sum())} ({y.mean()*100:.2f}%)")
        print(f"  feature sets: 33 -> {len(cols33)} cols, "
              f"curve set -> {len(cols19)} cols")
        print(f"  curve factors present: "
              f"{[c for c in be.CURVE_FEATURES if c in cols19]}")
    return panel, X33.reset_index(drop=True), X19, y, cols33, cols19


# ============================================================== evaluate =====
def _fit_prob(name, Xtr, ytr, Xte, return_model=False):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = cl.classifiers()[name]()
    m.fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return (p, m) if return_model else p


def evaluate_sets(panel, X33, X19, y, cols33, cols19, top_k=TOP_K,
                  budget=BUDGET, verbose=True):
    from sklearn.metrics import roc_auc_score

    A33, A19 = X33.to_numpy(float), X19.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    ev = sorted(panel.loc[y == 1, "issuer_code"].dropna().unique())
    if verbose:
        print(f"\n  leave-one-issuer-out over {len(ev)} defaulted issuers, "
              f"top_k={top_k}")

    rows, picks = [], []
    store = {}          # (model, feature_set) -> pooled test probabilities
    y_store = None
    for name in TREE_MODELS:
        if name not in cl.classifiers():
            continue
        t0 = time.time()
        preds = {SET_ALL: [], SET_TOP: [], SET_CURVE: []}
        oy = []
        chosen = Counter()
        for held in ev:
            tr = groups != held
            te = ~tr
            if yv[tr].sum() < 2:
                continue
            oy.append(yv[te])

            # full 33
            preds[SET_ALL].append(_fit_prob(name, A33[tr], yv[tr], A33[te]))

            # top-k chosen from THIS fold's training rows only
            _p, m = _fit_prob(name, A33[tr], yv[tr], A33[te], return_model=True)
            imp = np.asarray(getattr(m, "feature_importances_",
                                     np.zeros(len(cols33))), dtype=float)
            idx = np.argsort(imp)[::-1][:top_k]
            for j in idx:
                chosen[cols33[j]] += 1
            preds[SET_TOP].append(
                _fit_prob(name, A33[np.ix_(tr, idx)] if False else A33[tr][:, idx],
                          yv[tr], A33[te][:, idx]))

            # 19-feature curve set
            preds[SET_CURVE].append(_fit_prob(name, A19[tr], yv[tr], A19[te]))

        if not oy:
            continue
        yy = np.concatenate(oy)
        if y_store is None:
            y_store = yy
        for sname, plist in preds.items():
            pp = np.concatenate(plist)
            store[(name, sname)] = pp
            auc = float(roc_auc_score(yy, pp)) if 0 < yy.sum() < len(yy) else np.nan
            prec, rec, f1, k = cl._budget_metrics(yy, pp, budget)
            rows.append(dict(model=name, feature_set=sname, n_features=(
                len(cols33) if sname == SET_ALL else
                (top_k if sname == SET_TOP else len(cols19))),
                auc_oos=auc, f1=f1, recall=rec, precision=prec,
                n_flagged=k, n_eval=int(len(yy)), n_pos=int(yy.sum())))
        for feat, cnt in chosen.most_common():
            picks.append(dict(model=name, feature=feat,
                              pretty=be.PRETTY.get(feat, feat) if hasattr(be, "PRETTY")
                              else feat,
                              times_selected=int(cnt), n_folds=len(oy),
                              share=cnt / max(len(oy), 1)))
        if verbose:
            d = pd.DataFrame([r for r in rows if r["model"] == name])
            line = "  ".join(f"{r['feature_set']}={r['auc_oos']:.3f}"
                             for _, r in d.iterrows())
            print(f"    {name:16s} AUC  {line}   ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows), pd.DataFrame(picks), store, y_store


def dm_between_sets(store, y, verbose=True):
    """For each model, test the two reduced sets against the full 33."""
    rows = []
    models = sorted({m for m, _ in store})
    for m in [x for x in TREE_MODELS if x in models]:
        if (m, SET_ALL) not in store:
            continue
        base_p = store[(m, SET_ALL)]
        for sname in (SET_TOP, SET_CURVE):
            if (m, sname) not in store:
                continue
            stat, pv = dm_brier(base_p, store[(m, sname)], y)
            better = ("reduced set" if stat is not None and stat > 0 else "all 33")
            rows.append(dict(model=m, feature_set=sname, dm_stat=stat, p_value=pv,
                             lower_loss=("--" if np.isnan(stat) else better),
                             verdict=("not distinguishable" if (np.isnan(pv) or pv >= 0.05)
                                      else f"{better} better (p<0.05)")))
    tab = pd.DataFrame(rows)
    if verbose and not tab.empty:
        print()
        print("  DM test on Brier loss, each reduced set against all 33")
        for _, r in tab.iterrows():
            pv = "--" if pd.isna(r["p_value"]) else f"{r['p_value']:.4f}"
            print(f"    {r['model']:16s} {r['feature_set']:16s} p={pv:>8s}  "
                  f"{r['verdict']}")
    return tab


def permutation_importance_oos(panel, X, y, cols, model_name="XGBoost", n_rep=3,
                               verbose=True):
    """Out-of-sample permutation importance: shuffle one determinant in the held-out
    rows and measure how much AUC falls. Unlike gain or SHAP this is measured on data
    the model never saw, so it answers 'does this determinant actually carry
    predictive information' rather than 'did the trees use it'."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    ev = sorted(panel.loc[y == 1, "issuer_code"].dropna().unique())
    rng = np.random.default_rng(42)
    base_scores, drops = [], {c: [] for c in cols}
    for held in ev:
        tr, te = groups != held, groups == held
        if yv[tr].sum() < 2 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(A[tr])
        m = cl.classifiers()[model_name]()
        m.fit(sc.transform(A[tr]), yv[tr])
        Xte = A[te].copy()
        # a single held-out issuer has one class only, so AUC must be pooled;
        # score the held-out rows against the full training distribution instead
        p_base = m.predict_proba(sc.transform(Xte))[:, 1]
        base_scores.append((yv[te], p_base))
        for j, c in enumerate(cols):
            ds = []
            for _ in range(n_rep):
                Xp = Xte.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                ds.append(m.predict_proba(sc.transform(Xp))[:, 1])
            drops[c].append(np.mean(ds, axis=0))
    if not base_scores:
        return pd.DataFrame()
    yy = np.concatenate([a for a, _ in base_scores])
    pp = np.concatenate([b for _, b in base_scores])
    auc0 = roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan
    rows = []
    for c in cols:
        pc = np.concatenate(drops[c])
        aucc = roc_auc_score(yy, pc) if 0 < yy.sum() < len(yy) else np.nan
        stat, pv = dm_brier(pc, pp, yy)      # loss with feature shuffled vs intact
        rows.append(dict(feature=c, auc_full=float(auc0), auc_permuted=float(aucc),
                         auc_drop=float(auc0 - aucc), dm_stat=stat, p_value=pv,
                         significant=bool(pv is not None and not np.isnan(pv)
                                          and pv < 0.05 and auc0 > aucc)))
    tab = pd.DataFrame(rows).sort_values("auc_drop", ascending=False)
    tab = tab.reset_index(drop=True)
    if verbose:
        print()
        print(f"  out-of-sample permutation importance ({model_name}), "
              f"baseline AUC {auc0:.3f}")
        for _, r in tab.head(12).iterrows():
            mark = "*" if r["significant"] else " "
            pv = "--" if pd.isna(r["p_value"]) else f"{r['p_value']:.4f}"
            print(f"   {mark} {r['feature']:28s} AUC drop {r['auc_drop']:+.4f}  "
                  f"p={pv}")
    return tab


# ================================================================ output =====
def cross_model_randomisation(panel, X, y, cols, n_rep=25, verbose=True):
    """Crossed model x determinant randomisation test.

    For every combination of the four models and the 33 determinants, the
    determinant is randomly shuffled in the held-out rows `n_rep` times and the
    resulting AUC recorded. That gives an empirical distribution of "what happens
    when this determinant carries no information", from which two quantities follow:

        prob_influential  the share of random shuffles that made AUC WORSE.
                          Read as: the probability, measured from the randomisation
                          itself, that the determinant is carrying signal rather than
                          noise. A value near 0.5 is exactly what an irrelevant
                          determinant produces.
        mean_auc_drop     the average loss of AUC across shuffles.

    Nothing is refitted inside the loop: each model is fitted once per fold and the
    shuffling happens only on the evaluation rows, so the measurement stays
    out-of-sample and the cost stays manageable.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    ev = sorted(panel.loc[y == 1, "issuer_code"].dropna().unique())
    models = [m for m in TREE_MODELS if m in cl.classifiers()]
    rng = np.random.default_rng(2024)
    rows = []

    for name in models:
        t0 = time.time()
        fits = []
        for held in ev:
            tr, te = groups != held, groups == held
            if yv[tr].sum() < 2 or te.sum() == 0:
                continue
            sc = StandardScaler().fit(A[tr])
            m = cl.classifiers()[name]()
            m.fit(sc.transform(A[tr]), yv[tr])
            fits.append((sc, m, te))
        if not fits:
            continue
        yy = np.concatenate([yv[te] for _, _, te in fits])
        p0 = np.concatenate([m.predict_proba(sc.transform(A[te]))[:, 1]
                             for sc, m, te in fits])
        auc0 = (roc_auc_score(yy, p0) if 0 < yy.sum() < len(yy) else np.nan)

        for j, c in enumerate(cols):
            aucs = []
            for _ in range(n_rep):
                parts = []
                for sc, m, te in fits:
                    Xp = A[te].copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    parts.append(m.predict_proba(sc.transform(Xp))[:, 1])
                pp = np.concatenate(parts)
                aucs.append(roc_auc_score(yy, pp)
                            if 0 < yy.sum() < len(yy) else np.nan)
            aucs = np.array(aucs, dtype=float)
            ok = np.isfinite(aucs)
            if not ok.any():
                continue
            worse = float(np.mean(aucs[ok] < auc0))
            rows.append(dict(model=name, feature=c, auc_full=float(auc0),
                             auc_shuffled_mean=float(np.mean(aucs[ok])),
                             mean_auc_drop=float(auc0 - np.mean(aucs[ok])),
                             prob_influential=worse, n_rep=int(ok.sum())))
        if verbose:
            print(f"    {name:16s} {len(cols)} determinants x {n_rep} shuffles "
                  f"({time.time()-t0:.0f}s)")

    d = pd.DataFrame(rows)
    if d.empty:
        return d, pd.DataFrame()
    # aggregate across models: a determinant that matters should matter everywhere
    agg = (d.groupby("feature")
           .agg(mean_prob=("prob_influential", "mean"),
                min_prob=("prob_influential", "min"),
                mean_drop=("mean_auc_drop", "mean"),
                n_models=("model", "nunique"))
           .reset_index())
    agg["models_above_0.9"] = [
        int((d[d.feature == f]["prob_influential"] >= 0.9).sum())
        for f in agg["feature"]]
    agg = agg.sort_values("mean_prob", ascending=False).reset_index(drop=True)
    if verbose:
        print("\n  crossed randomisation, probability the determinant is influential")
        print(f"  {'determinant':28s} {'mean p':>8s} {'min p':>8s} "
              f"{'AUC drop':>10s} {'models>=0.9':>12s}")
        for _, r in agg.head(12).iterrows():
            print(f"  {r['feature']:28s} {r['mean_prob']:8.2f} {r['min_prob']:8.2f} "
                  f"{r['mean_drop']:+10.4f} {int(r['models_above_0.9']):12d}")
    return d, agg


def write_outputs(res, picks, top_k=TOP_K, dm=None, perm=None, cross=None):
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"

    # main table: one row per model x feature set
    d = res.copy()
    order_m = {m: i for i, m in enumerate(TREE_MODELS)}
    order_s = {SET_ALL: 0, SET_TOP: 1, SET_CURVE: 2}
    d["_m"] = d["model"].map(order_m).fillna(99)
    d["_s"] = d["feature_set"].map(order_s).fillna(99)
    d = d.sort_values(["_m", "_s"]).drop(columns=["_m", "_s"]).reset_index(drop=True)
    best = d.loc[d["auc_oos"].idxmax()]
    tm.write_tex_table(
        d, out("tab_featsel.tex"),
        f"Comparison of feature sets on the iBond panel: all 33 determinants, the "
        f"top {top_k} chosen by each model, and the {int(d[d.feature_set==SET_CURVE].n_features.iloc[0]) if (d.feature_set==SET_CURVE).any() else 19}-determinant "
        f"specification that includes the yield-curve factors",
        "tab:featsel",
        cols=["model", "feature_set", "n_features", "auc_oos", "f1", "recall",
              "precision"],
        fmt={"auc_oos": f3, "f1": f3, "recall": f3, "precision": f3},
        bold_row=lambda r, b=best: (r["model"] == b["model"]
                                    and r["feature_set"] == b["feature_set"]),
        note=(f"Leave-one-issuer-out on identical rows for all three feature sets. "
              f"The top {top_k} determinants are re-selected from the training issuers "
              f"of every fold, so the selection never sees the held-out issuer. "
              f"F1, recall and precision use a matched alarm budget of "
              f"{BUDGET*100:.0f}\\% of issuer-months."))
    res.to_csv(out("featsel_result.csv"), index=False)

    # pivot table: AUC by model x feature set, with the change vs all-33
    piv = res.pivot_table(index="model", columns="feature_set", values="auc_oos")
    piv = piv.reindex([m for m in TREE_MODELS if m in piv.index])
    for c in (SET_TOP, SET_CURVE):
        if c in piv.columns and SET_ALL in piv.columns:
            piv[f"{c} vs 33"] = piv[c] - piv[SET_ALL]
    tab = piv.reset_index()
    cols = ["model"] + [c for c in (SET_ALL, SET_TOP, SET_CURVE) if c in tab.columns] \
        + [c for c in tab.columns if "vs 33" in c]
    tm.write_tex_table(
        tab, out("tab_featsel_delta.tex"),
        "Out-of-sample AUC by feature set, and the change relative to using all 33 "
        "determinants", "tab:featsel-delta",
        cols=cols,
        fmt={**{c: f3 for c in cols[1:] if "vs 33" not in c},
             **{c: (lambda v: "--" if pd.isna(v) else f"{v:+.3f}")
                for c in cols if "vs 33" in c}},
        note=("A positive change means the smaller set did better than the full 33. "
              "The curve set replaces most accounting detail with the yield-curve "
              "level, slope and curvature and their twelve-month changes."))

    # selection stability
    if not picks.empty:
        st = (picks.groupby(["feature", "pretty"])["times_selected"].sum()
              .reset_index().sort_values("times_selected", ascending=False).head(15))
        n_tot = picks["n_folds"].max() * picks["model"].nunique()
        st["share_of_all"] = st["times_selected"] / max(n_tot, 1)
        tm.write_tex_table(
            st, out("tab_featsel_stability.tex"),
            f"Determinants most often selected into the top {top_k} across models "
            "and folds", "tab:featsel-stab",
            cols=["pretty", "times_selected", "share_of_all"],
            fmt={"share_of_all": lambda v: f"{v*100:.0f}\\%"},
            bold_row=lambda r: r["pretty"] == st.iloc[0]["pretty"],
            note=("Counted over every model and every leave-one-issuer-out fold. A "
                  "determinant chosen in nearly all folds is a stable signal; one "
                  "chosen occasionally reflects fold-to-fold noise."))
        picks.to_csv(out("featsel_picked.csv"), index=False)

    if dm is not None and not dm.empty:
        tm.write_tex_table(
            dm, out("tab_featsel_dm.tex"),
            "Diebold-Mariano test on Brier loss: each reduced feature set against "
            "all 33 determinants", "tab:featsel-dm",
            cols=["model", "feature_set", "dm_stat", "p_value", "verdict"],
            fmt={"dm_stat": lambda v: "--" if pd.isna(v) else f"{v:.3f}",
                 "p_value": lambda v: "--" if pd.isna(v) else f"{v:.4f}"},
            note=("For a binary outcome the counterpart of squared error is the Brier "
                  "loss $(y-p)^2$, so the same test used on the regression sections "
                  "applies here. \emph{not distinguishable} means the accuracy "
                  "difference between the reduced set and the full 33 is within "
                  "noise, which supports using the smaller set on grounds of "
                  "parsimony."))
        dm.to_csv(out("featsel_dm.csv"), index=False)

    if perm is not None and not perm.empty:
        d2 = perm.head(15).copy()
        tm.write_tex_table(
            d2, out("tab_featsel_perm.tex"),
            "Out-of-sample permutation importance: change in AUC when each "
            "determinant is shuffled in the held-out rows", "tab:featsel-perm",
            cols=["feature", "auc_permuted", "auc_drop", "p_value", "significant"],
            fmt={"auc_permuted": lambda v: f"{v:.3f}",
                 "auc_drop": lambda v: f"{v:+.4f}",
                 "p_value": lambda v: "--" if pd.isna(v) else f"{v:.4f}",
                 "significant": lambda v: "yes" if v else "no"},
            bold_row=lambda r: bool(r["significant"]),
            note=("Each determinant is shuffled in the held-out rows only, so the "
                  "measurement is out-of-sample. A positive AUC drop means the model "
                  "loses accuracy without that determinant. The p-value comes from a "
                  "Diebold-Mariano test on Brier loss with and without the shuffle, "
                  "which distinguishes a genuine loss of information from ordinary "
                  "sampling variation."))
        perm.to_csv(out("featsel_perm.csv"), index=False)

    if cross is not None and not cross.empty:
        d3 = cross.head(15).copy()
        tm.write_tex_table(
            d3, out("tab_featsel_cross.tex"),
            "Crossed model-by-determinant randomisation: probability that a "
            "determinant carries predictive information", "tab:featsel-cross",
            cols=["feature", "mean_prob", "min_prob", "mean_drop",
                  "models_above_0.9"],
            fmt={"mean_prob": lambda v: f"{v:.2f}",
                 "min_prob": lambda v: f"{v:.2f}",
                 "mean_drop": lambda v: f"{v:+.4f}"},
            bold_row=lambda r: r["mean_prob"] >= 0.9 and r["models_above_0.9"] >= 3,
            note=("Each determinant was shuffled in the held-out rows 25 times per "
                  "model. The probability is the share of shuffles that reduced AUC. "
                  "A determinant carrying no information yields a value near 0.5, so "
                  "only values close to 1 across several models indicate a genuine "
                  "contribution. The last column counts how many of the four models "
                  "gave a probability of at least 0.9."))
        cross.to_csv(out("featsel_cross.csv"), index=False)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.4))
    ax = axes[0]
    models = [m for m in TREE_MODELS if m in set(res["model"])]
    xs = np.arange(len(models))
    w = 0.26
    for i, sname in enumerate((SET_ALL, SET_TOP, SET_CURVE)):
        vals = [res[(res.model == m) & (res.feature_set == sname)]["auc_oos"].mean()
                for m in models]
        ax.bar(xs + (i - 1) * w, vals, width=w, color=SC[sname], alpha=0.92,
               label=sname, edgecolor="white")
    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=8.5, rotation=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("out-of-sample AUC")
    ax.set_title("AUC by feature set", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    if not picks.empty:
        st = (picks.groupby("pretty")["times_selected"].sum()
              .sort_values(ascending=False).head(12))
        ax.barh(np.arange(len(st)), st.values, color="#a8501a", alpha=0.92)
        ax.set_yticks(np.arange(len(st)))
        ax.set_yticklabels(st.index, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("times selected into the top set")
        ax.set_title("Selection frequency across models and folds",
                     fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.axis("off")
    fig.tight_layout()
    tm.save_fig(fig, "fig_featsel.png")


def run(top_k=TOP_K, budget=BUDGET, save=True, verbose=True):
    print("=" * 78)
    print("Feature-set comparison on the iBond panel")
    print("=" * 78)
    panel, X33, X19, y, cols33, cols19 = load_joint(verbose)
    if y.sum() < 5:
        raise RuntimeError("too few positive months")
    res, picks, store, yy = evaluate_sets(panel, X33, X19, y, cols33, cols19,
                                          top_k, budget, verbose)
    dm = dm_between_sets(store, yy, verbose) if store else pd.DataFrame()
    champ = "XGBoost"
    if not res.empty:
        r_all = res[res.feature_set == SET_ALL]
        if not r_all.empty:
            champ = r_all.loc[r_all["auc_oos"].idxmax(), "model"]
    perm = permutation_importance_oos(panel, X33, y, cols33, champ, verbose=verbose)
    print()
    print("  crossed model x determinant randomisation ...")
    cross_raw, cross = cross_model_randomisation(panel, X33, y, cols33,
                                                 verbose=verbose)
    write_outputs(res, picks, top_k, dm, perm, cross)
    if save:
        con = sqlite3.connect(DB)
        res.to_sql(T_RESULT, con, if_exists="replace", index=False)
        if not picks.empty:
            picks.to_sql(T_PICKED, con, if_exists="replace", index=False)
        if dm is not None and not dm.empty:
            dm.to_sql(T_DM, con, if_exists="replace", index=False)
        if perm is not None and not perm.empty:
            perm.to_sql(T_PERM, con, if_exists="replace", index=False)
        if cross is not None and not cross.empty:
            cross.to_sql("cmdf_featsel_cross", con, if_exists="replace", index=False)
            cross_raw.to_sql("cmdf_featsel_cross_raw", con, if_exists="replace",
                             index=False)
        con.commit(); con.close()
    return res, picks, dm, perm, cross


def main():
    a = sys.argv
    k = int(a[a.index("--top") + 1]) if "--top" in a else TOP_K
    res, picks, dm, perm, cross = run(top_k=k, save="--no-save" not in a)
    print("\n" + "=" * 92)
    print("AUC BY FEATURE SET")
    print("=" * 92)
    piv = res.pivot_table(index="model", columns="feature_set", values="auc_oos")
    piv = piv.reindex([m for m in TREE_MODELS if m in piv.index])
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    if SET_ALL in piv.columns:
        print("\nchange relative to all 33:")
        for c in (SET_TOP, SET_CURVE):
            if c in piv.columns:
                d = (piv[c] - piv[SET_ALL])
                print(f"  {c:16s} mean {d.mean():+.3f}   "
                      f"better in {int((d > 0).sum())}/{len(d)} models")
    if not picks.empty:
        print("\nmost frequently selected determinants")
        st = (picks.groupby("pretty")["times_selected"].sum()
              .sort_values(ascending=False).head(10))
        for kk, v in st.items():
            print(f"  {kk:26s} {v}")
    print("\nArtefacts: tex_out/tab_featsel.tex, tab_featsel_delta.tex, "
          "tab_featsel_stability.tex, fig_featsel.png")
    print("Done.")


if __name__ == "__main__":
    main()
