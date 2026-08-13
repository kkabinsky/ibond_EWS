# -*- coding: utf-8 -*-
"""
firm_shock_panel.py -- per-issuer shock and threshold diagnostics for the GUI.

The heavy work is kept out of app.py: this module loads the real iBond panel, fits the
scoring model once, caches it, and then answers per-issuer questions cheaply. app.py
imports `build_panel_for_issuer` and renders whatever comes back.

WHAT AN ISSUER PANEL CONTAINS
    1  the issuer's PD path against the review-capacity threshold
    2  where the issuer sits on the two determinant pairs that move PD most, with the
       iso-PD contour drawn as the monitoring boundary
    3  the shock ladder: how far each determinant would have to move, on its own, to
       push this issuer across the threshold
    4  a headline row: current PD, percentile, margin, and the binding determinant

EVERY NUMBER COMES FROM ibond_33features_panel. Nothing is synthetic.

The scoring probabilities are out-of-fold under grouped cross-validation over issuer
identity, so an issuer's own rows never trained the model that scores it. The response
surfaces used for the contours are fitted on the full panel, which is the right choice
for describing a response rather than forecasting one.
"""
from __future__ import annotations

import base64
import io
import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_CACHE = {}
DEFAULT_WORKLOAD = 0.05          # detection saturates here; 10% costs twice the load
BACKGROUND = 120
GRID = 45
SCORER = "CatBoost"
SEED = 42


def _fig_b64(fig, dpi=118):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def load_state(workload=DEFAULT_WORKLOAD, force=False):
    """Load the panel, fit the model, cache everything. Safe to call repeatedly."""
    if not force and _CACHE.get("ready") and _CACHE.get("workload") == workload:
        return _CACHE

    import cmdf_tree_classify as cl
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier

    panel, X, y, cols = cl.load_panel(verbose=False)
    panel = panel.reset_index(drop=True)
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()

    # The threshold is read off the out-of-fold distribution, so every counterfactual
    # must be evaluated by the SAME model that produced those probabilities. Scoring a
    # shock with a model fitted on all rows puts the surface on a different scale from
    # the threshold: a full-fit model has memorised each issuer's own outcome and gives
    # a safe issuer a probability an order of magnitude below its out-of-fold value,
    # which made single-determinant shocks look as though they lowered PD.
    from sklearn.model_selection import StratifiedGroupKFold
    fold_of = np.full(len(A), -1, int)
    fold_models = {}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(A, yv, groups)):
        if yv[tr].sum() < 2:
            continue
        sck = StandardScaler().fit(A[tr])
        mk = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                                l2_leaf_reg=3.0, auto_class_weights="Balanced",
                                random_seed=SEED, verbose=0,
                                allow_writing_files=False).fit(sck.transform(A[tr]),
                                                               yv[tr])
        fold_models[k] = (sck, mk)
        fold_of[te] = k

    oof = np.full(len(A), np.nan)
    for k, (sck, mk) in fold_models.items():
        m = fold_of == k
        oof[m] = mk.predict_proba(sck.transform(A[m]))[:, 1]
    ok = np.isfinite(oof)

    # The threshold is a review-capacity rule, and the review queue is built from the
    # CURRENT cross-section: the team looks at today's issuers, not at every month
    # since 2007. Taking the quantile over all 16,686 issuer-months therefore sets the
    # line too low, because the latest month of each issuer is systematically riskier
    # than the historical pool it is being compared against. That put 12.6% of issuers
    # above a line meant to select 5%. Ranking within the latest cross-section makes
    # the flagged count match the capacity it was derived from.
    last_rows = (panel.assign(_i=np.arange(len(panel)))
                 .sort_values("month_dt").groupby("issuer_code").tail(1)["_i"]
                 .to_numpy())
    cross = oof[last_rows]
    cross = cross[np.isfinite(cross)]
    thr = float(np.quantile(cross, 1 - workload))

    # kept only for the population-level contour panels, never for a single issuer
    sc = StandardScaler().fit(A)
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0,
                            allow_writing_files=False).fit(sc.transform(A), yv)

    imp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tex_out", "importance_default_event.csv")
    # Two rankings are kept, not one. `feats` averages gain over the four learners and
    # drives the shock ladder. `xgb_feats` is XGBoost alone, which orders the panel
    # differently -- it puts TDTA and amihud_monthly_100 near the top where the average
    # does not -- so the pairs it suggests are not the pairs the average suggests.
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path)
        gains = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
        feats = [f for f in gains.index if f in cols][:8]
        xg = (imp[imp["model"] == "XGBoost"].groupby("feature")["gain"].mean()
              .sort_values(ascending=False))
        xgb_feats = [f for f in xg.index if f in cols][:6]
        xgb_gains = xg
    else:
        feats = list(cols[:8])
        gains = pd.Series(dtype=float)
        xgb_feats = feats[:6]
        xgb_gains = pd.Series(dtype=float)
    if not xgb_feats:
        xgb_feats = feats[:6]

    rng = np.random.default_rng(SEED)
    _CACHE.update(dict(
        ready=True, workload=workload, panel=panel, A=A, y=yv, cols=list(cols),
        idx={c: i for i, c in enumerate(cols)}, sd=A.std(0, ddof=1),
        med=np.median(A, axis=0), oof=oof, thr=thr, sc=sc, model=cb,
        fold_of=fold_of, fold_models=fold_models,
        cross=cross, last_rows=last_rows,
        feats=feats, gains=gains, xgb_feats=xgb_feats, xgb_gains=xgb_gains,
        BG=A[rng.choice(len(A), size=BACKGROUND, replace=False)],
        issuers=sorted(panel["issuer_code"].dropna().unique().tolist())))
    return _CACHE


def _pd_of(S, B):
    """Population-level surface, fitted on all rows. Not for single-issuer shocks."""
    return S["model"].predict_proba(S["sc"].transform(B))[:, 1]


def _pd_fold(S, B, row):
    """Score with the fold model that never saw the issuer owning `row`, so the value
    is on the same scale as the out-of-fold threshold."""
    k = int(S["fold_of"][row])
    if k not in S["fold_models"]:
        return _pd_of(S, B)
    sck, mk = S["fold_models"][k]
    return mk.predict_proba(sck.transform(B))[:, 1]


def issuer_summary(issuer, workload=DEFAULT_WORKLOAD):
    """Headline numbers for one issuer, all from the real panel."""
    S = load_state(workload)
    p = S["panel"]
    rows = p.index[p["issuer_code"] == issuer].to_numpy()
    if len(rows) == 0:
        return None
    rows = rows[np.argsort(p.loc[rows, "month_dt"].to_numpy())]
    last = rows[-1]
    cur = float(S["oof"][last])
    # percentile against the same cross-section the threshold came from, so the two
    # numbers in the table cannot disagree about where an issuer stands
    pct = float((S["cross"] < cur).mean() * 100)
    return dict(issuer=issuer, n_months=len(rows),
                first_month=str(p.loc[rows[0], "month"]),
                last_month=str(p.loc[last, "month"]),
                pd_now=cur, percentile=pct, threshold=S["thr"],
                margin=S["thr"] - cur, breach=bool(cur >= S["thr"]),
                event=bool(S["y"][rows].max() == 1),
                n_event_months=int(S["y"][rows].sum()))


def shock_ladder(issuer, workload=DEFAULT_WORKLOAD, max_sd=4.0, steps=41):
    """How far each determinant must move, alone, to take this issuer past the line."""
    S = load_state(workload)
    p = S["panel"]
    rows = p.index[p["issuer_code"] == issuer].to_numpy()
    if len(rows) == 0:
        return pd.DataFrame()
    last = rows[np.argmax(p.loc[rows, "month_dt"].to_numpy())]
    x0 = S["A"][last].copy()

    from sklearn.linear_model import LogisticRegression
    if "beta" not in S:
        S["beta"] = LogisticRegression(
            max_iter=5000, C=0.1, class_weight="balanced").fit(
            S["sc"].transform(S["A"]), S["y"]).coef_[0]
    beta = S["beta"]

    pd_now = float(_pd_fold(S, x0[None, :], last)[0])
    already = pd_now >= S["thr"]

    # Both directions are scanned, not just the one the global logistic coefficient
    # calls adverse. A boosted tree is not monotone, so the direction that raises PD
    # across the panel can lower it for a particular issuer at its own position; fixing
    # the direction in advance then reports "cannot reach" for issuers that in fact
    # cross the line by moving the other way.
    grid = np.linspace(-max_sd, max_sd, 2 * steps - 1)
    out = []
    for f in S["feats"]:
        j = S["idx"][f]
        B = np.tile(x0, (len(grid), 1))
        B[:, j] = x0[j] + grid * S["sd"][j]
        curve = _pd_fold(S, B, last)
        target = (curve < S["thr"]) if already else (curve >= S["thr"])
        hit = np.where(target)[0]
        if len(hit):
            k = hit[np.argmin(np.abs(grid[hit]))]     # smallest move that works
            sd_needed, val, direc = abs(grid[k]), float(B[k, j]), \
                ("up" if grid[k] > 0 else "down")
        else:
            sd_needed, val, direc = np.nan, np.nan, "--"
        # how close it gets, which is what matters when the line is out of reach
        best = float(curve.min() if already else curve.max())
        out.append(dict(feature=f, mode="recover" if already else "breach",
                        direction=direc, current=float(x0[j]),
                        sd_needed=sd_needed, value_needed=val,
                        pd_best=best, pd_now=pd_now,
                        closeness=(S["thr"] / best if already and best > 0
                                   else (best / S["thr"] if S["thr"] > 0 else np.nan)),
                        reachable=bool(len(hit))))
    d = pd.DataFrame(out)
    return d.sort_values(["reachable", "sd_needed"],
                         ascending=[False, True],
                         na_position="last").reset_index(drop=True)


def build_panel_for_issuer(issuer, workload=DEFAULT_WORKLOAD):
    """Return {'summary': dict, 'ladder': DataFrame, 'figures': {name: b64}}."""
    S = load_state(workload)
    summ = issuer_summary(issuer, workload)
    if summ is None:
        return None
    lad = shock_ladder(issuer, workload)
    p = S["panel"]
    rows = p.index[p["issuer_code"] == issuer].to_numpy()
    rows = rows[np.argsort(p.loc[rows, "month_dt"].to_numpy())]
    figs = {}

    # ---- 1. PD path -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    dts = pd.to_datetime(p.loc[rows, "month_dt"])
    vals = S["oof"][rows]
    ax.plot(dts, vals, lw=1.9, color="#1d4ed8", marker="o", ms=3.2)
    ax.axhline(S["thr"], color="#b91c1c", ls="--", lw=1.8,
               label=f"threshold at {workload:.0%} capacity ({S['thr']:.5f})")
    over = vals >= S["thr"]
    if over.any():
        ax.scatter(dts[over], vals[over], s=52, color="#b91c1c", zorder=5,
                   label=f"{int(over.sum())} months above the line")
    ev = S["y"][rows] == 1
    if ev.any():
        for d0 in dts[ev]:
            ax.axvline(d0, color="#16a34a", lw=1.2, alpha=0.6)
        ax.plot([], [], color="#16a34a", lw=1.2, label="recorded event window")
    ax.set_yscale("log")
    ax.set_ylabel("out-of-fold PD (log)")
    ax.set_title(f"{issuer}: PD path against the monitoring threshold",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.autofmt_xdate()
    figs["path"] = _fig_b64(fig)

    # ---- 2. four pairs with the boundary ----------------------------------
    # Two pairs come from the cross-model ranking, two from XGBoost's own ranking.
    # They are not the same pairs: XGBoost puts TDTA and amihud_monthly_100 high,
    # so it points at boundaries the averaged ranking never draws.
    last = rows[-1]
    x0 = S["A"][last]

    def _rank_pairs(pool):
        """Order candidate pairs by how much a joint one-SD move shifts THIS issuer's
        PD, so the panels drawn are the ones that actually move this firm."""
        scored = []
        for f1, f2 in itertools.combinations(pool, 2):
            j1, j2 = S["idx"][f1], S["idx"][f2]
            B = np.tile(x0, (2, 1))
            B[1, j1] += S["sd"][j1]
            B[1, j2] += S["sd"][j2]
            scored.append((abs(np.diff(_pd_fold(S, B, last))[0]), f1, f2))
        scored.sort(reverse=True)
        return [(f1, f2) for _, f1, f2 in scored]

    pick = _rank_pairs([f for f in S["feats"][:5] if f in S["idx"]])[:2]
    seen = {frozenset(q) for q in pick}
    xpick = []
    for q in _rank_pairs([f for f in S["xgb_feats"][:5] if f in S["idx"]]):
        if frozenset(q) in seen:
            continue
        xpick.append(q)
        seen.add(frozenset(q))
        if len(xpick) == 2:
            break

    panels = ([(f1, f2, "all four models, mean gain") for f1, f2 in pick]
              + [(f1, f2, "XGBoost gain") for f1, f2 in xpick])

    nrow = int(np.ceil(len(panels) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(11.6, 4.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (f1, f2, src) in zip(axes, panels):
        j1, j2 = S["idx"][f1], S["idx"][f2]
        lo1, hi1 = np.percentile(S["A"][:, j1], [1, 99])
        lo2, hi2 = np.percentile(S["A"][:, j2], [1, 99])
        G1, G2 = np.meshgrid(np.linspace(lo1, hi1, GRID),
                             np.linspace(lo2, hi2, GRID))
        big = np.tile(S["BG"], (G1.size, 1))
        big[:, j1] = np.repeat(G1.ravel(), len(S["BG"]))
        big[:, j2] = np.repeat(G2.ravel(), len(S["BG"]))
        P = _pd_of(S, big).reshape(G1.size, len(S["BG"])).mean(1).reshape(G1.shape)
        im = ax.contourf(G1, G2, P, levels=22, cmap="viridis")
        if P.min() < S["thr"] < P.max():
            cs = ax.contour(G1, G2, P, levels=[S["thr"]], colors="#f8fafc",
                            linewidths=2.4)
            ax.clabel(cs, fmt={S["thr"]: "threshold"}, fontsize=7)
        ax.scatter(S["A"][:, j1], S["A"][:, j2], s=3, color="#cbd5e1", alpha=0.20)
        ax.plot(S["A"][rows, j1], S["A"][rows, j2], color="#f59e0b", lw=1.4,
                alpha=0.9, zorder=6, label=f"{issuer} path")
        ax.scatter([x0[j1]], [x0[j2]], s=140, marker="*", color="#b91c1c",
                   edgecolors="white", linewidth=1.0, zorder=7, label="latest month")
        ax.set_xlabel(f1, fontsize=9)
        ax.set_ylabel(f2, fontsize=9)
        ax.set_xlim(lo1, hi1)
        ax.set_ylim(lo2, hi2)
        ax.set_title(f1 + " x " + f2 + "\n" + "ranked by " + src, fontsize=10,
                     fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{issuer}: position against the monitoring boundary",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95 if nrow > 1 else 0.93])
    figs["pairs"] = _fig_b64(fig)

    # ---- 3. shock ladder --------------------------------------------------
    lp = lad.copy()
    recover = (not lp.empty) and lp["mode"].iloc[0] == "recover"
    n_reach = int(lp.reachable.sum())
    max_sd = 4.0
    verb = ("favourable movement to come back under the line" if recover
            else "adverse movement to cross the line")

    if n_reach == 0:
        # Nothing reaches the line on its own. Plotting only the reachable bars would
        # leave an empty frame, which reads as a failure rather than as the finding it
        # is; the informative quantity is how close each determinant gets.
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        vals = (lp.closeness * 100).to_numpy(float)
        ax.barh(np.arange(len(lp)), vals, color="#64748b", alpha=0.85)
        for i, (_, r) in enumerate(lp.iterrows()):
            ax.text(vals[i], i, f"  PD reaches {r.pd_best:.6f}", va="center",
                    fontsize=8)
        ax.axvline(100, color="#b91c1c", lw=2.0)
        ax.text(100, -0.7, " the line", color="#b91c1c", fontsize=8.5,
                fontweight="bold")
        ax.set_yticks(np.arange(len(lp)))
        ax.set_yticklabels(lp.feature, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("closest approach to the threshold, as a percentage of it")
        ax.set_title(f"{issuer}: no determinant crosses the line alone within "
                     f"{max_sd:.0f} SD\ncurrent PD {lad.pd_now.iloc[0]:.6f}, "
                     f"threshold {S['thr']:.6f}; bars show how close each one gets",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
    else:
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        col = "#16a34a" if recover else "#b91c1c"
        vals, cols, hatch = [], [], []
        for _, r in lp.iterrows():
            if r.reachable:
                vals.append(r.sd_needed); cols.append(col); hatch.append("")
            else:
                vals.append(max_sd); cols.append("#cbd5e1"); hatch.append("//")
        bars = ax.barh(np.arange(len(lp)), vals, color=cols, alpha=0.88)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h)
        for i, (_, r) in enumerate(lp.iterrows()):
            if r.reachable:
                ax.text(r.sd_needed, i, f"  {r.sd_needed:.2f} SD ({r.direction})",
                        va="center", fontsize=8)
            else:
                ax.text(max_sd, i, f"  > {max_sd:.0f} SD, PD reaches "
                                   f"{r.pd_best:.6f}", va="center", fontsize=7.5,
                        color="#475569")
        ax.set_yticks(np.arange(len(lp)))
        ax.set_yticklabels(lp.feature, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("standard deviations of movement needed, "
                      "one determinant at a time")
        ax.set_title(f"{issuer}: shock ladder, {verb}\n"
                     f"{n_reach} of {len(lp)} determinants can do it alone; "
                     f"hatched bars cannot within {max_sd:.0f} SD",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
    figs["ladder"] = _fig_b64(fig)

    return dict(summary=summ, ladder=lad, figures=figs,
                pairs=[f"{a} x {b}" for a, b, _ in panels],
                pairs_mean=[f"{a} x {b}" for a, b in pick],
                pairs_xgb=[f"{a} x {b}" for a, b in xpick])


def issuer_table(workload=DEFAULT_WORKLOAD, limit=None):
    """Ranking of every issuer by current PD, for the overview grid."""
    S = load_state(workload)
    p = S["panel"]
    last = (p.sort_values("month_dt").groupby("issuer_code").tail(1).index.to_numpy())
    d = pd.DataFrame(dict(
        issuer=p.loc[last, "issuer_code"].values,
        month=p.loc[last, "month"].values,
        pd_now=S["oof"][last]))
    d["percentile"] = [float((S["cross"] < v).mean() * 100) for v in d.pd_now]
    d["margin"] = S["thr"] - d.pd_now
    # The wording matters. These bands say where an issuer sits relative to the review
    # threshold; they do not say the issuer defaulted. Only 8 issuers in this panel
    # ever recorded an event, while far more sit above the threshold at any capacity,
    # so labelling the top band "default" would be wrong by a wide margin. The names
    # match the bands the rest of the application already uses.
    d["status"] = np.where(d.pd_now >= S["thr"], "HIGH RISK",
                           np.where(d.percentile >= 100 * (1 - workload) - 2,
                                    "WATCH", "OK"))
    d = d.sort_values("pd_now", ascending=False).reset_index(drop=True)
    return d.head(limit) if limit else d
