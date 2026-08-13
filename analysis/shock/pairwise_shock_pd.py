# -*- coding: utf-8 -*-
"""
pairwise_shock_pd.py -- shock the real determinants two at a time and measure what
happens to the predicted default probability across the whole 33-determinant panel.

WHAT IS SHOCKED
    Determinants are ranked by mean gain against the observed default event, and the
    leading ones are shocked in pairs. Each shock moves a determinant by one
    within-panel standard deviation in its ADVERSE direction, where adverse is read off
    the sign of the fitted logistic coefficient rather than assumed, so the direction
    is a property of the data and not of the analyst.

HORIZON
    The 33-determinant panel carries a three-month event label, so the quantity being
    moved is PD over three months. It is written PD throughout rather than PD12 to
    avoid implying a twelve-month horizon this panel cannot support.

TWO MODELS, ON PURPOSE
    Logistic     the index is additive, so a joint shock can differ from the sum of two
                 single shocks only through the curvature of the link.
    CatBoost     tree ensembles represent interactions directly, so any remaining gap
                 between the joint effect and the sum of the parts is an interaction
                 the linear model cannot express.
    Reporting both separates curvature from genuine interaction.

SCOPE
    This is a sensitivity analysis of a fitted response surface, not a validation
    exercise: both models are fitted once on the full panel, because the question is
    how the fitted surface responds to a perturbation, not how well it generalises.

RUN
    python pairwise_shock_pd.py
    python pairwise_shock_pd.py --top 6
"""
from __future__ import annotations

import itertools
import os
import sqlite3
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cmdf_tree_classify as cl
import cmdf_tree_models as tm

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out

TOP = 8
SHOCK_SD = 1.0
WORKLOAD = 0.02
SEED = 42


def esc(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def ranked_features(cols):
    """Mean gain across the four tree models, restricted to determinants present."""
    p = out("importance_default_event.csv")
    if not os.path.exists(p):
        raise SystemExit("run make_importance_default.py first")
    imp = pd.read_csv(p)
    r = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
    return [f for f in r.index if f in cols], r


def fit_models(A, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(A)
    As = sc.transform(A)

    lg = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced")
    lg.fit(As, y)

    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0, allow_writing_files=False)
    cb.fit(As, y)
    return sc, {"Logistic": lg, "CatBoost": cb}


def pd_of(model, sc, A):
    return model.predict_proba(sc.transform(A))[:, 1]


def main():
    top = TOP
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])

    print("=" * 100)
    print("Pairwise determinant shocks and the response of the predicted default "
          "probability")
    print("=" * 100)
    panel, X, y, cols = cl.load_panel(verbose=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    sd = A.std(0, ddof=1)

    order, gains = ranked_features(cols)
    idx = {c: i for i, c in enumerate(cols)}
    chosen = order[:top]
    print(f"\n  determinants shocked (top {top} by mean gain):")
    for i, f in enumerate(chosen, 1):
        print(f"    {i}. {f:26s} gain {gains[f]:.4f}   sd {sd[idx[f]]:.4f}")

    sc, models = fit_models(A, yv)

    # adverse direction from the logistic coefficients: +1 if raising the determinant
    # raises PD, -1 otherwise. Coefficients are on the standardised scale.
    beta = models["Logistic"].coef_[0]
    direction = {f: (1.0 if beta[idx[f]] >= 0 else -1.0) for f in chosen}
    print("\n  adverse direction inferred from the fitted logistic coefficients:")
    for f in chosen:
        arrow = "increase" if direction[f] > 0 else "decrease"
        print(f"    {f:26s} beta {beta[idx[f]]:+.4f}  ->  adverse = {arrow}")

    base = {m: pd_of(models[m], sc, A) for m in models}
    thr = {m: np.quantile(base[m], 1 - WORKLOAD) for m in models}
    print("\n  baseline mean PD:  " +
          "   ".join(f"{m} {base[m].mean():.5f}" for m in models))

    def shocked(feats):
        B = A.copy()
        for f in feats:
            j = idx[f]
            B[:, j] = B[:, j] + direction[f] * SHOCK_SD * sd[j]
        return B

    single = {}
    for f in chosen:
        Bf = shocked([f])
        single[f] = {m: pd_of(models[m], sc, Bf) - base[m] for m in models}

    rows = []
    for f1, f2 in itertools.combinations(chosen, 2):
        Bj = shocked([f1, f2])
        for m in models:
            joint = pd_of(models[m], sc, Bj) - base[m]
            d1 = single[f1][m].mean()
            d2 = single[f2][m].mean()
            alarm0 = (base[m] >= thr[m]).mean()
            alarm1 = ((base[m] + joint) >= thr[m]).mean()
            rows.append(dict(model=m, f1=f1, f2=f2,
                             d1=d1, d2=d2, joint=joint.mean(),
                             interaction=joint.mean() - (d1 + d2),
                             pct_chg=100 * joint.mean() / base[m].mean(),
                             alarm_base=100 * alarm0, alarm_shock=100 * alarm1,
                             alarm_delta_pp=100 * (alarm1 - alarm0)))
    d = pd.DataFrame(rows)

    for m in models:
        s = d[d.model == m].sort_values("joint", ascending=False)
        print("\n" + "=" * 100)
        print(f"{m}: pairs ranked by the joint effect on mean PD")
        print("=" * 100)
        print(f"  {'pair':40s} {'d1':>9} {'d2':>9} {'joint':>9} "
              f"{'interact':>9} {'%chg':>8} {'alarm pp':>9}")
        for _, r in s.head(12).iterrows():
            print(f"  {r.f1[:18]+' + '+r.f2[:18]:40s} {r.d1:>9.5f} {r.d2:>9.5f} "
                  f"{r.joint:>9.5f} {r.interaction:>+9.5f} {r.pct_chg:>7.1f}% "
                  f"{r.alarm_delta_pp:>+8.2f}")

    print("\n=== single-determinant effects, for reference ===")
    print(f"  {'determinant':26s} " +
          "  ".join(f"{m:>12s}" for m in models))
    for f in chosen:
        print(f"  {f:26s} " +
              "  ".join(f"{single[f][m].mean():>12.5f}" for m in models))

    # ------------------------------------------------------------- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))
    for ax, m in zip(axes, models):
        M = np.full((top, top), np.nan)
        for _, r in d[d.model == m].iterrows():
            i, j = chosen.index(r.f1), chosen.index(r.f2)
            M[i, j] = M[j, i] = r.joint
        for i, f in enumerate(chosen):
            M[i, i] = single[f][m].mean()
        vmax = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(top)); ax.set_yticks(range(top))
        ax.set_xticklabels(chosen, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(chosen, fontsize=8)
        for i in range(top):
            for j in range(top):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]*1000:.1f}", ha="center", va="center",
                            fontsize=6.8,
                            color="white" if abs(M[i, j]) > vmax * 0.55 else "#111")
        ax.set_title(f"{m}: mean change in PD  ($\\times 10^{{-3}}$)\n"
                     f"diagonal = single shock, off-diagonal = joint",
                     fontsize=10.5, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.suptitle(f"Adverse shock of {SHOCK_SD:.0f} standard deviation, applied to the "
                 f"top {top} determinants singly and in pairs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUTDIR, "fig_pairwise_shock.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    d.to_csv(out("pairwise_shock_pd.csv"), index=False)
    con = sqlite3.connect(DB)
    d.to_sql("cmdf_pairwise_shock", con, if_exists="replace", index=False)
    con.commit(); con.close()
    print(f"\n  wrote {p}")
    print("  wrote tex_out/pairwise_shock_pd.csv, table cmdf_pairwise_shock")


if __name__ == "__main__":
    main()
