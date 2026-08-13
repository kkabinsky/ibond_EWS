# -*- coding: utf-8 -*-
"""
triple_shock_pd.py -- extend the shock analysis from pairs to triples, rank every
three-determinant combination by its effect on the default probability, and write the
result to an Excel workbook.

WHY THREE IS NOT JUST "ONE MORE"
    With two determinants the joint effect splits into two single effects and one
    interaction. With three it splits into three singles, three pairwise interactions
    and one genuine three-way term. Writing D for a shock effect,

        D_123 = D_1 + D_2 + D_3
              + I_12 + I_13 + I_23
              + I_123

    where each pairwise interaction is I_ab = D_ab - D_a - D_b and the three-way term
    is whatever the pairwise decomposition fails to explain,

        I_123 = D_123 - (D_1 + D_2 + D_3) - (I_12 + I_13 + I_23).

    This is the standard ANOVA-style decomposition. A large I_123 means the three
    determinants matter together in a way no pair of them reveals, which is the only
    thing a triple analysis can add over the pairwise one.

HOW TO SEE FOUR DIMENSIONS
    Three determinants plus the probability is four dimensions, so a single surface is
    not available. The figure shows the three practical alternatives applied to the
    leading triple: a coloured 3-D scatter, a set of 2-D surfaces sliced at fixed
    levels of the third determinant, and the decomposition itself as a bar chart.

RUN
    python triple_shock_pd.py
    python triple_shock_pd.py --top 10
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

import cmdf_tree_classify as cl
import cmdf_tree_models as tm

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out

TOP = 10
SHOCK_SD = 1.0
WORKLOAD = 0.02
SEED = 42
GRID = 26
SLICES = 3


def main():
    top = TOP
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])

    print("=" * 100)
    print("Three-determinant shocks: ranking, interaction decomposition, Excel output")
    print("=" * 100)
    panel, X, y, cols = cl.load_panel(verbose=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    sd = A.std(0, ddof=1)

    imp = pd.read_csv(out("importance_default_event.csv"))
    gains = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
    idx = {c: i for i, c in enumerate(cols)}
    chosen = [f for f in gains.index if f in idx][:top]
    print(f"\n  determinants considered: {', '.join(chosen)}")
    print(f"  combinations: {len(list(itertools.combinations(chosen,3)))} triples, "
          f"{len(list(itertools.combinations(chosen,2)))} pairs, {top} singles")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier
    sc = StandardScaler().fit(A)
    As = sc.transform(A)

    lg = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced").fit(As, yv)
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0,
                            allow_writing_files=False).fit(As, yv)
    models = {"Logistic": lg, "CatBoost": cb}
    beta = lg.coef_[0]
    direction = {f: (1.0 if beta[idx[f]] >= 0 else -1.0) for f in chosen}

    def pd_of(m, B):
        return models[m].predict_proba(sc.transform(B))[:, 1]

    def shocked(feats):
        B = A.copy()
        for f in feats:
            j = idx[f]
            B[:, j] = B[:, j] + direction[f] * SHOCK_SD * sd[j]
        return B

    base = {m: pd_of(m, A) for m in models}
    thr = {m: np.quantile(base[m], 1 - WORKLOAD) for m in models}
    print("\n  baseline mean PD:  " +
          "   ".join(f"{m} {base[m].mean():.5f}" for m in models))

    D1 = {m: {} for m in models}
    for f in chosen:
        B = shocked([f])
        for m in models:
            D1[m][f] = float((pd_of(m, B) - base[m]).mean())

    I2 = {m: {} for m in models}
    pair_rows = []
    for a, b in itertools.combinations(chosen, 2):
        B = shocked([a, b])
        for m in models:
            d_ab = float((pd_of(m, B) - base[m]).mean())
            I2[m][(a, b)] = d_ab - D1[m][a] - D1[m][b]
            pair_rows.append(dict(model=m, f1=a, f2=b, joint=d_ab,
                                  interaction=I2[m][(a, b)]))

    print("\n  scoring triples ...")
    rows = []
    for a, b, c in itertools.combinations(chosen, 3):
        B = shocked([a, b, c])
        for m in models:
            p = pd_of(m, B)
            d_abc = float((p - base[m]).mean())
            s1 = D1[m][a] + D1[m][b] + D1[m][c]
            s2 = I2[m][(a, b)] + I2[m][(a, c)] + I2[m][(b, c)]
            alarm0 = float((base[m] >= thr[m]).mean())
            alarm1 = float((p >= thr[m]).mean())
            rows.append(dict(
                model=m, f1=a, f2=b, f3=c,
                d_f1=D1[m][a], d_f2=D1[m][b], d_f3=D1[m][c],
                sum_singles=s1,
                I_12=I2[m][(a, b)], I_13=I2[m][(a, c)], I_23=I2[m][(b, c)],
                sum_pairwise=s2,
                joint=d_abc, three_way=d_abc - s1 - s2,
                pct_of_base=100 * d_abc / base[m].mean(),
                alarm_base_pct=100 * alarm0, alarm_shock_pct=100 * alarm1,
                alarm_delta_pp=100 * (alarm1 - alarm0)))
    d = pd.DataFrame(rows)

    for m in models:
        s = d[d.model == m].sort_values("joint", ascending=False)
        print("\n" + "=" * 100)
        print(f"{m}: top triples by joint effect on mean PD")
        print("=" * 100)
        print(f"  {'triple':48s} {'joint':>9} {'singles':>9} {'pairwise':>9} "
              f"{'3-way':>9} {'%base':>8}")
        for _, r in s.head(10).iterrows():
            name = f"{r.f1[:14]}+{r.f2[:14]}+{r.f3[:14]}"
            print(f"  {name:48s} {r.joint:>9.5f} {r.sum_singles:>9.5f} "
                  f"{r.sum_pairwise:>9.5f} {r.three_way:>+9.5f} "
                  f"{r.pct_of_base:>7.0f}%")
        b3 = s.reindex(s.three_way.abs().sort_values(ascending=False).index).head(3)
        print(f"  largest three-way term:")
        for _, r in b3.iterrows():
            print(f"    {r.f1}+{r.f2}+{r.f3}  I123 = {r.three_way:+.5f}")

    # ------------------------------------------------------------- Excel ----
    xl = out("triple_shock_pd.xlsx")
    singles = pd.DataFrame([dict(model=m, feature=f, delta_pd=D1[m][f],
                                 gain=gains.get(f, np.nan),
                                 adverse=("increase" if direction[f] > 0 else
                                          "decrease"))
                            for m in models for f in chosen])
    notes = pd.DataFrame({
        "item": ["panel", "issuer-months", "issuers", "events", "prevalence",
                 "shock size", "adverse direction", "workload for alarm rate",
                 "surface model", "horizon", "decomposition", "caveat"],
        "value": ["ibond_33features_panel", f"{len(A):,}",
                  f"{panel['issuer_code'].nunique()}", f"{int(yv.sum())}",
                  f"{yv.mean():.4%}", f"{SHOCK_SD:.0f} standard deviation",
                  "sign of the fitted logistic coefficient",
                  f"{WORKLOAD:.0%} of issuer-months",
                  "CatBoost and Logistic, both fitted on the full panel",
                  "three months",
                  "D_123 = sum singles + sum pairwise interactions + three-way term",
                  "models fitted on all rows; this is a sensitivity analysis of the "
                  "fitted surface, not an out-of-sample result"]})
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        for m in models:
            (d[d.model == m].sort_values("joint", ascending=False)
             .to_excel(w, sheet_name=f"triples {m}"[:31], index=False))
        pd.DataFrame(pair_rows).sort_values(["model", "joint"], ascending=[True, False]) \
            .to_excel(w, sheet_name="pairs", index=False)
        singles.to_excel(w, sheet_name="singles", index=False)
        notes.to_excel(w, sheet_name="method", index=False)
    print(f"\n  wrote {xl}")

    # ------------------------------------------------------------- figure ---
    best = d[d.model == "CatBoost"].sort_values("joint", ascending=False).iloc[0]
    f1, f2, f3 = best.f1, best.f2, best.f3
    j1, j2, j3 = idx[f1], idx[f2], idx[f3]
    print(f"\n  figure uses the leading CatBoost triple: {f1}, {f2}, {f3}")

    rng = np.random.default_rng(SEED)
    take = rng.choice(len(A), size=2500, replace=False)
    p_pt = base["CatBoost"][take]

    fig = plt.figure(figsize=(16.5, 5.6))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    # PD is extremely skewed, so almost every point takes the same colour on a linear
    # scale; the percentile rank spreads the palette over the observed ordering
    rank = pd.Series(p_pt).rank(pct=True).to_numpy() * 100
    sca = ax.scatter(A[take, j1], A[take, j2], A[take, j3], c=rank, cmap="viridis",
                     s=9, alpha=0.60, edgecolors="none")
    ax.set_xlabel(f1, fontsize=8); ax.set_ylabel(f2, fontsize=8)
    ax.set_zlabel(f3, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=22, azim=-128)
    ax.set_title("A. three determinants as axes,\nPD as colour", fontsize=10,
                 fontweight="bold")
    fig.colorbar(sca, ax=ax, fraction=0.03, pad=0.10,
                 label="PD percentile")

    lo1, hi1 = np.percentile(A[:, j1], [2, 98])
    lo2, hi2 = np.percentile(A[:, j2], [2, 98])
    qs = np.percentile(A[:, j3], [15, 50, 85])
    G1, G2 = np.meshgrid(np.linspace(lo1, hi1, GRID), np.linspace(lo2, hi2, GRID))
    BG = A[rng.choice(len(A), size=200, replace=False)]
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    cmaps = ["Blues", "Oranges", "Reds"]
    for q, cm in zip(qs, cmaps):
        big = np.tile(BG, (G1.size, 1))
        big[:, j1] = np.repeat(G1.ravel(), len(BG))
        big[:, j2] = np.repeat(G2.ravel(), len(BG))
        big[:, j3] = q
        P = pd_of("CatBoost", big).reshape(G1.size, len(BG)).mean(1).reshape(G1.shape)
        ax.plot_surface(G1, G2, P, cmap=cm, alpha=0.80, linewidth=0,
                        antialiased=False)
        ax.text(hi1, hi2, P.max(), f"  {f3}={q:.2f}", fontsize=7.5)
    ax.set_xlabel(f1, fontsize=8); ax.set_ylabel(f2, fontsize=8)
    ax.set_zlabel("PD", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=24, azim=-130)
    ax.set_title(f"B. surfaces over ({f1}, {f2})\nsliced at three levels of {f3}",
                 fontsize=10, fontweight="bold")

    ax = fig.add_subplot(1, 3, 3)
    top10 = d[d.model == "CatBoost"].sort_values("joint", ascending=False).head(10)
    lbl = [f"{r.f1}\n+{r.f2}\n+{r.f3}" for _, r in top10.iterrows()]
    ys = np.arange(len(top10))
    # the three-way term is often negative, and stacked bars misrepresent a negative
    # component by drawing it back over the others; grouped bars keep the sign honest
    h = 0.26
    ax.barh(ys - h, top10.sum_singles, height=h, color="#3b82f6",
            label="sum of singles")
    ax.barh(ys, top10.sum_pairwise, height=h, color="#f59e0b",
            label="pairwise interactions")
    ax.barh(ys + h, top10.three_way, height=h, color="#b91c1c",
            label="three-way term")
    ax.scatter(top10.joint, ys, s=46, marker="D", color="#111827", zorder=6,
               label="joint effect")
    for i, v in enumerate(top10.joint):
        ax.text(v, i - 0.42, f"{v:.4f}", va="center", ha="center", fontsize=7,
                fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(lbl, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="#334155", lw=0.9)
    ax.set_xlabel(r"change in mean PD")
    ax.set_title("C. decomposition of the ten\nlargest triples (CatBoost)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="x")

    fig.suptitle("From two determinants to three: how to look at four dimensions, and "
                 "what the third determinant adds",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUTDIR, "fig_triple_shock.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    d.to_csv(out("triple_shock_pd.csv"), index=False)
    con = sqlite3.connect(DB)
    d.to_sql("cmdf_triple_shock", con, if_exists="replace", index=False)
    con.commit(); con.close()
    print(f"  wrote {p}")
    print("  wrote tex_out/triple_shock_pd.csv, table cmdf_triple_shock")


if __name__ == "__main__":
    main()
