# -*- coding: utf-8 -*-
"""
roe_triple_figures.py -- one figure per three-determinant combination containing ROE,
saved as a separate JPG named after its determinants.

WHY ROE IS HELD FIXED
    Fixing one determinant across every figure is what makes the set comparable: the
    panels differ only in the two determinants added to ROE, so any change in the
    surface or in the decomposition is attributable to those two rather than to a
    change of all three at once.

EACH FIGURE
    A  the three determinants as axes with the default probability as colour, on the
       percentile scale because PD is far too skewed for a linear palette
    B  the (ROE, second determinant) surface drawn at three levels of the third, which
       is how a fourth dimension is read off a three-dimensional plot
    C  the ANOVA-style decomposition for that triple: three single effects, three
       pairwise interactions, and the three-way remainder, with the joint effect marked

FILE NAMES
    tex_out/triples_roe/pd3_<rank>_ROE__<second>__<third>.jpg
    The rank prefix keeps the directory listing in the same order as the ranking table,
    so the strongest combinations are found without opening anything.

RUN
    python roe_triple_figures.py
    python roe_triple_figures.py --n 40 --grid 22
"""
from __future__ import annotations

import itertools
import os
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
out = tm.out
FIGDIR = os.path.join(OUTDIR, "triples_roe")

ANCHOR = "ROE"
N_FIGS = 40
TOP = 11                   # anchor + 10 partners -> C(10,2) = 45 candidate triples
GRID = 22
BACKGROUND = 120
SHOCK_SD = 1.0
SEED = 42
DPI = 130


def main():
    n_figs, grid = N_FIGS, GRID
    if "--n" in sys.argv:
        n_figs = int(sys.argv[sys.argv.index("--n") + 1])
    if "--grid" in sys.argv:
        grid = int(sys.argv[sys.argv.index("--grid") + 1])
    os.makedirs(FIGDIR, exist_ok=True)

    print("=" * 96)
    print(f"One JPG per triple containing {ANCHOR}")
    print("=" * 96)
    panel, X, y, cols = cl.load_panel(verbose=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    sd = A.std(0, ddof=1)

    imp = pd.read_csv(out("importance_default_event.csv"))
    gains = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
    idx = {c: i for i, c in enumerate(cols)}
    ranked = [f for f in gains.index if f in idx]
    if ANCHOR not in ranked:
        raise SystemExit(f"{ANCHOR} not in the panel")
    partners = [f for f in ranked if f != ANCHOR][:TOP - 1]
    print(f"\n  anchor   : {ANCHOR}")
    print(f"  partners : {', '.join(partners)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier
    sc = StandardScaler().fit(A)
    As = sc.transform(A)
    beta = LogisticRegression(max_iter=5000, C=0.1,
                              class_weight="balanced").fit(As, yv).coef_[0]
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0,
                            allow_writing_files=False).fit(As, yv)

    def pdf(B):
        return cb.predict_proba(sc.transform(B))[:, 1]

    def direction(f):
        j = idx[f]
        return 1.0 if beta[j] >= 0 else -1.0

    def shocked(feats):
        B = A.copy()
        for f in feats:
            j = idx[f]
            B[:, j] = B[:, j] + direction(f) * SHOCK_SD * sd[j]
        return B

    base = pdf(A)
    base_mean = float(base.mean())
    print(f"  baseline mean PD (CatBoost): {base_mean:.5f}")

    feats_all = [ANCHOR] + partners
    D1 = {f: float((pdf(shocked([f])) - base).mean()) for f in feats_all}
    I2 = {}
    for a, b in itertools.combinations(feats_all, 2):
        d_ab = float((pdf(shocked([a, b])) - base).mean())
        I2[frozenset((a, b))] = d_ab - D1[a] - D1[b]

    rows = []
    for b, c in itertools.combinations(partners, 2):
        trio = (ANCHOR, b, c)
        d_abc = float((pdf(shocked(list(trio))) - base).mean())
        s1 = sum(D1[f] for f in trio)
        s2 = (I2[frozenset((ANCHOR, b))] + I2[frozenset((ANCHOR, c))]
              + I2[frozenset((b, c))])
        rows.append(dict(f1=ANCHOR, f2=b, f3=c, joint=d_abc, sum_singles=s1,
                         sum_pairwise=s2, three_way=d_abc - s1 - s2,
                         pct_of_base=100 * d_abc / base_mean,
                         I_12=I2[frozenset((ANCHOR, b))],
                         I_13=I2[frozenset((ANCHOR, c))],
                         I_23=I2[frozenset((b, c))],
                         d_f1=D1[ANCHOR], d_f2=D1[b], d_f3=D1[c]))
    d = pd.DataFrame(rows).sort_values("joint", ascending=False).reset_index(drop=True)
    d = d.head(n_figs)
    print(f"\n  {len(d)} triples selected, ranked by joint effect")
    print(f"  {'rank':>4}  {'triple':46s} {'joint':>9} {'%base':>8} {'3-way':>10}")
    for i, r in d.head(10).iterrows():
        print(f"  {i+1:>4}  {r.f1+' + '+r.f2+' + '+r.f3:46s} {r.joint:>9.5f} "
              f"{r.pct_of_base:>7.0f}% {r.three_way:>+10.5f}")

    rng = np.random.default_rng(SEED)
    take = rng.choice(len(A), size=2200, replace=False)
    rank_col = pd.Series(base[take]).rank(pct=True).to_numpy() * 100
    BG = A[rng.choice(len(A), size=BACKGROUND, replace=False)]

    print(f"\n  rendering {len(d)} JPGs into {FIGDIR} ...")
    for i, r in d.iterrows():
        f1, f2, f3 = r.f1, r.f2, r.f3
        j1, j2, j3 = idx[f1], idx[f2], idx[f3]

        fig = plt.figure(figsize=(15.6, 5.1))

        ax = fig.add_subplot(1, 3, 1, projection="3d")
        s = ax.scatter(A[take, j1], A[take, j2], A[take, j3], c=rank_col,
                       cmap="viridis", s=8, alpha=0.55, edgecolors="none")
        ax.set_xlabel(f1, fontsize=8); ax.set_ylabel(f2, fontsize=8)
        ax.set_zlabel(f3, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=22, azim=-128)
        ax.set_title("A. determinants as axes, PD percentile as colour",
                     fontsize=9.5, fontweight="bold")
        fig.colorbar(s, ax=ax, fraction=0.028, pad=0.10, label="PD pct")

        ax = fig.add_subplot(1, 3, 2, projection="3d")
        lo1, hi1 = np.percentile(A[:, j1], [2, 98])
        lo2, hi2 = np.percentile(A[:, j2], [2, 98])
        G1, G2 = np.meshgrid(np.linspace(lo1, hi1, grid),
                             np.linspace(lo2, hi2, grid))
        qs = np.percentile(A[:, j3], [15, 50, 85])
        for q, cm in zip(qs, ["Blues", "Oranges", "Reds"]):
            big = np.tile(BG, (G1.size, 1))
            big[:, j1] = np.repeat(G1.ravel(), len(BG))
            big[:, j2] = np.repeat(G2.ravel(), len(BG))
            big[:, j3] = q
            P = pdf(big).reshape(G1.size, len(BG)).mean(1).reshape(G1.shape)
            ax.plot_surface(G1, G2, P, cmap=cm, alpha=0.78, linewidth=0,
                            antialiased=False)
            ax.text(hi1, hi2, P.max(), f"  {f3}={q:.2f}", fontsize=7)
        ax.set_xlabel(f1, fontsize=8); ax.set_ylabel(f2, fontsize=8)
        ax.set_zlabel("PD", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=24, azim=-130)
        ax.set_title(f"B. surface over ({f1}, {f2}) at three levels of {f3}",
                     fontsize=9.5, fontweight="bold")

        ax = fig.add_subplot(1, 3, 3)
        names = [f"$D$ {f1}", f"$D$ {f2}", f"$D$ {f3}",
                 f"$I$ {f1}-{f2}", f"$I$ {f1}-{f3}", f"$I$ {f2}-{f3}",
                 "$I$ three-way"]
        vals = [r.d_f1, r.d_f2, r.d_f3, r.I_12, r.I_13, r.I_23, r.three_way]
        colr = ["#3b82f6"] * 3 + ["#f59e0b"] * 3 + ["#b91c1c"]
        ys = np.arange(len(vals))
        ax.barh(ys, vals, color=colr, edgecolor="white")
        for yy, v in zip(ys, vals):
            ax.text(v, yy, f" {v:+.5f}", va="center", fontsize=7.5,
                    ha="left" if v >= 0 else "right")
        ax.axvline(0, color="#334155", lw=1.0)
        ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("contribution to the change in mean PD", fontsize=8.5)
        ax.set_title(f"C. decomposition   joint = {r.joint:+.5f} "
                     f"({r.pct_of_base:+.0f}% of base)",
                     fontsize=9.5, fontweight="bold")
        ax.grid(alpha=0.25, axis="x")
        lim = max(abs(min(vals)), abs(max(vals))) * 1.45
        ax.set_xlim(-lim, lim)

        fig.suptitle(f"#{i+1}   {f1} + {f2} + {f3}      "
                     f"baseline mean PD {base_mean:.5f}  ->  "
                     f"{base_mean + r.joint:.5f}",
                     fontsize=12.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        safe = f"pd3_{i+1:02d}_{f1}__{f2}__{f3}".replace("/", "-")
        p = os.path.join(FIGDIR, f"{safe}.jpg")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", format="jpg",
                    pil_kwargs={"quality": 90})
        plt.close(fig)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    {i+1:>3}/{len(d)}  {os.path.basename(p)}")

    d.insert(0, "rank", np.arange(1, len(d) + 1))
    d["file"] = [f"pd3_{k+1:02d}_{r.f1}__{r.f2}__{r.f3}.jpg"
                 for k, r in d.iterrows()]
    d.to_csv(out("roe_triples_index.csv"), index=False)
    with pd.ExcelWriter(out("roe_triples.xlsx"), engine="openpyxl") as w:
        d.to_excel(w, sheet_name="ROE triples", index=False)
    print(f"\n  wrote {len(d)} JPGs in {FIGDIR}")
    print("  wrote tex_out/roe_triples_index.csv and roe_triples.xlsx")


if __name__ == "__main__":
    main()
