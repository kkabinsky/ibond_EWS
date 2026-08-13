# -*- coding: utf-8 -*-
"""
pd_surface_3d.py -- the same determinant pairs as knn_cluster_shock.py, redrawn as
three-dimensional response surfaces with the default probability on the z axis.

WHAT THE SURFACE IS
    For a pair (x_a, x_b) a regular grid is laid over the observed range of the two
    determinants. At each grid point the pair is set to that value for a sample of real
    issuer-months while every OTHER determinant keeps its own observed value, and the
    predictions are averaged. That is the standard partial-dependence construction.

    Freezing the other determinants at their medians was tried first and produced a
    surface flat at PD < 0.0001 everywhere: the median issuer is safe on all 31
    remaining determinants, so no movement in two of them can lift the probability.
    Averaging over the real distribution keeps the background variation that gives the
    surface its level.

    Because the model is a gradient-boosted ensemble, the surface is piecewise
    constant. The steps are a real property of the fitted function, not a rendering
    artefact, and they show where the trees placed their splits.

WHAT IS MARKED ON IT
    Two vertical stems: the panel centroid before the shock and after it, each drawn at
    the height the model assigns to that point. The gap between the two heads is the
    change in probability the shock produces at the centroid, which is printed in the
    panel title.

HORIZON
    The 33-determinant panel carries a three-month event label, so the height is PD
    over three months. It is written PD rather than PD12 for that reason.

RUN
    python pd_surface_3d.py
    python pd_surface_3d.py --pairs 12 --grid 40
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

N_PAIRS = 40
GRID = 30
TOP_FEATURES = 10
SHOCK_SD = 1.0
SHOCK_TOP = 5
SCORER = "CatBoost"        # highest PR-AUC in the out-of-fold reanalysis
SEED = 42
BACKGROUND = 250       # real issuer-months averaged over at each grid point
CLIP = (2, 98)             # percentile window for the grid, keeps outliers off the axes


def main():
    n_pairs, grid = N_PAIRS, GRID
    if "--pairs" in sys.argv:
        n_pairs = int(sys.argv[sys.argv.index("--pairs") + 1])
    if "--grid" in sys.argv:
        grid = int(sys.argv[sys.argv.index("--grid") + 1])

    print("=" * 92)
    print("Three-dimensional PD response surfaces over determinant pairs")
    print("=" * 92)
    panel, X, y, cols = cl.load_panel(verbose=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    sd = A.std(0, ddof=1)
    mu = A.mean(0)
    As = (A - mu) / np.where(sd > 0, sd, 1.0)
    med = np.median(As, axis=0)

    imp = pd.read_csv(out("importance_default_event.csv"))
    order = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
    idx = {c: i for i, c in enumerate(cols)}
    feats = [f for f in order.index if f in idx][:TOP_FEATURES]
    shocked_feats = [f for f in order.index if f in idx][:SHOCK_TOP]
    print(f"\n  plotted : {', '.join(feats)}")
    print(f"  shocked : {', '.join(shocked_feats)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(A)
    beta = LogisticRegression(max_iter=5000, C=0.1,
                              class_weight="balanced").fit(As, yv).coef_[0]
    shock_vec = np.zeros(len(cols))
    for f in shocked_feats:
        j = idx[f]
        shock_vec[j] = np.sign(beta[j] if beta[j] != 0 else 1.0) * SHOCK_SD

    est = cl.classifiers()[SCORER]()
    est.fit(sc.transform(A), yv)
    print(f"  surface model: {SCORER}")

    def pd_at(Zstd):
        """Zstd is in standardised units; the model was fitted on the raw scale."""
        return est.predict_proba(sc.transform(Zstd * sd + mu))[:, 1]

    rng = np.random.default_rng(SEED)
    BG = As[rng.choice(len(As), size=min(BACKGROUND, len(As)), replace=False)]
    base_c = BG.mean(0)
    # per-panel shock vectors are built inside the loop

    pairs = list(itertools.combinations(feats, 2))[:n_pairs]
    print(f"\n  building {len(pairs)} surfaces on a {grid}x{grid} grid ...")

    rows = []
    surfaces = []
    for f1, f2 in pairs:
        j1, j2 = idx[f1], idx[f2]
        lo1, hi1 = np.percentile(As[:, j1], CLIP)
        lo2, hi2 = np.percentile(As[:, j2], CLIP)
        g1 = np.linspace(lo1, hi1, grid)
        g2 = np.linspace(lo2, hi2, grid)
        G1, G2 = np.meshgrid(g1, g2)
        # partial dependence: hold the pair at the grid value, keep every other
        # determinant at its real value, average over the background sample
        nb = len(BG)
        big = np.tile(BG, (G1.size, 1))
        big[:, j1] = np.repeat(G1.ravel(), nb)
        big[:, j2] = np.repeat(G2.ravel(), nb)
        P = pd_at(big).reshape(G1.size, nb).mean(1).reshape(G1.shape)

        # shock only the pair on display, so each panel reports its own effect
        pair_vec = np.zeros(len(cols))
        for jj in (j1, j2):
            pair_vec[jj] = np.sign(beta[jj] if beta[jj] != 0 else 1.0) * SHOCK_SD
        p_base = float(pd_at(BG).mean())
        p_shock = float(pd_at(BG + pair_vec).mean())
        surfaces.append((f1, f2, G1, G2, P, j1, j2, p_base, p_shock))
        rows.append(dict(f1=f1, f2=f2, pd_min=P.min(), pd_max=P.max(),
                         pd_range=P.max() - P.min(),
                         pd_at_base=p_base, pd_at_shock=p_shock,
                         delta=p_shock - p_base,
                         shocked=(f1 in shocked_feats) or (f2 in shocked_feats)))
    d = pd.DataFrame(rows)

    print(f"\n  {'pair':44s} {'PD range':>10} {'PD base':>9} {'PD shock':>10}")
    for _, r in d.sort_values("pd_range", ascending=False).head(10).iterrows():
        print(f"  {r.f1[:20]+' x '+r.f2[:20]:44s} {r.pd_range:>10.5f} "
              f"{r.pd_at_base:>9.5f} {r.pd_at_shock:>10.5f}")

    order_plot = d.sort_values("pd_range", ascending=False).index.tolist()

    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig = plt.figure(figsize=(4.3 * ncol, 3.7 * nrow))
    vmax = max(s[4].max() for s in surfaces)

    for pos, k in enumerate(order_plot, 1):
        f1, f2, G1, G2, P, j1, j2, p_base, p_shock = surfaces[k]
        ax = fig.add_subplot(nrow, ncol, pos, projection="3d")
        ax.plot_surface(G1, G2, P, cmap="viridis", vmin=0, vmax=vmax,
                        linewidth=0, antialiased=False, alpha=0.92,
                        rstride=1, cstride=1)
        dx1 = np.sign(beta[j1] if beta[j1] != 0 else 1.0) * SHOCK_SD
        dx2 = np.sign(beta[j2] if beta[j2] != 0 else 1.0) * SHOCK_SD
        for xc, yc, zc, col, mk in (
                (base_c[j1], base_c[j2], p_base, "#1d4ed8", "o"),
                (base_c[j1] + dx1, base_c[j2] + dx2, p_shock, "#b91c1c", "X")):
            ax.plot([xc, xc], [yc, yc], [0, zc], color=col, lw=1.6, zorder=10)
            ax.scatter([xc], [yc], [zc], color=col, s=42, marker=mk,
                       edgecolors="white", linewidth=0.6, zorder=11, depthshade=False)
        star = "*" if d.loc[k, "shocked"] else ""
        ax.set_title(f"{f1} x {f2}{star}\nPD {p_base:.4f} $\\rightarrow$ "
                     f"{p_shock:.4f}", fontsize=8.5, fontweight="bold", pad=2)
        ax.set_xlabel(f1, fontsize=6.5, labelpad=-4)
        ax.set_ylabel(f2, fontsize=6.5, labelpad=-4)
        ax.set_zlabel("PD", fontsize=6.5, labelpad=-6)
        ax.tick_params(labelsize=5.5, pad=-2)
        ax.view_init(elev=26, azim=-125)
        ax.set_zlim(0, vmax)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#1d4ed8",
                      markersize=9, label="centroid, baseline"),
               Line2D([0], [0], marker="X", color="w", markerfacecolor="#b91c1c",
                      markersize=10, label="centroid, after shock")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, 0.006))
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    cax = fig.add_axes([0.965, 0.30, 0.010, 0.40])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("PD, three-month horizon", fontsize=10)

    fig.suptitle("Partial-dependence surfaces of the default probability over "
                 "determinant pairs\n"
                 f"height = {SCORER} PD averaged over {BACKGROUND} real "
                 f"issuer-months at each grid point;  markers = cohort centroid "
                 f"before and after a {SHOCK_SD:.0f} SD adverse shock applied to that "
                 f"panel's own pair;  * = pair is in the top-{SHOCK_TOP} determinants",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.028, 0.955, 0.955])
    p = os.path.join(OUTDIR, "fig_pd_surface_3d.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)

    d.to_csv(out("pd_surface_3d.csv"), index=False)
    print(f"\n  wrote {p}")
    print("  wrote tex_out/pd_surface_3d.csv")


if __name__ == "__main__":
    main()
