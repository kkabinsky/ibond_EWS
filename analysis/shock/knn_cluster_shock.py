# -*- coding: utf-8 -*-
"""
knn_cluster_shock.py -- forty pairwise views of the determinant cloud, before and after
a shock, with the clusters marked and the displacement measured.

WHAT IS DRAWN
    Issuer-months are clustered once in the full 33-determinant space by k-means, so
    every point carries a cluster label that does not depend on which pair happens to
    be plotted. Each panel then shows one determinant pair: baseline points as filled
    circles coloured by cluster, shocked points as crosses.

WHAT IS MEASURED
    A scatter can suggest that a cloud has moved without establishing it. Each panel
    therefore also carries a k-nearest-neighbour separability score: a KNN classifier
    is trained, on that pair of determinants alone, to tell baseline points from
    shocked ones, and the cross-validated accuracy is reported.

        0.50   the two clouds are indistinguishable in this projection
        1.00   the shock is fully visible in these two determinants

    This is the quantity the eye is trying to judge, so it is better computed than
    guessed. Panels are ordered by it, most separable first.

RUN
    python knn_cluster_shock.py
    python knn_cluster_shock.py --pairs 40 --k 4
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

N_PAIRS = 40
N_CLUSTERS = 4
TOP_FEATURES = 10          # C(10,2) = 45 pairs, the first N_PAIRS are drawn
SHOCK_SD = 1.0
SHOCK_TOP = 5
SUBSAMPLE = 1200
KNN_K = 15
SEED = 42

CCOL = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#be123c"]


def knn_separability(a, b, k=KNN_K, seed=SEED):
    """Cross-validated accuracy of a KNN classifier separating the two clouds."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Z = np.vstack([a, b])
    lab = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    return float(cross_val_score(pipe, Z, lab, cv=3, scoring="accuracy").mean())


def main():
    n_pairs, k_clusters = N_PAIRS, N_CLUSTERS
    if "--pairs" in sys.argv:
        n_pairs = int(sys.argv[sys.argv.index("--pairs") + 1])
    if "--k" in sys.argv:
        k_clusters = int(sys.argv[sys.argv.index("--k") + 1])

    print("=" * 96)
    print("Pairwise determinant views with clusters, baseline versus shocked")
    print("=" * 96)
    panel, X, y, cols = cl.load_panel(verbose=True)
    A = X.to_numpy(float)
    sd = A.std(0, ddof=1)
    mu = A.mean(0)
    As = (A - mu) / np.where(sd > 0, sd, 1.0)

    imp = pd.read_csv(out("importance_default_event.csv"))
    order = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
    idx = {c: i for i, c in enumerate(cols)}
    feats = [f for f in order.index if f in idx][:TOP_FEATURES]
    shocked_feats = [f for f in order.index if f in idx][:SHOCK_TOP]
    print(f"\n  determinants plotted : {', '.join(feats)}")
    print(f"  determinants shocked : {', '.join(shocked_feats)}")

    from sklearn.linear_model import LogisticRegression
    beta = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced") \
        .fit(As, y.to_numpy(int)).coef_[0]
    Ash = As.copy()
    for f in shocked_feats:
        j = idx[f]
        Ash[:, j] += np.sign(beta[j] if beta[j] != 0 else 1.0) * SHOCK_SD

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=SEED).fit(As)
    lab = km.labels_
    sizes = np.bincount(lab)
    print(f"\n  k-means on all 33 determinants, k = {k_clusters}")
    for c in range(k_clusters):
        print(f"    cluster {c}: {sizes[c]:,} issuer-months "
              f"({100*sizes[c]/len(lab):.1f}%)   "
              f"event rate {100*y.to_numpy()[lab==c].mean():.3f}%")

    rng = np.random.default_rng(SEED)
    take = rng.choice(len(As), size=min(SUBSAMPLE, len(As)), replace=False)

    pairs = list(itertools.combinations(feats, 2))[:n_pairs]
    print(f"\n  scoring {len(pairs)} pairs with a {KNN_K}-nearest-neighbour "
          f"separability test ...")
    rows = []
    for f1, f2 in pairs:
        j1, j2 = idx[f1], idx[f2]
        a = As[take][:, [j1, j2]]
        b = Ash[take][:, [j1, j2]]
        acc = knn_separability(a, b)
        shift = np.linalg.norm(b.mean(0) - a.mean(0))
        rows.append(dict(f1=f1, f2=f2, knn_acc=acc, centroid_shift=shift,
                         shocked_1=f1 in shocked_feats,
                         shocked_2=f2 in shocked_feats))
    d = pd.DataFrame(rows).sort_values("knn_acc", ascending=False)

    print(f"\n  {'pair':44s} {'KNN acc':>8} {'shift':>8}  shocked")
    for _, r in d.head(12).iterrows():
        tag = ("both" if r.shocked_1 and r.shocked_2
               else ("one" if r.shocked_1 or r.shocked_2 else "neither"))
        print(f"  {r.f1[:20]+' vs '+r.f2[:20]:44s} {r.knn_acc:>8.3f} "
              f"{r.centroid_shift:>8.3f}  {tag}")
    print(f"\n  {'':44s} {'lowest':>8}")
    for _, r in d.tail(3).iterrows():
        print(f"  {r.f1[:20]+' vs '+r.f2[:20]:44s} {r.knn_acc:>8.3f} "
              f"{r.centroid_shift:>8.3f}")

    # ------------------------------------------------------------- figure ----
    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, d.iterrows()):
        j1, j2 = idx[r.f1], idx[r.f2]
        for c in range(k_clusters):
            sel = take[lab[take] == c]
            ax.scatter(As[sel, j1], As[sel, j2], s=7, alpha=0.42,
                       color=CCOL[c % len(CCOL)], edgecolors="none",
                       label=f"cluster {c}" if ax is axes[0] else None)
        ax.scatter(Ash[take][:, j1], Ash[take][:, j2], s=8, marker="x",
                   alpha=0.30, color="#b91c1c", linewidth=0.6,
                   label="shocked" if ax is axes[0] else None)
        ax.annotate("", xy=(Ash[take][:, j1].mean(), Ash[take][:, j2].mean()),
                    xytext=(As[take][:, j1].mean(), As[take][:, j2].mean()),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#111827",
                                    mutation_scale=13), zorder=6)
        star = "*" if (r.shocked_1 or r.shocked_2) else ""
        ax.set_title(f"{r.f1} vs {r.f2}{star}\nKNN {r.knn_acc:.3f}   "
                     f"shift {r.centroid_shift:.2f}", fontsize=8.5,
                     fontweight="bold")
        ax.set_xlabel(r.f1, fontsize=7.5)
        ax.set_ylabel(r.f2, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.18)
        lo1, hi1 = np.percentile(np.r_[As[take][:, j1], Ash[take][:, j1]], [1, 99])
        lo2, hi2 = np.percentile(np.r_[As[take][:, j2], Ash[take][:, j2]], [1, 99])
        ax.set_xlim(lo1, hi1); ax.set_ylim(lo2, hi2)
    for ax in axes[len(pairs):]:
        ax.axis("off")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=k_clusters + 1, fontsize=10,
               markerscale=3.0, frameon=True, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Determinant pairs, clustered in the full 33-determinant space, "
                 "baseline versus shocked\n"
                 f"circles = baseline coloured by k-means cluster (k={k_clusters}); "
                 f"crosses = after a {SHOCK_SD:.0f} SD adverse shock to "
                 f"{', '.join(shocked_feats[:3])}...;  "
                 "arrow = centroid displacement;  "
                 "* = at least one determinant in the pair was shocked",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    p = os.path.join(OUTDIR, "fig_knn_cluster_shock.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)

    d.to_csv(out("knn_cluster_shock.csv"), index=False)
    con = sqlite3.connect(DB)
    d.to_sql("cmdf_knn_cluster_shock", con, if_exists="replace", index=False)
    con.commit(); con.close()
    print(f"\n  wrote {p}")
    print("  wrote tex_out/knn_cluster_shock.csv")


if __name__ == "__main__":
    main()
