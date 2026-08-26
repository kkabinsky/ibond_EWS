# -*- coding: utf-8 -*-
"""
compare_datasets_shock.py -- run the shock analysis on both panels and put the two
results side by side, which is the only way the question "what does dataset 2 do
compared with the original" can actually be answered.

THE QUESTION
    The shock family (knn_cluster_shock, pairwise_shock_pd, triple_shock_pd,
    roe_triple_figures) was written against one panel. It now runs against either,
    selected with --dataset. Running it twice is not yet a comparison: each run
    ranks determinants by its own importance table, so the two runs shock DIFFERENT
    determinants and their figures are not on the same axes. Reading one against
    the other by eye invites exactly the wrong conclusion.

    This module makes the comparison explicit.

WHAT IS COMPARED, AND HOW

    1  ROE-triple response surface
       The top-ranked ROE triple of each panel, drawn on its own axes. This is the
       .jpg the report carries; here the two panels' versions sit in one frame.

    2  Transition path
       Both clouds are standardised over the 33 determinants and projected onto
       their own first two principal components. The baseline cloud, the cloud
       after a one-standard-deviation adverse shock, and the arrow between the two
       centroids are drawn together. The arrow IS the transition path.

       The path is drawn twice per panel:

         own shock      each panel shocks the five determinants it ranks highest,
                        which is what the standalone programs do
         matched shock  both panels shock the SAME five determinants (dataset 1's)

       Without the matched version a difference in path length cannot be told apart
       from a difference in which determinants were shocked. With it, the two
       causes separate.

    3  Rank agreement
       ROE plus the ten partners both panels rank highly is decomposed on BOTH
       panels, so all 45 triples are shared. Left to their own importance tables
       the two panels pick almost disjoint partner sets and only three triples
       overlap, which is far too few to read a correlation from. Each triple's
       joint effect on PD in one panel is plotted against the other, with
       Spearman's rho: a high rho would mean the panels disagree on magnitudes but
       agree on ordering, which is a very different finding from disagreeing on
       both.

WHAT THIS IS NOT
    Not a validation exercise. The two panels have different universes (293 bond
    issuers against 941 listed firms) and different label constructions (derived
    from the ThaiBMA register against the stored y_pre3m), so neither panel's
    numbers are evidence about the other's accuracy. What is being compared is how
    a fitted response surface reacts to a perturbation.

RUN
    python compare_datasets_shock.py
    python compare_datasets_shock.py --skip-chain   # reuse existing per-dataset runs
    python compare_datasets_shock.py --grid 18      # coarser surfaces, faster
"""
from __future__ import annotations

import itertools
import os
import shutil
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
import ibond_dataset as ds
import make_importance_default as mid

SHOCK_SD = 1.0
SHOCK_TOP = 5              # how many determinants the adverse shock moves
TOP_FEATURES = 11          # ROE plus ten partners, as roe_triple_figures uses
GRID = 20
BACKGROUND = 120           # rows averaged over when drawing a response surface
SCATTER_N = 1800
SEED = 42

DSCOL = {1: "#1f3a5f", 2: "#a8501a"}
INK, GRID_C = "#1a1a1a", "#d8d8d8"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "grid.color": GRID_C, "figure.facecolor": "white"})

CHAIN = ["knn_cluster_shock", "pairwise_shock_pd", "triple_shock_pd",
         "roe_triple_figures"]


# ============================================================ per dataset ====
def run_chain(choice, skip=False):
    """Make sure each per-dataset program has produced its CSV, running it if not."""
    ds.use(choice)
    missing = [m for m, f in zip(CHAIN, ["knn_cluster_shock.csv",
                                         "pairwise_shock_pd.csv",
                                         "triple_shock_pd.csv",
                                         "roe_triples_index.csv"])
               if not os.path.exists(ds.out(f))]
    if not missing:
        return
    if skip:
        print(f"  dataset {choice}: missing {missing} and --skip-chain was given")
        return
    import runpy
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in missing:
        path = None
        for dirpath, _, files in os.walk(root):
            if name + ".py" in files:
                path = os.path.join(dirpath, name + ".py")
                break
        print(f"\n  --- dataset {choice}: running {name} ---")
        saved = sys.argv
        sys.argv = [path]
        try:
            runpy.run_path(path, run_name="__main__")
        finally:
            sys.argv = saved
        ds.use(choice)          # the sub-program may have re-read the command line


def prepare(choice, shock_feats=None):
    """Everything the comparison figure needs from one panel.

    ``shock_feats`` forces a matched shock; when None the panel's own top five by
    gain are used.
    """
    ds.use(choice)
    panel, X, y, cols = cl.load_panel(verbose=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    sd = A.std(0, ddof=1)
    idx = {c: i for i, c in enumerate(cols)}

    imp = pd.read_csv(mid.ensure_csv(panel, X, y, cols))
    gain = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
    order = [f for f in gain.index if f in idx]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(A)
    As = sc.transform(A)
    lg = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced").fit(As, yv)
    beta = lg.coef_[0]
    # adverse direction is read off the fitted coefficient, not assumed
    direction = {c: (1.0 if beta[idx[c]] >= 0 else -1.0) for c in cols}

    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0,
                            allow_writing_files=False).fit(As, yv)

    def pdf(B):
        return cb.predict_proba(sc.transform(B))[:, 1]

    own = order[:SHOCK_TOP]
    feats = [f for f in (shock_feats or own) if f in idx] or own

    def shocked(names, base=None):
        B = (A if base is None else base).copy()
        for f in names:
            j = idx[f]
            B[:, j] = B[:, j] + direction[f] * SHOCK_SD * sd[j]
        return B

    return dict(choice=choice, info=ds.DATASETS[choice], panel=panel, cols=cols,
                idx=idx, A=A, As=As, y=yv, sd=sd, gain=gain, order=order,
                own=own, matched=feats, direction=direction, sc=sc, pdf=pdf,
                shocked=shocked, base_pd=pdf(A))


def transition_path(st, feats):
    """Baseline and shocked clouds in the panel's own PC1-PC2, plus the arrow.

    The basis is fitted on the BASELINE cloud only and the shocked cloud is
    projected into it. Refitting on the shocked cloud would rotate the axes with
    the data and hide the very displacement the figure exists to show.
    """
    from sklearn.decomposition import PCA
    As = st["As"]
    Ash = st["sc"].transform(st["shocked"](feats))
    p = PCA(n_components=2, random_state=SEED).fit(As)
    Zb, Zs = p.transform(As), p.transform(Ash)
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(Zb), size=min(SCATTER_N, len(Zb)), replace=False)
    d0, d1 = Zb.mean(0), Zs.mean(0)
    return dict(Zb=Zb[take], Zs=Zs[take], c0=d0, c1=d1,
                shift=float(np.linalg.norm(d1 - d0)),
                var=p.explained_variance_ratio_,
                dpd=float((st["pdf"](st["shocked"](feats)) - st["base_pd"]).mean()))


def common_features(S, anchor="ROE", n_partners=10):
    """ROE plus the partners both panels rank highly, by summed rank.

    Each panel's own roe_triple run picks its partners from its own importance
    table, and the two tables barely overlap, so only a handful of triples are
    evaluated by both. A comparison built on that handful is a comparison of
    almost nothing. Choosing one determinant set and evaluating it on both panels
    turns every triple into a shared observation.
    """
    rank = {}
    for c in (1, 2):
        for i, f in enumerate(S[c]["order"]):
            if f != anchor:
                rank[f] = rank.get(f, 0) + i
    both = [f for f in rank if f in S[1]["idx"] and f in S[2]["idx"]]
    partners = sorted(both, key=lambda f: rank[f])[:n_partners]
    return anchor, partners


def decompose(st, anchor, partners):
    """ANOVA-style split of every triple's shock effect, on one panel.

        D_123 = D_1 + D_2 + D_3 + I_12 + I_13 + I_23 + I_123

    The same decomposition triple_shock_pd.py performs, computed here so that both
    panels are decomposed over an identical determinant set.
    """
    feats = [anchor] + partners
    base = st["base_pd"]
    D1 = {f: float((st["pdf"](st["shocked"]([f])) - base).mean()) for f in feats}
    I2 = {}
    for a, b in itertools.combinations(feats, 2):
        d_ab = float((st["pdf"](st["shocked"]([a, b])) - base).mean())
        I2[frozenset((a, b))] = d_ab - D1[a] - D1[b]
    rows = []
    for b, c in itertools.combinations(partners, 2):
        trio = (anchor, b, c)
        d_abc = float((st["pdf"](st["shocked"](list(trio))) - base).mean())
        s1 = sum(D1[f] for f in trio)
        s2 = (I2[frozenset((anchor, b))] + I2[frozenset((anchor, c))]
              + I2[frozenset((b, c))])
        rows.append(dict(f1=anchor, f2=b, f3=c,
                         k=" + ".join(sorted(trio)),
                         joint=d_abc, sum_singles=s1, sum_pairwise=s2,
                         three_way=d_abc - s1 - s2,
                         pct_of_base=100 * d_abc / base.mean()))
    return pd.DataFrame(rows).sort_values("joint", ascending=False)


def matched_triples(S, anchor, partners):
    """The common decomposition on both panels, merged into one table."""
    out = {}
    for c in (1, 2):
        print(f"  decomposing {len(partners)} partners on dataset {c} ...")
        out[c] = decompose(S[c], anchor, partners)
    m = out[1].merge(out[2], on="k", suffixes=("_ds1", "_ds2"))
    m["rank_ds1"] = m["joint_ds1"].rank(ascending=False).astype(int)
    m["rank_ds2"] = m["joint_ds2"].rank(ascending=False).astype(int)
    m["rank_delta"] = m["rank_ds2"] - m["rank_ds1"]
    keep = ["k", "f1_ds1", "f2_ds1", "f3_ds1", "joint_ds1", "joint_ds2",
            "three_way_ds1", "three_way_ds2", "pct_of_base_ds1",
            "pct_of_base_ds2", "sum_singles_ds1", "sum_singles_ds2",
            "sum_pairwise_ds1", "sum_pairwise_ds2",
            "rank_ds1", "rank_ds2", "rank_delta"]
    m = m[[c for c in keep if c in m.columns]].rename(
        columns={"f1_ds1": "f1", "f2_ds1": "f2", "f3_ds1": "f3"})
    return m.sort_values("joint_ds1", ascending=False).reset_index(drop=True)


# ================================================================ figures ====
def draw_surface(ax, st, trio, grid):
    """PD over two determinants at three levels of the third, as roe_triple does."""
    f1, f2, f3 = trio
    j1, j2, j3 = st["idx"][f1], st["idx"][f2], st["idx"][f3]
    A = st["A"]
    rng = np.random.default_rng(SEED)
    BG = A[rng.choice(len(A), size=BACKGROUND, replace=False)]
    lo1, hi1 = np.percentile(A[:, j1], [2, 98])
    lo2, hi2 = np.percentile(A[:, j2], [2, 98])
    G1, G2 = np.meshgrid(np.linspace(lo1, hi1, grid), np.linspace(lo2, hi2, grid))
    qs = np.percentile(A[:, j3], [15, 50, 85])
    for q, cm in zip(qs, ["Blues", "Oranges", "Reds"]):
        big = np.tile(BG, (G1.size, 1))
        big[:, j1] = np.repeat(G1.ravel(), len(BG))
        big[:, j2] = np.repeat(G2.ravel(), len(BG))
        big[:, j3] = q
        P = st["pdf"](big).reshape(G1.size, len(BG)).mean(1).reshape(G1.shape)
        ax.plot_surface(G1, G2, P, cmap=cm, alpha=0.78, linewidth=0,
                        antialiased=False)
    ax.set_xlabel(f1, fontsize=7.5)
    ax.set_ylabel(f2, fontsize=7.5)
    ax.set_zlabel("PD", fontsize=7.5)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=24, azim=-130)
    # the three sheets are named here rather than in the plot, where the labels
    # sat on top of each other whenever two levels produced similar surfaces
    ax.text2D(0.0, -0.07,
              f"{f3} at p15 / p50 / p85 = {qs[0]:.2f} (blue) / {qs[1]:.2f} "
              f"(orange) / {qs[2]:.2f} (red)",
              transform=ax.transAxes, fontsize=7, color="#444")


def _arrow(ax, tp):
    """Baseline centroid, shocked centroid, and the path between them."""
    ax.annotate("", xy=tuple(tp["c1"]), xytext=tuple(tp["c0"]),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#111827",
                                mutation_scale=16), zorder=6)
    ax.scatter(*tp["c0"], s=48, color="#111827", zorder=7, marker="o")
    ax.scatter(*tp["c1"], s=80, color="#111827", zorder=7, marker="*")


def draw_path(ax, st, tp, feats, title):
    c = DSCOL[st["choice"]]
    # shocked cloud underneath, baseline on top: drawn the other way round the
    # baseline disappears entirely under the crosses and the panel shows one cloud
    ax.scatter(tp["Zs"][:, 0], tp["Zs"][:, 1], s=7, marker="x", alpha=0.22,
               color="#b91c1c", linewidth=0.5, label="after shock", zorder=2)
    ax.scatter(tp["Zb"][:, 0], tp["Zb"][:, 1], s=6, alpha=0.45, color=c,
               edgecolors="none", label="baseline", zorder=3)
    _arrow(ax, tp)
    lo1, hi1 = np.percentile(np.r_[tp["Zb"][:, 0], tp["Zs"][:, 0]], [0.5, 99.5])
    lo2, hi2 = np.percentile(np.r_[tp["Zb"][:, 1], tp["Zs"][:, 1]], [0.5, 99.5])
    ax.set_xlim(lo1, hi1); ax.set_ylim(lo2, hi2)

    # The path is a fraction of a standard deviation and the cloud spans several,
    # so at cloud scale the arrow is a dot. The inset shows the same arrow at the
    # scale of the displacement itself; the panel keeps the honest scale.
    ins = ax.inset_axes([0.63, 0.60, 0.35, 0.36])
    d = tp["c1"] - tp["c0"]
    pad = max(float(np.abs(d).max()), 1e-6) * 1.6
    mid = (tp["c0"] + tp["c1"]) / 2
    ins.set_xlim(mid[0] - pad, mid[0] + pad)
    ins.set_ylim(mid[1] - pad, mid[1] + pad)
    ins.scatter(tp["Zs"][:, 0], tp["Zs"][:, 1], s=5, marker="x", alpha=0.25,
                color="#b91c1c", linewidth=0.5)
    ins.scatter(tp["Zb"][:, 0], tp["Zb"][:, 1], s=4, alpha=0.35, color=c,
                edgecolors="none")
    _arrow(ins, tp)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("centroid displacement, zoomed", fontsize=6.5, pad=2)
    for sp in ins.spines.values():
        sp.set_edgecolor("#888")
    ax.set_xlabel(f"PC1 ({100*tp['var'][0]:.1f}% of variance)", fontsize=8)
    ax.set_ylabel(f"PC2 ({100*tp['var'][1]:.1f}%)", fontsize=8)
    ax.set_title(f"{title}\npath length {tp['shift']:.3f} SD   "
                 f"mean $\\Delta$PD {tp['dpd']*1000:+.2f}$\\times10^{{-3}}$",
                 fontsize=9, fontweight="bold")
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.02, "shocked: " + ", ".join(f[:14] for f in feats),
            transform=ax.transAxes, fontsize=6.2, color="#444",
            va="bottom", ha="left")


def main_figure(S, T_own, T_matched, triples, agree, grid, path):
    fig = plt.figure(figsize=(15.5, 16.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.0, 1.0, 0.95], hspace=0.34,
                          wspace=0.22)

    # ---- row A: the ROE triple each panel ranks first
    for k, c in enumerate((1, 2)):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        trio = triples[c]
        draw_surface(ax, S[c], trio, grid)
        ax.set_title(f"A{k+1}. Dataset {c}: top ROE triple\n"
                     f"{trio[0]} + {trio[1]} + {trio[2]}",
                     fontsize=10, fontweight="bold", color=DSCOL[c])

    # ---- row B: transition path under each panel's OWN shock
    for k, c in enumerate((1, 2)):
        ax = fig.add_subplot(gs[1, k])
        draw_path(ax, S[c], T_own[c], S[c]["own"],
                  f"B{k+1}. Dataset {c}: transition path, own top-5 shock")
        if k == 0:
            ax.legend(loc="upper left", fontsize=7, markerscale=2.2, framealpha=0.9)

    # ---- row C: the same shock applied to both panels
    for k, c in enumerate((1, 2)):
        ax = fig.add_subplot(gs[2, k])
        draw_path(ax, S[c], T_matched[c], S[c]["matched"],
                  f"C{k+1}. Dataset {c}: transition path, MATCHED shock")

    # ---- row D left: rank agreement on triples
    ax = fig.add_subplot(gs[3, 0])
    if not agree.empty:
        ax.scatter(agree["joint_ds1"] * 1000, agree["joint_ds2"] * 1000, s=26,
                   color="#2e7d4f", alpha=0.75, edgecolors="white", linewidth=0.5)
        lim = np.r_[agree["joint_ds1"] * 1000, agree["joint_ds2"] * 1000]
        pad = 0.06 * (lim.max() - lim.min() + 1e-9)
        line = np.array([lim.min() - pad, lim.max() + pad])
        ax.plot(line, line, ls="--", lw=1.0, color="#888")
        ax.axhline(0, lw=0.7, color="#bbb"); ax.axvline(0, lw=0.7, color="#bbb")
        rho = agree["joint_ds1"].corr(agree["joint_ds2"], method="spearman")
        pear = agree["joint_ds1"].corr(agree["joint_ds2"])
        ax.set_title(f"D1. Joint effect of the same triple, both panels\n"
                     f"{len(agree)} shared triples   Spearman $\\rho$ = {rho:.3f}   "
                     f"Pearson r = {pear:.3f}", fontsize=9.5, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no triple is evaluated by both panels",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D1. Joint effect of the same triple", fontsize=9.5,
                     fontweight="bold")
    ax.set_xlabel(r"Dataset 1  mean $\Delta$PD  ($\times 10^{-3}$)", fontsize=8)
    ax.set_ylabel(r"Dataset 2  mean $\Delta$PD  ($\times 10^{-3}$)", fontsize=8)
    ax.grid(alpha=0.2)

    # ---- row D right: which determinants each panel leans on
    ax = fig.add_subplot(gs[3, 1])
    feats = list(dict.fromkeys(S[1]["order"][:10] + S[2]["order"][:10]))
    yy = np.arange(len(feats))
    w = 0.4
    for off, c in ((-w / 2, 1), (w / 2, 2)):
        vals = [float(S[c]["gain"].get(f, 0.0)) for f in feats]
        ax.barh(yy + off, vals, height=w, color=DSCOL[c], alpha=0.9,
                label=f"Dataset {c}")
    ax.set_yticks(yy)
    ax.set_yticklabels(feats, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean gain across the four tree models", fontsize=8)
    ax.set_title("D2. Determinant ranking behind every shock above\n"
                 "union of each panel's top ten", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle(
        "Shock analysis on two panels: Dataset 1 (293 bond issuers, 16,986 "
        "firm-months) against Dataset 2 (941 listed firms, 187,007 firm-months)\n"
        "adverse shock of one standard deviation; direction taken from the fitted "
        "logistic coefficient; PD from CatBoost fitted on each panel",
        fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(path, dpi=140, bbox_inches="tight", format="jpg",
                pil_kwargs={"quality": 92})
    plt.close(fig)
    return path


def side_by_side_triples(S, triples, grid, outdir, n=12):
    """One JPG per matched triple, dataset 1 beside dataset 2 on the same axes."""
    os.makedirs(outdir, exist_ok=True)
    written = []
    for i, r in enumerate(triples.head(n).itertuples(), 1):
        trio = (r.f1, r.f2, r.f3)
        fig = plt.figure(figsize=(12.4, 5.0))
        for k, c in enumerate((1, 2)):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            draw_surface(ax, S[c], trio, grid)
            j = getattr(r, f"joint_ds{k+1}")
            t3 = getattr(r, f"three_way_ds{k+1}")
            ax.set_title(f"Dataset {c}   joint $\\Delta$PD {j*1000:+.2f}"
                         f"$\\times10^{{-3}}$   three-way {t3*1000:+.2f}",
                         fontsize=9.5, fontweight="bold", color=DSCOL[c])
        fig.suptitle(f"{trio[0]} + {trio[1]} + {trio[2]}  --  same triple, both panels",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        safe = f"cmp_{i:02d}_" + "__".join(trio).replace("/", "_")
        p = os.path.join(outdir, safe + ".jpg")
        fig.savefig(p, dpi=130, bbox_inches="tight", format="jpg",
                    pil_kwargs={"quality": 90})
        plt.close(fig)
        written.append(p)
    return written


# ================================================================= tables ====
def _ds_csv(choice, name):
    return os.path.join(ds.DATA_ROOT, "tex_out", ds.DATASETS[choice]["key"], name)


def _merge_pairs(name, cols_keep, extra_key=None):
    """Merge a per-dataset CSV on its unordered determinant key."""
    frames = {}
    for c in (1, 2):
        p = _ds_csv(c, name)
        if not os.path.exists(p):
            return pd.DataFrame()
        d = pd.read_csv(p)
        keycols = [k for k in ("f1", "f2", "f3") if k in d.columns]
        d["k"] = d[keycols].apply(lambda r: " + ".join(sorted(map(str, r))), axis=1)
        if extra_key and extra_key in d.columns:
            d["k"] = d[extra_key].astype(str) + " | " + d["k"]
        frames[c] = d[["k"] + [x for x in cols_keep if x in d.columns]]
    return frames[1].merge(frames[2], on="k", suffixes=("_ds1", "_ds2"))


def build_tables():
    triples = _merge_pairs("roe_triples_index.csv",
                           ["joint", "three_way", "pct_of_base", "sum_singles",
                            "sum_pairwise"])
    if not triples.empty:
        triples["rank_ds1"] = triples["joint_ds1"].rank(ascending=False).astype(int)
        triples["rank_ds2"] = triples["joint_ds2"].rank(ascending=False).astype(int)
        triples["rank_delta"] = triples["rank_ds2"] - triples["rank_ds1"]
        triples = triples.sort_values("joint_ds1", ascending=False)
    pairs = _merge_pairs("pairwise_shock_pd.csv",
                         ["joint", "interaction", "pct_chg", "alarm_delta_pp"],
                         extra_key="model")
    knn = _merge_pairs("knn_cluster_shock.csv", ["knn_acc", "centroid_shift"])
    return triples, pairs, knn


def summary_rows(S, T_own, T_matched, triples):
    rows = []
    for c in (1, 2):
        st, info = S[c], ds.DATASETS[c]
        rows.append(dict(
            dataset=c,
            label=info["label"],
            table=info["table"],
            firm_months=len(st["panel"]),
            firms=int(st["panel"]["issuer_code"].nunique()),
            positives=int(st["y"].sum()),
            positive_firms=int(pd.Series(st["panel"]["issuer_code"])[st["y"] == 1]
                               .nunique()),
            event_rate_pct=100 * float(st["y"].mean()),
            baseline_mean_pd=float(st["base_pd"].mean()),
            top5_by_gain=", ".join(st["own"]),
            path_len_own=T_own[c]["shift"],
            dpd_own=T_own[c]["dpd"],
            path_len_matched=T_matched[c]["shift"],
            dpd_matched=T_matched[c]["dpd"],
            pc1_var_pct=100 * float(T_own[c]["var"][0]),
            pc2_var_pct=100 * float(T_own[c]["var"][1]),
        ))
    d = pd.DataFrame(rows)
    if not triples.empty:
        d["triple_spearman"] = triples["joint_ds1"].corr(triples["joint_ds2"],
                                                         method="spearman")
    return d


# =================================================================== main ====
def main():
    grid = GRID
    if "--grid" in sys.argv:
        grid = int(sys.argv[sys.argv.index("--grid") + 1])
    skip = "--skip-chain" in sys.argv

    print("=" * 100)
    print("Shock analysis compared across the two panels")
    print("=" * 100)

    for c in (1, 2):
        run_chain(c, skip=skip)

    print("\n  fitting dataset 1 ...")
    S = {1: prepare(1)}
    matched = S[1]["own"]                       # dataset 1's shock, reused on both
    print("\n  fitting dataset 2 ...")
    S[2] = prepare(2, shock_feats=matched)
    S[1]["matched"] = matched

    T_own = {c: transition_path(S[c], S[c]["own"]) for c in (1, 2)}
    T_matched = {c: transition_path(S[c], S[c]["matched"]) for c in (1, 2)}

    anchor, partners = common_features(S)
    print("\n  common determinant set: " + anchor + " + " + ", ".join(partners))
    triples = matched_triples(S, anchor, partners)

    print("\n  transition path")
    print(f"  {'':10s} {'own shock':>34s}   {'matched shock':>34s}")
    print(f"  {'dataset':10s} {'length':>9} {'mean dPD':>11} {'shocked':>11}   "
          f"{'length':>9} {'mean dPD':>11}")
    for c in (1, 2):
        print(f"  {c:<10d} {T_own[c]['shift']:>9.4f} {T_own[c]['dpd']:>11.6f} "
              f"{len(S[c]['own']):>11d}   {T_matched[c]['shift']:>9.4f} "
              f"{T_matched[c]['dpd']:>11.6f}")

    own_triples, pairs, knn = build_tables()
    if not triples.empty:
        rho = triples["joint_ds1"].corr(triples["joint_ds2"], method="spearman")
        print(f"\n  {len(triples)} triples evaluated by both panels, "
              f"Spearman rho = {rho:.3f}")
        print(f"\n  {'triple':52s} {'dPD ds1':>10} {'dPD ds2':>10} {'rank move':>10}")
        for r in triples.head(12).itertuples():
            print(f"  {r.k[:52]:52s} {r.joint_ds1*1000:>10.3f} "
                  f"{r.joint_ds2*1000:>10.3f} {r.rank_delta:>+10d}")

    summary = summary_rows(S, T_own, T_matched, triples)

    # ------------------------------------------------------------- outputs ---
    figp = main_figure(S, T_own, T_matched,
                       {c: _top_triple(c) for c in (1, 2)}, triples, grid,
                       ds.result_out("fig_shock_dataset_compare.jpg"))
    print(f"\n  wrote {figp}")

    sbs_dir = os.path.join(ds.RESULTS_DIR, "triples_side_by_side")
    written = side_by_side_triples(S, triples, grid, sbs_dir)
    print(f"  wrote {len(written)} side-by-side triple JPGs into {sbs_dir}")

    for c in (1, 2):
        src = os.path.join(ds.DATA_ROOT, "tex_out", ds.DATASETS[c]["key"],
                           "triples_roe")
        dst = os.path.join(ds.RESULTS_DIR, f"triples_roe_ds{c}")
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            n = 0
            for f in sorted(os.listdir(src)):
                if f.endswith(".jpg"):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                    n += 1
            print(f"  copied {n} dataset-{c} triple JPGs into {dst}")
        for f in ("fig_knn_cluster_shock.png", "fig_pairwise_shock.png",
                  "fig_triple_shock.png", "fig_pca_shock.png",
                  "fig_pca_shock_simulation.png"):
            p = _ds_csv(c, f)
            if os.path.exists(p):
                shutil.copy2(p, ds.result_out(f"ds{c}_" + f))

    summary.to_csv(ds.result_out("shock_dataset_summary.csv"), index=False)
    if not triples.empty:
        triples.to_csv(ds.result_out("shock_dataset_compare.csv"), index=False)
    xl = ds.result_out("shock_dataset_compare.xlsx")
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        if not triples.empty:
            triples.to_excel(w, sheet_name="ROE triples", index=False)
        if not own_triples.empty:
            own_triples.to_excel(w, sheet_name="ROE triples (own sets)",
                                 index=False)
        if not pairs.empty:
            pairs.to_excel(w, sheet_name="pairs", index=False)
        if not knn.empty:
            knn.to_excel(w, sheet_name="knn separability", index=False)
    print(f"  wrote {xl}")

    con = sqlite3.connect(ds.RESULT_DB)
    summary.to_sql("cmdf_shock_dataset_summary", con, if_exists="replace",
                   index=False)
    if not triples.empty:
        triples.to_sql("cmdf_shock_dataset_compare", con, if_exists="replace",
                       index=False)
    con.commit(); con.close()

    write_summary_th(S, T_own, T_matched, triples, summary)
    print("\n  done")


def _top_triple(choice):
    p = _ds_csv(choice, "roe_triples_index.csv")
    d = pd.read_csv(p).sort_values("joint", ascending=False).iloc[0]
    return (d.f1, d.f2, d.f3)


def write_summary_th(S, T_own, T_matched, triples, summary):
    """Thai narrative, written from the numbers rather than typed in by hand."""
    L = []
    A = L.append
    A("# ผลการเปรียบเทียบ shock feature: Dataset 1 กับ Dataset 2\n")
    A("ไฟล์นี้สร้างอัตโนมัติจาก `analysis/shock/compare_datasets_shock.py` "
      "ตัวเลขทุกตัวอ่านจากผลการรันจริง ไม่ได้พิมพ์เข้าไปเอง\n")

    A("## ขนาดของข้อมูลและจำนวนเหตุการณ์\n")
    A("| รายการ | Dataset 1 | Dataset 2 |")
    A("|---|---:|---:|")
    r1, r2 = summary.iloc[0], summary.iloc[1]
    A(f"| ตาราง | `{r1.table}` | `{r2.table}` |")
    A(f"| firm-months | {r1.firm_months:,} | {r2.firm_months:,} |")
    A(f"| จำนวนบริษัท | {r1.firms:,} | {r2.firms:,} |")
    A(f"| เดือนที่เป็นเหตุการณ์ | {r1.positives:,} | {r2.positives:,} |")
    A(f"| บริษัทที่มีเหตุการณ์ | {r1.positive_firms} | {r2.positive_firms} |")
    A(f"| อัตราเหตุการณ์ | {r1.event_rate_pct:.3f}% | {r2.event_rate_pct:.3f}% |")
    A(f"| PD เฉลี่ยฐาน (CatBoost) | {r1.baseline_mean_pd:.5f} | "
      f"{r2.baseline_mean_pd:.5f} |\n")

    A("## ตัวแปรที่ถูก shock\n")
    A(f"- Dataset 1 (5 อันดับแรกตาม gain): `{r1.top5_by_gain}`")
    A(f"- Dataset 2 (5 อันดับแรกตาม gain): `{r2.top5_by_gain}`\n")
    same = set(S[1]["own"]) & set(S[2]["own"])
    A(f"ตัวแปรที่ตรงกันมี {len(same)} ตัว"
      + (f": `{', '.join(sorted(same))}`" if same else " (ไม่มีเลย)") + "\n")

    A("## Transition path (ลูกศรการเคลื่อนของ centroid ในพิกัด PC1-PC2)\n")
    A("| | ความยาว path (own shock) | ΔPD เฉลี่ย (own) | ความยาว path (matched shock) | ΔPD เฉลี่ย (matched) |")
    A("|---|---:|---:|---:|---:|")
    for c in (1, 2):
        A(f"| Dataset {c} | {T_own[c]['shift']:.4f} | {T_own[c]['dpd']*1000:+.3f}e-3 "
          f"| {T_matched[c]['shift']:.4f} | {T_matched[c]['dpd']*1000:+.3f}e-3 |")
    A("")
    A("`own shock` คือแต่ละชุดขยับตัวแปร 5 ตัวที่ตัวเองจัดอันดับสูงสุด "
      "ซึ่งเป็นสิ่งที่โปรแกรมเดิมทำ ส่วน `matched shock` บังคับให้ทั้งสองชุด "
      "ขยับตัวแปรชุดเดียวกัน (ของ Dataset 1) ถ้าไม่มีคอลัมน์ matched "
      "จะแยกไม่ออกว่าความยาว path ที่ต่างกันมาจากข้อมูลหรือมาจากการที่ "
      "shock คนละตัวแปร\n")

    if not triples.empty:
        rho = triples["joint_ds1"].corr(triples["joint_ds2"], method="spearman")
        A("## ความสอดคล้องของอันดับ ROE triple\n")
        A(f"- triple ที่ทั้งสองชุดประเมินร่วมกัน: {len(triples)} ชุด")
        A(f"- Spearman rho ของ joint ΔPD: **{rho:.3f}**\n")
        A("| triple | ΔPD ds1 (x1e-3) | ΔPD ds2 (x1e-3) | อันดับขยับ |")
        A("|---|---:|---:|---:|")
        for r in triples.head(10).itertuples():
            A(f"| {r.k} | {r.joint_ds1*1000:.3f} | {r.joint_ds2*1000:.3f} "
              f"| {r.rank_delta:+d} |")
        A("")

    if not triples.empty:
        n = len(triples)
        neg1 = int((triples["joint_ds1"] < 0).sum())
        neg2 = int((triples["joint_ds2"] < 0).sum())
        A("## ทิศทางของ shock กับผิวตอบสนองของ CatBoost\n")
        A(f"ในบรรดา {n} triple ที่ประเมินร่วมกัน ค่า ΔPD ออกมาติดลบ "
          f"{neg1} ครั้งใน Dataset 1 และ {neg2} ครั้งใน Dataset 2 "
          "ทั้งที่ทิศทางถูกกำหนดให้เป็น *ทิศทางร้าย*\n")
        A("สาเหตุอยู่ในตัวโปรแกรมเดิมอยู่แล้ว ไม่ใช่ผลจากการเพิ่ม dataset ที่สอง "
          "ทิศทางร้ายอ่านจากเครื่องหมายของสัมประสิทธิ์ logistic "
          "แต่ค่า PD ที่นำมาวัดมาจาก CatBoost เมื่อทั้งสองแบบจำลอง "
          "ไม่เห็นตรงกันว่าตัวแปรตัวไหนเพิ่มความเสี่ยง "
          "การขยับตามทิศทางของ logistic จึงทำให้ผิวของ CatBoost ลดลงได้ "
          "อ่านผลจึงควรอ่านที่ **ขนาด** ของการเปลี่ยนแปลงและอันดับ "
          "มากกว่าจะอ่านเครื่องหมายว่าเป็นการยืนยันทิศทาง\n")
        A("ข้อสังเกตนี้เป็นคนละเรื่องกับคำถามว่า dataset2 ให้ผลอย่างไร "
          "แต่จำเป็นต้องระบุ เพราะมันเปลี่ยนวิธีอ่านตัวเลขในตารางข้างบน\n")

    A("## ข้อควรระวังในการอ่านผล\n")
    A("- นิยาม target ไม่เหมือนกัน Dataset 1 สร้าง label จาก "
      "`ibond_default_payment` (เดือน t ถึง t+3 ก่อนการผิดนัดครั้งแรก) "
      "ส่วน Dataset 2 ใช้คอลัมน์ `y_pre3m` ที่เก็บมาแล้วในตาราง")
    A("- universe ต่างกัน ชุดแรกเป็นผู้ออกหุ้นกู้ ชุดที่สองเป็นบริษัทจดทะเบียน "
      "SET/mai ทั้งหมด อัตราเหตุการณ์จึงเจือจางกว่า")
    A("- การวิเคราะห์นี้ fit บนข้อมูลทั้งชุด เป็น sensitivity ของ response "
      "surface ไม่ใช่การวัดความแม่นยำนอกกลุ่มตัวอย่าง")
    A("- Dataset 1 มีบริษัทที่เกิดเหตุการณ์เพียง "
      f"{r1.positive_firms} ราย ค่า gain จึงเปราะ "
      "อันดับตัวแปรที่ต่างกันระหว่างสองชุดอาจมาจากขนาดตัวอย่าง "
      "ไม่ใช่จากข้อมูลใหม่")

    p = ds.result_out("summary_th.md")
    open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"  wrote {p}")
    return p


if __name__ == "__main__":
    main()
