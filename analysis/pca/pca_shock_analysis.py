# -*- coding: utf-8 -*-
"""
pca_shock_analysis.py -- PCA of the two leading determinants, the inverse map, and
what a threshold-level shock does to the regression outcome.

SETTING
    The regression of interest is y = beta' x + eps with x the determinant vector. We
    restrict to the two determinants the importance analysis puts forward, x1 = ROE and
    x2 = DE, and ask three questions:

        1  what orthogonal basis does PCA give for the (ROE, DE) plane,
        2  where does a cluster of issuers move in that basis when the determinants are
           shocked to their anomaly thresholds,
        3  how much does y change as a result, decomposed by principal direction.

WHY THE SCALING CHOICE MATTERS
    ROE and DE are measured in different units and differ in dispersion by an order of
    magnitude. PCA on the raw covariance is then dominated by whichever variable
    happens to have the larger variance and says nothing about co-movement. The
    correlation-matrix version answers the co-movement question and is the one used for
    interpretation; both are reported so the difference is visible rather than hidden
    in a preprocessing choice.

RUN
    python pca_shock_analysis.py
"""
from __future__ import annotations

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
os.makedirs(OUTDIR, exist_ok=True)

BETA1 = [1, 2, 3, 4, 5]        # coefficient on ROE
BETA2 = [1, 2, 3, 4]           # coefficient on DE
WINSOR = (0.01, 0.99)
SEED = 42

# anomaly thresholds: DE from the risk-zone table, ROE from the adverse tail
DE_AMBER, DE_RED = 1.606, 2.116


def load_xy():
    con = sqlite3.connect(DB)
    d = pd.read_sql("SELECT ROE, DE FROM ibond_33features_panel", con)
    con.close()
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    for c in d.columns:
        lo, hi = d[c].quantile(WINSOR)
        d[c] = d[c].clip(lo, hi)
    return d


def eig2(S):
    """Closed-form eigenpairs of a symmetric 2x2 matrix, sorted by decreasing value.

    Written out rather than delegated so the algebra in the manuscript can be checked
    against the numbers directly."""
    a, b, c = S[0, 0], S[0, 1], S[1, 1]
    tr, det = a + c, a * c - b * b
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    def vec(l):
        v = np.array([b, l - a]) if abs(b) > 1e-12 else np.array([1.0, 0.0])
        return v / np.linalg.norm(v)
    v1, v2 = vec(l1), vec(l2)
    if v1[0] < 0:
        v1 = -v1
    v2 = np.array([-v1[1], v1[0]])          # orthogonal complement, right-handed
    return np.array([l1, l2]), np.column_stack([v1, v2])


def report_basis(name, S, cols):
    lam, V = eig2(S)
    print(f"\n=== {name} ===")
    print("  covariance / correlation matrix")
    print("   ", np.array2string(S, precision=4, suppress_small=True).replace("\n", "\n    "))
    tot = lam.sum()
    for k in (0, 1):
        v = V[:, k]
        print(f"  PC{k+1}: eigenvalue {lam[k]:.4f}  "
              f"({100*lam[k]/tot:.1f}% of variance)   "
              f"eigenvector ({cols[0]} {v[0]:+.4f}, {cols[1]} {v[1]:+.4f})")
    print(f"  check: V orthonormal -> V'V = I ? "
          f"{np.allclose(V.T @ V, np.eye(2))}")
    print(f"  check: sum of eigenvalues = trace ? "
          f"{np.isclose(lam.sum(), np.trace(S))}")
    return lam, V


def gamma_table(V, lam):
    """gamma = V' beta is the loading of y on each principal direction.

    Var(y) contributed by PC k is gamma_k^2 * lambda_k, because the scores are
    uncorrelated with variances equal to the eigenvalues."""
    rows = []
    for b1 in BETA1:
        for b2 in BETA2:
            beta = np.array([b1, b2], float)
            g = V.T @ beta
            var = g ** 2 * lam
            rows.append(dict(beta1=b1, beta2=b2,
                             gamma1=g[0], gamma2=g[1],
                             var_pc1=var[0], var_pc2=var[1],
                             share_pc1=var[0] / var.sum(),
                             norm_beta=np.linalg.norm(beta),
                             angle_beta_pc1=np.degrees(
                                 np.arccos(abs(g[0]) / np.linalg.norm(beta)))))
    return pd.DataFrame(rows)


def shock_analysis(d, S, V, lam, cols, mu, sd):
    """Move the cluster centroid to the anomaly thresholds and follow it through."""
    med = d.median().to_numpy(float)
    roe_low = d["ROE"].quantile(0.10)
    targets = {
        "DE to amber (1.606)": np.array([med[0], DE_AMBER]),
        "DE to red (2.116)": np.array([med[0], DE_RED]),
        "ROE to 10th pct": np.array([roe_low, med[1]]),
        "both adverse": np.array([roe_low, DE_RED]),
    }
    rows = []
    for name, x_new in targets.items():
        dx_raw = x_new - med
        dx_std = dx_raw / sd                     # shock in standardised units
        dz = V.T @ dx_std                        # displacement in PC coordinates
        rows.append(dict(shock=name,
                         d_roe=dx_raw[0], d_de=dx_raw[1],
                         d_roe_sd=dx_std[0], d_de_sd=dx_std[1],
                         dz1=dz[0], dz2=dz[1],
                         mahalanobis=float(np.sqrt(dz @ (dz / lam)))))
    return pd.DataFrame(rows)


def delta_y(shocks, V, lam):
    """Change in y for each (beta, shock) pair, split by principal direction."""
    rows = []
    for _, s in shocks.iterrows():
        dz = np.array([s["dz1"], s["dz2"]])
        for b1 in BETA1:
            for b2 in BETA2:
                beta = np.array([b1, b2], float)
                g = V.T @ beta
                contrib = g * dz
                rows.append(dict(shock=s["shock"], beta1=b1, beta2=b2,
                                 dy_pc1=contrib[0], dy_pc2=contrib[1],
                                 dy_total=contrib.sum()))
    return pd.DataFrame(rows)


def worst_case(V, lam):
    """Over the compact set ||dx_std|| <= r the largest |dy| is r*||beta||, attained
    along beta itself, not along PC1. Verified numerically below."""
    rng = np.random.default_rng(SEED)
    out = []
    for b1 in BETA1:
        for b2 in BETA2:
            beta = np.array([b1, b2], float)
            ang = rng.uniform(0, 2 * np.pi, 20000)
            u = np.column_stack([np.cos(ang), np.sin(ang)])   # unit ball boundary
            dy = u @ beta
            out.append(dict(beta1=b1, beta2=b2,
                            max_dy_numeric=dy.max(),
                            norm_beta=np.linalg.norm(beta),
                            dy_along_pc1=float(V[:, 0] @ beta)))
    return pd.DataFrame(out)


def make_figure(Z, d, shocks, V, lam):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))

    ax = axes[0]
    idx = np.random.default_rng(SEED).choice(len(Z), size=min(4000, len(Z)),
                                             replace=False)
    ax.scatter(Z[idx, 0], Z[idx, 1], s=5, alpha=0.20, color="#64748b",
               edgecolors="none", label="issuer-months")
    for _, s in shocks.iterrows():
        ax.annotate("", xy=(s["dz1"], s["dz2"]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#b91c1c"))
        ax.text(s["dz1"], s["dz2"], "  " + s["shock"], fontsize=8,
                color="#b91c1c", va="center")
    ax.axhline(0, color="#333", lw=0.8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel(f"PC1 score  (eigenvalue {lam[0]:.3f})")
    ax.set_ylabel(f"PC2 score  (eigenvalue {lam[1]:.3f})")
    ax.set_title("Where the cluster centroid moves under a threshold shock",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    dsub = d.sample(min(4000, len(d)), random_state=SEED)
    ax.scatter(dsub["ROE"], dsub["DE"], s=5, alpha=0.20, color="#64748b",
               edgecolors="none")
    med = d.median()
    sd = d.std()
    for k, col in ((0, "#1d4ed8"), (1, "#ea580c")):
        v = V[:, k] * np.sqrt(lam[k]) * 2.2
        ax.annotate("", xy=(med["ROE"] + v[0] * sd["ROE"],
                            med["DE"] + v[1] * sd["DE"]),
                    xytext=(med["ROE"], med["DE"]),
                    arrowprops=dict(arrowstyle="->", lw=2.4, color=col))
        ax.text(med["ROE"] + v[0] * sd["ROE"], med["DE"] + v[1] * sd["DE"],
                f" PC{k+1}", color=col, fontsize=10, fontweight="bold")
    ax.axhline(DE_AMBER, color="#f59e0b", ls="--", lw=1.2)
    ax.axhline(DE_RED, color="#dc2626", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0], DE_AMBER, " DE amber 1.606", fontsize=7.5,
            color="#f59e0b", va="bottom")
    ax.text(ax.get_xlim()[0], DE_RED, " DE red 2.116", fontsize=7.5,
            color="#dc2626", va="bottom")
    ax.set_xlabel("ROE (winsorised)")
    ax.set_ylabel("DE (winsorised)")
    ax.set_title("Principal directions in the original coordinates",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.25)

    fig.suptitle("PCA of the two leading determinants, and the effect of a "
                 "threshold-level shock", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUTDIR, "fig_pca_shock.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {p}")


def main():
    print("=" * 96)
    print("PCA of (ROE, DE), the inverse map, and threshold-shock propagation to y")
    print("=" * 96)
    d = load_xy()
    cols = list(d.columns)
    mu = d.mean().to_numpy(float)
    sd = d.std(ddof=1).to_numpy(float)
    print(f"  rows {len(d):,}   winsorised at {WINSOR[0]:.0%}/{WINSOR[1]:.0%}")
    print(f"  mean  ROE {mu[0]:.4f}   DE {mu[1]:.4f}")
    print(f"  sd    ROE {sd[0]:.4f}   DE {sd[1]:.4f}")

    Sraw = np.cov(d.to_numpy(float).T, ddof=1)
    Zs = (d.to_numpy(float) - mu) / sd
    Scor = np.cov(Zs.T, ddof=1)

    report_basis("PCA on the raw covariance matrix", Sraw, cols)
    lam, V = report_basis("PCA on the correlation matrix (used below)", Scor, cols)

    rho = Scor[0, 1]
    print(f"\n  closed form for a correlation matrix: "
          f"lambda = 1 +/- |rho| = {1+abs(rho):.4f}, {1-abs(rho):.4f}")
    print(f"  eigenvectors are (1, {'+' if rho>0 else '-'}1)/sqrt(2) and its "
          f"orthogonal complement, independent of the size of rho")

    Z = Zs @ V

    g = gamma_table(V, lam)
    print("\n=== gamma = V'beta, and the variance of y by principal direction ===")
    print(f"{'b1':>3} {'b2':>3} {'gamma1':>9} {'gamma2':>9} "
          f"{'Var_PC1':>9} {'Var_PC2':>9} {'PC1 share':>10}")
    for _, r in g.iterrows():
        print(f"{int(r.beta1):>3} {int(r.beta2):>3} {r.gamma1:>9.4f} "
              f"{r.gamma2:>9.4f} {r.var_pc1:>9.4f} {r.var_pc2:>9.4f} "
              f"{100*r.share_pc1:>9.1f}%")

    shocks = shock_analysis(d, Scor, V, lam, cols, mu, sd)
    print("\n=== cluster displacement under a threshold shock ===")
    print(shocks.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    dy = delta_y(shocks, V, lam)
    print("\n=== change in y, split by principal direction (selected betas) ===")
    sel = dy[(dy.beta1.isin([1, 3, 5])) & (dy.beta2.isin([1, 4]))]
    print(sel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    wc = worst_case(V, lam)
    ok = np.allclose(wc["max_dy_numeric"], wc["norm_beta"], atol=2e-3)
    print(f"\n=== worst-case shock on the unit ball ||dx||<=1 ===")
    print(f"  max |dy| equals ||beta|| for every beta: {ok}")
    print("  the worst direction is beta/||beta||, NOT PC1; PC1 gives "
          "only |beta'v1|:")
    print(wc.head(6).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    make_figure(Z, d, shocks, V, lam)
    g.to_csv(os.path.join(OUTDIR, "pca_gamma.csv"), index=False)
    shocks.to_csv(os.path.join(OUTDIR, "pca_shocks.csv"), index=False)
    dy.to_csv(os.path.join(OUTDIR, "pca_delta_y.csv"), index=False)
    print("  wrote tex_out/pca_gamma.csv, pca_shocks.csv, pca_delta_y.csv")


if __name__ == "__main__":
    main()
