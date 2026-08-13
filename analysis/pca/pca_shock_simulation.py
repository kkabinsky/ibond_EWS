# -*- coding: utf-8 -*-
"""
pca_shock_simulation.py -- synthetic-data illustration of how a feature shock moves an
issuer cluster in PCA coordinates, and what that does to y.

DESIGN
    Baseline issuers are drawn from a bivariate normal whose mean, dispersion and
    correlation match the winsorised (ROE, DE) panel, so the geometry is realistic
    while the data are entirely synthetic. A shocked cohort is then displaced by a
    fixed vector corresponding to an anomaly threshold, and both cohorts are projected
    into the SAME principal basis.

    Keeping the basis fixed is the point. Re-estimating the eigenvectors after the
    shock rotates the coordinate system along with the data and largely hides the
    displacement, which is the usual reason a shock looks invisible in a PCA scatter
    that was refitted on the post-shock sample.

OUTPUT
    tex_out/fig_pca_shock_simulation.png with four panels:
      A  original (ROE, DE) coordinates with the anomaly thresholds
      B  the same cohorts in PCA coordinates, with centroid displacement arrows
      C  distribution of y before and after each shock, for one choice of beta
      D  the displacement decomposed into its PC1 and PC2 contributions to y

RUN
    python pca_shock_simulation.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42
N_BASE = 4000
N_SHOCK = 700

# moments taken from the winsorised iBond panel so the synthetic geometry is realistic
MU = np.array([4.5128, 1.5249])          # ROE, DE
SD = np.array([17.2001, 1.3998])
RHO = -0.2737

DE_AMBER, DE_RED = 1.606, 2.116
ROE_P10_SHIFT = -22.0820                 # ROE median -> 10th percentile

SHOCKS = {
    "DE to red": np.array([0.0, DE_RED - 1.1837]),        # median DE 1.1837
    "ROE to p10": np.array([ROE_P10_SHIFT, 0.0]),
    "both adverse": np.array([ROE_P10_SHIFT, DE_RED - 1.1837]),
}
COLORS = {"DE to red": "#d97706", "ROE to p10": "#2563eb",
          "both adverse": "#b91c1c"}
BETAS = [(1, 1), (3, 4), (5, 1)]


def simulate():
    rng = np.random.default_rng(SEED)
    corr = np.array([[1.0, RHO], [RHO, 1.0]])
    cov = np.outer(SD, SD) * corr
    X = rng.multivariate_normal(MU, cov, size=N_BASE)
    return X, rng


def basis_from(X):
    """Principal basis of the BASELINE cohort, standardised. Returned once and reused
    for every shocked cohort."""
    mu = X.mean(0)
    sd = X.std(0, ddof=1)
    Z = (X - mu) / sd
    C = np.cov(Z.T, ddof=1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if V[0, 0] < 0:
        V[:, 0] = -V[:, 0]
    if V[0, 1] < 0:
        V[:, 1] = -V[:, 1]
    return mu, sd, w, V


def project(X, mu, sd, V):
    return ((X - mu) / sd) @ V


def main():
    print("=" * 90)
    print("Synthetic simulation: feature shock seen in PCA coordinates")
    print("=" * 90)
    X, rng = simulate()
    mu, sd, lam, V = basis_from(X)
    print(f"  baseline n = {N_BASE:,}   shocked cohort n = {N_SHOCK:,} each")
    print(f"  eigenvalues  PC1 {lam[0]:.4f} ({100*lam[0]/lam.sum():.1f}%)   "
          f"PC2 {lam[1]:.4f} ({100*lam[1]/lam.sum():.1f}%)")
    print(f"  PC1 = ({V[0,0]:+.4f}, {V[1,0]:+.4f})   "
          f"PC2 = ({V[0,1]:+.4f}, {V[1,1]:+.4f})")

    Zb = project(X, mu, sd, V)
    idx = rng.choice(N_BASE, size=N_SHOCK, replace=False)
    cohorts = {}
    for name, dx in SHOCKS.items():
        Xs = X[idx] + dx
        cohorts[name] = dict(X=Xs, Z=project(Xs, mu, sd, V), dx=dx)

    print("\n  centroid displacement in PC coordinates")
    for name, c in cohorts.items():
        dz = c["Z"].mean(0) - Zb.mean(0)
        c["dz"] = dz
        print(f"    {name:14s} dz1 {dz[0]:+.4f}  dz2 {dz[1]:+.4f}   "
              f"|dz| {np.linalg.norm(dz):.4f}")

    print("\n  change in y (mean over the cohort)")
    print(f"    {'shock':14s} " + "  ".join(f"beta={b}" for b in BETAS))
    for name, c in cohorts.items():
        row = []
        for b in BETAS:
            beta = np.array(b, float)
            dy = (c["X"] - X[idx]) @ beta
            row.append(f"{dy.mean():+9.3f}")
        print(f"    {name:14s} " + "  ".join(row))

    # ---------------------------------------------------------------- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.4))

    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.18, color="#94a3b8",
               edgecolors="none", label="baseline")
    for name, c in cohorts.items():
        ax.scatter(c["X"][:, 0], c["X"][:, 1], s=6, alpha=0.30,
                   color=COLORS[name], edgecolors="none", label=name)
    ax.axhline(DE_AMBER, color="#f59e0b", ls="--", lw=1.1)
    ax.axhline(DE_RED, color="#dc2626", ls="--", lw=1.1)
    ax.text(ax.get_xlim()[0], DE_AMBER, " DE amber", fontsize=7.5,
            color="#f59e0b", va="bottom")
    ax.text(ax.get_xlim()[0], DE_RED, " DE red", fontsize=7.5,
            color="#dc2626", va="bottom")
    ax.set_xlabel("ROE"); ax.set_ylabel("DE")
    ax.set_title("A. Original coordinates", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.5, markerscale=2, loc="upper right")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.scatter(Zb[:, 0], Zb[:, 1], s=6, alpha=0.18, color="#94a3b8",
               edgecolors="none", label="baseline")
    for name, c in cohorts.items():
        ax.scatter(c["Z"][:, 0], c["Z"][:, 1], s=6, alpha=0.30,
                   color=COLORS[name], edgecolors="none", label=name)
    for name, c in cohorts.items():
        ax.annotate("", xy=tuple(Zb.mean(0) + c["dz"]), xytext=tuple(Zb.mean(0)),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=COLORS[name],
                                    mutation_scale=18))
    ax.axhline(0, color="#334155", lw=0.8); ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel(f"PC1 score   (eigenvalue {lam[0]:.3f}, "
                  f"{100*lam[0]/lam.sum():.0f}% of feature variance)")
    ax.set_ylabel(f"PC2 score   (eigenvalue {lam[1]:.3f})")
    ax.set_title("B. PCA coordinates, basis fixed on the baseline cohort",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.5, markerscale=2, loc="upper right")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    beta = np.array(BETAS[1], float)
    y0 = X[idx] @ beta
    ax.hist(y0, bins=60, alpha=0.55, color="#94a3b8", label="baseline")
    for name, c in cohorts.items():
        ax.hist(c["X"] @ beta, bins=60, alpha=0.45, color=COLORS[name], label=name)
    ax.set_xlabel(r"$y=\beta_1\,$ROE$+\beta_2\,$DE" +
                  f"    with beta = {tuple(int(b) for b in beta)}")
    ax.set_ylabel("count")
    ax.set_title("C. Distribution of y, before and after the shock",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1, 1]
    names = list(cohorts)
    w = 0.8 / len(BETAS)
    xs = np.arange(len(names))
    for i, b in enumerate(BETAS):
        beta = np.array(b, float)
        g = V.T @ beta
        c1 = [cohorts[n]["dz"][0] * g[0] for n in names]
        c2 = [cohorts[n]["dz"][1] * g[1] for n in names]
        pos = xs + i * w - 0.4 + w / 2
        ax.bar(pos, c1, width=w * 0.92, color="#1d4ed8", alpha=0.85,
               label="via PC1" if i == 0 else None)
        ax.bar(pos, c2, width=w * 0.92, bottom=c1, color="#ea580c", alpha=0.85,
               label="via PC2" if i == 0 else None)
        for p, t in zip(pos, np.array(c1) + np.array(c2)):
            ax.text(p, t + (0.4 if t >= 0 else -0.9), f"{t:+.1f}", ha="center",
                    fontsize=7.2, fontweight="bold")
        ax.text(pos.mean(), ax.get_ylim()[1] * 0.02, "", ha="center")
    ax.axhline(0, color="#334155", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n" + "  ".join(str(b) for b in BETAS)
                        for n in names], fontsize=8)
    ax.set_ylabel(r"$\Delta y$ contribution")
    ax.set_title(r"D. $\Delta y=\gamma_1\Delta z_1+\gamma_2\Delta z_2$, "
                 r"three $\beta$ per shock", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle("Feature shock in PCA space: synthetic issuers, moments matched to "
                 "the winsorised (ROE, DE) panel", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(OUTDIR, "fig_pca_shock_simulation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {p}")

    # what happens if the basis is refitted after the shock
    print("\n  refitting the basis on a shocked cohort instead of fixing it:")
    for name, c in cohorts.items():
        mu2, sd2, lam2, V2 = basis_from(c["X"])
        dz_ref = project(c["X"], mu2, sd2, V2).mean(0) - Zb.mean(0)
        print(f"    {name:14s} |dz| fixed basis {np.linalg.norm(c['dz']):.4f}   "
              f"refitted {np.linalg.norm(dz_ref):.4f}")


if __name__ == "__main__":
    main()
