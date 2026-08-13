# -*- coding: utf-8 -*-
"""
pca_inverse_surface.py -- the inverse PCA map for two and for three determinants,
drawn as response surfaces, on the same 20-issuer shock simulation.

WHAT THIS ADDS OVER pca_inverse_derivation.py
    That script worked the algebra through with numbers. This one asks the question the
    numbers do not answer on their own: if you throw away a principal component and map
    back, how wrong is the DEFAULT PROBABILITY, and where on the plane is it wrong?
    The answer is a surface, not a table, because the logistic link is non-linear and a
    fixed reconstruction error in x turns into a different error in PD depending on
    where you stand.

TWO DETERMINANTS
    x = (ROE, DE), the same 20 synthetic issuers and the same three shock levels used
    in pca_shock_20points.py, so every number here lines up with
    tex_out/pca_inverse_20points.csv.

THREE DETERMINANTS
    x = (ROE, DE, TDTA), calibrated to the correlations of the real winsorised panel
    (-0.2733, -0.1005, 0.7104), so the geometry is realistic while the data stay
    synthetic. Three determinants give two ways to truncate, rank two and rank one, and
    the gap between them is the point of the exercise.

THE IDENTITY THAT BREAKS UNDER SHOCK
    On the sample the basis was fitted to, the mean squared rank-k reconstruction error
    in standardised units equals the sum of the discarded eigenvalues, times (n-1)/n.
    That holds at the baseline and only there. Once the cohort is shocked it translates
    away from the centre the basis was built around, the discarded direction starts
    carrying real signal, and the error grows. Truncated PCA therefore degrades exactly
    in the regime the shock analysis is about. This script measures that.

RUN
    python pca_inverse_surface.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)

from pca_shock_20points import B0, B1, B2, LEVELS, N, STEP, make_issuers, sigmoid
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
os.makedirs(OUTDIR, exist_ok=True)

SEED3 = 11
GRID = 46
LCOL = {0: "#64748b", 1: "#f59e0b", 2: "#ea580c", 3: "#b91c1c"}
LNAME = {0: "baseline", 1: "shock 1", 2: "shock 2", 3: "shock 3"}

# three-determinant index, calibrated so the baseline sits at a plausible PD
C0, C1, C2, C3 = -4.6, -0.05, 0.80, 1.50
STEP3 = np.array([-5.0, 0.30, 0.05])

# correlations of the real winsorised panel, ROE / DE / TDTA
RHO = np.array([[1.0000, -0.2733, -0.1005],
                [-0.2733, 1.0000, 0.7104],
                [-0.1005, 0.7104, 1.0000]])
MU3 = np.array([6.101523, 1.048342, 0.480000])
SD3 = np.array([4.687106, 0.442937, 0.130000])


# ---------------------------------------------------------------- helpers
def pca_fit(X):
    """Correlation-matrix PCA. Returns mu, sd, eigenvalues, eigenvectors as columns."""
    mu, sd = X.mean(0), X.std(0, ddof=1)
    Z = (X - mu) / sd
    C = np.corrcoef(Z.T)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return mu, sd, w[o], V[:, o]


def project(X, mu, sd, V):
    return ((X - mu) / sd) @ V


def inverse(Zk, mu, sd, V, k):
    """Map scores back to the original units keeping only the first k components."""
    return mu + sd * (Zk[:, :k] @ V[:, :k].T)


def report_eig(name, mu, sd, w, V, cols):
    print(f"\n=== {name} ===")
    print("  mu = " + ", ".join(f"{c} {m:.6f}" for c, m in zip(cols, mu)))
    print("  sd = " + ", ".join(f"{c} {s:.6f}" for c, s in zip(cols, sd)))
    for k in range(len(w)):
        vec = ", ".join(f"{c} {V[i, k]:+.6f}" for i, c in enumerate(cols))
        print(f"  PC{k+1}: eigenvalue {w[k]:.6f} "
              f"({100*w[k]/w.sum():.2f}% of variance)   eigenvector ({vec})")
    print(f"  check: sum of eigenvalues {w.sum():.6f} = number of determinants "
          f"{len(w)}")


# ---------------------------------------------------------------- 2 determinants
def part_two():
    X0 = make_issuers()
    mu, sd, w, V = pca_fit(X0)
    report_eig("สอง determinants: ROE, DE", mu, sd, w, V, ["ROE", "DE"])

    rows = []
    for k in LEVELS:
        Xk = X0 + k * STEP
        Zk = project(Xk, mu, sd, V)
        Xr = inverse(Zk, mu, sd, V, 1)
        eta = B0 + B1 * Xk[:, 0] + B2 * Xk[:, 1]
        etr = B0 + B1 * Xr[:, 0] + B2 * Xr[:, 1]
        err = np.sqrt((((Xk - Xr) / sd) ** 2).sum(1))
        rows.append(pd.DataFrame(dict(
            level=k, ROE=Xk[:, 0], DE=Xk[:, 1], z1=Zk[:, 0], z2=Zk[:, 1],
            ROE_r1=Xr[:, 0], DE_r1=Xr[:, 1], err_std=err,
            PD=sigmoid(eta), PD_r1=sigmoid(etr))))
    d = pd.concat(rows, ignore_index=True)
    d["pd_err"] = (d.PD - d.PD_r1).abs()

    print("\n  ผลของการตัด PC2 ทิ้ง แยกตามระดับ shock")
    print("  level |  MSE(std)  | lambda2*(n-1)/n |  PD เฉลี่ย | PD rank1 | |ผิด| เฉลี่ย")
    for k, g in d.groupby("level"):
        print(f"    {k}   |  {(g.err_std**2).mean():8.6f}  |    {w[1]*(N-1)/N:8.6f}     "
              f"|  {g.PD.mean():.6f} | {g.PD_r1.mean():.6f} |  {g.pd_err.mean():.6f}")

    # --- figure ---------------------------------------------------------
    g1 = np.linspace(d.ROE.min() - 1, d.ROE.max() + 1, GRID)
    g2 = np.linspace(d.DE.min() - 0.1, d.DE.max() + 0.1, GRID)
    G1, G2 = np.meshgrid(g1, g2)
    P_true = sigmoid(B0 + B1 * G1 + B2 * G2)

    flat = np.column_stack([G1.ravel(), G2.ravel()])
    Zf = project(flat, mu, sd, V)
    Rf = inverse(Zf, mu, sd, V, 1)
    P_r1 = sigmoid(B0 + B1 * Rf[:, 0] + B2 * Rf[:, 1]).reshape(G1.shape)

    fig = plt.figure(figsize=(13.2, 10.4))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot_surface(G1, G2, P_true, cmap="viridis", alpha=0.72,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    for k in LEVELS:
        g = d[d.level == k]
        ax.scatter(g.ROE, g.DE, g.PD, s=26, color=LCOL[k], depthshade=False,
                   edgecolors="white", linewidth=0.4, label=LNAME[k])
    ax.set_xlabel("ROE", fontsize=9)
    ax.set_ylabel("DE", fontsize=9)
    ax.set_zlabel("PD12", fontsize=9)
    ax.set_title("A. true surface, 20 issuers at 4 shock levels",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.view_init(elev=24, azim=-128)

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.plot_surface(G1, G2, P_r1, cmap="magma", alpha=0.72,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    for k in LEVELS:
        g = d[d.level == k]
        ax.scatter(g.ROE_r1, g.DE_r1, g.PD_r1, s=26, color=LCOL[k],
                   depthshade=False, edgecolors="white", linewidth=0.4)
    ax.set_xlabel("ROE", fontsize=9)
    ax.set_ylabel("DE", fontsize=9)
    ax.set_zlabel("PD12 from rank-1", fontsize=9)
    ax.set_title("B. after inverse PCA keeping PC1 only",
                 fontsize=10.5, fontweight="bold")
    ax.view_init(elev=24, azim=-128)

    ax = fig.add_subplot(2, 2, 3)
    im = ax.contourf(G1, G2, P_r1 - P_true, levels=24, cmap="coolwarm")
    cs = ax.contour(G1, G2, P_r1 - P_true, levels=[0], colors="black",
                    linewidths=1.2)
    ax.clabel(cs, fmt={0: "no error"}, fontsize=8)
    for k in LEVELS:
        g = d[d.level == k]
        ax.scatter(g.ROE, g.DE, s=22, color=LCOL[k], edgecolors="white",
                   linewidth=0.5, zorder=5)
    # the discarded direction, drawn through the centre of the baseline cloud
    v2 = V[:, 1] * sd
    ax.annotate("", xy=(mu[0] + 2.2 * v2[0], mu[1] + 2.2 * v2[1]),
                xytext=(mu[0] - 2.2 * v2[0], mu[1] - 2.2 * v2[1]),
                arrowprops=dict(arrowstyle="<->", color="#111827", lw=1.6))
    ax.text(mu[0] + 2.4 * v2[0], mu[1] + 2.4 * v2[1], " PC2 (dropped)",
            fontsize=8.5, fontweight="bold")
    ax.set_xlabel("ROE", fontsize=9)
    ax.set_ylabel("DE", fontsize=9)
    ax.set_title("C. error introduced by dropping PC2:  PD(rank-1) - PD(true)",
                 fontsize=10.5, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    ax = fig.add_subplot(2, 2, 4)
    ks = sorted(d.level.unique())
    mse = [float((d[d.level == k].err_std ** 2).mean()) for k in ks]
    ax.bar(ks, mse, color=[LCOL[k] for k in ks], alpha=0.9, width=0.62)
    ax.axhline(w[1] * (N - 1) / N, color="#111827", ls="--", lw=1.6,
               label=f"$\\lambda_2 (n-1)/n$ = {w[1]*(N-1)/N:.4f}")
    for k, v in zip(ks, mse):
        ax.text(k, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(ks)
    ax.set_xticklabels([LNAME[k] for k in ks], fontsize=8.5)
    ax.set_ylabel("mean squared rank-1 error, standardised units")
    ax.set_title("D. the identity holds at baseline and fails under shock",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Inverse PCA with two determinants: what dropping PC2 costs in PD",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = os.path.join(OUTDIR, "fig_pca_surface_2f.png")
    fig.savefig(fn, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {fn}")

    d.to_csv(os.path.join(OUTDIR, "pca_surface_2f.csv"), index=False)
    return d, (mu, sd, w, V)


# ---------------------------------------------------------------- 3 determinants
def make_issuers3():
    rng = np.random.default_rng(SEED3)
    L = np.linalg.cholesky(RHO)
    Z = rng.standard_normal((N, 3)) @ L.T
    X = MU3 + SD3 * Z
    X[:, 1] = np.clip(X[:, 1], 0.15, None)
    X[:, 2] = np.clip(X[:, 2], 0.05, 0.95)
    return X


def part_three():
    X0 = make_issuers3()
    mu, sd, w, V = pca_fit(X0)
    report_eig("สาม determinants: ROE, DE, TDTA", mu, sd, w, V, ["ROE", "DE", "TDTA"])

    rows = []
    for k in LEVELS:
        Xk = X0 + k * STEP3
        Zk = project(Xk, mu, sd, V)
        rec = {r: inverse(Zk, mu, sd, V, r) for r in (1, 2)}
        eta = C0 + C1 * Xk[:, 0] + C2 * Xk[:, 1] + C3 * Xk[:, 2]
        rec_eta = {r: C0 + C1 * X[:, 0] + C2 * X[:, 1] + C3 * X[:, 2]
                   for r, X in rec.items()}
        rows.append(pd.DataFrame(dict(
            level=k, ROE=Xk[:, 0], DE=Xk[:, 1], TDTA=Xk[:, 2],
            z1=Zk[:, 0], z2=Zk[:, 1], z3=Zk[:, 2],
            err_r2=np.sqrt((((Xk - rec[2]) / sd) ** 2).sum(1)),
            err_r1=np.sqrt((((Xk - rec[1]) / sd) ** 2).sum(1)),
            PD=sigmoid(eta), PD_r2=sigmoid(rec_eta[2]), PD_r1=sigmoid(rec_eta[1]))))
    d = pd.concat(rows, ignore_index=True)
    d["pd_err_r2"] = (d.PD - d.PD_r2).abs()
    d["pd_err_r1"] = (d.PD - d.PD_r1).abs()

    print("\n  ผลของการตัดองค์ประกอบทิ้ง แยกตามระดับ shock")
    print("  level | MSE rank2 | lam3*(n-1)/n | MSE rank1 | (lam2+lam3)*(n-1)/n "
          "| PD    | |ผิด| r2 | |ผิด| r1")
    for k, g in d.groupby("level"):
        print(f"    {k}   | {(g.err_r2**2).mean():9.6f} | {w[2]*(N-1)/N:12.6f} "
              f"| {(g.err_r1**2).mean():9.6f} | {(w[1]+w[2])*(N-1)/N:19.6f} "
              f"| {g.PD.mean():.4f} | {g.pd_err_r2.mean():8.6f} "
              f"| {g.pd_err_r1.mean():8.6f}")

    # --- figure: PD surfaces over (ROE, DE) sliced at three levels of TDTA ----
    g1 = np.linspace(d.ROE.min() - 1, d.ROE.max() + 1, GRID)
    g2 = np.linspace(d.DE.min() - 0.1, d.DE.max() + 0.1, GRID)
    G1, G2 = np.meshgrid(g1, g2)
    slices = np.quantile(d.TDTA, [0.10, 0.50, 0.90])

    fig = plt.figure(figsize=(13.6, 9.6))
    for i, t in enumerate(slices):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        P = sigmoid(C0 + C1 * G1 + C2 * G2 + C3 * t)
        ax.plot_surface(G1, G2, P, cmap="viridis", alpha=0.75, linewidth=0,
                        rstride=1, cstride=1)
        near = d[np.abs(d.TDTA - t) < 0.06]
        for k in LEVELS:
            g = near[near.level == k]
            if len(g):
                ax.scatter(g.ROE, g.DE, g.PD, s=30, color=LCOL[k],
                           depthshade=False, edgecolors="white", linewidth=0.4,
                           label=LNAME[k] if i == 0 else None)
        ax.set_xlabel("ROE", fontsize=8.5)
        ax.set_ylabel("DE", fontsize=8.5)
        ax.set_zlabel("PD", fontsize=8.5)
        ax.set_title(f"slice at TDTA = {t:.3f}", fontsize=10, fontweight="bold")
        ax.view_init(elev=24, azim=-128)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(2, 3, 4)
    ks = sorted(d.level.unique())
    x = np.arange(len(ks))
    m2 = [float((d[d.level == k].err_r2 ** 2).mean()) for k in ks]
    m1 = [float((d[d.level == k].err_r1 ** 2).mean()) for k in ks]
    ax.bar(x - 0.19, m2, width=0.36, color="#2563eb", alpha=0.9, label="keep PC1+PC2")
    ax.bar(x + 0.19, m1, width=0.36, color="#b91c1c", alpha=0.9, label="keep PC1 only")
    ax.axhline(w[2] * (N - 1) / N, color="#2563eb", ls="--", lw=1.4)
    ax.axhline((w[1] + w[2]) * (N - 1) / N, color="#b91c1c", ls="--", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([LNAME[k] for k in ks], fontsize=8)
    ax.set_ylabel("mean squared error, standardised units", fontsize=9)
    ax.set_title("reconstruction error, dashed = discarded eigenvalues",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(2, 3, 5)
    p_true = [float(d[d.level == k].PD.mean()) for k in ks]
    p_r2 = [float(d[d.level == k].PD_r2.mean()) for k in ks]
    p_r1 = [float(d[d.level == k].PD_r1.mean()) for k in ks]
    ax.plot(x, p_true, "o-", color="#111827", lw=2.0, label="true")
    ax.plot(x, p_r2, "s--", color="#2563eb", lw=1.7, label="PC1+PC2")
    ax.plot(x, p_r1, "^--", color="#b91c1c", lw=1.7, label="PC1 only")
    for xi, v in zip(x, p_true):
        ax.text(xi, v, f"  {v:.4f}", fontsize=8, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([LNAME[k] for k in ks], fontsize=8)
    ax.set_ylabel("mean PD over the 20 issuers", fontsize=9)
    ax.set_title("PD after truncation follows the true path only at rank 2",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 6)
    lab = ["PC1", "PC2", "PC3"]
    ax.bar(lab, w, color=["#16a34a", "#2563eb", "#b91c1c"], alpha=0.9)
    for i, v in enumerate(w):
        ax.text(i, v, f"{v:.4f}\n{100*v/w.sum():.1f}%", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_ylabel("eigenvalue of the correlation matrix", fontsize=9)
    ax.set_title("three determinants: how the variance splits",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, w.max() * 1.28)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Inverse PCA with three determinants (ROE, DE, TDTA): "
                 "surfaces sliced on the third determinant",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fn = os.path.join(OUTDIR, "fig_pca_surface_3f.png")
    fig.savefig(fn, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {fn}")

    d.to_csv(os.path.join(OUTDIR, "pca_surface_3f.csv"), index=False)

    print("\n  ตัวอย่างแทนค่าทีละขั้น สามตัวแรกที่ระดับ shock 3")
    g = d[d.level == 3].head(3)
    print(g[["ROE", "DE", "TDTA", "z1", "z2", "z3", "PD", "PD_r2", "PD_r1"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return d, (mu, sd, w, V)


if __name__ == "__main__":
    part_two()
    part_three()
    print("\nเสร็จ")
