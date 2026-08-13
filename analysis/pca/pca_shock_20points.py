# -*- coding: utf-8 -*-
"""
pca_shock_20points.py -- a small, readable version of the shock experiment.

Twenty synthetic issuers, two determinants (ROE, DE), three shock levels, and for each
state the resulting 12-month default probability and its momentum.

    linear index      eta = b0 + b1*ROE + b2*DE
    PD12              sigma(eta)
    momentum          PD12 at the shocked state / PD12 at baseline

The shock is applied in equal steps, so level k moves every issuer by k times the same
vector. The PCA basis is computed once on the baseline states and then held fixed;
projecting the shocked states with a refitted basis would remove the displacement,
because PCA centres the data and a common shock is a pure translation.

RUN
    python pca_shock_20points.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
os.makedirs(OUTDIR, exist_ok=True)

SEED = 7
N = 20
B0, B1, B2 = -4.0, -0.05, 0.80          # index: ROE lowers risk, DE raises it
STEP = np.array([-5.0, 0.30])           # one shock step: ROE -5, DE +0.30
LEVELS = [0, 1, 2, 3]
LCOL = {0: "#64748b", 1: "#f59e0b", 2: "#ea580c", 3: "#b91c1c"}
LNAME = {0: "baseline", 1: "shock 1", 2: "shock 2", 3: "shock 3"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_issuers():
    rng = np.random.default_rng(SEED)
    roe = rng.normal(8.0, 6.0, N)
    de = np.clip(1.2 - 0.03 * (roe - 8.0) + rng.normal(0, 0.45, N), 0.15, None)
    return np.column_stack([roe, de])


def pd12(X):
    return sigmoid(B0 + B1 * X[:, 0] + B2 * X[:, 1])


def main():
    X0 = make_issuers()
    p0 = pd12(X0)

    # basis fixed on the baseline states
    mu, sd = X0.mean(0), X0.std(0, ddof=1)
    Z0 = (X0 - mu) / sd
    C = np.cov(Z0.T, ddof=1)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    if V[0, 0] < 0:
        V[:, 0] = -V[:, 0]
    if V[0, 1] < 0:
        V[:, 1] = -V[:, 1]

    print("=" * 84)
    print(f"{N} synthetic issuers, three shock levels, step = "
          f"(ROE {STEP[0]:+.1f}, DE {STEP[1]:+.2f})")
    print("=" * 84)
    print(f"  index: eta = {B0} + ({B1})*ROE + {B2}*DE ,  PD12 = sigmoid(eta)")
    print(f"  correlation(ROE, DE) on the baseline = {C[0,1]:+.4f}")
    print(f"  PC1 eigenvalue {w[0]:.4f} ({100*w[0]/w.sum():.1f}%)   "
          f"vector ({V[0,0]:+.4f}, {V[1,0]:+.4f})")
    print(f"  PC2 eigenvalue {w[1]:.4f} ({100*w[1]/w.sum():.1f}%)   "
          f"vector ({V[0,1]:+.4f}, {V[1,1]:+.4f})")

    rows = []
    states = {}
    for k in LEVELS:
        Xk = X0 + k * STEP
        Zk = ((Xk - mu) / sd) @ V
        pk = pd12(Xk)
        states[k] = dict(X=Xk, Z=Zk, pd=pk)
        for i in range(N):
            rows.append(dict(issuer=f"F{i+1:02d}", level=k,
                             ROE=Xk[i, 0], DE=Xk[i, 1],
                             PC1=Zk[i, 0], PC2=Zk[i, 1],
                             PD12=pk[i], momentum=pk[i] / p0[i]))
    d = pd.DataFrame(rows)

    print("\n=== all 20 issuers, baseline and three shock levels ===")
    print(f"{'issuer':7s}{'lvl':>4}{'ROE':>8}{'DE':>7}{'PC1':>8}{'PC2':>8}"
          f"{'PD12':>9}{'moment.':>9}")
    for i in range(N):
        for k in LEVELS:
            r = d[(d.issuer == f"F{i+1:02d}") & (d.level == k)].iloc[0]
            print(f"{r.issuer if k==0 else '':7s}{k:>4}{r.ROE:>8.2f}{r.DE:>7.2f}"
                  f"{r.PC1:>8.3f}{r.PC2:>8.3f}{r.PD12:>9.4f}{r.momentum:>9.3f}")
        print("  " + "-" * 58)

    print("=== cohort means ===")
    g = d.groupby("level").agg(ROE=("ROE", "mean"), DE=("DE", "mean"),
                               PC1=("PC1", "mean"), PC2=("PC2", "mean"),
                               PD12=("PD12", "mean"),
                               momentum=("momentum", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  centroid displacement from baseline, in PC coordinates")
    for k in LEVELS[1:]:
        dz = np.array([g.PC1[k] - g.PC1[0], g.PC2[k] - g.PC2[0]])
        print(f"    level {k}:  dz1 {dz[0]:+.4f}   dz2 {dz[1]:+.4f}   "
              f"|dz| {np.linalg.norm(dz):.4f}   "
              f"mean PD12 {g.PD12[k]:.4f}   mean momentum {g.momentum[k]:.3f}")

    # ------------------------------------------------------------- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))

    ax = axes[0]
    for k in LEVELS:
        ax.scatter(states[k]["X"][:, 0], states[k]["X"][:, 1], s=55,
                   color=LCOL[k], alpha=0.85, edgecolors="white", linewidth=0.8,
                   label=LNAME[k], zorder=3)
    for i in range(N):
        xs = [states[k]["X"][i, 0] for k in LEVELS]
        ys = [states[k]["X"][i, 1] for k in LEVELS]
        ax.plot(xs, ys, color="#94a3b8", lw=0.8, alpha=0.7, zorder=1)
    for i in range(N):
        ax.annotate(f"F{i+1}", (states[0]["X"][i, 0], states[0]["X"][i, 1]),
                    fontsize=6.5, color="#334155",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("ROE"); ax.set_ylabel("DE")
    ax.set_title("A. Original coordinates", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    for k in LEVELS:
        ax.scatter(states[k]["Z"][:, 0], states[k]["Z"][:, 1], s=55,
                   color=LCOL[k], alpha=0.85, edgecolors="white", linewidth=0.8,
                   label=LNAME[k], zorder=3)
    for i in range(N):
        xs = [states[k]["Z"][i, 0] for k in LEVELS]
        ys = [states[k]["Z"][i, 1] for k in LEVELS]
        ax.plot(xs, ys, color="#94a3b8", lw=0.8, alpha=0.7, zorder=1)
    c0 = states[0]["Z"].mean(0)
    for k in LEVELS[1:]:
        ck = states[k]["Z"].mean(0)
        ax.annotate("", xy=tuple(ck), xytext=tuple(c0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.6, color=LCOL[k],
                                    mutation_scale=18), zorder=5)
    ax.axhline(0, color="#334155", lw=0.8); ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel(f"PC1  ({100*w[0]/w.sum():.0f}% of variance)")
    ax.set_ylabel(f"PC2  ({100*w[1]/w.sum():.0f}%)")
    ax.set_title("B. PCA coordinates, basis fixed on the baseline",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2]
    for i in range(N):
        ax.plot(LEVELS, [states[k]["pd"][i] for k in LEVELS],
                marker="o", ms=3.5, lw=1.0, color="#94a3b8", alpha=0.65)
    ax.plot(LEVELS, [states[k]["pd"].mean() for k in LEVELS],
            marker="s", ms=8, lw=2.6, color="#b91c1c", label="cohort mean")
    for k in LEVELS:
        ax.annotate(f"{states[k]['pd'].mean():.3f}\nM={g.momentum[k]:.2f}",
                    (k, states[k]["pd"].mean()), fontsize=8, fontweight="bold",
                    color="#b91c1c", xytext=(6, -6), textcoords="offset points")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([LNAME[k] for k in LEVELS], fontsize=9)
    ax.set_xlabel("shock level"); ax.set_ylabel(r"$PD_{12}$")
    ax.set_title(r"C. $PD_{12}$ per issuer, and momentum $M$ of the mean",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"{N} synthetic issuers under three equal shock steps of "
                 f"(ROE {STEP[0]:+.0f}, DE {STEP[1]:+.2f})",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUTDIR, "fig_pca_shock_20points.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    d.to_csv(os.path.join(OUTDIR, "pca_shock_20points.csv"), index=False)
    print(f"\n  wrote {p}")
    print("  wrote tex_out/pca_shock_20points.csv")


if __name__ == "__main__":
    main()
