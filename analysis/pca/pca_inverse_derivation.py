# -*- coding: utf-8 -*-
"""
pca_inverse_derivation.py -- the inverse PCA map for the 20-point simulation, worked
through with explicit numbers.

THE CHAIN USED IN pca_shock_20points.py
    standardise      u = (x - mu) / sd          elementwise
    project          z = V' u
    inverse          u = V z                    because V'V = I so V^{-1} = V'
    de-standardise   x = mu + sd * (V z)        elementwise

    Only the second and third steps are the PCA proper; the first and fourth are
    scaling, and they must be inverted in the right order or the reconstruction is
    silently wrong.

TRUNCATION
    Dropping PC2 gives the rank-one reconstruction
        x_hat = mu + sd * (v1 z1)
    whose squared error, averaged over the sample, equals the discarded eigenvalue
    lambda2 in standardised units. That identity is checked numerically below, and the
    consequence for PD12 is reported, since a reconstruction error in x propagates
    through the logistic link and is not proportional to it.

RUN
    python pca_inverse_derivation.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pca_shock_20points import (B0, B1, B2, LEVELS, N, STEP, make_issuers,
                                sigmoid)
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
SHOW = 5                    # how many issuers to print in full detail


def build():
    X0 = make_issuers()
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
    return X0, mu, sd, w, V


def main():
    X0, mu, sd, lam, V = build()
    v1, v2 = V[:, 0], V[:, 1]
    r2 = np.sqrt(2.0)

    print("=" * 92)
    print("INVERSE PCA FOR THE 20-POINT SIMULATION")
    print("=" * 92)
    print(f"  mu = (ROE {mu[0]:.4f}, DE {mu[1]:.4f})")
    print(f"  sd = (ROE {sd[0]:.4f}, DE {sd[1]:.4f})")
    print(f"  V  = [ v1 v2 ] = [[{V[0,0]:+.6f} {V[0,1]:+.6f}]")
    print(f"                    [{V[1,0]:+.6f} {V[1,1]:+.6f}]]")
    print(f"  eigenvalues  lambda1 {lam[0]:.6f}   lambda2 {lam[1]:.6f}")
    print(f"  V'V = I ? {np.allclose(V.T @ V, np.eye(2))}    "
          f"det(V) = {np.linalg.det(V):+.6f}")

    print("\n  because the eigenvectors are (1,-1)/sqrt2 and (1,1)/sqrt2 exactly,")
    print("  the forward and inverse maps have closed forms:")
    print("      z1 = (u1 - u2)/sqrt2        u1 = (z1 + z2)/sqrt2")
    print("      z2 = (u1 + u2)/sqrt2        u2 = (z2 - z1)/sqrt2")

    rows = []
    print("\n" + "=" * 92)
    print(f"STEP-BY-STEP FOR THE FIRST {SHOW} ISSUERS, ALL {len(LEVELS)} STATES")
    print("=" * 92)
    for i in range(N):
        for k in LEVELS:
            x = X0[i] + k * STEP
            u = (x - mu) / sd                       # standardise
            z = V.T @ u                             # project
            u_back = V @ z                          # inverse projection
            x_back = mu + sd * u_back               # de-standardise
            # rank-one reconstruction, PC2 discarded
            u1c = v1 * z[0]
            x1c = mu + sd * u1c
            p_true = sigmoid(B0 + B1 * x[0] + B2 * x[1])
            p_rank1 = sigmoid(B0 + B1 * x1c[0] + B2 * x1c[1])
            rows.append(dict(issuer=f"F{i+1:02d}", level=k,
                             ROE=x[0], DE=x[1], u1=u[0], u2=u[1],
                             z1=z[0], z2=z[1],
                             ROE_back=x_back[0], DE_back=x_back[1],
                             err_full=np.linalg.norm(x - x_back),
                             ROE_r1=x1c[0], DE_r1=x1c[1],
                             err_rank1_std=np.linalg.norm(u - u1c),
                             PD12=p_true, PD12_rank1=p_rank1))
            if i < SHOW:
                if k == 0:
                    print(f"\n--- F{i+1:02d} ---")
                print(f"  level {k}")
                print(f"    x      = ({x[0]:+9.4f}, {x[1]:+7.4f})")
                print(f"    u      = (x-mu)/sd = (({x[0]:+.4f}-{mu[0]:.4f})/{sd[0]:.4f},"
                      f" ({x[1]:+.4f}-{mu[1]:.4f})/{sd[1]:.4f})"
                      f" = ({u[0]:+.4f}, {u[1]:+.4f})")
                print(f"    z1     = (u1-u2)/sqrt2 = ({u[0]:+.4f} - {u[1]:+.4f})/{r2:.4f}"
                      f" = {z[0]:+.4f}")
                print(f"    z2     = (u1+u2)/sqrt2 = ({u[0]:+.4f} + {u[1]:+.4f})/{r2:.4f}"
                      f" = {z[1]:+.4f}")
                print(f"    back:  u1 = (z1+z2)/sqrt2 = ({z[0]:+.4f} + {z[1]:+.4f})"
                      f"/{r2:.4f} = {u_back[0]:+.4f}")
                print(f"           u2 = (z2-z1)/sqrt2 = ({z[1]:+.4f} - {z[0]:+.4f})"
                      f"/{r2:.4f} = {u_back[1]:+.4f}")
                print(f"           x  = mu + sd*u = ({mu[0]:.4f} + {sd[0]:.4f}"
                      f"*{u_back[0]:+.4f}, {mu[1]:.4f} + {sd[1]:.4f}*{u_back[1]:+.4f})")
                print(f"              = ({x_back[0]:+9.4f}, {x_back[1]:+7.4f})   "
                      f"error {np.linalg.norm(x-x_back):.2e}")
                print(f"    rank-1 (PC2 dropped): x_hat = "
                      f"({x1c[0]:+9.4f}, {x1c[1]:+7.4f})   "
                      f"PD12 {p_true:.4f} -> {p_rank1:.4f}")

    d = pd.DataFrame(rows)

    print("\n" + "=" * 92)
    print("CHECKS OVER ALL 80 STATES")
    print("=" * 92)
    print(f"  max reconstruction error, both components kept : "
          f"{d.err_full.max():.3e}   (machine precision)")
    mse_std = (d.err_rank1_std ** 2).mean()
    print(f"  mean squared error in standardised units, PC2 dropped : {mse_std:.6f}")
    print(f"  discarded eigenvalue lambda2                          : {lam[1]:.6f}")
    print(f"  the two agree on the baseline states only, because the shocked states")
    print(f"  are not centred on the baseline mean; on level 0 alone:")
    m0 = (d[d.level == 0].err_rank1_std ** 2).mean()
    print(f"      level-0 mean squared error = {m0:.6f}  vs  lambda2 = {lam[1]:.6f}"
          f"   ratio {m0/lam[1]:.4f}")

    print(f"\n  effect of dropping PC2 on PD12")
    d["pd_abs_err"] = (d.PD12_rank1 - d.PD12).abs()
    print(f"    mean absolute error {d.pd_abs_err.mean():.4f}   "
          f"max {d.pd_abs_err.max():.4f}")
    for k in LEVELS:
        s = d[d.level == k]
        print(f"    level {k}: true mean PD12 {s.PD12.mean():.4f}   "
              f"rank-1 mean {s.PD12_rank1.mean():.4f}   "
              f"mean abs error {s.pd_abs_err.mean():.4f}")

    out = os.path.join(OUTDIR, "pca_inverse_20points.csv")
    d.to_csv(out, index=False)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
