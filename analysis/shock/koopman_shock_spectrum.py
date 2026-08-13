# -*- coding: utf-8 -*-
"""
koopman_shock_spectrum.py -- Koopman eigenvalues of the 33-determinant dynamics on the
unit circle, one panel per model, before and after a determinant shock.

THE OPERATOR
    Consecutive issuer-months give snapshot pairs (x_t, x_{t+1}) within each issuer.
    A linear Koopman approximation K is fitted by ridge least squares,

        X_next  ~  X_now K',        K' = (X'X + rho I)^{-1} X'Y,

    the same estimator already used in koopman.py. Its eigenvalues describe how each
    dynamic mode of the determinant vector propagates one month forward: |lambda| < 1
    decays, |lambda| = 1 persists, |lambda| > 1 grows.

WHERE THE MODELS ENTER
    A raw Koopman operator on the determinants knows nothing about default, so it is
    identical for every model. Reweighting the observables by importance does NOT fix
    this: for a diagonal W the fitted operator becomes W^{-1} K W, a similarity
    transform, and eigenvalues are invariant under similarity. That route was tried
    first and produced four identical spectra, which is a property of the algebra
    rather than a finding about the models.

    The observable vector is therefore made genuinely model-specific:

        g_m(x) = [ x_{S_m} ,  PD_m(x) ]

    where S_m is the set of determinants that model m ranks highest by gain, and
    PD_m is that model's own predicted default probability, which is a nonlinear
    observable of the state. Restricting to a subspace is a projection, not a
    similarity, and appending PD_m adds a coordinate no other model shares, so the
    resulting operators differ in substance.

THE SHOCK
    The same adverse one-standard-deviation shock as in pairwise_shock_pd.py is applied
    to the leading determinants, and the operator is refitted on the shocked
    trajectories. The interesting quantity is not the absolute eigenvalues but whether
    the shock pushes modes outward, towards or past the unit circle.

RUN
    python koopman_shock_spectrum.py
    python koopman_shock_spectrum.py --shock-top 4
"""
from __future__ import annotations

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

MODELS = ["XGBoost", "CatBoost", "LightGBM", "Random Forest"]
MC = {"XGBoost": "#1f3a5f", "CatBoost": "#a8501a",
      "LightGBM": "#e0a52e", "Random Forest": "#2e7d4f"}
RIDGE = 1e-3
SHOCK_SD = 1.0
SHOCK_TOP = 5
MIN_PAIRS = 200
SUBSPACE = 8          # determinants kept per model, plus that model's PD


def fit_K(Xn, Xx, ridge=RIDGE):
    """Ridge least squares, matching fit_linear_koopman_K in koopman.py."""
    d = Xn.shape[1]
    A = Xn.T @ Xn + ridge * np.eye(d)
    B = Xn.T @ Xx
    return np.linalg.solve(A, B).T


def snapshots(panel, A, cols):
    """Consecutive within-issuer month pairs. Gaps in the calendar are skipped so a
    missing month is never treated as a one-step transition."""
    p = panel.copy()
    p["_i"] = np.arange(len(p))
    p["month_dt"] = pd.to_datetime(p["month"], errors="coerce")
    p = p.sort_values(["issuer_code", "month_dt"])
    now, nxt = [], []
    for _, g in p.groupby("issuer_code", sort=False):
        ii = g["_i"].to_numpy()
        dt = g["month_dt"].to_numpy()
        if len(ii) < 2:
            continue
        gap = (dt[1:] - dt[:-1]).astype("timedelta64[D]").astype(int)
        ok = (gap >= 28) & (gap <= 32)          # exactly one calendar month
        now.append(ii[:-1][ok]); nxt.append(ii[1:][ok])
    now = np.concatenate(now); nxt = np.concatenate(nxt)
    return A[now], A[nxt], len(now)


def spectrum(Gn, Gx):
    """Koopman eigenvalues for an already-built observable pair."""
    K = fit_K(Gn, Gx)
    ev = np.linalg.eigvals(K)
    return ev[np.argsort(-np.abs(ev))]


def observables(A, sub, pd_vec):
    """g(x) = [ x on the model's own subspace , logit of that model's PD ].

    The probability enters on the logit scale because it is bounded in [0,1] and a
    linear operator cannot respect that bound; the logit is unbounded and is the
    natural coordinate for a linear propagator."""
    q = np.clip(pd_vec, 1e-6, 1 - 1e-6)
    lg = np.log(q / (1 - q))
    lg = (lg - lg.mean()) / (lg.std(ddof=1) + 1e-12)
    return np.column_stack([A[:, sub], lg])


def half_life(lmbda):
    a = abs(lmbda)
    if a <= 0 or a >= 1:
        return np.inf
    return np.log(0.5) / np.log(a)


def main():
    shock_top = SHOCK_TOP
    if "--shock-top" in sys.argv:
        shock_top = int(sys.argv[sys.argv.index("--shock-top") + 1])

    print("=" * 96)
    print("Koopman spectrum of the 33-determinant dynamics, by model, before and "
          "after a shock")
    print("=" * 96)
    panel, X, y, cols = cl.load_panel(verbose=True)
    panel = panel.reset_index(drop=True)
    A = X.to_numpy(float)
    sd = A.std(0, ddof=1)
    mu = A.mean(0)
    As = (A - mu) / np.where(sd > 0, sd, 1.0)      # standardise once

    imp = pd.read_csv(out("importance_default_event.csv"))
    piv = imp.pivot_table(index="feature", columns="model", values="gain")
    order = piv.mean(1).sort_values(ascending=False)
    idx = {c: i for i, c in enumerate(cols)}
    shocked_feats = [f for f in order.index if f in idx][:shock_top]
    print(f"\n  determinants shocked: {', '.join(shocked_feats)}")

    # adverse direction from a logistic fit, as in the pairwise analysis
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(max_iter=5000, C=0.1,
                            class_weight="balanced").fit(As, y.to_numpy(int))
    beta = lg.coef_[0]
    Ash = As.copy()
    for f in shocked_feats:
        j = idx[f]
        Ash[:, j] += np.sign(beta[j] if beta[j] != 0 else 1.0) * SHOCK_SD

    Xn0, Xx0, npair = snapshots(panel, As, cols)
    Xn1, Xx1, _ = snapshots(panel, Ash, cols)
    print(f"  one-month snapshot pairs: {npair:,}")
    if npair < MIN_PAIRS:
        raise SystemExit("too few consecutive month pairs to fit a Koopman operator")

    # each model contributes its own PD as an extra observable
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(A)
    clf = cl.classifiers()
    pdv = {}
    for m in MODELS:
        if m not in clf:
            continue
        est = clf[m]()
        est.fit(sc.transform(A), y.to_numpy(int))
        pdv[m] = {"base": est.predict_proba(sc.transform(A))[:, 1],
                  "shock": est.predict_proba(
                      sc.transform(Ash * sd + mu))[:, 1]}

    rows, spec = [], {}
    for m in MODELS:
        if m not in piv.columns or m not in pdv:
            print(f"  {m}: no importance record or model, skipped")
            continue
        w = piv[m].reindex(cols).fillna(0.0).to_numpy(float)
        sub = np.argsort(-w)[:SUBSPACE]                 # this model's own subspace
        G0 = observables(As, sub, pdv[m]["base"])
        G1 = observables(Ash, sub, pdv[m]["shock"])
        Gn0, Gx0, _ = snapshots(panel, G0, cols)
        Gn1, Gx1, _ = snapshots(panel, G1, cols)
        e0 = spectrum(Gn0, Gx0)
        e1 = spectrum(Gn1, Gx1)
        spec[m] = (e0, e1)
        print(f"      {m:15s} subspace = "
              f"{', '.join(np.array(cols)[sub][:4])} ... + logit PD")
        r0, r1 = np.abs(e0).max(), np.abs(e1).max()
        rows.append(dict(model=m, radius_base=r0, radius_shock=r1,
                         d_radius=r1 - r0,
                         n_outside_base=int((np.abs(e0) > 1).sum()),
                         n_outside_shock=int((np.abs(e1) > 1).sum()),
                         n_complex_base=int((np.abs(e0.imag) > 1e-9).sum()),
                         half_life_base=half_life(e0[0]),
                         half_life_shock=half_life(e1[0]),
                         mean_mod_base=float(np.abs(e0).mean()),
                         mean_mod_shock=float(np.abs(e1).mean())))
        print(f"    {m:15s} spectral radius {r0:.4f} -> {r1:.4f} "
              f"({r1-r0:+.4f})   modes outside unit circle {int((np.abs(e0)>1).sum())}"
              f" -> {int((np.abs(e1)>1).sum())}")
    res = pd.DataFrame(rows)

    print("\n=== slowest mode, in months to halve ===")
    for _, r in res.iterrows():
        h0 = r.half_life_base; h1 = r.half_life_shock
        f0 = "never" if not np.isfinite(h0) else f"{h0:.1f}"
        f1 = "never" if not np.isfinite(h1) else f"{h1:.1f}"
        print(f"  {r.model:15s} baseline {f0:>7s} months   shocked {f1:>7s} months")

    # ------------------------------------------------------------- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 12.0))
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, m in zip(axes.ravel(), spec):
        e0, e1 = spec[m]
        ax.plot(np.cos(th), np.sin(th), color="#334155", lw=1.6, zorder=2)
        ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color="#cbd5e1", lw=0.9,
                ls="--", zorder=1)
        ax.scatter(e0.real, e0.imag, s=52, color=MC[m], alpha=0.85,
                   edgecolors="white", linewidth=0.7, label="baseline", zorder=4)
        ax.scatter(e1.real, e1.imag, s=52, marker="x", color="#b91c1c",
                   alpha=0.9, linewidth=1.6, label="after shock", zorder=5)
        for a, b in zip(e0[:6], e1[:6]):
            ax.annotate("", xy=(b.real, b.imag), xytext=(a.real, a.imag),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="#94a3b8"),
                        zorder=3)
        ax.axhline(0, color="#94a3b8", lw=0.6)
        ax.axvline(0, color="#94a3b8", lw=0.6)
        ax.set_aspect("equal")
        lim = max(1.15, np.abs(np.r_[e0, e1]).max() * 1.1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        # every eigenvalue sits in a tight cluster just inside or on the circle at
        # Re(lambda) close to 1, so the full view alone shows nothing; the inset
        # carries the detail that actually distinguishes baseline from shock
        axi = ax.inset_axes([0.06, 0.60, 0.38, 0.36])
        axi.plot(np.cos(th), np.sin(th), color="#334155", lw=1.2)
        axi.scatter(e0.real, e0.imag, s=30, color=MC[m], alpha=0.9,
                    edgecolors="white", linewidth=0.5, zorder=4)
        axi.scatter(e1.real, e1.imag, s=34, marker="x", color="#b91c1c",
                    linewidth=1.4, zorder=5)
        axi.axhline(0, color="#94a3b8", lw=0.5)
        axi.set_xlim(0.86, 1.05); axi.set_ylim(-0.06, 0.06)
        axi.set_title("zoom near unity", fontsize=7.5)
        axi.tick_params(labelsize=6)
        axi.grid(alpha=0.25)
        r0, r1 = np.abs(e0).max(), np.abs(e1).max()
        ax.set_title(f"{m}\nspectral radius {r0:.4f} $\\rightarrow$ {r1:.4f}",
                     fontsize=11, fontweight="bold", color=MC[m])
        ax.set_xlabel(r"Re $\lambda$"); ax.set_ylabel(r"Im $\lambda$")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.2)
    fig.suptitle("Koopman eigenvalues of the monthly determinant dynamics\n"
                 f"observables = each model's top {SUBSPACE} determinants plus its own "
                 f"logit PD;  shock = {SHOCK_SD:.0f} SD adverse move on "
                 f"{', '.join(shocked_feats[:3])}",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    p = os.path.join(OUTDIR, "fig_koopman_shock_spectrum.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ev = []
    for m, (e0, e1) in spec.items():
        for k, (a, b) in enumerate(zip(e0, e1), 1):
            ev.append(dict(model=m, mode=k,
                           re_base=a.real, im_base=a.imag, mod_base=abs(a),
                           re_shock=b.real, im_shock=b.imag, mod_shock=abs(b)))
    pd.DataFrame(ev).to_csv(out("koopman_eigenvalues.csv"), index=False)
    res.to_csv(out("koopman_spectrum_summary.csv"), index=False)
    con = sqlite3.connect(DB)
    res.to_sql("cmdf_koopman_spectrum", con, if_exists="replace", index=False)
    con.commit(); con.close()
    print(f"\n  wrote {p}")
    print("  wrote tex_out/koopman_eigenvalues.csv, koopman_spectrum_summary.csv")


if __name__ == "__main__":
    main()
