# -*- coding: utf-8 -*-
"""
shock_direction.py -- which way is "adverse", decided once and by the model whose
surface the shock is measured on.

THE DEFECT THIS FIXES
    Six modules each re-fitted a logistic regression and took the adverse direction
    from the sign of its coefficient::

        beta = LogisticRegression(...).fit(As, y).coef_[0]
        direction = 1.0 if beta[j] >= 0 else -1.0

    They then measured the effect of that move on a CatBoost surface. Where the two
    models disagree about which way a determinant pushes risk -- and on these panels
    they disagree often -- moving in the logistic's adverse direction makes the
    CatBoost probability FALL. On the 45 triples compared across the two panels the
    reported change came out negative 40 and 45 times respectively, so the sign of
    every published shock effect meant nothing.

    The direction has to come from the same surface the effect is read off. That is
    all this module does.

HOW THE DIRECTION IS MEASURED
    Empirically, with the same perturbation the analysis actually applies: move the
    determinant +1 SD across the whole panel, and look at which way the model's mean
    predicted probability moves.

        dir(j) = sign( mean[ f(A with x_j + SD_j) ] - mean[ f(A) ] )

    A finite difference rather than a coefficient, because a tree ensemble has no
    coefficient, and because the finite difference is the quantity the shock
    analysis is about to compute anyway. For the logistic the finite-difference sign
    and the coefficient sign always agree -- the link is monotone -- so nothing is
    lost by measuring both the same way.

    Single-determinant effects are then non-negative by construction. That is the
    point: "adverse" now means adverse, and what stays informative is the SIZE of
    each effect, the interactions between determinants, and the cases where the two
    models disagree, which this module reports rather than hides.

ONE TABLE, EVERY FIGURE
    The result is cached per dataset at tex_out/ds<N>/shock_direction.csv so all six
    modules displace the cloud the same way. Previously each re-fitted its own
    logistic and small differences in fitting could point two figures in opposite
    directions with nothing on the page to say so.

RUN
    python shock_direction.py                # dataset 1
    python shock_direction.py --dataset 2
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cmdf_tree_classify as cl
import ibond_dataset as ds

CSV_NAME = "shock_direction.csv"
SHOCK_SD = 1.0
SEED = 42

SURFACE = "surface"        # the fix: direction from the model that measures PD
LOGISTIC = "logistic"      # the old behaviour, kept so the earlier figures redraw


def source_from_argv(default=SURFACE):
    """Read --direction surface|logistic, and remove it from sys.argv."""
    if "--direction" in sys.argv:
        i = sys.argv.index("--direction")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--direction expects 'surface' or 'logistic'")
        v = sys.argv[i + 1].strip().lower()
        del sys.argv[i:i + 2]
        if v not in (SURFACE, LOGISTIC):
            raise SystemExit(f"--direction expects 'surface' or 'logistic', got {v!r}")
        return v
    return default


def _fit(A, yv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier
    sc = StandardScaler().fit(A)
    As = sc.transform(A)
    lg = LogisticRegression(max_iter=5000, C=0.1,
                            class_weight="balanced").fit(As, yv)
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0,
                            allow_writing_files=False).fit(As, yv)
    return sc, lg, cb


def build(panel=None, X=None, y=None, cols=None, verbose=True):
    """Measure the adverse direction of every determinant under both models."""
    if panel is None:
        panel, X, y, cols = cl.load_panel(verbose=verbose)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    sd = A.std(0, ddof=1)
    sc, lg, cb = _fit(A, yv)

    def mean_pd(model, B):
        return float(model.predict_proba(sc.transform(B))[:, 1].mean())

    base_lg, base_cb = mean_pd(lg, A), mean_pd(cb, A)
    beta = lg.coef_[0]

    rows = []
    for j, c in enumerate(cols):
        B = A.copy()
        B[:, j] = B[:, j] + SHOCK_SD * sd[j]
        d_lg = mean_pd(lg, B) - base_lg
        d_cb = mean_pd(cb, B) - base_cb
        rows.append(dict(
            feature=c,
            beta_logistic=float(beta[j]),
            dpd_logistic_up=d_lg,
            dpd_surface_up=d_cb,
            dir_logistic=1 if beta[j] >= 0 else -1,
            dir_surface=1 if d_cb >= 0 else -1,
        ))
    d = pd.DataFrame(rows)
    d["agree"] = d["dir_logistic"] == d["dir_surface"]
    d["flat_surface"] = d["dpd_surface_up"].abs() < 1e-9

    if verbose:
        n, k = len(d), int((~d["agree"]).sum())
        print(f"  adverse direction measured on {n} determinants")
        print(f"  logistic and CatBoost disagree on {k} of {n} "
              f"({100*k/n:.0f}%)")
        flat = int(d["flat_surface"].sum())
        if flat:
            print(f"  {flat} determinant(s) move the CatBoost surface not at all; "
                  f"their direction is arbitrary and their effect is zero either way")
        if k:
            print(f"  {'determinant':28s} {'beta':>9} {'dPD logit':>11} "
                  f"{'dPD surface':>12}")
            for r in d.loc[~d["agree"]].itertuples():
                print(f"  {r.feature:28s} {r.beta_logistic:>+9.4f} "
                      f"{r.dpd_logistic_up:>+11.5f} {r.dpd_surface_up:>+12.5f}")
    return d


def table(panel=None, X=None, y=None, cols=None, verbose=True, rebuild=False):
    """The direction table for the active dataset, computed once and cached."""
    p = ds.out(CSV_NAME)
    if os.path.exists(p) and not rebuild:
        return pd.read_csv(p)
    d = build(panel, X, y, cols, verbose=verbose)
    d.to_csv(p, index=False)
    if verbose:
        print(f"  wrote {p}")
    return d


def directions(cols, source=SURFACE, panel=None, X=None, y=None, verbose=False):
    """{determinant: +1 or -1} under the requested source.

    ``surface`` is the corrected behaviour and the default; ``logistic`` reproduces
    the figures produced before this module existed.
    """
    d = table(panel, X, y, cols, verbose=verbose)
    col = "dir_surface" if source == SURFACE else "dir_logistic"
    m = dict(zip(d["feature"], d[col]))
    return {c: float(m.get(c, 1)) for c in cols}


def describe(source):
    return ("adverse direction from the CatBoost surface being measured "
            "(finite difference at +1 SD)" if source == SURFACE else
            "adverse direction from the fitted logistic coefficient "
            "(pre-fix behaviour, kept for comparison)")


def main():
    source = source_from_argv()
    print("=" * 92)
    print("Adverse direction per determinant, measured on both models")
    print("=" * 92)
    panel, X, y, cols = cl.load_panel(verbose=True)
    d = table(panel, X, y, cols, verbose=True, rebuild=True)
    print(f"\n  active source: {describe(source)}")
    print(f"\n  {'determinant':28s} {'logistic':>10} {'surface':>10} {'agree':>7}")
    for r in d.itertuples():
        print(f"  {r.feature:28s} {'up' if r.dir_logistic > 0 else 'down':>10} "
              f"{'up' if r.dir_surface > 0 else 'down':>10} "
              f"{'yes' if r.agree else 'NO':>7}")


if __name__ == "__main__":
    main()
