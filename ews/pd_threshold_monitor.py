# -*- coding: utf-8 -*-
"""
pd_threshold_monitor.py -- derive monitoring thresholds on determinant pairs from the
shock analysis, so an issuer can be checked against a stated PD ceiling.

THE QUESTION
    "Keep PD below c" is a statement about the outcome, but an analyst can only observe
    determinants. The task is therefore to translate a ceiling on PD into a boundary in
    determinant space, and then to measure how far each issuer sits from it.

HOW THE THRESHOLD IS BUILT
    For a determinant pair the fitted model defines a partial-dependence surface
    PD(x_a, x_b). The set {(x_a, x_b) : PD = c} is the iso-PD contour, and it is the
    boundary being sought. Three products come out of it:

      1  MARGINAL TRIPWIRE   the value of one determinant at which PD reaches c while
                             the other sits at its median. One number per determinant,
                             usable in a spreadsheet.
      2  PAIR BOUNDARY       a straight line w_a x_a + w_b x_b = k fitted to the
                             contour, with the R^2 of that fit reported so a curved
                             contour is not silently reported as a line.
      3  MARGIN              for each issuer, the signed distance to the boundary in
                             standard deviations. Positive means inside the safe
                             region. This is the quantity to monitor, because a breach
                             flag alone gives no warning that one is approaching.

TWO WAYS TO SET c, AND WHY THE SECOND IS PREFERRED
    ABSOLUTE   c is a probability, for example 0.01. This is the natural reading of
               "PD must not exceed one per cent", but it relies on the probabilities
               being calibrated. On this panel they are not: the Brier Skill Score is
               negative for every model, so an absolute ceiling is not trustworthy.
    WORKLOAD   c is the PD at a chosen review capacity, for example the 98th percentile
               of the out-of-fold PD distribution when the team can review 2% of
               issuer-months. This is invariant to miscalibration, because it depends
               only on the ordering, and it is the form we recommend.

    Both are computed. The workload-anchored thresholds are the ones to deploy.

VALIDATION DISCIPLINE
    The PD distribution that anchors the workload thresholds comes from grouped
    out-of-fold predictions, so the level is not set by rows the model has memorised.
    The surface itself is fitted on the full panel, which is appropriate because it is
    being used to describe a response, not to forecast.

RUN
    python pd_threshold_monitor.py
    python pd_threshold_monitor.py --workloads 0.01,0.02,0.05
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
from reanalysis_oof import out_of_fold

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out

TOP = 8
GRID = 60
BACKGROUND = 150
WORKLOADS = (0.01, 0.02, 0.05)
ABS_TARGETS = (0.005, 0.010, 0.020)
SCORER = "CatBoost"
SEED = 42


def contour_points(G1, G2, P, c):
    """Points on the iso-PD contour, taken from matplotlib's contour extractor."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(G1, G2, P, levels=[c])
    pts = []
    for path in cs.get_paths():
        v = path.vertices
        if len(v):
            pts.append(v)
    plt.close(fig)
    return np.vstack(pts) if pts else np.empty((0, 2))


def fit_line(pts):
    """Least-squares line w1*x + w2*y = k with unit normal, plus the R^2 of the fit.

    Total least squares via the smaller principal direction, because neither
    determinant is the dependent variable here."""
    if len(pts) < 4:
        return None
    mu = pts.mean(0)
    Z = pts - mu
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    normal = Vt[1]                       # direction of least spread
    k = normal @ mu
    resid = Z @ normal
    tot = np.sum(Z ** 2)
    r2 = 1.0 - (np.sum(resid ** 2) / tot) if tot > 0 else np.nan
    return dict(w1=float(normal[0]), w2=float(normal[1]), k=float(k),
                r2_linear=float(r2), n_points=len(pts))


def main():
    workloads = list(WORKLOADS)
    if "--workloads" in sys.argv:
        workloads = [float(x) for x in
                     sys.argv[sys.argv.index("--workloads") + 1].split(",")]

    print("=" * 100)
    print("Monitoring thresholds on determinant pairs, derived from the PD surface")
    print("=" * 100)
    panel, X, y, cols = cl.load_panel(verbose=True)
    panel = panel.reset_index(drop=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    sd = A.std(0, ddof=1)
    med = np.median(A, axis=0)
    groups = panel["issuer_code"].to_numpy()

    imp = pd.read_csv(out("importance_default_event.csv"))
    gains = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
    idx = {c: i for i, c in enumerate(cols)}
    feats = [f for f in gains.index if f in idx][:TOP]
    print(f"\n  determinants monitored: {', '.join(feats)}")

    print(f"\n  out-of-fold PD for the threshold levels ({SCORER}) ...")
    oof = out_of_fold(SCORER, X, y, groups)
    ok = np.isfinite(oof)
    levels = {}
    for w in workloads:
        levels[f"workload {w:.0%}"] = float(np.quantile(oof[ok], 1 - w))
    for a in ABS_TARGETS:
        levels[f"absolute {a:.3f}"] = a
    print("  threshold levels")
    for k, v in levels.items():
        share = float((oof[ok] >= v).mean())
        print(f"    {k:16s} PD = {v:.6f}   flags {100*share:5.2f}% of issuer-months")

    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier
    sc = StandardScaler().fit(A)
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0,
                            allow_writing_files=False).fit(sc.transform(A), yv)

    def pdf(B):
        return cb.predict_proba(sc.transform(B))[:, 1]

    rng = np.random.default_rng(SEED)
    BG = A[rng.choice(len(A), size=BACKGROUND, replace=False)]

    print(f"\n  building surfaces and extracting contours ...")
    pair_rows, marg_rows, surfaces = [], [], {}
    for f1, f2 in itertools.combinations(feats, 2):
        j1, j2 = idx[f1], idx[f2]
        lo1, hi1 = np.percentile(A[:, j1], [1, 99])
        lo2, hi2 = np.percentile(A[:, j2], [1, 99])
        G1, G2 = np.meshgrid(np.linspace(lo1, hi1, GRID),
                             np.linspace(lo2, hi2, GRID))
        big = np.tile(BG, (G1.size, 1))
        big[:, j1] = np.repeat(G1.ravel(), len(BG))
        big[:, j2] = np.repeat(G2.ravel(), len(BG))
        P = pdf(big).reshape(G1.size, len(BG)).mean(1).reshape(G1.shape)
        surfaces[(f1, f2)] = (G1, G2, P)

        for lname, c in levels.items():
            if not (P.min() < c < P.max()):
                pair_rows.append(dict(f1=f1, f2=f2, level=lname, target_pd=c,
                                      reachable=False))
                continue
            pts = contour_points(G1, G2, P, c)
            fit = fit_line(pts)
            row = dict(f1=f1, f2=f2, level=lname, target_pd=c, reachable=True,
                       pd_min=float(P.min()), pd_max=float(P.max()))
            if fit:
                row.update(fit)
                # express the rule in standard deviations so the two determinants are
                # on a common footing
                row["w1_sd"] = fit["w1"] * sd[j1]
                row["w2_sd"] = fit["w2"] * sd[j2]
            pair_rows.append(row)

    # marginal tripwires: one determinant moves, the rest stay at their real values
    print("  marginal tripwires ...")
    for f in feats:
        j = idx[f]
        lo, hi = np.percentile(A[:, j], [1, 99])
        gvals = np.linspace(lo, hi, 400)
        big = np.tile(BG, (len(gvals), 1))
        big[:, j] = np.repeat(gvals, len(BG))
        curve = pdf(big).reshape(len(gvals), len(BG)).mean(1)
        rising = curve[-1] >= curve[0]
        for lname, c in levels.items():
            hit = np.where(curve >= c)[0] if rising else np.where(curve <= c)[0]
            val = float(gvals[hit[0]]) if len(hit) else np.nan
            marg_rows.append(dict(feature=f, level=lname, target_pd=c,
                                  direction="rising" if rising else "falling",
                                  threshold=val,
                                  threshold_sd=((val - med[j]) / sd[j])
                                  if np.isfinite(val) else np.nan,
                                  pd_at_min=float(curve[0]),
                                  pd_at_max=float(curve[-1]),
                                  reachable=bool(len(hit))))
    pairs_df = pd.DataFrame(pair_rows)
    marg_df = pd.DataFrame(marg_rows)

    key = f"workload {workloads[1]:.0%}" if len(workloads) > 1 else list(levels)[0]
    print(f"\n=== marginal tripwires at {key} (PD = {levels[key]:.6f}) ===")
    m = marg_df[marg_df.level == key]
    print(f"  {'determinant':22s} {'dir':>8} {'threshold':>12} {'in SD':>8} reachable")
    for _, r in m.iterrows():
        t = "--" if not np.isfinite(r.threshold) else f"{r.threshold:.4f}"
        s = "--" if not np.isfinite(r.threshold_sd) else f"{r.threshold_sd:+.2f}"
        print(f"  {r.feature:22s} {r.direction:>8} {t:>12} {s:>8} "
              f"{'yes' if r.reachable else 'no'}")

    print(f"\n=== pair boundaries at {key}, ranked by how linear the contour is ===")
    pp = pairs_df[(pairs_df.level == key) & (pairs_df.reachable == True)]
    if "r2_linear" in pp.columns:
        pp = pp.dropna(subset=["r2_linear"]).sort_values("r2_linear", ascending=False)
        print(f"  {'pair':40s} {'R2 linear':>10} {'rule in SD units':>34}")
        for _, r in pp.head(10).iterrows():
            rule = f"{r.w1_sd:+.3f}*{r.f1[:12]} {r.w2_sd:+.3f}*{r.f2[:12]} = {r.k:.3f}"
            print(f"  {r.f1[:18]+' + '+r.f2[:18]:40s} {r.r2_linear:>10.4f} "
                  f"{rule:>34}")

    # ------------------------------------------------- current issuer status --
    print("\n  scoring the latest month of every issuer ...")
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    last = panel.sort_values("month_dt").groupby("issuer_code").tail(1).index.to_numpy()
    status = pd.DataFrame(dict(issuer_code=panel.loc[last, "issuer_code"].values,
                               month=panel.loc[last, "month"].values,
                               pd_oof=oof[last]))
    for lname, c in levels.items():
        status[f"breach {lname}"] = status.pd_oof >= c
    status["margin_pd"] = levels[key] - status.pd_oof
    status = status.sort_values("pd_oof", ascending=False)
    nb = int(status[f"breach {key}"].sum())
    print(f"    {nb} of {len(status)} issuers breach the {key} threshold")
    print(status.head(8).to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # ------------------------------------------------------------- figure ----
    show = list(itertools.combinations(feats, 2))[:9]
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 13.0))
    for ax, (f1, f2) in zip(axes.ravel(), show):
        G1, G2, P = surfaces[(f1, f2)]
        im = ax.contourf(G1, G2, P, levels=24, cmap="viridis")
        for lname, c, col, ls in ((f"workload {workloads[1]:.0%}",
                                   levels[key], "#f8fafc", "-"),
                                  ("absolute 0.010", levels.get("absolute 0.010"),
                                   "#fb7185", "--")):
            if c is not None and P.min() < c < P.max():
                cs = ax.contour(G1, G2, P, levels=[c], colors=col,
                                linewidths=2.0, linestyles=ls)
                ax.clabel(cs, fmt={c: lname}, fontsize=7)
        j1, j2 = idx[f1], idx[f2]
        ax.scatter(A[last, j1], A[last, j2], s=12, color="#e2e8f0",
                   edgecolors="#0f172a", linewidth=0.4, alpha=0.85, zorder=5,
                   label="issuers, latest month")
        ax.set_xlabel(f1, fontsize=8.5); ax.set_ylabel(f2, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{f1} x {f2}", fontsize=10, fontweight="bold")
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=7.5, loc="upper right")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Iso-PD contours as monitoring boundaries\n"
                 f"filled surface = {SCORER} PD averaged over {BACKGROUND} real "
                 f"issuer-months;  white line = the {key} threshold "
                 f"(PD {levels[key]:.5f});  dashed = absolute PD 0.010",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUTDIR, "fig_pd_threshold_monitor.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    xl = out("pd_threshold_monitor.xlsx")
    guide = pd.DataFrame({
        "topic": ["recommended threshold type", "why not absolute PD",
                  "what to monitor", "when to re-derive", "known weakness"],
        "guidance": [
            "Workload-anchored. Set the ceiling at the PD reached by the share of "
            "issuer-months the review team can actually examine.",
            "The Brier Skill Score is negative for every model on this panel, so the "
            "probabilities are not calibrated and an absolute ceiling such as 1% does "
            "not mean what it says.",
            "The signed margin to the boundary, not the breach flag. A breach flag "
            "gives no warning that an issuer is approaching the line.",
            "Whenever the panel is extended, and at minimum annually. The contour is "
            "a property of the fitted surface, which moves as the data move.",
            "Policyrate enters most boundaries, and on this panel all 32 event months "
            "fall in a single Policyrate regime, so thresholds involving it may be "
            "tracking calendar time rather than credit risk."]})
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        marg_df.to_excel(w, sheet_name="marginal tripwires", index=False)
        pairs_df.to_excel(w, sheet_name="pair boundaries", index=False)
        status.to_excel(w, sheet_name="issuer status", index=False)
        pd.DataFrame([dict(level=k, target_pd=v,
                           flagged_pct=100 * float((oof[ok] >= v).mean()))
                      for k, v in levels.items()]).to_excel(
            w, sheet_name="levels", index=False)
        guide.to_excel(w, sheet_name="how to use", index=False)

    con = sqlite3.connect(DB)
    marg_df.to_sql("cmdf_pd_tripwires", con, if_exists="replace", index=False)
    pairs_df.to_sql("cmdf_pd_boundaries", con, if_exists="replace", index=False)
    status.to_sql("cmdf_pd_issuer_status", con, if_exists="replace", index=False)
    con.commit(); con.close()
    print(f"\n  wrote {p}")
    print(f"  wrote {xl}")


if __name__ == "__main__":
    main()
