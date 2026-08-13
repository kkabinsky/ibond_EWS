# -*- coding: utf-8 -*-
"""
threshold_design_figure.py -- a worked example of setting a monitoring threshold the
way pd_threshold_monitor.py recommends: anchored on review capacity, not on an
absolute probability.

THE FOUR STEPS THE FIGURE WALKS THROUGH
    A  Start from the out-of-fold PD distribution. A review capacity, expressed as a
       share of issuer-months the team can examine, maps to a percentile, and the
       percentile maps to a PD cut-off. Nothing here depends on the probabilities being
       correctly scaled, only on their ordering, which is why this survives the
       negative Brier Skill Score.

    B  Choose the capacity by looking at what it buys. Detection rises with workload
       and so does the false-alarm load; the operating point is a decision about which
       trade the institution is willing to make, and it should be made on this curve
       rather than by picking a round number.

    C  Carry the chosen cut-off back into determinant space. The same threshold becomes
       an iso-PD contour on a determinant pair, which is the object an analyst can
       actually check an issuer against.

    D  Report the margin rather than the breach. The distribution of distance-to-
       threshold shows how many issuers sit just inside the line, which a breach count
       cannot.

RUN
    python threshold_design_figure.py
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cmdf_tree_classify as cl
import cmdf_tree_models as tm
from reanalysis_oof import lead_and_burden, out_of_fold

OUTDIR = tm.OUTDIR
out = tm.out

SCORER = "CatBoost"
SEED = 42
PAIR = ("TDTA", "Policyrate")     # the pair with the strongest interaction
MARKS = (0.02, 0.05, 0.10)        # capacities highlighted in the figure
CHOSEN = 0.10                     # a team of 3 reviewing 30 of 289 names a month
GRID = 70
BACKGROUND = 150
TEAM_SIZE = 3
TEAM_CAPACITY = 30                # names the team can review per month


def main():
    print("=" * 92)
    print("Worked example: building a monitoring threshold from review capacity")
    print("=" * 92)
    panel, X, y, cols = cl.load_panel(verbose=True)
    panel = panel.reset_index(drop=True)
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    panel["y"] = y.to_numpy(int)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    idx = {c: i for i, c in enumerate(cols)}
    groups = panel["issuer_code"].to_numpy()
    n_iss = panel["issuer_code"].nunique()

    print(f"\n  out-of-fold PD from {SCORER} ...")
    oof = out_of_fold(SCORER, X, y, groups)
    ok = np.isfinite(oof)
    p = oof[ok]

    # ---- step B: what each capacity buys -----------------------------------
    ws = np.unique(np.r_[np.linspace(0.002, 0.10, 60), MARKS])
    rows = []
    for w in ws:
        thr = float(np.quantile(p, 1 - w))
        alarm = (np.nan_to_num(oof, nan=-1) >= thr).astype(int)
        lt, fa, fa_yr = lead_and_burden(panel, alarm, len(panel))
        n_ev = int(len(lt)); nc = int(lt["caught"].sum()) if n_ev else 0
        rows.append(dict(workload=w, threshold=thr, flagged=int(alarm.sum()),
                         share=float(alarm.mean()),
                         detect=nc / n_ev if n_ev else np.nan, n_caught=nc,
                         n_events=n_ev, fa_per_issuer_year=fa_yr,
                         names_per_month=alarm.sum() / panel["month"].nunique()))
    tr = pd.DataFrame(rows)

    print(f"\n  {'capacity':>9} {'PD cut-off':>12} {'flagged':>9} {'detection':>11} "
          f"{'FA/issuer-yr':>13} {'names/month':>12}")
    for w in MARKS:
        r = tr.iloc[(tr.workload - w).abs().argmin()]
        print(f"  {r.workload:>8.1%} {r.threshold:>12.6f} {int(r.flagged):>9,} "
              f"{int(r.n_caught)}/{int(r.n_events)}{'':>6} "
              f"{r.fa_per_issuer_year:>13.3f} {r.names_per_month:>12.1f}")

    chosen = tr.iloc[(tr.workload - CHOSEN).abs().argmin()]
    thr_c = float(chosen.threshold)
    print(f"\n  chosen operating point: {chosen.workload:.1%} of issuer-months, "
          f"PD cut-off {thr_c:.6f}")

    # ---- step C: the same cut-off as a contour -----------------------------
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier
    sc = StandardScaler().fit(A)
    cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                            l2_leaf_reg=3.0, auto_class_weights="Balanced",
                            random_seed=SEED, verbose=0,
                            allow_writing_files=False).fit(sc.transform(A), yv)
    rng = np.random.default_rng(SEED)
    BG = A[rng.choice(len(A), size=BACKGROUND, replace=False)]
    f1, f2 = PAIR
    j1, j2 = idx[f1], idx[f2]
    lo1, hi1 = np.percentile(A[:, j1], [1, 99])
    lo2, hi2 = np.percentile(A[:, j2], [1, 99])
    G1, G2 = np.meshgrid(np.linspace(lo1, hi1, GRID), np.linspace(lo2, hi2, GRID))
    big = np.tile(BG, (G1.size, 1))
    big[:, j1] = np.repeat(G1.ravel(), len(BG))
    big[:, j2] = np.repeat(G2.ravel(), len(BG))
    P = cb.predict_proba(sc.transform(big))[:, 1] \
        .reshape(G1.size, len(BG)).mean(1).reshape(G1.shape)
    print(f"  surface over ({f1}, {f2}): PD from {P.min():.6f} to {P.max():.6f}")

    # ---- step D: margins on the latest month -------------------------------
    last = panel.sort_values("month_dt").groupby("issuer_code").tail(1).index.to_numpy()
    pd_last = oof[last]
    margin_pct = np.array([100 * (1 - (p < v).mean()) for v in pd_last])
    margin_w = chosen.workload * 100 - margin_pct        # positive = safe side

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 11.2))

    # A ---------------------------------------------------------------------
    ax = axes[0, 0]
    srt = np.sort(p)[::-1]
    share = np.arange(1, len(srt) + 1) / len(srt) * 100
    ax.plot(share, srt, color="#0f172a", lw=1.8)
    ax.set_yscale("log")
    for w, col in zip(MARKS, ["#0891b2", "#b91c1c", "#f59e0b"]):
        t = float(np.quantile(p, 1 - w))
        ax.axvline(w * 100, color=col, ls="--", lw=1.4)
        ax.axhline(t, color=col, ls=":", lw=1.2)
        ax.plot([w * 100], [t], marker="o", ms=8, color=col)
        ax.annotate(f" {w:.0%} capacity\n PD = {t:.5f}", (w * 100, t),
                    fontsize=8, color=col, fontweight="bold",
                    xytext=(9, 4), textcoords="offset points")
    ax.set_xlim(0, 12)
    ax.set_xlabel("share of issuer-months reviewed, highest PD first (%)")
    ax.set_ylabel("out-of-fold PD (log scale)")
    ax.set_title("A. capacity maps to a percentile, the percentile maps to a cut-off",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # B ---------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(tr.workload * 100, tr.detect * 100, color="#16a34a", lw=2.2,
            marker="o", ms=3, label="detection of the 8 event issuers")
    ax.set_xlabel("review capacity, share of issuer-months (%)")
    ax.set_ylabel("actionable detection (%)", color="#16a34a")
    ax.tick_params(axis="y", labelcolor="#16a34a")
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.plot(tr.workload * 100, tr.fa_per_issuer_year, color="#b91c1c", lw=2.2,
             ls="--", marker="s", ms=3, label="false alarms per issuer-year")
    ax2.set_ylabel("false alarms per issuer-year", color="#b91c1c")
    ax2.tick_params(axis="y", labelcolor="#b91c1c")
    ax.axvline(CHOSEN * 100, color="#0f172a", lw=1.6, ls="-.")
    ax.annotate(f"chosen: {CHOSEN:.0%}\n"
                f"{int(chosen.n_caught)}/{int(chosen.n_events)} detected\n"
                f"{chosen.fa_per_issuer_year:.2f} FA/issuer-yr\n"
                f"about {chosen.names_per_month:.0f} names a month",
                (CHOSEN * 100, 40), fontsize=8.5, fontweight="bold",
                xytext=(14, 0), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#0f172a"))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.set_title("B. choose the capacity from what it buys, not from a round number",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3)

    # C ---------------------------------------------------------------------
    ax = axes[1, 0]
    im = ax.contourf(G1, G2, P, levels=26, cmap="viridis")
    for w, col in zip(MARKS, ["#67e8f9", "#fecaca", "#fde68a"]):
        t = float(np.quantile(p, 1 - w))
        if P.min() < t < P.max():
            cs = ax.contour(G1, G2, P, levels=[t], colors=col, linewidths=2.4)
            ax.clabel(cs, fmt={t: f"{w:.0%}"}, fontsize=8)
    ax.scatter(A[last, j1], A[last, j2], s=16, color="#e2e8f0",
               edgecolors="#0f172a", linewidth=0.5, alpha=0.9, zorder=5,
               label="issuers, latest month")
    ax.set_xlabel(f1); ax.set_ylabel(f2)
    ax.set_title(f"C. the cut-off becomes a boundary on ({f1}, {f2})",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="PD")

    # D ---------------------------------------------------------------------
    ax = axes[1, 1]
    near = margin_w[(margin_w > -6) & (margin_w < 20)]
    ax.hist(margin_w, bins=np.arange(-8, 40, 1.0), color="#94a3b8",
            edgecolor="white")
    ax.axvline(0, color="#b91c1c", lw=2.2)
    n_breach = int((margin_w <= 0).sum())
    n_close = int(((margin_w > 0) & (margin_w <= 2)).sum())
    ax.annotate(f"{n_breach} issuers past the line",
                (0, ax.get_ylim()[1] * 0.86), fontsize=9, color="#b91c1c",
                fontweight="bold", xytext=(-118, 0), textcoords="offset points")
    ax.axvspan(0, 2, color="#f59e0b", alpha=0.22)
    ax.annotate(f"{n_close} within 2 points\nof the line",
                (2, ax.get_ylim()[1] * 0.60), fontsize=8.5, color="#b45309",
                fontweight="bold", xytext=(10, 0), textcoords="offset points")
    ax.set_xlabel("margin to the threshold, in percentile points "
                  "(positive = safe side)")
    ax.set_ylabel("issuers")
    ax.set_title("D. monitor the margin; a breach count gives no early warning",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Setting a monitoring threshold from review capacity\n"
                 f"{TEAM_CAPACITY} of {n_iss} issuers is {TEAM_CAPACITY/n_iss:.1%}, "
                 f"so the cut-off sits at the {100-100*TEAM_CAPACITY/n_iss:.0f}th "
                 f"percentile of the out-of-fold PD;  {len(panel):,} issuer-months, "
                 f"{int(yv.sum())} event months",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fp = os.path.join(OUTDIR, "fig_threshold_design.png")
    fig.savefig(fp, dpi=135, bbox_inches="tight")
    plt.close(fig)

    tr.to_csv(out("threshold_tradeoff.csv"), index=False)
    print(f"\n  breaches at the chosen point: {n_breach} of {len(margin_w)} issuers")
    print(f"  within 2 percentile points of the line: {n_close}")
    print(f"\n  wrote {fp}")
    print("  wrote tex_out/threshold_tradeoff.csv")


if __name__ == "__main__":
    main()
