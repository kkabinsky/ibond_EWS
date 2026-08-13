# -*- coding: utf-8 -*-
"""
hyperbolic_boundary_panel.py -- momentum and the hyperbolic decision boundary on the
real iBond panel, as a command-line tool and as a panel inside app.py.

THE RULE
    The level rule asks only "is PD high now". The boundary rule asks a second question,
    "is PD RISING fast from where it was", and combines the two:

        Momentum(t) = PD(t) / PD(t-1)
        alarm       <=>  Momentum(t) * PD(t-1)^alpha  >=  K
                    <=>  log M(t) + alpha * log PD(t-1)  >=  log K

    In the (PD, M) plane the boundary M = K / PD^alpha is a hyperbola, which is where the
    name comes from. alpha sets how much a high level is allowed to substitute for fast
    growth. At alpha = 0 the rule ignores the level and looks only at momentum; as alpha
    grows the level dominates and the rule converges on the plain PD threshold.

    This is the same definition survival.py uses. What is different here is the source of
    PD: this module takes the OUT-OF-FOLD CatBoost probabilities from firm_shock_panel,
    so the boundary, the PD path chart and the shock ladder all sit on one scale.

WHY log K IS NOT A FREE PARAMETER
    Under workload matching K carries no information. Writing s = log M + alpha log P,
    the alarm set is {s >= log K}, and

        quantile_{1-w}(s - log K) = quantile_{1-w}(s) - log K

    so choosing log K to alarm on a fixed fraction w gives the same ranking whatever K
    is. Only alpha changes the ordering. The tool therefore scans alpha and fixes log K
    by workload, rather than treating both as free.

WHAT IS HONEST AND WHAT IS NOT
    The panel holds 32 event months from 8 issuers. Selecting alpha by how well the rule
    flags those 32 months is fitting on the outcome, in sample. Every selection number
    printed under "in-sample" carries that caveat, and the tool always prints the plain
    PD rule beside the boundary rule so the reader can see whether the extra parameter
    bought anything.

RUN
    python hyperbolic_boundary_panel.py
    python hyperbolic_boundary_panel.py --workload 0.10
    python hyperbolic_boundary_panel.py --issuer TPOLY
    python hyperbolic_boundary_panel.py --alpha 1.2 --no-figure
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
EPS = 1e-12
ALPHAS = np.round(np.linspace(0.0, 2.5, 26), 3)

_CACHE = {}


def _fig_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ data
def build_frame(workload=0.05):
    """One row per issuer-month with PD, previous PD, momentum and the event label."""
    key = ("frame", workload)
    if key in _CACHE:
        return _CACHE[key]

    import firm_shock_panel as fsp
    S = fsp.load_state(workload)

    p = S["panel"].copy()
    p["pd_now"] = S["oof"]
    p["event"] = S["y"]
    p = p.sort_values(["issuer_code", "month_dt"]).reset_index(drop=True)

    # previous month within the same issuer, and only when the months really are
    # consecutive -- a gap in the calendar would make the ratio meaningless
    g = p.groupby("issuer_code")
    p["pd_prev"] = g["pd_now"].shift(1)
    gap = g["month_dt"].diff().dt.days
    p.loc[(gap < 27) | (gap > 32), "pd_prev"] = np.nan

    p["momentum"] = p["pd_now"] / p["pd_prev"].replace(0.0, np.nan)
    ok = (p.pd_prev.notna() & (p.pd_prev > EPS) & (p.momentum > 0)
          & np.isfinite(p.pd_now))
    p["usable"] = ok
    p["log_M"] = np.where(ok, np.log(p.momentum.where(ok, 1.0)), np.nan)
    p["log_P"] = np.where(ok, np.log(p.pd_prev.where(ok, 1.0)), np.nan)

    out = dict(frame=p, thr=S["thr"], workload=workload,
               n_events=int(p.event.sum()),
               event_issuers=sorted(p.loc[p.event == 1, "issuer_code"].unique()))
    _CACHE[key] = out
    return out


def score(p, alpha):
    """s = log M + alpha log P. Higher means closer to alarming."""
    return p["log_M"] + alpha * p["log_P"]


def boundary_at(p, alpha, workload):
    """log K fixed by workload on the usable rows."""
    s = score(p, alpha)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan
    return float(np.quantile(s, 1 - workload))


def confusion(flag, y):
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    tn = int(((flag == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, mcc=mcc)


def scan_alpha(workload=0.05):
    """Boundary rule at every alpha, plus the plain PD rule, on the same rows."""
    D = build_frame(workload)
    p = D["frame"]
    u = p[p.usable].copy()
    y = u.event.to_numpy(int)

    # the level rule, restricted to the same rows so the comparison is like for like
    pd_cut = float(np.quantile(u.pd_now, 1 - workload))
    lvl = confusion((u.pd_now >= pd_cut).to_numpy(int), y)
    lvl.update(rule="PD level only", alpha=np.nan, logK=np.nan, cut=pd_cut)

    rows = [lvl]
    for a in ALPHAS:
        lk = boundary_at(u, a, workload)
        c = confusion((score(u, a) >= lk).to_numpy(int), y)
        c.update(rule="hyperbolic boundary", alpha=float(a), logK=lk,
                 K=float(np.exp(lk)), cut=np.nan)
        rows.append(c)
    d = pd.DataFrame(rows)
    return d, D


def pick_alpha(workload=0.05):
    d, D = scan_alpha(workload)
    b = d[d.rule == "hyperbolic boundary"]
    best = b.loc[b.mcc.idxmax()]
    lvl = d[d.rule == "PD level only"].iloc[0]
    return float(best.alpha), float(best.logK), best, lvl, d, D


# ------------------------------------------------------------------ report
def issuer_positions(alpha, logK, workload=0.05):
    """Where each issuer's latest usable month sits relative to the boundary."""
    D = build_frame(workload)
    p = D["frame"]
    u = p[p.usable].copy()
    last = u.sort_values("month_dt").groupby("issuer_code").tail(1)
    last = last.assign(s=score(last, alpha))
    last["margin"] = last.s - logK
    last["side"] = np.where(last.margin >= 0, "ALARM", "below")
    last["ever_event"] = last.issuer_code.isin(D["event_issuers"])
    cols = ["issuer_code", "month", "pd_prev", "pd_now", "momentum", "s",
            "margin", "side", "ever_event"]
    return last[cols].sort_values("margin", ascending=False).reset_index(drop=True)


def event_month_positions(alpha, logK, workload=0.05):
    """Every recorded event month, with its momentum and its side of the boundary."""
    D = build_frame(workload)
    p = D["frame"]
    e = p[(p.event == 1)].copy()
    e["s"] = score(e, alpha)
    e["margin"] = e.s - logK
    e["side"] = np.where(e.usable & (e.margin >= 0), "ALARM",
                         np.where(e.usable, "below", "no previous month"))
    return e[["issuer_code", "month", "pd_prev", "pd_now", "momentum", "s",
              "margin", "side"]].sort_values(["issuer_code", "month"])


# ------------------------------------------------------------------ figure
def figure_boundary(alpha, logK, workload=0.05, issuer=None):
    D = build_frame(workload)
    p = D["frame"]
    u = p[p.usable]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))

    # --- left: the (log PD, log M) plane with the boundary as a straight line ---
    ax = axes[0]
    ax.scatter(u.log_P, u.log_M, s=4, color="#cbd5e1", alpha=0.35,
               label=f"issuer-months, n={len(u):,}")
    ev = u[u.event == 1]
    ax.scatter(ev.log_P, ev.log_M, s=64, marker="X", color="#16a34a",
               edgecolors="white", linewidth=0.8, zorder=6,
               label=f"recorded event months, n={len(ev)}")
    xs = np.linspace(u.log_P.min(), u.log_P.max(), 100)
    ax.plot(xs, logK - alpha * xs, color="#b91c1c", lw=2.4,
            label=f"boundary  $\\log M + {alpha:.2f}\\log P = {logK:.3f}$")
    if issuer:
        gi = u[u.issuer_code == issuer]
        if len(gi):
            ax.plot(gi.log_P, gi.log_M, color="#f59e0b", lw=1.3, alpha=0.9, zorder=7)
            la = gi.sort_values("month_dt").iloc[-1]
            ax.scatter([la.log_P], [la.log_M], s=150, marker="*", color="#b91c1c",
                       edgecolors="white", linewidth=1.0, zorder=8,
                       label=f"{issuer}, latest")
    ax.set_xlabel("$\\log PD(t-1)$", fontsize=10)
    ax.set_ylabel("$\\log M(t) = \\log\\, PD(t)/PD(t-1)$", fontsize=10)
    ax.set_title("A. the boundary is a straight line in log coordinates",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.6, loc="upper right")
    ax.grid(alpha=0.3)

    # --- right: the same thing in PD and M, where it is a hyperbola ---
    ax = axes[1]
    ax.scatter(u.pd_prev, u.momentum, s=4, color="#cbd5e1", alpha=0.35)
    ax.scatter(ev.pd_prev, ev.momentum, s=64, marker="X", color="#16a34a",
               edgecolors="white", linewidth=0.8, zorder=6,
               label="recorded event months")
    lo = max(float(u.pd_prev.min()), 1e-9)
    pg = np.logspace(np.log10(lo), np.log10(float(u.pd_prev.max())), 200)
    ax.plot(pg, np.exp(logK) / pg ** alpha, color="#b91c1c", lw=2.4,
            label=f"$M = K/PD^{{{alpha:.2f}}}$,  $K={np.exp(logK):.4g}$")
    ax.axvline(D["thr"], color="#1d4ed8", ls="--", lw=1.6,
               label=f"PD level threshold {D['thr']:.5f}")
    if issuer:
        gi = u[u.issuer_code == issuer]
        if len(gi):
            ax.plot(gi.pd_prev, gi.momentum, color="#f59e0b", lw=1.3, zorder=7)
            la = gi.sort_values("month_dt").iloc[-1]
            ax.scatter([la.pd_prev], [la.momentum], s=150, marker="*",
                       color="#b91c1c", edgecolors="white", linewidth=1.0, zorder=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$PD(t-1)$", fontsize=10)
    ax.set_ylabel("momentum $M(t)$", fontsize=10)
    ax.set_title("B. the same rule in the original units is a hyperbola",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.6, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    ttl = (f"Hyperbolic decision boundary at {workload:.0%} review capacity"
           + (f" -- {issuer}" if issuer else ""))
    fig.suptitle(ttl, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _fig_b64(fig)


def figure_alpha_scan(workload=0.05):
    d, D = scan_alpha(workload)
    b = d[d.rule == "hyperbolic boundary"]
    lvl = d[d.rule == "PD level only"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax = axes[0]
    ax.plot(b.alpha, b.mcc, "o-", color="#1d4ed8", lw=2.0, ms=4)
    ax.axhline(lvl.mcc, color="#b91c1c", ls="--", lw=1.8,
               label=f"PD level only, MCC {lvl.mcc:.4f}")
    k = b.loc[b.mcc.idxmax()]
    ax.scatter([k.alpha], [k.mcc], s=130, marker="*", color="#f59e0b",
               edgecolors="#111827", zorder=6,
               label=f"best $\\alpha$ = {k.alpha:.2f}, MCC {k.mcc:.4f}")
    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_ylabel("MCC (in sample, 32 events)", fontsize=10)
    ax.set_title("choosing $\\alpha$ at fixed workload", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(b.alpha, b.recall, "o-", color="#16a34a", lw=1.9, ms=4,
            label="recall, boundary")
    ax.plot(b.alpha, b.precision, "s-", color="#7c3aed", lw=1.9, ms=4,
            label="precision, boundary")
    ax.axhline(lvl.recall, color="#16a34a", ls="--", lw=1.5,
               label=f"recall, PD level {lvl.recall:.3f}")
    ax.axhline(lvl.precision, color="#7c3aed", ls="--", lw=1.5,
               label=f"precision, PD level {lvl.precision:.3f}")
    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_title(f"both rules alarm on {workload:.0%} of rows by construction",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _fig_b64(fig)


def figure_all_labeled(alpha, logK, workload=0.05, label_top=45):
    """Every issuer's latest month in one plane, labelled, with the boundary."""
    D = build_frame(workload)
    pos = issuer_positions(alpha, logK, workload)
    pos = pos.assign(lP=np.log(pos.pd_prev), lM=np.log(pos.momentum))

    fig, ax = plt.subplots(figsize=(13.0, 8.4))
    ok = pos[~pos.ever_event]
    ev = pos[pos.ever_event]
    ax.scatter(ok.lP, ok.lM, s=34, color="#94a3b8", alpha=0.75,
               edgecolors="white", linewidth=0.4,
               label=f"issuers with no recorded event, n={len(ok)}")
    ax.scatter(ev.lP, ev.lM, s=170, marker="X", color="#16a34a",
               edgecolors="white", linewidth=1.0, zorder=6,
               label=f"issuers that recorded an event, n={len(ev)}")

    xs = np.linspace(pos.lP.min() - 0.4, pos.lP.max() + 0.4, 100)
    ax.plot(xs, logK - alpha * xs, color="#b91c1c", lw=2.6, zorder=5,
            label=f"boundary  $\\log M + {alpha:.2f}\\log P = {logK:.3f}$")
    ax.axvline(np.log(D["thr"]), color="#1d4ed8", ls="--", lw=1.8,
               label=f"PD level threshold {D['thr']:.5f}")
    ax.axhline(0.0, color="#64748b", ls=":", lw=1.4,
               label="$M = 1$, PD unchanged from last month")

    # Almost every point lands on log M = 0, so labels stack on one line. They are
    # staggered over several tiers with leader lines, and only the names worth
    # reading are drawn: every event issuer plus the riskiest by margin.
    show = pos[pos.ever_event | pos.index.isin(pos.head(label_top).index)]
    show = show.sort_values("lP").reset_index(drop=True)
    tiers = [0.42, 0.78, 1.14, 1.50, -0.42, -0.78, -1.14, -1.50]
    for i, r in show.iterrows():
        dy = tiers[i % len(tiers)]
        ax.annotate(str(r.issuer_code), xy=(r.lP, r.lM),
                    xytext=(r.lP, r.lM + dy),
                    fontsize=7.4, ha="center",
                    va="bottom" if dy > 0 else "top",
                    color="#065f46" if r.ever_event else "#334155",
                    fontweight="bold" if r.ever_event else "normal",
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color="#16a34a" if r.ever_event else "#cbd5e1",
                                    shrinkA=0, shrinkB=1.5))

    # keep the view on the data; the boundary line runs far outside it
    lo, hi = float(pos.lM.min()), float(pos.lM.max())
    pad = max(2.2, 0.35 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    cross = (logK - 0.0) / alpha if alpha else np.nan
    if np.isfinite(cross):
        ax.annotate(f"boundary meets $M=1$ at $\\log P = {cross:.2f}$\n"
                    f"($PD = {np.exp(cross):.5f}$), so at $M=1$ the rule\n"
                    f"is just a cut on the level",
                    xy=(cross, 0.0),
                    xytext=(float(pos.lP.min()) + 0.5, lo - pad * 0.72),
                    fontsize=8.8, color="#b91c1c", fontweight="bold", ha="left",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#b91c1c", lw=0.9, alpha=0.92),
                    arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.3,
                                    connectionstyle="arc3,rad=0.12"))

    ax.set_xlabel("$\\log PD(t-1)$", fontsize=11)
    ax.set_ylabel("$\\log M(t)$", fontsize=11)
    ax.set_title(f"All {len(pos)} issuers on their latest month, "
                 f"against the boundary at {workload:.0%} capacity\n"
                 f"almost every issuer sits on $M=1$, so the boundary reduces to a "
                 f"vertical cut on PD",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_b64(fig)


def figure_issuer_momentum(issuer, alpha, logK, workload=0.05):
    """Momentum and PD over time for one issuer, with the event months marked."""
    D = build_frame(workload)
    p = D["frame"]
    g = p[p.issuer_code == issuer].sort_values("month_dt")
    if g.empty:
        return None
    dts = pd.to_datetime(g.month_dt)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)

    ax = axes[0]
    ax.plot(dts, g.pd_now, lw=1.9, color="#1d4ed8", marker="o", ms=3.2)
    ax.axhline(D["thr"], color="#b91c1c", ls="--", lw=1.7,
               label=f"PD threshold {D['thr']:.5f}")
    over = g.pd_now >= D["thr"]
    if over.any():
        ax.scatter(dts[over], g.pd_now[over], s=46, color="#b91c1c", zorder=5,
                   label=f"{int(over.sum())} months above the line")
    ax.set_yscale("log")
    ax.set_ylabel("out-of-fold PD (log)", fontsize=9)
    ax.set_title(f"{issuer}: PD and momentum", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(dts, g.momentum, lw=1.8, color="#7c3aed", marker="o", ms=3.2)
    ax.axhline(1.0, color="#64748b", ls=":", lw=1.6,
               label="$M = 1$, PD unchanged")
    ev = g.event == 1
    if ev.any():
        for d0 in dts[ev]:
            ax.axvline(d0, color="#16a34a", lw=1.2, alpha=0.6)
        ax.plot([], [], color="#16a34a", lw=1.2, label="recorded event month")
    mx = g.momentum.replace([np.inf, -np.inf], np.nan).dropna()
    if len(mx):
        pk = mx.idxmax()
        ax.annotate(f"peak M = {g.loc[pk,'momentum']:.2f}",
                    (pd.to_datetime(g.loc[pk, "month_dt"]), g.loc[pk, "momentum"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points",
                    color="#7c3aed", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("momentum $M(t)$", fontsize=9)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.3, which="both")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_b64(fig)


def figure_issuer_trajectory(issuer, alpha, logK, workload=0.05):
    """One issuer's walk through the (log PD, log M) plane, coloured by time."""
    D = build_frame(workload)
    p = D["frame"]
    u = p[p.usable]
    g = u[u.issuer_code == issuer].sort_values("month_dt")
    if g.empty:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.scatter(u.log_P, u.log_M, s=3, color="#e2e8f0", alpha=0.5, zorder=1)
    ax.plot(g.log_P, g.log_M, color="#f59e0b", lw=1.2, alpha=0.75, zorder=4)
    sc = ax.scatter(g.log_P, g.log_M, c=np.arange(len(g)), cmap="plasma",
                    s=44, zorder=5, edgecolors="white", linewidth=0.5)
    ev = g.event == 1
    if ev.any():
        ax.scatter(g.log_P[ev], g.log_M[ev], s=190, marker="X",
                   facecolors="none", edgecolors="#16a34a", linewidth=2.2,
                   zorder=7, label=f"{int(ev.sum())} recorded event months")
    la = g.iloc[-1]
    ax.scatter([la.log_P], [la.log_M], s=210, marker="*", color="#b91c1c",
               edgecolors="white", linewidth=1.1, zorder=8,
               label=f"latest month {la.month}")

    xs = np.linspace(u.log_P.min(), u.log_P.max(), 100)
    ax.plot(xs, logK - alpha * xs, color="#b91c1c", lw=2.4, zorder=6,
            label=f"boundary  $\\alpha = {alpha:.2f}$")
    ax.axvline(np.log(D["thr"]), color="#1d4ed8", ls="--", lw=1.6,
               label="PD level threshold")
    ax.axhline(0.0, color="#64748b", ls=":", lw=1.3, label="$M = 1$")

    cb = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label("month order, dark = early", fontsize=8.5)
    ax.set_xlabel("$\\log PD(t-1)$", fontsize=10)
    ax.set_ylabel("$\\log M(t)$", fontsize=10)
    mar = float(la.log_M + alpha * la.log_P - logK)
    ax.set_title(f"{issuer}: path through the boundary plane, "
                 f"{len(g)} usable months\n"
                 f"latest margin {mar:+.4f} "
                 f"({'above' if mar >= 0 else 'below'} the boundary)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_b64(fig)


def momentum_table(alpha, logK, workload=0.05):
    """Latest-month coordinates for every issuer, with peak momentum over history."""
    D = build_frame(workload)
    p = D["frame"]
    u = p[p.usable]
    pk = (u.groupby("issuer_code")
            .agg(peak_M=("momentum", "max"),
                 median_M=("momentum", "median"),
                 n_usable=("momentum", "size")))
    pos = issuer_positions(alpha, logK, workload).set_index("issuer_code")
    d = pos.join(pk)
    d["log_P"] = np.log(d.pd_prev)
    d["log_M"] = np.log(d.momentum)
    d = d.reset_index()
    return d.sort_values(["momentum", "margin"], ascending=[False, False]) \
            .reset_index(drop=True)


# ------------------------------------------------------------------ app.py entry
def build_panel(workload=0.05, issuer=None):
    """What app.py calls. Returns figures as base64 plus the tables."""
    alpha, logK, best, lvl, scan, D = pick_alpha(workload)
    return dict(
        alpha=alpha, logK=logK, K=float(np.exp(logK)),
        workload=workload, threshold=D["thr"],
        n_usable=int(D["frame"].usable.sum()), n_events=D["n_events"],
        boundary=dict(best), level=dict(lvl),
        scan=scan, issuers=issuer_positions(alpha, logK, workload),
        events=event_month_positions(alpha, logK, workload),
        figures=dict(boundary=figure_boundary(alpha, logK, workload, issuer),
                     alpha=figure_alpha_scan(workload)))


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(
        description="momentum and the hyperbolic decision boundary on the iBond panel")
    ap.add_argument("--workload", type=float, default=0.05,
                    help="share of issuer-months the team can review, default 0.05")
    ap.add_argument("--alpha", type=float, default=None,
                    help="fix alpha instead of selecting it")
    ap.add_argument("--issuer", type=str, default=None,
                    help="draw one issuer's path on the figure")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    scan, D = scan_alpha(a.workload)
    lvl = scan[scan.rule == "PD level only"].iloc[0]
    b = scan[scan.rule == "hyperbolic boundary"]

    if a.alpha is None:
        best = b.loc[b.mcc.idxmax()]
        alpha, logK = float(best.alpha), float(best.logK)
        how = "selected by MCC, IN SAMPLE"
    else:
        alpha = float(a.alpha)
        logK = boundary_at(D["frame"][D["frame"].usable], alpha, a.workload)
        best = b.iloc[(b.alpha - alpha).abs().argmin()]
        how = "fixed on the command line"

    bar = "=" * 78
    print(bar)
    print("HYPERBOLIC DECISION BOUNDARY ON THE REAL iBond PANEL")
    print(bar)
    print(f"  rule            log M(t) + alpha * log PD(t-1) >= log K")
    print(f"  momentum        M(t) = PD(t) / PD(t-1), consecutive months only")
    print(f"  PD source       out-of-fold CatBoost from firm_shock_panel")
    print(f"  rows usable     {int(D['frame'].usable.sum()):,} of "
          f"{len(D['frame']):,} issuer-months")
    print(f"  events          {D['n_events']} months from "
          f"{len(D['event_issuers'])} issuers")
    print(f"  review capacity {a.workload:.0%}")
    print(f"  alpha           {alpha:.2f}   ({how})")
    print(f"  log K           {logK:.6f}      K = {np.exp(logK):.6g}")
    print(f"  PD threshold    {D['thr']:.6f}")

    print("\n" + bar)
    print("DOES THE BOUNDARY BEAT THE PLAIN PD RULE?  both alarm on the same share")
    print(bar)
    print(f"{'rule':>24s} {'TP':>4s} {'FP':>6s} {'FN':>4s} {'recall':>8s} "
          f"{'prec':>8s} {'MCC':>8s}")
    for nm, r in (("PD level only", lvl), (f"boundary a={alpha:.2f}", best)):
        print(f"{nm:>24s} {int(r.tp):4d} {int(r.fp):6d} {int(r.fn):4d} "
              f"{r.recall:8.4f} {r.precision:8.4f} {r.mcc:8.4f}")
    d_mcc = float(best.mcc) - float(lvl.mcc)
    print(f"\n  difference in MCC: {d_mcc:+.4f}")
    if d_mcc <= 0:
        print("  the extra parameter did NOT help on this panel")
    else:
        print("  NOTE: alpha was chosen on these same 32 events, so this gap is")
        print("        optimistic. It is not out-of-sample evidence.")

    print("\n" + bar)
    print("EVERY RECORDED EVENT MONTH: MOMENTUM AND SIDE OF THE BOUNDARY")
    print(bar)
    ev = event_month_positions(alpha, logK, a.workload)
    print(f"{'issuer':>7s} {'month':>8s} {'PD(t-1)':>10s} {'PD(t)':>10s} "
          f"{'momentum':>10s} {'s':>9s} {'margin':>9s}  side")
    for _, r in ev.iterrows():
        pp = "--" if not np.isfinite(r.pd_prev) else f"{r.pd_prev:.6f}"
        mm = "--" if not np.isfinite(r.momentum) else f"{r.momentum:.4f}"
        ss = "--" if not np.isfinite(r.s) else f"{r.s:+.4f}"
        gg = "--" if not np.isfinite(r.margin) else f"{r.margin:+.4f}"
        print(f"{r.issuer_code:>7s} {r.month:>8s} {pp:>10s} {r.pd_now:10.6f} "
              f"{mm:>10s} {ss:>9s} {gg:>9s}  {r.side}")
    n_al = int((ev.side == "ALARM").sum())
    print(f"\n  {n_al} of {len(ev)} event months are on the alarm side")

    print("\n" + bar)
    print("LATEST MONTH OF EVERY ISSUER, TOP 20 BY MARGIN")
    print(bar)
    pos = issuer_positions(alpha, logK, a.workload)
    print(f"{'issuer':>7s} {'month':>8s} {'PD(t)':>10s} {'momentum':>10s} "
          f"{'margin':>9s}  {'side':>6s}  ever had an event")
    for _, r in pos.head(20).iterrows():
        print(f"{r.issuer_code:>7s} {r.month:>8s} {r.pd_now:10.6f} "
              f"{r.momentum:10.4f} {r.margin:+9.4f}  {r.side:>6s}  "
              f"{'yes' if r.ever_event else 'no'}")
    n_alarm = int((pos.side == "ALARM").sum())
    print(f"\n  {n_alarm} of {len(pos)} issuers alarm on their latest month")
    ev_pos = pos[pos.ever_event]
    print(f"  of the {len(ev_pos)} issuers that ever had an event, "
          f"{int((ev_pos.side == 'ALARM').sum())} alarm now")

    os.makedirs(OUTDIR, exist_ok=True)
    pos.to_csv(os.path.join(OUTDIR, "hyperbolic_issuer_positions.csv"), index=False)
    ev.to_csv(os.path.join(OUTDIR, "hyperbolic_event_months.csv"), index=False)
    scan.to_csv(os.path.join(OUTDIR, "hyperbolic_alpha_scan.csv"), index=False)

    if not a.no_figure:
        for nm, b64 in (("boundary", figure_boundary(alpha, logK, a.workload,
                                                     a.issuer)),
                        ("alpha", figure_alpha_scan(a.workload))):
            fn = os.path.join(OUTDIR, f"fig_hyperbolic_{nm}.png")
            open(fn, "wb").write(base64.b64decode(b64))
            print(f"  wrote {fn}")
    print("  wrote tex_out/hyperbolic_issuer_positions.csv, "
          "hyperbolic_event_months.csv, hyperbolic_alpha_scan.csv")


if __name__ == "__main__":
    main()
