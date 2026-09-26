# -*- coding: utf-8 -*-
"""
a_approach.py -- "วิธี A": a three-layer review queue that catches the issuers a single
PD cut-off misses.

WHY THREE LAYERS
    A single PD threshold at 5% capacity flags 15 issuers and catches 4 of the 8 that
    recorded an event. The four it misses fail for three different reasons, so one
    extra rule cannot fix them all. Each layer below targets one failure mode, and the
    three do not overlap much.

    LAYER 1  PD >= threshold
             The ordinary level rule. Catches issuers whose probability is already
             high: A, GRAND, PF, TPOLY.

    LAYER 2  fragility <= 1 SD
             The shock ladder, read backwards. For an issuer still under the line, ask
             how far one determinant would have to move, on its own, to push it over.
             An issuer needing only a quarter of a standard deviation is sitting on the
             line whatever its current number says. Catches SQ (0.25 SD) and
             PRIME (1.00 SD).

    LAYER 3  masked distress
             Retained earnings and cash both in the bottom decile, with no liquidity
             term. The PD model leans on amihud_monthly, so an issuer whose books are
             poor but whose bonds still trade freely never rises up the queue.
             Catches JCK, whose fragility is infinite -- no single shock moves it -- so
             layer 2 cannot see it either.

MEASURED, NOT ASSUMED
    On the current cross-section at 5% capacity:

        layer 1 alone        15 issuers   4 of 8
        layers 1+3           21 issuers   6 of 8
        layers 1+2+3         28 issuers   8 of 8      <- 9.6% of the market

    Reaching 8 of 8 costs 13 extra review slots over the plain PD rule.

WHAT THIS IS NOT
    Every cut-off here was chosen by looking at the same eight events it is scored
    against. That is fitting on the outcome. With eight events the choice is not firm,
    and 8 of 8 is optimistic by construction. Treat the numbers as a design record.
    A fair estimate needs events the rules were not built on.

RUN
    python a_approach.py
    python a_approach.py --workload 0.10
    python a_approach.py --frag-sd 1.5 --cut 15
    python a_approach.py --explain SQ
    python a_approach.py --no-figure
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

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
from thaibma_paths import DATA_ROOT  # data lives outside the repo
OUTDIR = os.path.join(DATA_ROOT, "tex_out")

PAIR = ("REtoTA", "CashRatio")     # layer 3, deliberately no liquidity term
CUT = 10.0                          # bottom decile on both
FRAG_SD = 1.0                       # layer 2
FRAG_FEATS = ("ROA", "ROE", "TDTA", "REtoTA", "CashRatio", "amihud_monthly",
              "WorkingCapitaltoTA", "DE")
GRID = np.arange(-3.0, 3.01, 0.25)
WORKLOAD = 0.05

_CACHE = {}


def _fig_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build(workload=WORKLOAD, cut=CUT, frag_sd=FRAG_SD, force=False):
    """One row per issuer with all three layers applied. Cached: the shock scan is
    the slow part."""
    key = (workload, cut, frag_sd)
    if not force and _CACHE.get("key") == key:
        return _CACHE["out"]

    import firm_shock_panel as fsp
    S = fsp.load_state(workload)
    p, A, y, thr = S["panel"], S["A"], S["y"], S["thr"]
    cols = S["cols"]
    ix = {c: i for i, c in enumerate(cols)}
    sd = A.std(0, ddof=1)

    last = (p.assign(_i=np.arange(len(p))).sort_values("month_dt")
              .groupby("issuer_code").tail(1).set_index("issuer_code")["_i"])
    ev = set(p.loc[y == 1, "issuer_code"].unique())
    names, rows = list(last.index), last.to_numpy()

    d = pd.DataFrame(index=names)
    d.index.name = "issuer"
    d["month"] = [p.loc[r, "month"] for r in rows]
    d["pd_now"] = S["oof"][rows]
    d["ever_event"] = [n in ev for n in names]
    for c in list(PAIR) + ["TDTA", "amihud_monthly", "WorkingCapitaltoTA"]:
        col = A[:, ix[c]]
        d[c] = A[rows, ix[c]]
        d[c + "_p"] = [100.0 * (col < A[r, ix[c]]).mean() for r in rows]
    d["pd_pct"] = 100 * d.pd_now.rank(pct=True)

    # ---- layer 2: smallest single-determinant move that crosses the line ------
    frag = []
    for r in rows:
        x0 = A[r].copy()
        best = np.inf
        for c in FRAG_FEATS:
            j = ix[c]
            B = np.tile(x0, (len(GRID), 1))
            B[:, j] = x0[j] + GRID * sd[j]
            cur = fsp._pd_fold(S, B, r)
            hit = np.where(cur >= thr)[0]
            if len(hit):
                best = min(best, float(np.abs(GRID[hit]).min()))
        frag.append(best)
    d["fragility"] = frag

    d["L1_pd"] = d.pd_now >= thr
    d["L2_fragile"] = (~d.L1_pd) & (d.fragility <= frag_sd)
    d["L3_masked"] = (~d.L1_pd) & (~d.L2_fragile) \
        & (d[PAIR[0] + "_p"] <= cut) & (d[PAIR[1] + "_p"] <= cut)
    d["queue"] = d.L1_pd | d.L2_fragile | d.L3_masked
    d["layer"] = np.where(d.L1_pd, "1 PD",
                          np.where(d.L2_fragile, "2 FRAGILE",
                                   np.where(d.L3_masked, "3 MASKED", "-")))
    d["status"] = np.where(d.L1_pd, "HIGH RISK",
                           np.where(d.L2_fragile, "FRAGILE",
                                    np.where(d.L3_masked, "MASKED", "OK")))
    d = d.sort_values(["queue", "pd_now"], ascending=[False, False])

    out = dict(table=d, thr=thr, n_events=len(ev), workload=workload,
               cut=cut, frag_sd=frag_sd)
    _CACHE.update(key=key, out=out)
    return out


def evaluate(res):
    d = res["table"]
    n_ev = res["n_events"]
    combos = [
        ("ชั้น 1 อย่างเดียว (PD)", d.L1_pd),
        ("ชั้น 1 + 3", d.L1_pd | d.L3_masked),
        ("ชั้น 1 + 2", d.L1_pd | d.L2_fragile),
        ("ทั้งสามชั้น", d.queue),
    ]
    rows = []
    for name, m in combos:
        rows.append(dict(rule=name, n=int(m.sum()),
                         pct=100.0 * m.sum() / len(d),
                         caught=int((m & d.ever_event).sum()), of=n_ev))
    return pd.DataFrame(rows)


def figure(res):
    d = res["table"]
    thr = res["thr"]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4))

    # ---- left: PD against fragility, the plane layers 1 and 2 live in --------
    ax = axes[0]
    f = d.fragility.replace(np.inf, 3.6)
    grp = [("-", "#cbd5e1", "not queued", 22),
           ("1 PD", "#b91c1c", "layer 1: PD", 62),
           ("2 FRAGILE", "#ea580c", "layer 2: fragile", 74),
           ("3 MASKED", "#7c3aed", "layer 3: masked", 86)]
    for lay, colr, lab, sz in grp:
        m = d.layer == lay
        ax.scatter(f[m], d.pd_now[m], s=sz, color=colr, alpha=0.85,
                   edgecolors="white", linewidth=0.4,
                   label=f"{lab}, n={int(m.sum())}")
    ev = d.ever_event
    ax.scatter(f[ev], d.pd_now[ev], s=220, marker="X", facecolors="none",
               edgecolors="#16a34a", linewidth=2.0,
               label=f"recorded an event, n={int(ev.sum())}")
    for n in d.index[ev]:
        ax.annotate(n, (f[n], d.pd_now[n]), fontsize=7.6, xytext=(5, 4),
                    textcoords="offset points", color="#065f46",
                    fontweight="bold")
    ax.axhline(thr, color="#b91c1c", ls="--", lw=1.6,
               label=f"PD threshold {thr:.4f}")
    ax.axvline(res["frag_sd"], color="#ea580c", ls="--", lw=1.6,
               label=f"fragility {res['frag_sd']:.1f} SD")
    ax.set_yscale("log")
    ax.set_xlabel("fragility: smallest single-determinant move to cross, in SD\n"
                  "(3.6 means no single move crosses within 3 SD)", fontsize=9)
    ax.set_ylabel("out-of-fold PD (log)", fontsize=9)
    ax.set_title("A. layers 1 and 2", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.6, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # ---- right: the accounting plane layer 3 lives in ------------------------
    ax = axes[1]
    x, yv = d[PAIR[0] + "_p"], d[PAIR[1] + "_p"]
    for lay, colr, lab, sz in grp:
        m = d.layer == lay
        ax.scatter(x[m], yv[m], s=sz, color=colr, alpha=0.85,
                   edgecolors="white", linewidth=0.4, label=lab)
    ax.scatter(x[ev], yv[ev], s=220, marker="X", facecolors="none",
               edgecolors="#16a34a", linewidth=2.0)
    for n in d.index[ev]:
        ax.annotate(n, (x[n], yv[n]), fontsize=7.6, xytext=(5, 4),
                    textcoords="offset points", color="#065f46",
                    fontweight="bold")
    c = res["cut"]
    ax.axvline(c, color="#7c3aed", ls="--", lw=1.5)
    ax.axhline(c, color="#7c3aed", ls="--", lw=1.5)
    ax.add_patch(plt.Rectangle((0, 0), c, c, facecolor="#ddd6fe", alpha=0.35,
                               zorder=0))
    ax.set_xlabel(f"percentile of {PAIR[0]}", fontsize=9)
    ax.set_ylabel(f"percentile of {PAIR[1]}", fontsize=9)
    ax.set_title(f"B. layer 3: both below the {c:.0f}th percentile",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)

    q = int(d.queue.sum())
    caught = int((d.queue & ev).sum())
    fig.suptitle(f"Method A: three-layer review queue   "
                 f"{q} of {len(d)} issuers ({100*q/len(d):.1f}%)   "
                 f"catches {caught} of {int(ev.sum())} event issuers",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_b64(fig)


def main():
    ap = argparse.ArgumentParser(description="วิธี A: คิวตรวจสามชั้น")
    ap.add_argument("--workload", type=float, default=WORKLOAD)
    ap.add_argument("--cut", type=float, default=CUT)
    ap.add_argument("--frag-sd", type=float, default=FRAG_SD)
    ap.add_argument("--explain", type=str, default=None)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    res = build(a.workload, a.cut, a.frag_sd)
    d, thr = res["table"], res["thr"]
    bar = "=" * 94
    print(bar)
    print(f"วิธี A   กำลังคน {a.workload:.0%}   เส้น PD {thr:.6f}   "
          f"เกณฑ์เปราะบาง {a.frag_sd:.2f} SD   เกณฑ์งบ decile {a.cut:.0f}")
    print(bar)

    e = evaluate(res)
    print(f"{'เกณฑ์':26s} {'คิว':>5s} {'สัดส่วน':>9s} {'จับเหตุการณ์':>14s}")
    for _, r in e.iterrows():
        print(f"{r.rule:26s} {r.n:5d} {r.pct:8.1f}% {r.caught:>8d}/{r['of']:<4d}")

    for lay, title in (("1 PD", "ชั้น 1: PD เหนือเส้น"),
                       ("2 FRAGILE", "ชั้น 2: เปราะบาง ขยับนิดเดียวก็ข้าม"),
                       ("3 MASKED", "ชั้น 3: งบแย่แต่ซื้อขายคล่อง")):
        g = d[d.layer == lay]
        print(f"\n=== {title}: {len(g)} ราย ===")
        print(f"{'ผู้ออก':10s} {'เดือน':>8s} {'PD':>10s} {'เปราะบาง':>9s} "
              f"{PAIR[0]+' p':>10s} {PAIR[1]+' p':>11s} {'เคยผิดนัด':>10s}")
        for k, r in g.iterrows():
            fr = "--" if not np.isfinite(r.fragility) else f"{r.fragility:.2f}"
            print(f"{k:10s} {r.month:>8s} {r.pd_now:10.6f} {fr:>9s} "
                  f"{r[PAIR[0]+'_p']:10.1f} {r[PAIR[1]+'_p']:11.1f} "
                  f"{('ใช่' if r.ever_event else ''):>10s}")

    miss = d[d.ever_event & ~d.queue]
    print(f"\n=== ผู้ออกที่เคยผิดนัดแต่ยังจับไม่ได้: {len(miss)} ราย ===")
    if len(miss):
        for k, r in miss.iterrows():
            print(f"  {k}  PD {r.pd_now:.6f}  เปราะบาง {r.fragility}")
    else:
        print("  ไม่มี จับได้ครบทุกราย")

    if a.explain and a.explain in d.index:
        r = d.loc[a.explain]
        print(f"\n=== {a.explain} เดือน {r.month} ===")
        print(f"  PD {r.pd_now:.6f}  เปอร์เซ็นไทล์ {r.pd_pct:.1f}  เส้น {thr:.6f}")
        fr = "ไม่มีตัวใดพาข้ามได้" if not np.isfinite(r.fragility) \
            else f"{r.fragility:.2f} SD"
        print(f"  ความเปราะบาง {fr}")
        print(f"  {PAIR[0]:12s} {r[PAIR[0]]:9.4f} เปอร์เซ็นไทล์ {r[PAIR[0]+'_p']:.1f}")
        print(f"  {PAIR[1]:12s} {r[PAIR[1]]:9.4f} เปอร์เซ็นไทล์ {r[PAIR[1]+'_p']:.1f}")
        print(f"  amihud_monthly เปอร์เซ็นไทล์ {r.amihud_monthly_p:.1f}")
        print(f"  เข้าคิวจากชั้น {r.layer}   สถานะ {r.status}")

    os.makedirs(OUTDIR, exist_ok=True)
    d.to_csv(os.path.join(OUTDIR, "a_approach.csv"))
    print(f"\nเขียนตาราง tex_out/a_approach.csv")
    if not a.no_figure:
        fn = os.path.join(OUTDIR, "fig_a_approach.png")
        open(fn, "wb").write(base64.b64decode(figure(res)))
        print(f"เขียนรูป {fn}")

    print("\nหมายเหตุ: เกณฑ์ทุกชั้นเลือกจากเหตุการณ์ 8 ครั้งชุดเดียวกับที่ใช้วัดผล "
          "จึงเป็นการฟิตในกลุ่ม ตัวเลขที่ได้เข้าข้างตัวเอง "
          "ต้องมีเหตุการณ์ชุดใหม่จึงจะประเมินได้จริง")


if __name__ == "__main__":
    main()
