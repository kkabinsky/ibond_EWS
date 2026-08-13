# -*- coding: utf-8 -*-
"""
reanalysis_rules.py -- the decision-boundary comparison the reviewers asked for
(fix-list item 6), computed rather than asserted.

WHAT IS COMPARED
    Every rule maps the same out-of-fold probability vector to an alarm. None of them
    is allowed to gain by flagging more names than another: the comparison is run in
    two explicit regimes.

    REGIME A, detection-matched.
        For each rule, sweep the threshold from strict to loose and record the SMALLEST
        review workload at which the rule first reaches a given detection count. This
        answers "what does it cost to catch k of the 8 events with this rule".

    REGIME B, workload-matched.
        Fix the workload and read off the detection. This answers "given a fixed
        analyst capacity, which rule catches most".

    Reporting only one regime is what allows a weaker rule to look better, which is why
    both are produced here.

RULES
    PD level              threshold on the out-of-fold probability
    Momentum only         threshold on PD/median(previous 12 months)
    Logistic score        logistic regression on (log PD_prev, log Momentum),
                          fitted inside the training folds only
    Spline / GAM          an additive spline model: SplineTransformer on each of
                          (log PD_prev, log Momentum) followed by logistic
                          regression, again fitted inside the training folds. This is
                          the non-linear frontier that was previously described in the
                          manuscript but never implemented; no GAM code existed in the
                          repository before this module.
    Power-law boundary    log M + alpha*log P_prev >= log K, the rule this paper
                          proposes

    The two fitted rules use the SAME grouped folds as the probability model, so their
    frontiers are held out exactly like everything else.

RUN
    python reanalysis_rules.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cmdf_tree_classify as cl
import cmdf_tree_models as tm
from reanalysis_oof import (ACT_MAX, ACT_MIN, N_SPLITS, SEED, WORKLOAD,
                            lead_and_burden, out_of_fold)

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out
MOM_WINDOW = 12
HYPER_K, HYPER_ALPHA = 0.35, 0.55
SCORER = "CatBoost"          # highest PR-AUC in the out-of-fold reanalysis
WORKLOADS = (0.0062, 0.02)   # the rate claimed in the submitted draft, and 2%


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def signals(panel, p):
    """PD, lagged PD and momentum, in the panel's original row order."""
    d = panel.copy()
    d["PD_3M"] = np.asarray(p, float)
    d["_ord"] = np.arange(len(d))
    d = d.sort_values(["issuer_code", "month_dt"])
    d["PD_prev"] = d.groupby("issuer_code")["PD_3M"].shift(1).fillna(d["PD_3M"])
    # the first month of each issuer has no prior window, so the rolling median is
    # undefined there; fall back to the issuer's own current level, which makes the
    # momentum of a first observation exactly 1 rather than missing. Leaving it NaN
    # silently removed 289 rows from every rule that has to be fitted.
    d["_base"] = (d.groupby("issuer_code")["PD_3M"]
                  .transform(lambda s: s.shift(1)
                             .rolling(MOM_WINDOW, min_periods=1).median()))
    d["_base"] = d["_base"].fillna(d["PD_3M"])
    d["Momentum"] = (d["PD_3M"] / (d["_base"] + 1e-4)).clip(0, 10)
    return d.sort_values("_ord").drop(columns=["_ord", "_base"]).reset_index(drop=True)


def fit_frontier(kind, d, y, groups):
    """Out-of-fold score from a rule that has to be estimated, not just thresholded.

    kind is either 'logistic' or 'gam'. Both take the two signals as input and are
    fitted inside the training folds of the same grouped split used elsewhere.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    Z = np.column_stack([np.log(np.clip(d["PD_prev"].to_numpy(float), 1e-9, None)),
                         np.log(np.clip(d["Momentum"].to_numpy(float), 1e-9, None))])
    # a defensive guard: any residual non-finite value would silently drop the whole
    # fold rather than the single row that caused it
    if not np.isfinite(Z).all():
        med = np.nanmedian(np.where(np.isfinite(Z), Z, np.nan), axis=0)
        bad = ~np.isfinite(Z)
        Z[bad] = np.take(med, np.where(bad)[1])
        print(f"      replaced {int(bad.sum())} non-finite signal values with the "
              f"column median")
    yv = np.asarray(y, int)
    oof = np.full(len(Z), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(Z, yv, groups):
        if yv[tr].sum() < 2:
            continue
        if kind == "gam":
            # one spline basis per input, then an additive logistic link: the standard
            # way to obtain a GAM-style frontier without an extra dependency
            # knots at quantiles rather than at equal spacing: both log-signals are
            # heavily skewed, so uniform knots would put most of the flexibility in a
            # sparse tail and leave the dense centre under-resolved. This gives the
            # flexible frontier its best chance against the two-parameter boundary.
            model = make_pipeline(
                SplineTransformer(n_knots=5, degree=3, include_bias=False,
                                  knots="quantile"),
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))
        try:
            model.fit(Z[tr], yv[tr])
            oof[te] = model.predict_proba(Z[te])[:, 1]
        except Exception as ex:
            print(f"      {kind} fold failed: {ex}")
    return oof


def rule_scores(d, y, groups):
    """One score vector per rule. Higher means more alarming for every rule."""
    pdv = d["PD_3M"].to_numpy(float)
    prev = d["PD_prev"].to_numpy(float)
    mom = d["Momentum"].to_numpy(float)
    power = (np.log(np.clip(mom, 1e-9, None))
             + HYPER_ALPHA * np.log(np.clip(prev, 1e-9, None)) - np.log(HYPER_K))
    print("      fitting the logistic frontier out of fold ...")
    lg = fit_frontier("logistic", d, y, groups)
    print("      fitting the spline/GAM frontier out of fold ...")
    gam = fit_frontier("gam", d, y, groups)
    return {"PD level only": pdv,
            "Momentum only": mom,
            "Logistic score (2 signals)": lg,
            "Spline/GAM frontier (2 signals)": gam,
            f"Power-law boundary $K$={HYPER_K}, $\\alpha$={HYPER_ALPHA}": power}


def sweep(panel, score, n_grid=260):
    """Detection, workload and false-alarm load across the whole threshold range."""
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    if ok.sum() == 0:
        return pd.DataFrame()
    qs = np.linspace(0.90, 0.99995, n_grid)
    rows = []
    months = len(panel)
    for q in qs:
        thr = np.nanquantile(s[ok], q)
        alarm = (np.nan_to_num(s, nan=-np.inf) >= thr).astype(int)
        share = alarm.mean()
        if share <= 0:
            continue
        lt, fa, fa_yr = lead_and_burden(panel, alarm, months)
        if lt.empty:
            continue
        act = lt.loc[lt["actionable_days"].notna(), "actionable_days"]
        pers = lt.loc[lt["persistent_days"].notna(), "persistent_days"]
        rows.append(dict(q=q, threshold=float(thr), n_alarm=int(alarm.sum()),
                         share=float(share) * 100,
                         n_caught=int(lt["caught"].sum()), n_events=int(len(lt)),
                         act_median=float(act.median()) if len(act) else np.nan,
                         pers_median=float(pers.median()) if len(pers) else np.nan,
                         fa_per_issuer_year=fa_yr))
    return pd.DataFrame(rows)


def regime_a(sweeps, target=None):
    """Cheapest workload at which each rule first reaches the detection target."""
    rows = []
    if target is None:
        target = int(min(s["n_caught"].max() for s in sweeps.values() if not s.empty))
    for name, s in sweeps.items():
        if s.empty:
            continue
        hit = s[s["n_caught"] >= target]
        best_possible = int(s["n_caught"].max())
        if hit.empty:
            rows.append(dict(rule=name, target=target, reached=False,
                             max_detect=best_possible, share=np.nan,
                             n_alarm=np.nan, act_median=np.nan, pers_median=np.nan,
                             fa_per_issuer_year=np.nan))
            continue
        r = hit.loc[hit["share"].idxmin()]
        rows.append(dict(rule=name, target=target, reached=True,
                         max_detect=best_possible, share=r["share"],
                         n_alarm=int(r["n_alarm"]), act_median=r["act_median"],
                         pers_median=r["pers_median"],
                         fa_per_issuer_year=r["fa_per_issuer_year"]))
    return pd.DataFrame(rows), target


def regime_b(sweeps, workload):
    """Detection at a fixed workload."""
    rows = []
    for name, s in sweeps.items():
        if s.empty:
            continue
        r = s.iloc[(s["share"] - workload * 100).abs().argmin()]
        rows.append(dict(rule=name, workload=workload * 100, share=r["share"],
                         n_alarm=int(r["n_alarm"]), n_caught=int(r["n_caught"]),
                         n_events=int(r["n_events"]), act_median=r["act_median"],
                         pers_median=r["pers_median"],
                         fa_per_issuer_year=r["fa_per_issuer_year"]))
    return pd.DataFrame(rows)


def write_tex(a, target, bs):
    f0 = lambda v: "--" if pd.isna(v) else f"{v:.0f}"
    f2 = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"

    L = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\caption{Decision rules compared in two regimes on identical out-of-fold "
         r"probabilities. Panel A fixes the detection target and reports the workload "
         r"each rule needs; Panel B fixes the workload and reports detection.}",
         r"\label{tab:boundary-baselines}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"\multicolumn{6}{@{}l}{\textit{Panel A. Detection-matched: cheapest "
         f"workload reaching {target} of 8 events" + r"}} \\", r"\midrule",
         r"\textbf{Decision rule} & \textbf{Max detect.} & \textbf{Workload} & "
         r"\textbf{Action med.} & \textbf{Persist. med.} & \textbf{FA / issuer-yr} \\",
         r"\midrule"]
    best = a.loc[a["share"].idxmin(), "rule"] if a["share"].notna().any() else None
    for _, r in a.iterrows():
        cells = [esc(r["rule"]) if "$" not in str(r["rule"]) else str(r["rule"]),
                 f"{int(r['max_detect'])}/8",
                 "never" if not r["reached"] else f"{r['share']:.2f}\\%",
                 f0(r["act_median"]) + (" d" if pd.notna(r["act_median"]) else ""),
                 f0(r["pers_median"]) + (" d" if pd.notna(r["pers_median"]) else ""),
                 f2(r["fa_per_issuer_year"])]
        if r["rule"] == best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        L.append(" & ".join(cells) + r" \\")

    for w, b in bs:
        L += [r"\midrule",
              r"\multicolumn{6}{@{}l}{\textit{Panel B. Workload-matched at "
              f"{w*100:.2f}\\% of issuer-months" + r"}} \\", r"\midrule",
              r"\textbf{Decision rule} & \textbf{Detection} & \textbf{Workload} & "
              r"\textbf{Action med.} & \textbf{Persist. med.} & "
              r"\textbf{FA / issuer-yr} \\", r"\midrule"]
        bb = b.loc[b["n_caught"].idxmax(), "rule"] if not b.empty else None
        for _, r in b.iterrows():
            cells = [esc(r["rule"]) if "$" not in str(r["rule"]) else str(r["rule"]),
                     f"{int(r['n_caught'])}/{int(r['n_events'])}",
                     f"{r['share']:.2f}\\%",
                     f0(r["act_median"]) + (" d" if pd.notna(r["act_median"]) else ""),
                     f0(r["pers_median"]) + (" d" if pd.notna(r["pers_median"]) else ""),
                     f2(r["fa_per_issuer_year"])]
            if r["rule"] == bb:
                cells = [r"\textbf{" + c + "}" for c in cells]
            L.append(" & ".join(cells) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[3pt] {\footnotesize All rules are applied to the same out-of-fold "
          f"probabilities from {SCORER}, which attains the highest PR-AUC. "
          r"The logistic and spline frontiers are themselves fitted inside the "
          r"training folds of the same grouped split, so no rule sees an issuer it is "
          r"later scored on. \emph{Max detect.} is the highest detection the rule "
          r"attains at any threshold. A false alarm is a flagged issuer-month outside "
          r"the event window, expressed per issuer-year over the "
          r"289-issuer panel.}", r"\end{table}"]

    frag = ("\\section*{Decision-rule comparison}\n\n" + "\n".join(L) + "\n")
    p = out("section_rules.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(frag)
    return p


def main():
    print("=" * 100)
    print("Decision-rule comparison, computed out of fold")
    print("=" * 100)
    panel, X, y, cols = cl.load_panel(verbose=True)
    panel = panel.reset_index(drop=True)
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    panel["y"] = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()

    print(f"\n  out-of-fold probabilities from {SCORER} ...")
    p_oof = out_of_fold(SCORER, X, y, groups)
    d = signals(panel, p_oof)
    d["y"] = panel["y"].values
    d["event_date"] = panel["event_date"].values

    print("  building rule scores ...")
    scores = rule_scores(d, y, groups)

    print("\n  sweeping thresholds ...")
    sweeps = {}
    for name, s in scores.items():
        t0 = time.time()
        sweeps[name] = sweep(d, s)
        mx = int(sweeps[name]["n_caught"].max()) if not sweeps[name].empty else 0
        print(f"    {name[:44]:46s} max detection {mx}/8  ({time.time()-t0:.0f}s)")

    a, target = regime_a(sweeps)
    bs = [(w, regime_b(sweeps, w)) for w in WORKLOADS]

    print(f"\n  PANEL A  cheapest workload reaching {target}/8")
    for _, r in a.iterrows():
        print(f"    {r['rule'][:44]:46s} max {int(r['max_detect'])}/8  "
              + ("never reaches target" if not r["reached"]
                 else f"workload {r['share']:.2f}%  FA/yr {r['fa_per_issuer_year']:.2f}"))
    for w, b in bs:
        print(f"\n  PANEL B  workload {w*100:.2f}%")
        for _, r in b.iterrows():
            print(f"    {r['rule'][:44]:46s} {int(r['n_caught'])}/8  "
                  f"flagged {int(r['n_alarm']):,} ({r['share']:.2f}%)  "
                  f"FA/yr {r['fa_per_issuer_year']:.2f}")

    p = write_tex(a, target, bs)
    a.to_csv(out("rules_regime_a.csv"), index=False)
    pd.concat([b.assign(target_workload=w) for w, b in bs],
              ignore_index=True).to_csv(out("rules_regime_b.csv"), index=False)
    pd.concat([s.assign(rule=n) for n, s in sweeps.items()],
              ignore_index=True).to_csv(out("rules_sweep.csv"), index=False)
    con = sqlite3.connect(DB)
    a.to_sql("cmdf_rules_a", con, if_exists="replace", index=False)
    pd.concat([b.assign(target_workload=w) for w, b in bs],
              ignore_index=True).to_sql("cmdf_rules_b", con, if_exists="replace",
                                        index=False)
    con.commit(); con.close()
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
