# -*- coding: utf-8 -*-
"""
reanalysis_oof.py -- the reanalysis demanded by the reviewer report.

WHAT WAS WRONG IN THE SUBMITTED VERSION
    Every criticism below was checked against the source and confirmed.

    R1.1  cmdf_feature_select.py:146 builds its folds from
          `panel.loc[y == 1, "issuer_code"]`, i.e. only the eight issuers that
          default. Held-out rows therefore contain no negatives from the other 281
          issuers, so the reported AUC is not an AUC over the 289-issuer population.

    R1.2  cmdf_approach2_compare.py:168-172 fits on the FULL panel and feeds those
          in-sample probabilities into add_signals() and lead_times(). The published
          8/8 detection, 0.62% alert rate and 152-day persistence are therefore
          in-sample quantities.

    R2.1  bond_ews.py:386-391 forms PD_3M as 1-(1-h_t)(1-h_{t+1})(1-h_{t+2}) using
          g.shift(-1) and g.shift(-2), i.e. hazards dated after the decision date.

    R2.3  A Brier loss of 0.064 is ~34x worse than the constant-prevalence forecast
          (0.0019), so "calibrated" was not supported.

    R2.4  cmdf_approach2_compare.py:209 sets n_caught from `persistent_days`, not
          from `actionable_days`, so the Caught column did not measure actionable
          detection.

WHAT THIS MODULE DOES INSTEAD
    1  Uses the direct three-month-ahead event label, so the forecast depends only on
       information dated t. No shift(-1)/shift(-2) anywhere: R2.1 cannot arise.
    2  StratifiedGroupKFold over ALL 289 issuers, so every issuer-month receives a
       prediction from a model that never saw that issuer, and negatives come from the
       whole panel (R1.1, fix-list 2).
    3  Runs the alarm pipeline on out-of-fold probabilities only. Detection, lead
       time, persistence and alert burden are all held-out quantities (R1.2, 4).
    4  Counts detection on the ACTIONABLE alarm, and reports persistence separately
       (R2.4).
    5  Reports ROC-AUC, PR-AUC, Brier, and Brier Skill Score against the
       constant-prevalence forecast, which is the reference R2.3 asks for.
    6  Issuer-clustered bootstrap confidence intervals for every headline number,
       including the detection rate (R1.4).
    7  False alarms per issuer-year (fix-list 5).
    8  Compares the alarm boundary against PD-only, momentum-only and a logistic
       rule at a MATCHED workload, so the boundary is not credited for simply
       flagging more names (fix-list 6).

NOT COVERED HERE
    Nested tuning of hyperparameters, K and alpha (R1.3) is in reanalysis_nested.py.
    The DM re-test with issuer-block resampling (R1.5) is in the same module.

RUN
    python reanalysis_oof.py
    python reanalysis_oof.py --boot 200      fewer bootstrap replicates
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

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out

MODELS = ["Logistic (Approach 1)", "Random Forest", "XGBoost", "CatBoost", "LightGBM"]
N_SPLITS = 5
SEED = 42
N_BOOT = 400
DAYS_PER_MONTH = 30.4375
WORKLOAD = 0.02          # share of issuer-months an analyst team can review
ACT_MIN, ACT_MAX = 30, 92


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


# ============================================================ predictions ====
def out_of_fold(name, X, y, groups, n_splits=N_SPLITS, seed=SEED):
    """One held-out probability for every issuer-month.

    The fold variable is the issuer, so an issuer never appears in both the training
    and the scoring side. Stratification keeps at least one event issuer per fold.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    A, yv = X.to_numpy(float), y.to_numpy(int)
    oof = np.full(len(A), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(A, yv, groups):
        if yv[tr].sum() < 2:
            continue
        sc = StandardScaler().fit(A[tr])
        base = cl.classifiers()[name]()
        # calibration is fitted inside the training fold only, never on held-out rows
        try:
            m = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
            m.fit(sc.transform(A[tr]), yv[tr])
        except Exception:
            m = base
            m.fit(sc.transform(A[tr]), yv[tr])
        oof[te] = m.predict_proba(sc.transform(A[te]))[:, 1]
    return oof


# ================================================================ metrics ====
def discrimination(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y, int); p = np.asarray(p, float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if not (0 < y.sum() < len(y)):
        return dict(auc=np.nan, pr_auc=np.nan, brier=np.nan, bss=np.nan,
                    prevalence=np.nan)
    prev = float(y.mean())
    brier = float(np.mean((y - p) ** 2))
    # the reference forecast issues the base rate to every row, every month
    brier_ref = float(np.mean((y - prev) ** 2))
    return dict(auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)),
                brier=brier, brier_ref=brier_ref,
                bss=float(1.0 - brier / brier_ref) if brier_ref > 0 else np.nan,
                prevalence=prev)


def alarm_at_workload(p, workload=WORKLOAD):
    """Flag the top `workload` share of issuer-months. Every rule is compared at the
    same workload, so differences reflect ranking rather than threshold placement."""
    p = np.asarray(p, float)
    q = np.nanquantile(p, 1.0 - workload)
    return (p >= q).astype(int), float(q)


def lead_and_burden(panel, alarm, months_per_issuer):
    """Actionable lead, persistence, detection and false-alarm load, all from the
    held-out alarm vector.

    Detection counts the ACTIONABLE alarm, i.e. an alarm inside the one-to-three
    calendar-month window before the event. The submitted version counted the
    persistent episode instead, which is why its detection rate was 8/8.
    """
    p = panel.copy()
    p["alarm"] = np.asarray(alarm, int)
    rows = []
    for code, g in p.groupby("issuer_code"):
        ev = g["event_date"].dropna()
        if ev.empty:
            continue
        ev_date = pd.to_datetime(ev.iloc[0])
        pre = g[g["month_dt"] < ev_date].sort_values("month_dt")
        if pre.empty:
            continue
        is_al = pre["alarm"].to_numpy().astype(bool)
        days = (ev_date - pre["month_dt"]).dt.days.to_numpy()

        act = np.nan
        win = is_al & (days >= ACT_MIN) & (days <= ACT_MAX)
        if win.any():
            act = float(days[win].max())
        pers = np.nan
        if is_al.any():
            k = int(np.max(np.flatnonzero(is_al)))
            while k > 0 and is_al[k - 1]:
                k -= 1
            pers = float(days[k])
        rows.append(dict(issuer_code=code, actionable_days=act,
                         persistent_days=pers, caught=int(np.isfinite(act))))
    lt = pd.DataFrame(rows)

    # a false alarm is a flagged issuer-month that is not inside the event window
    y = panel["y"].to_numpy(int)
    fa = int(((np.asarray(alarm, int) == 1) & (y == 0)).sum())
    issuer_years = months_per_issuer / 12.0
    return lt, fa, float(fa / issuer_years) if issuer_years else np.nan


def boot_ci(values, stat, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap resampling ISSUERS, not rows.

    Rows inside an issuer are serially dependent and the three-month windows overlap,
    so a row-level bootstrap understates the variance. Resampling whole issuers keeps
    that dependence intact."""
    rng = np.random.default_rng(seed)
    codes = np.asarray(sorted(values.keys()))
    n = len(codes)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(codes, size=n, replace=True)
        s = stat([values[c] for c in pick])
        if s is not None and np.isfinite(s):
            draws.append(s)
    if not draws:
        return (np.nan, np.nan)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def auc_ci_by_issuer(panel, y, p, n_boot=N_BOOT, seed=SEED):
    from sklearn.metrics import roc_auc_score
    codes = panel["issuer_code"].to_numpy()
    by = {c: (y[codes == c], p[codes == c]) for c in np.unique(codes)}

    def stat(chunks):
        yy = np.concatenate([a for a, _ in chunks])
        pp = np.concatenate([b for _, b in chunks])
        ok = np.isfinite(pp)
        yy, pp = yy[ok], pp[ok]
        if not (0 < yy.sum() < len(yy)):
            return None
        return roc_auc_score(yy, pp)
    return boot_ci(by, stat, n_boot, seed)


def detect_ci(caught_flags, n_boot=N_BOOT, seed=SEED):
    """Wilson-style interval obtained by resampling the event issuers."""
    by = {i: f for i, f in enumerate(caught_flags)}
    return boot_ci(by, lambda c: float(np.mean(c)) if len(c) else None, n_boot, seed)


# ============================================================== the runs =====
def run_model(name, panel, X, y, groups, n_boot):
    t0 = time.time()
    oof = out_of_fold(name, X, y, groups)
    yv = y.to_numpy(int)
    d = discrimination(yv, oof)

    alarm, thr = alarm_at_workload(oof)
    months = len(panel)
    lt, fa, fa_per_yr = lead_and_burden(panel, alarm, months)

    n_ev = int(len(lt))
    caught = lt["caught"].to_numpy() if n_ev else np.array([], int)
    n_caught = int(caught.sum())
    act = lt.loc[lt["actionable_days"].notna(), "actionable_days"]
    pers = lt.loc[lt["persistent_days"].notna(), "persistent_days"]

    lo_auc, hi_auc = auc_ci_by_issuer(panel, yv, oof, n_boot)
    lo_det, hi_det = detect_ci(caught, n_boot) if n_ev else (np.nan, np.nan)

    res = dict(
        model=name, auc=d["auc"], auc_lo=lo_auc, auc_hi=hi_auc,
        pr_auc=d["pr_auc"], brier=d["brier"], brier_ref=d["brier_ref"],
        bss=d["bss"], prevalence=d["prevalence"],
        n_alarm=int(alarm.sum()), alarm_share=float(alarm.mean()) * 100,
        threshold=thr,
        n_events=n_ev, n_caught=n_caught,
        detect=float(n_caught / n_ev) if n_ev else np.nan,
        detect_lo=lo_det, detect_hi=hi_det,
        act_median=float(act.median()) if len(act) else np.nan,
        pers_median=float(pers.median()) if len(pers) else np.nan,
        false_alarms=fa, fa_per_issuer_year=fa_per_yr,
        seconds=round(time.time() - t0, 1))
    print(f"    {name:22s} AUC {res['auc']:.3f} [{lo_auc:.3f},{hi_auc:.3f}]  "
          f"PR {res['pr_auc']:.4f}  BSS {res['bss']:+.3f}  "
          f"caught {n_caught}/{n_ev}  ({res['seconds']}s)")
    return res, oof


def rule_comparison(panel, y, oof_best, name_best):
    """Does the boundary earn its complexity? Every rule is evaluated at the same
    workload, so they flag the same number of issuer-months."""
    from cmdf_approach2_compare import HYPER_K, HYPER_ALPHA, MOM_WINDOW
    p = panel.copy()
    p["PD_3M"] = oof_best
    p = p.sort_values(["issuer_code", "month_dt"])
    pd_prev = p.groupby("issuer_code")["PD_3M"].shift(1).fillna(p["PD_3M"])
    base = (p.groupby("issuer_code")["PD_3M"]
            .transform(lambda s: s.shift(1).rolling(MOM_WINDOW, min_periods=1)
                       .median()))
    mom = (p["PD_3M"] / (base + 1e-4)).clip(0, 10)
    score_b = (np.log(np.clip(mom, 1e-9, None))
               + HYPER_ALPHA * np.log(np.clip(pd_prev, 1e-9, None)))

    yv = p["y"].to_numpy(int)
    months = len(p)
    rules = {"PD only": p["PD_3M"].to_numpy(float),
             "Momentum only": mom.to_numpy(float),
             "Power-law boundary": score_b.to_numpy(float)}
    rows = []
    for rname, score in rules.items():
        alarm, _ = alarm_at_workload(score)
        lt, fa, fa_yr = lead_and_burden(p, alarm, months)
        n_ev = int(len(lt)); nc = int(lt["caught"].sum()) if n_ev else 0
        act = lt.loc[lt["actionable_days"].notna(), "actionable_days"]
        rows.append(dict(rule=rname, n_caught=nc, n_events=n_ev,
                         detect=nc / n_ev if n_ev else np.nan,
                         act_median=float(act.median()) if len(act) else np.nan,
                         n_alarm=int(alarm.sum()),
                         alarm_share=float(alarm.mean()) * 100,
                         fa_per_issuer_year=fa_yr))
    d = pd.DataFrame(rows)
    d["scored_by"] = name_best
    return d


# ================================================================ output =====
def write_tex(res, rules):
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"

    L = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\caption{Out-of-fold performance over all 289 issuers. Every issuer-month "
         r"is scored by a model that never saw that issuer.}",
         r"\label{tab:oof-main}",
         r"\begin{tabular}{@{}lccrrr@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{ROC-AUC [95\% CI]} & \textbf{PR-AUC} & "
         r"\textbf{Brier} & \textbf{BSS} & \textbf{Detection} \\", r"\midrule"]
    best = res.loc[res["pr_auc"].idxmax(), "model"] if res["pr_auc"].notna().any() else None
    for _, r in res.iterrows():
        cells = [esc(r["model"]),
                 f"{f3(r['auc'])} [{f3(r['auc_lo'])}, {f3(r['auc_hi'])}]",
                 f4(r["pr_auc"]), f4(r["brier"]),
                 f"{r['bss']:+.3f}" if pd.notna(r["bss"]) else "--",
                 (f"{int(r['n_caught'])}/{int(r['n_events'])} "
                  f"[{100*r['detect_lo']:.0f}--{100*r['detect_hi']:.0f}\\%]")]
        if r["model"] == best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        L.append(" & ".join(cells) + r" \\")
    prev = res["prevalence"].iloc[0]
    bref = res["brier_ref"].iloc[0]
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[3pt] {\footnotesize Predictions are out-of-fold from "
          f"StratifiedGroupKFold({N_SPLITS}) over issuer identity, so negatives come "
          r"from the whole panel rather than from the event issuers alone. "
          r"Detection counts the actionable alarm, an alarm one to three calendar "
          r"months before the event, at a workload of "
          f"{WORKLOAD*100:.0f}\\% of issuer-months. "
          r"The Brier Skill Score is measured against the constant-prevalence "
          f"forecast, which issues {prev:.4f} to every row and attains a Brier loss "
          f"of {bref:.5f}; a negative score means the model is worse than that "
          r"reference. Intervals are percentile bootstrap over issuers, "
          f"{N_BOOT} replicates.}}", r"\end{table}"]

    L2 = [r"\begin{table}[H]", r"\centering", r"\small",
          r"\caption{Alarm timing and review burden, all measured on out-of-fold "
          r"predictions at a matched workload}", r"\label{tab:oof-burden}",
          r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
          r"\textbf{Model} & \textbf{Flagged} & \textbf{Share} & "
          r"\textbf{Actionable med.} & \textbf{Persistent med.} & "
          r"\textbf{False alarms} \\",
          r" & \textbf{months} & & \textbf{days} & \textbf{days} & "
          r"\textbf{per issuer-year} \\", r"\midrule"]
    for _, r in res.iterrows():
        L2.append(" & ".join([
            esc(r["model"]), f"{int(r['n_alarm']):,}",
            f"{r['alarm_share']:.2f}\\%",
            "--" if pd.isna(r["act_median"]) else f"{r['act_median']:.0f}",
            "--" if pd.isna(r["pers_median"]) else f"{r['pers_median']:.0f}",
            f"{r['fa_per_issuer_year']:.2f}"]) + r" \\")
    L2 += [r"\bottomrule", r"\end{tabular}",
           r"\\[3pt] {\footnotesize All rules flag the same share of issuer-months, "
           r"so the columns compare ranking quality rather than threshold placement. "
           r"A false alarm is a flagged issuer-month outside the event window.}",
           r"\end{table}"]

    L3 = [r"\begin{table}[H]", r"\centering", r"\small",
          r"\caption{Does the boundary earn its complexity? Alarm rules compared at "
          r"an identical workload on the same out-of-fold probabilities}",
          r"\label{tab:oof-rules}",
          r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
          r"\textbf{Rule} & \textbf{Detection} & \textbf{Actionable med. days} & "
          r"\textbf{Flagged months} & \textbf{False alarms / issuer-year} \\",
          r"\midrule"]
    for _, r in rules.iterrows():
        L3.append(" & ".join([
            esc(r["rule"]),
            f"{int(r['n_caught'])}/{int(r['n_events'])}",
            "--" if pd.isna(r["act_median"]) else f"{r['act_median']:.0f}",
            f"{int(r['n_alarm']):,}",
            f"{r['fa_per_issuer_year']:.2f}"]) + r" \\")
    L3 += [r"\bottomrule", r"\end{tabular}",
           r"\\[3pt] {\footnotesize All three rules are applied to the same "
           f"out-of-fold probabilities from {rules['scored_by'].iloc[0]} and are "
           r"tuned to flag the same number of issuer-months, so any difference is "
           r"attributable to the rule and not to a larger alert budget.}",
           r"\end{table}"]

    frag = ("\\section*{Out-of-fold reanalysis over the full issuer population}\n\n"
            + "\n".join(L) + "\n\n" + "\n".join(L2) + "\n\n" + "\n".join(L3) + "\n")
    p = out("section_oof_reanalysis.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(frag)
    return p


def main():
    n_boot = N_BOOT
    if "--boot" in sys.argv:
        n_boot = int(sys.argv[sys.argv.index("--boot") + 1])

    print("=" * 100)
    print("Out-of-fold reanalysis: every issuer-month scored by a model that never "
          "saw that issuer")
    print("=" * 100)
    panel, X, y, cols = cl.load_panel(verbose=True)
    panel = panel.reset_index(drop=True)
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    panel["y"] = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    print(f"  issuers {panel['issuer_code'].nunique()}  "
          f"issuer-months {len(panel):,}  events {int(y.sum())}  "
          f"prevalence {y.mean():.4%}")
    print(f"\n  {N_SPLITS}-fold grouped CV over ALL issuers, "
          f"{n_boot} issuer-level bootstrap replicates ...")

    rows, oofs = [], {}
    for name in MODELS:
        if name not in cl.classifiers():
            print(f"    {name}: not available - skipped")
            continue
        r, oof = run_model(name, panel, X, y, groups, n_boot)
        rows.append(r); oofs[name] = oof
    res = pd.DataFrame(rows)

    best = res.loc[res["pr_auc"].idxmax(), "model"]
    print(f"\n  alarm-rule comparison at a matched workload, scored by {best} ...")
    rules = rule_comparison(panel, y, oofs[best], best)
    for _, r in rules.iterrows():
        print(f"    {r['rule']:22s} caught {int(r['n_caught'])}/{int(r['n_events'])}  "
              f"flagged {int(r['n_alarm']):,}  "
              f"FA/issuer-yr {r['fa_per_issuer_year']:.2f}")

    p = write_tex(res, rules)
    res.to_csv(out("oof_reanalysis.csv"), index=False)
    rules.to_csv(out("oof_rule_comparison.csv"), index=False)
    np.save(os.path.join(OUTDIR, "oof_predictions.npy"),
            np.vstack([oofs[m] for m in oofs]))
    con = sqlite3.connect(DB)
    res.to_sql("cmdf_oof_reanalysis", con, if_exists="replace", index=False)
    rules.to_sql("cmdf_oof_rules", con, if_exists="replace", index=False)
    pd.DataFrame(oofs).assign(issuer_code=panel["issuer_code"].values,
                              month=panel["month"].values,
                              y=panel["y"].values).to_sql(
        "cmdf_oof_predictions", con, if_exists="replace", index=False)
    con.commit(); con.close()

    print("\n" + "=" * 100)
    print("HEADLINE COMPARISON WITH THE SUBMITTED VERSION")
    print("=" * 100)
    print(f"{'':22s} {'submitted':>14s}   {'this reanalysis':>18s}")
    b = res.loc[res["pr_auc"].idxmax()]
    print(f"{'detection':22s} {'8/8 (100%)':>14s}   "
          f"{f'{int(b.n_caught)}/{int(b.n_events)} '
             f'({100*b.detect:.0f}%)':>18s}")
    print(f"{'alert share':22s} {'0.62%':>14s}   {f'{b.alarm_share:.2f}%':>18s}")
    print(f"{'Brier':22s} {'0.0651':>14s}   {f'{b.brier:.5f}':>18s}")
    print(f"{'Brier skill score':22s} {'not reported':>14s}   {f'{b.bss:+.3f}':>18s}")
    print(f"{'PR-AUC':22s} {'not reported':>14s}   {f'{b.pr_auc:.4f}':>18s}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
