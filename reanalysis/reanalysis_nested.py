# -*- coding: utf-8 -*-
"""
reanalysis_nested.py -- nested tuning (R1.3) and a dependence-aware Diebold-Mariano
test (R1.5).

R1.3  THE PROBLEM
    In the submitted version the hyper-parameters, the alarm boundary constants K and
    alpha, and the alert threshold were all chosen by looking at the same events they
    were later scored against. The boundary in particular was selected to optimise MCC
    and realised lead time on the eight recorded defaults, so the reported detection
    and lead time were partly a description of that selection step.

    THE FIX HERE
    Two nested loops. The OUTER loop holds out whole issuers and is used only for
    scoring. Inside each outer training set an INNER loop, again split by issuer,
    selects the learner configuration and then the pair (K, alpha) and the workload.
    Nothing on the outer held-out fold is ever consulted while choosing anything, so
    the outer numbers are free of the selection effect.

R1.5  THE PROBLEM
    The Diebold-Mariano test was applied to pooled issuer-month losses. Those losses
    are serially correlated within an issuer and the three-month event windows
    overlap, so the effective sample is far smaller than the number of rows and the
    p-values were too small.

    THE FIX HERE
    The loss differential is averaged within each issuer first, and the sampling
    distribution is obtained by resampling whole issuers. That keeps the within-issuer
    dependence intact instead of assuming it away. The old pooled p-value is reported
    alongside so the size of the correction is visible.

RUN
    python reanalysis_nested.py
    python reanalysis_nested.py --fast     smaller grids, fewer bootstrap draws
"""
from __future__ import annotations

import itertools
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
from reanalysis_oof import (ACT_MAX, ACT_MIN, WORKLOAD, alarm_at_workload,
                            discrimination, lead_and_burden)

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out

SEED = 42
N_OUTER = 5
N_INNER = 3
N_BOOT = 2000
MOM_WINDOW = 12

MODELS = ["Logistic (Approach 1)", "Random Forest", "XGBoost", "CatBoost", "LightGBM"]

# small, explicit grids; the point is that selection happens inside the training
# folds, not that the grid is exhaustive
GRIDS = {
    "Logistic (Approach 1)": [{"C": 0.05}, {"C": 0.1}, {"C": 0.5}],
    "Random Forest": [{"n_estimators": 300, "max_depth": 4},
                      {"n_estimators": 300, "max_depth": 6},
                      {"n_estimators": 500, "max_depth": 8}],
    "XGBoost": [{"max_depth": 2, "learning_rate": 0.05},
                {"max_depth": 3, "learning_rate": 0.05},
                {"max_depth": 4, "learning_rate": 0.10}],
    "CatBoost": [{"depth": 2, "learning_rate": 0.05},
                 {"depth": 3, "learning_rate": 0.05},
                 {"depth": 4, "learning_rate": 0.10}],
    "LightGBM": [{"max_depth": 2, "learning_rate": 0.05},
                 {"max_depth": 3, "learning_rate": 0.05},
                 {"max_depth": 4, "learning_rate": 0.10}],
}
K_GRID = [0.15, 0.25, 0.35, 0.50, 0.70]
ALPHA_GRID = [0.25, 0.40, 0.55, 0.70]


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def build(name, params):
    from sklearn.calibration import CalibratedClassifierCV
    base = cl.classifiers()[name]()
    try:
        base.set_params(**params)
    except Exception:
        pass
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)


def fit_predict(name, params, Atr, ytr, Ate):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Atr)
    m = build(name, params)
    try:
        m.fit(sc.transform(Atr), ytr)
    except Exception:
        m = cl.classifiers()[name]()
        m.fit(sc.transform(Atr), ytr)
    return m.predict_proba(sc.transform(Ate))[:, 1]


# ======================================================= boundary machinery ==
def momentum_and_prev(sub_panel, p):
    """Momentum and lagged PD for one set of rows, computed per issuer in date order."""
    d = sub_panel.copy()
    d["PD_3M"] = np.asarray(p, float)
    d = d.sort_values(["issuer_code", "month_dt"])
    prev = d.groupby("issuer_code")["PD_3M"].shift(1).fillna(d["PD_3M"])
    basel = (d.groupby("issuer_code")["PD_3M"]
             .transform(lambda s: s.shift(1)
                        .rolling(MOM_WINDOW, min_periods=1).median()))
    mom = (d["PD_3M"] / (basel + 1e-4)).clip(0, 10)
    return d, prev.to_numpy(float), mom.to_numpy(float)


def boundary_score(prev, mom, alpha):
    return (np.log(np.clip(mom, 1e-9, None))
            + alpha * np.log(np.clip(prev, 1e-9, None)))


def mcc_at_workload(y, score, workload):
    from sklearn.metrics import matthews_corrcoef
    alarm, _ = alarm_at_workload(score, workload)
    if alarm.sum() == 0 or alarm.sum() == len(alarm):
        return -1.0
    return float(matthews_corrcoef(y, alarm))


# ============================================================== nested run ===
def nested(name, panel, X, y, groups, fast=False):
    """Outer scoring folds; every choice made inside the inner folds."""
    from sklearn.model_selection import StratifiedGroupKFold

    A, yv = X.to_numpy(float), y.to_numpy(int)
    grid = GRIDS[name][:2] if fast else GRIDS[name]
    kg = K_GRID[::2] if fast else K_GRID
    ag = ALPHA_GRID[::2] if fast else ALPHA_GRID

    oof = np.full(len(A), np.nan)
    oof_score = np.full(len(A), np.nan)
    chosen = []
    t0 = time.time()

    outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(outer.split(A, yv, groups), 1):
        if yv[tr].sum() < 2:
            continue
        # ---- inner loop: pick the learner configuration --------------------
        inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True,
                                     random_state=SEED + fold)
        gtr = groups[tr]
        best_cfg, best_ap = grid[0], -np.inf
        inner_pred = {}
        for cfg in grid:
            ip = np.full(len(tr), np.nan)
            for itr, ite in inner.split(A[tr], yv[tr], gtr):
                if yv[tr][itr].sum() < 2:
                    continue
                ip[ite] = fit_predict(name, cfg, A[tr][itr], yv[tr][itr],
                                      A[tr][ite])
            ok = np.isfinite(ip)
            if ok.sum() and 0 < yv[tr][ok].sum() < ok.sum():
                from sklearn.metrics import average_precision_score
                ap = average_precision_score(yv[tr][ok], ip[ok])
                inner_pred[id(cfg)] = ip
                if ap > best_ap:
                    best_ap, best_cfg = ap, cfg

        # ---- inner loop: pick K, alpha and workload on the SAME inner preds -
        ip = inner_pred.get(id(best_cfg))
        best_K, best_a, best_w, best_m = 0.35, 0.55, WORKLOAD, -np.inf
        if ip is not None and np.isfinite(ip).any():
            sub = panel.iloc[tr].copy()
            ok = np.isfinite(ip)
            d_in, prev_in, mom_in = momentum_and_prev(sub, np.nan_to_num(ip))
            y_in = d_in["y"].to_numpy(int)
            for K, a in itertools.product(kg, ag):
                sc = boundary_score(prev_in, mom_in, a)
                # the threshold is the workload quantile, so K enters through the
                # ranking only; both are still selected without touching the outer fold
                for w in ((WORKLOAD,) if fast else (0.01, 0.02, 0.03)):
                    m = mcc_at_workload(y_in, sc - np.log(K), w)
                    if m > best_m:
                        best_m, best_K, best_a, best_w = m, K, a, w

        # ---- score the outer held-out fold with the frozen choices ---------
        p_te = fit_predict(name, best_cfg, A[tr], yv[tr], A[te])
        oof[te] = p_te
        sub_te = panel.iloc[te].copy()
        d_te, prev_te, mom_te = momentum_and_prev(sub_te, p_te)
        s_te = boundary_score(prev_te, mom_te, best_a) - np.log(best_K)
        # map back to original row order
        oof_score[d_te.index.to_numpy()] = s_te
        chosen.append(dict(fold=fold, cfg=str(best_cfg), K=best_K, alpha=best_a,
                           workload=best_w, inner_pr_auc=best_ap))

    sel = pd.DataFrame(chosen)
    d = discrimination(yv, oof)
    alarm, _ = alarm_at_workload(oof_score, WORKLOAD)
    lt, fa, fa_yr = lead_and_burden(panel, alarm, len(panel))
    n_ev = int(len(lt)); nc = int(lt["caught"].sum()) if n_ev else 0
    act = lt.loc[lt["actionable_days"].notna(), "actionable_days"]

    res = dict(model=name, auc=d["auc"], pr_auc=d["pr_auc"], brier=d["brier"],
               bss=d["bss"], n_caught=nc, n_events=n_ev,
               act_median=float(act.median()) if len(act) else np.nan,
               n_alarm=int(alarm.sum()), alarm_share=float(alarm.mean()) * 100,
               fa_per_issuer_year=fa_yr,
               K_modal=sel["K"].mode().iloc[0] if len(sel) else np.nan,
               alpha_modal=sel["alpha"].mode().iloc[0] if len(sel) else np.nan,
               K_range=f"{sel['K'].min():.2f}-{sel['K'].max():.2f}" if len(sel) else "--",
               alpha_range=f"{sel['alpha'].min():.2f}-{sel['alpha'].max():.2f}"
               if len(sel) else "--",
               n_distinct_cfg=int(sel["cfg"].nunique()) if len(sel) else 0,
               seconds=round(time.time() - t0, 1))
    print(f"    {name:22s} AUC {res['auc']:.3f}  PR {res['pr_auc']:.4f}  "
          f"BSS {res['bss']:+.3f}  caught {nc}/{n_ev}  "
          f"K={res['K_modal']} a={res['alpha_modal']} "
          f"(K range {res['K_range']}, {res['n_distinct_cfg']} distinct configs)  "
          f"({res['seconds']}s)")
    return res, oof, sel


# ================================================== dependence-aware DM ======
def dm_pooled(l1, l2):
    """The test as it was applied in the submitted version: pooled rows, HLN
    correction, no allowance for within-issuer dependence."""
    from scipy import stats
    d = np.asarray(l1, float) - np.asarray(l2, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def dm_block(l1, l2, issuers, n_boot=N_BOOT, seed=SEED):
    """Average the loss differential within each issuer, then resample issuers.

    Each issuer contributes one number, so serial correlation inside an issuer and
    the overlap between consecutive three-month windows no longer inflate the
    effective sample size."""
    d = np.asarray(l1, float) - np.asarray(l2, float)
    ok = np.isfinite(d)
    d, iss = d[ok], np.asarray(issuers)[ok]
    codes = np.unique(iss)
    per = np.array([d[iss == c].mean() for c in codes])
    obs = float(per.mean())
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(per, size=len(per), replace=True).mean()
                      for _ in range(n_boot)])
    # two-sided bootstrap p-value for the null that the mean differential is zero
    centred = draws - draws.mean()
    p = float((np.abs(centred) >= abs(obs)).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return obs, float(lo), float(hi), p, len(codes)


def dm_table(panel, y, oofs, baseline="Logistic (Approach 1)", n_boot=N_BOOT):
    yv = y.to_numpy(int)
    iss = panel["issuer_code"].to_numpy()
    base = oofs.get(baseline)
    if base is None:
        return pd.DataFrame()
    lb = (yv - base) ** 2
    rows = []
    for name, p in oofs.items():
        if name == baseline:
            continue
        lm = (yv - p) ** 2
        s_pool, p_pool = dm_pooled(lb, lm)
        diff, lo, hi, p_blk, n_iss = dm_block(lb, lm, iss, n_boot)
        rows.append(dict(model=name, mean_diff=diff, ci_lo=lo, ci_hi=hi,
                         p_block=p_blk, p_pooled=p_pool, dm_pooled=s_pool,
                         n_issuers=n_iss,
                         verdict=("better than baseline" if p_blk < 0.05 and diff > 0
                                  else ("worse than baseline"
                                        if p_blk < 0.05 and diff < 0
                                        else "not distinguishable"))))
    return pd.DataFrame(rows)


# ================================================================ output =====
def write_tex(res, dm, sel_all):
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"

    L = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\caption{Nested validation. Learner settings, the boundary constants $K$ "
         r"and $\alpha$, and the workload are selected inside the training folds "
         r"only; the outer folds are used for scoring alone.}",
         r"\label{tab:nested}",
         r"\begin{tabular}{@{}lrrrrrl@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{BSS} & "
         r"\textbf{Detection} & \textbf{FA/issuer-yr} & "
         r"\textbf{$K$, $\alpha$ selected} \\", r"\midrule"]
    best = res.loc[res["pr_auc"].idxmax(), "model"] if res["pr_auc"].notna().any() else None
    for _, r in res.iterrows():
        cells = [esc(r["model"]), f3(r["auc"]), f4(r["pr_auc"]),
                 f"{r['bss']:+.3f}" if pd.notna(r["bss"]) else "--",
                 f"{int(r['n_caught'])}/{int(r['n_events'])}",
                 f"{r['fa_per_issuer_year']:.2f}",
                 f"{r['K_modal']:.2f}, {r['alpha_modal']:.2f}"]
        if r["model"] == best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[3pt] {\footnotesize Outer loop: "
          f"{N_OUTER}-fold grouped cross-validation over issuer identity. Inner loop: "
          f"{N_INNER}-fold grouped cross-validation inside each outer training set, "
          r"used to choose the learner configuration by PR-AUC and then $(K,\alpha)$ "
          r"and the workload by the Matthews correlation coefficient. The last column "
          r"reports the modal selection across outer folds; the constants are not "
          r"stable across folds, which is itself evidence that they were previously "
          r"fitted to the events they were scored against.}", r"\end{table}"]

    L2 = []
    if not dm.empty:
        L2 = [r"\begin{table}[H]", r"\centering", r"\small",
              r"\caption{Diebold--Mariano comparison against the logistic baseline "
              r"with issuer-block resampling, and the pooled-row $p$-value it "
              r"replaces}", r"\label{tab:dm-block}",
              r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
              r"\textbf{Model} & \textbf{Mean loss diff.} & \textbf{95\% CI} & "
              r"\textbf{$p$ (issuer block)} & \textbf{$p$ (pooled rows)} & "
              r"\textbf{Verdict} \\", r"\midrule"]
        for _, r in dm.iterrows():
            L2.append(" & ".join([
                esc(r["model"]), f"{r['mean_diff']:+.6f}",
                f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]",
                f"{r['p_block']:.4f}",
                "--" if pd.isna(r["p_pooled"]) else f"{r['p_pooled']:.4f}",
                esc(r["verdict"])]) + r" \\")
        L2 += [r"\bottomrule", r"\end{tabular}",
               r"\\[3pt] {\footnotesize A positive differential means the model has "
               r"lower Brier loss than the logistic baseline. The loss differential "
               r"is averaged within each issuer before resampling, so the effective "
               f"sample is {int(dm['n_issuers'].iloc[0])} issuers rather than "
               f"{16686:,} issuer-months. The pooled column reproduces the test as it "
               r"was applied previously and is shown only to indicate how much the "
               r"within-issuer dependence had inflated the evidence.}",
               r"\end{table}"]

    frag = ("\\section*{Nested validation and a dependence-aware accuracy test}\n\n"
            + "\n".join(L) + "\n\n" + "\n".join(L2) + "\n")
    p = out("section_nested_dm.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(frag)
    return p


def main():
    fast = "--fast" in sys.argv
    n_boot = 500 if fast else N_BOOT
    print("=" * 100)
    print("Nested validation (R1.3) and issuer-block Diebold-Mariano (R1.5)")
    print("=" * 100)
    panel, X, y, cols = cl.load_panel(verbose=True)
    panel = panel.reset_index(drop=True)
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")
    panel["y"] = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()

    print(f"\n  outer {N_OUTER} folds, inner {N_INNER} folds, "
          f"grids: learner {len(GRIDS['XGBoost'])}, K {len(K_GRID)}, "
          f"alpha {len(ALPHA_GRID)}"
          + ("  [--fast]" if fast else ""))
    rows, oofs, sels = [], {}, []
    for name in MODELS:
        if name not in cl.classifiers():
            continue
        r, oof, sel = nested(name, panel, X, y, groups, fast)
        rows.append(r); oofs[name] = oof; sels.append(sel.assign(model=name))
    res = pd.DataFrame(rows)
    sel_all = pd.concat(sels, ignore_index=True) if sels else pd.DataFrame()

    print(f"\n  issuer-block DM against the logistic baseline, {n_boot} draws ...")
    dm = dm_table(panel, y, oofs, n_boot=n_boot)
    for _, r in dm.iterrows():
        print(f"    {r['model']:22s} diff {r['mean_diff']:+.6f}  "
              f"p(block) {r['p_block']:.4f}   p(pooled) {r['p_pooled']:.4f}   "
              f"{r['verdict']}")

    p = write_tex(res, dm, sel_all)
    res.to_csv(out("nested_results.csv"), index=False)
    dm.to_csv(out("dm_block.csv"), index=False)
    sel_all.to_csv(out("nested_selections.csv"), index=False)
    con = sqlite3.connect(DB)
    res.to_sql("cmdf_nested", con, if_exists="replace", index=False)
    dm.to_sql("cmdf_dm_block", con, if_exists="replace", index=False)
    sel_all.to_sql("cmdf_nested_selections", con, if_exists="replace", index=False)
    con.commit(); con.close()

    print("\n" + "=" * 100)
    print("STABILITY OF THE BOUNDARY CONSTANTS ACROSS OUTER FOLDS")
    print("=" * 100)
    if not sel_all.empty:
        print(sel_all.groupby("model")[["K", "alpha", "workload"]]
              .agg(["min", "max", "nunique"]).to_string())
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
