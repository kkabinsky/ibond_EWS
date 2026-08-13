# -*- coding: utf-8 -*-
"""
compare_models.py  --  Head-to-head comparison of the two Approach-1 hazard engines.

    survivor2.py       -> Logistic (discrete-time logistic hazard)
    machine_survior.py -> XGBoost  (gradient-boosted-trees hazard) + SHAP

Runs BOTH on the same panel / features / pipeline, then produces a metric-by-metric
table showing WHO OUTPERFORMS BY WHAT %, a per-firm lead-time comparison, and the
statistical verdict. Everything is persisted to SQLite so the Flet GUI can render it
without re-running the models.

SQLite schema
-------------
  model_compare_metrics   metric | direction | logistic | xgboost | diff | pct_outperform | winner
  model_compare_leadtime  firm_id | default_date | lead_lr | lead_xgb | diff_days | winner
  model_compare_summary   single row: wins, AUC delta, bootstrap CI/p, McNemar, verdict, run_at

Run:  python compare_models.py            (run + print + save to SQLite)
      python compare_models.py --no-save  (print only)
"""
from __future__ import annotations
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import lead_metrics
import survival
import survivor2
import machine_survior as ms
from survivor2 import DAYS_PER_MONTH, P_STAR_DEFAULT as P_STAR
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")

T_METRICS = "model_compare_metrics"
T_LEAD = "model_compare_leadtime"
T_SUMMARY = "model_compare_summary"

# metric direction: +1 = higher is better, -1 = lower is better, 0 = informational
METRIC_DIRECTION = {
    "PD_3M AUC (out-of-sample)": +1,
    "PD_3M AUC (in-sample)": 0,
    "Overfit gap (in - out)": -1,
    "MCC (out-of-sample)": +1,
    "Precision (out-of-sample)": +1,
    "Recall (out-of-sample)": +1,
    "F1 (out-of-sample)": +1,
    "Flagged volume % (out-of-sample)": -1,
    "Detection rate % (before default)": +1,
    "Median actionable lead (1-3M, months)": 0,
    "Mean actionable lead (1-3M, months)": 0,
    "Median persistent alarm duration (months)": 0,
}


def _pct_outperform(better: float, worse: float) -> float:
    """relative advantage of the winner over the loser, in %."""
    if worse is None or np.isnan(worse) or abs(worse) < 1e-12:
        return float("inf") if (better or 0) > 0 else 0.0
    return abs(better - worse) / abs(worse) * 100.0


def build_metric_table(R_lr: dict, R_xgb: dict, n_events: int) -> pd.DataFrame:
    el, ex = R_lr["eval"], R_xgb["eval"]
    det_lr = int((R_lr["lt"]["status"] == "detected").sum())
    det_xgb = int((R_xgb["lt"]["status"] == "detected").sum())
    L_lr, L_xgb = R_lr["leads"], R_xgb["leads"]

    def med(a):
        return float(np.median(a) / DAYS_PER_MONTH) if len(a) else float("nan")

    def mean(a):
        return float(np.mean(a) / DAYS_PER_MONTH) if len(a) else float("nan")

    def persistent_med(table):
        values = pd.to_numeric(
            table["persistent_alarm_days"], errors="coerce"
        ).dropna()
        return float(values.median() / DAYS_PER_MONTH) if len(values) else float("nan")

    rows = [
        ("PD_3M AUC (out-of-sample)", R_lr["auc_oos"], R_xgb["auc_oos"]),
        ("PD_3M AUC (in-sample)", R_lr["meta"]["pd_auc"], R_xgb["meta"]["pd_auc"]),
        ("Overfit gap (in - out)",
         R_lr["meta"]["pd_auc"] - R_lr["auc_oos"], R_xgb["meta"]["pd_auc"] - R_xgb["auc_oos"]),
        ("MCC (out-of-sample)", el["MCC"], ex["MCC"]),
        ("Precision (out-of-sample)", el["precision"], ex["precision"]),
        ("Recall (out-of-sample)", el["recall"], ex["recall"]),
        ("F1 (out-of-sample)", R_lr["f1"], R_xgb["f1"]),
        ("Flagged volume % (out-of-sample)", el["volume"] * 100, ex["volume"] * 100),
        ("Detection rate % (before default)",
         det_lr / max(n_events, 1) * 100, det_xgb / max(n_events, 1) * 100),
        ("Median actionable lead (1-3M, months)", med(L_lr), med(L_xgb)),
        ("Mean actionable lead (1-3M, months)", mean(L_lr), mean(L_xgb)),
        ("Median persistent alarm duration (months)",
         persistent_med(R_lr["lt"]), persistent_med(R_xgb["lt"])),
    ]

    recs = []
    for name, v_lr, v_xgb in rows:
        d = METRIC_DIRECTION.get(name, 0)
        diff = (v_xgb - v_lr) if (v_lr == v_lr and v_xgb == v_xgb) else float("nan")
        if d == 0 or not (v_lr == v_lr and v_xgb == v_xgb) or abs(diff) < 1e-12:
            winner, pct = ("-", 0.0) if d == 0 or abs(diff) < 1e-12 else ("-", 0.0)
        else:
            xgb_better = (diff > 0) if d > 0 else (diff < 0)
            winner = "XGBoost" if xgb_better else "Logistic"
            better, worse = (v_xgb, v_lr) if xgb_better else (v_lr, v_xgb)
            pct = _pct_outperform(better, worse)
        recs.append(dict(metric=name,
                         direction={1: "higher better", -1: "lower better", 0: "info"}[d],
                         logistic=float(v_lr) if v_lr == v_lr else None,
                         xgboost=float(v_xgb) if v_xgb == v_xgb else None,
                         diff=float(diff) if diff == diff else None,
                         pct_outperform=float(pct) if np.isfinite(pct) else None,
                         winner=winner))
    return pd.DataFrame(recs)


def build_leadtime_table(R_lr: dict, R_xgb: dict) -> pd.DataFrame:
    a = R_lr["lt"][[
        "firm_id", "default_date", "lead_days",
        "persistent_alarm_days", "status",
    ]].rename(columns={
        "lead_days": "lead_lr",
        "persistent_alarm_days": "persistent_lr",
        "status": "status_lr",
    })
    b = R_xgb["lt"][[
        "firm_id", "lead_days", "persistent_alarm_days", "status",
    ]].rename(columns={
        "lead_days": "lead_xgb",
        "persistent_alarm_days": "persistent_xgb",
        "status": "status_xgb",
    })
    m = a.merge(b, on="firm_id")
    m = m[(m["status_lr"] != "censored") | (m["status_xgb"] != "censored")].copy()

    def winner(r):
        lr, xg = r["lead_lr"], r["lead_xgb"]
        if pd.isna(lr) and pd.isna(xg):
            return "both missed"
        if pd.isna(lr):
            return "XGBoost (LR missed)"
        if pd.isna(xg):
            return "Logistic (XGB missed)"
        return "-"                       # both detected: lead length is contextual
    m["diff_days"] = m["lead_xgb"] - m["lead_lr"]
    m["winner"] = m.apply(winner, axis=1)
    m["default_date"] = m["default_date"].astype(str)
    return m.sort_values("default_date").reset_index(drop=True)


def run_comparison(verbose: bool = True) -> dict:
    if verbose:
        print("[1/4] loading panel ...")
    panel = survivor2.load_bond_dated()
    n_events = int(panel["event"].sum())
    clean = panel.drop(columns=survivor2.HAZARD_DROP, errors="ignore")

    if verbose:
        print(f"      {len(panel):,} firm-months | {panel['firm_id'].nunique()} firms | "
              f"events {n_events}")
        print("[2/4] fitting LOGISTIC hazard ...")
    df_lr, meta_lr, te_lr = ms.run_with(clean, ms.fit_hazard_logistic)
    e_lr = meta_lr["oos_pd"] or survival.evaluate(df_lr, "flag_PD")
    lt_lr = survivor2.lead_time_table(df_lr, P_STAR)
    R_lr = dict(label="Logistic", df=df_lr, meta=meta_lr, te=te_lr, lt=lt_lr, eval=e_lr,
                f1=ms._f1(e_lr["precision"], e_lr["recall"]), auc_oos=meta_lr["pd_auc_oos"],
                leads=lt_lr.loc[lt_lr["status"] == "detected", "lead_days"].astype(float).values)

    if verbose:
        print("[3/4] fitting XGBOOST hazard ...")
    df_x, meta_x, te_x = ms.run_with(clean, ms.fit_hazard_xgb)
    e_x = meta_x["oos_pd"] or survival.evaluate(df_x, "flag_PD")
    lt_x = survivor2.lead_time_table(df_x, P_STAR)
    R_xgb = dict(label="XGBoost", df=df_x, meta=meta_x, te=te_x, lt=lt_x, eval=e_x,
                 f1=ms._f1(e_x["precision"], e_x["recall"]), auc_oos=meta_x["pd_auc_oos"],
                 leads=lt_x.loc[lt_x["status"] == "detected", "lead_days"].astype(float).values)

    if verbose:
        print("[4/4] statistical tests ...")
    metrics = build_metric_table(R_lr, R_xgb, n_events)
    leadtime = build_leadtime_table(R_lr, R_xgb)

    # ---- statistical tests on the common OOS rows --------------------------
    auc_delta = lo = hi = pval = float("nan")
    b = c = 0
    pmc = float("nan")
    n_paired = n_pos = 0
    if te_lr is not None and te_x is not None:
        key = ["firm_id", "month_index"]
        m = (te_lr[key + ["y_fwd", "PD_3M", "flag_PD"]]
             .rename(columns={"PD_3M": "p_lr", "flag_PD": "f_lr"})
             .merge(te_x[key + ["PD_3M", "flag_PD"]]
                    .rename(columns={"PD_3M": "p_xgb", "flag_PD": "f_xgb"}), on=key))
        y = m["y_fwd"].astype(int).values
        n_paired, n_pos = len(m), int(y.sum())
        if n_paired and len(np.unique(y)) > 1:
            auc_delta, lo, hi, pval = ms.bootstrap_auc_diff(y, m["p_lr"].values, m["p_xgb"].values)
            b, c, pmc = ms.mcnemar_test(y, m["f_lr"].values, m["f_xgb"].values)

    scored = metrics[metrics["winner"].isin(["Logistic", "XGBoost"])]
    wins_lr = int((scored["winner"] == "Logistic").sum())
    wins_xgb = int((scored["winner"] == "XGBoost").sum())
    if pval == pval and pval < 0.05:
        stat_txt = (f"{'XGBoost' if auc_delta > 0 else 'Logistic'} has a statistically "
                    f"significant higher OOS AUC (p={pval:.3f}).")
    else:
        stat_txt = (f"No statistically significant AUC difference (p={pval:.3f}); "
                    f"the {abs(auc_delta):.3f} gap is within noise.")
    overall = ("Logistic" if wins_lr > wins_xgb else
               "XGBoost" if wins_xgb > wins_lr else "Tie")
    verdict = (f"{overall} wins {max(wins_lr, wins_xgb)}/{len(scored)} scored metrics. "
               f"{stat_txt} Recommendation: use Logistic as the deliverable baseline "
               f"(simpler, fewer false alarms, interpretable); keep XGBoost + SHAP when "
               f"maximum recall matters.")

    summary = dict(**lead_metrics.summary_metadata(),
                   n_firm_months=int(len(panel)), n_firms=int(panel["firm_id"].nunique()),
                   n_events=n_events, wins_logistic=wins_lr, wins_xgboost=wins_xgb,
                   n_scored_metrics=int(len(scored)),
                   auc_lr=float(R_lr["auc_oos"]), auc_xgb=float(R_xgb["auc_oos"]),
                   auc_delta=float(auc_delta) if auc_delta == auc_delta else None,
                   ci_low=float(lo) if lo == lo else None,
                   ci_high=float(hi) if hi == hi else None,
                   p_bootstrap=float(pval) if pval == pval else None,
                   mcnemar_lr_only=int(b), mcnemar_xgb_only=int(c),
                   p_mcnemar=float(pmc) if pmc == pmc else None,
                   n_paired_rows=int(n_paired), n_positives=int(n_pos),
                   median_persistent_alarm_days_logistic=float(
                       pd.to_numeric(lt_lr["persistent_alarm_days"], errors="coerce").median()
                   ),
                   median_persistent_alarm_days_xgboost=float(
                       pd.to_numeric(lt_x["persistent_alarm_days"], errors="coerce").median()
                   ),
                   overall_winner=overall, verdict=verdict)
    return dict(metrics=metrics, leadtime=leadtime, summary=summary,
                shap_top=ms.shap_top(meta_x["model"], clean)[0],
                odds_top=ms.odds_ratios_top(meta_lr["model"])[0])


# ------------------------------------------------------------- persistence ----
def save_to_sqlite(res: dict, db=DB) -> None:
    con = sqlite3.connect(db)
    res["metrics"].to_sql(T_METRICS, con, if_exists="replace", index=False)
    res["leadtime"].to_sql(T_LEAD, con, if_exists="replace", index=False)
    pd.DataFrame([res["summary"]]).to_sql(T_SUMMARY, con, if_exists="replace", index=False)
    con.commit()
    con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    try:
        metrics = pd.read_sql_query(f"SELECT * FROM {T_METRICS}", con)
        leadtime = pd.read_sql_query(f"SELECT * FROM {T_LEAD}", con)
        summary = pd.read_sql_query(f"SELECT * FROM {T_SUMMARY} LIMIT 1", con)
    except Exception:
        metrics = leadtime = summary = pd.DataFrame()
    finally:
        con.close()
    return metrics, leadtime, summary


# ------------------------------------------------------------------ print ----
def print_report(res: dict) -> None:
    m, lt, s = res["metrics"], res["leadtime"], res["summary"]
    print("\n" + "=" * 88)
    print("TABLE 1 -- METRIC-BY-METRIC COMPARISON   (Logistic = survivor2, XGBoost = machine_survior)")
    print("=" * 88)
    print(f"  {'Metric':34s} {'Logistic':>10s} {'XGBoost':>10s} {'Winner':>10s} {'Outperform':>12s}")
    print("  " + "-" * 82)
    for _, r in m.iterrows():
        lrv = "-" if r["logistic"] is None else f"{r['logistic']:.3f}"
        xgv = "-" if r["xgboost"] is None else f"{r['xgboost']:.3f}"
        pct = "" if (r["winner"] == "-" or r["pct_outperform"] is None) else f"+{r['pct_outperform']:.1f}%"
        print(f"  {r['metric']:34s} {lrv:>10s} {xgv:>10s} {r['winner']:>10s} {pct:>12s}")
    print("  " + "-" * 82)
    print(f"  scored metrics: Logistic wins {s['wins_logistic']}, XGBoost wins {s['wins_xgboost']}"
          f"  (of {s['n_scored_metrics']})")

    print("\n" + "=" * 88)
    print("TABLE 2 -- PER-FIRM ACTIONABLE 1-3M LEAD TIME (days)")
    print("=" * 88)
    print(f"  {'Firm':>6s} {'Default':>12s} {'Logistic':>9s} {'XGBoost':>9s} {'Diff':>8s}  Note")
    print("  " + "-" * 66)
    for _, r in lt.iterrows():
        lr = "MISS" if pd.isna(r["lead_lr"]) else f"{int(r['lead_lr'])}"
        xg = "MISS" if pd.isna(r["lead_xgb"]) else f"{int(r['lead_xgb'])}"
        df_ = "" if pd.isna(r["diff_days"]) else f"{int(r['diff_days']):+d}"
        note = "" if r["winner"] == "-" else r["winner"]
        print(f"  {int(r['firm_id']):>6d} {str(r['default_date']):>12s} {lr:>9s} {xg:>9s} {df_:>8s}  {note}")

    print("\n" + "=" * 88)
    print("TABLE 3 -- EXPLAINABILITY (top drivers)")
    print("=" * 88)
    print(f"  {'Logistic (odds ratio)':42s} {'XGBoost (mean |SHAP|)':42s}")
    print("  " + "-" * 84)
    for (k1, v1), (k2, v2) in zip(res["odds_top"], res["shap_top"]):
        print(f"  {k1:28s} {v1:11.3f}   {k2:28s} {v2:9.4f}")

    print("\n" + "=" * 88)
    print("TABLE 4 -- STATISTICAL TEST & VERDICT")
    print("=" * 88)
    print(f"  paired OOS rows {s['n_paired_rows']:,}  positives {s['n_positives']}")
    if s["auc_delta"] is not None:
        print(f"  Bootstrap AUC(XGB) - AUC(LR) = {s['auc_delta']:+.3f}"
              f"   95% CI [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]   p = {s['p_bootstrap']:.3f}")
        print(f"  McNemar: LR-only-correct {s['mcnemar_lr_only']}, "
              f"XGB-only-correct {s['mcnemar_xgb_only']}, p = {s['p_mcnemar']:.3f}")
    print(f"\n  OVERALL WINNER: {s['overall_winner']}")
    print(f"  {s['verdict']}")


def main():
    res = run_comparison(verbose=True)
    print_report(res)
    if "--no-save" not in sys.argv:
        save_to_sqlite(res)
        print(f"\nSaved to SQLite: {T_METRICS}, {T_LEAD}, {T_SUMMARY}  ({DB})")
    print("\nDone.")


if __name__ == "__main__":
    main()
