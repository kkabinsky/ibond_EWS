# -*- coding: utf-8 -*-
"""
cmdf_approach2_compare.py -- the Approach-2 calibrated hazard pipeline run with four
different base learners, compared on identical data.

Approach 2 as implemented in run_survivor_ews_33features_xgb.py is a pipeline, not a
single model:

    calibrated classifier  ->  PD_3M
                           ->  Momentum = PD_3M(t) / median(PD_3M, previous 12m)
                           ->  hyperbolic flag: log M + 0.55 log PD_prev >= log 0.35
                           ->  alert bands
                           ->  lead time before the recorded default

Everything above the first arrow is what changes here. The base learner is swapped
between XGBoost, CatBoost, LightGBM and Random Forest; every other step, constant and
threshold is left exactly as in the original module so the comparison isolates the
learner.

WHY CALIBRATION MATTERS IN THIS PIPELINE
    The downstream steps use PD_3M as a NUMBER, not just as a ranking: Momentum is a
    ratio of consecutive PDs and the alert bands are absolute cut-offs at 0.05 and
    0.15. A learner that ranks well but is badly calibrated will therefore produce
    the wrong momentum and the wrong bands even when its AUC is high. Each base
    learner is wrapped in CalibratedClassifierCV for that reason.

VALIDATION
    Leave-one-issuer-out for the reported AUC, since every recorded default falls in
    the last two years. In-sample AUC is shown only to expose the overfitting gap.

RUN
    python cmdf_approach2_compare.py
    python cmdf_approach2_compare.py --no-save
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
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

# constants copied from run_survivor_ews_33features_xgb.py so the pipeline is identical
ALARM_PD = 0.05
BAND_WATCH, BAND_ELEV = 0.05, 0.15
HYPER_K, HYPER_ALPHA = 0.35, 0.55
MOM_WINDOW = 12
DAYS_PER_MONTH = 30.4375

LEARNERS = ["XGBoost", "CatBoost", "LightGBM", "Random Forest"]
LC = {"XGBoost": "#1f3a5f", "CatBoost": "#a8501a",
      "LightGBM": "#e0a52e", "Random Forest": "#2e7d4f"}

plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})


def base_learner(name, seed=42):
    """Same hyper-parameters as the original Approach-2 module for XGBoost, and
    matching depth / rate / tree budget for the others."""
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.03,
                                 eval_metric="logloss", random_state=seed,
                                 verbosity=0)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=60, max_depth=3, learning_rate=0.03,
                                  random_state=seed, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=60, depth=3, learning_rate=0.03,
                                  random_seed=seed, verbose=0,
                                  allow_writing_files=False)
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=60, max_depth=3, n_jobs=-1,
                                      random_state=seed)
    raise ValueError(name)


def calibrated(name, seed=42):
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(estimator=base_learner(name, seed),
                                  method="sigmoid", cv=3)


# ============================================================ the pipeline ===
def add_signals(panel):
    """Steps 2-4 of Approach 2, unchanged from the original module."""
    p = panel.sort_values(["issuer_code", "month_dt"]).reset_index(drop=True)
    p["PD_prev"] = p.groupby("issuer_code")["PD_3M"].shift(1).fillna(p["PD_3M"])
    p["PD_lag_med"] = (p.groupby("issuer_code")["PD_3M"]
                       .transform(lambda s: s.shift(1)
                                  .rolling(MOM_WINDOW, min_periods=1).median()))
    p["Momentum"] = (p["PD_3M"] / (p["PD_lag_med"] + 1e-4)).clip(0.0, 10.0)
    score = (np.log(np.clip(p["Momentum"], 1e-9, None))
             + HYPER_ALPHA * np.log(np.clip(p["PD_prev"], 1e-9, None)))
    p["flag_hyper"] = (score >= np.log(HYPER_K)).astype(int)
    p["alert_level"] = "OK"
    p.loc[p["PD_3M"] >= BAND_WATCH, "alert_level"] = "WATCH"
    p.loc[p["PD_3M"] >= BAND_ELEV, "alert_level"] = "ELEVATED"
    p.loc[(p["flag_hyper"] == 1)
          | ((p["PD_3M"] >= BAND_ELEV) & (p["Momentum"] >= 1.2)),
          "alert_level"] = "HIGH RISK"
    return p


def lead_times(p):
    """Two metrics, as stored by the current engines.

    actionable  first alarm inside the 1-3 calendar-month window before the event
    persistent  start of the final continuous monthly alarm episode
    """
    rows = []
    for code, g in p.groupby("issuer_code"):
        ev = g["event_date"].dropna()
        if ev.empty:
            continue
        ev_date = pd.to_datetime(ev.iloc[0])
        pre = g[g["month_dt"] < ev_date].sort_values("month_dt")
        if pre.empty:
            continue
        is_alarm = ((pre["PD_3M"] >= ALARM_PD)
                    | (pre["flag_hyper"] == 1)).to_numpy()
        days = (ev_date - pre["month_dt"]).dt.days.to_numpy()

        act = np.nan
        in_win = is_alarm & (days >= 30) & (days <= 92)
        if in_win.any():
            act = float(days[in_win].max())      # earliest inside the window

        pers = np.nan
        if is_alarm.any():
            pos = int(np.max(np.flatnonzero(is_alarm)))
            while pos > 0 and is_alarm[pos - 1]:
                pos -= 1
            pers = float(days[pos])
        rows.append(dict(issuer_code=code, default_date=ev_date.date(),
                         actionable_days=act, persistent_days=pers,
                         caught=int(np.isfinite(act) or np.isfinite(pers))))
    return pd.DataFrame(rows)


def run_one(name, panel_raw, X, y, verbose=True):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    A, yv = X.to_numpy(float), y.to_numpy(int)
    groups = panel_raw["issuer_code"].to_numpy()
    ev = sorted(panel_raw.loc[y == 1, "issuer_code"].dropna().unique())
    t0 = time.time()

    # in-sample fit, used for the signals and the alert bands
    sc = StandardScaler().fit(A)
    m = calibrated(name)
    m.fit(sc.transform(A), yv)
    p_in = m.predict_proba(sc.transform(A))[:, 1]
    auc_in = float(roc_auc_score(yv, p_in)) if yv.sum() else np.nan

    # leave-one-issuer-out, the honest number
    oy, op = [], []
    for held in ev:
        tr = groups != held
        if yv[tr].sum() < 2:
            continue
        sci = StandardScaler().fit(A[tr])
        mi = calibrated(name)
        try:
            mi.fit(sci.transform(A[tr]), yv[tr])
        except Exception:
            continue
        oy.append(yv[~tr])
        op.append(mi.predict_proba(sci.transform(A[~tr]))[:, 1])
    auc_oos = np.nan
    brier = np.nan
    if oy:
        yy, pp = np.concatenate(oy), np.concatenate(op)
        if 0 < yy.sum() < len(yy):
            auc_oos = float(roc_auc_score(yy, pp))
            brier = float(np.mean((yy - pp) ** 2))

    panel = panel_raw.copy()
    panel["PD_3M"] = p_in
    panel = add_signals(panel)
    lt = lead_times(panel)

    n = len(panel)
    vc = panel["alert_level"].value_counts()
    got_a = lt["actionable_days"].notna() if not lt.empty else pd.Series(dtype=bool)
    got_p = lt["persistent_days"].notna() if not lt.empty else pd.Series(dtype=bool)
    res = dict(
        learner=name, auc_in=auc_in, auc_oos=auc_oos, brier_oos=brier,
        n_events=int(len(lt)),
        n_caught=int(got_p.sum()) if not lt.empty else 0,
        actionable_median=float(lt.loc[got_a, "actionable_days"].median())
        if got_a.any() else np.nan,
        persistent_median=float(lt.loc[got_p, "persistent_days"].median())
        if got_p.any() else np.nan,
        persistent_months=(float(lt.loc[got_p, "persistent_days"].median())
                           / DAYS_PER_MONTH) if got_p.any() else np.nan,
        n_high=int(vc.get("HIGH RISK", 0)),
        pct_high=float(vc.get("HIGH RISK", 0)) / n * 100 if n else np.nan,
        pd_median=float(panel["PD_3M"].median()),
        pd_p99=float(panel["PD_3M"].quantile(0.99)),
        seconds=round(time.time() - t0, 1))
    if verbose:
        print(f"    {name:15s} AUC oos {auc_oos:.3f} (in {auc_in:.3f})  "
              f"caught {res['n_caught']}/{res['n_events']}  "
              f"persistent {res['persistent_median']:.0f}d  "
              f"HIGH {res['pct_high']:.2f}%  ({res['seconds']}s)")
    lt = lt.assign(learner=name)
    return res, lt, panel[["issuer_code", "month", "PD_3M", "Momentum",
                           "flag_hyper", "alert_level"]].assign(learner=name)


# ================================================================ outputs ====
def write_outputs(res, lt_all):
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    f0 = lambda v: "--" if pd.isna(v) else f"{v:.0f}"

    best = res.loc[res["auc_oos"].idxmax(), "learner"]
    tm.write_tex_table(
        res, out("tab_a2_compare.tex"),
        "Approach 2 (calibrated hazard pipeline) รันด้วย base learner สี่แบบ "
        "บนข้อมูล iBond ชุดเดียวกัน", "tab:a2-cmp",
        cols=["learner", "auc_in", "auc_oos", "brier_oos", "n_caught", "n_events",
              "persistent_median", "pct_high"],
        fmt={"auc_in": f3, "auc_oos": f3, "brier_oos": lambda v: f"{v:.5f}",
             "persistent_median": f0,
             "pct_high": lambda v: "--" if pd.isna(v) else f"{v:.2f}\\%"},
        bold_row=lambda r, b=best: r["learner"] == b,
        note=("ทุกขั้นตอนหลังการประมาณค่า PD เหมือนกันทั้งหมด ได้แก่ Momentum "
              "เส้นแบ่งไฮเพอร์โบลา เกณฑ์ระดับสัญญาณ และวิธีคำนวณเวลานำ "
              "ความต่างที่เห็นจึงมาจาก base learner เท่านั้น "
              "ค่า Brier loss ยิ่งต่ำยิ่งดี และเป็นค่าที่สะท้อนคุณภาพของ"
              "ความน่าจะเป็นโดยตรง ต่างจาก AUC ที่วัดเฉพาะการจัดลำดับ"))
    res.to_csv(out("a2_compare.csv"), index=False)

    if not lt_all.empty:
        piv = lt_all.pivot_table(index="issuer_code", columns="learner",
                                 values="persistent_days")
        piv = piv.reindex([c for c in piv.index], columns=[c for c in LEARNERS
                                                           if c in piv.columns])
        tab = piv.reset_index()
        tm.write_tex_table(
            tab, out("tab_a2_leadtime.tex"),
            "จุดเริ่มของช่วงสัญญาณต่อเนื่อง (วัน) รายผู้ออกตราสาร จำแนกตาม base learner",
            "tab:a2-lead", cols=list(tab.columns),
            fmt={c: f0 for c in tab.columns if c != "issuer_code"},
            note=("ตัวเลขคือจำนวนวันก่อนวันผิดนัดชำระที่แบบจำลองเริ่มส่งสัญญาณ"
                  "ต่อเนื่องและไม่หยุด ค่าที่น้อยกว่าหมายถึงเตือนใกล้เหตุการณ์กว่า "
                  "ซึ่งลดช่วงเวลาที่สัญญาณค้างอยู่โดยไม่มีเหตุการณ์เกิดขึ้น"))
        lt_all.to_csv(out("a2_leadtime_detail.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    ax = axes[0]
    xs = np.arange(len(res))
    w = 0.38
    ax.bar(xs - w / 2, res["auc_in"], width=w, color="#cbd5e1",
           label="in-sample", edgecolor="white")
    ax.bar(xs + w / 2, res["auc_oos"], width=w,
           color=[LC.get(l, "#888") for l in res["learner"]],
           label="out-of-sample", edgecolor="white")
    for x, v in zip(xs, res["auc_oos"]):
        if pd.notna(v):
            ax.text(x + w / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(res["learner"], fontsize=8, rotation=14)
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Discrimination", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(xs, res["persistent_median"],
           color=[LC.get(l, "#888") for l in res["learner"]], alpha=0.92,
           edgecolor="white")
    for x, r in zip(xs, res.itertuples()):
        if pd.notna(r.persistent_median):
            ax.text(x, r.persistent_median + 6,
                    f"{r.persistent_median:.0f} d\n({r.n_caught}/{r.n_events})",
                    ha="center", fontsize=7.5)
    ax.axhline(91, color="#dc2626", lw=1.2, ls="--")
    ax.text(len(res) - 0.5, 98, "3-month window", fontsize=7, color="#dc2626",
            ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(res["learner"], fontsize=8, rotation=14)
    ax.set_ylabel("median persistent alarm start (days)")
    ax.set_title("Lead time", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(xs, res["pct_high"], color=[LC.get(l, "#888") for l in res["learner"]],
           alpha=0.92, edgecolor="white")
    for x, r in zip(xs, res.itertuples()):
        ax.text(x, r.pct_high + 0.05, f"{r.pct_high:.2f}%\n({r.n_high:,})",
                ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(res["learner"], fontsize=8, rotation=14)
    ax.set_ylabel("issuer-months flagged HIGH RISK (%)")
    ax.set_title("Alarm load", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Approach 2 pipeline with four base learners", fontsize=11.5,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    tm.save_fig(fig, "fig_a2_compare.png")


def run(save=True, verbose=True):
    print("=" * 78)
    print("Approach 2 calibrated hazard pipeline: four base learners")
    print("=" * 78)
    panel, X, y, cols = cl.load_panel(verbose=verbose)
    if y.sum() < 5:
        raise RuntimeError("too few positive months")
    if "event_date" not in panel.columns:
        raise RuntimeError("panel has no event_date column")
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")

    rows, lts, alerts = [], [], []
    for name in LEARNERS:
        try:
            base_learner(name)
        except ImportError:
            print(f"    {name}: library not installed — skipped")
            continue
        r, lt, al = run_one(name, panel, X, y, verbose)
        rows.append(r)
        lts.append(lt)
        alerts.append(al)

    res = pd.DataFrame(rows)
    lt_all = pd.concat(lts, ignore_index=True) if lts else pd.DataFrame()
    write_outputs(res, lt_all)
    if save:
        con = sqlite3.connect(DB)
        res.to_sql("cmdf_a2_compare", con, if_exists="replace", index=False)
        if not lt_all.empty:
            lt_all.to_sql("cmdf_a2_leadtime", con, if_exists="replace", index=False)
        if alerts:
            pd.concat(alerts, ignore_index=True).to_sql(
                "cmdf_a2_alerts", con, if_exists="replace", index=False)
        con.commit(); con.close()
    return res, lt_all


def main():
    res, lt_all = run(save="--no-save" not in sys.argv)
    print("\n" + "=" * 92)
    print("APPROACH 2 WITH FOUR BASE LEARNERS")
    print("=" * 92)
    print(res[["learner", "auc_in", "auc_oos", "brier_oos", "n_caught", "n_events",
               "persistent_median", "pct_high"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not res.empty:
        b = res.loc[res["auc_oos"].idxmax()]
        print(f"\nbest discrimination: {b['learner']} (AUC {b['auc_oos']:.3f})")
        bb = res.loc[res["brier_oos"].idxmin()]
        print(f"best calibration:    {bb['learner']} (Brier {bb['brier_oos']:.5f})")
        # A low alarm load is only a virtue if the model also ranks well. With eight
        # events a learner can catch all of them on a narrow alarm while ranking the
        # rest of the panel poorly, so require usable discrimination first.
        USABLE = 0.75
        full = res[(res["n_caught"] == res["n_events"]) & (res["auc_oos"] >= USABLE)]
        if not full.empty:
            t = full.loc[full["pct_high"].idxmin()]
            print(f"caught all events, usable AUC, lowest alarm load: {t['learner']} "
                  f"({t['pct_high']:.2f}% flagged, AUC {t['auc_oos']:.3f})")
        weak = res[res["auc_oos"] < USABLE]
        for _, r in weak.iterrows():
            print(f"  NOTE: {r['learner']} caught {r['n_caught']}/{r['n_events']} on a "
                  f"{r['pct_high']:.2f}% alarm but its out-of-sample AUC is only "
                  f"{r['auc_oos']:.3f}, so that result should not be read as skill")
    print("\nArtefacts: tex_out/tab_a2_compare.tex, tab_a2_leadtime.tex, "
          "fig_a2_compare.png")
    print("Done.")


if __name__ == "__main__":
    main()
