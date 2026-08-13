# -*- coding: utf-8 -*-
"""
make_table10.py -- Table 10 filled in for every model, not just the two original rows.

WHAT THE ORIGINAL TABLE 10 HELD
    Two rows only, Approach 1 logistic and Approach 2 calibrated XGBoost, taken from
    ibond_model_compare_33features. Random Forest, CatBoost and LightGBM had never
    been scored on the same six columns, which is what this module adds.

WHY THE NUMBERS ARE RECOMPUTED RATHER THAN COPIED
    AUC, F1, precision and recall in the stored two-row table come from a validation
    path whose "out-of-sample" AUC sits 0.0007 to 0.0107 from its own in-sample value.
    Filling three more rows from a different path would produce a table whose rows are
    not comparable to each other. Every row here is therefore produced by one
    identical procedure:

        StratifiedGroupKFold(5) over issuer_code
            every issuer-month is predicted by a model that never saw that issuer,
            so the panel gets a complete out-of-fold PD vector and precision has real
            true negatives to work with. Leave-one-issuer-out cannot do this: it only
            holds out the eight defaulted issuers, so the held-out rows contain no
            negatives from anywhere else in the panel.

        the out-of-fold PD then runs through the unchanged Approach-2 pipeline
            Momentum -> hyperbolic boundary -> alert bands -> lead time

        alarm = PD_3M >= 0.05 or the hyperbolic flag
            the same rule the lead-time function uses, so the F1 column and the
            lead-time column describe the same decision rather than two different ones

    The stored two-row figures are still printed underneath so the difference between
    the two validation paths stays visible.

RUN
    python make_table10.py
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

import cmdf_approach2_compare as a2
import cmdf_tree_classify as cl
import cmdf_tree_models as tm

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out
DAYS_PER_MONTH = a2.DAYS_PER_MONTH
ALARM_PD = a2.ALARM_PD

MODELS = ["XGBoost", "CatBoost", "LightGBM", "Random Forest", "Logistic"]
N_SPLITS = 5
SEED = 42
BUDGET = 0.02          # share of issuer-months every model is allowed to flag


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def estimator(name, seed=SEED):
    """Approach-2 wrapping for every learner, logistic included, so the calibration
    step is not something only the tree models receive."""
    from sklearn.calibration import CalibratedClassifierCV
    if name == "Logistic":
        from sklearn.linear_model import LogisticRegression
        base = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    else:
        base = a2.base_learner(name, seed)
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)


def out_of_fold(name, X, y, groups, n_splits=N_SPLITS):
    """One out-of-fold probability per row, from a model that never saw the issuer."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    A, yv = X.to_numpy(float), y.to_numpy(int)
    oof = np.full(len(A), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in cv.split(A, yv, groups):
        if yv[tr].sum() < 2:
            continue
        sc = StandardScaler().fit(A[tr])
        m = estimator(name)
        try:
            m.fit(sc.transform(A[tr]), yv[tr])
        except Exception:
            continue
        oof[te] = m.predict_proba(sc.transform(A[te]))[:, 1]
    return oof


def score_alarm(y_true, alarm):
    """Precision, recall and F1 of the alarm rule itself, not of an arbitrary 0.5 cut
    on a probability. This is the decision an analyst actually acts on."""
    y = np.asarray(y_true, int)
    a = np.asarray(alarm, int)
    tp = int(((a == 1) & (y == 1)).sum())
    fp = int(((a == 1) & (y == 0)).sum())
    fn = int(((a == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = (2 * prec * rec / (prec + rec)
          if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else np.nan)
    return prec, rec, f1, tp, fp, fn


def pipeline_metrics(panel_raw, pd_vec, y_true, alarm):
    """Lead time and detection under whichever alarm vector is passed in, so the fixed
    rule and the matched budget are measured the same way."""
    p = panel_raw.copy()
    p["PD_3M"] = pd_vec
    p = a2.add_signals(p)
    # override the rule the lead-time function reads, so both variants share one path
    p["flag_hyper"] = np.asarray(alarm, int)
    p["PD_3M"] = np.where(np.asarray(alarm, int) == 1,
                          np.maximum(p["PD_3M"], ALARM_PD), p["PD_3M"])
    lt = a2.lead_times(p)
    got_a = lt["actionable_days"].notna() if not lt.empty else pd.Series(dtype=bool)
    got_p = lt["persistent_days"].notna() if not lt.empty else pd.Series(dtype=bool)
    return dict(
        n_events=int(len(lt)),
        n_caught=int((got_a | got_p).sum()) if not lt.empty else 0,
        act_med=float(lt.loc[got_a, "actionable_days"].median())
        if got_a.any() else np.nan,
        pers_med=float(lt.loc[got_p, "persistent_days"].median())
        if got_p.any() else np.nan)


def run_model(name, panel_raw, X, y, verbose=True):
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    groups = panel_raw["issuer_code"].to_numpy()
    oof = out_of_fold(name, X, y, groups)

    ok = np.isfinite(oof)
    yv = y.to_numpy(int)
    auc_oos = (float(roc_auc_score(yv[ok], oof[ok]))
               if ok.sum() and 0 < yv[ok].sum() < ok.sum() else np.nan)
    brier = float(np.mean((yv[ok] - oof[ok]) ** 2)) if ok.sum() else np.nan

    # the pipeline, unchanged, driven by the out-of-fold PD
    panel = panel_raw.copy()
    panel["PD_3M"] = np.where(ok, oof, np.nanmedian(oof))
    panel = a2.add_signals(panel)
    lt = a2.lead_times(panel)

    alarm = ((panel["PD_3M"] >= ALARM_PD) | (panel["flag_hyper"] == 1)).astype(int)
    prec, rec, f1, tp, fp, fn = score_alarm(yv, alarm)

    got_a = lt["actionable_days"].notna() if not lt.empty else pd.Series(dtype=bool)
    got_p = lt["persistent_days"].notna() if not lt.empty else pd.Series(dtype=bool)
    n_ev = int(len(lt))
    n_caught = int((got_a | got_p).sum()) if not lt.empty else 0
    act_med = float(lt.loc[got_a, "actionable_days"].median()) if got_a.any() else np.nan
    pers_med = (float(lt.loc[got_p, "persistent_days"].median())
                if got_p.any() else np.nan)

    # ---- matched alarm budget -------------------------------------------------
    # The fixed 0.05 cut-off was set against in-sample PD values. Out of fold the PD
    # scale drops by more than an order of magnitude, so that same cut-off flags a very
    # different share of the panel for each learner and F1 stops measuring skill. The
    # budget variant gives every learner the same number of alarms, which is also the
    # real operating constraint: an analyst team can only review so many names.
    pdv = panel["PD_3M"].to_numpy(float)
    thr = float(np.quantile(pdv, 1.0 - BUDGET))
    alarm_b = (pdv >= thr).astype(int)
    prec_b, rec_b, f1_b, tp_b, fp_b, fn_b = score_alarm(yv, alarm_b)
    pm = pipeline_metrics(panel_raw, pdv, yv, alarm_b)

    vc = panel["alert_level"].value_counts()
    res = dict(
        model=name, auc_oos=auc_oos, f1=f1, precision=prec, recall=rec,
        lead_months=act_med / DAYS_PER_MONTH if np.isfinite(act_med) else np.nan,
        lead_days=act_med, n_caught=n_caught, n_events=n_ev,
        persistent_days=pers_med,
        persistent_months=(pers_med / DAYS_PER_MONTH
                           if np.isfinite(pers_med) else np.nan),
        brier_oos=brier, tp=tp, fp=fp, fn=fn,
        n_alarm=int(alarm.sum()), n_rows=int(len(panel)),
        alarm_share=float(alarm.mean()) * 100,
        n_high=int(vc.get("HIGH RISK", 0)),
        pct_high=float(vc.get("HIGH RISK", 0)) / len(panel) * 100,
        # budget variant
        f1_bud=f1_b, precision_bud=prec_b, recall_bud=rec_b,
        thr_bud=thr, n_alarm_bud=int(alarm_b.sum()),
        n_caught_bud=pm["n_caught"], lead_days_bud=pm["act_med"],
        lead_months_bud=(pm["act_med"] / DAYS_PER_MONTH
                         if np.isfinite(pm["act_med"]) else np.nan),
        persistent_days_bud=pm["pers_med"],
        seconds=round(time.time() - t0, 1))
    if verbose:
        print(f"    {name:15s} AUC {auc_oos:.4f} | fixed rule F1 {f1:.4f} "
              f"caught {n_caught}/{n_ev} | budget F1 {f1_b:.4f} "
              f"P {prec_b:.4f} R {rec_b:.4f} caught {pm['n_caught']}/{n_ev} "
              f"({res['seconds']}s)")
    return res, lt.assign(model=name)


def _row_table(res, caption, label, f1c, pc, rc, leadc, caughtc, note):
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{AUC (OOS)} & \textbf{F1} & "
             r"\textbf{Precision} & \textbf{Recall} & \textbf{Lead-Time} & "
             r"\textbf{Events Caught} \\", r"\midrule"]
    best = res.loc[res[f1c].idxmax(), "model"] if res[f1c].notna().any() else None
    for _, r in res.iterrows():
        pct = (r[caughtc] / r["n_events"] * 100) if r["n_events"] else np.nan
        cells = [esc(r["model"]), f4(r["auc_oos"]), f4(r[f1c]),
                 f4(r[pc]), f4(r[rc]),
                 ("--" if pd.isna(r[leadc]) else f"{r[leadc]:.2f} เดือน"),
                 (f"{int(r[caughtc])} / {int(r['n_events'])} ({pct:.0f}\\%)"
                  if r["n_events"] else "--")]
        if r["model"] == best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[3pt] {\footnotesize " + note + "}", r"\end{table}"]
    return lines


def write_tex(res, stored):
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"

    common = (r"ทุกแถวคำนวณด้วยขั้นตอนเดียวกันทั้งหมด คือแบ่งข้อมูลด้วย "
              r"StratifiedGroupKFold จำนวน 5 ชั้นตามรหัสผู้ออก "
              r"ทำให้ทุกแถวของแผงข้อมูลได้ค่าพยากรณ์จากแบบจำลองที่ไม่เคยเห็น"
              r"ผู้ออกรายนั้น จากนั้นส่งค่า PD ที่ได้เข้าสายการคำนวณของ "
              r"Approach 2 ตามเดิม ได้แก่ Momentum เส้นแบ่งไฮเพอร์โบลา "
              r"และเกณฑ์ระดับสัญญาณ ")

    lines = _row_table(
        res,
        r"Table 10 -- การเปรียบเทียบสมรรถนะแบบจำลองบนชุด iBond 33 ปัจจัย "
        r"ที่งบสัญญาณเท่ากันร้อยละ 2",
        "tab:table10-all", "f1_bud", "precision_bud", "recall_bud",
        "lead_months_bud", "n_caught_bud",
        common + (r"ตารางนี้กำหนดให้ทุกแบบจำลองจุดสัญญาณได้เท่ากันที่ร้อยละ 2 "
                  r"ของบริษัท-เดือน คือ 334 จาก 16,686 แถว "
                  r"เพราะเกณฑ์ตายตัวที่ $PD_{3M} \ge 0.05$ ถูกตั้งไว้กับค่า PD "
                  r"ที่ประมาณจากข้อมูลทั้งชุด เมื่อเปลี่ยนมาใช้ค่านอกกลุ่มตัวอย่าง "
                  r"สเกลของ PD ลดลงมากกว่าสิบเท่า เกณฑ์เดิมจึงทำให้แต่ละแบบจำลอง "
                  r"จุดสัญญาณคนละสัดส่วนกัน และค่า F1 สะท้อนตำแหน่งของเกณฑ์ "
                  r"มากกว่าความสามารถของแบบจำลอง การให้งบสัญญาณเท่ากันยังตรงกับ"
                  r"ข้อจำกัดจริงในการทำงาน เพราะทีมตรวจสอบรับงานได้จำกัดจำนวน"))

    lines2 = _row_table(
        res,
        r"ผลของแบบจำลองเดียวกันเมื่อใช้เกณฑ์ตายตัวเดิมของสายการคำนวณ",
        "tab:table10-fixed", "f1", "precision", "recall",
        "lead_months", "n_caught",
        common + (r"ตารางนี้ใช้กฎเดิมของสายการคำนวณโดยไม่แก้ไข คือ "
                  r"$PD_{3M} \ge 0.05$ หรือเข้าเงื่อนไขไฮเพอร์โบลา "
                  r"อัตราการดักจับที่ลดลงจาก 8 จาก 8 มาอยู่ในช่วง 1 ถึง 6 จาก 8 "
                  r"เกิดจากการที่ผู้ออกที่ผิดนัดชำระไม่ได้อยู่ในชุดฝึกของแบบจำลอง"
                  r"ที่ทำนายผู้ออกรายนั้น ค่า PD ที่ได้จึงไม่สูงพอจะข้ามเกณฑ์ 0.05 "
                  r"ตัวเลขในตารางนี้แสดงไว้เพื่อชี้ว่าเกณฑ์ตายตัวต้องปรับใหม่ "
                  r"หากจะนำสายการคำนวณไปใช้กับผู้ออกที่ยังไม่เคยอยู่ในชุดฝึก"))

    lines = lines + [""] + lines2

    # the load table: what each model costs to run
    l2 = [r"\begin{table}[H]", r"\centering", r"\small",
          r"\caption{ภาระการตรวจสอบและคุณภาพของความน่าจะเป็น}",
          r"\label{tab:table10-load}",
          r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
          r"\textbf{Model} & \textbf{Brier} & \textbf{แถวที่จุดสัญญาณ} & "
          r"\textbf{สัดส่วน} & \textbf{HIGH RISK} & "
          r"\textbf{เวลานำต่อเนื่อง} \\", r"\midrule"]
    for _, r in res.iterrows():
        l2.append(" & ".join([
            esc(r["model"]), f"{r['brier_oos']:.5f}", f"{int(r['n_alarm']):,}",
            f"{r['alarm_share']:.2f}\\%", f"{int(r['n_high']):,}",
            ("--" if pd.isna(r["persistent_days"])
             else f"{r['persistent_days']:.0f} วัน")]) + r" \\")
    l2 += [r"\bottomrule", r"\end{tabular}",
           r"\\[3pt] {\footnotesize ค่า Brier loss วัดคุณภาพของความน่าจะเป็นโดยตรง "
           r"ยิ่งต่ำยิ่งดี ต่างจาก AUC ที่วัดเฉพาะการจัดลำดับ "
           r"คอลัมน์แถวที่จุดสัญญาณคือจำนวนบริษัท-เดือนที่เข้าเกณฑ์สัญญาณ "
           r"จากทั้งหมด " + f"{int(res.iloc[0]['n_rows']):,}" + r" แถว "
           r"เวลานำต่อเนื่องคือจุดเริ่มของช่วงสัญญาณต่อเนื่องช่วงสุดท้าย "
           r"ซึ่งไม่ถูกจำกัดด้วยกรอบ 1 ถึง 3 เดือน จึงแยกความสามารถ"
           r"ของแบบจำลองได้ชัดกว่าคอลัมน์เวลานำในตารางก่อนหน้า}",
           r"\end{table}"]

    # stored two-row reference
    l3 = []
    if not stored.empty:
        l3 = [r"\begin{table}[H]", r"\centering", r"\small",
              r"\caption{ค่าที่บันทึกไว้เดิมของสองแถวแรก เพื่อเทียบที่มา}",
              r"\label{tab:table10-stored}",
              r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
              r"\textbf{Model} & \textbf{AUC in} & \textbf{AUC OOS} & "
              r"\textbf{F1} & \textbf{Precision} & \textbf{Recall} \\", r"\midrule"]
        for _, r in stored.iterrows():
            nm = ("Approach 1: Logistic" if "Logistic" in str(r["model_approach"])
                  else "Approach 2: Calibrated XGBoost")
            l3.append(" & ".join([
                nm, f"{float(r['auc_in_sample']):.4f}",
                f"{float(r['auc_out_sample']):.4f}",
                f"{float(r['f1_out_sample']):.4f}",
                f"{float(r['precision_oos']):.4f}",
                f"{float(r['recall_out_sample']):.4f}"]) + r" \\")
        l3 += [r"\bottomrule", r"\end{tabular}",
               r"\\[3pt] {\footnotesize ค่าจากตาราง "
               r"\texttt{ibond\_model\_compare\_33features} "
               r"ซึ่งใช้การแบ่งข้อมูลต่างจากตารางด้านบน "
               r"ค่าที่ระบุว่านอกกลุ่มตัวอย่างต่างจากค่าในกลุ่มฝึกของตัวเอง "
               r"เพียง 0.0007 ถึง 0.0107 จึงแสดงไว้เพื่อเทียบที่มา "
               r"และไม่ได้นำมารวมในตารางหลัก}",
               r"\end{table}"]

    frag = ("\\section*{Table 10 การเปรียบเทียบสมรรถนะครบทุกแบบจำลอง}\n\n"
            + "\n".join(lines) + "\n\n" + "\n".join(l2) + "\n\n"
            + "\n".join(l3) + "\n")
    p = out("section_table10.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(frag)
    return p


def main():
    print("=" * 96)
    print("Table 10 for every model on the iBond 33-determinant panel")
    print("=" * 96)
    panel, X, y, cols = cl.load_panel(verbose=True)
    if "event_date" not in panel.columns:
        raise RuntimeError("panel has no event_date column")
    panel["month_dt"] = pd.to_datetime(panel["month"], errors="coerce")

    print(f"\n  out-of-fold predictions, StratifiedGroupKFold({N_SPLITS}) "
          f"over issuer_code ...")
    rows, lts = [], []
    for name in MODELS:
        if name != "Logistic":
            try:
                a2.base_learner(name)
            except ImportError:
                print(f"    {name}: library not installed -- skipped")
                continue
        r, lt = run_model(name, panel, X, y)
        rows.append(r)
        lts.append(lt)

    res = pd.DataFrame(rows)
    lt_all = pd.concat(lts, ignore_index=True) if lts else pd.DataFrame()

    con = sqlite3.connect(DB)
    try:
        stored = pd.read_sql("SELECT * FROM ibond_model_compare_33features", con)
    except Exception:
        stored = pd.DataFrame()
    con.close()

    p = write_tex(res, stored)
    res.to_csv(out("table10.csv"), index=False)
    if "--no-save" not in sys.argv:
        con = sqlite3.connect(DB)
        res.to_sql("cmdf_table10", con, if_exists="replace", index=False)
        if not lt_all.empty:
            lt_all.to_sql("cmdf_table10_leadtime", con, if_exists="replace",
                          index=False)
        con.commit(); con.close()

    print("\n" + "=" * 96)
    print("TABLE 10")
    print("=" * 96)
    print("matched alarm budget 2% of issuer-months (the reported table)")
    for _, r in res.iterrows():
        print(f"  {r['model']:16s} AUC {r['auc_oos']:.4f}  F1 {r['f1_bud']:.4f}  "
              f"P {r['precision_bud']:.4f}  R {r['recall_bud']:.4f}  "
              f"lead {r['lead_months_bud']:.2f} mo  "
              f"caught {int(r['n_caught_bud'])}/{int(r['n_events'])}")
    print("\nfixed rule PD>=0.05 or hyperbolic flag, unchanged from the pipeline")
    for _, r in res.iterrows():
        print(f"  {r['model']:16s} AUC {r['auc_oos']:.4f}  F1 {r['f1']:.4f}  "
              f"P {r['precision']:.4f}  R {r['recall']:.4f}  "
              f"lead {r['lead_months']:.2f} mo  "
              f"caught {int(r['n_caught'])}/{int(r['n_events'])}")
    print("-" * 96)
    print(res[["model", "brier_oos", "n_alarm", "alarm_share", "n_high",
               "persistent_days"]].to_string(index=False,
                                             float_format=lambda v: f"{v:.4f}"))
    b = res.loc[res["f1_bud"].idxmax()]
    print(f"\nhighest F1 at the matched budget: {b['model']} ({b['f1_bud']:.4f})")
    ba = res.loc[res["auc_oos"].idxmax()]
    print(f"highest AUC: {ba['model']} ({ba['auc_oos']:.4f})")
    bb = res.loc[res["brier_oos"].idxmin()]
    print(f"lowest Brier: {bb['model']} ({bb['brier_oos']:.5f})")
    print(f"\nwrote {p}")
    print(r"add with:  \input{section_table10.tex}")


if __name__ == "__main__":
    main()
