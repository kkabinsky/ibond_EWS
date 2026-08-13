# -*- coding: utf-8 -*-
"""
make_table9_dm.py -- Table 9 for every method, plus a Diebold-Mariano test against
the Approach-1 baseline.

TABLE 9 (extended)
    For each model on the real iBond 33-determinant panel:
        AUC in-sample
        AUC leave-one-issuer-out   measured here
        AUC as stored              from ibond_model_compare_33features
        gap                        stored minus leave-one-issuer-out

    The stored table holds only two engines, Approach-1 logistic and Approach-2
    XGBoost, so the other three rows carry no stored value. Those rows are kept with
    the column marked as absent rather than dropped, which is why the original
    two-row version looked incomplete.

DM TEST
    Diebold-Mariano on the Brier loss (y - p)^2 of the pooled out-of-sample
    predictions, each model against the Approach-1 logistic baseline. AUC alone
    cannot say whether a gap is larger than sampling noise; this test can, and it
    needs no threshold.

RUN
    python make_table9_dm.py
"""
from __future__ import annotations

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cmdf_tree_classify as cl
import cmdf_tree_models as tm
from cmdf_feature_select import dm_brier

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out
BASELINE = cl.BASELINE
ORDER = ["XGBoost", "CatBoost", "LightGBM", "Random Forest", BASELINE]


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def fit_all(panel, X, y, verbose=True):
    """Pooled leave-one-issuer-out probabilities for every model, plus in-sample AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    A, yv = X.to_numpy(float), y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    ev = sorted(panel.loc[y == 1, "issuer_code"].dropna().unique())
    clf = cl.classifiers()
    preds, auc_in, yy = {}, {}, None

    for name in ORDER:
        if name not in clf:
            continue
        sc = StandardScaler().fit(A)
        m = clf[name]()
        m.fit(sc.transform(A), yv)
        auc_in[name] = float(roc_auc_score(yv, m.predict_proba(sc.transform(A))[:, 1]))

        oy, op = [], []
        for held in ev:
            tr = groups != held
            if yv[tr].sum() < 2:
                continue
            sci = StandardScaler().fit(A[tr])
            mi = clf[name]()
            try:
                mi.fit(sci.transform(A[tr]), yv[tr])
            except Exception:
                continue
            oy.append(yv[~tr])
            op.append(mi.predict_proba(sci.transform(A[~tr]))[:, 1])
        if not oy:
            continue
        if yy is None:
            yy = np.concatenate(oy)
        preds[name] = np.concatenate(op)
        if verbose:
            print(f"    {name:22s} in {auc_in[name]:.4f}  "
                  f"loio {roc_auc_score(yy, preds[name]):.4f}")
    return preds, auc_in, yy


def build(preds, auc_in, yy, stored):
    from sklearn.metrics import roc_auc_score
    st = {}
    if not stored.empty:
        for _, r in stored.iterrows():
            nm = str(r["model_approach"])
            key = ("XGBoost" if "XGBoost" in nm
                   else (BASELINE if "Logistic" in nm else nm))
            st[key] = float(r["auc_out_sample"])

    base_p = preds.get(BASELINE)
    rows = []
    for name in ORDER:
        if name not in preds:
            continue
        p = preds[name]
        auc = float(roc_auc_score(yy, p))
        brier = float(np.mean((yy - p) ** 2))
        stat = pv = np.nan
        verdict = "baseline"
        if base_p is not None and name != BASELINE:
            stat, pv = dm_brier(base_p, p, yy)
            if np.isnan(pv):
                verdict = "not computable"
            elif pv < 0.05:
                verdict = ("ดีกว่า A1 (p<0.05)" if stat > 0
                           else "แย่กว่า A1 (p<0.05)")
            else:
                verdict = "ไม่ต่างจาก A1"
        rows.append(dict(model=name, auc_in=auc_in.get(name, np.nan),
                         auc_loio=auc, auc_stored=st.get(name, np.nan),
                         gap=(st[name] - auc) if name in st else np.nan,
                         brier=brier, dm_stat=stat, p_value=pv, verdict=verdict))
    return pd.DataFrame(rows)


def write_tex(d):
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"

    # Table 9
    t9 = [r"\begin{table}[H]", r"\centering", r"\small",
          r"\caption{การเทียบค่า AUC ที่วัดในรายงานนี้กับค่าที่บันทึกไว้ใน "
          r"\texttt{ibond\_model\_compare\_33features} ครบทุกวิธี}",
          r"\label{tab:table9-all}",
          r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
          r"\textbf{Model} & \textbf{AUC in-sample} & "
          r"\textbf{AUC (leave-one-issuer-out)} & \textbf{AUC as stored} & "
          r"\textbf{ส่วนต่าง} \\", r"\midrule"]
    for _, r in d.iterrows():
        has = pd.notna(r["auc_stored"])
        t9.append(" & ".join([
            esc(r["model"]), f4(r["auc_in"]), f4(r["auc_loio"]),
            f4(r["auc_stored"]) if has else r"ไม่มีบันทึก",
            f"{r['gap']:+.4f}" if has else "--"]) + r" \\")
    t9 += [r"\bottomrule", r"\end{tabular}",
           r"\\[3pt] {\footnotesize ตาราง "
           r"\texttt{ibond\_model\_compare\_33features} บันทึกไว้เพียงสองเอนจิน "
           r"คือ Approach 1 logistic และ Approach 2 XGBoost อีกสามวิธีจึงไม่มีค่า "
           r"ในคอลัมน์นั้น ค่าที่บันทึกไว้สูงกว่าค่าที่วัดด้วยการกันผู้ออกออก "
           r"ทีละราย และต่างจากค่า in-sample ของตัวเองเพียงเล็กน้อย}",
           r"\end{table}"]

    # DM table
    dm = d[d["model"] != BASELINE]
    best = dm.loc[dm["auc_loio"].idxmax(), "model"] if not dm.empty else None
    t10 = [r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{การทดสอบ Diebold--Mariano บนค่า Brier loss "
           r"เทียบกับแบบจำลอง Approach 1}", r"\label{tab:table10-dm}",
           r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
           r"\textbf{Model} & \textbf{AUC (LOIO)} & \textbf{Brier} & "
           r"\textbf{DM stat} & \textbf{p-value} & \textbf{ผลการทดสอบ} \\",
           r"\midrule"]
    for _, r in d.iterrows():
        cells = [esc(r["model"]), f4(r["auc_loio"]), f"{r['brier']:.5f}",
                 f4(r["dm_stat"]), f4(r["p_value"]), esc(r["verdict"])]
        if r["model"] == best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        t10.append(" & ".join(cells) + r" \\")
    t10 += [r"\bottomrule", r"\end{tabular}",
            r"\\[3pt] {\footnotesize ค่า Brier loss คือความคลาดเคลื่อนกำลังสอง "
            r"ของความน่าจะเป็นที่พยากรณ์ ยิ่งต่ำยิ่งดี "
            r"สถิติ DM ที่เป็นบวกหมายถึงแบบจำลองนั้นมีค่าความคลาดเคลื่อนต่ำกว่า "
            r"แบบจำลอง Approach 1 การทดสอบใช้ค่าความคลาดเคลื่อนรายจุดโดยตรง "
            r"จึงไม่ต้องกำหนดเกณฑ์ตัดใด ๆ ต่างจากค่า F1}",
            r"\end{table}"]

    frag = ("\\section*{การเทียบค่า AUC ทุกวิธี และการทดสอบ Diebold--Mariano}\n\n"
            + "\n".join(t9) + "\n\n" + "\n".join(t10) + "\n")
    p = out("section_table9_dm.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(frag)
    return p


def main():
    print("=" * 92)
    print("Table 9 for every method, plus the Diebold-Mariano test against Approach 1")
    print("=" * 92)
    panel, X, y, cols = cl.load_panel(verbose=True)
    print("\n  fitting leave-one-issuer-out for every model ...")
    preds, auc_in, yy = fit_all(panel, X, y)
    con = sqlite3.connect(DB)
    try:
        stored = pd.read_sql("SELECT * FROM ibond_model_compare_33features", con)
    except Exception:
        stored = pd.DataFrame()
    con.close()

    d = build(preds, auc_in, yy, stored)
    p = write_tex(d)
    d.to_csv(out("table9_dm.csv"), index=False)
    con = sqlite3.connect(DB)
    d.to_sql("cmdf_table9_dm", con, if_exists="replace", index=False)
    con.commit(); con.close()

    print("\nTABLE 9")
    print(d[["model", "auc_in", "auc_loio", "auc_stored", "gap"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDM TEST vs APPROACH 1 (Brier loss)")
    print(d[["model", "auc_loio", "brier", "dm_stat", "p_value", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sig = d[(d["model"] != BASELINE) & (d["p_value"] < 0.05) & (d["dm_stat"] > 0)]
    print(f"\n{len(sig)} of {len(d)-1} models beat Approach 1 significantly "
          f"(p<0.05): {', '.join(sig['model']) if len(sig) else 'none'}")
    print(f"\nwrote {p}")
    print(r"add with:  \input{section_table9_dm.tex}")


if __name__ == "__main__":
    main()
