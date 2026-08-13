# -*- coding: utf-8 -*-
"""
make_auc_f1_table.py -- one table holding AUC and F1 for every method built in this
folder, each expressed as a percentage improvement over the Approach-1 baseline.

BASELINE
    Approach 1 logistic on the 33-determinant iBond panel, as scored in
    cmdf_classify_metrics. That row is the fair reference because it was produced by
    the same pipeline as the other classification rows: same panel, same
    leave-one-issuer-out folds, same 2% alarm budget for F1. The separately stored
    Approach-1 figure in bond_ews_summary_33 comes from a different validation path
    and is reported at the bottom for completeness rather than used as the baseline.

GROUPS COVERED
    A  iBond 33 determinants, classification of the real default event
    B  iBond 19 determinants including the yield-curve factors
    C  Approach-2 calibrated hazard pipeline with four base learners

Writes a LaTeX fragment (\input-able, nothing overwritten) plus a CSV.

RUN
    python make_auc_f1_table.py
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
os.makedirs(OUTDIR, exist_ok=True)

BASELINE = "Logistic (Approach 1)"


def q(con, sql):
    try:
        return pd.read_sql(sql, con)
    except Exception:
        return pd.DataFrame()


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def gather():
    con = sqlite3.connect(DB)
    cls = q(con, "SELECT * FROM cmdf_classify_metrics")
    fs = q(con, "SELECT * FROM cmdf_featsel_result")
    a2 = q(con, "SELECT * FROM cmdf_a2_compare")
    s33 = q(con, "SELECT * FROM bond_ews_summary_33")
    s19 = q(con, "SELECT * FROM bond_ews_summary")
    con.close()
    return cls, fs, a2, s33, s19


def build(cls, fs, a2, s33, s19):
    rows = []

    # ---- baseline
    base = cls[cls["model"] == BASELINE]
    if base.empty:
        raise SystemExit("baseline row not found in cmdf_classify_metrics")
    b_auc = float(base.iloc[0]["auc_oos"])
    b_f1 = float(base.iloc[0]["f1"])

    rows.append(dict(group="A. iBond 33 determinants", method=BASELINE,
                     features=33, auc=b_auc, f1=float(base.iloc[0]["f1"]),
                     is_base=True))
    for _, r in cls.iterrows():
        if str(r["model"]) == BASELINE:
            continue
        rows.append(dict(group="A. iBond 33 determinants", method=str(r["model"]),
                         features=33, auc=float(r["auc_oos"]), f1=float(r["f1"]),
                         is_base=False))

    # ---- 19-determinant curve set, every model
    if not fs.empty:
        g = fs[fs["feature_set"].astype(str).str.startswith("19")]
        for _, r in g.iterrows():
            rows.append(dict(group="B. iBond 19 determinants (curve set)",
                             method=str(r["model"]),
                             features=int(r["n_features"]),
                             auc=float(r["auc_oos"]), f1=float(r["f1"]),
                             is_base=False))

    # ---- Approach-2 calibrated pipeline
    if not a2.empty:
        for _, r in a2.iterrows():
            rows.append(dict(group="C. Approach 2 calibrated pipeline",
                             method=f"{r['learner']} (calibrated)",
                             features=33, auc=float(r["auc_oos"]), f1=np.nan,
                             is_base=False))

    d = pd.DataFrame(rows)
    d["auc_vs_base_pct"] = (d["auc"] - b_auc) / abs(b_auc) * 100
    d["f1_vs_base_pct"] = (d["f1"] - b_f1) / abs(b_f1) * 100
    d.loc[d["is_base"], ["auc_vs_base_pct", "f1_vs_base_pct"]] = np.nan
    d["beats_auc"] = d["auc"] > b_auc
    d["beats_f1"] = d["f1"] > b_f1

    # sort inside each group by AUC, baseline first in group A
    d["_g"] = d["group"].map({g: i for i, g in enumerate(
        ["A. iBond 33 determinants", "B. iBond 19 determinants (curve set)",
         "C. Approach 2 calibrated pipeline"])}).fillna(9)
    d["_b"] = (~d["is_base"]).astype(int)
    d = d.sort_values(["_g", "_b", "auc"], ascending=[True, True, False])
    d = d.drop(columns=["_g", "_b"]).reset_index(drop=True)
    return d, b_auc, b_f1


def write_tex(d, b_auc, b_f1, s33, s19):
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{ค่า AUC และ F1 ของทุกวิธี พร้อมร้อยละที่ดีกว่าแบบจำลอง "
             r"Approach 1}", r"\label{tab:auc-f1-all}",
             r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"\textbf{กลุ่ม} & \textbf{วิธี} & \textbf{ตัวแปร} & "
             r"\textbf{AUC} & \textbf{F1} & \textbf{AUC vs A1} & "
             r"\textbf{F1 vs A1} \\", r"\midrule"]
    last = None
    for _, r in d.iterrows():
        grp = "" if r["group"] == last else esc(r["group"])
        last = r["group"]
        auc = f"{r['auc']:.4f}"
        f1 = "--" if pd.isna(r["f1"]) else f"{r['f1']:.4f}"
        ap = ("baseline" if r["is_base"] else
              ("--" if pd.isna(r["auc_vs_base_pct"])
               else f"{r['auc_vs_base_pct']:+.1f}\\%"))
        fp = ("baseline" if r["is_base"] else
              ("--" if pd.isna(r["f1_vs_base_pct"])
               else f"{r['f1_vs_base_pct']:+.1f}\\%"))
        cells = [grp, esc(r["method"]), str(int(r["features"])), auc, f1, ap, fp]
        if bool(r["beats_auc"]) and bool(r["beats_f1"]):
            cells = [r"\textbf{" + c + "}" if c else c for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    n_auc = int(d.loc[~d["is_base"], "beats_auc"].sum())
    n_tot = int((~d["is_base"]).sum())
    n_f1 = int(d.loc[~d["is_base"] & d["f1"].notna(), "beats_f1"].sum())
    n_f1_tot = int((~d["is_base"] & d["f1"].notna()).sum())
    note = (f"แบบจำลองอ้างอิงคือ Approach 1 logistic บนชุดตัวแปร 33 ตัว "
            f"ซึ่งให้ AUC {b_auc:.4f} และ F1 {b_f1:.4f} "
            f"ประเมินด้วยวิธี leave-one-issuer-out ชุดเดียวกันทั้งหมด "
            f"และค่า F1 วัดที่งบสัญญาณเท่ากันร้อยละ 2 ของบริษัท-เดือน "
            f"มี {n_auc} จาก {n_tot} วิธีที่ให้ AUC สูงกว่าแบบจำลองอ้างอิง "
            f"และ {n_f1} จาก {n_f1_tot} วิธีที่ให้ F1 สูงกว่า "
            f"แถวที่เน้นตัวหนาคือวิธีที่ดีกว่าทั้งสองค่า "
            f"กลุ่ม C ไม่รายงานค่า F1 เพราะสายการคำนวณนั้นใช้เกณฑ์ระดับสัญญาณ"
            f"ของตัวเองแทนงบสัญญาณคงที่")
    lines.append(r"\\[3pt] {\footnotesize " + note + "}")
    lines.append(r"\end{table}")

    # a short second table: the separately stored Approach-1 numbers
    extra = []
    if not s33.empty or not s19.empty:
        extra = [r"\begin{table}[H]", r"\centering", r"\small",
                 r"\caption{ค่า AUC ของ Approach 1 ที่บันทึกจากสายการคำนวณเดิม "
                 r"เพื่อความครบถ้วน}", r"\label{tab:a1-stored}",
                 r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
                 r"\textbf{แหล่ง} & \textbf{ตัวแปร} & \textbf{AUC in-sample} & "
                 r"\textbf{AUC out-of-sample} \\", r"\midrule"]
        if not s19.empty:
            r = s19.iloc[0]
            extra.append(r"\texttt{bond\_ews\_summary} & 19 & "
                         f"{float(r['auc_in']):.4f} & {float(r['auc_oos']):.4f}" + r" \\")
        if not s33.empty:
            r = s33.iloc[0]
            extra.append(r"\texttt{bond\_ews\_summary\_33} & 33 & "
                         f"{float(r['auc_in']):.4f} & {float(r['auc_oos']):.4f}" + r" \\")
        extra += [r"\bottomrule", r"\end{tabular}",
                  r"\\[3pt] {\footnotesize ค่าเหล่านี้มาจากสายการคำนวณของ Approach 1 "
                  r"เอง ซึ่งใช้การแบ่งข้อมูลต่างจากตารางด้านบน จึงแสดงไว้เพื่อ"
                  r"ความครบถ้วนและไม่ได้ใช้เป็นฐานในการคำนวณร้อยละ}",
                  r"\end{table}"]

    frag = ("\\section*{ค่า AUC และ F1 ของทุกวิธี เทียบกับ Approach 1}\n\n"
            + "\n".join(lines) + "\n\n" + "\n".join(extra) + "\n")
    p = os.path.join(OUTDIR, "section_auc_f1_all.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(frag)
    return p, n_auc, n_tot, n_f1, n_f1_tot


def main():
    cls, fs, a2, s33, s19 = gather()
    d, b_auc, b_f1 = build(cls, fs, a2, s33, s19)
    p, n_auc, n_tot, n_f1, n_f1_tot = write_tex(d, b_auc, b_f1, s33, s19)
    d.to_csv(os.path.join(OUTDIR, "auc_f1_all.csv"), index=False)

    print("=" * 96)
    print(f"BASELINE  Approach 1 logistic (33 features): AUC {b_auc:.4f}  F1 {b_f1:.4f}")
    print("=" * 96)
    show = d[["group", "method", "features", "auc", "f1",
              "auc_vs_base_pct", "f1_vs_base_pct"]].copy()
    last = None
    for _, r in show.iterrows():
        g = "" if r["group"] == last else r["group"]
        last = r["group"]
        f1 = "  --  " if pd.isna(r["f1"]) else f"{r['f1']:.4f}"
        ap = "  base " if pd.isna(r["auc_vs_base_pct"]) else f"{r['auc_vs_base_pct']:+6.1f}%"
        fp = "  base " if pd.isna(r["f1_vs_base_pct"]) else f"{r['f1_vs_base_pct']:+6.1f}%"
        print(f"{g[:34]:36s} {r['method'][:26]:28s} {int(r['features']):>3}  "
              f"{r['auc']:.4f}  {f1}  {ap}  {fp}")
    print("-" * 96)
    print(f"beats the baseline on AUC: {n_auc}/{n_tot} methods")
    print(f"beats the baseline on F1 : {n_f1}/{n_f1_tot} methods")
    best_a = d.loc[d["auc"].idxmax()]
    print(f"highest AUC: {best_a['method']} ({best_a['auc']:.4f}, "
          f"{best_a['auc_vs_base_pct']:+.1f}% over the baseline)")
    if d["f1"].notna().any():
        best_f = d.loc[d["f1"].idxmax()]
        print(f"highest F1 : {best_f['method']} ({best_f['f1']:.4f}, "
              f"{best_f['f1_vs_base_pct']:+.1f}% over the baseline)")
    print(f"\nwrote {p}")
    print(r"add to a document with:  \input{section_auc_f1_all.tex}")


if __name__ == "__main__":
    main()
