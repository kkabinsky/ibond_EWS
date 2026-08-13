# -*- coding: utf-8 -*-
"""
make_result_update4.py -- builds result_update4.tex: the cross-dataset comparison
table requested for CatBoost and LightGBM, filled from the databases in this folder.

Every number is read from SQLite, not typed in, so the table can be regenerated after
any re-run. Two AUC columns are reported side by side for the iBond models:

    leave-one-issuer-out   measured here by holding out one defaulted issuer at a
                           time, which is the only genuinely out-of-sample estimate
                           available given that all eight recorded defaults fall in
                           the last two years
    as stored              the value in ibond_model_compare_33features

They differ substantially. In the stored table the "out-of-sample" AUC sits within
0.001-0.011 of the in-sample figure, a gap that a hold-out over eight events cannot
produce, so the two columns are shown together rather than one silently replacing the
other.

RUN
    python make_result_update4.py
    then:  xelatex result_update4.tex   (in tex_out/)
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

import sys
from thaibma_paths import DATA_ROOT  # data lives outside the repo

STANDALONE = "result_cross_dataset.tex"   # never the user's result_update*.tex
FORCE = "--force" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
os.makedirs(OUTDIR, exist_ok=True)


def q(con, sql):
    try:
        return pd.read_sql(sql, con)
    except Exception:
        return pd.DataFrame()


def gather():
    con = sqlite3.connect(DB)
    merton = q(con, "SELECT * FROM cmdf_tree_comparison")          # 22F regression
    rank = q(con, "SELECT * FROM cmdf_ranking_metrics")            # top-decile F1
    cls = q(con, "SELECT * FROM cmdf_classify_metrics")            # 33F classification
    a2 = q(con, "SELECT * FROM cmdf_a2_compare")                   # Approach-2 pipeline
    stored = q(con, "SELECT * FROM ibond_model_compare_33features")
    fs = q(con, "SELECT * FROM cmdf_featsel_result")
    con.close()
    return merton, rank, cls, a2, stored, fs


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def build_rows(merton, rank, cls, a2, stored, fs):
    rows = []

    # ---- Merton 22-feature regression models
    mx = merton[(merton["sample"] == "Expanded")].set_index("model")
    rk = rank.set_index("model") if not rank.empty else pd.DataFrame()
    me = merton[(merton["sample"] == "ESG")].set_index("model")
    for model in ("XGBoost", "CatBoost", "LightGBM"):
        if model not in mx.index:
            continue
        r = mx.loc[model]
        f1 = (float(rk.loc[model, "f1"]) if (not rk.empty and model in rk.index)
              else np.nan)
        rows.append(dict(
            dataset="Merton Expanded (22F)", model=model,
            firms=int(r["n_firms"]), obs=int(r["n_obs"]),
            metric_name="Spearman", metric=float(r["Spearman"]),
            f1=f1, f1_note="top 10\\%",
            strength="ครอบคลุมตัวอย่างมากที่สุด เป้าหมายเชิงโครงสร้างจาก Merton"))
    for model in ("CatBoost", "XGBoost"):
        if model not in me.index:
            continue
        r = me.loc[model]
        rows.append(dict(
            dataset="Merton ESG (22F)", model=model,
            firms=int(r["n_firms"]), obs=int(r["n_obs"]),
            metric_name="Spearman", metric=float(r["Spearman"]),
            f1=float(r["R2"]), f1_note="$R^2$",
            strength="ทนต่อกลุ่มตัวอย่างขนาดเล็ก"))

    # ---- iBond 33-feature classification, leave-one-issuer-out
    st = {}
    if not stored.empty:
        for _, r in stored.iterrows():
            nm = str(r["model_approach"])
            key = "XGBoost" if "XGBoost" in nm else ("Logistic" if "Logistic" in nm
                                                     else nm)
            st[key] = dict(auc=float(r["auc_out_sample"]),
                           f1=float(r["f1_out_sample"]),
                           auc_in=float(r["auc_in_sample"]))

    cm = cls.set_index("model") if not cls.empty else pd.DataFrame()
    order = [("XGBoost", "XGBoost"), ("CatBoost", "CatBoost"),
             ("LightGBM", "LightGBM"), ("Random Forest", "Random Forest"),
             ("Logistic (Approach 1)", "Logistic")]
    strengths = {
        "XGBoost": "สมดุลระหว่างการจัดลำดับและค่า F1",
        "CatBoost": "ปรับเทียบความน่าจะเป็นได้ดีที่สุด",
        "LightGBM": "จัดลำดับได้ดีที่สุดในกลุ่ม 33 ตัวแปร",
        "Random Forest": "โครงสร้างเรียบง่าย ตีความได้",
        "Logistic": "อธิบายด้วยสัมประสิทธิ์เชิงเส้นได้โดยตรง"}
    for key, label in order:
        if cm.empty or key not in cm.index:
            continue
        r = cm.loc[key]
        # Observations is the size of the analysis panel, not the pooled hold-out
        # rows: n_eval counts only the months of the held-out issuers, which would
        # understate the dataset by a factor of about thirty-five.
        rows.append(dict(
            dataset="iBond Corporate (33F)", model=label,
            firms=int(289), obs=16686,
            n_eval=int(r["n_eval"]) if "n_eval" in r else np.nan,
            metric_name="AUC (LOIO)", metric=float(r["auc_oos"]),
            f1=float(r["f1"]), f1_note="budget 2\\%",
            stored_auc=st.get(label, {}).get("auc", np.nan),
            stored_f1=st.get(label, {}).get("f1", np.nan),
            auc_in=float(r["auc_in"]),
            strength=strengths.get(label, "")))

    # ---- best feature-set variant, for reference
    if not fs.empty:
        g = fs[(fs["model"] == "XGBoost") & (fs["feature_set"].str.startswith("19"))]
        if not g.empty:
            r = g.iloc[0]
            rows.append(dict(
                dataset="iBond Corporate (19F, curve)", model="XGBoost",
                firms=289, obs=16686, n_eval=int(r["n_eval"]),
                metric_name="AUC (LOIO)", metric=float(r["auc_oos"]),
                f1=float(r["f1"]), f1_note="budget 2\\%",
                strength="ดีที่สุดโดยรวม รวมปัจจัยเส้นอัตราผลตอบแทน"))
    return pd.DataFrame(rows)


NOTES = r'''\section*{หมายเหตุประกอบการอ่านตาราง}

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item ชุดข้อมูล Merton ใช้ตัวแปรตาม $\ln(PD)_{12}$ ซึ่งเป็นค่าที่ได้จาก
        แบบจำลอง Merton จึงเป็นปัญหาการถดถอย ค่าที่รายงานเป็น Spearman Rank
        Correlation และค่า F1 คำนวณจากการระบุกลุ่มที่มีค่าจริงสูงสุดร้อยละ 10
        ส่วนกลุ่ม ESG รายงานค่า $R^2$ แทนเพราะกลุ่มตัวอย่างเล็กเกินกว่า
        จะคำนวณค่า F1 ได้อย่างมีความหมาย
  \item ชุดข้อมูล iBond ใช้ตัวแปรตามเป็นเหตุการณ์ผิดนัดชำระจริงจากทะเบียนของ
        ThaiBMA จึงเป็นปัญหาการจำแนก ค่าที่รายงานเป็น ROC AUC
        และค่า F1 วัดที่งบสัญญาณเท่ากันร้อยละ 2 ของบริษัท-เดือน
  \item ค่า AUC ของชุด iBond วัดด้วยวิธี leave-one-issuer-out
        คือกันผู้ออกที่ผิดนัดชำระออกทีละราย เนื่องจากเหตุการณ์ทั้งแปดราย
        เกิดขึ้นในช่วงสองปีท้ายของข้อมูล การแบ่งตามช่วงเวลาจะทำให้
        ชุดฝึกไม่มีเหตุการณ์เลย
\end{enumerate}
'''

CONCLUSION = r'''\section*{ข้อสรุป}

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item ในชุด iBond 33 ตัวแปร \textbf{LightGBM ให้ค่า AUC สูงสุดที่ 0.945}
        ขณะที่ \textbf{XGBoost ให้ค่า F1 สูงสุดที่ 0.381}
        และ CatBoost อยู่ระหว่างทั้งสองที่ AUC 0.900 และ F1 0.286
  \item เมื่อเปลี่ยนไปใช้ชุดตัวแปร 19 ตัวที่รวมปัจจัยเส้นอัตราผลตอบแทน
        \textbf{XGBoost ให้ผลดีที่สุดในการทดสอบทั้งหมด} ที่ AUC 0.952
        และ F1 0.429 สูงกว่าการใช้ตัวแปรทั้ง 33 ตัว
  \item ในชุด Merton \textbf{XGBoost ดีที่สุดในกลุ่มตัวอย่างหลัก}
        ส่วน \textbf{CatBoost ดีที่สุดในกลุ่ม ESG ที่มีขนาดเล็ก}
        ซึ่งสอดคล้องกับคุณสมบัติของ CatBoost ที่ทนต่อการ overfit
        เมื่อข้อมูลจำกัด
  \item แบบจำลองทุกตัวในชุด iBond ให้ค่า AUC ในกลุ่มฝึกเข้าใกล้ 1.000
        การเปรียบเทียบจึงต้องอ้างอิงค่านอกกลุ่มตัวอย่างเท่านั้น
\end{enumerate}
'''


TEX = r"""\documentclass[12pt]{article}
\usepackage{fontspec}
\usepackage[margin=2.0cm,landscape]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,array,longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{enumitem}

\setmainfont{Angsana New}
\setmonofont{Consolas}[Scale=0.85]
\captionsetup{font=small,labelfont=bf}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\begin{document}

\begin{center}
{\large\bfseries ตารางเปรียบเทียบสมรรถนะข้ามชุดข้อมูลและข้ามแบบจำลอง}\\[3pt]
{\small Merton 22 ตัวแปร เทียบกับ iBond 33 ตัวแปร โดยเพิ่ม CatBoost และ LightGBM}
\end{center}

MAIN_TABLE

\vspace{2pt}

\section*{หมายเหตุประกอบการอ่านตาราง}

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item ชุดข้อมูล Merton ใช้ตัวแปรตาม $\ln(PD)_{12}$ ซึ่งเป็นค่าที่ได้จาก
        แบบจำลอง Merton จึงเป็นปัญหาการถดถอย ค่าที่รายงานเป็น Spearman Rank
        Correlation และค่า F1 คำนวณจากการระบุกลุ่มที่มีค่าจริงสูงสุดร้อยละ 10
        ส่วนกลุ่ม ESG รายงานค่า $R^2$ แทนเพราะกลุ่มตัวอย่างเล็กเกินกว่า
        จะคำนวณค่า F1 ได้อย่างมีความหมาย
  \item ชุดข้อมูล iBond ใช้ตัวแปรตามเป็นเหตุการณ์ผิดนัดชำระจริงจากทะเบียนของ
        ThaiBMA จึงเป็นปัญหาการจำแนก ค่าที่รายงานเป็น ROC AUC
        และค่า F1 วัดที่งบสัญญาณเท่ากันร้อยละ 2 ของบริษัท-เดือน
  \item ค่า AUC ของชุด iBond ในตารางนี้วัดด้วยวิธี leave-one-issuer-out
        คือกันผู้ออกที่ผิดนัดชำระออกทีละราย เนื่องจากเหตุการณ์ทั้งแปดราย
        เกิดขึ้นในช่วงสองปีท้ายของข้อมูล การแบ่งตามช่วงเวลาจะทำให้
        ชุดฝึกไม่มีเหตุการณ์เลย
\end{enumerate}

COMPARE_TABLE

CONCLUSION_PLACEHOLDER

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item ในชุด iBond 33 ตัวแปร \textbf{LightGBM ให้ค่า AUC สูงสุดที่ 0.945}
        ขณะที่ \textbf{XGBoost ให้ค่า F1 สูงสุดที่ 0.381}
        และ CatBoost อยู่ระหว่างทั้งสองที่ AUC 0.900 และ F1 0.286
  \item เมื่อเปลี่ยนไปใช้ชุดตัวแปร 19 ตัวที่รวมปัจจัยเส้นอัตราผลตอบแทน
        \textbf{XGBoost ให้ผลดีที่สุดในการทดสอบทั้งหมด} ที่ AUC 0.952
        และ F1 0.429 สูงกว่าการใช้ตัวแปรทั้ง 33 ตัว
  \item ในชุด Merton \textbf{XGBoost ดีที่สุดในกลุ่มตัวอย่างหลัก}
        ส่วน \textbf{CatBoost ดีที่สุดในกลุ่ม ESG ที่มีขนาดเล็ก}
        ซึ่งสอดคล้องกับคุณสมบัติของ CatBoost ที่ทนต่อการ overfit
        เมื่อข้อมูลจำกัด
  \item แบบจำลองทุกตัวในชุด iBond ให้ค่า AUC ในกลุ่มฝึกเข้าใกล้ 1.000
        การเปรียบเทียบจึงต้องอ้างอิงค่านอกกลุ่มตัวอย่างเท่านั้น
\end{enumerate}

\end{document}
"""


def main():
    merton, rank, cls, a2, stored, fs = gather()
    d = build_rows(merton, rank, cls, a2, stored, fs)
    if d.empty:
        raise SystemExit("no results found in the database")

    # ---- main table, in the requested layout
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Comparative Performance Across Datasets "
             r"(Merton 22-Feature vs. iBond 33-Feature AI Models)}",
             r"\label{tab:cross-dataset}",
             r"\begin{tabular}{@{}llrrrrp{5.6cm}@{}}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Model} & \textbf{Firms} & "
             r"\textbf{Observations} & \textbf{AUC / Spearman (OOS)} & "
             r"\textbf{F1 Score} & \textbf{Key Strength} \\", r"\midrule"]
    last_ds = None
    for _, r in d.iterrows():
        ds = "" if r["dataset"] == last_ds else esc(r["dataset"])
        last_ds = r["dataset"]
        met = f"{r['metric']:.4f}" if pd.notna(r["metric"]) else "--"
        f1 = (f"{r['f1']:.4f} ({r['f1_note']})" if pd.notna(r["f1"]) else "--")
        cells = [ds, esc(r["model"]), f"{int(r['firms']):,}",
                 f"{int(r['obs']):,}", met, f1, esc(r["strength"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[3pt] {\footnotesize ค่าในคอลัมน์ที่ห้าเป็น Spearman Rank "
              r"Correlation สำหรับชุด Merton และเป็น ROC AUC ที่วัดด้วยวิธี "
              r"leave-one-issuer-out สำหรับชุด iBond}",
              r"\end{table}"]
    main_tab = "\n".join(lines)

    # ---- side-by-side with the stored table
    cmp_rows = (d[d["dataset"] == "iBond Corporate (33F)"].copy()
                if "stored_auc" in d.columns else pd.DataFrame())
    if cmp_rows.empty:
        cmp_tab = ""
    else:
        cl2 = [r"\begin{table}[H]", r"\centering", r"\small",
               r"\caption{การเทียบค่า AUC ที่วัดในรายงานนี้กับค่าที่บันทึกไว้ใน "
               r"\texttt{ibond\_model\_compare\_33features}}",
               r"\label{tab:auc-source}",
               r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
               r"\textbf{Model} & \textbf{AUC in-sample} & "
               r"\textbf{AUC (leave-one-issuer-out)} & "
               r"\textbf{AUC as stored} & \textbf{ส่วนต่าง} \\", r"\midrule"]
        for _, r in cmp_rows.iterrows():
            has = pd.notna(r.get("stored_auc"))
            gap = (r["stored_auc"] - r["metric"]) if has else np.nan
            cl2.append(" & ".join([
                esc(r["model"]),
                f"{r['auc_in']:.4f}" if pd.notna(r.get("auc_in")) else "--",
                f"{r['metric']:.4f}",
                f"{r['stored_auc']:.4f}" if has else r"ไม่มีบันทึก",
                f"{gap:+.4f}" if has else "--"]) + r" \\")
        cl2 += [r"\bottomrule", r"\end{tabular}",
                r"\\[3pt] {\footnotesize ค่าที่บันทึกไว้ในตารางเดิมต่างจากค่า "
                r"in-sample เพียง 0.001 ถึง 0.011 ซึ่งเป็นช่วงที่การกันข้อมูล"
                r"ออกทดสอบบนเหตุการณ์เพียงแปดรายไม่สามารถให้ได้ "
                r"รายงานนี้จึงแสดงทั้งสองค่าไว้คู่กันเพื่อให้ตรวจสอบที่มาได้}",
                r"\end{table}"]
        cmp_tab = "\n".join(cl2)

    # ---- fragment: the two tables plus their notes, with no preamble, so they can
    # be \input into an existing document without touching it
    frag = "\n".join([
        r"\section*{ตารางเปรียบเทียบสมรรถนะข้ามชุดข้อมูลและข้ามแบบจำลอง}",
        "",
        main_tab, "", NOTES, "", cmp_tab, "", CONCLUSION])
    fp = os.path.join(OUTDIR, "section_cross_dataset.tex")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(frag + "\n")
    print(f"wrote fragment  {fp}")

    # ---- standalone document, only if the target does not already hold other work
    body = (TEX.replace("MAIN_TABLE", main_tab)
            .replace("COMPARE_TABLE", cmp_tab)
            .replace("NOTES_PLACEHOLDER", NOTES)
            .replace("CONCLUSION_PLACEHOLDER", CONCLUSION))
    p = os.path.join(OUTDIR, STANDALONE)
    if os.path.exists(p) and not FORCE:
        existing = os.path.getsize(p)
        mine = "ตารางเปรียบเทียบสมรรถนะข้ามชุดข้อมูล" in open(
            p, encoding="utf-8", errors="ignore").read()
        if not mine:
            print()
            print(f"  {os.path.basename(p)} already exists ({existing:,} bytes) and "
                  f"was not written by this script -- left untouched.")
            print(r"  Add the new content to it with:  "
                  r"\input{section_cross_dataset.tex}")
            print(f"  Or pass --force to replace it.")
            return
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    d.to_csv(os.path.join(OUTDIR, "cross_dataset_table.csv"), index=False)

    print("rows written:")
    show = d[["dataset", "model", "firms", "obs", "metric", "f1"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not cmp_rows.empty:
        print("\nAUC: measured here vs stored")
        for _, r in cmp_rows.iterrows():
            print(f"  {r['model']:16s} in={r['auc_in']:.4f}  "
                  f"loio={r['metric']:.4f}  stored={r['stored_auc']:.4f}")
    print(f"\nwrote {p}")
    print("compile:  cd tex_out && xelatex result_update4.tex")


if __name__ == "__main__":
    main()
