# -*- coding: utf-8 -*-
"""
cmdf_ai_pipeline_report.py -- consolidated development / testing / evaluation report
for every AI engine built in this folder against the real iBond data.

The individual modules each answer one question and write their own tables. This
module reads all of them back, puts every engine on the same footing, and emits the
tables and figures for the pipeline section of the report.

ENGINES COVERED
    bond_ews.py                          Approach 1, logistic, 19 determinants
                                         (10 bond-level + 3 twelve-month changes
                                          + 6 yield-curve factors)
    run_survivor_ews_33features.py       Approach 1, logistic, 33 determinants
    run_survivor_ews_33features_xgb.py   Approach 2, calibrated XGBoost, 33
    cmdf_tree_classify.py                Random Forest / XGBoost / CatBoost /
                                         LightGBM, 33 determinants
    cmdf_feature_select.py               feature-set and randomisation tests

WHAT IS PUT SIDE BY SIDE
    1. the data actually downloaded from iBond (issuers, bonds, default register)
    2. out-of-sample discrimination for every engine, on identical rows
    3. operational lead time -- how many days of warning each engine delivered
       before the real payment-default dates
    4. the alarm load each engine places on a supervisor

All engines are validated the same way: leave-one-issuer-out, because every recorded
default falls in the last two years and a time split would leave the training half
with no events.

RUN
    python cmdf_ai_pipeline_report.py
    python cmdf_ai_pipeline_report.py --no-save
"""
from __future__ import annotations

import os
import sqlite3
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cmdf_tree_models as tm

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out

EC = {"A1 logistic (19, curve)": "#6b7280",
      "A1 logistic (33)": "#9ca3af",
      "A2 XGBoost (33)": "#1f3a5f",
      "Random Forest (33)": "#2e7d4f",
      "XGBoost (33)": "#0369a1",
      "CatBoost (33)": "#a8501a",
      "LightGBM (33)": "#e0a52e",
      "XGBoost (19, curve)": "#7c3aed"}

plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})


def _q(con, sql):
    try:
        return pd.read_sql(sql, con)
    except Exception:
        return pd.DataFrame()


# =========================================================== stage 1: data ===
def data_inventory(con):
    """What the download actually produced from iBond."""
    rows = []
    for t, label in (("ibond_issuer", "ผู้ออกตราสาร (issuer master)"),
                     ("ibond_corp_bond", "รายการหุ้นกู้ (bond universe)"),
                     ("ibond_default_payment", "ทะเบียนการผิดนัดชำระ"),
                     ("ibond_33features_panel", "ตารางวิเคราะห์ 33 ตัวแปร"),
                     ("bond_ews_panel", "ตารางวิเคราะห์ 19 ตัวแปร")):
        d = _q(con, f"SELECT COUNT(*) AS n FROM {t}")
        if d.empty:
            continue
        rows.append({"table": t, "content": label, "rows": int(d["n"].iloc[0])})
    log = _q(con, "SELECT * FROM ibond_bond_log ORDER BY rowid DESC LIMIT 1")
    meta = {}
    if not log.empty:
        r = log.iloc[0]
        meta = {"downloaded_at": str(r.get("downloaded_at", "")),
                "n_issuers": int(r.get("n_issuers", 0)),
                "n_bonds": int(r.get("n_bonds", 0)),
                "n_defaults": int(r.get("n_defaults", 0)),
                "source": str(r.get("source", ""))}
    d = _q(con, "SELECT default_type_en, COUNT(*) AS n FROM ibond_default_payment "
                "GROUP BY default_type_en ORDER BY n DESC")
    return pd.DataFrame(rows), meta, d


# ==================================================== stage 2: discrimination =
def collect_discrimination(con):
    """Out-of-sample AUC / F1 for every engine, all leave-one-issuer-out."""
    rows = []

    s19 = _q(con, "SELECT * FROM bond_ews_summary")
    if not s19.empty:
        r = s19.iloc[0]
        rows.append(dict(engine="A1 logistic (19, curve)", approach="Approach 1",
                         n_features=19, auc_in=float(r["auc_in"]),
                         auc_oos=float(r["auc_oos"]), f1=np.nan,
                         source="bond_ews.py"))

    s33 = _q(con, "SELECT * FROM bond_ews_summary_33")
    if not s33.empty:
        r = s33.iloc[0]
        rows.append(dict(engine="A1 logistic (33)", approach="Approach 1",
                         n_features=33, auc_in=float(r["auc_in"]),
                         auc_oos=float(r["auc_oos"]), f1=np.nan,
                         source="run_survivor_ews_33features.py"))

    sx = _q(con, "SELECT * FROM bond_ews_xgb_summary_33")
    if not sx.empty:
        r = sx.iloc[0]
        rows.append(dict(engine="A2 XGBoost (33)", approach="Approach 2",
                         n_features=33, auc_in=float(r["auc_in"]),
                         auc_oos=float(r["auc_oos"]), f1=np.nan,
                         source="run_survivor_ews_33features_xgb.py"))

    cm = _q(con, "SELECT * FROM cmdf_classify_metrics")
    for _, r in cm.iterrows():
        nm = str(r["model"])
        if nm.startswith("Logistic"):
            continue          # already represented by the A1 rows above
        rows.append(dict(engine=f"{nm} (33)", approach="Tree ensemble",
                         n_features=33, auc_in=float(r["auc_in"]),
                         auc_oos=float(r["auc_oos"]), f1=float(r["f1"]),
                         source="cmdf_tree_classify.py"))

    fs = _q(con, "SELECT * FROM cmdf_featsel_result "
                 "WHERE feature_set LIKE '19%' AND model = 'XGBoost'")
    if not fs.empty:
        r = fs.iloc[0]
        rows.append(dict(engine="XGBoost (19, curve)", approach="Tree ensemble",
                         n_features=int(r["n_features"]), auc_in=np.nan,
                         auc_oos=float(r["auc_oos"]), f1=float(r["f1"]),
                         source="cmdf_feature_select.py"))

    d = pd.DataFrame(rows)
    if d.empty:
        return d
    return d.sort_values("auc_oos", ascending=False).reset_index(drop=True)


# ========================================================= stage 3: lead time =
def collect_leadtime(con):
    """Two different lead-time metrics are stored, and they answer different
    questions. Reporting only the first makes every engine look identical.

        lead_days              the ACTIONABLE lead time: the first alarm inside the
                               fixed 1-3 calendar-month window before the event. It
                               is bounded to roughly 30-91 days by construction, so
                               similar values across engines are a property of the
                               definition, not evidence that the engines agree.
        persistent_alarm_days  the start of the final continuous alarm episode. This
                               is the one that varies by engine and shows how early
                               each engine first went and stayed on alert.
    """
    specs = [("A1 logistic (19, curve)", "bond_ews_leadtime"),
             ("A1 logistic (33)", "bond_ews_leadtime_33"),
             ("A2 XGBoost (33)", "bond_ews_xgb_leadtime_33")]
    rows, detail = [], []
    for engine, t in specs:
        d = _q(con, f"SELECT * FROM {t}")
        if d.empty or "lead_days" not in d.columns:
            continue
        d["lead_days"] = pd.to_numeric(d["lead_days"], errors="coerce")
        has_pers = "persistent_alarm_days" in d.columns
        if has_pers:
            d["persistent_alarm_days"] = pd.to_numeric(d["persistent_alarm_days"],
                                                       errors="coerce")
        got = d["lead_days"].notna()
        pg = d["persistent_alarm_days"].notna() if has_pers else pd.Series(False,
                                                                          index=d.index)
        rows.append(dict(
            engine=engine, n_events=int(len(d)), n_caught=int(got.sum()),
            actionable_median=float(d.loc[got, "lead_days"].median())
            if got.any() else np.nan,
            actionable_min=float(d.loc[got, "lead_days"].min())
            if got.any() else np.nan,
            actionable_max=float(d.loc[got, "lead_days"].max())
            if got.any() else np.nan,
            persistent_median=float(d.loc[pg, "persistent_alarm_days"].median())
            if pg.any() else np.nan,
            persistent_months=(float(d.loc[pg, "persistent_alarm_days"].median()) / 30.44
                               if pg.any() else np.nan)))
        for _, r in d.iterrows():
            detail.append(dict(
                engine=engine, issuer_code=str(r["issuer_code"]),
                lead_days=(float(r["lead_days"]) if pd.notna(r["lead_days"])
                           else np.nan),
                persistent_days=(float(r["persistent_alarm_days"])
                                 if has_pers and pd.notna(r["persistent_alarm_days"])
                                 else np.nan)))
    return pd.DataFrame(rows), pd.DataFrame(detail)


# ======================================================== stage 4: alarm load =
def collect_alarm_load(con):
    """How many issuer-months each engine asks a supervisor to look at."""
    rows = []
    for engine, t, col in (("A1 logistic (19, curve)", "bond_ews_alert", "alert"),
                           ("A1 logistic (33)", "bond_ews_alert_33", "alert_level"),
                           ("A2 XGBoost (33)", "bond_ews_xgb_alert_33", "alert_level")):
        d = _q(con, f"SELECT * FROM {t}")
        if d.empty or col not in d.columns:
            continue
        n = len(d)
        high = int((d[col] == "HIGH RISK").sum())
        rows.append(dict(engine=engine, n_rows=n, n_high=high,
                         pct_high=high / n * 100 if n else np.nan,
                         n_elevated=int((d[col] == "ELEVATED").sum()),
                         n_watch=int((d[col] == "WATCH").sum())))
    return pd.DataFrame(rows)


# ================================================================== outputs ==
def write_outputs(inv, meta, dtypes, disc, lead, lead_detail, load):
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"

    if not inv.empty:
        tm.write_tex_table(
            inv, out("tab_ai_data.tex"),
            "ข้อมูลที่ดาวน์โหลดจาก iBond และตารางวิเคราะห์ที่สร้างขึ้น",
            "tab:ai-data", cols=["table", "content", "rows"],
            note=(f"ดาวน์โหลดครั้งล่าสุด {meta.get('downloaded_at','')} "
                  f"ผ่านช่องทาง {meta.get('source','')}. "
                  f"ตารางวิเคราะห์ทั้งสองชุดมีจำนวนแถวเท่ากันและอ้างอิง"
                  f"บริษัท-เดือนชุดเดียวกัน จึงเปรียบเทียบกันได้โดยตรง."))

    if not dtypes.empty:
        tm.write_tex_table(
            dtypes, out("tab_ai_defaults.tex"),
            "องค์ประกอบของทะเบียนการผิดนัดชำระที่ใช้เป็นตัวแปรเป้าหมาย",
            "tab:ai-def", cols=["default_type_en", "n"],
            note="เป็นเหตุการณ์จริงที่ ThaiBMA บันทึกไว้ ไม่ใช่ค่าที่ประมาณจากแบบจำลอง.")

    if not disc.empty:
        best = disc.iloc[0]["engine"]
        tm.write_tex_table(
            disc, out("tab_ai_discrimination.tex"),
            "ความสามารถในการจำแนกของทุกแบบจำลองที่พัฒนาขึ้น "
            "(leave-one-issuer-out บนข้อมูล iBond ชุดเดียวกัน)",
            "tab:ai-disc",
            cols=["engine", "approach", "n_features", "auc_in", "auc_oos", "f1"],
            fmt={"auc_in": f3, "auc_oos": f3, "f1": f3},
            bold_row=lambda r, b=best: r["engine"] == b,
            note=("ค่า AUC ในกลุ่มฝึกแสดงไว้เพื่อให้เห็นระยะห่างจากค่านอกกลุ่มฝึก "
                  "ซึ่งเป็นขนาดของการ overfit ไม่ใช่ตัวเลขที่ใช้อ้างสมรรถนะ "
                  "ค่า F1 วัดที่งบสัญญาณเท่ากันร้อยละ 2 ของบริษัท-เดือน "
                  "เครื่องหมายขีดหมายถึงโปรแกรมต้นทางไม่ได้รายงานค่านั้น."))
        disc.to_csv(out("ai_discrimination.csv"), index=False)

    if not lead.empty:
        b = lead.loc[lead["persistent_median"].idxmin(), "engine"]             if lead["persistent_median"].notna().any() else lead.iloc[0]["engine"]
        tm.write_tex_table(
            lead, out("tab_ai_leadtime.tex"),
            "เวลานำที่แต่ละแบบจำลองให้ได้ก่อนวันผิดนัดชำระจริง วัดด้วยสองนิยาม",
            "tab:ai-lead",
            cols=["engine", "n_caught", "n_events", "actionable_median",
                  "actionable_min", "actionable_max", "persistent_median",
                  "persistent_months"],
            fmt={c: (lambda v: "--" if pd.isna(v) else f"{v:.0f}")
                 for c in ("actionable_median", "actionable_min", "actionable_max",
                           "persistent_median")} |
                {"persistent_months": lambda v: "--" if pd.isna(v) else f"{v:.1f}"},
            bold_row=lambda r, bb=b: r["engine"] == bb,
            note=("นิยามที่หนึ่ง Actionable Lead Time คือสัญญาณแรกที่อยู่ในหน้าต่าง "
                  "1 ถึง 3 เดือนก่อนเหตุการณ์ ค่าจึงถูกจำกัดอยู่ในช่วงประมาณ 30 ถึง 91 "
                  "วันโดยนิยาม การที่ทุกแบบจำลองให้ค่าใกล้กันจึงเป็นคุณสมบัติของนิยาม "
                  "ไม่ใช่หลักฐานว่าแบบจำลองให้ผลเหมือนกัน "
                  "นิยามที่สอง Persistent Alarm Duration คือจุดเริ่มของช่วงสัญญาณ "
                  "ต่อเนื่องช่วงสุดท้ายก่อนเหตุการณ์ ซึ่งเป็นค่าที่แตกต่างกันตามแบบจำลอง "
                  "และสะท้อนว่าแบบจำลองเริ่มเตือนและคงสัญญาณไว้ตั้งแต่เมื่อใด."))
        lead.to_csv(out("ai_leadtime.csv"), index=False)

    if not load.empty:
        tm.write_tex_table(
            load, out("tab_ai_alarmload.tex"),
            "ภาระการตรวจสอบที่แต่ละแบบจำลองสร้างขึ้น",
            "tab:ai-load",
            cols=["engine", "n_rows", "n_high", "pct_high", "n_elevated", "n_watch"],
            fmt={"pct_high": lambda v: "--" if pd.isna(v) else f"{v:.1f}\\%"},
            note=("จำนวนบริษัท-เดือนที่ถูกจัดเป็นระดับเสี่ยงสูง "
                  "ตัวเลขนี้ต้องอ่านคู่กับเวลานำและอัตราการจับได้ "
                  "เพราะการจับได้ครบทุกรายย่อมทำได้ง่ายขึ้นเมื่อส่งสัญญาณกว้างขึ้น."))
        load.to_csv(out("ai_alarmload.csv"), index=False)

    # figure: three panels -- AUC, lead time, alarm load
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    ax = axes[0]
    if not disc.empty:
        d = disc.dropna(subset=["auc_oos"])
        ys = np.arange(len(d))
        ax.barh(ys, d["auc_oos"], color=[EC.get(e, "#888") for e in d["engine"]],
                alpha=0.92, edgecolor="white")
        for yy, v in zip(ys, d["auc_oos"]):
            ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(d["engine"], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.02)
        ax.axvline(0.5, color="#6b7280", lw=1.0, ls=":")
        ax.set_xlabel("out-of-sample AUC")
        ax.set_title("Discrimination", fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    if not lead.empty:
        ys = np.arange(len(lead))
        h = 0.38
        ax.barh(ys - h / 2, lead["actionable_median"], height=h, color="#9ca3af",
                alpha=0.95, edgecolor="white", label="actionable (1-3m window)")
        ax.barh(ys + h / 2, lead["persistent_median"], height=h,
                color=[EC.get(e, "#888") for e in lead["engine"]], alpha=0.95,
                edgecolor="white", label="persistent alarm start")
        for yy, r in zip(ys, lead.itertuples()):
            if not np.isnan(r.persistent_median):
                ax.text(r.persistent_median + 8, yy + h / 2,
                        f"{r.persistent_median:.0f} d", va="center", fontsize=7.5)
            if not np.isnan(r.actionable_median):
                ax.text(r.actionable_median + 8, yy - h / 2,
                        f"{r.actionable_median:.0f} d "
                        f"({r.n_caught}/{r.n_events})", va="center", fontsize=7)
        ax.axvline(91, color="#dc2626", lw=1.2, ls="--")
        ax.text(96, -0.45, "3-month window", fontsize=7, color="#dc2626")
        ax.legend(fontsize=7)
        ax.set_yticks(ys)
        ax.set_yticklabels(lead["engine"], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("median lead time (days)")
        ax.set_title("Operational lead time", fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    if not load.empty:
        ys = np.arange(len(load))
        ax.barh(ys, load["pct_high"],
                color=[EC.get(e, "#888") for e in load["engine"]], alpha=0.92,
                edgecolor="white")
        for yy, r in zip(ys, load.itertuples()):
            ax.text(r.pct_high + 0.05, yy, f"{r.pct_high:.1f}%  ({r.n_high:,})",
                    va="center", fontsize=7.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(load["engine"], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("share of issuer-months flagged HIGH RISK (%)")
        ax.set_title("Alarm load", fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("AI engines developed on the iBond data: discrimination, lead time "
                 "and alarm load", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    tm.save_fig(fig, "fig_ai_pipeline.png")

    # figure: lead time per issuer, per engine
    if not lead_detail.empty:
        val = ("persistent_days"
               if lead_detail["persistent_days"].notna().any() else "lead_days")
        piv = lead_detail.pivot_table(index="issuer_code", columns="engine",
                                      values=val)
        piv = piv.reindex(piv.mean(axis=1).sort_values().index)
        fig, ax = plt.subplots(figsize=(9.6, 0.46 * len(piv) + 1.8))
        engines = [c for c in piv.columns]
        h = 0.8 / max(len(engines), 1)
        ys = np.arange(len(piv))
        for i, e in enumerate(engines):
            ax.barh(ys + i * h - 0.4 + h / 2, piv[e].values, height=h,
                    color=EC.get(e, "#888"), alpha=0.92, label=e)
        ax.axvline(91, color="#dc2626", lw=1.2, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(piv.index, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("lead time (days)")
        ax.set_title("Persistent alarm start per defaulted issuer", fontsize=10.5,
                     fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        tm.save_fig(fig, "fig_ai_leadtime_issuer.png")


def run(save=True, verbose=True):
    print("=" * 78)
    print("Consolidated AI development / testing / evaluation report (iBond data)")
    print("=" * 78)
    con = sqlite3.connect(DB)
    try:
        inv, meta, dtypes = data_inventory(con)
        disc = collect_discrimination(con)
        lead, lead_detail = collect_leadtime(con)
        load = collect_alarm_load(con)
    finally:
        con.close()

    if verbose:
        print("\n[1] data downloaded from iBond")
        if meta:
            print(f"    {meta['downloaded_at']} via {meta['source']}: "
                  f"{meta['n_issuers']:,} issuers, {meta['n_bonds']:,} bonds, "
                  f"{meta['n_defaults']} default records")
        print(inv.to_string(index=False) if not inv.empty else "    (none)")

        print("\n[2] discrimination, all engines (leave-one-issuer-out)")
        if not disc.empty:
            print(disc[["engine", "approach", "n_features", "auc_in", "auc_oos", "f1"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        print("\n[3] operational lead time")
        if not lead.empty:
            print(lead.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

        print("\n[4] alarm load")
        if not load.empty:
            print(load.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    write_outputs(inv, meta, dtypes, disc, lead, lead_detail, load)
    if save:
        con = sqlite3.connect(DB)
        if not disc.empty:
            disc.to_sql("cmdf_ai_discrimination", con, if_exists="replace",
                        index=False)
        if not lead.empty:
            lead.to_sql("cmdf_ai_leadtime", con, if_exists="replace", index=False)
        if not load.empty:
            load.to_sql("cmdf_ai_alarmload", con, if_exists="replace", index=False)
        con.commit(); con.close()
    return inv, disc, lead, load


def main():
    inv, disc, lead, load = run(save="--no-save" not in sys.argv)
    if not disc.empty:
        b = disc.iloc[0]
        print(f"\nbest discrimination: {b['engine']} (AUC {b['auc_oos']:.3f})")
    if not lead.empty:
        if lead["persistent_median"].notna().any():
            bl = lead.loc[lead["persistent_median"].idxmin()]
            print(f"earliest-but-tightest persistent alarm: {bl['engine']} "
                  f"({bl['persistent_median']:.0f} days, caught "
                  f"{bl['n_caught']}/{bl['n_events']})")
    print("\nArtefacts: tex_out/tab_ai_data.tex, tab_ai_defaults.tex, "
          "tab_ai_discrimination.tex, tab_ai_leadtime.tex, tab_ai_alarmload.tex, "
          "fig_ai_pipeline.png, fig_ai_leadtime_issuer.png")
    print("Done.")


if __name__ == "__main__":
    main()
