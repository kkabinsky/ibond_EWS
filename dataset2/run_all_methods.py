# -*- coding: utf-8 -*-
"""Run every classifier on Dataset2 and write the tables the manual uses.

Writes into ./out
    run_summary.csv     one row per model: AUC, AP, recall, events caught
    lead_32.csv         one row per issuer per model: alarm month, lead, persist
    lead_wide.csv       the 32 issuers by model, actionable lead in days
    tbl_dataset2_allmethods_summary.tex
    tbl_dataset2_allmethods_lead32.tex

Each model is written as soon as it finishes, so a long run can be read while
it is still going and a crash late on does not lose the earlier models.

Run  python run_all_methods.py
     python run_all_methods.py --models XGBoost CatBoost
     python run_all_methods.py --workload 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data_layer as DL

OUT = Path(__file__).resolve().parent / "out"
REFERENCE = ["XGBoost", "CatBoost"]      # the two the manual already reports


def _fmt_days(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{float(value):.0f}"


def _fmt_month(value) -> str:
    return "--" if value in (None, "", "nan") or pd.isna(value) else str(value)[:7]


def latex_escape(text: str) -> str:
    out = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        out = out.replace(a, b)
    return out


def summary_latex(summary: pd.DataFrame, catalog: int, evaluable: int) -> str:
    rows = []
    for r in summary.itertuples(index=False):
        rate = r.all_detected / max(r.all_evaluable_events, 1)
        rows.append(
            f"{latex_escape(r.model)} & {r.auc_oof:.4f} & "
            f"{r.average_precision_oof:.4f} & {r.recall_at_workload:.4f} & "
            f"{int(r.all_detected)}/{int(r.all_evaluable_events)} & "
            f"{rate*100:.1f}\\% & {_fmt_days(r.app_lead_median_days)} & "
            f"{_fmt_days(r.app_persistent_median_days)} & "
            f"{r.runtime_seconds:.1f} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{ผลทุกวิธีบน Dataset2 เหตุการณ์ {catalog} บริษัท วัดได้ {evaluable} ราย
ที่ workload เท่ากัน และ fold เดียวกัน}}
\\label{{tab:dataset2-allmethods-summary}}
\\small
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrrrr}}
\\toprule
วิธี & AUC & AP & recall รายเดือน & ตรวจได้/วัดได้ & อัตราตรวจพบ &
Lead med. (วัน) & Persistent med. (วัน) & เวลา (วิ) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""


def lead32_latex(wide: pd.DataFrame, models: list[str]) -> str:
    head_models = " & ".join(latex_escape(m) for m in models)
    spec = "lll" + "r" * len(models)
    rows = []
    for r in wide.itertuples(index=False):
        cells = " & ".join(_fmt_days(getattr(r, m.replace(" ", "_"))) for m in models)
        rows.append(f"{latex_escape(r.firm_name)[:26]} & {latex_escape(r.event_type)} & "
                    f"{r.event_month} & {cells} \\\\")
    body = "\n".join(rows)
    ncol = 3 + len(models)
    return f"""\\begingroup
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}
\\scriptsize
\\begin{{longtable}}{{{spec}}}
\\caption{{Actionable lead time หน่วยวัน รายบริษัททั้ง {len(wide)} ราย เทียบทุกวิธี
เครื่องหมาย -- คือไม่มี alarm ในกรอบ 1--3 เดือน}}\\label{{tab:dataset2-allmethods-lead32}}\\\\
\\toprule
บริษัท & เหตุ & เดือนเหตุ & {head_models} \\\\
\\midrule
\\endfirsthead
\\multicolumn{{{ncol}}}{{c}}{{ตาราง \\ref{{tab:dataset2-allmethods-lead32}} (ต่อ)}}\\\\
\\toprule
บริษัท & เหตุ & เดือนเหตุ & {head_models} \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{{ncol}}}{{r}}{{มีหน้าถัดไป}}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
{body}
\\end{{longtable}}
\\endgroup
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workload", type=float, default=DL.DEFAULT_WORKLOAD)
    ap.add_argument("--risk-set", action="store_true",
                    help="drop months after an issuer's first event, the "
                         "convention Approach 1 and 2 use, so the numbers "
                         "compare directly with them")
    args = ap.parse_args()
    models = list(args.models or DL.MODEL_NAMES)

    OUT.mkdir(exist_ok=True)
    print(f"Loading Dataset2 from {DL.DEFAULT_SOURCE_DB.resolve()}", flush=True)
    data = DL.load_model_data(risk_set=args.risk_set)
    print(DL.banner(data), flush=True)
    if args.risk_set:
        print(f"Risk set: {data['excluded_post_event_rows']:,} months after a "
              f"first event removed, {len(data['panel']):,} months kept",
              flush=True)
    catalog = data["all_events"]
    print(f"\nEvaluating {len(models)} methods against the "
          f"{len(catalog)} issuers with an event", flush=True)

    rows, lead_frames, sweeps, curves = [], [], [], {}
    for name in models:
        print(f"\nTraining {name} with app-compatible grouped OOF protocol", flush=True)
        try:
            run = DL.fit_oof(data, name, args.workload)
        except Exception as exc:
            print(f"  {name} skipped: {exc}", flush=True)
            continue
        rows.append(DL.summarise_run(data, run))
        lead = DL.lead_table(data, run, catalog)
        lead_frames.append(lead)
        sweeps.append(DL.precision_recall_workload_sweep(
            data["y"], run["oof"], data["latest_rows"], name))
        curves[f"{name}__precision"] = run["pr_curve"]["precision"]
        curves[f"{name}__recall"] = run["pr_curve"]["recall"]
        # written after every model so a long run can be read while it goes
        pd.DataFrame(rows).to_csv(OUT / "run_summary.csv", index=False)
        pd.concat(lead_frames, ignore_index=True).to_csv(OUT / "lead_32.csv", index=False)
        pd.concat(sweeps, ignore_index=True).to_csv(
            OUT / "workload_sweep.csv", index=False)
        np.savez_compressed(OUT / "pr_curves.npz", **curves)
        r = rows[-1]
        print(f"  {name}: AUC {r['auc_oof']:.4f} | recall {r['recall_at_workload']:.4f} "
              f"| caught {int(r['all_detected'])}/{int(r['all_evaluable_events'])}",
              flush=True)

    if not rows:
        raise SystemExit("no model finished")

    summary = pd.DataFrame(rows)
    summary = summary[[c for c in DL.SUMMARY_COLUMNS if c in summary.columns]]
    # the two the manual already reports come first, then the rest by detection
    order = ([m for m in REFERENCE if m in set(summary["model"])]
             + [m for m in summary.sort_values("all_detected", ascending=False)["model"]
                if m not in REFERENCE])
    summary["_o"] = summary["model"].map({m: i for i, m in enumerate(order)})
    summary = summary.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    summary.to_csv(OUT / "run_summary.csv", index=False)

    all_lead = pd.concat(lead_frames, ignore_index=True)
    all_lead.to_csv(OUT / "lead_32.csv", index=False)
    pd.concat(sweeps, ignore_index=True).to_csv(
        OUT / "workload_sweep.csv", index=False)
    np.savez_compressed(OUT / "pr_curves.npz", **curves)

    wide = (all_lead.pivot_table(index=["firm_name", "event_type", "event_month"],
                                 columns="model", values="actionable_lead_days",
                                 aggfunc="first")
            .reset_index().sort_values("event_month"))
    wide.columns = [str(c).replace(" ", "_") for c in wide.columns]
    wide.to_csv(OUT / "lead_wide.csv", index=False)

    present = [m for m in summary["model"] if m.replace(" ", "_") in wide.columns]
    catalog_n = int(summary["all_event_catalog"].iloc[0])
    evaluable_n = int(summary["all_evaluable_events"].iloc[0])
    (OUT / "tbl_dataset2_allmethods_summary.tex").write_text(
        summary_latex(summary, catalog_n, evaluable_n), encoding="utf-8")
    (OUT / "tbl_dataset2_allmethods_lead32.tex").write_text(
        lead32_latex(wide, present), encoding="utf-8")

    print(DL.print_run_summary(summary))
    print(f"\nWritten to {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
