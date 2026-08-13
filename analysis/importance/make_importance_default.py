# -*- coding: utf-8 -*-
"""
make_importance_default.py -- feature importance measured against the REAL default
event, for every tree model.

WHY THIS MODULE EXISTS
    The importance table and figure in the report (Table 6, fig_importance_compare)
    come from cmdf_tree_models.py, whose TARGET is base.TARGET = "ln_pd12m", the
    Merton 12-month probability of default. That is a regression on a model-derived
    quantity, and the Merton PD is itself a function of leverage and asset volatility.
    Ranking determinants against it therefore puts D/E first in every model almost by
    construction: the target is largely a transformation of the leverage input.

    That is not the question a credit early-warning system needs answered. The
    question is which determinants carry information about an ACTUAL default or
    restructuring, and on that target D/E is not first. On the iBond panel its
    out-of-sample permutation AUC drop is +0.0008 with p = 0.4865, and it is selected
    into a model's own top five in 2 of 32 model-fold combinations.

    This module produces the importance figures and table against the real event so
    the report can show both and label each one for what it is.

THREE MEASURES, REPORTED SIDE BY SIDE
    gain          in-sample, what the trees used. With 32 positive months from 8
                  issuers this is the least trustworthy of the three and is shown
                  only for continuity with the existing table.
    mean |SHAP|   in-sample attribution per row, same caveat.
    perm AUC drop leave-one-issuer-out. The determinant is shuffled in the held-out
                  rows only, so a positive drop means real predictive information was
                  lost. Ranked on this column, with a Diebold-Mariano test on Brier
                  loss to separate a genuine loss from sampling noise.

RUN
    python make_importance_default.py
    python make_importance_default.py --no-perm     skip the slow permutation pass
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

import cmdf_feature_select as fs
import cmdf_tree_classify as cl
import cmdf_tree_models as tm

OUTDIR = tm.OUTDIR
DB = tm.DB
out = tm.out

MODELS = ["Random Forest", "XGBoost", "CatBoost", "LightGBM"]
MC = {"Random Forest": "#2e7d4f", "XGBoost": "#1f3a5f",
      "CatBoost": "#a8501a", "LightGBM": "#e0a52e"}
TOP_SHOW = 14
SHAP_N = 4000

plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def _slug(s):
    return s.lower().replace(" ", "_")


def gain_and_shap(panel, X, y, cols):
    """In-sample gain and mean |SHAP| per model, on the real default target."""
    from sklearn.preprocessing import StandardScaler
    try:
        import shap
    except ImportError:
        shap = None
        print("    shap not installed - SHAP columns will be blank")

    A, yv = X.to_numpy(float), y.to_numpy(int)
    rows, shap_store = [], {}
    for name in MODELS:
        ctor = cl.classifiers().get(name)
        if ctor is None:
            print(f"    {name}: not available - skipped")
            continue
        t0 = time.time()
        sc = StandardScaler().fit(A)
        m = ctor()
        m.fit(sc.transform(A), yv)

        imp = np.asarray(getattr(m, "feature_importances_",
                                 np.zeros(len(cols))), dtype=float)
        imp = imp / (imp.sum() or 1.0)

        mabs = np.full(len(cols), np.nan)
        if shap is not None:
            # the same rows for every model, so the columns of the SHAP table are
            # comparable in level and not just in ordering
            idx = np.random.default_rng(42).choice(
                len(A), size=min(SHAP_N, len(A)), replace=False)
            Xs = sc.transform(A[idx])
            try:
                sv = shap.TreeExplainer(m).shap_values(Xs)
                if isinstance(sv, list):
                    sv = sv[-1]
                sv = np.asarray(sv)
                if sv.ndim == 3:            # (rows, features, classes)
                    sv = sv[:, :, -1]
                mabs = np.abs(sv).mean(0)
                # keep the raw (unscaled) values too, so dependence plots read in the
                # determinant's own units rather than in z-scores
                shap_store[name] = (sv, Xs, A[idx])
            except Exception as ex:
                print(f"    {name}: SHAP failed ({ex})")

        for c, g, s in zip(cols, imp, mabs):
            rows.append(dict(model=name, feature=c, gain=float(g),
                             mean_abs_shap=float(s)))
        print(f"    {name:15s} gain + SHAP in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows), shap_store


def perm_all(panel, X, y, cols, n_rep=3):
    """Leave-one-issuer-out permutation importance for every model."""
    frames = []
    for name in MODELS:
        if name not in cl.classifiers():
            continue
        t0 = time.time()
        d = fs.permutation_importance_oos(panel, X, y, cols, model_name=name,
                                          n_rep=n_rep, verbose=False)
        if d.empty:
            continue
        d = d.assign(model=name)
        frames.append(d)
        best = d.sort_values("auc_drop", ascending=False).iloc[0]
        print(f"    {name:15s} {len(cols)} determinants x {n_rep} shuffles "
              f"({time.time()-t0:.0f}s)  top: {best['feature']} "
              f"{best['auc_drop']:+.4f}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ================================================================= figures ====
def per_model_figs(shap_store, cols):
    """Bar, beeswarm and dependence for every model, against the real default event.
    These replace the figures produced against the Merton PD regression target."""
    try:
        import shap
    except ImportError:
        return
    names = [tm.PRETTY.get(c, c) for c in cols]
    for model_name, (sv, Xs, Xraw) in shap_store.items():
        slug = _slug(model_name)
        mabs = np.abs(sv).mean(0)
        o = np.argsort(mabs)[::-1]

        fig, ax = plt.subplots(figsize=(8.0, 0.32 * len(o) + 1.4))
        ax.barh(np.arange(len(o)), mabs[o], color=MC[model_name], alpha=0.92)
        ax.set_yticks(np.arange(len(o)))
        ax.set_yticklabels([names[i] for i in o], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(f"Determinant importance vs the real default event: "
                     f"{model_name}", fontsize=10.5, fontweight="bold",
                     color=MC[model_name])
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        tm.save_fig(fig, f"fig_shapdef_bar_{slug}.png")

        plt.figure(figsize=(8.2, 6.0))
        shap.summary_plot(sv, Xs, feature_names=names, show=False, max_display=18)
        plt.title(f"SHAP summary vs the real default event: {model_name}",
                  fontsize=10.5, fontweight="bold", color=MC[model_name])
        tm.save_fig(plt.gcf(), f"fig_shapdef_beeswarm_{slug}.png")

        pick = o[:6]
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
        for ax, j in zip(axes.ravel(), pick):
            x = Xraw[:, j]
            ax.scatter(x, sv[:, j], s=5, alpha=0.35, color=MC[model_name],
                       edgecolors="none")
            ax.axhline(0, color="#444", lw=0.8, ls="--")
            ax.set_xlabel(names[j], fontsize=8.5)
            ax.set_ylabel("SHAP", fontsize=8.5)
            ax.grid(alpha=0.25)
            fin = np.isfinite(x)
            if fin.sum() > 20:
                lo, hi = np.nanpercentile(x[fin], [1, 99])
                if hi > lo:
                    ax.set_xlim(lo, hi)
        for ax in axes.ravel()[len(pick):]:
            ax.axis("off")
        fig.suptitle(f"SHAP dependence vs the real default event: {model_name} "
                     f"(above zero raises the default probability)",
                     fontsize=11, fontweight="bold", color=MC[model_name])
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        tm.save_fig(fig, f"fig_shapdef_dependence_{slug}.png")


def native_table(d, top=TOP_SHOW):
    """The direct counterpart of Table 6, but against the real default event.

    Table 6 in the report is built by cmdf_tree_models.step_importance on a different
    panel entirely: 87,019 firm-months of the SET/Merton sample, 22 winsorized lagged
    determinants, target ln(PD)_12m. Not one determinant name is shared with the iBond
    33-determinant panel used here, so the two tables cannot be read as versions of
    each other. This one is generated so the report carries a like-for-like table on
    the target the early-warning system actually has to predict."""
    piv = d.pivot_table(index="pretty", columns="model", values="gain")
    present = [m for m in MODELS if m in piv.columns]
    if not present:
        return None
    piv = piv[present]
    piv["Mean"] = piv.mean(axis=1)
    piv = piv.sort_values("Mean", ascending=False).head(top)
    tab = piv.reset_index().rename(columns={"pretty": "Determinant"})
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"
    cols = ["Determinant"] + present + ["Mean"]

    # determinant names carry underscores (ln_amihud, amihud_monthly_100), which are
    # math-shift characters in LaTeX and abort the compile if they reach the note raw
    firsts = {m: esc(d[d["model"] == m].sort_values("gain", ascending=False)
                     .iloc[0]["pretty"]) for m in present}
    same = len(set(firsts.values())) == 1
    lead = ", ".join(f"{esc(m)} เลือก {v}" for m, v in firsts.items())

    tm.write_tex_table(
        tab, out("tab_importance_default.tex"),
        f"Native importance of the top {top} determinants against the real default "
        f"event, by model (iBond 33-determinant panel)",
        "tab:importance-default", cols=cols, fmt={c: f4 for c in cols[1:]},
        bold_row=lambda r: r["Determinant"] == tab.iloc[0]["Determinant"],
        note=("ตารางนี้ใช้แผงข้อมูล iBond 16,686 บริษัท-เดือน จากผู้ออก 289 ราย "
              "ตัวแปร 33 ตัว และตัวแปรตามคือการผิดนัดชำระหรือปรับโครงสร้างหนี้จริง "
              "ภายในสามเดือนข้างหน้า ต่างจากตารางที่ 6 ซึ่งใช้แผงข้อมูล 87,019 "
              "บริษัท-เดือน ตัวแปร 22 ตัว และตัวแปรตามคือ $\\ln(PD)_{12m}$ ของ "
              "Merton ชื่อตัวแปรของสองแผงไม่ซ้ำกันแม้แต่ตัวเดียว "
              "จึงอ่านเป็นตารางเดียวกันคนละรุ่นไม่ได้ "
              + ("แบบจำลองทุกตัวให้อันดับหนึ่งตรงกัน "
                 if same else
                 "อันดับหนึ่งของแต่ละแบบจำลองไม่ตรงกัน คือ " + lead + " ")
              + "ค่าในตารางวัดในกลุ่มฝึก ซึ่งกับเหตุการณ์เพียง 8 ราย "
                "มีความไม่แน่นอนสูง การจัดอันดับที่เชื่อถือได้ที่สุดอยู่ในตาราง "
                "ที่วัดด้วยการสลับค่าปัจจัยนอกกลุ่มตัวอย่าง"))
    return tab


def shap_table(d):
    """mean |SHAP| per determinant for every model, on the real default event."""
    piv = d.pivot_table(index="pretty", columns="model", values="mean_abs_shap")
    present = [m for m in MODELS if m in piv.columns]
    if not present:
        return None
    piv = piv[present]
    piv["Mean"] = piv.mean(axis=1)
    piv = piv.sort_values("Mean", ascending=False)
    tab = piv.reset_index().rename(columns={"pretty": "Determinant"})
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"
    cols = ["Determinant"] + present + ["Mean"]
    tm.write_tex_table(
        tab, out("tab_shapdef_all.tex"),
        "Mean $|$SHAP$|$ per determinant against the real default event, "
        "all four tree models",
        "tab:shapdef-all", cols=cols, fmt={c: f4 for c in cols[1:]},
        bold_row=lambda r: r["Determinant"] == tab.iloc[0]["Determinant"],
        note=("Measured on the same 4,000 randomly drawn issuer-months for every "
              "model, so the columns are comparable in level as well as in ordering. "
              "The target is a real default or restructuring within the next three "
              "months, not the Merton PD, which is why D/E does not lead this table "
              "the way it leads the earlier one."))
    return tab


def fig_compare(d, value_col, title, fname, top=TOP_SHOW):
    piv = d.pivot_table(index="pretty", columns="model", values=value_col)
    present = [m for m in MODELS if m in piv.columns]
    if not present:
        return
    order = piv[present].mean(axis=1).sort_values(ascending=False).head(top).index
    piv = piv.loc[order]

    n = len(order)
    fig, ax = plt.subplots(figsize=(9.6, 0.44 * n + 1.8))
    ys = np.arange(n)
    h = 0.8 / len(present)
    for i, m in enumerate(present):
        ax.barh(ys + i * h - 0.4 + h / 2, piv[m].values, height=h,
                color=MC[m], label=m, edgecolor="white", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    tm.save_fig(fig, fname)


def fig_target_contrast(d_event, top=10):
    """Side by side: the same determinants ranked against the Merton PD regression
    target and against the real default event. This is the figure that shows why the
    two importance tables disagree."""
    try:
        g_pd = pd.read_csv(out("importance_all_models.csv"))
    except Exception:
        print("    importance_all_models.csv not found - contrast figure skipped")
        return
    boost = [m for m in ("XGBoost", "CatBoost", "LightGBM") if m in MODELS]

    a = (g_pd[g_pd["model"].isin(boost)].groupby("pretty")["importance"].mean()
         .sort_values(ascending=False).head(top))
    b = (d_event[d_event["model"].isin(boost)].groupby("pretty")["gain"].mean()
         .sort_values(ascending=False).head(top))

    # All text inside the figure is English on purpose. Matplotlib's default font
    # carries no Thai glyphs, so Thai strings render as empty boxes in the PNG. The
    # Thai explanation belongs in the LaTeX caption, which is typeset by XeLaTeX in
    # Angsana New and does render.
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.9))
    for ax, ser, ttl, col in (
            (axes[0], a, "Target: Merton ln(PD) 12m\n(regression)", "#7f1d1d"),
            (axes[1], b, "Target: actual default within 3 months\n(classification)",
             "#1e3a8a")):
        ys = np.arange(len(ser))
        ax.barh(ys, ser.values, color=col, alpha=0.9, edgecolor="white")
        ax.set_yticks(ys)
        ax.set_yticklabels(ser.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("mean gain share across XGBoost, CatBoost, LightGBM")
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", color=col)
        ax.grid(axis="x", alpha=0.3)
        ax.margins(x=0.14)
        for yv, v in zip(ys, ser.values):
            ax.text(v, yv, f" {v:.3f}", va="center", fontsize=7.5)
    fig.suptitle("Determinant ranking depends on the target  |  "
                 "Merton PD is itself computed from leverage, "
                 "so D/E leads the left panel almost by construction",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    tm.save_fig(fig, "fig_importance_target_contrast.png")


# ================================================================== output ====
def write_tex(d, perm):
    f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"

    piv_g = d.pivot_table(index="pretty", columns="model", values="gain")
    piv_s = d.pivot_table(index="pretty", columns="model", values="mean_abs_shap")
    present = [m for m in MODELS if m in piv_g.columns]

    if not perm.empty:
        pm = perm.copy()
        pm["pretty"] = pm["feature"].map(lambda c: tm.PRETTY.get(c, c))
        pdrop = pm.groupby("pretty")["auc_drop"].mean()
        pmin = pm.groupby("pretty")["p_value"].min()
        nsig = pm.assign(s=pm["significant"].astype(bool)).groupby("pretty")["s"].sum()
        order = pdrop.sort_values(ascending=False).head(TOP_SHOW).index
    else:
        pdrop = pmin = nsig = None
        order = piv_g[present].mean(axis=1).sort_values(ascending=False).head(
            TOP_SHOW).index

    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{ความสำคัญของปัจจัยเมื่อวัดกับเหตุการณ์ผิดนัดชำระจริง "
             r"เรียงตามค่า AUC ที่หายไปเมื่อสลับค่าปัจจัยนอกกลุ่มตัวอย่าง}",
             r"\label{tab:imp-default}",
             r"\begin{tabular}{@{}l" + "r" * (len(present) + 3) + r"@{}}",
             r"\toprule",
             r"\textbf{ปัจจัย} & " +
             " & ".join(r"\textbf{" + esc(m) + r" gain}" for m in present) +
             r" & \textbf{AUC drop} & \textbf{p ต่ำสุด} & "
             r"\textbf{โมเดลที่นัยสำคัญ} \\", r"\midrule"]
    for i, feat in enumerate(order):
        cells = [esc(feat)]
        cells += [f4(piv_g.loc[feat, m]) if feat in piv_g.index else "--"
                  for m in present]
        if pdrop is not None:
            cells += [f"{pdrop.get(feat, np.nan):+.4f}",
                      f4(pmin.get(feat, np.nan)),
                      f"{int(nsig.get(feat, 0))} / {len(present)}"]
        else:
            cells += ["--", "--", "--"]
        if i == 0:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[3pt] {\footnotesize คอลัมน์ gain วัดในกลุ่มฝึก "
              r"ซึ่งกับข้อมูลที่มีเดือนผิดนัด 32 แถวจากผู้ออก 8 ราย "
              r"เป็นค่าที่เชื่อถือได้น้อยที่สุดในตาราง จึงแสดงไว้เพื่อเทียบกับ"
              r"ตารางเดิมเท่านั้น คอลัมน์ AUC drop วัดด้วยการกันผู้ออกออกทีละราย "
              r"แล้วสลับค่าปัจจัยเฉพาะในแถวที่กันไว้ ค่าบวกหมายถึงแบบจำลอง"
              r"เสียความแม่นยำไปจริงเมื่อไม่มีปัจจัยนั้น "
              r"ค่า p มาจากการทดสอบ Diebold--Mariano บนค่า Brier loss "
              r"ระหว่างกรณีที่สลับค่ากับกรณีที่ไม่สลับ "
              r"การเรียงลำดับใช้คอลัมน์ AUC drop เพราะเป็นค่าเดียวที่วัด"
              r"นอกกลุ่มตัวอย่าง}",
              r"\end{table}"]

    body = ("\\section*{ความสำคัญของปัจจัยเมื่อวัดกับการผิดนัดชำระจริง}\n\n"
            "ตารางและรูปในหัวข้อก่อนหน้าวัดความสำคัญของปัจจัยกับค่า "
            "$\\ln(PD)_{12m}$ ของแบบจำลอง Merton ซึ่งเป็นค่าที่คำนวณมาจาก"
            "ภาระหนี้และความผันผวนของมูลค่าสินทรัพย์อยู่แล้ว "
            "การจัดอันดับกับ target นั้นจึงให้ D/E เป็นอันดับหนึ่งในทุกแบบจำลอง"
            "เกือบโดยนิยามของ target เอง ไม่ใช่เพราะเป็นข้อค้นพบเชิงประจักษ์\n\n"
            "หัวข้อนี้วัดความสำคัญกับเหตุการณ์ผิดนัดชำระหรือปรับโครงสร้างหนี้จริง "
            "ซึ่งเป็นคำถามที่ระบบเตือนภัยล่วงหน้าต้องตอบ ผลที่ได้ต่างจากเดิมชัดเจน\n\n"
            + "\n".join(lines) + "\n\n"
            "\\begin{figure}[H]\n\\centering\n"
            "\\includegraphics[width=\\textwidth]"
            "{fig_importance_target_contrast.png}\n"
            "\\caption{ปัจจัยสำคัญสิบอันดับแรกเมื่อใช้ target ต่างกัน "
            "ด้านซ้ายคือ $\\ln(PD)_{12m}$ ของ Merton ด้านขวาคือการผิดนัดชำระจริง "
            "ค่าเฉลี่ยจาก XGBoost CatBoost และ LightGBM}\n"
            "\\label{fig:imp-contrast}\n\\end{figure}\n\n"
            "\\begin{figure}[H]\n\\centering\n"
            "\\includegraphics[width=\\textwidth]{fig_importance_default_gain.png}\n"
            "\\caption{ค่า gain ของแต่ละแบบจำลองเมื่อวัดกับการผิดนัดชำระจริง}\n"
            "\\label{fig:imp-default-gain}\n\\end{figure}\n\n"
            "\\begin{figure}[H]\n\\centering\n"
            "\\includegraphics[width=\\textwidth]{fig_importance_default_shap.png}\n"
            "\\caption{ค่า mean $|$SHAP$|$ ของแต่ละแบบจำลองเมื่อวัดกับ"
            "การผิดนัดชำระจริง}\n"
            "\\label{fig:imp-default-shap}\n\\end{figure}\n")

    p = out("section_importance_default.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    return p


def main():
    print("=" * 96)
    print("Feature importance against the real default event, every tree model")
    print("=" * 96)
    panel, X, y, cols = cl.load_panel(verbose=True)

    print("\n  gain and SHAP per model ...")
    d, shap_store = gain_and_shap(panel, X, y, cols)
    d["pretty"] = d["feature"].map(lambda c: tm.PRETTY.get(c, c))

    perm = pd.DataFrame()
    if "--no-perm" not in sys.argv:
        print("\n  leave-one-issuer-out permutation importance per model ...")
        perm = perm_all(panel, X, y, cols)

    print("\n  figures ...")
    fig_compare(d, "gain", "Gain importance vs the real default event, by model",
                "fig_importance_default_gain.png")
    fig_compare(d, "mean_abs_shap",
                "mean |SHAP| vs the real default event, by model",
                "fig_importance_default_shap.png")
    if not perm.empty:
        pm = perm.copy()
        pm["pretty"] = pm["feature"].map(lambda c: tm.PRETTY.get(c, c))
        fig_compare(pm, "auc_drop",
                    "Out-of-sample permutation importance (AUC drop), by model",
                    "fig_importance_default_perm.png")
    fig_target_contrast(d)
    print("\n  per-model SHAP figures (bar, beeswarm, dependence) ...")
    per_model_figs(shap_store, cols)
    native_table(d)
    shap_table(d)

    p = write_tex(d, perm)
    d.to_csv(out("importance_default_event.csv"), index=False)
    if not perm.empty:
        perm.to_csv(out("importance_default_perm.csv"), index=False)
    con = sqlite3.connect(DB)
    d.to_sql("cmdf_importance_default", con, if_exists="replace", index=False)
    if not perm.empty:
        perm.to_sql("cmdf_importance_default_perm", con, if_exists="replace",
                    index=False)
    con.commit(); con.close()

    print("\n" + "=" * 96)
    print("TOP 5 PER MODEL, real default target")
    print("=" * 96)
    for m, g in d.groupby("model"):
        t = g.sort_values("gain", ascending=False).head(5)
        print(f"  gain  {m:15s} " +
              " | ".join(f"{r.pretty} {r.gain:.4f}" for r in t.itertuples()))
    if d["mean_abs_shap"].notna().any():
        print()
        for m, g in d.groupby("model"):
            t = g.sort_values("mean_abs_shap", ascending=False).head(5)
            print(f"  SHAP  {m:15s} " +
                  " | ".join(f"{r.pretty} {r.mean_abs_shap:.4f}"
                             for r in t.itertuples()))
    if not perm.empty:
        print("\n  ranked by out-of-sample permutation AUC drop (mean over models)")
        pm = perm.copy()
        pm["pretty"] = pm["feature"].map(lambda c: tm.PRETTY.get(c, c))
        agg = (pm.groupby("pretty")
               .agg(auc_drop=("auc_drop", "mean"), p_min=("p_value", "min"),
                    n_sig=("significant", "sum"))
               .sort_values("auc_drop", ascending=False).head(12))
        print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
        de = pm[pm["feature"].str.upper() == "DE"]
        if not de.empty:
            print(f"\n  D/E: mean AUC drop {de['auc_drop'].mean():+.4f}, "
                  f"lowest p {de['p_value'].min():.4f}, "
                  f"significant in {int(de['significant'].sum())} of "
                  f"{len(de)} models")
    print(f"\nwrote {p}")
    print(r"add with:  \input{section_importance_default.tex}")


if __name__ == "__main__":
    main()
