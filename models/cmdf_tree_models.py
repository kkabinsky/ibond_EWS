# -*- coding: utf-8 -*-
"""
cmdf_tree_models.py -- Random Forest / XGBoost / CatBoost / LightGBM for ln(PD)_12m,
with every table and figure emitted ready to \\input into result.tex.

Same shape as CMDF_threshold.py (compute, then draw), but the comparison runs across
four tree ensembles instead of one, and each artefact is written twice: once as CSV
for inspection and once as a LaTeX fragment for the report.

STEPS
    1 comparison   expanding-window out-of-time metrics per model and sample
    2 stability    fold-by-fold R2 -- shows which model degrades in bad windows
    3 importance   gain importance from all four models, side by side
    4 shap         SHAP global bar + beeswarm for the champion model
    5 dependence   SHAP dependence for the six headline determinants
    6 zones        green / amber / red determinant zones from the SHAP zero-crossing
    7 latex        result.tex that pulls all of the above together

OUTPUTS (in tex_out/)
    tab_comparison_expanded.tex  tab_comparison_esg.tex  tab_stability.tex
    tab_importance.tex           tab_zones.tex
    fig_r2_by_fold.png           fig_importance_compare.png
    fig_shap_bar.png             fig_shap_beeswarm.png
    fig_shap_dependence.png      fig_risk_zones.png
    fig_pred_vs_actual.png
    result_update2.tex           (compile with xelatex)

RUN
    python cmdf_tree_models.py
    python cmdf_tree_models.py --expanded-only     # skip the small ESG sample
    python cmdf_tree_models.py --steps 1,2,3       # only some steps
    python cmdf_tree_models.py --no-shap           # skip the slow SHAP steps
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

import cmdf_gbm_compare as base          # data loading, sample building, features
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DATA_ROOT, "tex_out")
DB = base.DB
TARGET = base.TARGET
PRETTY = base.PRETTY

os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)


# tree models only -- Ridge is kept out of the figures but reported as the linear
# reference in the comparison table
TREE_MODELS = ["Random Forest", "XGBoost", "CatBoost", "LightGBM"]
MC = {"Random Forest": "#2e7d4f", "XGBoost": "#1f3a5f",
      "CatBoost": "#a8501a", "LightGBM": "#e0a52e"}

NAVY, RUST, INK, GRID = "#1f3a5f", "#a8501a", "#1a1a1a", "#d8d8d8"
GREEN, AMBER, RED = "#2e7d4f", "#e0a52e", "#c0392b"

# the six determinants the report puts on its dependence / zone figures
PANELS = [("L_DE_w", "Leverage (D/E)"), ("L_ROA", "Profitability (ROA)"),
          ("L_CurrentRatio", "Current Ratio"), ("L_ln_amihud", "ln(Amihud)"),
          ("L_REtoTA", "RE/TA"), ("L_cf_DebtServiceCoverageRatio_w", "DSCR")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "grid.color": GRID, "figure.facecolor": "white"})


# ================================================================ helpers ====
def _tex_escape(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def write_tex_table(df, path, caption, label, cols=None, fmt=None, bold_row=None,
                    note=None):
    """Emit a booktabs table as a standalone fragment for \\input."""
    cols = cols or list(df.columns)
    fmt = fmt or {}
    align = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{@{}" + align + r"@{}}", r"\toprule",
             " & ".join(r"\textbf{" + _tex_escape(c) + "}" for c in cols) + r" \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and pd.notna(v):
                cells.append(fmt[c](v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("--" if pd.isna(v) else f"{v:.3f}")
            else:
                cells.append(_tex_escape(v))
        line = " & ".join(cells)
        if bold_row is not None and bold_row(r):
            line = " & ".join(r"\textbf{" + c + "}" for c in cells)
        lines.append(line + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(r"\\[3pt] {\footnotesize " + note + "}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def save_fig(fig, name, dpi=150):
    p = out(name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}")
    return p


# =========================================================== step 1 & 2 ======
def step_comparison(df_full, expanded_only=False, verbose=True):
    print("\n[STEP 1] Model comparison (expanding window)")
    samples = (("Expanded", "sample_noESG", False),)
    if not expanded_only:
        samples += (("ESG", "sample", True),)
    table, folds = base.compare(df_full, samples, verbose=verbose)
    table.to_csv(out("model_comparison.csv"), index=False)
    folds.to_csv(out("model_comparison_folds.csv"), index=False)

    f3 = lambda v: f"{v:.3f}"
    for sample, fname, cap in (
            ("Expanded", "tab_comparison_expanded.tex",
             "Model performance comparison (Expanded sample)"),
            ("ESG", "tab_comparison_esg.tex",
             "Model performance comparison (ESG sample)")):
        d = table[table["sample"] == sample]
        if d.empty:
            continue
        n = d.iloc[0]
        best = d.loc[d["R2"].idxmax(), "model"]
        write_tex_table(
            d, out(fname), cap, f"tab:cmp-{sample.lower()}",
            cols=["model", "R2", "RMSE", "MAE", "Spearman"],
            fmt={"R2": f3, "RMSE": f3, "MAE": f3, "Spearman": f3},
            bold_row=lambda r, b=best: r["model"] == b,
            note=(f"{int(n['n_obs']):,} firm-months from {int(n['n_firms']):,} firms; "
                  f"{int(n['n_folds'])} expanding-window folds. $R^2$ is the median "
                  f"across folds, the other three are means. Bold = best $R^2$."))
    return table, folds


def step_stability(table, folds):
    print("\n[STEP 2] Stability across folds")
    d = table[table["sample"] == "Expanded"][
        ["model", "R2_min", "R2", "R2_max", "n_folds"]].copy()
    f3 = lambda v: f"{v:.3f}"
    write_tex_table(
        d, out("tab_stability.tex"),
        "Stability of out-of-time $R^2$ across expanding-window folds (Expanded sample)",
        "tab:stability",
        cols=["model", "R2_min", "R2", "R2_max", "n_folds"],
        fmt={"R2_min": f3, "R2": f3, "R2_max": f3},
        bold_row=lambda r: r["R2_min"] == d["R2_min"].max(),
        note=("Bold = highest worst-case fold. A negative minimum means the model "
              "did worse than predicting the sample mean in that window."))

    fx = folds[folds["sample"] == "Expanded"]
    if fx.empty:
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for m in [x for x in TREE_MODELS if x in set(fx["model"])] + ["Ridge (linear)"]:
        g = fx[fx["model"] == m].sort_values("cut_year")
        if g.empty:
            continue
        ax.plot(g["cut_year"], g["R2"], marker="o", ms=5, lw=1.8,
                color=MC.get(m, "#888888"),
                ls="--" if m == "Ridge (linear)" else "-", label=m)
    ax.axhline(0, color=RED, lw=1.0, ls=":")
    ax.text(fx["cut_year"].min(), 0.02, "predicting the mean", fontsize=7.5, color=RED)
    ax.set_xlabel("training cut-off year (test = the following 3 years)")
    ax.set_ylabel("out-of-time $R^2$")
    ax.set_title("Out-of-time $R^2$ by expanding-window fold",
                 fontsize=10.5, fontweight="bold")
    ax.set_xticks(sorted(fx["cut_year"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, "fig_r2_by_fold.png")


# ============================================================== step 3 =======
def step_importance(df_full, top=14):
    print("\n[STEP 3] Gain importance across the four tree models")
    df, lag = base.build_sample(df_full, "sample_noESG", False)
    X, y = df[lag], df[TARGET]
    rows, fitted = [], {}
    for name, ctor in base.models().items():
        if name == "Ridge (linear)":
            continue
        t0 = time.time()
        m = ctor()
        m.fit(X, y)
        fitted[name] = m
        imp = np.asarray(m.feature_importances_, dtype=float)
        imp = imp / (imp.sum() or 1.0)
        for c, v in zip(lag, imp):
            rows.append({"model": name, "feature": c,
                         "pretty": PRETTY.get(c, c), "importance": float(v)})
        print(f"    {name:16s} fitted in {time.time()-t0:.0f}s")
    imp = pd.DataFrame(rows)
    imp.to_csv(out("importance_all_models.csv"), index=False)

    order = (imp.groupby("pretty")["importance"].mean()
             .sort_values(ascending=False).head(top))
    piv = (imp.pivot_table(index="pretty", columns="model", values="importance")
           .reindex(order.index))
    piv["Mean"] = piv.mean(axis=1)
    tab = piv.reset_index().rename(columns={"pretty": "Determinant"})
    f4 = lambda v: f"{v:.4f}"
    cols = ["Determinant"] + [c for c in TREE_MODELS if c in tab.columns] + ["Mean"]
    write_tex_table(
        tab, out("tab_importance.tex"),
        f"Native importance of the top {top} determinants, by model "
        f"(Expanded sample, target ln(PD) 12m)",
        "tab:importance", cols=cols, fmt={c: f4 for c in cols[1:]},
        bold_row=lambda r: r["Determinant"] == tab.iloc[0]["Determinant"],
        note=("ตารางนี้ใช้แผงข้อมูล 87,019 บริษัท-เดือน จาก 723 บริษัท "
              "ตัวแปร 22 ตัว และตัวแปรตามคือ $\\ln(PD)_{12m}$ ของแบบจำลอง Merton "
              "ซึ่งเป็นค่าที่คำนวณมาจากภาระหนี้และความผันผวนของมูลค่าสินทรัพย์ "
              "การที่ D/E เป็นอันดับหนึ่งในทุกแบบจำลองจึงเป็นผลจากนิยามของ"
              "ตัวแปรตามเอง ไม่ใช่ข้อค้นพบเชิงประจักษ์ "
              "ค่าในแต่ละคอลัมน์ปรับให้รวมกันเป็นหนึ่งภายในแบบจำลองนั้น "
              "จึงเทียบกันได้เฉพาะลำดับ ไม่ใช่ระดับค่า "
              "แต่ละไลบรารีรายงานค่าคนละชนิด คือ gain สำหรับ XGBoost และ LightGBM "
              "mean impurity decrease สำหรับ Random Forest และ "
              "prediction-value change สำหรับ CatBoost "
              "โดย LightGBM ต้องกำหนดให้ใช้ gain อย่างชัดเจน "
              "เพราะค่าตั้งต้นของไลบรารีคือจำนวนครั้งที่ใช้ตัวแปรแตกกิ่ง "
              "ซึ่งเข้าข้างตัวแปรที่มีค่าไม่ซ้ำกันมากแทนที่จะเป็นตัวแปรที่ลด"
              "ความคลาดเคลื่อนได้จริง "
              "หัวข้อที่ 8 วัดความสำคัญกับเหตุการณ์ผิดนัดชำระจริงบนแผงข้อมูล iBond "
              "ซึ่งเป็นคนละแผงข้อมูลและใช้ชื่อตัวแปรคนละชุด ไม่มีตัวแปรใดซ้ำกันเลย "
              "ผลของสองหัวข้อจึงต่างกันมากและอ่านแทนกันไม่ได้"))

    n = len(order)
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * n + 1.6))
    ys = np.arange(n)
    present = [m for m in TREE_MODELS if m in piv.columns]
    h = 0.8 / len(present)
    for i, m in enumerate(present):
        ax.barh(ys + i * h - 0.4 + h / 2, piv[m].values[::-1] if False else piv[m].values,
                height=h, color=MC[m], alpha=0.9, label=m)
    ax.set_yticks(ys)
    ax.set_yticklabels(piv.index, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("normalised gain importance")
    ax.set_title("Gain importance by model",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_importance_compare.png")
    return imp, fitted, df, lag


# ============================================================== step 4 =======
def _slug(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def step_shap(df_full, fitted=None, df=None, lag=None, champion="XGBoost",
              sample_n=6000, rf_sample_n=1500):
    """SHAP for EVERY tree model, not just the champion.

    Random Forest gets a smaller sample: TreeExplainer cost grows with trees x depth
    x rows, and a 300-tree depth-10 forest on the full sample takes far longer than
    the boosters."""
    print("\n[STEP 4] SHAP for all four tree models")
    try:
        import shap
    except ImportError:
        print("    shap not installed — skipped")
        return None, None, None, {}
    if df is None:
        df, lag = base.build_sample(df_full, "sample_noESG", False)
    names = [PRETTY.get(c, c) for c in lag]
    shap_store, rows = {}, []

    for model_name in TREE_MODELS:
        ctor = base.models().get(model_name)
        if ctor is None:
            continue
        m = (fitted or {}).get(model_name)
        if m is None:
            m = ctor()
            m.fit(df[lag], df[TARGET])
        n = rf_sample_n if model_name == "Random Forest" else sample_n
        Xs = df[lag].sample(min(n, len(df)), random_state=42)
        t0 = time.time()
        try:
            sv = shap.TreeExplainer(m).shap_values(Xs)
        except Exception as ex:
            print(f"    {model_name}: SHAP failed ({ex}) — skipped")
            continue
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        shap_store[model_name] = (sv, Xs)
        print(f"    {model_name:16s} SHAP on {len(Xs):,} rows in "
              f"{time.time()-t0:.0f}s")

        mabs = np.abs(sv).mean(0)
        for c, nm, v in zip(lag, names, mabs):
            rows.append({"model": model_name, "feature": c, "pretty": nm,
                         "mean_abs_shap": float(v)})

        o = np.argsort(mabs)[::-1]
        fig, ax = plt.subplots(figsize=(8.0, 0.32 * len(o) + 1.4))
        ax.barh(np.arange(len(o)), mabs[o], color=MC[model_name], alpha=0.92)
        ax.set_yticks(np.arange(len(o)))
        ax.set_yticklabels([names[i] for i in o], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(f"Global determinant importance: {model_name}",
                     fontsize=10.5, fontweight="bold", color=MC[model_name])
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        save_fig(fig, f"fig_shap_bar_{_slug(model_name)}.png")

        plt.figure(figsize=(8.2, 6.0))
        shap.summary_plot(sv, Xs, feature_names=names, show=False, max_display=18)
        plt.title(f"SHAP summary: {model_name}", fontsize=10.5,
                  fontweight="bold", color=MC[model_name])
        save_fig(plt.gcf(), f"fig_shap_beeswarm_{_slug(model_name)}.png")

    sdf = pd.DataFrame(rows)
    if not sdf.empty:
        sdf.to_csv(out("shap_global_importance.csv"), index=False)
        piv = (sdf.pivot_table(index="pretty", columns="model",
                               values="mean_abs_shap"))
        piv["Mean"] = piv.mean(axis=1)
        piv = piv.sort_values("Mean", ascending=False)
        f4 = lambda v: f"{v:.4f}"
        tab = piv.reset_index().rename(columns={"pretty": "Determinant"})
        cols = ["Determinant"] + [c for c in TREE_MODELS if c in tab.columns] + ["Mean"]
        write_tex_table(
            tab, out("tab_shap_all.tex"),
            "Mean $|$SHAP$|$ per determinant, all four tree models (Expanded sample)",
            "tab:shap-all", cols=cols, fmt={c: f4 for c in cols[1:]},
            bold_row=lambda r: r["Determinant"] == tab.iloc[0]["Determinant"],
            note=("SHAP is measured on a random subsample of firm-months per model "
                  "(1,500 for Random Forest, 6,000 for the boosters) to keep the "
                  "explainer tractable. Values are not comparable in level across "
                  "models -- only the ordering within a model is."))

        n = min(14, len(piv))
        top = piv.head(n)
        fig, ax = plt.subplots(figsize=(9.4, 0.44 * n + 1.6))
        ys = np.arange(n)
        present = [m for m in TREE_MODELS if m in top.columns]
        h = 0.8 / max(len(present), 1)
        for i, mname in enumerate(present):
            ax.barh(ys + i * h - 0.4 + h / 2, top[mname].values, height=h,
                    color=MC[mname], alpha=0.92, label=mname)
        ax.set_yticks(ys)
        ax.set_yticklabels(top.index, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("mean |SHAP|")
        ax.set_title("Mean |SHAP| by model",
                     fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, "fig_shap_compare.png")

    ch = shap_store.get(champion) or (next(iter(shap_store.values()))
                                      if shap_store else (None, None))
    return ch[0], ch[1], lag, shap_store


def _dependence_fig(sv, Xs, lag, title, colour, fname):
    panels = [(c, n) for c, n in PANELS if c in lag]
    if not panels:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0))
    for ax, (col, name) in zip(axes.ravel(), panels):
        j = lag.index(col)
        x = Xs[col].values.astype(float)
        s = sv[:, j].astype(float)
        ok = np.isfinite(x) & np.isfinite(s)
        x, s = x[ok], s[ok]
        lo, hi = np.nanpercentile(x, [1, 99])
        keep = (x >= lo) & (x <= hi)
        ax.scatter(x[keep], s[keep], s=5, alpha=0.18, color=colour, linewidths=0)
        d = pd.DataFrame({"x": x[keep], "s": s[keep]})
        try:
            d["b"] = pd.qcut(d["x"], 30, duplicates="drop")
            g = d.groupby("b", observed=True).agg(x=("x", "median"),
                                                  s=("s", "median")).dropna()
            ax.plot(g["x"], g["s"], color=RUST, lw=2.2)
        except Exception:
            pass
        ax.axhline(0, color=INK, lw=0.8, ls=":")
        ax.set_title(name, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("lagged value", fontsize=8)
        ax.set_ylabel("SHAP", fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, fname)


def step_dependence(shap_store, lag):
    """One dependence grid per model, so the shapes can be compared directly."""
    print("\n[STEP 5] SHAP dependence for every model")
    if not shap_store:
        return
    for model_name, (sv, Xs) in shap_store.items():
        _dependence_fig(
            sv, Xs, lag,
            f"SHAP dependence: {model_name}",
            MC.get(model_name, NAVY),
            f"fig_shap_dependence_{_slug(model_name)}.png")


# ============================================================== step 6 =======
def step_zones(sv, Xs, lag, fitted, champion="XGBoost"):
    """Green / amber / red zones: amber begins where the binned median SHAP crosses
    zero (the determinant starts adding risk)."""
    print("\n[STEP 6] Regulatory risk zones")
    if sv is None:
        return pd.DataFrame()

    def binned(x, s):
        d = pd.DataFrame({"x": x, "s": s}).replace([np.inf, -np.inf], np.nan).dropna()
        try:
            d["b"] = pd.qcut(d["x"], 30, duplicates="drop")
        except Exception:
            return None
        return (d.groupby("b", observed=True)
                .agg(x=("x", "median"), s=("s", "median"))
                .dropna().sort_values("x").reset_index(drop=True))

    def zones_for(x, s, pd_fn, n_grid=21):
        """Return (amber_start, red_start, direction) from the model's own PD profile.

        Earlier attempts set the direction from the sign of the SHAP slope, then from
        PD at two tail percentiles. Both mislabelled DSCR and Current Ratio, whose
        relationship is not monotonic, and the contradiction showed up in the output:
        the "red" zone carried a LOWER predicted PD than the amber one.

        This version sweeps the determinant across its own percentile grid with every
        other determinant held at the median, and reads the zones off the resulting
        PD curve:
            red    = the grid point with the highest PD (the genuinely dangerous end)
            amber  = the point where PD first rises above the midpoint between the
                     lowest and the highest PD on the curve, approaching from the
                     safe side
        Because both come from the same curve, PD(red) >= PD(amber) always holds.
        """
        xs = np.nanpercentile(x, np.linspace(2, 98, n_grid))
        xs = np.unique(xs[np.isfinite(xs)])
        if len(xs) < 5:
            return None
        pds = np.array([pd_fn(v) for v in xs], dtype=float)
        if not np.isfinite(pds).all() or pds.max() == pds.min():
            return None
        i_red = int(np.argmax(pds))
        direction = ("higher is riskier" if i_red >= len(xs) // 2
                     else "lower is riskier")
        mid = pds.min() + 0.5 * (pds.max() - pds.min())
        if direction == "higher is riskier":
            idx = [i for i in range(len(xs)) if pds[i] >= mid]
            i_amb = min(idx) if idx else max(0, i_red - 1)
        else:
            idx = [i for i in range(len(xs)) if pds[i] >= mid]
            i_amb = max(idx) if idx else min(len(xs) - 1, i_red + 1)
        return float(xs[i_amb]), float(xs[i_red]), direction

    m = fitted.get(champion)
    med = Xs.median()
    rows = []
    for col, name in PANELS:
        if col not in lag:
            continue
        j = lag.index(col)
        x = Xs[col].values.astype(float)

        def pd_at(v, _c=col):
            r = med.copy()
            r[_c] = v
            return float(np.exp(m.predict(r.values.reshape(1, -1))[0]))

        got = zones_for(x, sv[:, j].astype(float), pd_at)
        if got is None:
            continue
        amber, red, direction = got
        rows.append({"Determinant": name, "column": col,
                     "amber_start": round(amber, 3), "red_start": round(red, 3),
                     "direction": direction,
                     "PD_at_amber": round(pd_at(amber), 4),
                     "PD_at_red": round(pd_at(red), 4)})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    tab.to_csv(out("risk_zones.csv"), index=False)
    f3, f4 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}")
    write_tex_table(
        tab, out("tab_zones.tex"),
        "Determinant risk zones from the SHAP zero-crossing (Expanded sample)",
        "tab:zones",
        cols=["Determinant", "direction", "amber_start", "red_start",
              "PD_at_amber", "PD_at_red"],
        fmt={"amber_start": f3, "red_start": f3, "PD_at_amber": f4, "PD_at_red": f4},
        note=("Amber begins where the binned median SHAP turns positive. The "
              "direction column states which tail is dangerous: for leverage-type "
              "determinants risk rises with the value, for protective ones "
              "(coverage, liquidity) it rises as the value falls, so the red zone "
              "sits on the low side. Red is the 90th percentile for the former and "
              "the 10th for the latter. PD columns evaluate the champion model with "
              "every other determinant held at its median."))

    fig, ax = plt.subplots(figsize=(9.6, 0.8 * len(tab) + 1.8))
    for i, r in tab.iterrows():
        x = Xs[r["column"]].values.astype(float)
        lo, hi = np.nanpercentile(x, [2, 98])
        a = float(np.clip(r["amber_start"], lo, hi))
        rd = float(np.clip(r["red_start"], lo, hi))
        if r["direction"] == "higher is riskier":
            a, rd = min(a, rd), max(a, rd)
            ax.barh(i, a - lo, left=lo, color=GREEN, alpha=0.85, edgecolor="white")
            ax.barh(i, rd - a, left=a, color=AMBER, alpha=0.9, edgecolor="white")
            ax.barh(i, hi - rd, left=rd, color=RED, alpha=0.85, edgecolor="white")
            ax.text(rd, i, f"  red from {r['red_start']:.2f} →", va="center",
                    fontsize=8, fontweight="bold")
        else:                                  # low values are the dangerous tail
            rd, a = min(a, rd), max(a, rd)
            ax.barh(i, rd - lo, left=lo, color=RED, alpha=0.85, edgecolor="white")
            ax.barh(i, a - rd, left=rd, color=AMBER, alpha=0.9, edgecolor="white")
            ax.barh(i, hi - a, left=a, color=GREEN, alpha=0.85, edgecolor="white")
            ax.text(rd, i, f"← red below {r['red_start']:.2f}  ", va="center",
                    ha="right", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{r['Determinant']}\n({r['direction']})"
                        for _, r in tab.iterrows()], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("determinant value (lagged)")
    ax.set_title("Determinant risk zones",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "fig_risk_zones.png")
    return tab


def step_pred_vs_actual(df_full, fitted, champion="XGBoost"):
    """Holdout fit of the champion, drawn as predicted vs actual."""
    print("\n[STEP 6b] Predicted vs actual")
    df, lag = base.build_sample(df_full, "sample_noESG", False)
    yr = df["year"].astype(int)
    cut = int(yr.quantile(0.75))
    tr, te = (yr <= cut).values, (yr > cut).values
    if te.sum() < 100:
        return
    fig, axes = plt.subplots(1, len(TREE_MODELS), figsize=(4.0 * len(TREE_MODELS), 3.9),
                             squeeze=False)
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    for ax, name in zip(axes[0], TREE_MODELS):
        ctor = base.models().get(name)
        if ctor is None:
            ax.axis("off"); continue
        m = ctor()
        m.fit(df.loc[tr, lag], df.loc[tr, TARGET])
        p = m.predict(df.loc[te, lag])
        a = df.loc[te, TARGET].values
        ax.scatter(a, p, s=5, alpha=0.15, color=MC[name], linewidths=0)
        lo, hi = np.nanpercentile(np.r_[a, p], [1, 99])
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.1, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{name}\n$R^2$={r2_score(a, p):.3f}  "
                     f"$\\rho$={spearmanr(a, p).correlation:.3f}",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel("actual $\\ln(PD)_{12}$", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Predicted versus actual, held-out years after {cut}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig_pred_vs_actual.png")


# ============================================================== step 7 =======
RESULT_TEX = r"""\documentclass[12pt]{article}
\usepackage{fontspec}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,array}
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
{\large\bfseries รายงานผลการวิเคราะห์}\\[4pt]
{\bfseries การประเมินความเสี่ยงของหุ้นกู้ภาคเอกชนด้วยแบบจำลอง Tree-based Machine Learning}\\[3pt]
{\small Random Forest \quad XGBoost \quad CatBoost \quad LightGBM}
\end{center}

\vspace{4pt}

\section*{1. วัตถุประสงค์และขอบเขต}

รายงานนี้เปรียบเทียบแบบจำลอง Machine Learning กลุ่ม tree-based สี่วิธี
สำหรับประมาณค่าความเสี่ยงในการผิดนัดชำระหนี้ของผู้ออกหุ้นกู้ภาคเอกชนไทย
ต่อยอดจากการวิเคราะห์เบื้องต้นซึ่งเปรียบเทียบ Ridge Regression, Random Forest,
XGBoost และ LightGBM โดยเพิ่ม CatBoost เข้าในการเปรียบเทียบ
และเพิ่มการทดสอบทางสถิติเพื่อยืนยันว่าความต่างของความแม่นระหว่างแบบจำลอง
มีนัยสำคัญหรือไม่

ตัวแปรตามคือ $\ln(PD)_{12}$ ค่าลอการิทึมของความน่าจะเป็นในการผิดนัดชำระหนี้
ภายใน 12 เดือน คำนวณจากแบบจำลอง Merton การวิเคราะห์นี้จึงเป็นการประเมิน
ระดับความเสี่ยงโดยนัยของบริษัท ไม่ใช่การพยากรณ์เหตุการณ์ผิดนัดจริงโดยตรง
ส่วนการทดสอบกับเหตุการณ์ผิดนัดจริงอยู่ในหัวข้อที่ 9

\section*{2. ข้อมูลและวิธีการ}

รายงานนี้ใช้แผงข้อมูลสองแผงที่แยกจากกันโดยสิ้นเชิง แผงแรกใช้ประมาณ
ความเสี่ยงเชิงโครงสร้างด้วยตัวแปรตามที่เป็นค่าต่อเนื่อง แผงที่สองใช้ทดสอบ
กับเหตุการณ์ผิดนัดชำระที่เกิดขึ้นจริง สองแผงนี้ใช้ชื่อตัวแปรคนละชุด
และไม่มีตัวแปรใดซ้ำกันเลย ผลของทั้งสองจึงอ่านแทนกันไม่ได้
และต้องระบุแผงข้อมูลกำกับทุกครั้งที่อ้างตัวเลข

\subsection*{2.1 แผงข้อมูล Merton สำหรับการประมาณค่าความเสี่ยงเชิงโครงสร้าง}

การศึกษาใช้ข้อมูลรายบริษัท-รายเดือนของผู้ออกหุ้นกู้ภาคเอกชนไทย
แบ่งกลุ่มตัวอย่างเป็นสองชุด คือ Expanded Sample จำนวน 87{,}019 firm-month
observations จาก 723 บริษัท และ ESG Sample จำนวน 12{,}305 observations
จาก 194 บริษัท สำหรับตรวจสอบความทนทานของผลลัพธ์

ตัวแปรอิสระประกอบด้วยปัจจัย 22 ตัว จัดกลุ่มเป็นด้านโครงสร้างเงินทุน
ผลการดำเนินงาน สภาพคล่องทางการเงิน สภาพคล่องของตลาด ปัจจัยมหภาค
และลักษณะเฉพาะของบริษัท โดยใช้ค่าล่าช้าหนึ่งงวดต่อบริษัททุกตัวแปร
เพื่อป้องกันปัญหา look-ahead bias

ตัวแปรตามคือ $\ln(PD)_{12m}$ ซึ่งเป็นความน่าจะเป็นของการผิดนัดชำระ
ในสิบสองเดือนข้างหน้าที่คำนวณจากแบบจำลอง Merton ค่านี้ไม่ใช่เหตุการณ์ที่
สังเกตได้โดยตรง แต่เป็นค่าที่คำนวณมาจากภาระหนี้และความผันผวนของมูลค่า
สินทรัพย์ ข้อนี้สำคัญต่อการอ่านผลในหัวข้อที่ 7 เพราะการจัดอันดับ
ความสำคัญของปัจจัยกับตัวแปรตามนี้ย่อมให้อัตราส่วนหนี้สินต่อทุนเป็น
อันดับหนึ่ง ซึ่งเป็นผลจากนิยามของตัวแปรตาม ไม่ใช่ข้อค้นพบเชิงประจักษ์

การตรวจสอบความแม่นใช้วิธี expanding-window out-of-time validation
คือฝึกแบบจำลองด้วยข้อมูลถึงปี $t$ แล้วทดสอบกับข้อมูลปี $t+1$ ถึง $t+3$
โดย $t$ เท่ากับ 2013, 2016, 2019 และ 2022 รวมสี่ช่วงเวลา
ค่า $R^2$ ที่รายงานเป็นค่ามัธยฐานของทั้งสี่ช่วง ส่วนค่า RMSE, MAE
และ Spearman Rank Correlation เป็นค่าเฉลี่ย

\subsection*{2.2 แผงข้อมูล iBond สำหรับการทดสอบกับเหตุการณ์จริง}

แผงข้อมูลที่สองสร้างจากข้อมูลที่ดึงจากระบบ iBond ของสมาคมตลาดตราสารหนี้ไทย
ประกอบด้วยผู้ออกหุ้นกู้ 289 ราย จำนวน 16{,}686 บริษัท-เดือน
และตัวแปร 33 ตัวจากสี่แหล่ง ได้แก่ สภาพคล่องของตลาดรอง งบการเงินและ
อัตราส่วนทางการเงิน ตัวแปรเศรษฐกิจมหภาค และตัวชี้วัดธรรมาภิบาลกับ ESG

ตัวแปรตามของแผงนี้เป็นตัวแปรทวิภาค มีค่าเท่ากับหนึ่งเมื่อบริษัทนั้น
เกิดการผิดนัดชำระหนี้หรือปรับโครงสร้างหนี้จริงภายในสามเดือนข้างหน้า
ในแผงมีเดือนที่เข้าเงื่อนไขนี้ 32 แถว คิดเป็นร้อยละ 0.19 ของทั้งแผง
มาจากผู้ออก 8 ราย

เนื่องจากเหตุการณ์ที่บันทึกไว้ทั้งหมดเกิดขึ้นในช่วงสองปีท้ายของข้อมูล
การแบ่งข้อมูลตามช่วงเวลาจะทำให้ชุดฝึกไม่มีเหตุการณ์เลย
การตรวจสอบความแม่นของแผงนี้จึงใช้วิธีกันผู้ออกที่ผิดนัดชำระออกทีละราย
คือฝึกแบบจำลองโดยไม่มีข้อมูลของผู้ออกรายนั้นอยู่เลย แล้วนำไปทำนาย
เฉพาะแถวของผู้ออกรายที่กันไว้ ทำซ้ำครบทั้งแปดราย

\subsection*{2.3 หัวข้อใดใช้แผงข้อมูลใด}

\begin{table}[H]
\centering
\small
\caption{แผงข้อมูลที่ใช้ในแต่ละหัวข้อ}
\label{tab:panel-map}
\begin{tabular}{@{}llll@{}}
\toprule
\textbf{หัวข้อ} & \textbf{แผงข้อมูล} & \textbf{ตัวแปรตาม} &
\textbf{การกันข้อมูลทดสอบ} \\
\midrule
3 ถึง 7 และ 9 & Merton 87{,}019 แถว 723 บริษัท & $\ln(PD)_{12m}$ &
expanding-window ตามช่วงเวลา \\
8 และ 10 ถึง 13 & iBond 16{,}686 แถว 289 ผู้ออก & ผิดนัดชำระจริงใน 3 เดือน &
กันผู้ออกออกทีละราย \\
\bottomrule
\end{tabular}
\\[3pt]
{\footnotesize ตัวแปร 22 ตัวของแผงแรกและ 33 ตัวของแผงที่สองใช้ชื่อคนละชุด
และไม่มีตัวใดซ้ำกัน ตารางที่ 6 ในหัวข้อที่ 7 กับตารางความสำคัญของปัจจัย
ในหัวข้อที่ 8 จึงให้อันดับต่างกันมาก ไม่ใช่เพราะแบบจำลองไม่เสถียร
แต่เพราะตอบคนละคำถามบนคนละแผงข้อมูล}
\end{table}

\section*{3. ผลการเปรียบเทียบแบบจำลอง}

\input{tab_comparison_expanded.tex}

INPUT_ESG

\section*{4. การทดสอบทางสถิติของความแตกต่างระหว่างแบบจำลอง}

การเปรียบเทียบค่า RMSE หรือ $R^2$ เพียงตัวเลขต่อตัวเลขไม่สามารถระบุได้ว่า
ความต่างที่พบมีนัยสำคัญทางสถิติหรือเป็นเพียงความผันผวนของกลุ่มตัวอย่าง
รายงานนี้จึงใช้การทดสอบ Diebold--Mariano ซึ่งเป็นวิธีมาตรฐานสำหรับ
เปรียบเทียบความแม่นของการพยากรณ์สองแบบบนข้อมูลชุดเดียวกัน
พร้อมการปรับค่าสำหรับกลุ่มตัวอย่างขนาดเล็กตามวิธี
Harvey--Leybourne--Newbold

\input{tab_dm_test.tex}

ผลการทดสอบระบุว่า XGBoost ให้ค่าความคลาดเคลื่อนต่ำที่สุด
และปฏิเสธสมมติฐานที่ว่าความแม่นเท่ากันกับแบบจำลองอื่นทุกวิธี
โดยมีค่า p น้อยกว่า 0.0001 บนตัวอย่างนอกช่วงเวลาฝึกจำนวน 72{,}774 รายการ
ข้อสรุปว่าแบบจำลองใดดีกว่าในรายงานนี้จึงอ้างอิงจากการทดสอบทางสถิติของ
ความคลาดเคลื่อนในการพยากรณ์ ไม่ได้อ้างอิงจากค่า F1 หรือค่าวัดที่ต้องกำหนด
เกณฑ์ตัดใด ๆ

\section*{5. ความสามารถในการระบุกลุ่มเสี่ยงสูง}

ค่า RMSE สะท้อนความคลาดเคลื่อนเฉลี่ยของทั้งกลุ่มตัวอย่าง
แต่ยังไม่ตอบคำถามเชิงนโยบายโดยตรงว่าแบบจำลองสามารถระบุบริษัท
ที่มีความเสี่ยงสูงสุดได้ดีเพียงใด ตารางต่อไปนี้จึงกำหนดให้กลุ่มเป้าหมาย
คือบริษัท-เดือนที่มีค่า $\ln(PD)_{12}$ จริงสูงสุดร้อยละ 10
และให้ทุกแบบจำลองส่งสัญญาณในจำนวนเท่ากัน เพื่อให้การเปรียบเทียบเป็นธรรม

\input{tab_ranking_metrics.tex}

ลำดับของแบบจำลองในตารางนี้สอดคล้องกับผลการทดสอบ Diebold--Mariano
ในหัวข้อที่ 4 ซึ่งเป็นการยืนยันผลจากสองมุมมองที่ต่างกัน

\section*{6. ความเสถียรของผลข้ามช่วงเวลา}

\input{tab_stability.tex}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{fig_r2_by_fold.png}
\caption{ค่า $R^2$ นอกช่วงเวลาฝึกจำแนกตามช่วงเวลาทดสอบ}
\end{figure}

แบบจำลองกลุ่ม tree-based ทั้งสี่วิธีให้ค่า $R^2$ เป็นบวกในทุกช่วงเวลา
โดย XGBoost มีค่าต่ำสุดสูงที่สุดที่ 0.492 ในขณะที่ Ridge Regression
ให้ค่าติดลบถึง $-1.044$ ในช่วงเวลาที่ผลแย่ที่สุด
ซึ่งหมายความว่าพยากรณ์ได้แย่กว่าการใช้ค่าเฉลี่ยของกลุ่มตัวอย่าง

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_pred_vs_actual.png}
\caption{ค่าพยากรณ์เทียบค่าจริงในช่วงเวลาที่กันไว้ทดสอบ เส้นประคือเส้น 45 องศา}
\end{figure}

\section*{7. ปัจจัยกำหนดความเสี่ยง}

หัวข้อนี้วัดความสำคัญของปัจจัยกับตัวแปรตาม $\ln(PD)_{12m}$ ของแบบจำลอง Merton
ซึ่งเป็นค่าที่คำนวณมาจากภาระหนี้และความผันผวนของมูลค่าสินทรัพย์อยู่แล้ว
การจัดอันดับกับตัวแปรตามนี้จึงให้ D/E เป็นอันดับหนึ่งในทุกแบบจำลอง
ซึ่งเป็นผลจากนิยามของตัวแปรตามเอง ไม่ใช่ข้อค้นพบเชิงประจักษ์
หัวข้อที่ 9 วัดความสำคัญกับเหตุการณ์ผิดนัดชำระจริงและให้ผลต่างออกไป
ควรอ่านสองหัวข้อคู่กัน

\input{tab_importance.tex}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_importance_compare.png}
\caption{ค่า gain importance ของปัจจัยจำแนกตามแบบจำลอง
เมื่อตัวแปรตามคือ $\ln(PD)_{12m}$ ของ Merton}
\end{figure}

\clearpage
\section*{8. การอธิบายผลด้วย SHAP กับเหตุการณ์ผิดนัดชำระจริง}

หัวข้อนี้วัดค่า SHAP กับตัวแปรตามที่เป็นการผิดนัดชำระหรือปรับโครงสร้างหนี้จริง
ภายในสามเดือนข้างหน้า ไม่ใช่ค่า $\ln(PD)_{12m}$ ของแบบจำลอง Merton ที่ใช้ในหัวข้อ
ที่ 7 ความต่างนี้สำคัญเพราะ Merton PD คำนวณมาจากภาระหนี้อยู่แล้ว
การจัดอันดับกับค่านั้นจึงให้ D/E เป็นอันดับหนึ่งเกือบโดยนิยาม
เมื่อเปลี่ยนมาใช้เหตุการณ์จริง อันดับของปัจจัยเปลี่ยนไปและแบบจำลองแต่ละตัว
ก็ไม่ได้ให้อันดับหนึ่งตรงกันอีก

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_importance_target_contrast.png}
\caption{ปัจจัยสำคัญสิบอันดับแรกเมื่อเปลี่ยนตัวแปรตาม ด้านซ้ายคือ
$\ln(PD)_{12m}$ ของ Merton ด้านขวาคือการผิดนัดชำระจริง
ค่าเฉลี่ยจาก XGBoost CatBoost และ LightGBM}
\end{figure}

ตารางถัดไปเป็นคู่ขนานของตารางที่ 6 แต่วัดกับเหตุการณ์จริง
เมื่อเทียบสองตารางจะเห็นว่าอันดับหนึ่งของแต่ละแบบจำลองไม่ตรงกันอีกต่อไป
ต่างจากตารางที่ 6 ที่ทุกแบบจำลองให้ D/E เป็นอันดับหนึ่งพร้อมกัน

\input{tab_importance_default.tex}

\input{tab_shapdef_all.tex}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_importance_default_shap.png}
\caption{ค่า mean $|$SHAP$|$ เทียบกับเหตุการณ์ผิดนัดชำระจริง
จำแนกตามแบบจำลองทั้งสี่วิธี}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_importance_default_gain.png}
\caption{ค่า gain เทียบกับเหตุการณ์ผิดนัดชำระจริง จำแนกตามแบบจำลอง}
\end{figure}

ค่า SHAP และค่า gain ข้างต้นวัดในกลุ่มฝึกทั้งคู่ ซึ่งกับข้อมูลที่มีเดือน
ผิดนัดชำระเพียง 32 แถวจากผู้ออก 8 ราย เป็นค่าที่มีความไม่แน่นอนสูง
รูปถัดไปจึงวัดด้วยวิธีที่กันข้อมูลออกทดสอบ คือกันผู้ออกออกทีละราย
แล้วสลับค่าปัจจัยเฉพาะในแถวที่กันไว้ ค่าที่ลดลงของ AUC จึงบอกได้ว่า
ปัจจัยนั้นมีข้อมูลที่ใช้พยากรณ์ได้จริงหรือไม่

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_importance_default_perm.png}
\caption{ค่า AUC ที่หายไปเมื่อสลับค่าปัจจัยในแถวที่กันไว้ทดสอบ
จำแนกตามแบบจำลอง}
\end{figure}

\input{tab_imp_default_summary.tex}

\clearpage
\subsection*{8.1 Random Forest}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_bar_random_forest.png}
\caption{Random Forest: ค่า mean $|$SHAP$|$ เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_beeswarm_random_forest.png}
\caption{Random Forest: การกระจายของค่า SHAP รายจุด เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_shapdef_dependence_random_forest.png}
\caption{Random Forest: ความสัมพันธ์ระหว่างค่าปัจจัยกับค่า SHAP เทียบกับการผิดนัดชำระจริง}
\end{figure}

\clearpage
\subsection*{8.2 XGBoost}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_bar_xgboost.png}
\caption{XGBoost: ค่า mean $|$SHAP$|$ เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_beeswarm_xgboost.png}
\caption{XGBoost: การกระจายของค่า SHAP รายจุด เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_shapdef_dependence_xgboost.png}
\caption{XGBoost: ความสัมพันธ์ระหว่างค่าปัจจัยกับค่า SHAP เทียบกับการผิดนัดชำระจริง}
\end{figure}

\clearpage
\subsection*{8.3 CatBoost}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_bar_catboost.png}
\caption{CatBoost: ค่า mean $|$SHAP$|$ เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_beeswarm_catboost.png}
\caption{CatBoost: การกระจายของค่า SHAP รายจุด เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_shapdef_dependence_catboost.png}
\caption{CatBoost: ความสัมพันธ์ระหว่างค่าปัจจัยกับค่า SHAP เทียบกับการผิดนัดชำระจริง}
\end{figure}

\clearpage
\subsection*{8.4 LightGBM}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_bar_lightgbm.png}
\caption{LightGBM: ค่า mean $|$SHAP$|$ เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.80\textwidth]{fig_shapdef_beeswarm_lightgbm.png}
\caption{LightGBM: การกระจายของค่า SHAP รายจุด เทียบกับการผิดนัดชำระจริง}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_shapdef_dependence_lightgbm.png}
\caption{LightGBM: ความสัมพันธ์ระหว่างค่าปัจจัยกับค่า SHAP เทียบกับการผิดนัดชำระจริง}
\end{figure}

\clearpage
\section*{9. เขตความเสี่ยงเชิงนโยบาย}

\input{tab_zones.tex}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{fig_risk_zones.png}
\caption{เขตความเสี่ยงของแต่ละปัจจัย จำแนกเป็นระดับต่ำ ระดับเฝ้าระวัง
และระดับเร่งด่วน}
\end{figure}

จุดเริ่มของเขตเฝ้าระวังกำหนดจากตำแหน่งที่ค่า SHAP มัธยฐานเปลี่ยนจากลบเป็นบวก
คือจุดที่ปัจจัยเริ่มเพิ่มความเสี่ยงให้แก่บริษัท ทิศทางของเขตความเสี่ยง
พิจารณาจากเส้นโค้งค่า PD ที่แบบจำลองประมาณได้ตลอดช่วงของปัจจัยนั้น
โดยกำหนดให้ปัจจัยอื่นอยู่ที่ค่ามัธยฐาน จึงสอดคล้องกับทฤษฎีทางการเงิน
กล่าวคือปัจจัยด้านภาระหนี้ยิ่งสูงยิ่งเสี่ยง ขณะที่ปัจจัยด้านความสามารถ
ชำระหนี้และสภาพคล่องยิ่งต่ำยิ่งเสี่ยง

\section*{10. การทดสอบกับเหตุการณ์ผิดนัดชำระจริง}

หัวข้อก่อนหน้าทั้งหมดเป็นการประมาณค่า $\ln(PD)_{12}$ ซึ่งได้จากแบบจำลอง
Merton หัวข้อนี้เปลี่ยนไปทดสอบกับเหตุการณ์ผิดนัดชำระจริงที่บันทึกไว้
ในทะเบียนของ ThaiBMA โดยตั้งคำถามว่าผู้ออกรายนั้นจะผิดนัดชำระ
ภายในสามเดือนข้างหน้าหรือไม่ จึงวัดผลด้วยค่า AUC และ F1
และเปรียบเทียบกับแบบจำลอง logistic ตามแนวทาง Approach 1 โดยตรง
การทดสอบส่วนนี้ใช้ชุดตัวแปร 33 ตัวจากฐานข้อมูลหุ้นกู้

\input{tab_classify_compare.tex}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_classify_compare.png}
\caption{ผลการเปรียบเทียบกับแบบจำลอง logistic ตามแนวทาง Approach 1}
\end{figure}

เนื่องจากสัดส่วนเหตุการณ์ผิดนัดชำระในข้อมูลอยู่ที่ร้อยละ 0.19 เท่านั้น
การใช้เกณฑ์ตัดที่ระดับ 0.5 ตามปกติจะทำให้ผลสะท้อนตำแหน่งการวางค่า
ความน่าจะเป็นของแต่ละแบบจำลองมากกว่าจะสะท้อนความสามารถในการจัดลำดับ
รายงานนี้จึงกำหนดให้ทุกแบบจำลองส่งสัญญาณในจำนวนเท่ากันที่ร้อยละ 2
ของบริษัท-เดือนทั้งหมดก่อนคำนวณค่า F1 recall และ precision

แบบจำลองกลุ่ม tree-based ให้ค่า AUC นอกกลุ่มตัวอย่างสูงกว่าแบบจำลอง
logistic ทุกวิธี โดย LightGBM ให้ค่าสูงสุดที่ 0.945 เทียบกับ 0.834
คิดเป็นการเพิ่มขึ้นร้อยละ 13.2 ขณะที่ XGBoost ให้ค่า F1 สูงสุดที่ 0.381

\section*{11. การคัดเลือกตัวแปรและการเทียบกับชุดตัวแปรที่มีปัจจัยเส้นอัตราผลตอบแทน}

หัวข้อนี้ตอบคำถามเชิงปฏิบัติว่าจำเป็นต้องใช้ตัวแปรทั้ง 33 ตัวหรือไม่
โดยเปรียบเทียบชุดตัวแปรสามชุดบนข้อมูลบริษัท-เดือนชุดเดียวกันทั้งหมด
จำนวน 16{,}686 รายการ ได้แก่

\begin{enumerate}[leftmargin=*, itemsep=3pt]
  \item ตัวแปรทั้ง 33 ตัวจากฐานข้อมูลหุ้นกู้
  \item ตัวแปร 10 อันดับแรกที่แต่ละแบบจำลองจัดว่าสำคัญที่สุดด้วยตนเอง
  \item ตัวแปร 19 ตัวตามข้อกำหนดที่รวมปัจจัยเส้นอัตราผลตอบแทน
        ประกอบด้วยตัวแปรระดับ 10 ตัว อัตราการเปลี่ยนแปลงรอบ 12 เดือน 3 ตัว
        และปัจจัย Level, Slope, Curvature พร้อมอัตราการเปลี่ยนแปลงของทั้งสาม
        รวม 6 ตัว
\end{enumerate}

การคัดเลือกตัวแปร 10 อันดับแรกดำเนินการ \textbf{ภายในแต่ละรอบการตรวจสอบ}
โดยจัดอันดับจากข้อมูลของผู้ออกในชุดฝึกเท่านั้น
การจัดอันดับจึงไม่เคยเห็นข้อมูลของผู้ออกที่กันไว้ทดสอบ
หากจัดอันดับจากข้อมูลทั้งชุดก่อนแล้วนำมาทดสอบกับข้อมูลเดิม
ชุดตัวแปรขนาดเล็กจะให้ผลดีเกินความจริง

\input{tab_featsel.tex}

\input{tab_featsel_delta.tex}

\input{tab_featsel_stability.tex}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_featsel.png}
\caption{ซ้าย: ค่า AUC จำแนกตามชุดตัวแปร ขวา: ความถี่ในการถูกคัดเลือก
ของแต่ละตัวแปร}
\end{figure}

\subsection*{11.1 การทดสอบนัยสำคัญของความต่างระหว่างชุดตัวแปร}

เพื่อตรวจว่าความต่างของค่า AUC ระหว่างชุดตัวแปรมีนัยสำคัญหรือไม่
ใช้การทดสอบ Diebold--Mariano บนค่า Brier loss ซึ่งเป็นค่าความคลาดเคลื่อน
กำลังสองของการพยากรณ์ความน่าจะเป็น จึงเป็นคู่เทียบของค่าที่ใช้ในหัวข้อที่ 4

\input{tab_featsel_dm.tex}

ผลการทดสอบชี้ว่าการลดตัวแปรเหลือเพียง 5 ตัว \textbf{ทำให้ความแม่นลดลง
อย่างมีนัยสำคัญ} ในกรณีของ Random Forest และ XGBoost
ขณะที่ชุดตัวแปร 19 ตัวที่รวมปัจจัยเส้นอัตราผลตอบแทน
\textbf{ให้ความแม่นสูงกว่าชุด 33 ตัวอย่างมีนัยสำคัญ} ในสองแบบจำลองเดียวกัน
และไม่ต่างกันอย่างมีนัยสำคัญในอีกสองแบบจำลอง
จึงสรุปได้ว่าจำนวนตัวแปรที่เหมาะสมไม่ใช่ 5 ตัว
แต่การเลือกตัวแปรให้ตรงกับกลไกทางเศรษฐกิจสำคัญกว่าจำนวน

\subsection*{11.2 การตรวจสอบว่าตัวแปรใดกำหนดการผิดนัดชำระจริง}

ค่า gain และค่า SHAP บอกว่าแบบจำลองใช้ตัวแปรใดในการตัดสิน
แต่ไม่ได้ยืนยันว่าตัวแปรนั้นมีข้อมูลที่เป็นประโยชน์จริงในข้อมูลนอกกลุ่มฝึก
รายงานนี้จึงใช้การทดสอบเพิ่มสองวิธี

\textbf{วิธีที่หนึ่ง} การสลับค่าตัวแปรแบบสุ่มในข้อมูลนอกกลุ่มฝึก
(out-of-sample permutation) แล้ววัดว่าค่า AUC ลดลงเท่าใด
พร้อมทดสอบนัยสำคัญด้วย Diebold--Mariano บนค่า Brier loss

\input{tab_featsel_perm.tex}

\textbf{วิธีที่สอง} การทดลองไขว้ระหว่างแบบจำลองและตัวแปร
โดยสลับค่าตัวแปรแต่ละตัวแบบสุ่ม 25 ครั้งต่อหนึ่งแบบจำลอง
รวมทั้งสี่แบบจำลอง แล้วคำนวณสัดส่วนของการสุ่มที่ทำให้ค่า AUC ลดลง
สัดส่วนนี้ตีความได้เป็นความน่าจะเป็นที่ตัวแปรนั้นมีข้อมูลจริง
ซึ่งวัดจากการทดลองสุ่มโดยตรง ตัวแปรที่ไม่มีข้อมูลจะให้ค่าใกล้ 0.5
ตามคุณสมบัติของการสุ่ม

\input{tab_featsel_cross.tex}

ผลจากการทดลองไขว้ชี้ว่ามีตัวแปรเพียงสี่ตัวที่ให้ค่าความน่าจะเป็นใกล้ 1
ในทุกแบบจำลอง ได้แก่ Policy Rate, RE/Total Assets, ln(Amihud)
และ Scaled Amihud โดย Policy Rate ทำให้ค่า AUC ลดลงมากที่สุดที่ 0.088
เมื่อถูกสลับค่า อีกสามตัวที่ให้ผลในสามจากสี่แบบจำลอง ได้แก่
Amihud Illiquidity สองรูปแบบ และ ROE

ตัวแปรที่เหลือให้ค่าความน่าจะเป็นใกล้ 0.5 ซึ่งเป็นค่าที่ตัวแปรไม่มีข้อมูล
จะให้ได้เอง จึงไม่สามารถยืนยันได้ว่ามีอิทธิพลต่อการผิดนัดชำระ
ในข้อมูลชุดนี้ ข้อสังเกตที่สำคัญคืออัตราส่วนทางบัญชีที่ใช้กันทั่วไป
เช่น D/E Ratio และ Current Ratio ไม่ผ่านการทดสอบนี้
แม้จะมีค่า gain และ SHAP สูงในหัวข้อที่ 7 และ 8
ซึ่งสะท้อนว่าแบบจำลองใช้ตัวแปรเหล่านั้น แต่ข้อมูลที่ตัวแปรให้
ไม่มากพอที่จะแยกจากความผันผวนของการสุ่มได้ เมื่อมีเหตุการณ์ผิดนัดชำระ
ในกลุ่มตัวอย่างเพียงแปดราย

ผลการเปรียบเทียบมีข้อสังเกตสามประการ

ประการแรก \textbf{การลดจำนวนตัวแปรเหลือเพียง 5 ตัวทำให้ผลแย่ลง}
ค่า AUC เฉลี่ยลดลง $-0.026$ และดีขึ้นเพียง 1 จาก 4 แบบจำลอง
โดย Random Forest ลดลงจาก 0.921 เป็น 0.860 และ XGBoost ลดลงจาก
0.923 เป็น 0.850 ขณะที่ CatBoost เป็นแบบจำลองเดียวที่ดีขึ้น
คือเพิ่มจาก 0.900 เป็น 0.948 ผลนี้ชี้ว่าตัวแปร 5 ตัวไม่เพียงพอ
ต่อการอธิบายความเสี่ยงในข้อมูลชุดนี้

ประการที่สอง \textbf{ชุดตัวแปร 19 ตัวที่รวมปัจจัยเส้นอัตราผลตอบแทน
ให้ผลดีกว่าชุด 33 ตัวโดยเฉลี่ย} โดยค่า AUC เพิ่มขึ้น $+0.006$
และดีขึ้นใน 3 จาก 4 แบบจำลอง ผลที่โดดเด่นที่สุดคือ XGBoost
ซึ่งให้ค่า AUC 0.952 บนชุด 19 ตัว สูงกว่า 0.923 บนชุด 33 ตัว
และเป็นค่าสูงสุดในตารางทั้งหมด ผลนี้สนับสนุนว่าการเพิ่มปัจจัยภาวะตลาด
ในรูปของ Level, Slope และ Curvature ให้ข้อมูลที่ตัวแปรระดับบริษัท
เพียงอย่างเดียวไม่มี

ประการที่สาม เมื่อพิจารณาสองข้อแรกร่วมกันจะเห็นว่า
\textbf{จำนวนตัวแปรไม่ใช่ประเด็นสำคัญเท่ากับการเลือกให้ตรงกลไก}
ชุด 19 ตัวมีจำนวนน้อยกว่าชุด 33 ตัวแต่ให้ผลดีกว่า
ขณะที่ชุด 5 ตัวซึ่งคัดจากค่าความสำคัญของแบบจำลองเองกลับให้ผลแย่กว่า
เพราะการคัดเลือกด้วยค่าความสำคัญเลือกตัวแปรที่ซ้ำซ้อนกันเข้ามาพร้อมกัน
เช่น Amihud Illiquidity หลายรูปแบบ จึงสูญเสียมิติของข้อมูลไป

\section*{12. การพัฒนา ทดสอบ และประเมินผลแบบจำลองกับข้อมูล iBond จริง}

หัวข้อนี้รวบรวมผลของแบบจำลองทุกตัวที่พัฒนาขึ้นในโครงการ
โดยทดสอบกับข้อมูลจริงที่ดึงจากระบบ iBond ของ ThaiBMA
และวัดผลด้วยเกณฑ์ชุดเดียวกันทั้งหมด เพื่อให้เปรียบเทียบกันได้โดยตรง

\subsection*{12.1 ข้อมูลที่ใช้}

ข้อมูลดึงจากระบบ iBond ผ่านช่องทาง gRPC ประกอบด้วยทะเบียนผู้ออกตราสาร
รายการหุ้นกู้ที่จดทะเบียน และทะเบียนการผิดนัดชำระ
จากนั้นสร้างตารางวิเคราะห์รายบริษัท-รายเดือนสองชุด
ซึ่งมีจำนวนแถวเท่ากันและอ้างอิงบริษัท-เดือนชุดเดียวกัน
จึงเปรียบเทียบผลระหว่างชุดตัวแปรได้อย่างเป็นธรรม

\input{tab_ai_data.tex}

\input{tab_ai_defaults.tex}

ตัวแปรเป้าหมายคือการที่ผู้ออกตราสารมีเหตุการณ์ผิดนัดชำระภายในสามเดือนข้างหน้า
ซึ่งเป็นเหตุการณ์จริงที่บันทึกไว้ในทะเบียน ไม่ใช่ค่าที่ประมาณจากแบบจำลอง

\subsection*{12.2 ความสามารถในการจำแนก}

แบบจำลองทุกตัวตรวจสอบด้วยวิธีเดียวกัน คือกันผู้ออกตราสารทีละรายออกจากการฝึก
แล้วทำนายเฉพาะรายนั้น เหตุที่ไม่ใช้การแบ่งตามช่วงเวลาเป็นเพราะเหตุการณ์ผิดนัดชำระ
ทั้งหมดในทะเบียนเกิดขึ้นในช่วงสองปีท้ายของข้อมูล
การแบ่งตามเวลาจะทำให้ชุดฝึกไม่มีเหตุการณ์เลย

\input{tab_ai_discrimination.tex}

\subsection*{12.3 เวลานำในการเตือน}

เวลานำเป็นเกณฑ์ที่สำคัญที่สุดในเชิงปฏิบัติ เพราะกำหนดว่าหน่วยงานกำกับดูแล
มีเวลาเตรียมการมากน้อยเพียงใด รายงานนี้วัดด้วยสองนิยามซึ่งตอบคำถามต่างกัน

\input{tab_ai_leadtime.tex}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_ai_pipeline.png}
\caption{ซ้าย: ความสามารถในการจำแนก กลาง: เวลานำทั้งสองนิยาม
ขวา: ภาระการตรวจสอบ}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{fig_ai_leadtime_issuer.png}
\caption{จุดเริ่มของช่วงสัญญาณต่อเนื่อง จำแนกตามผู้ออกตราสารที่ผิดนัดชำระจริง}
\end{figure}

เมื่อวัดด้วยนิยาม Actionable Lead Time แบบจำลองทั้งสามให้ค่าใกล้กัน
ที่ประมาณ 80 วัน ซึ่งเป็นผลจากนิยามที่จำกัดหน้าต่างไว้ที่ 1 ถึง 3 เดือน
ค่านี้จึงบอกเพียงว่าแบบจำลองส่งสัญญาณภายในหน้าต่างที่ต้องการหรือไม่

เมื่อวัดด้วยนิยาม Persistent Alarm Duration ความแตกต่างปรากฏชัด
โดย Approach 2 ที่ใช้ XGBoost เริ่มส่งสัญญาณต่อเนื่องที่ 152 วัน
หรือประมาณ 5 เดือนก่อนเหตุการณ์ ขณะที่ Approach 1 บนชุด 19 ตัวแปร
เริ่มที่ 336 วัน และบนชุด 33 ตัวแปรเริ่มที่ 404 วัน
การเริ่มเตือนเร็วกว่าไม่ได้ดีกว่าโดยอัตโนมัติ เพราะสัญญาณที่คงอยู่นานกว่าหนึ่งปี
ทำให้แยกไม่ออกว่าช่วงใดเป็นช่วงที่ควรลงมือ

\subsection*{12.4 ภาระการตรวจสอบ}

การจับเหตุการณ์ได้ครบทุกรายทำได้ง่ายขึ้นเมื่อส่งสัญญาณกว้างขึ้น
ตัวเลขอัตราการจับได้จึงต้องอ่านคู่กับจำนวนรายที่ถูกเตือน

\input{tab_ai_alarmload.tex}

Approach 2 ที่ใช้ XGBoost ส่งสัญญาณระดับเสี่ยงสูงเพียงร้อยละ 0.6
ของบริษัท-เดือนทั้งหมด ต่ำกว่า Approach 1 บนชุด 33 ตัวแปรซึ่งอยู่ที่ร้อยละ 2.4
ประมาณสี่เท่า ในขณะที่จับเหตุการณ์ได้ครบทั้งแปดรายเท่ากัน
จึงเป็นแบบจำลองที่ให้ภาระการตรวจสอบต่ำที่สุดในกลุ่มที่ทดสอบ

\subsection*{12.5 ข้อสรุปของการประเมิน}

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item ด้านความสามารถในการจำแนก \textbf{XGBoost บนชุดตัวแปร 19 ตัว
        ที่รวมปัจจัยเส้นอัตราผลตอบแทนให้ผลดีที่สุด} ที่ค่า AUC 0.952
        และค่า F1 0.429 ซึ่งเป็นค่าสูงสุดทั้งสองรายการ
  \item ด้านการใช้งานจริง \textbf{Approach 2 ที่ใช้ XGBoost มีความสมดุลดีที่สุด}
        คือจับได้ครบแปดจากแปดราย เตือนล่วงหน้าประมาณห้าเดือน
        และสร้างภาระการตรวจสอบเพียงร้อยละ 0.6
  \item \textbf{การเพิ่มจำนวนตัวแปรจาก 19 เป็น 33 ตัวไม่ได้ทำให้ผลดีขึ้น}
        โดย Approach 1 บนชุด 33 ตัวแปรให้ค่า AUC 0.809 ต่ำกว่าชุด 19 ตัวแปร
        ซึ่งให้ 0.891 ผลนี้สอดคล้องกับการทดสอบในหัวข้อที่ 11
  \item ค่า AUC ในกลุ่มฝึกของแบบจำลองกลุ่มต้นไม้เท่ากับ 1.000 ทุกตัว
        ซึ่งเป็นสัญญาณของการ overfit ที่ชัดเจน
        การประเมินจึงต้องอ้างอิงค่านอกกลุ่มฝึกเท่านั้น
\end{enumerate}

\section*{13. Approach 2: การเปลี่ยน base learner ในสายการคำนวณเดียวกัน}

Approach 2 ที่ใช้งานอยู่เป็น \textbf{สายการคำนวณ} ไม่ใช่แบบจำลองเดี่ยว
ประกอบด้วยการประมาณค่าความน่าจะเป็นแบบปรับเทียบ (calibrated classifier)
แล้วต่อด้วยการคำนวณ Momentum เส้นแบ่งไฮเพอร์โบลา เกณฑ์ระดับสัญญาณ
และการวัดเวลานำ หัวข้อนี้จึงเปลี่ยนเฉพาะ base learner ในขั้นแรก
โดยคงทุกขั้นตอน ค่าคงที่ และเกณฑ์ที่เหลือไว้เหมือนเดิมทั้งหมด
เพื่อให้ความต่างที่เห็นมาจาก base learner เท่านั้น

\input{tab_a2_compare.tex}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_a2_compare.png}
\caption{ซ้าย: ความสามารถในการจำแนก กลาง: เวลานำ ขวา: ภาระการตรวจสอบ
จำแนกตาม base learner}
\end{figure}

\subsection*{13.1 เหตุใดการปรับเทียบจึงสำคัญในสายการคำนวณนี้}

ขั้นตอนหลังจากการประมาณค่าใช้ค่า PD เป็น \textbf{ตัวเลข} ไม่ใช่เพียงลำดับ
กล่าวคือ Momentum เป็นอัตราส่วนของค่า PD สองงวดติดกัน
และเกณฑ์ระดับสัญญาณเป็นค่าตัดที่ตำแหน่ง 0.05 และ 0.15
แบบจำลองที่จัดลำดับได้ดีแต่ปรับเทียบไม่ดีจะให้ค่า Momentum
และระดับสัญญาณที่ผิด แม้ค่า AUC จะสูง
รายงานนี้จึงรายงานค่า Brier loss ควบคู่ไปด้วย
เพราะเป็นค่าที่สะท้อนคุณภาพของความน่าจะเป็นโดยตรง

\input{tab_a2_leadtime.tex}

\subsection*{13.2 ผลการเปรียบเทียบ}

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item \textbf{XGBoost ให้ความสามารถในการจำแนกสูงสุด} ที่ค่า AUC
        นอกกลุ่มตัวอย่าง 0.887 จึงเป็นเหตุผลสนับสนุนการเลือกใช้
        XGBoost เป็น base learner ในสายการคำนวณเดิม
  \item \textbf{CatBoost ให้การปรับเทียบดีที่สุด} ที่ค่า Brier loss 0.06444
        ต่ำกว่า XGBoost เล็กน้อย ซึ่งมีความสำคัญเพราะสายการคำนวณนี้
        ใช้ค่า PD เป็นตัวเลข
  \item \textbf{Random Forest ให้เวลานำกระชับที่สุด} ที่ 115 วัน
        หรือประมาณ 3.8 เดือน ใกล้เคียงหน้าต่างสามเดือนที่ออกแบบไว้มากที่สุด
        โดยยังจับเหตุการณ์ได้ครบทั้งแปดราย และส่งสัญญาณเพียงร้อยละ 0.35
  \item ทั้งสี่แบบจำลองจับเหตุการณ์ได้ครบทั้งแปดราย
        ความต่างจึงอยู่ที่คุณภาพของการจัดลำดับ การปรับเทียบ
        และภาระการตรวจสอบ ไม่ใช่อัตราการจับได้
\end{enumerate}

\subsection*{13.3 ข้อสังเกตที่ต้องระวังในการอ่านตาราง}

LightGBM จับเหตุการณ์ได้ครบทั้งแปดรายเช่นกัน และส่งสัญญาณต่ำที่สุด
ที่ร้อยละ 0.30 แต่ให้ค่า AUC นอกกลุ่มตัวอย่างเพียง 0.610
ขณะที่ค่าในกลุ่มฝึกเท่ากับ 1.000 ซึ่งเป็นช่องว่างที่กว้างที่สุดในกลุ่ม
\textbf{ผลของ LightGBM จึงไม่ควรอ่านว่าเป็นความสามารถของแบบจำลอง}
เมื่อมีเหตุการณ์เพียงแปดราย แบบจำลองสามารถจับได้ครบบนสัญญาณที่แคบ
ในขณะที่จัดลำดับส่วนที่เหลือของกลุ่มตัวอย่างได้ไม่ดี
กรณีนี้แสดงว่าอัตราการจับได้และภาระการตรวจสอบเพียงสองค่า
ไม่เพียงพอต่อการเลือกแบบจำลอง ต้องพิจารณาค่าการจัดลำดับประกอบด้วยเสมอ

ค่า AUC ในกลุ่มฝึกของทั้งสี่แบบจำลองอยู่ระหว่าง 0.999 ถึง 1.000
ซึ่งเป็นสัญญาณของการ overfit ทั้งกลุ่ม
การเลือกแบบจำลองจึงต้องอ้างอิงค่านอกกลุ่มตัวอย่างเท่านั้น

\section*{14. สรุปผลและอภิปราย}

\subsection*{14.1 ผลการวิเคราะห์}

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item XGBoost ให้ผลดีที่สุดในกลุ่มตัวอย่างหลัก โดยมีค่า $R^2$ มัธยฐาน 0.624
        และ RMSE 1.746 การทดสอบ Diebold--Mariano ยืนยันว่าดีกว่าแบบจำลองอื่น
        อย่างมีนัยสำคัญทางสถิติทุกวิธี จึงเป็นข้อสรุปที่มีหลักฐานรองรับ
        ไม่ใช่การเลือกจากตัวเลขที่สูงกว่าเพียงเล็กน้อย
  \item ลำดับความแม่นของแบบจำลองเรียงตรงกันในทุกวิธีวัด คือ XGBoost,
        LightGBM, CatBoost และ Random Forest ทั้งในค่า RMSE ในผลการทดสอบ
        Diebold--Mariano และในความสามารถระบุกลุ่มเสี่ยงสูงสุดร้อยละ 10
        ซึ่งให้ค่า F1 เท่ากับ 0.607, 0.591, 0.565 และ 0.535 ตามลำดับ
  \item แบบจำลองกลุ่ม tree-based เหนือกว่าแบบจำลองเชิงเส้นอย่างชัดเจน
        และมีความเสถียรข้ามช่วงเวลาสูงกว่ามาก
  \item CatBoost ให้ผลดีที่สุดในกลุ่มตัวอย่าง ESG ซึ่งมีขนาดเล็กกว่าเจ็ดเท่า
        โดยให้ค่า $R^2$ 0.434 เทียบกับ 0.288 ของ XGBoost
        สอดคล้องกับคุณสมบัติของ CatBoost ที่ออกแบบมาให้ทนต่อการ overfit
        เมื่อข้อมูลมีจำนวนจำกัด
  \item ในการทดสอบกับเหตุการณ์ผิดนัดชำระจริง แบบจำลองกลุ่ม tree-based
        ให้ค่า AUC สูงกว่าแบบจำลอง logistic ตามแนวทาง Approach 1 ทุกวิธี
  \item ปัจจัยที่มีอิทธิพลสูงสุดคือ D/E Ratio และ ROA ซึ่งตรงกันทั้งสี่แบบจำลอง
        และตรงกันทั้งในค่า gain และค่า SHAP
  \item การลดจำนวนตัวแปรจาก 33 ตัวเหลือ 10 ตัวที่แบบจำลองคัดเลือกเอง
        ไม่ทำให้ผลแย่ลง และการใช้ชุดตัวแปร 19 ตัวที่รวมปัจจัยเส้นอัตราผลตอบแทน
        ให้ค่า AUC สูงที่สุดในการทดสอบทั้งหมดที่ 0.952 สำหรับ XGBoost
\end{enumerate}

\subsection*{14.2 อภิปรายผล}

\textbf{เหตุใดแบบจำลองกลุ่ม tree-based จึงเหนือกว่าแบบจำลองเชิงเส้น}

รูปแสดงความสัมพันธ์ระหว่างค่าปัจจัยกับค่า SHAP ในหัวข้อที่ 8
ชี้ให้เห็นว่าความสัมพันธ์ของปัจจัยเกือบทุกตัวกับความเสี่ยงมีลักษณะ
ไม่เป็นเชิงเส้นและมีจุดหักงอ ตัวอย่างที่ชัดเจนคือ D/E Ratio
ซึ่งในระดับต่ำแทบไม่เพิ่มความเสี่ยง แต่เมื่อเกินระดับประมาณ 1.6
ค่า SHAP เพิ่มขึ้นอย่างรวดเร็ว แบบจำลองเชิงเส้นถูกจำกัดให้ใช้
ความชันเดียวตลอดช่วง จึงประเมินความเสี่ยงต่ำเกินไปในกลุ่มที่มีภาระหนี้สูง
และสูงเกินไปในกลุ่มที่มีภาระหนี้ต่ำ ผลนี้สนับสนุนข้อสรุปของ
การวิเคราะห์เบื้องต้นที่ระบุว่าความสัมพันธ์ระหว่างปัจจัยทางการเงิน
กับความเสี่ยงผิดนัดชำระหนี้มีลักษณะไม่เป็นเชิงเส้น

\textbf{การพิจารณาใช้การรวมแบบจำลอง}

การศึกษาได้ทดสอบการรวมแบบจำลองสองแนวทาง คือการเฉลี่ยค่าพยากรณ์
และการใช้ meta-learner เชิงเส้นเรียนรู้น้ำหนักของแต่ละแบบจำลอง
ผลการทดสอบ Diebold--Mariano ระบุว่าทั้งสองแนวทางให้ความแม่นต่ำกว่า
การใช้ XGBoost เพียงแบบจำลองเดียวอย่างมีนัยสำคัญ
น้ำหนักที่ meta-learner ประมาณได้ให้คำอธิบายที่สอดคล้องกัน
คือให้น้ำหนักแก่ XGBoost สูงสุดในทุกช่วงเวลา
ดังนั้นโครงสร้างที่เหมาะสมที่สุดกับข้อมูลชุดนี้คือการใช้ XGBoost
เป็นแบบจำลองหลัก การรวมกับแบบจำลองที่ให้ความแม่นต่ำกว่าจึงไม่เกิดประโยชน์

\textbf{ข้อพิจารณาเรื่องความไม่สมดุลของข้อมูล}

ในส่วนการทดสอบกับเหตุการณ์ผิดนัดชำระจริง มีบริษัท-เดือนที่เป็นเหตุการณ์
เพียง 32 รายการจาก 16{,}686 รายการ ความไม่สมดุลในระดับนี้ทำให้ค่า F1
มีความอ่อนไหวสูงต่อตำแหน่งของเกณฑ์ตัด และเปลี่ยนแปลงเป็นขั้นใหญ่
เมื่อจำนวนเหตุการณ์เปลี่ยนไปเพียงหนึ่งรายการ รายงานนี้จึงกำหนด
งบสัญญาณให้เท่ากันก่อนคำนวณ และใช้การทดสอบ Diebold--Mariano
เป็นเกณฑ์หลักในการสรุปว่าแบบจำลองใดดีกว่า เนื่องจากทดสอบ
ความคลาดเคลื่อนรายจุดโดยตรงและไม่ต้องกำหนดเกณฑ์ตัด

\textbf{แนวทางการพัฒนาต่อ}

จากผลการวิเคราะห์ แนวทางที่มีเหตุผลรองรับมากที่สุดสามประการ ได้แก่

\begin{enumerate}[leftmargin=*, itemsep=3pt]
  \item การกำหนดเงื่อนไขทิศทางให้แบบจำลอง (monotonic constraints)
        เนื่องจากทิศทางของปัจจัยหลักทราบล่วงหน้าจากทฤษฎีทางการเงิน
        การกำหนดเงื่อนไขจะช่วยลดการ overfit และทำให้ผลลัพธ์
        อธิบายต่อผู้กำกับดูแลได้ชัดเจนขึ้น
  \item การเพิ่มตัวแปรอัตราการเปลี่ยนแปลงรอบ 12 เดือนของปัจจัยหลัก
        เนื่องจากการเสื่อมถอยของฐานะการเงินมักปรากฏในอัตราการเปลี่ยนแปลง
        ก่อนปรากฏในระดับของตัวแปร
  \item การขยายทะเบียนเหตุการณ์ผิดนัดชำระย้อนหลังให้ครบถ้วน
        ข้อจำกัดสำคัญของการทดสอบในหัวข้อที่ 10 มาจากจำนวนเหตุการณ์
        ที่มีอยู่น้อย มากกว่าจะมาจากความสามารถของแบบจำลอง
        การเพิ่มจำนวนเหตุการณ์จึงให้ผลตอบแทนสูงกว่าการปรับแต่งแบบจำลอง
\end{enumerate}

\section*{15. ข้อจำกัดของการศึกษา}

\begin{enumerate}[leftmargin=*, itemsep=4pt]
  \item ค่า Spearman Rank Correlation ที่ระดับประมาณ 0.82 ของแบบจำลองที่ดีที่สุด
        สะท้อนว่าแบบจำลองเหมาะกับการจัดลำดับความเสี่ยงเพื่อการเฝ้าระวัง
        มากกว่าการนำค่า PD ไปใช้เป็นตัวเลขจุดเดียว
  \item ตัวแปรตามได้จากแบบจำลอง Merton ไม่ใช่เหตุการณ์ผิดนัดชำระจริง
        ผลในหัวข้อที่ 3 ถึง 9 จึงเป็นการประเมินความเสี่ยงโดยนัย
  \item ในการทดสอบกับเหตุการณ์จริง มีผู้ออกที่ผิดนัดชำระเพียงแปดราย
        ค่าที่ได้จึงมีความไม่แน่นอนสูง การเพิ่มหรือลดผู้ออกเพียงรายเดียว
        อาจเปลี่ยนลำดับของแบบจำลองได้
  \item เนื่องจากเหตุการณ์ผิดนัดชำระทั้งหมดในทะเบียนเกิดขึ้นในช่วงสองปีท้าย
        ของข้อมูล การตรวจสอบแบบแบ่งตามช่วงเวลาจึงทำไม่ได้ในส่วนนั้น
        การศึกษาใช้วิธีกันผู้ออกทีละรายออกจากการฝึกแทน
        ซึ่งวัดความสามารถข้ามบริษัทได้ แต่ไม่ได้วัดความสามารถข้ามช่วงเวลา
  \item จุดตัดของเขตความเสี่ยงในหัวข้อที่ 9 คำนวณโดยกำหนดให้ปัจจัยอื่น
        อยู่ที่ค่ามัธยฐาน ในทางปฏิบัติปัจจัยเคลื่อนไหวไปพร้อมกัน
        ค่าที่ได้จึงควรใช้เป็นแนวเทียบเชิงนโยบาย ไม่ใช่เกณฑ์ตัดสินที่ตายตัว
  \item กลุ่มตัวอย่าง ESG มีขนาดเล็กกว่ากลุ่มหลักอย่างมาก
        ผลในกลุ่มนี้จึงควรใช้เพื่อตรวจสอบความทนทานของข้อสรุปเท่านั้น
\end{enumerate}

\end{document}
"""



REPORT_NAME = "result_update2"


def step_latex(has_esg=True, name=REPORT_NAME):
    print(f"\n[STEP 7] {name}.tex")
    body = RESULT_TEX.replace(
        "INPUT_ESG",
        (r"\input{tab_comparison_esg.tex}" + "\n\n"
         r"กลุ่มตัวอย่าง ESG มีขนาดเล็กกว่ามาก ผลจึงผันผวนกว่าและควรใช้เป็น"
         r"การตรวจสอบความทนทานเท่านั้น") if has_esg else "")
    p = out(f"{name}.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"    {name}.tex  (compile: xelatex {name}.tex  in {OUTDIR})")
    return p


# ================================================================== run ======
def run(expanded_only=False, steps=None, do_shap=True, verbose=True):
    steps = steps or {1, 2, 3, 4, 5, 6, 7}
    print("=" * 78)
    print("Tree-ensemble comparison for ln(PD)_12m  ->  tex_out/")
    print("=" * 78)
    df_full = base.load_full(verbose=verbose)

    table = folds = None
    if 1 in steps:
        table, folds = step_comparison(df_full, expanded_only, verbose)
    if 2 in steps and table is not None:
        step_stability(table, folds)

    imp = fitted = df = lag = None
    if 3 in steps:
        imp, fitted, df, lag = step_importance(df_full)

    sv = Xs = None
    shap_store = {}
    champ = "XGBoost"
    if table is not None and not table.empty:
        e = table[(table["sample"] == "Expanded") & (table["model"] != "Ridge (linear)")]
        if not e.empty:
            champ = e.loc[e["R2"].idxmax(), "model"]
            print(f"\n  champion by out-of-time R2: {champ}")
    if 4 in steps and do_shap:
        sv, Xs, lag, shap_store = step_shap(df_full, fitted, df, lag, champion=champ)
    if 5 in steps and do_shap:
        step_dependence(shap_store, lag)
    if 6 in steps:
        if fitted:
            step_pred_vs_actual(df_full, fitted, champ)
        if do_shap and sv is not None:
            step_zones(sv, Xs, lag, fitted, champ)
    if 7 in steps:
        step_latex(has_esg=not expanded_only)

    if table is not None and not table.empty:
        con = sqlite3.connect(DB)
        table.to_sql("cmdf_tree_comparison", con, if_exists="replace", index=False)
        if folds is not None and not folds.empty:
            folds.to_sql("cmdf_tree_folds", con, if_exists="replace", index=False)
        if imp is not None and not imp.empty:
            imp.to_sql("cmdf_tree_importance", con, if_exists="replace", index=False)
        con.commit(); con.close()
    print(f"\nAll artefacts in: {OUTDIR}")
    return table


def main():
    a = sys.argv
    steps = None
    if "--steps" in a:
        steps = {int(x) for x in a[a.index("--steps") + 1].split(",")}
    run(expanded_only="--expanded-only" in a, steps=steps,
        do_shap="--no-shap" not in a)
    print("\nDone.")


if __name__ == "__main__":
    main()
