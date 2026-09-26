# -*- coding: utf-8 -*-
"""Three-panel PR-AUC figure for every method, in one picture.

Panel 1  the full out-of-fold precision-recall curve of each method, with the
         1 to 10 per cent workload points marked on it
Panel 2  precision against review workload
Panel 3  recall against review workload

The review workload is the share of the highest-risk issuer-months a team
would actually open, so the two right-hand panels are the operating view: at
a fixed amount of work, how much precision and how much recall does each
method buy.

Reads   out/pr_curves.npz          the curves, written by run_all_methods.py
        out/workload_sweep.csv     the 1-10 per cent operating points
        out/run_summary.csv        AP per method for the legend
Writes  out/pr_auc_all_methods.jpg
        out/pr_auc_all_methods.pdf

Run  python make_pr_figure.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "out"

# one colour per method, kept stable across every figure in the report
COLORS = {
    "XGBoost": "#d97706",
    "CatBoost": "#15803d",
    "LightGBM": "#0891b2",
    "HistGradientBoosting": "#7c3aed",
    "GradientBoosting": "#be123c",
    "RandomForest": "#1d4ed8",
    "ExtraTrees": "#0f766e",
    "Logistic": "#a16207",
    "Probit": "#9333ea",
}


def main() -> None:
    summary = pd.read_csv(OUT / "run_summary.csv")
    sweep = pd.read_csv(OUT / "workload_sweep.csv")
    curves = np.load(OUT / "pr_curves.npz", allow_pickle=True)

    models = [m for m in summary["model"] if f"{m}__precision" in curves]
    prevalence = float(sweep["prevalence"].iloc[0])

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.color": "#e6ebf3", "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), constrained_layout=True)

    for name in models:
        color = COLORS.get(name, "#64748b")
        precision = np.asarray(curves[f"{name}__precision"], dtype=float)
        recall = np.asarray(curves[f"{name}__recall"], dtype=float)
        if len(precision) > 5000:
            take = np.unique(np.linspace(0, len(precision) - 1, 5000).astype(int))
            precision, recall = precision[take], recall[take]
        ap = float(summary.loc[summary["model"].eq(name),
                               "average_precision_oof"].iloc[0])
        axes[0].plot(recall * 100.0, precision * 100.0, color=color,
                     linewidth=1.5, label=f"{name}: AP={ap:.4f}")
        pts = sweep.loc[sweep["model"].eq(name)].sort_values("review_workload_pct")
        axes[0].scatter(pts["recall"] * 100.0, pts["precision"] * 100.0,
                        color=color, edgecolor="white", linewidth=0.6,
                        s=26, zorder=3)
        axes[1].plot(pts["review_workload_pct"], pts["precision"] * 100.0,
                     color=color, marker="o", markersize=4, linewidth=1.4,
                     label=name)
        axes[2].plot(pts["review_workload_pct"], pts["recall"] * 100.0,
                     color=color, marker="o", markersize=4, linewidth=1.4,
                     label=name)

    axes[0].axhline(prevalence * 100.0, color="#94a3b8", linestyle="--",
                    linewidth=1.0,
                    label=f"Random baseline={prevalence*100:.4f}%")
    axes[0].set_xlabel("Recall (%)")
    axes[0].set_ylabel("Precision (%)")
    axes[0].set_title("Full out-of-fold precision-recall curve", weight="bold")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=7.5, loc="upper right")

    axes[1].axhline(prevalence * 100.0, color="#94a3b8", linestyle="--",
                    linewidth=1.0)
    axes[1].set_xlabel("Top-risk issuer-months reviewed (%)")
    axes[1].set_ylabel("Precision (%)")
    axes[1].set_title("Precision by review workload", weight="bold")
    axes[1].set_xticks(list(range(1, 11)))
    axes[1].legend(fontsize=7.5)

    axes[2].set_xlabel("Top-risk issuer-months reviewed (%)")
    axes[2].set_ylabel("Recall (%)")
    axes[2].set_title("Recall by review workload", weight="bold")
    axes[2].set_xticks(list(range(1, 11)))
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=7.5)

    fig.suptitle("Dataset2 OOF PR-AUC and operating thresholds "
                 "(1%-10% review workload)", weight="bold", fontsize=13)

    jpg = OUT / "pr_auc_all_methods.jpg"
    fig.savefig(jpg, dpi=200, format="jpg", bbox_inches="tight")
    fig.savefig(OUT / "pr_auc_all_methods.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"written {jpg}")
    print(f"written {OUT / 'pr_auc_all_methods.pdf'}")
    print(f"{len(models)} methods drawn: {', '.join(models)}")


if __name__ == "__main__":
    main()
