# -*- coding: utf-8 -*-
"""ThaiBMA Credit Early Warning System — Dataset2 edition.

Same shell and the same panels as app.py, driven by Dataset2 instead of the
synthetic/33-feature Excel path:

    Loading Dataset2 from ../cmdf_credit.db
    Rows 187,007 | issuers 941 | positive months 124 | explicit event issuers 31
    | all dated/status event issuers 32

This folder is self-contained. It reads the SQLite file in the folder above and
nothing else, and it imports only data_layer.py from beside it.

Run  python app_dataset2.py            interactive GUI
     python app_dataset2.py --selftest headless backend check
     python app_dataset2.py --uitest   headless UI build check
"""
from __future__ import annotations

import base64
import io
import sys
import threading
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_layer as DL

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcff",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6ebf3",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1e293b",
    "axes.labelsize": 8.5,
    "axes.labelcolor": "#334155",
    "font.size": 9,
    "text.color": "#334155",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 120,
})

UI = {
    "page": "#f0f7ff",
    "surface": "#f8fafc",
    "sidebar": "#e0f2fe",
    "sidebar_panel": "#f0f9ff",
    "button": "#dbeafe",
    "primary": "#1e40af",
    "primary_dark": "#0f172a",
    "accent": "#2563eb",
    "text": "#0f172a",
    "muted": "#475569",
    "border": "#bfdbfe",
}
NAV_BUTTON_WIDTH = 272
NAV_BUTTON_HEIGHT = 44
BAND_COLOR = {"HIGH RISK": "#dc2626", "ELEVATED": "#ea580c",
              "WATCH": "#f59e0b", "OK": "#16a34a"}


def _b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _uri(b64):
    return "data:image/png;base64," + b64


# =============================================================== figures
def fig_pd_distribution(pd_values, threshold):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    finite = np.asarray(pd_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    ax.hist(finite, bins=60, color="#2563eb", alpha=0.85)
    ax.axvline(threshold, color="#dc2626", lw=1.4, ls="--",
               label=f"alarm threshold {threshold:.4f}")
    ax.set_xlabel("out-of-fold PD over the 1-3 month window")
    ax.set_ylabel("issuer-months")
    ax.set_title("Risk distribution, all issuer-months")
    ax.legend()
    return _b64(fig)


def fig_importance(imp: pd.DataFrame, top=15):
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.barh(range(len(s)), s["importance"].to_numpy(), color="#2563eb")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["feature"].tolist(), fontsize=8)
    ax.set_xlabel("model importance")
    ax.set_title(f"Feature importance, top {min(top, len(imp))} of {len(imp)}")
    return _b64(fig)


def fig_roe_vs_pd(cross: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    if "ROE" in cross.columns:
        x = pd.to_numeric(cross["ROE"], errors="coerce").to_numpy()
        y = pd.to_numeric(cross["pd_3m"], errors="coerce").to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        colors = [BAND_COLOR.get(b, "#64748b") for b in cross["band"].to_numpy()[ok]]
        ax.scatter(x[ok], y[ok], s=14, c=colors, alpha=0.8, edgecolors="none")
        lo, hi = np.nanpercentile(x[ok], [1, 99]) if ok.any() else (0, 1)
        ax.set_xlim(lo, hi)
    ax.set_xlabel("ROE")
    ax.set_ylabel("PD over the 1-3 month window")
    ax.set_title("ROE against credit risk, latest month of each issuer")
    return _b64(fig)


def fig_momentum(frame: pd.DataFrame, issuer: str):
    fig, ax = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True)
    t = frame["month_dt"]
    ax[0].plot(t, frame["pd_3m"], color="#1e40af", lw=1.3)
    alarms = frame[frame["alarm"].fillna(False).astype(bool)]
    if not alarms.empty:
        ax[0].scatter(alarms["month_dt"], alarms["pd_3m"], s=18,
                      color="#dc2626", zorder=3, label="alarm month")
        ax[0].legend()
    ax[0].set_ylabel("PD")
    ax[0].set_title(f"{issuer}: PD path and risk momentum")
    ax[1].axhline(0.0, color="#94a3b8", lw=0.8)
    ax[1].plot(t, frame["momentum_3m"], color="#b45309", lw=1.2)
    ax[1].set_ylabel("M(t), 3-month change")
    ax[1].set_xlabel("month")
    fig.autofmt_xdate()
    return _b64(fig)


def fig_hyperbolic(cross: pd.DataFrame):
    """PD against leverage, with the hyperbolic watch boundaries drawn on."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    lev_col = "TDTA" if "TDTA" in cross.columns else None
    if lev_col:
        x = pd.to_numeric(cross[lev_col], errors="coerce").to_numpy()
        y = pd.to_numeric(cross["pd_3m"], errors="coerce").to_numpy()
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        ax.scatter(x[ok], y[ok], s=13, c="#2563eb", alpha=0.65, edgecolors="none")
        if ok.any():
            grid = np.linspace(max(np.nanpercentile(x[ok], 1), 1e-6),
                               np.nanpercentile(x[ok], 99), 200)
            for k, color, label in ((0.02, "#f59e0b", "WATCH  x*y=0.02"),
                                    (0.05, "#ea580c", "ELEVATED  x*y=0.05"),
                                    (0.10, "#dc2626", "HIGH RISK  x*y=0.10")):
                ax.plot(grid, k / grid, color=color, lw=1.1, ls="--", label=label)
            ax.set_ylim(0, min(1.0, float(np.nanpercentile(y[ok], 99.5)) * 1.2 + 1e-3))
            ax.set_xlim(grid.min(), grid.max())
        ax.set_xlabel(f"leverage ({lev_col})")
        ax.legend(loc="upper right")
    ax.set_ylabel("PD over the 1-3 month window")
    ax.set_title("Hyperbolic risk boundaries")
    return _b64(fig)


def fig_lead_summary(lead: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.1))
    counts = lead["status"].value_counts()
    order = [s for s in ("detected", "earlier_only", "missed", "no_pre_event_data")
             if s in counts.index]
    colors = {"detected": "#16a34a", "earlier_only": "#f59e0b",
              "missed": "#dc2626", "no_pre_event_data": "#94a3b8"}
    ax[0].bar(range(len(order)), [counts[s] for s in order],
              color=[colors[s] for s in order])
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels([s.replace("_", "\n") for s in order], fontsize=7.5)
    ax[0].set_ylabel("issuers")
    ax[0].set_title("Event outcome")
    days = pd.to_numeric(lead.loc[lead["status"].eq("detected"),
                                  "actionable_lead_days"], errors="coerce").dropna()
    if len(days):
        ax[1].hist(days, bins=max(4, min(12, len(days))), color="#1e40af", alpha=0.85)
        ax[1].axvline(float(days.median()), color="#dc2626", ls="--", lw=1.2,
                      label=f"median {days.median():.0f} d")
        ax[1].legend()
    ax[1].set_xlabel("actionable lead time, days")
    ax[1].set_ylabel("issuers")
    ax[1].set_title("Lead time, 1-3 month window")
    fig.tight_layout()
    return _b64(fig)


def fig_model_compare(rows: list[dict]):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.1))
    names = [r["model"] for r in rows]
    ax[0].bar(range(len(rows)), [r["auc_oof"] for r in rows], color="#1e40af")
    ax[0].axhline(0.5, color="#94a3b8", ls="--", lw=0.9)
    ax[0].set_xticks(range(len(rows)))
    ax[0].set_xticklabels(names, rotation=20, fontsize=7.5, ha="right")
    ax[0].set_ylabel("AUC, out of fold")
    ax[0].set_title("Ranking quality")
    ax[1].bar(range(len(rows)), [r["average_precision_oof"] for r in rows],
              color="#b45309")
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels(names, rotation=20, fontsize=7.5, ha="right")
    ax[1].set_ylabel("average precision")
    ax[1].set_title("Precision-recall area")
    fig.tight_layout()
    return _b64(fig)


def fig_threshold_sweep(oof, y, latest_rows):
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    oof = np.asarray(oof, dtype=float)
    y = np.asarray(y)
    loads = np.linspace(0.01, 0.30, 30)
    prec, rec = [], []
    for w in loads:
        n = max(1, int(round(w * len(oof))))
        flag = np.zeros(len(oof), dtype=int)
        flag[np.argsort(oof)[::-1][:n]] = 1
        tp = int(((flag == 1) & (y == 1)).sum())
        prec.append(tp / max(n, 1))
        rec.append(tp / max(int(y.sum()), 1))
    ax.plot(loads * 100, prec, color="#1e40af", lw=1.3, label="precision")
    ax.plot(loads * 100, rec, color="#b45309", lw=1.3, label="recall")
    ax.set_xlabel("review workload, per cent of issuer-months flagged")
    ax.set_title("Precision and recall against workload")
    ax.legend()
    return _b64(fig)


# =============================================================== helpers
def df_table(df: pd.DataFrame, ft, max_rows=25, max_cols=12):
    import flet as _ft  # noqa: F401
    view = df.head(max_rows)
    cols = list(view.columns)[:max_cols]
    return ft.DataTable(
        heading_row_height=32,
        data_row_max_height=30,
        columns=[ft.DataColumn(ft.Text(str(c), size=10,
                                       weight=ft.FontWeight.BOLD)) for c in cols],
        rows=[
            ft.DataRow(cells=[
                ft.DataCell(ft.Text(
                    ("" if pd.isna(r[c]) else
                     (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])))[:38],
                    size=10))
                for c in cols])
            for _, r in view.iterrows()
        ],
    )


def scroll_box(ft, control, height=340):
    return ft.Container(
        content=ft.Column([ft.Row([control], scroll=ft.ScrollMode.AUTO)],
                          scroll=ft.ScrollMode.AUTO),
        height=height, bgcolor="white", border_radius=8,
        border=ft.Border.all(1, UI["border"]), padding=8,
    )


def card(ft, title, *controls):
    return ft.Container(
        content=ft.Column([
            ft.Text(title, size=13, weight=ft.FontWeight.BOLD, color=UI["primary"]),
            *controls,
        ], spacing=8),
        bgcolor=UI["surface"], border_radius=10, padding=14,
        border=ft.Border.all(1, UI["border"]),
    )


# =============================================================== the app
def main(page):
    import flet as ft

    page.title = "ThaiBMA Credit Early Warning — Dataset2"
    page.bgcolor = UI["page"]
    page.padding = 0
    page.theme_mode = ft.ThemeMode.LIGHT

    state = {"active_tab": 0, "data": None, "runs": {}, "model": "XGBoost",
             "workload": DL.DEFAULT_WORKLOAD, "issuer": None,
             "table": DL.DEFAULT_TABLE, "page_no": 0, "page_size": 25,
             "search": "", "summary": None, "summary_text": ""}

    status = ft.Text("starting", size=11, color=UI["muted"])
    banner_text = ft.Text("", size=11, color=UI["primary"],
                          weight=ft.FontWeight.BOLD, selectable=True)
    body = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=12)

    def say(message):
        status.value = message
        try:
            page.update()
        except Exception:
            pass

    def background(fn, message):
        def worker():
            say(message)
            try:
                fn()
            except Exception as exc:
                traceback.print_exc()
                say(f"failed: {exc}")
                return
            say("ready")
        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------- data access
    def ensure_data():
        if state["data"] is None:
            lines = []
            lines.append(f"Loading Dataset2 from {DL.DEFAULT_SOURCE_DB.resolve()}")
            data = DL.load_model_data()
            lines.append(DL.banner(data))
            for line in lines:
                print(line, flush=True)
            state["data"] = data
            banner_text.value = "\n".join(lines)
            if state["issuer"] is None and len(data["app_events"]):
                state["issuer"] = str(data["app_events"].iloc[0]["issuer_code"])
        return state["data"]

    def ensure_run(model=None):
        model = model or state["model"]
        key = (model, round(float(state["workload"]), 4))
        if key not in state["runs"]:
            data = ensure_data()
            state["runs"][key] = DL.fit_oof(data, model, state["workload"],
                                            progress=say)
        return state["runs"][key]

    # ------------------------------------------------------------- panels
    def panel_approach1():
        data = ensure_data()
        run = ensure_run()
        cross = DL.latest_cross_section(data, run)
        counts = cross["band"].value_counts()
        chips = ft.Row([
            ft.Container(
                content=ft.Column([
                    ft.Text(band, size=11, weight=ft.FontWeight.BOLD, color="white"),
                    ft.Text(f"{int(counts.get(band, 0))} issuers", size=16,
                            weight=ft.FontWeight.BOLD, color="white"),
                ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=BAND_COLOR[band], border_radius=8, padding=12, width=150)
            for band in ("HIGH RISK", "ELEVATED", "WATCH", "OK")
        ], spacing=10, wrap=True)
        m = run["metrics"]
        return [
            card(ft, "Approach 1: Dynamic Survival Hazard and Momentum",
                 ft.Text("Forward PD over the 1 to 3 month pre-event window, "
                         "scored out of fold with issuer-grouped folds so no firm "
                         "appears in both training and validation.",
                         size=11, color=UI["muted"]),
                 chips,
                 ft.Text(f"model {m['model']}  |  AUC {m['auc_oof']:.4f}  |  "
                         f"average precision {m['average_precision_oof']:.4f}  |  "
                         f"alarm threshold {m['threshold_latest_cross_section']:.4f}  |  "
                         f"workload {m['workload']:.0%}",
                         size=11, color=UI["text"])),
            ft.Row([
                card(ft, "ROE against credit risk",
                     ft.Image(src=_uri(fig_roe_vs_pd(cross)), width=520)),
                card(ft, "Hyperbolic risk boundaries",
                     ft.Image(src=_uri(fig_hyperbolic(cross)), width=520)),
            ], wrap=True, spacing=12),
            card(ft, f"Latest month of every issuer, ranked by PD "
                     f"({len(cross):,} issuers)",
                 scroll_box(ft, df_table(cross, ft, max_rows=40), 380)),
        ]

    def panel_approach2():
        data = ensure_data()
        run = ensure_run()
        m = run["metrics"]
        imp = DL.feature_importance(run, data["features"])
        return [
            card(ft, f"Approach 2: {m['model']} on the 33-feature panel",
                 ft.Text("Static financial, market and governance features with a "
                         "non-linear classifier. Importance is read from the "
                         "full-sample refit; every reported score is out of fold.",
                         size=11, color=UI["muted"]),
                 ft.Row([
                     ft.Text(f"AUC {m['auc_oof']:.4f}", size=12,
                             weight=ft.FontWeight.BOLD, color=UI["primary"]),
                     ft.Text(f"AP {m['average_precision_oof']:.4f}", size=12,
                             weight=ft.FontWeight.BOLD, color=UI["primary"]),
                     ft.Text(f"precision@{m['workload']:.0%} "
                             f"{m['precision_at_workload']:.4f}", size=12),
                     ft.Text(f"recall@{m['workload']:.0%} "
                             f"{m['recall_at_workload']:.4f}", size=12),
                     ft.Text(f"F1 {m['f1_at_workload']:.4f}", size=12),
                     ft.Text(f"{m['n_splits']} folds, "
                             f"{m['runtime_seconds']:.1f} s", size=11,
                             color=UI["muted"]),
                 ], wrap=True, spacing=16)),
            ft.Row([
                card(ft, "Feature importance",
                     ft.Image(src=_uri(fig_importance(imp)), width=520)),
                card(ft, "Risk distribution",
                     ft.Image(src=_uri(fig_pd_distribution(
                         run["oof"], run["threshold"])), width=520)),
            ], wrap=True, spacing=12),
            card(ft, "Precision and recall against review workload",
                 ft.Image(src=_uri(fig_threshold_sweep(
                     run["oof"], data["y"], data["latest_rows"])), width=560)),
            card(ft, "Feature importance table",
                 scroll_box(ft, df_table(imp, ft, max_rows=40), 320)),
        ]

    def panel_inspector():
        tables = DL.table_names()
        picker = ft.Dropdown(
            label="table", width=420, value=state["table"],
            options=[ft.dropdown.Option(t) for t in tables])
        search = ft.TextField(label="search issuer_code or name", width=300,
                              value=state["search"], dense=True)
        info = ft.Text("", size=11, color=UI["muted"])
        grid = ft.Column()

        def refresh(_=None):
            state["table"] = picker.value
            state["search"] = (search.value or "").strip()
            try:
                total = DL.row_count(state["table"])
                df = DL.read_table(state["table"], limit=4000)
                if state["search"]:
                    needle = state["search"].lower()
                    mask = pd.Series(False, index=df.index)
                    for c in df.columns:
                        if df[c].dtype == object:
                            mask |= df[c].astype(str).str.lower().str.contains(
                                needle, na=False)
                    df = df[mask]
                start = state["page_no"] * state["page_size"]
                view = df.iloc[start:start + state["page_size"]]
                info.value = (f"{state['table']}  |  {total:,} rows in table  |  "
                              f"{len(df):,} after search  |  "
                              f"showing {start + 1} to {start + len(view)}")
                grid.controls = [scroll_box(ft, df_table(
                    view, ft, max_rows=state["page_size"], max_cols=14), 400)]
            except Exception as exc:
                info.value = f"failed: {exc}"
                grid.controls = []
            page.update()

        def step(delta):
            def go(_):
                state["page_no"] = max(0, state["page_no"] + delta)
                refresh()
            return go

        picker.on_change = refresh
        search.on_submit = refresh
        refresh()
        return [
            card(ft, "Data Inspector and SQLite",
                 ft.Text(f"database {DL.DEFAULT_SOURCE_DB.resolve()}", size=11,
                         color=UI["muted"], selectable=True),
                 ft.Row([picker, search,
                         ft.Button("Search", icon=ft.Icons.SEARCH,
                                   on_click=refresh),
                         ft.Button("Prev", icon=ft.Icons.CHEVRON_LEFT,
                                   on_click=step(-1)),
                         ft.Button("Next", icon=ft.Icons.CHEVRON_RIGHT,
                                   on_click=step(1))], wrap=True, spacing=10),
                 info, grid),
        ]

    def panel_leadtime():
        data = ensure_data()
        run = ensure_run()
        app_lead = DL.lead_table(data, run, data["app_events"])
        all_lead = DL.lead_table(data, run, data["all_events"])
        s_app = DL.lead_summary(app_lead, "app")
        s_all = DL.lead_summary(all_lead, "all")

        def summary_row(s, prefix, label):
            return ft.Text(
                f"{label}: {s[prefix + '_detected']} of "
                f"{s[prefix + '_evaluable_events']} evaluable detected "
                f"({s[prefix + '_detection_rate']:.0%})  |  median lead "
                f"{s[prefix + '_lead_median_days']:.0f} d  |  range "
                f"{s[prefix + '_lead_min_days']:.0f} to "
                f"{s[prefix + '_lead_max_days']:.0f} d  |  median persistent run "
                f"{s[prefix + '_persistent_median_days']:.0f} d",
                size=11, color=UI["text"])

        return [
            card(ft, "Lead time: actionable 1 to 3 months, and persistent runs",
                 ft.Text("An alarm counts as actionable only if it fires one to "
                         "three months before the event month. An alarm in the "
                         "event month is not a warning, and a run that starts "
                         "earlier is reported separately.",
                         size=11, color=UI["muted"]),
                 summary_row(s_app, "app", "explicit onset issuers"),
                 summary_row(s_all, "all", "all dated or status issuers"),
                 ft.Text(f"metric version {DL.LEAD_METRIC_VERSION}", size=10,
                         color=UI["muted"])),
            card(ft, "Outcome and lead-time distribution",
                 ft.Image(src=_uri(fig_lead_summary(app_lead)), width=680)),
            card(ft, f"Per-issuer lead time, explicit onset "
                     f"({len(app_lead)} issuers)",
                 scroll_box(ft, df_table(app_lead, ft, max_rows=40, max_cols=11), 380)),
            card(ft, f"Per-issuer lead time, all dated or status events "
                     f"({len(all_lead)} issuers)",
                 scroll_box(ft, df_table(all_lead, ft, max_rows=40, max_cols=11), 380)),
        ]

    def panel_firm():
        data = ensure_data()
        run = ensure_run()
        events = data["app_events"]
        options = [ft.dropdown.Option(
            key=str(r.issuer_code),
            text=f"{r.issuer_code} — {r.firm_name} ({pd.Timestamp(r.event_month):%Y-%m})")
            for r in events.itertuples(index=False)]
        picker = ft.Dropdown(label="issuer with an event", width=520,
                             value=state["issuer"], options=options)
        holder = ft.Column()

        def refresh(_=None):
            state["issuer"] = picker.value
            frame = DL.issuer_frame(data, run, state["issuer"])
            ev = events.loc[events["issuer_code"].eq(str(state["issuer"]))]
            head = ""
            if len(ev):
                row = ev.iloc[0]
                head = (f"{row['firm_name']}  |  event "
                        f"{pd.Timestamp(row['event_month']):%Y-%m}  |  "
                        f"{row['event_type']}  |  source {row['event_source']}")
            lead = DL.compute_lead_for_issuer(
                frame.assign(alarm=frame["alarm"]),
                ev.iloc[0]["event_month"] if len(ev) else frame["month_dt"].max())
            holder.controls = [
                ft.Text(head, size=12, weight=ft.FontWeight.BOLD, color=UI["primary"]),
                ft.Text(f"actionable alarm {lead['actionable_alarm_month'] or 'none'}"
                        f"  |  lead {lead['actionable_lead_days'] if np.isfinite(lead['actionable_lead_days']) else float('nan'):.0f} d"
                        f"  |  persistent run {lead['persistent_alarm_start'] or 'none'}"
                        f" to {lead['persistent_alarm_end'] or 'none'}",
                        size=11, color=UI["text"]),
                ft.Image(src=_uri(fig_momentum(frame, str(state["issuer"]))), width=560),
                scroll_box(ft, df_table(
                    frame[["month", "pd_3m", "momentum_3m", "band", "alarm", "y_pre3m"]],
                    ft, max_rows=40), 340),
            ]
            page.update()

        picker.on_change = refresh
        refresh()
        return [card(ft, "Firm shock and PD threshold", picker, holder)]

    def panel_momentum():
        data = ensure_data()
        run = ensure_run()
        cross = DL.latest_cross_section(data, run)
        rising = cross.copy()
        return [
            card(ft, "Momentum and hyperbolic boundary",
                 ft.Text("The boundary is the curve where PD times leverage is "
                         "constant. A firm crosses a band either by raising PD or "
                         "by raising leverage, which is why the two are read "
                         "together rather than separately.",
                         size=11, color=UI["muted"]),
                 ft.Image(src=_uri(fig_hyperbolic(cross)), width=560)),
            card(ft, "Highest PD in the latest cross-section",
                 scroll_box(ft, df_table(rising.head(40), ft, max_rows=40), 380)),
        ]

    def panel_compare():
        data = ensure_data()
        rows = []
        for name in state.get("compare_models", ["XGBoost", "RandomForest", "Logistic"]):
            try:
                r = ensure_run(name)
                rows.append(r["metrics"])
            except Exception as exc:
                print(f"  {name} skipped: {exc}", flush=True)
        if not rows:
            return [card(ft, "Compare models", ft.Text("no model finished", size=11))]
        table = pd.DataFrame(rows)[[
            "model", "auc_oof", "average_precision_oof", "precision_at_workload",
            "recall_at_workload", "f1_at_workload", "n_splits", "runtime_seconds"]]
        lead_rows = []
        for name in table["model"]:
            run = ensure_run(name)
            lead = DL.lead_table(data, run, data["app_events"])
            s = DL.lead_summary(lead, "app")
            lead_rows.append({
                "model": name,
                "detected": s["app_detected"],
                "evaluable": s["app_evaluable_events"],
                "detection_rate": s["app_detection_rate"],
                "median_lead_days": s["app_lead_median_days"],
            })
        return [
            card(ft, "Compare models: Approach 1 against Approach 2",
                 ft.Text("Same panel, same folds, same workload. Only the "
                         "classifier changes.", size=11, color=UI["muted"]),
                 ft.Image(src=_uri(fig_model_compare(rows)), width=680)),
            card(ft, "Out-of-fold metrics",
                 scroll_box(ft, df_table(table, ft, max_rows=12), 240)),
            card(ft, "Lead time by model, explicit onset issuers",
                 scroll_box(ft, df_table(pd.DataFrame(lead_rows), ft, max_rows=12), 240)),
        ]

    def panel_zoo():
        state["compare_models"] = DL.MODEL_NAMES
        return panel_compare()

    def panel_run_summary():
        """Every model on the same folds, with how many defaults each one caught."""
        data = ensure_data()
        if state.get("summary") is None:
            summary, runs = DL.run_all_models(data, DL.MODEL_NAMES,
                                              state["workload"], progress=say)
            for name, run in runs.items():
                state["runs"][(name, round(float(state["workload"]), 4))] = run
            state["summary"] = summary
            state["summary_text"] = DL.print_run_summary(summary)
        summary = state["summary"]
        n_app = int(summary["app_evaluable_events"].iloc[0])
        n_all = int(summary["all_evaluable_events"].iloc[0])
        n_cat = int(summary["all_event_catalog"].iloc[0])

        chips = []
        for row in summary.itertuples(index=False):
            rate = row.all_detected / max(row.all_evaluable_events, 1)
            chips.append(ft.Container(
                content=ft.Column([
                    ft.Text(row.model, size=11, weight=ft.FontWeight.BOLD, color="white"),
                    ft.Text(f"{int(row.all_detected)} / {int(row.all_evaluable_events)}",
                            size=19, weight=ft.FontWeight.BOLD, color="white"),
                    ft.Text(f"caught  {rate:.0%}", size=10, color="white"),
                    ft.Text(f"recall {row.recall_at_workload:.4f}", size=10, color="white"),
                ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=("#16a34a" if rate >= 0.6 else
                         "#f59e0b" if rate >= 0.4 else "#dc2626"),
                border_radius=8, padding=12, width=168))

        return [
            card(ft, f"Defaults caught out of the {n_cat} issuers with an event",
                 ft.Text(f"Every model is fitted on the same issuer-grouped folds at "
                         f"a {state['workload']:.0%} review workload. The count is the "
                         f"number of issuers whose event was flagged one to three "
                         f"months ahead. Of the {n_cat} issuers in the catalogue, "
                         f"{n_all} can be evaluated; the rest have no panel month "
                         f"before the event, so no warning was ever possible. "
                         f"{n_app} carry a dated onset flag.",
                         size=11, color=UI["muted"]),
                 ft.Row(chips, spacing=10, wrap=True)),
            card(ft, "Run summary",
                 scroll_box(ft, df_table(summary, ft, max_rows=12,
                                         max_cols=len(summary.columns)), 260),
                 ft.Text(state.get("summary_text", ""), size=10,
                         font_family="Consolas", selectable=True,
                         color=UI["text"])),
            card(ft, "Ranking quality by model",
                 ft.Image(src=_uri(fig_model_compare(
                     summary.to_dict(orient="records"))), width=680)),
        ]

    def panel_about():
        data = ensure_data()
        panel = data["panel"]
        rows = [
            ("database", str(DL.DEFAULT_SOURCE_DB.resolve())),
            ("table", data["table"]),
            ("rows", f"{len(panel):,}"),
            ("issuers", f"{panel['issuer_code'].nunique():,}"),
            ("positive issuer-months", f"{int(np.asarray(data['y']).sum()):,}"),
            ("explicit event issuers", f"{len(data['app_events']):,}"),
            ("all dated or status event issuers", f"{len(data['all_events']):,}"),
            ("features", f"{len(data['features'])}"),
            ("month range", f"{panel['month_dt'].min():%Y-%m} to "
                            f"{panel['month_dt'].max():%Y-%m}"),
            ("target", "y_pre3m, the 1 to 3 month pre-event window"),
            ("validation", "StratifiedGroupKFold grouped on issuer_code, seed 42"),
        ]
        return [
            card(ft, "Dataset2 and how it is scored",
                 *[ft.Text(f"{k}: {v}", size=11, color=UI["text"], selectable=True)
                   for k, v in rows]),
            card(ft, "Event catalogue, explicit onset",
                 scroll_box(ft, df_table(data["app_events"], ft, max_rows=40), 380)),
            card(ft, "Event catalogue, all dated or status",
                 scroll_box(ft, df_table(data["all_events"], ft, max_rows=40), 380)),
        ]

    TABS = [
        ("Run summary: defaults caught by every model", ft.Icons.LEADERBOARD,
         panel_run_summary, "RUN SUMMARY"),
        ("Approach 1: Survival Dashboard", ft.Icons.TIMELINE, panel_approach1, "RISK MODELS"),
        ("Approach 2: XGBoost + SHAP", ft.Icons.PSYCHOLOGY, panel_approach2, "RISK MODELS"),
        ("Firm Shock & PD Threshold", ft.Icons.CRISIS_ALERT, panel_firm, "RISK MODELS"),
        ("Momentum & Hyperbolic Boundary", ft.Icons.SHOW_CHART, panel_momentum, "RISK MODELS"),
        ("Lead Time: 1-3M + Persistent", ft.Icons.SCHEDULE, panel_leadtime, "LEAD TIME & EVALUATION"),
        ("Compare Models: A1 vs A2", ft.Icons.COMPARE_ARROWS, panel_compare, "LEAD TIME & EVALUATION"),
        ("Model Zoo: all classifiers", ft.Icons.SCATTER_PLOT, panel_zoo, "LEAD TIME & EVALUATION"),
        ("Data Inspector & SQLite", ft.Icons.TABLE_CHART, panel_inspector, "DATASET2"),
        ("Dataset2 Summary", ft.Icons.INSIGHTS, panel_about, "DATASET2"),
    ]

    nav_styles = [("#e0f2fe", "#0369a1", "#1d4ed8"),
                  ("#fef3c7", "#b45309", "#d97706")]

    def show(idx):
        state["active_tab"] = idx
        for b in nav_buttons:
            active = (b.tab_idx == idx)
            def_bg, def_fg, act_bg = nav_styles[b.tab_idx % 2]
            b.bgcolor = act_bg if active else def_bg
            b.color = "white" if active else def_fg
        body.controls = [ft.Text(TABS[idx][0], size=17,
                                 weight=ft.FontWeight.BOLD, color=UI["primary"]),
                         ft.Text("working", size=11, color=UI["muted"])]
        page.update()

        def work():
            controls = TABS[idx][2]()
            body.controls = [ft.Text(TABS[idx][0], size=17,
                                     weight=ft.FontWeight.BOLD,
                                     color=UI["primary"]), *controls]
            page.update()

        background(work, f"building {TABS[idx][0]}")

    nav_buttons = []
    side_controls = []
    last_section = None
    for i, (label, icon, _fn, section) in enumerate(TABS):
        if section != last_section:
            side_controls.append(ft.Text(section, size=10,
                                         weight=ft.FontWeight.BOLD,
                                         color=UI["muted"]))
            last_section = section
        def_bg, def_fg, _act = nav_styles[i % 2]
        btn = ft.Button(label, icon=icon, bgcolor=def_bg, color=def_fg,
                        width=NAV_BUTTON_WIDTH, height=NAV_BUTTON_HEIGHT,
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=6),
                            padding=ft.Padding.symmetric(horizontal=14, vertical=10)))
        btn.tab_idx = i
        btn.on_click = (lambda idx: (lambda _e: show(idx)))(i)
        nav_buttons.append(btn)
        side_controls.append(btn)

    model_picker = ft.Dropdown(
        label="model", width=248, value=state["model"],
        options=[ft.dropdown.Option(m) for m in DL.MODEL_NAMES])
    workload_field = ft.TextField(label="workload", width=110, dense=True,
                                  value=f"{state['workload']:.2f}")

    def apply_settings(_):
        state["model"] = model_picker.value
        old_workload = state["workload"]
        try:
            state["workload"] = max(0.005, min(0.5, float(workload_field.value)))
        except Exception:
            pass
        if state["workload"] != old_workload:
            state["summary"] = None      # the whole table is workload dependent
        workload_field.value = f"{state['workload']:.2f}"
        show(state["active_tab"])

    sidebar = ft.Container(
        content=ft.Column([
            ft.Text("CMDF / ThaiBMA", size=15, weight=ft.FontWeight.BOLD,
                    color=UI["primary"]),
            ft.Text("Credit Early Warning — Dataset2", size=11, color=UI["muted"]),
            ft.Divider(height=8),
            model_picker,
            ft.Row([workload_field,
                    ft.Button("Apply", icon=ft.Icons.PLAY_ARROW,
                              on_click=apply_settings)], spacing=8),
            ft.Divider(height=8),
            ft.ListView(controls=side_controls, spacing=8, height=520,
                        auto_scroll=False),
            ft.Divider(height=8),
            banner_text,
            status,
        ], spacing=8),
        width=308, bgcolor=UI["sidebar"], padding=14,
    )

    page.add(ft.Row([sidebar,
                     ft.Container(content=body, expand=True, padding=16)],
                    expand=True, vertical_alignment=ft.CrossAxisAlignment.START))

    # the headless UI check builds every panel in turn instead of only the first
    if getattr(page, "build_all_panels", False):
        for i, (label, _icon, fn, _sec) in enumerate(TABS):
            controls = fn()
            print(f"  panel {i} {label:42s} {len(controls)} control(s)", flush=True)
        return
    show(0)


# =============================================================== entry points
def _selftest():
    print("app_dataset2 self test")
    data, _ = DL.load_with_banner()
    run = DL.fit_oof(data, "XGBoost", DL.DEFAULT_WORKLOAD)
    m = run["metrics"]
    print(f"AUC {m['auc_oof']:.4f} | AP {m['average_precision_oof']:.4f} | "
          f"threshold {m['threshold_latest_cross_section']:.4f}")
    lead = DL.lead_table(data, run, data["app_events"])
    s = DL.lead_summary(lead, "app")
    print(f"detected {s['app_detected']} of {s['app_evaluable_events']} "
          f"evaluable | median lead {s['app_lead_median_days']:.0f} d")
    cross = DL.latest_cross_section(data, run)
    imp = DL.feature_importance(run, data["features"])
    for name, fn in (
        ("pd distribution", lambda: fig_pd_distribution(run["oof"], run["threshold"])),
        ("importance", lambda: fig_importance(imp)),
        ("roe vs pd", lambda: fig_roe_vs_pd(cross)),
        ("hyperbolic", lambda: fig_hyperbolic(cross)),
        ("lead summary", lambda: fig_lead_summary(lead)),
        ("threshold sweep", lambda: fig_threshold_sweep(run["oof"], data["y"],
                                                        data["latest_rows"])),
        ("momentum", lambda: fig_momentum(
            DL.issuer_frame(data, run, str(data["app_events"].iloc[0]["issuer_code"])),
            str(data["app_events"].iloc[0]["issuer_code"]))),
        ("model compare", lambda: fig_model_compare([m])),
    ):
        png = fn()
        print(f"  figure {name:16s} {len(png):>8,} base64 chars")
    print("self test passed")


def _uitest():
    """Build every panel without a window, so a broken control is caught here."""
    import flet as ft

    class FakePage:
        def __init__(self):
            self.controls = []
            self.title = ""
            self.bgcolor = None
            self.padding = 0
            self.theme_mode = None

        def add(self, *c):
            self.controls.extend(c)

        def update(self):
            pass

    page = FakePage()
    page.build_all_panels = True
    main(page)
    print(f"ui test built {len(page.controls)} root control(s) and every panel")
    print("ui test passed")


def _summary_only():
    """The run summary without opening a window."""
    data, _ = DL.load_with_banner()
    summary, _runs = DL.run_all_models(data, DL.MODEL_NAMES, DL.DEFAULT_WORKLOAD)
    DL.print_run_summary(summary)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--uitest" in sys.argv:
        _uitest()
    elif "--summary" in sys.argv:
        _summary_only()
    else:
        import flet as ft
        ft.run(main)
