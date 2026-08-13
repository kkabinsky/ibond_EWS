# -*- coding: utf-8 -*-
"""
paper_replication.py
Replication of: Wattanatorn, W. & Wiriyadee, T.
"Determinants of Corporate Default Risk in the Thai Bond Market: A Multi-Horizon Analysis"
(Emerging Markets Finance and Trade, submission 262322061)

Sample (paper section 3.1)
    - non-financial SET-listed bond issuers, 2012-2025
    - Refinitiv ESG coverage  ->  206 issuers, ~13,000 firm-months
    - ratios winsorised at the 1st/99th percentile
    - every regressor lagged one period (within firm)

Dependent variables (section 3.2): Merton-based, computed in the source database
    ln(PD_h + 0.0001)  and  DD_h        for h = 12, 24, 36, 60 months

Baseline model (section 3.4)
    ln PD_it = a + bF F_{i,t-1} + bN N_{i,t-1} + bM M_{i,t-1} + bC C_{i,t-1} + bZ Z_{i,t-1}
               + industry FE + year FE + e_it        (SE clustered by firm)

Parsimonious proxies (section 3.5): profitability = ROA, leverage = D/E,
liquidity = current ratio, coverage = cash-flow debt-service coverage ratio.

Tables produced
    Table 4  baseline ln PD           4 horizons x {Amihud, Kang-Zhang}
    Table 5  distance to default      4 horizons x {Amihud, Kang-Zhang}
    Table 6  fractional logit on PD   4 horizons x {ln Amihud, Kang-Zhang}
    Table 7  ESG pillar scores        E / S / G entered separately
    Table 8  sensitivity to alternative ratio proxies
    Table 9  augmented macro (Panel A: no year FE, Panel B: with year FE)
    Table 10 factor model (PCA within construct: F_prof, F_lev, F_liq, F_cov)
    Table 11 external validation against actual ThaiBMA DP / RS events

Run:  python paper_replication.py                  (all tables)
      python paper_replication.py --tables 4,5     (selected)
      python paper_replication.py --expanded       (Appendix A: no ESG-coverage filter)
      python paper_replication.py --csv outdir     (also dump each table to CSV)
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from thaibma_paths import DATA_ROOT  # data lives outside the repo

warnings.filterwarnings("ignore")

DTA = r"D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta"
HERE = os.path.dirname(os.path.abspath(__file__))

YEAR_MIN, YEAR_MAX = 2012, 2025
HORIZONS = [12, 24, 36, 60]
PD_EPS = 1e-4                      # ln(PD + 0.0001), paper section 3.4

# ---- variable blocks --------------------------------------------------------
PROFIT = "ROA"
LEVER = "DE"
LIQUID = "CurrentRatio"
COVER = "cf_DebtServiceCoverageRatio"
NONFIN = ["lnTotalAssets", "AgeYear"]
MACRO = ["Policyrate"]
ESG = "ESGScore"
LIQ_PROXIES = {"Amihud": "amihud_monthly_100",
               "Kang-Zhang": "adj_illiq_kz",
               "ln Amihud": "ln_amihud"}

BASE_X = [PROFIT, LEVER, LIQUID, COVER] + NONFIN + MACRO + [ESG]

# alternatives used by the sensitivity table (paper section 5.1)
ALTERNATIVES = {
    "profitability": [PROFIT, "ROE", "EBITtoTA", "REtoTA"],
    "leverage": [LEVER, "TDTA", "LTDtoTA", "STDtoTA"],
    "liquidity": [LIQUID, "QuickRatio", "CashRatio", "WorkingCapitaltoTA"],
    "coverage": [COVER, "cf_Interestcoverageratio", "acc_DebtServiceCoverageRatio",
                 "cf_CashFlowCoverageRatio"],
}
WINSOR = [LEVER, "WorkingCapitaltoTA", COVER, "cf_Interestcoverageratio",
          "acc_DebtServiceCoverageRatio", "cf_CashFlowCoverageRatio", "adj_illiq_kz",
          "amihud_monthly_100", "ln_amihud", "ROE", "TDTA", "LTDtoTA", "STDtoTA",
          "QuickRatio", "CashRatio", "CurrentRatio", "ROA", "EBITtoTA", "REtoTA"]

PILLARS = ["EnvironmentalPillarScore", "SocialPillarScore", "GovernancePillarScore"]
EXTRA_MACRO = ["GDPgrowth", "UnemploymentratemodeledILOe"]
EVENT_COLS = ["d_DP_RS", "d_Default_Payment", "d_Restructure"]


# ============================================================= sample =========
def load_sample(expanded: bool = False) -> pd.DataFrame:
    """Build the estimation panel exactly as described in section 3.1."""
    need = (["firm_id", "month_year", "year", "SETIndustry", ESG]
            + [f"pd_{h}m" for h in HORIZONS] + [f"dd_{h}m" for h in HORIZONS]
            + BASE_X + list(LIQ_PROXIES.values()) + PILLARS + EXTRA_MACRO + EVENT_COLS
            + [v for vs in ALTERNATIVES.values() for v in vs])
    need = list(dict.fromkeys(need))
    df = pd.read_stata(DTA, columns=need, convert_categoricals=True)

    df["SETIndustry"] = df["SETIndustry"].astype(str)
    df = df[~df["SETIndustry"].str.lower().str.contains("financ", na=False)]   # exclude financials
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df["month_year"] = pd.to_datetime(df["month_year"])
    df = df.sort_values(["firm_id", "month_year"]).reset_index(drop=True)

    if not expanded:                          # main sample = Refinitiv ESG coverage
        df = df[df[ESG].notna()]
    df = df[df["pd_12m"].notna()]

    for c in WINSOR:                          # winsorise 1% / 99% (section 3.1)
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            lo, hi = v.quantile(0.01), v.quantile(0.99)
            df[c] = v.clip(lo, hi)

    # one-period lag of every regressor, within firm
    lag_cols = [c for c in df.columns
                if c not in ("firm_id", "month_year", "year", "SETIndustry")
                and not c.startswith(("pd_", "dd_", "d_"))]
    for c in lag_cols:
        df["L." + c] = df.groupby("firm_id")[c].shift(1)

    for h in HORIZONS:                        # dependent variables
        df[f"lnPD_{h}"] = np.log(pd.to_numeric(df[f"pd_{h}m"], errors="coerce") + PD_EPS)
        df[f"DD_{h}"] = pd.to_numeric(df[f"dd_{h}m"], errors="coerce")
    return df.reset_index(drop=True)


# ========================================================== estimation ========
def _design(d: pd.DataFrame, xs: list[str], year_fe: bool = True, industry_fe: bool = True):
    X = d[xs].astype(float).copy()
    if industry_fe:
        X = pd.concat([X, pd.get_dummies(d["SETIndustry"], prefix="ind", drop_first=True,
                                         dtype=float)], axis=1)
    if year_fe:
        X = pd.concat([X, pd.get_dummies(d["year"].astype(int), prefix="yr", drop_first=True,
                                         dtype=float)], axis=1)
    return sm.add_constant(X, has_constant="add")


def fit_ols(df, y, xs, year_fe=True, industry_fe=True):
    """OLS with industry + year fixed effects and firm-clustered standard errors."""
    d = df.dropna(subset=[y] + xs + ["SETIndustry", "year", "firm_id"]).copy()
    if d.empty:
        return None
    X = _design(d, xs, year_fe, industry_fe)
    res = sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["firm_id"].astype(str)})
    res._n_firms = d["firm_id"].nunique()
    res._within_r2 = _within_r2(d, y, xs)
    return res


def _within_r2(d, y, xs):
    """R^2 of the FE-demeaned model (industry x year absorbed)."""
    try:
        g = d.groupby(["SETIndustry", "year"])
        yy = d[y].astype(float) - g[y].transform("mean")
        XX = pd.DataFrame({c: d[c].astype(float) - g[c].transform("mean") for c in xs})
        return float(sm.OLS(yy, sm.add_constant(XX)).fit().rsquared)
    except Exception:
        return float("nan")


def fit_fraclogit(df, y_raw, xs, year_fe=True, industry_fe=True):
    """Fractional logit QMLE (Papke & Wooldridge 1996) on PD in [0,1]."""
    d = df.dropna(subset=[y_raw] + xs + ["SETIndustry", "year", "firm_id"]).copy()
    yv = pd.to_numeric(d[y_raw], errors="coerce").clip(0, 1)
    d = d.assign(_y=yv).dropna(subset=["_y"])
    if d.empty:
        return None
    X = _design(d, xs, year_fe, industry_fe)
    res = sm.GLM(d["_y"], X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["firm_id"].astype(str)})
    res._n_firms = d["firm_id"].nunique()
    res._within_r2 = float("nan")
    return res


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def coef_table(models: dict, xs: list[str], title: str, note: str = "") -> pd.DataFrame:
    """models: {column label -> fitted result}. Returns a tidy coefficient table."""
    rows = []
    for v in xs:
        cf, se = {}, {}
        for lab, m in models.items():
            if m is None or v not in m.params.index:
                cf[lab], se[lab] = "", ""
                continue
            cf[lab] = f"{m.params[v]:.4f}{stars(m.pvalues[v])}"
            se[lab] = f"({m.bse[v]:.4f})"
        rows.append({"variable": v, **cf})
        rows.append({"variable": "", **se})
    stat = lambda f: {lab: ("" if m is None else f(m)) for lab, m in models.items()}
    rows += [
        {"variable": "Observations", **stat(lambda m: f"{int(m.nobs):,}")},
        {"variable": "Firms", **stat(lambda m: f"{m._n_firms:,}")},
        {"variable": "Within R2", **stat(lambda m: f"{m._within_r2:.3f}"
                                         if m._within_r2 == m._within_r2 else "")},
        {"variable": "Industry FE", **stat(lambda m: "Yes")},
        {"variable": "Year FE", **stat(lambda m: "Yes")},
    ]
    out = pd.DataFrame(rows)
    out.attrs["title"] = title
    out.attrs["note"] = note
    return out


def show(tbl: pd.DataFrame, width: int = 13):
    print("\n" + "=" * 112)
    print(tbl.attrs.get("title", ""))
    print("=" * 112)
    cols = [c for c in tbl.columns if c != "variable"]
    print(f"  {'Variable':30s}" + "".join(f"{c:>{width}s}" for c in cols))
    print("  " + "-" * (30 + width * len(cols)))
    for _, r in tbl.iterrows():
        print(f"  {str(r['variable']):30s}" + "".join(f"{str(r[c]):>{width}s}" for c in cols))
    if tbl.attrs.get("note"):
        print("  " + tbl.attrs["note"])


# ============================================================== tables ========
def table4(df):
    """Baseline: ln(PD) at 4 horizons, Amihud vs Kang-Zhang liquidity."""
    models, xs_used = {}, None
    for lname in ["Amihud", "Kang-Zhang"]:
        xs = [f"L.{v}" for v in BASE_X[:-1]] + [f"L.{LIQ_PROXIES[lname]}", f"L.{ESG}"]
        xs_used = xs_used or xs
        for h in HORIZONS:
            models[f"{lname[:4]}-{h}m"] = fit_ols(df, f"lnPD_{h}", xs)
    xs_all = ([f"L.{v}" for v in BASE_X[:-1]]
              + [f"L.{LIQ_PROXIES['Amihud']}", f"L.{LIQ_PROXIES['Kang-Zhang']}", f"L.{ESG}"])
    return coef_table(models, xs_all,
                      "TABLE 4 -- Baseline panel regressions, dependent = ln(PD + 0.0001)",
                      "*** p<0.01, ** p<0.05, * p<0.10; SE clustered by firm. "
                      "Cols 1-4 Amihud, cols 5-8 Kang-Zhang.")


def table5(df):
    """Alternative dependent variable: distance to default."""
    models = {}
    for lname in ["Amihud", "Kang-Zhang"]:
        xs = [f"L.{v}" for v in BASE_X[:-1]] + [f"L.{LIQ_PROXIES[lname]}", f"L.{ESG}"]
        for h in HORIZONS:
            models[f"{lname[:4]}-{h}m"] = fit_ols(df, f"DD_{h}", xs)
    xs_all = ([f"L.{v}" for v in BASE_X[:-1]]
              + [f"L.{LIQ_PROXIES['Amihud']}", f"L.{LIQ_PROXIES['Kang-Zhang']}", f"L.{ESG}"])
    return coef_table(models, xs_all,
                      "TABLE 5 -- Distance to default (DD) as dependent variable",
                      "Signs mirror Table 4: higher DD = safer.")


def table6(df):
    """Fractional logit (Papke-Wooldridge) on the raw bounded PD."""
    models = {}
    for lname in ["ln Amihud", "Kang-Zhang"]:
        xs = [f"L.{v}" for v in BASE_X[:-1]] + [f"L.{LIQ_PROXIES[lname]}", f"L.{ESG}"]
        for h in HORIZONS:
            models[f"{lname[:4]}-{h}m"] = fit_fraclogit(df, f"pd_{h}m", xs)
    xs_all = ([f"L.{v}" for v in BASE_X[:-1]]
              + [f"L.{LIQ_PROXIES['ln Amihud']}", f"L.{LIQ_PROXIES['Kang-Zhang']}", f"L.{ESG}"])
    return coef_table(models, xs_all,
                      "TABLE 6 -- Fractional logit QMLE on raw PD (Papke & Wooldridge 1996)",
                      "Raw Amihud fails to converge (extreme right tail); ln Amihud used instead.")


def table7(df):
    """ESG pillars entered separately (Amihud specification)."""
    models = {}
    base = [f"L.{v}" for v in BASE_X[:-1]] + [f"L.{LIQ_PROXIES['Amihud']}"]
    for pil in PILLARS:
        for h in HORIZONS:
            models[f"{pil[:3]}-{h}m"] = fit_ols(df, f"lnPD_{h}", base + [f"L.{pil}"])
    return coef_table(models, base + [f"L.{p}" for p in PILLARS],
                      "TABLE 7 -- ESG pillar scores (Environmental / Social / Governance)",
                      "Each pillar replaces the aggregate ESG score in the baseline.")


def table8(df):
    """Sensitivity: swap one construct's proxy at a time (Kang-Zhang ln-PD spec)."""
    rows = []
    for construct, alts in ALTERNATIVES.items():
        for alt in alts:
            xs = [f"L.{v}" for v in
                  [alt if c == construct else b
                   for c, b in [("profitability", PROFIT), ("leverage", LEVER),
                                ("liquidity", LIQUID), ("coverage", COVER)]]]
            xs += [f"L.{v}" for v in NONFIN + MACRO]
            xs += [f"L.{LIQ_PROXIES['Kang-Zhang']}", f"L.{ESG}"]
            rec = {"construct": construct, "proxy": alt}
            for h in HORIZONS:
                m = fit_ols(df, f"lnPD_{h}", xs)
                key = f"L.{alt}"
                rec[f"{h}m"] = ("" if m is None or key not in m.params.index
                                else f"{m.params[key]:.4f}{stars(m.pvalues[key])}")
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs["title"] = ("TABLE 8 -- Sensitivity to alternative ratio proxies "
                          "(coefficient on the swapped proxy, ln PD)")
    out.attrs["note"] = "Baseline proxy listed first in each construct block."
    return out


def table9(df):
    """Augmented macro: Panel A drops year FE, Panel B keeps it."""
    models = {}
    xs = ([f"L.{v}" for v in BASE_X[:-1]] + [f"L.{LIQ_PROXIES['Amihud']}", f"L.{ESG}"]
          + [f"L.{v}" for v in EXTRA_MACRO])
    for h in HORIZONS:
        models[f"A-{h}m"] = fit_ols(df, f"lnPD_{h}", xs, year_fe=False)
    for h in HORIZONS:
        models[f"B-{h}m"] = fit_ols(df, f"lnPD_{h}", xs, year_fe=True)
    return coef_table(models, xs,
                      "TABLE 9 -- Augmented macroeconomic specification "
                      "(Panel A: no year FE | Panel B: with year FE)",
                      "GDP growth and unemployment added to the policy rate.")


def table10(df):
    """Factor model: first principal component within each construct."""
    from sklearn.decomposition import PCA
    d = df.copy()
    made = []
    for construct, alts in ALTERNATIVES.items():
        cols = [f"L.{a}" for a in alts if f"L.{a}" in d.columns]
        sub = d[cols].astype(float)
        sub = sub.fillna(sub.median())
        z = (sub - sub.mean()) / (sub.std() + 1e-9)
        name = {"profitability": "F_prof", "leverage": "F_lev",
                "liquidity": "F_liq", "coverage": "F_cov"}[construct]
        d[name] = PCA(n_components=1, random_state=0).fit_transform(z.values)[:, 0]
        made.append(name)
    xs = made + [f"L.{v}" for v in NONFIN + MACRO] + [f"L.{LIQ_PROXIES['Kang-Zhang']}", f"L.{ESG}"]
    models = {f"{h}m": fit_ols(d, f"lnPD_{h}", xs) for h in HORIZONS}
    return coef_table(models, xs,
                      "TABLE 10 -- Factor model (first PC within each construct)",
                      "F_prof, F_lev, F_liq, F_cov are orthogonal construct factors.")


def table11(df):
    """External validation: do the model-based measures line up with real DP/RS events?"""
    d = df.copy()
    d["event_firm"] = d.groupby("firm_id")["d_DP_RS"].transform(
        lambda s: (pd.to_numeric(s, errors="coerce") > 0).any())
    rows = []
    for h in HORIZONS:
        ev = d.loc[d["event_firm"], f"pd_{h}m"].dropna()
        nv = d.loc[~d["event_firm"], f"pd_{h}m"].dropna()
        dd_ev = d.loc[d["event_firm"], f"dd_{h}m"].dropna()
        dd_nv = d.loc[~d["event_firm"], f"dd_{h}m"].dropna()
        from scipy import stats as st
        t_pd = st.ttest_ind(ev, nv, equal_var=False)
        t_dd = st.ttest_ind(dd_ev, dd_nv, equal_var=False)
        y = np.r_[np.ones(len(ev)), np.zeros(len(nv))]
        s = np.r_[ev.values, nv.values]
        from sklearn.metrics import roc_auc_score
        rows.append({
            "horizon": f"{h}m",
            "PD event firms": f"{ev.mean():.4f}",
            "PD non-event": f"{nv.mean():.4f}",
            "diff t": f"{t_pd.statistic:.2f}{stars(t_pd.pvalue)}",
            "DD event firms": f"{dd_ev.mean():.3f}",
            "DD non-event": f"{dd_nv.mean():.3f}",
            "DD t": f"{t_dd.statistic:.2f}{stars(t_dd.pvalue)}",
            "PD AUC": f"{roc_auc_score(y, s):.3f}",
        })
    out = pd.DataFrame(rows)
    out.attrs["title"] = ("TABLE 11 -- External validation against actual ThaiBMA "
                          "payment-default / restructuring events")
    out.attrs["note"] = ("Event firms should show higher PD and lower DD. "
                         "AUC = ability of the Merton PD to rank event firms.")
    return out


def descriptives(df):
    keep = [PROFIT, LEVER, LIQUID, COVER] + NONFIN + MACRO + [ESG] + list(LIQ_PROXIES.values())
    rows = []
    for c in keep:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append({"variable": c, "N": f"{len(v):,}", "mean": f"{v.mean():.4f}",
                     "sd": f"{v.std():.4f}", "p25": f"{v.quantile(.25):.4f}",
                     "median": f"{v.median():.4f}", "p75": f"{v.quantile(.75):.4f}"})
    for h in HORIZONS:
        for pre in ("pd", "dd"):
            v = pd.to_numeric(df[f"{pre}_{h}m"], errors="coerce").dropna()
            rows.append({"variable": f"{pre.upper()}_{h}m", "N": f"{len(v):,}",
                         "mean": f"{v.mean():.4f}", "sd": f"{v.std():.4f}",
                         "p25": f"{v.quantile(.25):.4f}", "median": f"{v.median():.4f}",
                         "p75": f"{v.quantile(.75):.4f}"})
    out = pd.DataFrame(rows)
    out.attrs["title"] = "TABLE 2 -- Descriptive statistics"
    out.attrs["note"] = ""
    return out


TABLES = {"2": ("descriptives", descriptives), "4": ("baseline ln PD", table4),
          "5": ("distance to default", table5), "6": ("fractional logit", table6),
          "7": ("ESG pillars", table7), "8": ("proxy sensitivity", table8),
          "9": ("augmented macro", table9), "10": ("factor model", table10),
          "11": ("event validation", table11)}


# ================================================== horizon coefficient path ==
KEY_VARS = [f"L.{v}" for v in (PROFIT, LEVER, LIQUID, COVER, "lnTotalAssets", "Policyrate")]


def horizon_paths(df, liq="Kang-Zhang") -> pd.DataFrame:
    """Coefficient, SE and p-value of each key regressor at every horizon
    (the paper's central 'multi-horizon' result), for ln PD and for DD."""
    xs = [f"L.{v}" for v in BASE_X[:-1]] + [f"L.{LIQ_PROXIES[liq]}", f"L.{ESG}"]
    rows = []
    for dep, mk in (("lnPD", lambda h: f"lnPD_{h}"), ("DD", lambda h: f"DD_{h}")):
        for h in HORIZONS:
            m = fit_ols(df, mk(h), xs)
            if m is None:
                continue
            for v in KEY_VARS + [f"L.{LIQ_PROXIES[liq]}", f"L.{ESG}"]:
                if v not in m.params.index:
                    continue
                rows.append({"dependent": dep, "horizon": h, "variable": v,
                             "coef": float(m.params[v]), "se": float(m.bse[v]),
                             "pvalue": float(m.pvalues[v]), "stars": stars(m.pvalues[v])})
    return pd.DataFrame(rows)


# ============================================================== persistence ===
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")


def run_all(expanded: bool = False) -> dict:
    """Estimate every table plus the horizon coefficient paths."""
    import sqlite3  # noqa: F401  (kept local; only needed when saving)
    df = load_sample(expanded=expanded)
    out = {"sample": df}
    for key, (_name, fn) in TABLES.items():
        try:
            out[f"table{key}"] = fn(df)
        except Exception as ex:
            out[f"table{key}"] = pd.DataFrame([{"error": str(ex)}])
    out["paths"] = horizon_paths(df)
    out["summary"] = {
        "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample": "expanded (Appendix A)" if expanded else "main (ESG coverage)",
        "n_obs": int(len(df)), "n_firms": int(df["firm_id"].nunique()),
        "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
        "n_industries": int(df["SETIndustry"].nunique()),
        "paper_n_obs": 13072, "paper_n_firms": 206,
        "horizons": ",".join(str(h) for h in HORIZONS),
        "lambda_note": "ln(PD + 1e-4); regressors lagged 1 period; industry & year FE; SE clustered by firm",
    }
    return out


def save_to_sqlite(res: dict, db: str = DB) -> None:
    import sqlite3
    con = sqlite3.connect(db)
    for k, v in res.items():
        if k in ("sample", "summary"):
            continue
        if isinstance(v, pd.DataFrame):
            v.astype(str).to_sql(f"paper_{k}", con, if_exists="replace", index=False)
    res["paths"].to_sql("paper_paths", con, if_exists="replace", index=False)
    pd.DataFrame([res["summary"]]).to_sql("paper_summary", con, if_exists="replace", index=False)
    con.commit(); con.close()


def load_from_sqlite(db: str = DB):
    import sqlite3
    con = sqlite3.connect(db)
    tabs, paths, summ = {}, pd.DataFrame(), pd.DataFrame()
    try:
        for key in TABLES:
            try:
                tabs[key] = pd.read_sql_query(f"SELECT * FROM paper_table{key}", con)
            except Exception:
                pass
        paths = pd.read_sql_query("SELECT * FROM paper_paths", con)
        summ = pd.read_sql_query("SELECT * FROM paper_summary LIMIT 1", con)
    except Exception:
        pass
    finally:
        con.close()
    return tabs, paths, summ


def main():
    expanded = "--expanded" in sys.argv
    want = None
    if "--tables" in sys.argv:
        want = [t.strip() for t in sys.argv[sys.argv.index("--tables") + 1].split(",")]
    csv_dir = None
    if "--csv" in sys.argv:
        csv_dir = sys.argv[sys.argv.index("--csv") + 1]
        os.makedirs(csv_dir, exist_ok=True)

    if "--save" in sys.argv:                      # estimate everything -> SQLite (for the GUI)
        print("estimating all tables and saving to SQLite ...")
        res = run_all(expanded=expanded)
        save_to_sqlite(res)
        s = res["summary"]
        print(f"  {s['n_obs']:,} firm-months | {s['n_firms']} issuers | "
              f"{s['year_min']}-{s['year_max']} | sample: {s['sample']}")
        print(f"  saved: paper_table2/4/5/6/7/8/9/10/11, paper_paths, paper_summary ({DB})")
        return

    print("=" * 112)
    print("REPLICATION -- Determinants of Corporate Default Risk in the Thai Bond Market")
    print("               Wattanatorn & Wiriyadee, Emerging Markets Finance and Trade")
    print("=" * 112)
    print(f"loading sample ({'expanded / Appendix A' if expanded else 'main, ESG coverage'}) ...")
    df = load_sample(expanded=expanded)
    print(f"  firm-months {len(df):,} | issuers {df['firm_id'].nunique()} | "
          f"{df['year'].min():.0f}-{df['year'].max():.0f} | "
          f"industries {df['SETIndustry'].nunique()}")
    if not expanded:
        print("  (paper reports 206 issuers and 13,072 firm-months)")

    for key, (name, fn) in TABLES.items():
        if want and key not in want:
            continue
        try:
            tbl = fn(df)
            show(tbl)
            if csv_dir:
                p = os.path.join(csv_dir, f"table{key}.csv")
                tbl.to_csv(p, index=False)
                print(f"  -> {p}")
        except Exception as ex:
            print(f"\n[table {key} ({name}) failed: {ex}]")
    print("\nDone.")


if __name__ == "__main__":
    main()
