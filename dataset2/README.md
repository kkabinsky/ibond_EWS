# Dataset2 edition of the credit early-warning app

Self-contained apart from `thaibma_paths.py`, which only locates the database.
Nothing outside this folder is changed; the database is opened read-only.

## Data first

```
python dataset/build_db.py
```

run once from the repository root. It writes `cmdf_credit.db` there, including
the Dataset2 table `ibond_33features_panel_941firm`, and every program in this
folder finds it by itself.

## Run

```
python dataset2/app_dataset2.py             GUI
python dataset2/app_dataset2.py --selftest  backend check, prints metrics and figure sizes
python dataset2/app_dataset2.py --uitest    builds every panel headless (fits all nine models, about 13 min)
python dataset2/app_dataset2.py --summary   defaults caught by every model, no window
python dataset2/data_layer.py --selftest    data check only
python dataset2/leadtime_allmethods.py run  1-3 month lead time of nine models (about 11 min)
python dataset2/pd_curves.py build          3-month PD of Approach 1 and Approach 2
python run.py pd11_panel performance        the table the PD menus of app.py show
```

On start it prints

```
Loading Dataset2 from <repository>\cmdf_credit.db
Rows 187,007 | issuers 941 | positive months 124 | explicit event issuers 31 | all dated/status event issuers 32
```

## Files

| File | What it holds |
|---|---|
| `data_layer.py` | Dataset2 loading, the event catalogue, the nine classifiers, out-of-fold scoring, lead time, risk bands and momentum |
| `app_dataset2.py` | the Flet GUI: sidebar navigation, ten panels, all figures |
| `leadtime_allmethods.py` | the standalone nine-model lead-time engine (same file as `dataset2_leadtime_standalone.py` in iBond_LIME and leadning_time) |
| `pd_curves.py` | 3-month PD of Approach 1 (pooled logit) and Approach 2 (gradient boosting), per-issuer PD charts |
| `run_all_methods.py` | every classifier on Dataset2, tables written to `out/` |
| `make_pr_figure.py` | precision-recall of every method in one figure |

## Data

* table `ibond_33features_panel_941firm` in `cmdf_credit.db`
* target `y_pre3m`, the 1 to 3 month pre-event window
* 30 feature columns: liquidity, profitability, leverage, coverage, macro and ESG
* months 1984-01 to 2026-08

Validation is `StratifiedGroupKFold` grouped on `issuer_code` with seed 42, so
every month of one firm stays inside one fold and a firm the model has already
seen cannot inflate the held-out score.

## Panels

| Section | Panel |
|---|---|
| | Run summary: defaults caught by every model |
| RISK MODELS | Approach 1: Survival Dashboard |
| | Approach 2: XGBoost + SHAP |
| | Firm Shock & PD Threshold |
| | Momentum & Hyperbolic Boundary |
| LEAD TIME & EVALUATION | Lead Time: 1-3M + Persistent |
| | Compare Models: A1 vs A2 |
| | Model Zoo: all classifiers |
| DATASET2 | Data Inspector & SQLite |
| | Dataset2 Summary |

The sidebar carries a model picker and a workload field. Apply refits and
redraws the current panel. Each model and workload pair is fitted once and
cached for the session.

## Lead time

An alarm counts as actionable only when it fires one to three months before the
event month. An alarm in the event month is not a warning, and a run that
starts earlier than three months out is reported separately as a persistent run
rather than counted as a detection.

Metric version `lead_metrics_actionable_1_3m_persistent_v1`.

Installation, every command, and the results of running them on Dataset2 are in
`docs/githup_manual.pdf` (Thai).
