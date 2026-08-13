# -*- coding: utf-8 -*-
"""
benchmark_all.py -- Approach 1 vs Approach 2 vs basic Deep-Learning models.

Answers two questions on ONE common walk-forward out-of-sample split:

  (A) PREDICTION performance   -- ROC-AUC, PR-AUC, Brier, MCC/F1/precision/recall
                                  at a MATCHED alarm budget, and early-warning lead time.
  (B) FINANCIAL / ECONOMIC     -- Sarlin (2013) policymaker loss & relative usefulness,
                                  plus a THB cost-benefit (loss avoided vs review cost).

Models
------
Approach 1 (dynamic survival: hazard h(t|X) -> PD_3M = 1 - prod(1-h)):
    A1-Logistic     discrete-time logistic hazard          (survivor2.py)
    A1-XGBoost      gradient-boosted-trees hazard          (machine_survior.py)
    A1-MLP  [DL]    PyTorch MLP(32,16) hazard              (basic DL, architech Path 2)

Approach 2 (static classifier on the 33 features -> P(event within 3 months)):
    A2-Logistic, A2-RandomForest, A2-XGBoost               (proposal benchmark)
    A2-MLP  [DL]    PyTorch MLP(32,16) classifier          (basic DL)
    A2-GRU  [DL]    PyTorch GRU over the last L months     (basic sequence DL)

All models are trained on exactly the same expanding-window folds and scored on the
same firm-months predicting the same label (a real DP/RS credit event within the next
3 calendar months), so the comparison is like-for-like.

Run:  python benchmark_all.py                 (full run + save to SQLite)
      python benchmark_all.py --fast          (fewer folds / epochs)
      python benchmark_all.py --no-save
"""
from __future__ import annotations
import os
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)
import xgboost as xgb

import survival
import survivor2 as s2
from load_bond import BOND_FEATURES
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DATA_ROOT, "cmdf_credit.db")
T_PRED = "benchmark_prediction"
T_ECON = "benchmark_economics"
T_SUMMARY = "benchmark_summary"

HORIZON = 3                  # forward window (calendar months)
ALARM_BUDGET = 0.10          # matched alarm budget: flag the riskiest 10% of firm-months
SEQ_LEN = 6                  # months of history for the GRU

# ---- economic assumptions (all editable; reported with the results) ----------
EAD_MTHB = 1000.0            # exposure at default per issuer, million THB
LGD = 0.45                   # loss given default (Basel foundation IRB, senior unsecured)
MITIGATION = 0.30            # share of loss avoided when warned early enough to act
REVIEW_COST_MTHB = 0.05      # cost of reviewing one flagged firm-month, million THB


# ============================================================ data ============
def load_panel():
    panel = s2.load_bond_dated(horizon=HORIZON)
    keep = ["firm_id", "month_index", "month_year", "event", "default_3m"] + BOND_FEATURES
    return panel[keep].copy()


def _prep(train, test, feats):
    """median-impute + standardise using TRAIN statistics only."""
    med = train[feats].median()
    mu = train[feats].fillna(med).mean()
    sd = train[feats].fillna(med).std().replace(0, 1.0) + 1e-9
    def tf(d):
        x = d[feats].fillna(med)
        return np.nan_to_num(((x - mu) / sd).to_numpy(dtype=np.float32), nan=0.0)
    return tf(train), tf(test)


# ==================================================== Approach 1 fitters ======
def _a1_scores(train, test, feats, estimator):
    """Fit a monthly-hazard model, then convert to PD_3M for the test rows."""
    Xtr, Xte = _prep(train, test, feats)
    tmu, tsd = train["month_index"].mean(), train["month_index"].std() + 1e-9
    def tbasis(mi):
        m = ((np.asarray(mi, float) - tmu) / tsd)
        return np.column_stack([m, m ** 2, m ** 3]).astype(np.float32)
    Atr = np.hstack([tbasis(train["month_index"]), Xtr])
    ytr = train["event"].astype(int).to_numpy()
    model = estimator(Atr, ytr)
    S = np.ones(len(test))
    for k in range(1, HORIZON + 1):                     # X held fixed, time shifted
        h = model(np.hstack([tbasis(test["month_index"] + k), Xte]))
        S *= (1.0 - np.clip(h, 1e-9, 0.999))
    return 1.0 - S


def a1_logistic(train, test, feats):
    def est(A, y):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(A, y)
        return lambda B: clf.predict_proba(B)[:, 1]
    return _a1_scores(train, test, feats, est)


def a1_xgboost(train, test, feats):
    def est(A, y):
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        clf = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                min_child_weight=3, scale_pos_weight=spw,
                                objective="binary:logistic", eval_metric="logloss",
                                n_jobs=4, random_state=0, verbosity=0).fit(A, y)
        return lambda B: clf.predict_proba(B)[:, 1]
    return _a1_scores(train, test, feats, est)


def a1_mlp(train, test, feats, epochs=12):
    def est(A, y):
        net = _fit_torch_mlp(A, y, epochs=epochs)
        return lambda B: _torch_predict(net, B)
    return _a1_scores(train, test, feats, est)


# ==================================================== Approach 2 fitters ======
def a2_logistic(train, test, feats):
    Xtr, Xte = _prep(train, test, feats)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
        Xtr, train["default_3m"].astype(int))
    return clf.predict_proba(Xte)[:, 1]


def a2_rf(train, test, feats):
    Xtr, Xte = _prep(train, test, feats)
    clf = RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced",
                                 n_jobs=4, random_state=0).fit(
        Xtr, train["default_3m"].astype(int))
    return clf.predict_proba(Xte)[:, 1]


def a2_xgboost(train, test, feats):
    Xtr, Xte = _prep(train, test, feats)
    y = train["default_3m"].astype(int).to_numpy()
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
                            scale_pos_weight=spw, objective="binary:logistic",
                            eval_metric="aucpr", n_jobs=4, random_state=0,
                            verbosity=0).fit(Xtr, y)
    return clf.predict_proba(Xte)[:, 1]


def a2_mlp(train, test, feats, epochs=12):
    Xtr, Xte = _prep(train, test, feats)
    net = _fit_torch_mlp(Xtr, train["default_3m"].astype(int).to_numpy(), epochs=epochs)
    return _torch_predict(net, Xte)


def a2_gru(train, test, feats, epochs=8, seq_len=SEQ_LEN):
    """basic sequence DL: GRU over the firm's last `seq_len` monthly feature vectors."""
    import torch
    import torch.nn as nn
    Xtr, Xte = _prep(train, test, feats)

    def windows(df, X):
        """(n, seq_len, n_feat) tensor: pad the start of each firm with its first row."""
        idx = {}
        for pos, (f, m) in enumerate(zip(df["firm_id"].to_numpy(), df["month_index"].to_numpy())):
            idx[(f, m)] = pos
        out = np.empty((len(df), seq_len, X.shape[1]), dtype=np.float32)
        firms = df["firm_id"].to_numpy(); months = df["month_index"].to_numpy()
        for pos in range(len(df)):
            f, m = firms[pos], months[pos]
            for j, lag in enumerate(range(seq_len - 1, -1, -1)):
                out[pos, j] = X[idx.get((f, m - lag), pos)]
        return out

    Wtr, Wte = windows(train, Xtr), windows(test, Xte)
    ytr = train["default_3m"].astype(int).to_numpy()

    torch.manual_seed(0)
    net = nn.GRU(input_size=Wtr.shape[2], hidden_size=32, num_layers=1, batch_first=True)
    head = nn.Linear(32, 1)
    pos = max(ytr.sum(), 1); neg = max(len(ytr) - ytr.sum(), 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-3)
    Xt = torch.from_numpy(Wtr); yt = torch.from_numpy(ytr.astype(np.float32)).view(-1, 1)
    n, bs = len(Xt), 2048
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            o, _h = net(Xt[b])
            loss = lossf(head(o[:, -1, :]), yt[b])
            loss.backward(); opt.step()
    net.eval(); head.eval()
    with torch.no_grad():
        outs = []
        Xe = torch.from_numpy(Wte)
        for i in range(0, len(Xe), 4096):
            o, _h = net(Xe[i:i + 4096])
            outs.append(torch.sigmoid(head(o[:, -1, :])).view(-1).numpy())
    return np.concatenate(outs)


# ------------------------------------------------------------ torch MLP ------
def _fit_torch_mlp(X, y, hidden=(32, 16), epochs=12, lr=1e-3, batch=4096, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]; d = h
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers)
    y = np.asarray(y).astype(np.float32)
    pos = max(y.sum(), 1); neg = max(len(y) - y.sum(), 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(y).view(-1, 1)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward(); opt.step()
    net.eval()
    return net


def _torch_predict(net, X):
    import torch
    with torch.no_grad():
        out = []
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        for i in range(0, len(Xt), 8192):
            out.append(torch.sigmoid(net(Xt[i:i + 8192])).view(-1).numpy())
    return np.concatenate(out)


# ================================================== walk-forward evaluation ===
def walk_forward(panel, fit_predict, feats, n_folds=4, min_train_frac=0.45):
    """expanding window: every score is produced by a model trained on earlier months only."""
    months = np.sort(panel["month_index"].unique())
    start = months[int(len(months) * min_train_frac)]
    cuts = np.unique(np.linspace(start, months[-1] + 1, n_folds + 1).astype(int))
    parts = []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        train = panel[panel["month_index"] < c0]
        test = panel[(panel["month_index"] >= c0) & (panel["month_index"] < c1)]
        if len(test) == 0 or train["event"].sum() < 2 or train["default_3m"].sum() < 2:
            continue
        sc = fit_predict(train, test, feats)
        t = test[["firm_id", "month_index", "month_year", "event", "default_3m"]].copy()
        t["score"] = np.asarray(sc, dtype=float)
        parts.append(t)
    return pd.concat(parts).reset_index(drop=True) if parts else None


# ============================================== prediction & economic scoring =
def prediction_metrics(sc, budget=ALARM_BUDGET):
    y = sc["default_3m"].astype(int).to_numpy()
    p = sc["score"].to_numpy()
    # matched alarm budget by RANK (top-k), so heavy score ties -- e.g. PD_3M
    # saturating at 1.0 -- cannot silently blow the budget past `budget`.
    k = max(int(round(len(p) * budget)), 1)
    order = np.argsort(-p, kind="stable")[:k]
    f = np.zeros(len(p), dtype=int); f[order] = 1
    thr = float(p[order[-1]])                          # score of the k-th ranked row
    tp = int(((f == 1) & (y == 1)).sum()); fp = int(((f == 1) & (y == 0)).sum())
    fn = int(((f == 0) & (y == 1)).sum()); tn = int(((f == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        pr_auc=average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        brier=float(np.mean((p - y) ** 2)),
        mcc=matthews_corrcoef(y, f) if len(np.unique(f)) > 1 else 0.0,
        f1=2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        precision=prec, recall=rec, threshold=thr,
        flagged_pct=float(f.mean() * 100),             # actual alarm volume (should ≈ budget)
        ties_at_thr=int((p == thr).sum()),             # how saturated the score is
        tp=tp, fp=fp, fn=fn, tn=tn, n_rows=int(len(sc)), n_pos=int(y.sum()))


def lead_time_stats(sc, thr):
    """days between the start of the final sustained alarm run and the real event."""
    leads, detected, missed = [], 0, 0
    ev_firms = sc.loc[sc["event"] == 1, "firm_id"].unique()
    for f in ev_firms:
        g = sc[sc["firm_id"] == f].sort_values("month_index")
        e = g[g["event"] == 1]
        if e.empty:
            continue
        edate = e["month_year"].iloc[0]
        pre = g[g["month_year"] < edate].reset_index(drop=True)
        if pre.empty:
            continue
        inb = (pre["score"] >= thr).to_numpy()
        if not inb.any():
            missed += 1; continue
        j = int(np.where(inb)[0][-1])
        while j - 1 >= 0 and inb[j - 1]:
            j -= 1
        detected += 1
        leads.append((edate - pre["month_year"].iloc[j]).days)
    return dict(n_event_firms=int(len(ev_firms)), detected=detected, missed=missed,
                median_lead_days=float(np.median(leads)) if leads else float("nan"),
                mean_lead_days=float(np.mean(leads)) if leads else float("nan"))


def economic_metrics(m, lt):
    """(1) Sarlin (2013) usefulness -- firm-month level, with the policymaker weight mu
           DERIVED FROM the cost assumptions so the two economic views are consistent.
       (2) THB cost-benefit -- event level (a firm is either warned in time or not),
           which is how a credit desk actually books the saving."""
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    P1 = (tp + fn) / max(m["n_rows"], 1)                # unconditional P(event within 3m)
    P2 = 1.0 - P1
    T1 = fn / max(tp + fn, 1)                           # miss rate
    T2 = fp / max(fp + tn, 1)                           # false-alarm rate

    avoidable = EAD_MTHB * LGD * MITIGATION             # MTHB saved per early warning
    # mu = relative preference for avoiding a miss, implied by the money at stake
    mu = avoidable / (avoidable + REVIEW_COST_MTHB)
    L = mu * T1 * P1 + (1 - mu) * T2 * P2
    base = min(mu * P1, (1 - mu) * P2)                  # loss of the best trivial rule
    Ua = base - L
    Ur = Ua / base if base > 0 else float("nan")

    benefit = lt["detected"] * avoidable
    foregone = lt["missed"] * avoidable
    cost = fp * REVIEW_COST_MTHB
    net = benefit - cost
    return dict(miss_rate=T1, false_alarm_rate=T2, mu_implied=mu, sarlin_loss=L,
                usefulness_abs=Ua, usefulness_rel=Ur,
                benefit_mthb=benefit, review_cost_mthb=cost, net_benefit_mthb=net,
                foregone_mthb=foregone,
                roi=(net / cost) if cost > 0 else float("nan"),
                cost_per_detection_mthb=(cost / lt["detected"]) if lt["detected"] else float("nan"))


# ============================================================== main run ======
MODELS = [
    ("A1-Logistic", "Approach 1", "survival hazard - logistic", a1_logistic),
    ("A1-XGBoost", "Approach 1", "survival hazard - GBT", a1_xgboost),
    ("A1-MLP [DL]", "Approach 1 (DL)", "survival hazard - PyTorch MLP(32,16)", a1_mlp),
    ("A2-Logistic", "Approach 2", "static classifier", a2_logistic),
    ("A2-RandomForest", "Approach 2", "static classifier", a2_rf),
    ("A2-XGBoost", "Approach 2", "static classifier", a2_xgboost),
    ("A2-MLP [DL]", "Approach 2 (DL)", "static PyTorch MLP(32,16)", a2_mlp),
    ("A2-GRU [DL]", "Approach 2 (DL)", f"sequence GRU(32), {SEQ_LEN}m history", a2_gru),
]


def run_benchmark(fast=False, verbose=True):
    folds = 3 if fast else 4
    panel = load_panel()
    feats = list(BOND_FEATURES)
    if verbose:
        print(f"panel {len(panel):,} firm-months | {panel['firm_id'].nunique()} firms | "
              f"events {int(panel['event'].sum())} | 3m-positives {int(panel['default_3m'].sum())}")
        print(f"walk-forward folds={folds}, matched alarm budget={ALARM_BUDGET:.0%}\n")

    pred_rows, econ_rows = [], []
    for name, group, desc, fn in MODELS:
        t0 = time.time()
        if verbose:
            print(f"  fitting {name:18s} ...", end="", flush=True)
        try:
            f = fn
            if fast and "[DL]" in name:
                f = (lambda tr, te, ft, _f=fn: _f(tr, te, ft, epochs=5))
            sc = walk_forward(panel, f, feats, n_folds=folds)
            if sc is None or sc.empty:
                if verbose:
                    print("  no folds"); continue
            m = prediction_metrics(sc)
            lt = lead_time_stats(sc, m["threshold"])
            ec = economic_metrics(m, lt)
            pred_rows.append(dict(model=name, group=group, detail=desc, **m, **lt,
                                  seconds=round(time.time() - t0, 1)))
            econ_rows.append(dict(model=name, group=group, **ec))
            if verbose:
                print(f" AUC {m['auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  "
                      f"rec {m['recall']:.2f}  net {ec['net_benefit_mthb']:+.0f} MTHB  "
                      f"({time.time()-t0:.0f}s)")
        except Exception as ex:
            if verbose:
                print(f"  FAILED: {ex}")
    pred = pd.DataFrame(pred_rows)
    econ = pd.DataFrame(econ_rows)

    # ---- group-level answer: which APPROACH wins -------------------------
    merged = pred.merge(econ[["model", "usefulness_rel", "net_benefit_mthb", "roi"]], on="model")
    merged["approach"] = np.where(merged["group"].str.startswith("Approach 1"), "Approach 1", "Approach 2")
    grp = merged.groupby("approach").agg(
        best_auc=("auc", "max"), mean_auc=("auc", "mean"),
        best_pr_auc=("pr_auc", "max"), best_recall=("recall", "max"),
        best_usefulness=("usefulness_rel", "max"),
        best_net_mthb=("net_benefit_mthb", "max"),
        median_lead_days=("median_lead_days", "median")).reset_index()

    best_pred = merged.loc[merged["auc"].idxmax()]
    best_econ = merged.loc[merged["net_benefit_mthb"].idxmax()]
    a1 = merged[merged["approach"] == "Approach 1"]
    a2 = merged[merged["approach"] == "Approach 2"]
    dl = merged[merged["group"].str.contains(r"\(DL\)")]
    nodl = merged[~merged["group"].str.contains(r"\(DL\)")]
    summary = dict(
        run_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_firm_months=int(len(panel)), n_firms=int(panel["firm_id"].nunique()),
        n_events=int(panel["event"].sum()), folds=folds, alarm_budget=ALARM_BUDGET,
        best_prediction_model=str(best_pred["model"]), best_prediction_auc=float(best_pred["auc"]),
        best_economic_model=str(best_econ["model"]),
        best_economic_net_mthb=float(best_econ["net_benefit_mthb"]),
        a1_best_auc=float(a1["auc"].max()), a2_best_auc=float(a2["auc"].max()),
        a1_best_net=float(a1["net_benefit_mthb"].max()), a2_best_net=float(a2["net_benefit_mthb"].max()),
        dl_best_auc=float(dl["auc"].max()) if len(dl) else float("nan"),
        classic_best_auc=float(nodl["auc"].max()) if len(nodl) else float("nan"),
        ead_mthb=EAD_MTHB, lgd=LGD, mitigation=MITIGATION, review_cost_mthb=REVIEW_COST_MTHB,
    )
    winner_pred = "Approach 1" if summary["a1_best_auc"] >= summary["a2_best_auc"] else "Approach 2"
    winner_econ = "Approach 1" if summary["a1_best_net"] >= summary["a2_best_net"] else "Approach 2"
    dl_helps = summary["dl_best_auc"] > summary["classic_best_auc"]
    summary["winner_prediction"] = winner_pred
    summary["winner_economic"] = winner_econ
    summary["dl_beats_classic"] = bool(dl_helps)
    summary["verdict"] = (
        f"Prediction: {winner_pred} leads (best AUC {max(summary['a1_best_auc'], summary['a2_best_auc']):.3f} "
        f"by {summary['best_prediction_model']}). "
        f"Economics: {winner_econ} leads (best net benefit "
        f"{max(summary['a1_best_net'], summary['a2_best_net']):,.0f} MTHB by {summary['best_economic_model']}). "
        f"Basic DL {'does' if dl_helps else 'does NOT'} beat the classical models "
        f"(DL best AUC {summary['dl_best_auc']:.3f} vs classical {summary['classic_best_auc']:.3f}).")
    return dict(prediction=pred, economics=econ, groups=grp, summary=summary)


def save_to_sqlite(res, db=DB):
    con = sqlite3.connect(db)
    res["prediction"].to_sql(T_PRED, con, if_exists="replace", index=False)
    res["economics"].to_sql(T_ECON, con, if_exists="replace", index=False)
    pd.DataFrame([res["summary"]]).to_sql(T_SUMMARY, con, if_exists="replace", index=False)
    con.commit(); con.close()


def load_from_sqlite(db=DB):
    con = sqlite3.connect(db)
    try:
        p = pd.read_sql_query(f"SELECT * FROM {T_PRED}", con)
        e = pd.read_sql_query(f"SELECT * FROM {T_ECON}", con)
        s = pd.read_sql_query(f"SELECT * FROM {T_SUMMARY} LIMIT 1", con)
    except Exception:
        p = e = s = pd.DataFrame()
    finally:
        con.close()
    return p, e, s


def print_report(res):
    p, e, g, s = res["prediction"], res["economics"], res["groups"], res["summary"]
    print("\n" + "=" * 104)
    print("TABLE 1 -- PREDICTION PERFORMANCE  (walk-forward out-of-sample, matched "
          f"{ALARM_BUDGET:.0%} alarm budget)")
    print("=" * 104)
    print(f"  {'Model':18s} {'Group':17s} {'AUC':>6s} {'PR-AUC':>7s} {'Brier':>7s} {'MCC':>6s} "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Flag%':>6s} {'Detect':>8s} {'Lead(d)':>8s}")
    print("  " + "-" * 108)
    for _, r in p.sort_values("auc", ascending=False).iterrows():
        lead = "-" if r["median_lead_days"] != r["median_lead_days"] else f"{r['median_lead_days']:.0f}"
        print(f"  {r['model']:18s} {r['group']:17s} {r['auc']:6.3f} {r['pr_auc']:7.3f} "
              f"{r['brier']:7.4f} {r['mcc']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f1']:6.3f} {r['flagged_pct']:6.1f} {r['detected']:3d}/{r['n_event_firms']:<4d} {lead:>8s}")

    print("\n" + "=" * 104)
    print("TABLE 2 -- FINANCIAL / ECONOMIC PERFORMANCE")
    print(f"  assumptions: EAD {EAD_MTHB:,.0f} MTHB/issuer, LGD {LGD:.0%}, "
          f"loss mitigated when warned {MITIGATION:.0%}, review cost {REVIEW_COST_MTHB} MTHB per flagged firm-month")
    print(f"  -> saving per early warning {EAD_MTHB*LGD*MITIGATION:,.0f} MTHB; implied Sarlin "
          f"mu = {e['mu_implied'].iloc[0]:.5f} (derived from these costs, not assumed)")
    print("  NOTE: Sarlin Ur is firm-month level; NET (MTHB) is event level "
          "(a firm is either warned in time or not).")
    print("=" * 104)
    print(f"  {'Model':18s} {'MissRate':>9s} {'FA-Rate':>8s} {'Sarlin Ur':>10s} "
          f"{'Benefit':>10s} {'Cost':>9s} {'NET (MTHB)':>12s} {'ROI':>7s}")
    print("  " + "-" * 100)
    for _, r in e.merge(p[["model", "auc"]], on="model").sort_values("net_benefit_mthb", ascending=False).iterrows():
        print(f"  {r['model']:18s} {r['miss_rate']:9.3f} {r['false_alarm_rate']:8.3f} "
              f"{r['usefulness_rel']:10.3f} {r['benefit_mthb']:10,.0f} {r['review_cost_mthb']:9,.0f} "
              f"{r['net_benefit_mthb']:12,.0f} {r['roi']:7.2f}")

    print("\n" + "=" * 104)
    print("TABLE 3 -- APPROACH-LEVEL COMPARISON")
    print("=" * 104)
    print(f"  {'Approach':12s} {'best AUC':>9s} {'mean AUC':>9s} {'best PR-AUC':>12s} "
          f"{'best recall':>12s} {'best Ur':>9s} {'best NET':>12s} {'lead(d)':>9s}")
    print("  " + "-" * 100)
    for _, r in g.iterrows():
        print(f"  {r['approach']:12s} {r['best_auc']:9.3f} {r['mean_auc']:9.3f} "
              f"{r['best_pr_auc']:12.3f} {r['best_recall']:12.3f} {r['best_usefulness']:9.3f} "
              f"{r['best_net_mthb']:12,.0f} {r['median_lead_days']:9.0f}")

    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    print(f"  best prediction : {s['best_prediction_model']}  (AUC {s['best_prediction_auc']:.3f})")
    print(f"  best economics  : {s['best_economic_model']}  "
          f"(net {s['best_economic_net_mthb']:,.0f} MTHB)")
    print(f"  Approach 1 best AUC {s['a1_best_auc']:.3f} | Approach 2 best AUC {s['a2_best_auc']:.3f}")
    print(f"  DL best AUC {s['dl_best_auc']:.3f} | classical best AUC {s['classic_best_auc']:.3f}")
    print(f"\n  {s['verdict']}")


def main():
    fast = "--fast" in sys.argv
    res = run_benchmark(fast=fast, verbose=True)
    print_report(res)
    if "--no-save" not in sys.argv:
        save_to_sqlite(res)
        print(f"\nSaved to SQLite: {T_PRED}, {T_ECON}, {T_SUMMARY}  ({DB})")
    print("\nDone.")


if __name__ == "__main__":
    main()
