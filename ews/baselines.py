"""
baselines.py - anomaly scoring of the factor panel with the eight baseline
detectors requested in the ExpoGAF-AnoNet review
(``ref_gaf_latex_source/answer_iran_market_crash_v11_reviewer4.tex``):

    1. Isolation Forest        5. OmniAnomaly
    2. One-Class SVM           6. USAD
    3. Deep SVDD               7. TranAD
    4. DAGMM                   8. Anomaly Transformer

Protocol (identical for every detector, so the comparison is fair):

  * inputs are the standardised factors of ``feature_bond.xlsx`` arranged into
    per-firm sliding windows of length ``W``;
  * the label of a window is ``target_RS`` at its last row (the RS default
    event), and the anomaly score is compared against that label;
  * training uses **normal windows of training firms only** (unsupervised);
  * the split is **leave-firms-out** - a firm never appears on both sides,
    because the RS target is rare and clustered inside a few firms;
  * evaluation reports ROC-AUC and PR-AUC against the base rate.

The deep detectors are compact reference implementations of the published
architectures (Deep SVDD, DAGMM, OmniAnomaly, USAD, TranAD, Anomaly
Transformer), not the original authors' code; they reproduce the scoring
principle of each method rather than every training heuristic.

Standalone check:
    python baselines.py --selftest
"""

from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import koopman_gaf as kg
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DEFAULT = os.path.join(DATA_ROOT, "cmdf_credit.db")
BASELINE_TABLE = "baseline_anomaly"

BASELINES = ("IsolationForest", "OneClassSVM", "DeepSVDD", "DAGMM",
             "OmniAnomaly", "USAD", "TranAD", "AnomalyTransformer")


# ------------------------------------------------------------- windows -------
def build_windows(df: pd.DataFrame | None = None, window: int = 8,
                  target: str = "target_RS"):
    """(n_windows, window, n_factors) plus label, firm id and end-date."""
    df = kg.load_bond_panel() if df is None else df
    feats = kg.feature_columns(df)
    raw = df[feats].astype(float)
    mu, sd = raw.mean(), raw.std().replace(0, 1.0)
    Z = ((raw - mu) / sd).fillna(0.0).to_numpy(dtype=np.float32)
    y_all = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy().astype(int)
    firm_all = df["firm_id"].astype(str).to_numpy()

    Xs, ys, fs, idx = [], [], [], []
    order = np.argsort(firm_all, kind="stable")
    for firm in pd.unique(firm_all[order]):
        pos = np.where(firm_all == firm)[0]
        if len(pos) < window:
            continue
        for e in range(window - 1, len(pos)):
            sl = pos[e - window + 1:e + 1]
            Xs.append(Z[sl]); ys.append(y_all[pos[e]]); fs.append(firm); idx.append(pos[e])
    if not Xs:
        raise ValueError("no firm has at least %d observations" % window)
    return (np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=int),
            np.asarray(fs), np.asarray(idx), feats)


# ---------------------------------------------------------- torch helpers ----
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _train_loop(model, params, Xtr, epochs, batch, lr, step_fn, seed=42):
    torch, _ = _torch()
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    n = len(Xtr)
    for _ in range(int(epochs)):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            xb = Xtr[perm[i:i + batch]]
            loss = step_fn(xb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ------------------------------------------------------------- detectors -----
def _fit_score_isoforest(Xtr, Xall, seed=42, **kw):
    from sklearn.ensemble import IsolationForest
    a, b = Xtr.reshape(len(Xtr), -1), Xall.reshape(len(Xall), -1)
    m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(a)
    return -m.score_samples(b)


def _fit_score_ocsvm(Xtr, Xall, seed=42, max_train=8000, **kw):
    from sklearn.svm import OneClassSVM
    rng = np.random.default_rng(seed)
    a = Xtr.reshape(len(Xtr), -1)
    if len(a) > max_train:                       # OC-SVM is O(n^2); subsample
        a = a[rng.choice(len(a), max_train, replace=False)]
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(a)
    return -m.decision_function(Xall.reshape(len(Xall), -1))


def _fit_score_deepsvdd(Xtr, Xall, epochs=8, batch=512, lr=1e-3, hidden=64, zdim=16, **kw):
    torch, nn = _torch()
    d = Xtr.shape[1] * Xtr.shape[2]
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, zdim, bias=False))
    xtr = torch.from_numpy(Xtr.reshape(len(Xtr), -1))
    with torch.no_grad():
        c = net(xtr).mean(0)
        c[c.abs() < 1e-6] = 1e-6                 # avoid the trivial solution
    _train_loop(net, net.parameters(), xtr, epochs, batch, lr,
                lambda xb: ((net(xb) - c) ** 2).sum(1).mean())
    with torch.no_grad():
        s = ((net(torch.from_numpy(Xall.reshape(len(Xall), -1))) - c) ** 2).sum(1)
    return s.numpy()


def _fit_score_dagmm(Xtr, Xall, epochs=8, batch=512, lr=1e-3, hidden=64, zdim=4,
                     n_gmm=4, **kw):
    """Autoencoder + estimation network; score = GMM sample energy."""
    torch, nn = _torch()
    d = Xtr.shape[1] * Xtr.shape[2]
    enc = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, zdim))
    dec = nn.Sequential(nn.Linear(zdim, hidden), nn.Tanh(), nn.Linear(hidden, d))
    est = nn.Sequential(nn.Linear(zdim + 1, hidden), nn.Tanh(), nn.Dropout(0.2),
                        nn.Linear(hidden, n_gmm), nn.Softmax(dim=1))
    params = list(enc.parameters()) + list(dec.parameters()) + list(est.parameters())

    def latent(xb):
        z = enc(xb); xh = dec(z)
        cos = torch.nn.functional.cosine_similarity(xb, xh, dim=1).unsqueeze(1)
        return z, xh, torch.cat([z, cos], dim=1)

    def step(xb):
        z, xh, zc = latent(xb)
        gamma = est(zc)
        rec = ((xb - xh) ** 2).mean()
        # keep the mixture responsibilities from collapsing onto one component
        pk = gamma.mean(0)
        ent = -(pk * torch.log(pk + 1e-8)).sum()
        return rec - 0.05 * ent

    xtr = torch.from_numpy(Xtr.reshape(len(Xtr), -1))
    _train_loop(None, params, xtr, epochs, batch, lr, step)

    with torch.no_grad():
        _, _, zc_tr = latent(xtr)
        g = est(zc_tr)
        w = g.sum(0) + 1e-8
        mu = (g.t() @ zc_tr) / w.unsqueeze(1)
        cov = []
        for k in range(g.shape[1]):
            dxk = zc_tr - mu[k]
            cov.append((g[:, k].unsqueeze(1) * dxk).t() @ dxk / w[k]
                       + 1e-3 * torch.eye(zc_tr.shape[1]))
        cov = torch.stack(cov)
        _, _, zc_all = latent(torch.from_numpy(Xall.reshape(len(Xall), -1)))
        energies = []
        phi = (w / w.sum())
        for k in range(len(cov)):
            diff = zc_all - mu[k]
            inv = torch.linalg.inv(cov[k])
            m = (diff @ inv * diff).sum(1)
            logdet = torch.logdet(cov[k])
            energies.append(torch.log(phi[k] + 1e-8) - 0.5 * (m + logdet))
        e = torch.stack(energies, 1)
        score = -torch.logsumexp(e, dim=1)
    return score.numpy()


def _fit_score_usad(Xtr, Xall, epochs=8, batch=512, lr=1e-3, hidden=64, zdim=16,
                    alpha=0.5, **kw):
    """Shared encoder with two decoders trained adversarially (USAD)."""
    torch, nn = _torch()
    d = Xtr.shape[1] * Xtr.shape[2]
    enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, zdim), nn.ReLU())
    d1 = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Linear(hidden, d))
    d2 = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Linear(hidden, d))
    params = list(enc.parameters()) + list(d1.parameters()) + list(d2.parameters())

    state = {"ep": 1}

    def step(xb):
        n = float(state["ep"])
        z = enc(xb)
        w1, w2 = d1(z), d2(z)
        w3 = d2(enc(w1))
        l1 = (1 / n) * ((xb - w1) ** 2).mean() + (1 - 1 / n) * ((xb - w3) ** 2).mean()
        l2 = (1 / n) * ((xb - w2) ** 2).mean() - (1 - 1 / n) * ((xb - w3) ** 2).mean()
        return l1 + l2

    xtr = torch.from_numpy(Xtr.reshape(len(Xtr), -1))
    for ep in range(int(epochs)):
        state["ep"] = ep + 1
        _train_loop(None, params, xtr, 1, batch, lr, step)

    with torch.no_grad():
        xa = torch.from_numpy(Xall.reshape(len(Xall), -1))
        z = enc(xa); w1 = d1(z); w3 = d2(enc(w1))
        s = alpha * ((xa - w1) ** 2).mean(1) + (1 - alpha) * ((xa - w3) ** 2).mean(1)
    return s.numpy()


def _fit_score_omnianomaly(Xtr, Xall, epochs=8, batch=256, lr=1e-3, hidden=64,
                           zdim=8, **kw):
    """GRU stochastic recurrent VAE; score = negative reconstruction likelihood."""
    torch, nn = _torch()
    F = Xtr.shape[2]

    class Omni(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(F, hidden, batch_first=True)
            self.mu = nn.Linear(hidden, zdim); self.lv = nn.Linear(hidden, zdim)
            self.dgru = nn.GRU(zdim, hidden, batch_first=True)
            self.out = nn.Linear(hidden, F)

        def forward(self, x):
            h, _ = self.gru(x)
            mu, lv = self.mu(h), self.lv(h).clamp(-8, 8)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            g, _ = self.dgru(z)
            return self.out(g), mu, lv

    m = Omni()
    xtr = torch.from_numpy(Xtr)

    def step(xb):
        recon, mu, lv = m(xb)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        return ((recon - xb) ** 2).mean() + 0.05 * kl

    _train_loop(m, m.parameters(), xtr, epochs, batch, lr, step)
    with torch.no_grad():
        recon, _, _ = m(torch.from_numpy(Xall))
        s = ((recon - torch.from_numpy(Xall)) ** 2).mean(dim=(1, 2))
    return s.numpy()


def _transformer_core(F, hidden, heads=4, layers=1):
    torch, nn = _torch()
    enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                           dim_feedforward=hidden * 2,
                                           batch_first=True, dropout=0.0)
    return (nn.Linear(F, hidden), nn.TransformerEncoder(enc_layer, layers),
            nn.Linear(hidden, F))


def _fit_score_tranad(Xtr, Xall, epochs=8, batch=256, lr=1e-3, hidden=32, **kw):
    """Transformer with the two-phase reconstruction used by TranAD."""
    torch, nn = _torch()
    F = Xtr.shape[2]
    inp, tr, out = _transformer_core(F, hidden)
    params = list(inp.parameters()) + list(tr.parameters()) + list(out.parameters())

    def recon(x, focus=None):
        h = inp(x if focus is None else x + focus)
        return out(tr(h))

    def step(xb):
        o1 = recon(xb)                                   # phase 1
        focus = (o1 - xb).detach() ** 2
        o2 = recon(xb, focus)                            # phase 2, error-focused
        return ((o1 - xb) ** 2).mean() + ((o2 - xb) ** 2).mean()

    xtr = torch.from_numpy(Xtr)
    _train_loop(None, params, xtr, epochs, batch, lr, step)
    with torch.no_grad():
        xa = torch.from_numpy(Xall)
        o1 = recon(xa)
        o2 = recon(xa, (o1 - xa) ** 2)
        s = 0.5 * (((o1 - xa) ** 2).mean(dim=(1, 2)) + ((o2 - xa) ** 2).mean(dim=(1, 2)))
    return s.numpy()


def _fit_score_anomaly_transformer(Xtr, Xall, epochs=8, batch=256, lr=1e-3,
                                   hidden=32, **kw):
    """Transformer reconstruction plus an association-discrepancy term."""
    torch, nn = _torch()
    F, W = Xtr.shape[2], Xtr.shape[1]
    inp, tr, out = _transformer_core(F, hidden)
    params = list(inp.parameters()) + list(tr.parameters()) + list(out.parameters())

    # prior association: Gaussian kernel over the relative time distance
    pos = torch.arange(W, dtype=torch.float32)
    prior = torch.exp(-((pos[:, None] - pos[None, :]) ** 2) / (2.0 * (W / 4.0) ** 2))
    prior = prior / prior.sum(-1, keepdim=True)

    def encode(x):
        h = inp(x)
        z = tr(h)
        # series association from the encoded sequence
        att = torch.softmax(z @ z.transpose(1, 2) / (hidden ** 0.5), dim=-1)
        return out(z), att

    def step(xb):
        rec, att = encode(xb)
        disc = (att - prior.unsqueeze(0)).abs().mean()
        return ((rec - xb) ** 2).mean() + 0.1 * disc

    xtr = torch.from_numpy(Xtr)
    _train_loop(None, params, xtr, epochs, batch, lr, step)
    with torch.no_grad():
        xa = torch.from_numpy(Xall)
        rec, att = encode(xa)
        rec_err = ((rec - xa) ** 2).mean(dim=(1, 2))
        disc = (att - prior.unsqueeze(0)).abs().mean(dim=(1, 2))
        s = rec_err * (1.0 + disc)
    return s.numpy()


_DISPATCH = {
    "IsolationForest": _fit_score_isoforest,
    "OneClassSVM": _fit_score_ocsvm,
    "DeepSVDD": _fit_score_deepsvdd,
    "DAGMM": _fit_score_dagmm,
    "OmniAnomaly": _fit_score_omnianomaly,
    "USAD": _fit_score_usad,
    "TranAD": _fit_score_tranad,
    "AnomalyTransformer": _fit_score_anomaly_transformer,
}


# ------------------------------------------------------------- runner --------
def run_baseline(name: str, df: pd.DataFrame | None = None, window: int = 8,
                 test_size: float = 0.3, seed: int = 42, epochs: int = 8,
                 max_train: int = 30000) -> dict:
    """Train one detector on normal training-firm windows and score everything."""
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, average_precision_score

    if name not in _DISPATCH:
        raise ValueError("unknown baseline %r" % name)
    X, y, firms, idx, feats = build_windows(df, window=window)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=seed).split(X, y, groups=firms))
    # unsupervised: fit on NORMAL windows of the training firms only
    tr_norm = tr[y[tr] == 0]
    rng = np.random.default_rng(seed)
    if len(tr_norm) > max_train:
        tr_norm = rng.choice(tr_norm, max_train, replace=False)

    scores = _DISPATCH[name](X[tr_norm], X, seed=seed, epochs=epochs)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0,
                           posinf=0.0, neginf=0.0)

    metrics = {"base_rate": float(y[te].mean()), "n_test_pos": float(int(y[te].sum())),
               "n_train_normal": float(len(tr_norm)), "n_windows": float(len(X))}
    if len(np.unique(y[te])) > 1:
        metrics["AUC"] = float(roc_auc_score(y[te], scores[te]))
        metrics["PR_AUC"] = float(average_precision_score(y[te], scores[te]))
        metrics["lift"] = float(metrics["PR_AUC"] / max(metrics["base_rate"], 1e-12))
    else:
        metrics["AUC"] = metrics["PR_AUC"] = metrics["lift"] = float("nan")

    return {"name": name, "scores": scores, "y": y, "firms": firms,
            "test_idx": te, "metrics": metrics, "window": window,
            "n_factors": len(feats)}


def run_all_baselines(df: pd.DataFrame | None = None, window: int = 8,
                      epochs: int = 8, names=None, **kw) -> pd.DataFrame:
    rows = []
    for nm in (names or BASELINES):
        try:
            r = run_baseline(nm, df=df, window=window, epochs=epochs, **kw)
            m = r["metrics"]
            rows.append({"model": nm, "AUC": m["AUC"], "PR_AUC": m["PR_AUC"],
                         "lift": m["lift"], "base_rate": m["base_rate"],
                         "n_test_pos": m["n_test_pos"]})
        except Exception as exc:
            rows.append({"model": nm, "AUC": float("nan"), "PR_AUC": float("nan"),
                         "lift": float("nan"), "base_rate": float("nan"),
                         "n_test_pos": float("nan"), "error": str(exc)[:90]})
    out = pd.DataFrame(rows).sort_values("AUC", ascending=False, na_position="last")
    return out.reset_index(drop=True)


LEAD_TABLE = "baseline_lead_time"


def baseline_lead_time(name: str, df: pd.DataFrame | None = None, window: int = 8,
                       test_size: float = 0.3, seed: int = 42, epochs: int = 8,
                       q: float = 0.95, horizon_days: float = 365.0,
                       max_train: int = 30000) -> dict:
    """Lead time of every RS firm under one baseline detector.

    The alarm threshold is the **train-normal q-quantile** of the detector's own
    anomaly score, as prescribed for the constituent detectors in the reference
    protocol. The lead time follows the same definition used throughout the
    project, ``L = t_event - t_first_alarm``, counting only alarms inside the
    pre-event horizon so that an unrelated crossing years earlier is not
    reported as a multi-year early warning.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if name not in _DISPATCH:
        raise ValueError("unknown baseline %r" % name)
    panel = kg.load_bond_panel() if df is None else df
    X, y, firms, idx, feats = build_windows(panel, window=window)
    dates = pd.to_datetime(panel["dt"], errors="coerce").to_numpy()[idx]

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=seed).split(X, y, groups=firms))
    tr_norm = tr[y[tr] == 0]
    rng = np.random.default_rng(seed)
    if len(tr_norm) > max_train:
        tr_norm = rng.choice(tr_norm, max_train, replace=False)

    scores = np.nan_to_num(np.asarray(_DISPATCH[name](X[tr_norm], X, seed=seed,
                                                      epochs=epochs), dtype=float),
                           nan=0.0, posinf=0.0, neginf=0.0)
    tau = float(np.quantile(scores[tr_norm], q))          # train-normal quantile
    alarm = scores >= tau

    scored = pd.DataFrame({"firm_id": firms, "dt": dates, "y": y,
                           "score": scores, "alarm": alarm})
    rows = []
    for firm, g in scored.groupby("firm_id", sort=False):
        g = g.sort_values("dt")
        ev = g[g["y"] == 1]
        if ev.empty:
            continue
        t_event = ev["dt"].iloc[0]
        start = t_event - np.timedelta64(int(horizon_days), "D")
        pre = g[(g["alarm"]) & (g["dt"] < t_event) & (g["dt"] >= start)]
        if pre.empty:
            # highest score the detector ever gave this firm inside the horizon,
            # so a miss can be told apart from a firm with no data before the event
            win = g[(g["dt"] < t_event) & (g["dt"] >= start)]
            rows.append({"firm_id": firm, "event_date": t_event, "first_alarm": pd.NaT,
                         "lead_time_days": np.nan, "detected": 0, "censored": 0,
                         "n_windows": int(len(win)),
                         "max_score": float(win["score"].max()) if len(win) else np.nan})
        else:
            t_first = pre["dt"].iloc[0]
            lead_d = float((t_event - t_first) / np.timedelta64(1, "D"))
            win = g[(g["dt"] < t_event) & (g["dt"] >= start)]
            # the alarm already stands at the first window of the horizon, so the true
            # lead time is at least this long and the reported value is right-censored
            rows.append({"firm_id": firm, "event_date": t_event, "first_alarm": t_first,
                         "lead_time_days": lead_d, "detected": 1,
                         "censored": int(lead_d >= horizon_days - 31.0),
                         "n_windows": int(len(win)),
                         "max_score": float(win["score"].max())})
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        tbl.insert(0, "model", name)
        tbl = tbl.sort_values("lead_time_days", ascending=False,
                              na_position="last").reset_index(drop=True)
    lead = tbl["lead_time_days"].dropna() if not tbl.empty else pd.Series(dtype=float)
    summary = {
        "model": name, "tau": tau, "q": float(q), "horizon_days": float(horizon_days),
        "event_firms": int(len(tbl)),
        "detected": int(tbl["detected"].sum()) if not tbl.empty else 0,
        "detection_rate": float(tbl["detected"].mean()) if not tbl.empty else 0.0,
        "median_lead_days": float(lead.median()) if len(lead) else float("nan"),
        "mean_lead_days": float(lead.mean()) if len(lead) else float("nan"),
        "max_lead_days": float(lead.max()) if len(lead) else float("nan"),
        "censored": int(tbl["censored"].sum()) if not tbl.empty else 0,
        "alarm_rate": float(alarm.mean()),
    }
    return {"name": name, "table": tbl, "summary": summary, "tau": tau}


def fig_baseline_lead(res: dict):
    """Lead-time bars per RS firm for one detector."""
    tbl, s = res["table"], res["summary"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    if tbl.empty:
        ax.text(0.5, 0.5, "no RS firm in this panel", ha="center", va="center")
    else:
        d = tbl.copy()
        d["lead_m"] = d["lead_time_days"] / 30.44
        labels = d["firm_id"].astype(str)
        colors = ["#16a34a" if v == 1 else "#cbd5e1" for v in d["detected"]]
        ax.barh(labels, d["lead_m"].fillna(0.0), color=colors)
        for i, (v, det, cen) in enumerate(zip(d["lead_m"], d["detected"],
                                              d.get("censored", d["detected"] * 0))):
            txt = ("not detected" if det == 0 else
                   (">= %.1f mo (censored)" % v if cen == 1 else "%.1f mo" % v))
            ax.text(0.05, i, txt, va="center", fontsize=8,
                    color="#dc2626" if det == 0 else "#065f46")
        ax.set_xlabel("lead time before the RS event (months)")
    ax.set_title("%s - detected %d/%d RS firms  (tau = train-normal q%.2f)"
                 % (s["model"], s["detected"], s["event_firms"], s["q"]), fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def save_baseline_lead(res: dict, db_path: str | None = None) -> pd.DataFrame:
    table = "%s_%s" % (LEAD_TABLE, res["name"].lower())
    tbl = res["table"].copy()
    for c in ("event_date", "first_alarm"):
        if c in tbl.columns:
            tbl[c] = tbl[c].astype(str)
    con = sqlite3.connect(db_path or DB_DEFAULT)
    try:
        tbl.to_sql(table, con, if_exists="replace", index=False)
        pd.DataFrame([{"metric": k, "value": v} for k, v in res["summary"].items()]
                     ).astype({"value": str}).to_sql(
            table + "_summary", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return tbl


def load_baseline_lead(name: str, db_path: str | None = None) -> pd.DataFrame:
    con = sqlite3.connect(db_path or DB_DEFAULT)
    try:
        return pd.read_sql_query("SELECT * FROM %s_%s" % (LEAD_TABLE, name.lower()), con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def fig_baselines(cmp_df: pd.DataFrame):
    d = cmp_df.dropna(subset=["AUC"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.0))
    ax1.barh(d["model"][::-1], d["AUC"][::-1], color="#0ea5e9")
    ax1.axvline(0.5, color="#dc2626", ls="--", lw=1.2, label="random")
    ax1.set_xlim(0, 1); ax1.set_xlabel("ROC-AUC"); ax1.legend(fontsize=8)
    ax1.set_title("Eight baseline detectors on the factor panel (target = RS)", fontsize=10)
    ax1.grid(alpha=0.3, axis="x")

    ax2.barh(d["model"][::-1], d["lift"][::-1], color="#7c3aed")
    ax2.axvline(1.0, color="#dc2626", ls="--", lw=1.2, label="no lift")
    ax2.set_xlabel("PR-AUC / base rate  (lift)"); ax2.legend(fontsize=8)
    ax2.set_title("Precision-recall lift over the base rate", fontsize=10)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def save_baselines(cmp_df: pd.DataFrame, db_path: str | None = None,
                   table: str = BASELINE_TABLE) -> pd.DataFrame:
    con = sqlite3.connect(db_path or DB_DEFAULT)
    try:
        cmp_df.to_sql(table, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return cmp_df


def load_baselines(db_path: str | None = None, table: str = BASELINE_TABLE) -> pd.DataFrame:
    con = sqlite3.connect(db_path or DB_DEFAULT)
    try:
        return pd.read_sql_query("SELECT * FROM %s" % table, con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def _selftest() -> None:
    X, y, firms, idx, feats = build_windows(window=8)
    print("windows: %d x %d x %d factors | positives %d"
          % (X.shape[0], X.shape[1], len(feats), int(y.sum())))
    cmp_df = run_all_baselines(window=8, epochs=3)
    print(cmp_df.to_string(index=False))
    save_baselines(cmp_df)
    fig = fig_baselines(cmp_df); plt.close(fig)
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__.strip().splitlines()[0])
        print("\nusage: python baselines.py --selftest")
