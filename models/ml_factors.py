"""
ml_factors.py - alternative factor engines for the CMDF bond project.

Two additions to Approach 2:

  1. **LightGBM / CatBoost factor importance** - the same design matrix used by
     the Koopman model (raw factors plus their one-step Koopman forecast) is fed
     to LightGBM and to CatBoost. Each engine yields its own ranking of the
     factors, and the top factors become the watch-list that drives alerting.
     Validation always splits by *firm* (never by row), because the RS target is
     rare and clustered inside a handful of firms.

  2. **VAE latent factors** - the firm factor sequences are compressed by a
     temporal VAE (LSTM encoder -> mu / logvar -> reparameterise -> LSTM
     decoder, the architecture used in the research script
     ``D:\\vae\\vae_bond_backup3.py``). The posterior mean ``mu`` of each firm is
     stored as its latent factor vector.

Standalone check:
    python ml_factors.py --selftest
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
IMPORTANCE_TABLE = "factor_importance"      # + "_lightgbm" / "_catboost"
VAE_TABLE = "vae_latent"


# ------------------------------------------------------------- design --------
def build_design(df: pd.DataFrame | None = None, ridge: float = 1e-3,
                 target: str = "target_RS"):
    """Standardised factors + their one-step Koopman forecast, with firm index."""
    df = kg.load_bond_panel() if df is None else df
    feats = kg.feature_columns(df)
    raw = df[feats].astype(float)
    mu, sd = raw.mean(), raw.std().replace(0, 1.0)
    dfs = df.copy()
    dfs[feats] = ((raw - mu) / sd).fillna(0.0)

    X_now, X_next, rows = kg.build_koopman_pairs(dfs, feats)
    K = kg.fit_linear_koopman_K(X_now, X_next, ridge=ridge)
    Z = X_now @ K.T
    X = np.hstack([X_now, Z, np.linalg.norm(Z, axis=1, keepdims=True)])
    names = list(feats) + ["koop1_%s" % c for c in feats] + ["koop1_norm"]
    y = pd.to_numeric(df.loc[rows, target], errors="coerce").fillna(0).to_numpy().astype(int)
    firms = df.loc[rows, "firm_id"].astype(str).to_numpy()
    return X, y, names, firms, feats


# --------------------------------------------------- LightGBM / CatBoost -----
def _make_model(engine: str, seed: int = 42):
    engine = engine.lower()
    # The RS target is ~0.06% positive, so every engine must be told to balance
    # the classes; without it the learner collapses onto the majority class and
    # the ranking becomes meaningless (an AUC well below 0.5).
    if engine == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              subsample=0.9, colsample_bytree=0.9,
                              class_weight="balanced",
                              random_state=seed, verbose=-1)
    if engine == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                                  auto_class_weights="Balanced",
                                  random_seed=seed, verbose=0, allow_writing_files=False)
    if engine == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="logloss", random_state=seed)
    raise ValueError("unknown engine %r" % engine)


def run_importance(engine: str = "lightgbm", df: pd.DataFrame | None = None,
                   test_size: float = 0.3, seed: int = 42, top_watch: int = 10) -> dict:
    """Train the engine with a leave-firms-out split and rank the factors."""
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

    X, y, names, firms, feats = build_design(df)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=seed).split(X, y, groups=firms))
    model = _make_model(engine, seed)
    model.fit(X[tr], y[tr])

    p = model.predict_proba(X[te])[:, 1]
    metrics: dict[str, float] = {"base_rate": float(y[te].mean()),
                                 "n_test_pos": float(int(y[te].sum()))}
    if len(np.unique(y[te])) > 1:
        metrics["AUC"] = float(roc_auc_score(y[te], p))
        metrics["PR_AUC"] = float(average_precision_score(y[te], p))
        metrics["MCC"] = float(matthews_corrcoef(y[te], (p >= 0.5).astype(int)))
    else:
        metrics.update({"AUC": float("nan"), "PR_AUC": float("nan"), "MCC": float("nan")})

    imp = np.asarray(getattr(model, "feature_importances_", np.zeros(len(names))), dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    table = (pd.DataFrame({"feature": names, "importance": imp})
             .sort_values("importance", ascending=False).reset_index(drop=True))
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["kind"] = np.where(table["feature"].astype(str).str.startswith("koop1_"),
                             "koopman", "raw")
    # the watch-list: the underlying factor names of the strongest signals
    watch = []
    for f in table["feature"].astype(str):
        base = f[len("koop1_"):] if f.startswith("koop1_") else f
        if base in feats and base not in watch:
            watch.append(base)
        if len(watch) >= top_watch:
            break

    return {"engine": engine, "metrics": metrics, "importance": table,
            "watchlist": watch, "n_firms": int(len(np.unique(firms))),
            "n_rows": int(X.shape[0])}


def fig_model_importance(res: dict, top: int = 20):
    d = res["importance"].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(d) + 1.3))
    colors = ["#7c3aed" if k == "koopman" else "#0ea5e9" for k in d["kind"]]
    ax.barh(d["feature"].astype(str), d["importance"], color=colors)
    ax.set_xlabel("normalised importance")
    ax.set_title("%s: top %d factors  (purple = Koopman forecast feature)"
                 % (res["engine"].upper(), top), fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def save_importance(res: dict, db_path: str | None = None) -> pd.DataFrame:
    db_path = db_path or DB_DEFAULT
    table = "%s_%s" % (IMPORTANCE_TABLE, res["engine"])
    con = sqlite3.connect(db_path)
    try:
        res["importance"].to_sql(table, con, if_exists="replace", index=False)
        pd.DataFrame([{"metric": k, "value": float(v)} for k, v in res["metrics"].items()]
                     + [{"metric": "n_firms", "value": float(res["n_firms"])}]
                     ).to_sql(table + "_metrics", con, if_exists="replace", index=False)
        pd.DataFrame({"rank": np.arange(1, len(res["watchlist"]) + 1),
                      "factor": res["watchlist"]}).to_sql(
            table + "_watchlist", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return res["importance"]


def load_table(db_path: str | None = None, table: str = "factor_importance_lightgbm") -> pd.DataFrame:
    con = sqlite3.connect(db_path or DB_DEFAULT)
    try:
        return pd.read_sql_query("SELECT * FROM %s" % table, con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


# ------------------------------------------------------- VAE latent ----------
def build_sequences(df: pd.DataFrame | None = None, seq_len: int = 12,
                    target: str = "target_RS"):
    """(n_firms, seq_len, n_factors) tensor of the most recent observations."""
    df = kg.load_bond_panel() if df is None else df
    feats = kg.feature_columns(df)
    raw = df[feats].astype(float)
    mu, sd = raw.mean(), raw.std().replace(0, 1.0)
    dfs = df.copy()
    dfs[feats] = ((raw - mu) / sd).fillna(0.0)

    seqs, ids, labels = [], [], []
    for firm, g in dfs.groupby("firm_id", sort=False):
        g = g.sort_values("dt")
        if len(g) < seq_len:
            continue
        seqs.append(g[feats].to_numpy()[-seq_len:])
        ids.append(str(firm))
        labels.append(int(pd.to_numeric(g[target], errors="coerce").fillna(0).max() > 0)
                      if target in g.columns else 0)
    if not seqs:
        raise ValueError("no firm has at least %d observations" % seq_len)
    return np.asarray(seqs, dtype=np.float32), ids, np.asarray(labels), feats


def vae_latent(df: pd.DataFrame | None = None, latent_dim: int = 4, seq_len: int = 12,
               hidden_dim: int = 64, epochs: int = 40, lr: float = 1e-3,
               seed: int = 42) -> dict:
    """Temporal VAE over the firm factor sequences; returns the latent factors.

    Same architecture as the research script: LSTM encoder -> (mu, logvar) ->
    reparameterise -> LSTM decoder, trained with reconstruction + KL loss.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X, ids, labels, feats = build_sequences(df, seq_len=seq_len)
    n, T, F = X.shape

    class TemporalVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.LSTM(F, hidden_dim, batch_first=True)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            self.dec = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
            self.fc_out = nn.Linear(hidden_dim, F)

        def encode(self, x):
            _, (h, _) = self.enc(x)
            h = h[-1]
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        def decode(self, z):
            out, _ = self.dec(z.unsqueeze(1).expand(-1, T, -1))
            return self.fc_out(out)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    model = TemporalVAE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(X)
    batch = min(128, n)
    history = []
    model.train()
    for ep in range(int(epochs)):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            xb = xt[perm[i:i + batch]]
            recon, mu, logvar = model(xb)
            rec_loss = ((recon - xb) ** 2).mean()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
            loss = rec_loss + 0.01 * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(xb)
        history.append(tot / n)

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(xt)
        recon, _, _ = model(xt)
        per_firm_err = ((recon - xt) ** 2).mean(dim=(1, 2)).numpy()
    Zm = mu.numpy()

    table = pd.DataFrame(Zm, columns=["z%d" % (i + 1) for i in range(latent_dim)])
    table.insert(0, "firm_id", ids)
    table["event"] = labels
    table["recon_error"] = per_firm_err.round(6)
    table = table.round(6)
    return {"table": table, "latent": Zm, "labels": labels, "ids": ids,
            "history": history, "n_firms": n, "seq_len": T, "n_factors": F,
            "latent_dim": latent_dim,
            "final_loss": float(history[-1]) if history else float("nan")}


# ------------------------------- tabular latent family (AE/VAE/AAE/PAE) ------
LATENT_METHODS = ("AE", "VAE", "AAE", "PAE")
LATENT_FEATURE_TABLE = "latent_features"


def latent_features(method: str = "VAE", df: pd.DataFrame | None = None,
                    latent_dim: int = 8, hidden_dim: int = 64, epochs: int = 25,
                    beta: float = 0.1, adv_w: float = 1.0, test_size: float = 0.3,
                    seed: int = 42, lr: float = 1e-3) -> dict:
    """Latent features from one of the tabular representation models.

    Architectures follow ``D:\\vae\\run_dtd.py`` (``TabAE`` / ``TabVAE`` /
    ``TabPAE`` and the adversarial variant used for ``AAE``):

      AE   encoder -> z -> decoder,                 loss = MSE
      VAE  encoder -> (mu, logvar) -> z -> decoder, loss = MSE + beta * KL
      AAE  AE plus a discriminator that pushes q(z) towards N(0, I)
      PAE  encoder -> z -> decoder returning (mu, logvar), Gaussian NLL

    The encoder is trained **unsupervised on training firms only**; the latent
    vector is then scored by a balanced logistic regression, exactly as
    ``add_latent_method`` does in the research script.
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    method = method.upper()
    if method not in LATENT_METHODS:
        raise ValueError("method must be one of %s" % (LATENT_METHODS,))

    torch.manual_seed(seed)
    X, y, names, firms, feats = build_design(df)
    X = X[:, :len(feats)]                       # raw standardised factors only
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=seed).split(X, y, groups=firms))
    d_in = X.shape[1]
    xt = torch.from_numpy(X.astype(np.float32))
    xtr = xt[tr]

    class Enc(nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, out_dim))
        def forward(self, x): return self.net(x)

    dec = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                        nn.Linear(hidden_dim, d_in))

    if method == "VAE":
        enc_h = nn.Sequential(nn.Linear(d_in, hidden_dim), nn.ReLU())
        fc_mu, fc_lv = nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)
        params = list(enc_h.parameters()) + list(fc_mu.parameters()) + \
                 list(fc_lv.parameters()) + list(dec.parameters())
    elif method == "PAE":
        enc = Enc(latent_dim)
        dec_h = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        p_mu, p_lv = nn.Linear(hidden_dim, d_in), nn.Linear(hidden_dim, d_in)
        params = list(enc.parameters()) + list(dec_h.parameters()) + \
                 list(p_mu.parameters()) + list(p_lv.parameters())
    else:                                        # AE and AAE share the encoder
        enc = Enc(latent_dim)
        params = list(enc.parameters()) + list(dec.parameters())

    opt = torch.optim.Adam(params, lr=lr)
    disc = opt_d = None
    if method == "AAE":
        disc = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                             nn.Linear(hidden_dim, 1))
        opt_d = torch.optim.Adam(disc.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()

    n, batch, history = len(xtr), min(512, len(xtr)), []
    for _ in range(int(epochs)):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            xb = xtr[perm[i:i + batch]]
            if method == "VAE":
                h = enc_h(xb); mu, lv = fc_mu(h), fc_lv(h).clamp(-8, 8)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                loss = ((dec(z) - xb) ** 2).mean() + beta * (
                    -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean())
            elif method == "PAE":
                z = enc(xb); h = dec_h(z)
                mu, lv = p_mu(h), p_lv(h).clamp(-8, 8)
                loss = (0.5 * (lv + (xb - mu) ** 2 / lv.exp())).mean()
            else:
                z = enc(xb)
                loss = ((dec(z) - xb) ** 2).mean()
                if method == "AAE":
                    real = torch.randn_like(z)
                    d_loss = bce(disc(real), torch.ones(len(z), 1)) + \
                             bce(disc(z.detach()), torch.zeros(len(z), 1))
                    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                    loss = loss + adv_w * bce(disc(z), torch.ones(len(z), 1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(xb)
        history.append(tot / max(1, n))

    with torch.no_grad():
        if method == "VAE":
            Z = fc_mu(enc_h(xt)).numpy()
        else:
            Z = enc(xt).numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=5000)
    metrics: dict[str, float] = {"base_rate": float(y[te].mean()),
                                 "n_test_pos": float(int(y[te].sum())),
                                 "final_loss": float(history[-1]) if history else float("nan")}
    if len(np.unique(y[tr])) > 1 and len(np.unique(y[te])) > 1:
        clf.fit(Z[tr], y[tr])
        p = clf.predict_proba(Z[te])[:, 1]
        metrics["AUC"] = float(roc_auc_score(y[te], p))
        metrics["PR_AUC"] = float(average_precision_score(y[te], p))
    else:
        metrics["AUC"] = metrics["PR_AUC"] = float("nan")

    table = pd.DataFrame(Z, columns=["z%d" % (i + 1) for i in range(latent_dim)]).round(6)
    table.insert(0, "firm_id", firms)
    table["event"] = y
    return {"method": method, "table": table, "latent": Z, "labels": y,
            "firms": firms, "history": history, "metrics": metrics,
            "latent_dim": latent_dim, "n_rows": int(X.shape[0]),
            "n_firms": int(len(np.unique(firms)))}


def compare_latent_methods(df: pd.DataFrame | None = None, latent_dim: int = 8,
                           epochs: int = 20, **kw) -> pd.DataFrame:
    """Run every tabular latent method and rank them by out-of-firm AUC."""
    out = []
    for m in LATENT_METHODS:
        try:
            r = latent_features(m, df=df, latent_dim=latent_dim, epochs=epochs, **kw)
            out.append({"method": m, "latent_dim": latent_dim,
                        "AUC": r["metrics"]["AUC"], "PR_AUC": r["metrics"]["PR_AUC"],
                        "final_loss": r["metrics"]["final_loss"]})
        except Exception as exc:
            out.append({"method": m, "latent_dim": latent_dim, "AUC": float("nan"),
                        "PR_AUC": float("nan"), "final_loss": float("nan"),
                        "error": str(exc)[:80]})
    return pd.DataFrame(out).sort_values("AUC", ascending=False).reset_index(drop=True)


def save_latent_features(res: dict, db_path: str | None = None) -> pd.DataFrame:
    table = "%s_%s" % (LATENT_FEATURE_TABLE, res["method"].lower())
    con = sqlite3.connect(db_path or DB_DEFAULT)
    try:
        res["table"].head(5000).to_sql(table, con, if_exists="replace", index=False)
        pd.DataFrame([{"metric": k, "value": float(v)}
                      for k, v in res["metrics"].items()]).to_sql(
            table + "_metrics", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return res["table"]


def fig_latent_compare(cmp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    d = cmp_df.dropna(subset=["AUC"])
    ax.bar(d["method"], d["AUC"], color="#0ea5e9")
    ax.axhline(0.5, color="#dc2626", ls="--", lw=1.2, label="random (0.5)")
    ax.set_ylabel("out-of-firm AUC"); ax.set_ylim(0, 1)
    ax.set_title("Latent representation methods on the factor panel", fontsize=10)
    for i, (m, v) in enumerate(zip(d["method"], d["AUC"])):
        ax.text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_vae_latent(res: dict):
    """Latent space (first two dimensions or PCA) plus the training curve."""
    Z, lab = res["latent"], np.asarray(res["labels"])
    if Z.shape[1] > 2:
        Zc = Z - Z.mean(0)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        P = Zc @ Vt[:2].T
        xlab, ylab = "PC1", "PC2"
    else:
        P, xlab, ylab = Z[:, :2], "z1", "z2"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.2))
    ax1.scatter(P[lab == 0, 0], P[lab == 0, 1], s=14, c="#94a3b8",
                label="no event", alpha=0.7)
    if (lab == 1).any():
        ax1.scatter(P[lab == 1, 0], P[lab == 1, 1], s=60, c="#dc2626",
                    edgecolor="black", label="RS event", zorder=3)
    ax1.set_xlabel(xlab); ax1.set_ylabel(ylab)
    ax1.set_title("VAE latent factors - %d firms, latent dim %d"
                  % (res["n_firms"], res["latent_dim"]), fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(res["history"], color="#7c3aed", lw=1.5)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("recon + KL loss")
    ax2.set_title("VAE training loss (final %.4f)" % res["final_loss"], fontsize=10)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_vae(res: dict, db_path: str | None = None, table: str = VAE_TABLE) -> pd.DataFrame:
    con = sqlite3.connect(db_path or DB_DEFAULT)
    try:
        res["table"].to_sql(table, con, if_exists="replace", index=False)
        pd.DataFrame([{"metric": "n_firms", "value": float(res["n_firms"])},
                      {"metric": "latent_dim", "value": float(res["latent_dim"])},
                      {"metric": "seq_len", "value": float(res["seq_len"])},
                      {"metric": "n_factors", "value": float(res["n_factors"])},
                      {"metric": "final_loss", "value": float(res["final_loss"])}]
                     ).to_sql(table + "_metrics", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return res["table"]


# ------------------------------------------------------------------ CLI ------
def _selftest() -> None:
    for engine in ("lightgbm", "catboost"):
        res = run_importance(engine)
        save_importance(res)
        m = res["metrics"]
        print("%-9s AUC %.4f  PR-AUC %.4f  base %.5f | top: %s"
              % (engine, m["AUC"], m["PR_AUC"], m["base_rate"],
                 ", ".join(res["watchlist"][:4])))
        fig = fig_model_importance(res); plt.close(fig)

    v = vae_latent(epochs=8)
    save_vae(v)
    print("VAE       %d firms, latent %d, seq %d, final loss %.4f"
          % (v["n_firms"], v["latent_dim"], v["seq_len"], v["final_loss"]))
    fig = fig_vae_latent(v); plt.close(fig)
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__.strip().splitlines()[0])
        print("\nusage: python ml_factors.py --selftest")
