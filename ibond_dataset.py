# -*- coding: utf-8 -*-
"""
ibond_dataset.py -- one place that decides WHICH panel the analysis runs on.

THE PROBLEM THIS SOLVES
    Every analysis script reached the data through one function,
    ``cmdf_tree_classify.load_panel()``, which hard-coded a single table::

        panel = pd.read_sql("SELECT * FROM ibond_33features_panel", con)

    That is the 293-issuer bond panel, and it was the only panel the repository
    could see. The research now has a second panel -- 941 SET/mai firms -- and the
    question being asked is what the shock analysis does on it compared with the
    original. One hard-coded table name cannot answer that.

    This module turns the table name into a choice. It is a port of the DataAdapter
    written for the iBond_LIME project, with the database search order taken from
    the leadning_time project, so all three code bases now agree on what "dataset 1"
    and "dataset 2" mean.

THE TWO DATASETS

    1  ibond_33features_panel          16,986 issuer-months, 293 bond issuers
       The original panel. The event label is NOT stored in the table: it is
       derived from ibond_default_payment, flagging the three months before an
       issuer's first recorded missed payment. Roughly 32 positive months from
       8 issuers.

    2  ibond_33features_panel_941firm  187,007 firm-months, 941 SET/mai firms
       The wider panel. It carries its own label column, y_pre3m, already built on
       the same three-month-ahead convention, with d_DP_RS as a fallback.
       124 positive months from 31 firms.

    The two are NOT interchangeable. Different universes, different label
    construction, different event counts. Comparing them compares how the fitted
    response surface behaves, not which model is more accurate.

CHOOSING A DATASET
    Nothing changes unless asked. Dataset 1 is the default, so every existing
    command keeps producing exactly what it produced before::

        python run.py knn_cluster_shock                 # dataset 1, as always
        python run.py knn_cluster_shock --dataset 2     # the 941-firm panel
        IBOND_DATASET=2 python run.py roe_triple_figures

    ``--dataset N`` is consumed here and removed from sys.argv, so the scripts'
    own argument parsing never sees it.

WHERE THINGS ARE WRITTEN
    Two runs must not overwrite each other, or there is nothing left to compare.

        tex_out/ds1/    everything produced under dataset 1
        tex_out/ds2/    everything produced under dataset 2
        tex_out/compare/    figures and tables that put the two side by side

    Result tables in SQLite are suffixed the same way: ``cmdf_pairwise_shock`` for
    dataset 1, ``cmdf_pairwise_shock_ds2`` for dataset 2. Results are always
    written to cmdf_credit.db, never to the 127 MB lime_credit.db that dataset 2
    is read from.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

RESULT_DB_NAME = "cmdf_credit.db"

# databases that may hold a panel, in the order leadning_time searches them
DB_NAMES = ("cmdf_credit.db", "lime_credit.db", "bond_financials.db")


DATASETS = {
    1: {
        "table": "ibond_33features_panel",
        "target": None,                     # derived from ibond_default_payment
        "target_fallback": None,
        "key": "ds1",
        "suffix": "",
        "short": "Dataset 1",
        "label": "Dataset 1 -- iBond bond issuers (293 issuers)",
        "label_th": "ชุดข้อมูลเดิม: ผู้ออกหุ้นกู้ 293 ราย (16,986 firm-months)",
    },
    2: {
        "table": "ibond_33features_panel_941firm",
        "target": "y_pre3m",
        "target_fallback": "d_DP_RS",
        "key": "ds2",
        "suffix": "_ds2",
        "short": "Dataset 2",
        "label": "Dataset 2 -- SET/mai listed firms (941 firms)",
        "label_th": "ชุดข้อมูลใหม่: บริษัทจดทะเบียน 941 ราย (187,007 firm-months)",
    },
}


# ============================================================== selection ====
def _read_choice():
    """Read --dataset N from the command line, or IBOND_DATASET, else 1.

    The flag is removed from sys.argv so the calling script's own parsing is
    unaffected: it never learns that the flag was there.
    """
    argv = sys.argv
    for flag in ("--dataset", "--ds"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                try:
                    n = int(argv[i + 1])
                except ValueError:
                    raise SystemExit(f"{flag} expects 1 or 2, got {argv[i + 1]!r}")
                del argv[i:i + 2]
                if n not in DATASETS:
                    raise SystemExit(f"{flag} expects 1 or 2, got {n}")
                return n
            raise SystemExit(f"{flag} expects a number")
    env = os.environ.get("IBOND_DATASET", "").strip()
    if env:
        try:
            n = int(env)
        except ValueError:
            raise SystemExit(f"IBOND_DATASET expects 1 or 2, got {env!r}")
        if n not in DATASETS:
            raise SystemExit(f"IBOND_DATASET expects 1 or 2, got {n}")
        return n
    return 1


CHOICE = _read_choice()
INFO = DATASETS[CHOICE]
TABLE = INFO["table"]


def use(choice):
    """Switch dataset from inside a program, for code that runs both in one process."""
    global CHOICE, INFO, TABLE, DATA_ROOT, READ_DB, RESULT_DB, OUTDIR
    if choice not in DATASETS:
        raise ValueError(f"dataset must be 1 or 2, got {choice!r}")
    CHOICE = choice
    INFO = DATASETS[choice]
    TABLE = INFO["table"]
    DATA_ROOT, READ_DB = _resolve(choice)
    RESULT_DB = os.path.join(DATA_ROOT, RESULT_DB_NAME)
    OUTDIR = os.path.join(DATA_ROOT, "tex_out", INFO["key"])
    return INFO


# ================================================================= search ====
_SEARCHED = []


def _folders():
    """Folders that might hold a database, nearest first.

    Same order as leadning_time uses, plus the sibling datasets_bond/ folder where
    the 941-firm database is kept, since it is far too large to live in the repo.
    """
    env = os.environ.get("THAIBMA_DATA")
    if env:
        yield os.path.abspath(env)
    d = REPO_ROOT
    for _ in range(8):
        yield d
        yield os.path.join(d, "data")
        yield os.path.join(d, "datasets_bond")
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent


def _unzip_if_needed(folder):
    """lime_credit.db ships zipped in the iBond_LIME repo; unpack it once."""
    zp = os.path.join(folder, "lime_credit.db.zip")
    db = os.path.join(folder, "lime_credit.db")
    if os.path.exists(db) or not os.path.exists(zp):
        return
    print(f"[ibond_dataset] extracting {zp} ...")
    try:
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(folder)
    except Exception as ex:                              # a corrupt zip is not fatal
        print(f"[ibond_dataset] could not extract: {ex}")


def _has_table(db_path, table):
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        cur = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            (table,))
        return cur.fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        con.close()


def _write_root():
    """The one folder that everything is written to, whichever dataset is active.

    Deliberately independent of where a panel is READ from. Dataset 2 is read from
    datasets_bond/lime_credit.db, but its figures and result tables belong beside
    dataset 1's, or the two runs cannot be compared without hunting across folders.
    """
    for folder in _folders():
        if os.path.isdir(folder) and os.path.exists(
                os.path.join(folder, RESULT_DB_NAME)):
            return folder
    return REPO_ROOT


def _resolve(choice):
    """Find (write root, database holding this dataset's table).

    Never raises. If no database holds the table, the path returned simply does
    not exist, so importing this module always succeeds and only code that
    actually opens the database fails -- with a message saying where it looked.
    """
    table = DATASETS[choice]["table"]
    root = _write_root()
    del _SEARCHED[:]
    for folder in _folders():
        if folder in _SEARCHED:
            continue
        _SEARCHED.append(folder)
        if not os.path.isdir(folder):
            continue
        _unzip_if_needed(folder)
        for name in DB_NAMES:
            p = os.path.join(folder, name)
            if (os.path.exists(p) and os.path.getsize(p) > 1024
                    and _has_table(p, table)):
                return root, p
    return root, os.path.join(root, RESULT_DB_NAME)


DATA_ROOT, READ_DB = _resolve(CHOICE)
RESULT_DB = os.path.join(DATA_ROOT, RESULT_DB_NAME)
OUTDIR = os.path.join(DATA_ROOT, "tex_out", INFO["key"])
COMPAREDIR = os.path.join(DATA_ROOT, "tex_out", "compare")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "dataset2_compare")


def require_db():
    """Call before opening the database when a clear failure message helps."""
    if os.path.exists(READ_DB) and _has_table(READ_DB, TABLE):
        return READ_DB
    looked = "\n  ".join(_SEARCHED)
    extra = ""
    if CHOICE == 2:
        extra = ("\nDataset 2 lives in lime_credit.db, which is not committed "
                 "anywhere. Unzip lime_credit.db.zip from the iBond_LIME repository "
                 "into a datasets_bond/ folder beside this one, or set "
                 "THAIBMA_DATA to the folder that holds it.")
    raise FileNotFoundError(
        f"table {TABLE!r} (dataset {CHOICE}) was not found in any database. This "
        f"repository holds code only; the panels are ThaiBMA material and are not "
        f"committed.{extra}\nLooked in:\n  {looked}")


# ================================================================ outputs ====
def out(name, dataset_dir=True):
    """Path inside this dataset's output folder, creating it on first use."""
    d = OUTDIR if dataset_dir else os.path.join(DATA_ROOT, "tex_out")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def compare_out(name):
    """Path inside the side-by-side comparison folder."""
    os.makedirs(COMPAREDIR, exist_ok=True)
    return os.path.join(COMPAREDIR, name)


def result_out(*parts):
    """Path inside results/dataset2_compare/, the folder that IS committed."""
    d = os.path.join(RESULTS_DIR, *parts[:-1]) if len(parts) > 1 else RESULTS_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, parts[-1])


def tname(base):
    """Result-table name for the active dataset: suffixed so two runs coexist."""
    return base + INFO["suffix"]


def describe():
    ok = "found" if os.path.exists(READ_DB) else "MISSING"
    return (f"dataset  = {CHOICE}  ({INFO['label']})\n"
            f"table    = {TABLE}\n"
            f"read DB  = {READ_DB}  ({ok})\n"
            f"result DB= {RESULT_DB}\n"
            f"OUTDIR   = {OUTDIR}")


if __name__ == "__main__":
    for n in (1, 2):
        use(n)
        print(describe())
        print()
