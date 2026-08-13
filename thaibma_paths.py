# -*- coding: utf-8 -*-
"""
thaibma_paths.py -- one place that decides where the data lives.

THE PROBLEM THIS SOLVES
    Every script used to resolve the database from its own location::

        HERE = os.path.dirname(os.path.abspath(__file__))
        DB   = os.path.join(HERE, "cmdf_credit.db")

    That is fine while all 111 files sit in a single directory beside the data. This
    repository sorts them into app/, ews/, models/ and so on, so each HERE points
    somewhere different and none of those folders holds the database.

    It also matters for a second reason: this repository contains code only. The iBond
    panel is ThaiBMA data and is deliberately not committed, so the data root is
    somewhere outside the checkout on every machine.

NOTHING NEEDS CONFIGURING
    The data root is found automatically. Nobody has to set a variable before the
    program will start, and importing this module never fails, so installing and
    setting up the project works even when no database is present yet.

    The search runs in this order and stops at the first hit:

    1. ``THAIBMA_DATA`` -- optional, only for an unusual layout
    2. the repository folder itself
    3. each parent folder above it, up to eight levels, looking for cmdf_credit.db
    4. a ``data`` folder beside or above the repository

    If none of them holds the database, DATA_ROOT still resolves to a sensible
    writable folder and DB simply points at a file that does not exist yet. Code that
    only imports the module keeps working; only code that actually opens the database
    fails, and then with a message that says where it looked.

WHAT LIVES IN THE DATA ROOT
    cmdf_credit.db      the SQLite database, including ibond_33features_panel
    tex_out/            every figure, table and CSV the scripts write

CREDENTIALS ARE NOT HERE
    iBond and SMTP credentials are read from environment variables
    (``THAIBMA_USER`` / ``THAIBMA_PASS``, ``SMTP_USER`` / ``SMTP_PASS``) and are never
    stored in this repository. ThaiBMA's terms make the user ID and password
    confidential, so they must be set by the account holder on their own machine.
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


DB_NAME = "cmdf_credit.db"
_SEARCHED = []


def _candidates():
    env = os.environ.get("THAIBMA_DATA")
    if env:
        yield os.path.abspath(env)
    d = REPO_ROOT
    for _ in range(8):                       # the repo, then upwards
        yield d
        yield os.path.join(d, "data")
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent


def _resolve():
    """Find the folder holding the database. Never raises; always returns a path."""
    for c in _candidates():
        if c in _SEARCHED:
            continue
        _SEARCHED.append(c)
        if os.path.isdir(c) and os.path.exists(os.path.join(c, DB_NAME)):
            return c
    # nothing found: fall back to the repo so imports and setup still succeed
    return REPO_ROOT


DATA_ROOT = _resolve()
DB = os.path.join(DATA_ROOT, DB_NAME)
OUTDIR = os.path.join(DATA_ROOT, "tex_out")


def require_db():
    """Call before opening the database when a clear failure message helps."""
    if os.path.exists(DB):
        return DB
    looked = "\n  ".join(_SEARCHED)
    raise FileNotFoundError(
        f"{DB_NAME} was not found. This repository holds code only; the iBond panel "
        f"is ThaiBMA data and is not committed.\nPut {DB_NAME} in the repository "
        f"folder, or set THAIBMA_DATA to the folder that holds it.\nLooked in:\n  "
        f"{looked}")


def out(name):
    """Path inside the output folder, creating it on first use."""
    os.makedirs(OUTDIR, exist_ok=True)
    return os.path.join(OUTDIR, name)


def describe():
    return (f"DATA_ROOT = {DATA_ROOT}\n"
            f"DB        = {DB}  "
            f"({'found' if os.path.exists(DB) else 'MISSING'})\n"
            f"OUTDIR    = {OUTDIR}  "
            f"({'exists' if os.path.isdir(OUTDIR) else 'will be created'})")


if __name__ == "__main__":
    print(describe())
