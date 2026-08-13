# -*- coding: utf-8 -*-
"""
run.py -- launcher that makes the sorted folders behave like the flat layout.

WHY THIS EXISTS
    The scripts import each other by bare module name (``import cmdf_tree_classify``,
    ``import firm_shock_panel``, and 478 more). That works when every file sits in one
    directory. Sorting them into app/, ews/, models/ and the rest is better to read but
    breaks every one of those imports.

    Rather than rewrite 480 import statements -- which would mean touching almost every
    file and re-testing all of it -- this launcher puts every project folder on
    ``sys.path`` before handing control to the requested script. The scripts stay
    exactly as they were, and the folder structure stays readable.

USAGE
    python run.py app                       # the GUI
    python run.py hyperbolic_boundary_panel # any module, by bare name
    python run.py firm_shock_panel --help   # arguments pass straight through
    python run.py --list                    # show every runnable module

    Importing also works once paths are set up:

        import run                      # side effect: sys.path is prepared
        import firm_shock_panel as fsp
"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def add_paths():
    """Put every directory that holds a .py file on sys.path, nearest first."""
    added = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {"__pycache__", ".git", ".github"}]
        if any(f.endswith(".py") for f in filenames):
            if dirpath not in sys.path:
                sys.path.insert(0, dirpath)
                added.append(dirpath)
    return added


add_paths()


def find_module(name):
    """Locate a module by bare name anywhere in the tree."""
    target = name if name.endswith(".py") else name + ".py"
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if target in filenames:
            return os.path.join(dirpath, target)
    return None


def list_modules():
    rows = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = os.path.relpath(dirpath, ROOT)
        for f in sorted(filenames):
            if f.endswith(".py") and f != "run.py":
                rows.append((rel if rel != "." else "", f[:-3]))
    return rows


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        rows = list_modules()
        cur = None
        for folder, mod in sorted(rows):
            if folder != cur:
                cur = folder
                print(f"\n{folder or '.'}/")
            print(f"    {mod}")
        print(f"\n{len(rows)} modules")
        return 0

    name = argv[0]
    path = find_module(name)
    if path is None:
        print(f"no module named {name!r}; try: python run.py --list",
              file=sys.stderr)
        return 2

    # the script sees only its own arguments, exactly as if it were run directly
    sys.argv = [path] + argv[1:]
    runpy.run_path(path, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
