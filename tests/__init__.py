# -*- coding: utf-8 -*-
"""Put the project folders on sys.path before any test module is imported.

The test modules import project code by bare name (``import lead_metrics``), the
same convention the rest of the repository uses. That works under ``python run.py``
because the launcher prepares sys.path first, but ``python -m unittest discover -s
tests`` imports the test modules directly and every one of them died on
ModuleNotFoundError. Making tests/ a package gives discovery somewhere to run the
same preparation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run  # noqa: F401  -- importing run is what walks the tree and sets sys.path
