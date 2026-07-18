"""sys.path bootstrap for Streamlit dashboard files.

Streamlit prepends the script's directory to sys.path, which causes
the file ``app/dashboard/app.py`` to shadow the ``app`` package when
Python tries to resolve ``from app.config import ...``.

Importing this module as the very first import in any dashboard file
ensures the project root is at the front of sys.path and the true
``app`` package is found correctly.

Usage (MUST be the first import in every dashboard file)::

    import app.dashboard._path_fix  # noqa: F401
"""

from __future__ import annotations

import pathlib
import sys

# Project root = grandparent of this file:
#   <root>/app/dashboard/_path_fix.py  →  <root>
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
