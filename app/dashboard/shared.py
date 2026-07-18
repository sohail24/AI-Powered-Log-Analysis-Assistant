"""Shared Streamlit resources for all dashboard pages.

Uses ``@st.cache_resource`` to maintain a single DatabaseManager
and BatchRepository across page navigations.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pathlib
_root = str(_pathlib.Path(__file__).resolve().parents[2])
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import streamlit as st

from app.config.settings import Settings
from app.storage.database import DatabaseManager
from app.storage.repository import BatchRepository


@st.cache_resource
def _get_db(db_path: str) -> DatabaseManager:
    """Return a cached DatabaseManager (one per db_path)."""
    db = DatabaseManager(db_path)
    db.initialize()
    return db


def get_repo() -> BatchRepository:
    """Return a ``BatchRepository`` backed by the cached DB.

    Settings are read once from environment / .env; the
    DatabaseManager connection is cached for the session lifetime.
    """
    settings = Settings()
    db = _get_db(settings.db_path)
    return BatchRepository(db)


def get_settings() -> Settings:
    """Return application settings (cached on session state)."""
    if "settings" not in st.session_state:
        st.session_state["settings"] = Settings()
    return st.session_state["settings"]
