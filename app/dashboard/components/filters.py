"""Reusable filter widgets for the dashboard.

All functions return pure Python values (dates, strings, etc.) so
that pages can use them without coupling to the widget implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import streamlit as st

from app.storage.repository import BatchRepository


def date_range_filter(
    key_prefix: str = "global",
    default_days: int = 7,
) -> tuple[date, date]:
    """Render a two-column date range picker.

    Args:
        key_prefix: Prefix for session_state keys (avoid collision).
        default_days: How many days back to default start_date.

    Returns:
        ``(start_date, end_date)`` tuple.
    """
    col1, col2 = st.columns(2)
    today = date.today()
    default_start = today - timedelta(days=default_days - 1)

    with col1:
        start = st.date_input(
            "From",
            value=default_start,
            max_value=today,
            key=f"{key_prefix}_start_date",
        )
    with col2:
        end = st.date_input(
            "To",
            value=today,
            max_value=today,
            key=f"{key_prefix}_end_date",
        )

    # Ensure start ≤ end.
    if start > end:
        st.warning("⚠️ Start date is after end date — swapping.")
        start, end = end, start

    return start, end


def environment_filter(key: str = "global_env") -> str:
    """Render an environment selector.

    Args:
        key: Session state key.

    Returns:
        Selected environment string.
    """
    return st.selectbox(
        "Environment",
        options=["prod", "staging", "uat"],
        index=0,
        key=key,
    )


def job_selector(
    repo: BatchRepository,
    key: str = "global_job",
    include_all: bool = True,
) -> Optional[str]:
    """Render a job-name selector populated from the database.

    Args:
        repo: Repository to query for distinct job names.
        key: Session state key.
        include_all: If True, adds an "All Jobs" option at the top.

    Returns:
        Selected job name, or ``None`` if "All Jobs" selected.
    """
    rows = repo.get_all_job_names()
    names = [r["job_name"] for r in rows] if rows else []

    if not names:
        st.info("No jobs found in database.")
        return None

    options = ["All Jobs"] + names if include_all else names
    selected = st.selectbox("Job Name", options=options, key=key)
    return None if selected == "All Jobs" else selected


def level_multiselect(key: str = "global_levels") -> list[str]:
    """Render a log-level multiselect.

    Returns:
        List of selected level strings.
    """
    return st.multiselect(
        "Log Level",
        options=["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"],
        default=["INFO", "WARN", "ERROR", "FATAL", "CRITICAL"],
        key=key,
    )
