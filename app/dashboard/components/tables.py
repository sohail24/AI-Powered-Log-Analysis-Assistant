"""Reusable styled dataframe helpers for the dashboard.

All functions call ``st.dataframe`` / ``st.table`` directly so they
encapsulate both data shaping and rendering in one place.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

# ── Colour maps ──────────────────────────────────────────────────

_STATUS_BG = {
    "SUCCESS": "background-color: #14532d; color: #86efac;",
    "FAILED": "background-color: #7f1d1d; color: #fca5a5;",
    "PARTIAL": "background-color: #7c2d12; color: #fdba74;",
    "UNKNOWN": "background-color: #1e293b; color: #94a3b8;",
    "RUNNING": "background-color: #1e3a5f; color: #93c5fd;",
}

_SEVERITY_BG = {
    "CRITICAL": "background-color: #450a0a; color: #fca5a5;",
    "HIGH": "background-color: #7f1d1d; color: #fca5a5;",
    "MEDIUM": "background-color: #431407; color: #fdba74;",
    "LOW": "background-color: #172554; color: #93c5fd;",
}


def styled_batch_table(data: pd.DataFrame) -> None:
    """Render a styled batch execution table.

    Colours the ``status`` column based on status value.  All other
    columns are rendered in the default style.

    Args:
        data: DataFrame with at minimum a ``status`` column.
    """
    if data.empty:
        st.info("No batch executions found for the selected filters.")
        return

    def _colour_status(val: str) -> str:
        return _STATUS_BG.get(str(val).upper(), "")

    styled = data.style.map(_colour_status, subset=["status"])  # type: ignore[arg-type]

    # Format numeric columns nicely.
    fmt: dict = {}
    if "duration_seconds" in data.columns:
        fmt["duration_seconds"] = "{:.1f}s"
    if "error_rate_percent" in data.columns:
        fmt["error_rate_percent"] = "{:.1f}%"

    if fmt:
        styled = styled.format(fmt, na_rep="—")

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(35 * len(data) + 38, 500),
    )


def error_summary_table(data: pd.DataFrame) -> None:
    """Render the error summary table with severity colouring.

    Args:
        data: DataFrame with at minimum ``severity`` and
              ``error_category`` columns.
    """
    if data.empty:
        st.info("No errors recorded for the selected filters.")
        return

    def _colour_severity(val: str) -> str:
        return _SEVERITY_BG.get(str(val).upper(), "")

    styled = data.style.map(_colour_severity, subset=["severity"])  # type: ignore[arg-type]
    st.dataframe(styled, use_container_width=True)


def metrics_row(
    total: int,
    success_rate: float,
    failures: int,
    total_errors: int,
) -> None:
    """Render the 4-metric top bar.

    Args:
        total: Total batches run.
        success_rate: Success percentage (0-100).
        failures: Total failed executions.
        total_errors: Total error log lines.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Batches", total)
    c2.metric(
        "Success Rate",
        f"{success_rate:.1f}%",
        delta=None,
    )
    c3.metric(
        "Total Failures",
        failures,
        delta=None,
    )
    c4.metric("Total Errors", total_errors)


def execution_metadata_card(row: dict) -> None:
    """Render a compact metadata strip for a single execution.

    Args:
        row: Dict-like with execution fields.
    """
    cols = st.columns(5)
    cols[0].metric("Job", row.get("job_name", "—"))
    cols[1].metric("Status", row.get("status", "—"))
    dur = row.get("duration_seconds")
    cols[2].metric("Duration", f"{dur:.1f}s" if dur else "—")
    cols[3].metric("Errors", row.get("error_count", 0))
    cols[4].metric("Run #", row.get("run_number", "—"))
