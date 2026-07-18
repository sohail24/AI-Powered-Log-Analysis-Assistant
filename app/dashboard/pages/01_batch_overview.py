"""Batch Overview — daily snapshot of all batch executions.

Shows top-line metrics, a styled status grid, a status pie chart,
and a Gantt-style timeline for the selected date + environment.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_root = str(_pathlib.Path(__file__).resolve().parents[3])
if _root not in _sys.path: _sys.path.insert(0, _root)

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Batch Overview | Batch Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

from app.dashboard.components.charts import batch_gantt_chart, status_pie_chart
from app.dashboard.components.filters import date_range_filter, environment_filter
from app.dashboard.components.tables import metrics_row, styled_batch_table
from app.dashboard.shared import get_repo

# ── Repo ─────────────────────────────────────────────────────────
repo = get_repo()

# ── Sidebar filters ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    env = environment_filter(key="overview_env")
    start_date, end_date = date_range_filter(key_prefix="overview", default_days=1)

st.title("📊 Batch Overview")
st.caption(f"Showing **{env}** | {start_date} → {end_date}")
st.divider()

# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading batch data…"):
    rows = repo.get_executions_by_date_range(start_date, end_date, env)

if not rows:
    st.info("🔍 No batch executions found for the selected date range and environment.")
    st.stop()

df = pd.DataFrame([dict(r) for r in rows])

# ── Metric row ───────────────────────────────────────────────────
total = len(df)
success_count = (df["status"] == "SUCCESS").sum()
failure_count = (df["status"] == "FAILED").sum()
total_errors = int(df["error_count"].fillna(0).sum()) if "error_count" in df.columns else 0
success_rate = (success_count / total * 100) if total > 0 else 0.0

metrics_row(total, success_rate, failure_count, total_errors)
st.divider()

# ── Status grid + pie chart ──────────────────────────────────────
col_table, col_pie = st.columns([3, 1], gap="large")

with col_table:
    st.subheader("Batch Status Grid")
    display_cols = [c for c in [
        "job_name", "run_number", "attempt_type", "status",
        "start_time", "end_time", "duration_seconds", "error_count",
    ] if c in df.columns]
    rename_map = {
        "job_name": "Job Name",
        "run_number": "Run #",
        "attempt_type": "Attempt",
        "status": "Status",
        "start_time": "Start",
        "end_time": "End",
        "duration_seconds": "Duration (s)",
        "error_count": "Errors",
    }
    table_df = df[display_cols].rename(columns=rename_map).copy()
    styled_batch_table(table_df.rename(columns={"Status": "status"}))

with col_pie:
    st.subheader("Status Breakdown")
    status_counts = (
        df.groupby("status")
        .size()
        .reset_index(name="count")
    )
    st.plotly_chart(
        status_pie_chart(status_counts),
        use_container_width=True,
    )

st.divider()

# ── Timeline ─────────────────────────────────────────────────────
st.subheader("Execution Timeline")

gantt_df = df[
    [c for c in ["job_name", "start_time", "end_time", "status", "correlation_id"]
     if c in df.columns]
].copy()

# Parse timestamps.
for col in ("start_time", "end_time"):
    if col in gantt_df.columns:
        gantt_df[col] = pd.to_datetime(gantt_df[col], errors="coerce")

gantt_df = gantt_df.dropna(subset=["start_time"])

if gantt_df.empty:
    st.info("No timestamped executions to display in timeline.")
else:
    st.plotly_chart(
        batch_gantt_chart(gantt_df),
        use_container_width=True,
    )
