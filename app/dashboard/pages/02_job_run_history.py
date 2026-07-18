"""Job Run History — per-job execution history over a date range.

Shows a filterable history table, a stacked daily run-outcome chart,
and an attempt-type distribution chart.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_root = str(_pathlib.Path(__file__).resolve().parents[3])
if _root not in _sys.path: _sys.path.insert(0, _root)

from datetime import date, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Job Run History | Batch Intelligence Platform",
    page_icon="📈",
    layout="wide",
)

from app.dashboard.components.charts import attempt_type_chart, run_history_chart
from app.dashboard.components.filters import date_range_filter, environment_filter
from app.dashboard.components.tables import styled_batch_table
from app.dashboard.shared import get_repo

repo = get_repo()

# ── Sidebar filters ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    env = environment_filter(key="history_env")
    start_date, end_date = date_range_filter(key_prefix="history", default_days=7)

    # Job selector populated from DB.
    job_rows = repo.get_all_job_names()
    job_names = [r["job_name"] for r in job_rows] if job_rows else []
    if not job_names:
        st.info("No jobs in database.")
        st.stop()

    selected_job = st.selectbox("Job Name", options=job_names, key="history_job")

st.title("📈 Job Run History")
st.caption(f"**{selected_job}** | {env} | {start_date} → {end_date}")
st.divider()

# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading run history…"):
    rows = repo.get_executions_by_job_and_date_range(
        selected_job, start_date, end_date, env
    )

if not rows:
    st.info(
        f"🔍 No executions found for **{selected_job}** "
        f"in the selected date range."
    )
    st.stop()

df = pd.DataFrame([dict(r) for r in rows])

# ── Run History Table ────────────────────────────────────────────
st.subheader("Execution History")

display_cols = [c for c in [
    "run_date", "run_number", "attempt_type", "status",
    "duration_seconds", "error_count", "warn_count",
] if c in df.columns]
rename_map = {
    "run_date": "Date",
    "run_number": "Run #",
    "attempt_type": "Attempt Type",
    "status": "Status",
    "duration_seconds": "Duration (s)",
    "error_count": "Errors",
    "warn_count": "Warnings",
}
table_df = df[display_cols].rename(columns=rename_map).copy()
styled_batch_table(table_df.rename(columns={"Status": "status"}))

st.divider()

# ── Charts ───────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("Daily Run Outcomes")
    if "run_date" in df.columns and "status" in df.columns:
        run_counts = (
            df.groupby(["run_date", "status"])
            .size()
            .reset_index(name="count")
        )
        st.plotly_chart(
            run_history_chart(run_counts),
            use_container_width=True,
        )
    else:
        st.info("No date/status data available.")

with col_right:
    st.subheader("Attempt Type Distribution")
    if "run_date" in df.columns and "attempt_type" in df.columns:
        attempt_counts = (
            df.groupby(["run_date", "attempt_type"])
            .size()
            .reset_index(name="count")
        )
        st.plotly_chart(
            attempt_type_chart(attempt_counts),
            use_container_width=True,
        )
    else:
        st.info("No attempt type data available.")

# ── Summary stats ────────────────────────────────────────────────
st.divider()
st.subheader("Summary Statistics")

total = len(df)
success = (df["status"] == "SUCCESS").sum() if "status" in df.columns else 0
auto_retries = (
    (df["attempt_type"] == "AUTO_RETRY").sum()
    if "attempt_type" in df.columns else 0
)
manual_retries = (
    (df["attempt_type"] == "MANUAL_RETRY").sum()
    if "attempt_type" in df.columns else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Runs", total)
c2.metric("Successful", success)
c3.metric("Auto Retries", auto_retries)
c4.metric("Manual Retries", manual_retries)
