"""Error Analysis — aggregated error breakdown across all batches.

Shows a horizontal bar chart of error categories, an expandable
detail table with sample lines, and a 7-day error trend line chart.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_root = str(_pathlib.Path(__file__).resolve().parents[3])
if _root not in _sys.path: _sys.path.insert(0, _root)

from datetime import date, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Error Analysis | Batch Intelligence Platform",
    page_icon="🔥",
    layout="wide",
)

from app.dashboard.components.charts import error_bar_chart, error_trend_chart
from app.dashboard.components.filters import date_range_filter, environment_filter
from app.dashboard.components.tables import error_summary_table
from app.dashboard.shared import get_repo

repo = get_repo()

# ── Sidebar filters ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    env = environment_filter(key="errors_env")
    start_date, end_date = date_range_filter(key_prefix="errors", default_days=7)
    severity_filter = st.multiselect(
        "Severity",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        key="errors_severity",
    )

st.title("🔥 Error Analysis")
st.caption(f"Showing **{env}** | {start_date} → {end_date}")
st.divider()

# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading error summaries…"):
    rows = repo.get_error_summaries_by_date_range(start_date, end_date, env)

if not rows:
    st.info("✅ No errors found for the selected date range and environment.")
    st.stop()

df = pd.DataFrame([dict(r) for r in rows])

# Apply severity filter.
if severity_filter and "severity" in df.columns:
    df = df[df["severity"].isin(severity_filter)]

if df.empty:
    st.info("No errors match the selected severity filters.")
    st.stop()

# ── Top Error Categories chart ────────────────────────────────────
st.subheader("Top Error Categories")

category_totals = (
    df.groupby(["error_category", "severity"])
    .agg(total_count=("count", "sum"))
    .reset_index()
    .sort_values("total_count", ascending=False)
    .head(15)
)

st.plotly_chart(
    error_bar_chart(category_totals),
    use_container_width=True,
)

st.divider()

# ── Error Detail Table ───────────────────────────────────────────
st.subheader("Error Detail")

detail_cols = [c for c in [
    "job_name", "run_number", "error_category", "count",
    "severity", "first_seen", "last_seen",
] if c in df.columns]

display_df = df[detail_cols].sort_values(
    ["severity", "count"],
    ascending=[False, False],
    key=lambda col: col.map(
        {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    ) if col.name == "severity" else col,
) if "severity" in df.columns else df[detail_cols]

error_summary_table(display_df)

# ── Expandable sample lines ───────────────────────────────────────
if "error_category" in df.columns:
    st.markdown("---")
    st.subheader("Sample Error Lines")
    st.caption("Select a category to inspect sample log lines.")

    categories = sorted(df["error_category"].unique().tolist())
    selected_cat = st.selectbox(
        "Error Category",
        options=categories,
        key="errors_cat_select",
    )

    cat_rows = df[df["error_category"] == selected_cat]
    for _, row in cat_rows.iterrows():
        samples = row.get("sample_lines") or []
        if isinstance(samples, str):
            import json
            try:
                samples = json.loads(samples)
            except Exception:
                samples = [samples]
        if samples:
            with st.expander(
                f"📋 {row.get('job_name', '?')} — {row.get('count', 0)} occurrences",
                expanded=False,
            ):
                for s in samples:
                    st.code(s, language=None)

st.divider()

# ── Error Trend chart ────────────────────────────────────────────
st.subheader("Error Trend (Last 7 Days)")

trend_end = date.today()
trend_start = trend_end - timedelta(days=6)

with st.spinner("Loading trend data…"):
    trend_rows = repo.get_error_summaries_by_date_range(trend_start, trend_end, env)

if trend_rows:
    trend_df = pd.DataFrame([dict(r) for r in trend_rows])
    if "first_seen" in trend_df.columns and "error_category" in trend_df.columns:
        trend_df["call_date"] = pd.to_datetime(
            trend_df["first_seen"], errors="coerce"
        ).dt.date

        # Top 5 categories by total count.
        top5 = (
            trend_df.groupby("error_category")["count"]
            .sum()
            .nlargest(5)
            .index.tolist()
        )
        trend_agg = (
            trend_df[trend_df["error_category"].isin(top5)]
            .groupby(["call_date", "error_category"])
            .agg(count=("count", "sum"))
            .reset_index()
        )
        st.plotly_chart(
            error_trend_chart(trend_agg),
            use_container_width=True,
        )
    else:
        st.info("Insufficient data for trend chart.")
else:
    st.info("No trend data available.")
