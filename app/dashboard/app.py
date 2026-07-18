"""Batch Intelligence Platform — Streamlit entry point.

Renders the home / landing page with:
  - Sidebar: env selector, date range, Run Pipeline button
  - Home: platform overview cards, quick stats, recent activity
"""

from __future__ import annotations

# ── Path bootstrap (must be before all app.* imports) ───────────
import sys as _sys
import pathlib as _pathlib
_root = str(_pathlib.Path(__file__).resolve().parents[2])
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import traceback
from datetime import date, datetime, timedelta

import streamlit as st

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Batch Intelligence Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-Powered Batch Log Analysis Platform",
    },
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }

    /* Sidebar dark glass */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
        backdrop-filter: blur(12px);
    }

    /* Card-style metric blocks */
    div[data-testid="stMetric"] {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: border-color 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.6);
    }

    /* Feature card */
    .feature-card {
        background: rgba(30, 27, 75, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    .feature-card:hover {
        border-color: rgba(139, 92, 246, 0.5);
        background: rgba(30, 27, 75, 0.7);
        transform: translateY(-2px);
    }
    .feature-card h3 {
        color: #a5b4fc;
        margin-top: 0;
    }
    .feature-card p {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* Hero gradient text */
    .hero-title {
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    /* Glowing divider */
    hr {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.25);
        margin: 1.5rem 0;
    }

    /* Pipeline status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-ready { background: rgba(34,197,94,0.2); color: #86efac; }
    .badge-running { background: rgba(59,130,246,0.2); color: #93c5fd; }
    .badge-error { background: rgba(239,68,68,0.2); color: #fca5a5; }
    </style>
    """,
    unsafe_allow_html=True,
)

from app.config.settings import Settings
from app.dashboard.shared import get_repo, get_settings
from app.pipeline import IntelligencePipeline

settings = get_settings()
repo = get_repo()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 1rem 0;'>"
        "<span style='font-size:2rem'>🔍</span><br>"
        "<span style='color:#a5b4fc; font-weight:700; font-size:1.1rem'>"
        "Batch Intelligence</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Environment.
    env = st.selectbox(
        "Environment",
        options=["prod", "staging", "uat"],
        key="sidebar_env",
    )

    # Date range.
    today = date.today()
    start_date = st.date_input(
        "From Date",
        value=today,
        max_value=today,
        key="sidebar_start",
    )
    end_date = st.date_input(
        "To Date",
        value=today,
        max_value=today,
        key="sidebar_end",
    )

    st.divider()

    # Log directory.
    log_dir = st.text_input(
        "Log Directory",
        value=settings.log_directory,
        key="sidebar_log_dir",
        help="Directory containing raw log files to process.",
    )

    # Run Pipeline button.
    run_clicked = st.button(
        "▶ Run Pipeline",
        type="primary",
        use_container_width=True,
        key="sidebar_run",
    )

    # Pipeline status.
    if "last_run_time" in st.session_state:
        last = st.session_state["last_run_time"]
        batches = st.session_state.get("last_run_batches", 0)
        st.markdown(
            f"<small style='color:#64748b'>Last run: "
            f"{last.strftime('%H:%M:%S')}<br>"
            f"Batches: {batches}</small>",
            unsafe_allow_html=True,
        )

# ── Run Pipeline ─────────────────────────────────────────────────
if run_clicked:
    st.session_state.pop("pipeline_error", None)
    with st.spinner("🔄 Running ingestion pipeline…"):
        try:
            pipeline = IntelligencePipeline(settings)
            result = pipeline.run(
                log_directory=log_dir,
                environment=env,
            )
            st.session_state["last_run_time"] = datetime.now()
            st.session_state["last_run_batches"] = result.executions_stored
            st.session_state["last_pipeline_result"] = result
        except Exception as exc:
            st.session_state["pipeline_error"] = str(exc)
            st.session_state["pipeline_traceback"] = traceback.format_exc()

    if "pipeline_error" in st.session_state:
        st.error(
            f"Pipeline failed: {st.session_state['pipeline_error']}"
        )
        with st.expander("Traceback"):
            st.code(st.session_state["pipeline_traceback"])
    elif "last_pipeline_result" in st.session_state:
        r = st.session_state["last_pipeline_result"]
        st.success(
            f"✅ Pipeline complete — **{r.executions_stored}** batches, "
            f"**{r.errors_stored}** errors, **{r.chunks_stored}** chunks"
        )
        if r.warnings:
            with st.expander(f"⚠️ {len(r.warnings)} warnings"):
                for w in r.warnings:
                    st.warning(w)

# ── Hero section ─────────────────────────────────────────────────
st.markdown(
    "<h1 class='hero-title'>Batch Intelligence Platform</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#94a3b8; font-size:1.1rem; max-width:640px'>"
    "AI-powered batch log analysis — ingest, de-interleave, and surface "
    "actionable insights from multi-job log streams without manual sifting."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── Quick stats from DB ───────────────────────────────────────────
with st.spinner("Loading stats…"):
    try:
        today_execs = repo.get_executions_by_date_range(today, today, env)
        total_today = len(today_execs)
        success_today = sum(1 for r in today_execs if r["status"] == "SUCCESS")
        failed_today = sum(1 for r in today_execs if r["status"] == "FAILED")
        total_errors_today = sum((r["error_count"] or 0) for r in today_execs)
    except Exception:
        total_today = success_today = failed_today = total_errors_today = 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Today's Batches", total_today)
c2.metric(
    "Success Rate",
    f"{(success_today / total_today * 100):.0f}%" if total_today else "—",
)
c3.metric("Failures Today", failed_today)
c4.metric("Total Errors Today", total_errors_today)

st.divider()

# ── Feature cards ────────────────────────────────────────────────
st.subheader("📋 Dashboard Pages")

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown(
        "<div class='feature-card'>"
        "<h3>📊 Batch Overview</h3>"
        "<p>Daily snapshot — status grid, pie chart breakdown, "
        "and Gantt execution timeline.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with f2:
    st.markdown(
        "<div class='feature-card'>"
        "<h3>📈 Job Run History</h3>"
        "<p>Per-job historical runs — attempt types, "
        "retry patterns, and daily outcome trends.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with f3:
    st.markdown(
        "<div class='feature-card'>"
        "<h3>🔥 Error Analysis</h3>"
        "<p>Rule-based error categorisation by severity — "
        "sample lines, trend charts, top-category breakdown.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with f4:
    st.markdown(
        "<div class='feature-card'>"
        "<h3>📄 Raw Log Viewer</h3>"
        "<p>Paginated, filterable raw log display with level "
        "and keyword filtering per batch.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Recent activity ───────────────────────────────────────────────
st.subheader("🕐 Recent Executions")

try:
    recent_rows = repo.get_executions_by_date_range(
        today - timedelta(days=1), today, env
    )
    if recent_rows:
        import pandas as pd

        df = pd.DataFrame([dict(r) for r in recent_rows[-10:]])
        display_cols = [c for c in [
            "job_name", "correlation_id", "status",
            "start_time", "duration_seconds", "error_count",
        ] if c in df.columns]

        STATUS_EMOJIS = {
            "SUCCESS": "✅", "FAILED": "❌",
            "PARTIAL": "⚠️", "UNKNOWN": "❓",
        }
        if "status" in df.columns:
            df["status"] = df["status"].map(
                lambda s: f"{STATUS_EMOJIS.get(s, '')} {s}"
            )

        st.dataframe(
            df[display_cols].rename(columns={
                "job_name": "Job", "correlation_id": "CID",
                "status": "Status", "start_time": "Started",
                "duration_seconds": "Duration (s)", "error_count": "Errors",
            }),
            use_container_width=True,
            height=min(38 * len(df) + 38, 420),
        )
    else:
        st.info("🔍 No executions in the last 24 hours. Run the pipeline to process logs.")
except Exception as exc:
    st.warning(f"Could not load recent executions: {exc}")
