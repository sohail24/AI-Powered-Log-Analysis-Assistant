"""Raw Log Viewer — paginated, filterable view of a single batch.

Loads log chunks from the database for the selected correlation_id
and renders them in a scrollable code block with level + keyword
filtering and line-number display.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_root = str(_pathlib.Path(__file__).resolve().parents[3])
if _root not in _sys.path: _sys.path.insert(0, _root)

from datetime import date
from typing import Optional

import streamlit as st

st.set_page_config(
    page_title="Raw Log Viewer | Batch Intelligence Platform",
    page_icon="📄",
    layout="wide",
)

from app.dashboard.components.filters import date_range_filter, environment_filter
from app.dashboard.components.tables import execution_metadata_card
from app.dashboard.shared import get_repo

repo = get_repo()

_LINES_PER_PAGE = 200

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    env = environment_filter(key="viewer_env")
    start_date, end_date = date_range_filter(key_prefix="viewer", default_days=1)

    # Level multiselect.
    level_filter = st.multiselect(
        "Log Level",
        options=["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"],
        default=["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"],
        key="viewer_levels",
    )

    # Keyword search.
    keyword = st.text_input(
        "Keyword Search",
        placeholder="Filter lines containing…",
        key="viewer_keyword",
    )

    show_orphans = st.checkbox("Show orphan lines", value=True, key="viewer_orphans")

st.title("📄 Raw Log Viewer")

# ── Batch selector ───────────────────────────────────────────────
with st.spinner("Loading available batches…"):
    exec_rows = repo.get_executions_by_date_range(start_date, end_date, env)

if not exec_rows:
    st.info("No batches found for the selected date range.")
    st.stop()

cid_options = [r["correlation_id"] for r in exec_rows]
cid_labels = {
    r["correlation_id"]: f"{r['correlation_id']}  [{r['job_name']} | {r['status']}]"
    for r in exec_rows
}

selected_cid = st.selectbox(
    "Select Batch (Correlation ID)",
    options=cid_options,
    format_func=lambda c: cid_labels.get(c, c),
    key="viewer_cid",
)

if not selected_cid:
    st.stop()

# ── Execution metadata ───────────────────────────────────────────
exec_row = repo.get_execution_by_cid(selected_cid)
if exec_row:
    execution_metadata_card(dict(exec_row))

st.divider()

# ── Load log chunks ──────────────────────────────────────────────
with st.spinner("Loading log chunks…"):
    chunks = repo.get_chunks_for_cid(selected_cid)

if not chunks:
    st.info(f"No log chunks stored for **{selected_cid}**.")
    st.stop()

# Assemble all lines from chunks (preserving order by start_line).
sorted_chunks = sorted(chunks, key=lambda c: c["start_line"])
all_lines: list[tuple[int, str]] = []  # (unified_line_no, raw_text)

for chunk in sorted_chunks:
    content = chunk["content"] or ""
    start_line = chunk["start_line"] or 0
    for i, raw in enumerate(content.split("\n")):
        if raw:
            all_lines.append((start_line + i, raw))

# ── Apply filters ─────────────────────────────────────────────────

def _extract_level(raw: str) -> str:
    """Best-effort level extraction from a raw log line."""
    import re
    m = re.search(r"\[(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\]", raw, re.I)
    return m.group(1).upper() if m else "INFO"


filtered: list[tuple[int, str]] = []
for line_no, raw in all_lines:
    level = _extract_level(raw)
    if level_filter and level not in level_filter:
        # Also accept lines where level maps WARN→WARNING.
        if not (level == "WARN" and "WARNING" in level_filter):
            if not (level == "WARNING" and "WARN" in level_filter):
                continue
    if keyword and keyword.lower() not in raw.lower():
        continue
    filtered.append((line_no, raw))

total_filtered = len(filtered)

if total_filtered == 0:
    st.info("No lines match the selected filters.")
    st.stop()

# ── Pagination ───────────────────────────────────────────────────
total_pages = max(1, (total_filtered + _LINES_PER_PAGE - 1) // _LINES_PER_PAGE)

if "viewer_page" not in st.session_state:
    st.session_state["viewer_page"] = 0

page = st.session_state["viewer_page"]

col_info, col_prev, col_next = st.columns([4, 1, 1])
with col_info:
    st.caption(
        f"Showing {total_filtered} lines · Page {page + 1} / {total_pages}"
    )
with col_prev:
    if st.button("◀ Prev", disabled=page == 0, key="viewer_prev"):
        st.session_state["viewer_page"] = max(0, page - 1)
        st.rerun()
with col_next:
    if st.button("Next ▶", disabled=page >= total_pages - 1, key="viewer_next"):
        st.session_state["viewer_page"] = min(total_pages - 1, page + 1)
        st.rerun()

# ── Render lines ─────────────────────────────────────────────────
page_start = page * _LINES_PER_PAGE
page_end = min(page_start + _LINES_PER_PAGE, total_filtered)
page_lines = filtered[page_start:page_end]

# Build the display block.
display_parts: list[str] = []
error_levels = {"ERROR", "FATAL", "CRITICAL"}

for line_no, raw in page_lines:
    level = _extract_level(raw)
    prefix = f"[{line_no:>6}] "
    display_parts.append(prefix + raw)

display_text = "\n".join(display_parts)

st.code(display_text, language=None)

# ── Chunk type legend ─────────────────────────────────────────────
with st.expander("📦 Chunk Types in this Batch"):
    chunk_info = [
        {"Chunk #": c["chunk_index"], "Type": c["chunk_type"],
         "Lines": f"{c['start_line']}–{c['end_line']}",
         "Source": c["source_file"]}
        for c in sorted_chunks
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(chunk_info), use_container_width=True)
