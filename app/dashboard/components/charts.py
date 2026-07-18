"""Reusable Plotly chart builders for the dashboard.

All functions accept a ``pd.DataFrame`` and return a ``go.Figure``
so pages can call ``st.plotly_chart(fig, use_container_width=True)``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Colour palettes ──────────────────────────────────────────────

STATUS_COLOURS = {
    "SUCCESS": "#22c55e",   # green-500
    "FAILED": "#ef4444",    # red-500
    "PARTIAL": "#f97316",   # orange-500
    "UNKNOWN": "#94a3b8",   # slate-400
    "RUNNING": "#3b82f6",   # blue-500
}

SEVERITY_COLOURS = {
    "LOW": "#60a5fa",       # blue-400
    "MEDIUM": "#fb923c",    # orange-400
    "HIGH": "#f87171",      # red-400
    "CRITICAL": "#7f1d1d",  # red-900
}


def status_pie_chart(data: pd.DataFrame) -> go.Figure:
    """Pie chart of batch execution status distribution.

    Args:
        data: DataFrame with columns ``status`` and ``count``.

    Returns:
        Plotly figure.
    """
    if data.empty:
        return _empty_figure("No data for status breakdown")

    colours = [STATUS_COLOURS.get(s, "#94a3b8") for s in data["status"]]
    fig = go.Figure(
        go.Pie(
            labels=data["status"],
            values=data["count"],
            marker_colors=colours,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        **_dark_layout(title="Status Breakdown"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def batch_gantt_chart(data: pd.DataFrame) -> go.Figure:
    """Horizontal Gantt-style chart of batch executions.

    Args:
        data: DataFrame with columns:
              ``job_name``, ``start_time``, ``end_time``,
              ``status``, ``correlation_id``.

    Returns:
        Plotly figure.
    """
    if data.empty:
        return _empty_figure("No executions to display")

    # Fill missing end_time with start_time + 1 min so bars are visible.
    data = data.copy()
    data["end_time"] = data["end_time"].fillna(
        data["start_time"] + pd.Timedelta(minutes=1)
    )
    data["colour"] = data["status"].map(STATUS_COLOURS).fillna("#94a3b8")

    fig = px.timeline(
        data,
        x_start="start_time",
        x_end="end_time",
        y="job_name",
        color="status",
        color_discrete_map=STATUS_COLOURS,
        custom_data=["correlation_id", "status"],
        title="Batch Execution Timeline",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "CID: %{customdata[0]}<br>"
            "Status: %{customdata[1]}<br>"
            "Start: %{x}<extra></extra>"
        )
    )
    fig.update_layout(**_dark_layout())
    fig.update_yaxes(autorange="reversed")
    return fig


def error_bar_chart(data: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of error categories by count.

    Args:
        data: DataFrame with columns:
              ``error_category``, ``total_count``, ``severity``.

    Returns:
        Plotly figure.
    """
    if data.empty:
        return _empty_figure("No errors recorded")

    data = data.sort_values("total_count", ascending=True)
    colours = [
        SEVERITY_COLOURS.get(str(s).upper(), "#60a5fa")
        for s in data["severity"]
    ]

    fig = go.Figure(
        go.Bar(
            x=data["total_count"],
            y=data["error_category"],
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}: %{x} occurrences<extra></extra>",
        )
    )
    fig.update_layout(
        **_dark_layout(title="Error Categories"),
        xaxis_title="Count",
        yaxis_title="Category",
    )
    return fig


def run_history_chart(data: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of run outcomes by date.

    Args:
        data: DataFrame with columns:
              ``run_date``, ``status``, ``count``.

    Returns:
        Plotly figure.
    """
    if data.empty:
        return _empty_figure("No run history data")

    fig = px.bar(
        data,
        x="run_date",
        y="count",
        color="status",
        color_discrete_map=STATUS_COLOURS,
        barmode="stack",
        title="Daily Run Outcomes",
    )
    fig.update_layout(
        **_dark_layout(),
        xaxis_title="Date",
        yaxis_title="Run Count",
        legend_title="Status",
    )
    return fig


def attempt_type_chart(data: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of attempt types by date.

    Args:
        data: DataFrame with columns:
              ``run_date``, ``attempt_type``, ``count``.

    Returns:
        Plotly figure.
    """
    if data.empty:
        return _empty_figure("No attempt type data")

    type_colours = {
        "SCHEDULED": "#22c55e",
        "AUTO_RETRY": "#f97316",
        "MANUAL_RETRY": "#ef4444",
        "UNKNOWN": "#94a3b8",
    }
    fig = px.bar(
        data,
        x="run_date",
        y="count",
        color="attempt_type",
        color_discrete_map=type_colours,
        barmode="group",
        title="Attempt Type Distribution",
    )
    fig.update_layout(
        **_dark_layout(),
        xaxis_title="Date",
        yaxis_title="Count",
    )
    return fig


def error_trend_chart(data: pd.DataFrame) -> go.Figure:
    """Line chart of top error categories over time.

    Args:
        data: DataFrame with columns:
              ``call_date``, ``error_category``, ``count``.

    Returns:
        Plotly figure.
    """
    if data.empty:
        return _empty_figure("No error trend data")

    fig = px.line(
        data,
        x="call_date",
        y="count",
        color="error_category",
        markers=True,
        title="Error Trend (Last 7 Days)",
    )
    fig.update_layout(
        **_dark_layout(),
        xaxis_title="Date",
        yaxis_title="Error Count",
    )
    return fig


# ── Private helpers ──────────────────────────────────────────────


def _dark_layout(title: Optional[str] = None) -> dict:
    """Return a shared dark-theme Plotly layout dict."""
    base: dict = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(15,23,42,0.0)",
        "plot_bgcolor": "rgba(15,23,42,0.0)",
        "font": {"family": "Inter, sans-serif", "size": 13},
        "margin": {"t": 48, "b": 24, "l": 12, "r": 12},
    }
    if title:
        base["title"] = {"text": title, "font": {"size": 16}}
    return base


def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with an annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#94a3b8"},
    )
    fig.update_layout(**_dark_layout())
    return fig
