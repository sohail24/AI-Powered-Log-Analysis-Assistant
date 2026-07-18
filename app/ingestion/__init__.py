"""Ingestion module — log stitching and parsing.

Public API:
    LogStitcher  — merges multiple log files into a unified timeline.
    RawLogLine   — per-line data model.
    StitchedLog  — merged result data model.
"""

from app.ingestion.models import RawLogLine, StitchedLog
from app.ingestion.stitcher import LogStitcher

__all__ = ["LogStitcher", "RawLogLine", "StitchedLog"]
