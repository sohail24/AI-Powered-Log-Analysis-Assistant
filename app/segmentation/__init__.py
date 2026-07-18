"""Segmentation module — CID extraction and de-interleaving.

Public API:
    LineParser            — extracts CID, level, job markers from a line.
    DeinterleavingEngine  — splits a stitched log into per-batch executions.
    ParsedLogLine         — enriched per-line data model.
    BatchExecution        — per-batch container.
    DeinterleavedResult   — engine output.
"""

from app.segmentation.extractor import LineParser
from app.segmentation.deinterleaver import DeinterleavingEngine
from app.segmentation.models import (
    BatchExecution,
    DeinterleavedResult,
    ParsedLogLine,
)

__all__ = [
    "LineParser",
    "DeinterleavingEngine",
    "ParsedLogLine",
    "BatchExecution",
    "DeinterleavedResult",
]
