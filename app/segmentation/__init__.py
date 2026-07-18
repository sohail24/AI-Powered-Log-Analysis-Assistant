"""Segmentation module — CID extraction, de-interleaving, and job grouping.

Public API:
    LineParser            — extracts CID, level, job markers from a line.
    DeinterleavingEngine  — splits a stitched log into per-batch executions.
    JobGrouper            — groups executions by (job_name, date).
    ParsedLogLine         — enriched per-line data model.
    BatchExecution        — per-batch container.
    DeinterleavedResult   — engine output.
    JobRunGroup           — per-job-per-day container.
"""

from app.segmentation.extractor import LineParser
from app.segmentation.deinterleaver import DeinterleavingEngine
from app.segmentation.job_grouper import JobGrouper, JobRunGroup
from app.segmentation.models import (
    BatchExecution,
    DeinterleavedResult,
    ParsedLogLine,
)

__all__ = [
    "LineParser",
    "DeinterleavingEngine",
    "JobGrouper",
    "JobRunGroup",
    "ParsedLogLine",
    "BatchExecution",
    "DeinterleavedResult",
]
