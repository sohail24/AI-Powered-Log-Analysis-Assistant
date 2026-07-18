"""Data models for the segmentation layer.

Defines the enriched log line (``ParsedLogLine``), the per-batch
container (``BatchExecution``), and the top-level result produced
by the de-interleaving engine (``DeinterleavedResult``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from app.config.constants import BatchStatus


@dataclass
class ParsedLogLine:
    """Enriched version of RawLogLine with extracted fields.

    Attributes:
        raw: Original unmodified line text.
        source_file: Absolute path of the origin file.
        file_line_number: 1-indexed line number in the source file.
        unified_line_number: 1-indexed position in unified output.
        ingestion_id: UUID4 string unique to this line.
        parsed_timestamp: Extracted timestamp, or None if unparseable.
        level: Normalised log level in uppercase, ``"UNKNOWN"`` if undetected.
        correlation_id: Extracted CID, or None if absent.
        message: Raw line text (same as *raw* for now).
        is_orphan: True if no CID was found on this line.
        orphan_attributed_to: CID this orphan was later attributed to.
    """

    raw: str
    source_file: str
    file_line_number: int
    unified_line_number: int
    ingestion_id: str
    parsed_timestamp: Optional[datetime]
    level: str = "UNKNOWN"
    correlation_id: Optional[str] = None
    message: str = ""
    is_orphan: bool = False
    orphan_attributed_to: Optional[str] = None


@dataclass
class BatchExecution:
    """Container for all lines belonging to a single batch execution.

    Attributes:
        correlation_id: The CID that groups these lines.
        lines: Ordered list of ParsedLogLine for this batch.
        source_files: Unique files that contributed lines.
        start_time: Earliest parsed timestamp in this batch.
        end_time: Latest parsed timestamp in this batch.
        status: Lifecycle status (SUCCESS, FAILED, etc.).
        job_name: Detected job name, ``"unknown_job"`` if undetectable.
        error_count: Number of lines with ERROR / FATAL / CRITICAL level.
        warn_count: Number of lines with WARN / WARNING level.
        total_lines: Number of lines in this batch.
        orphan_lines_count: How many orphan lines were attributed here.
        has_start_marker: Whether a START marker was detected.
        has_end_marker: Whether an END marker was detected.
    """

    correlation_id: str
    lines: List[ParsedLogLine] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: BatchStatus = BatchStatus.UNKNOWN
    job_name: str = "unknown_job"
    error_count: int = 0
    warn_count: int = 0
    total_lines: int = 0
    orphan_lines_count: int = 0
    has_start_marker: bool = False
    has_end_marker: bool = False


@dataclass
class DeinterleavedResult:
    """Top-level output of the de-interleaving engine.

    Attributes:
        batches: Map of correlation_id → BatchExecution.
        unattributed_lines: Truly orphaned lines with no CID found.
        total_lines_processed: Total lines fed into the engine.
        total_batches_found: Number of distinct CIDs discovered.
        batches_with_no_start_marker: CIDs missing a START pattern.
        batches_with_no_end_marker: CIDs missing an END pattern.
        cid_coverage_percent: Percentage of lines that had a CID.
    """

    batches: Dict[str, BatchExecution] = field(default_factory=dict)
    unattributed_lines: List[ParsedLogLine] = field(default_factory=list)
    total_lines_processed: int = 0
    total_batches_found: int = 0
    batches_with_no_start_marker: List[str] = field(default_factory=list)
    batches_with_no_end_marker: List[str] = field(default_factory=list)
    cid_coverage_percent: float = 0.0
