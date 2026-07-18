"""Shared data models for the ingestion layer.

All models use dataclasses with full type hints. These are the
canonical representations that flow through ingestion → segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4


@dataclass
class RawLogLine:
    """A single line from a log file with parsed metadata.

    Attributes:
        raw: Original unmodified line text.
        source_file: Absolute path of the origin file.
        file_line_number: 1-indexed line number in the source file.
        parsed_timestamp: Extracted timestamp, or None if unparseable.
        unified_line_number: 1-indexed position in unified output
            (assigned after stitching / sorting).
        ingestion_id: UUID4 string unique to this line.
    """

    raw: str
    source_file: str
    file_line_number: int
    parsed_timestamp: Optional[datetime]
    unified_line_number: int = 0
    ingestion_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class StitchedLog:
    """Result of merging multiple log files into one timeline.

    Attributes:
        lines: Chronologically sorted list of RawLogLine.
        source_files: Ordered list of file paths that contributed.
        start_time: Earliest parsed timestamp across all lines.
        end_time: Latest parsed timestamp across all lines.
        total_lines: Number of lines in the unified output.
        lines_without_timestamp: Count of lines with no parsed ts.
        created_at: When the stitching operation ran.
    """

    lines: List[RawLogLine]
    source_files: List[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_lines: int
    lines_without_timestamp: int
    created_at: datetime = field(default_factory=datetime.utcnow)
