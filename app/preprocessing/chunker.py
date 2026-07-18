"""Semantic log chunker for batch executions.

Splits a ``BatchExecution`` into meaningful chunks (HEADER,
ERROR_CLUSTER, RECOVERY, FOOTER, BODY) based on line position,
error proximity, and keyword matching.  Chunks are later stored
in the database and fed into the vector DB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Set

from app.config.constants import ERROR_LEVELS, LogLevel
from app.segmentation.models import BatchExecution, ParsedLogLine

# Keywords that indicate recovery / retry behaviour.
_RECOVERY_RE = re.compile(
    r"\bretr(?:y|ying|ied)\b|\breconnect\b|\bfallback\b|\brecovering\b|\brecovered\b",
    re.IGNORECASE,
)

# How many seconds around an error line to include in the cluster.
_ERROR_CLUSTER_WINDOW_SEC = 30


@dataclass
class LogChunk:
    """A semantically meaningful slice of a batch log.

    Attributes:
        correlation_id: The batch's CID.
        chunk_index: 0-based sequential index.
        chunk_type: HEADER | ERROR_CLUSTER | RECOVERY | FOOTER | BODY.
        source_file: Basename of the primary source file.
        start_unified_line: Unified line number of the first line.
        end_unified_line: Unified line number of the last line.
        content: Joined text of all lines in this chunk.
        line_count: Number of lines in this chunk.
    """

    correlation_id: str
    chunk_index: int
    chunk_type: str
    source_file: str
    start_unified_line: int
    end_unified_line: int
    content: str
    line_count: int


class LogChunker:
    """Split a ``BatchExecution`` into semantic log chunks.

    Usage::

        chunker = LogChunker()
        chunks = chunker.chunk(execution)
    """

    def chunk(
        self,
        execution: BatchExecution,
        header_lines: int = 20,
        footer_lines: int = 20,
    ) -> List[LogChunk]:
        """Segment *execution* into semantic chunks.

        Steps:
          1. Mark HEADER (first *header_lines* lines).
          2. Mark FOOTER (last *footer_lines* lines).
          3. Mark ERROR_CLUSTER (lines within 30 s of an error).
          4. Mark RECOVERY (post-error-cluster lines with retry keywords).
          5. Remaining lines become BODY.
          6. Merge consecutive same-type lines into single chunks.

        Args:
            execution: The batch to chunk.
            header_lines: How many leading lines form the HEADER.
            footer_lines: How many trailing lines form the FOOTER.

        Returns:
            Ordered list of ``LogChunk``.
        """
        lines = execution.lines
        if not lines:
            return []

        n = len(lines)
        labels: Dict[int, str] = {}  # index → chunk_type

        # Step 1: HEADER
        for i in range(min(header_lines, n)):
            labels[i] = "HEADER"

        # Step 2: FOOTER (may overlap HEADER for tiny batches)
        for i in range(max(0, n - footer_lines), n):
            # Footer overrides BODY but not ERROR_CLUSTER.
            if i not in labels:
                labels[i] = "FOOTER"

        # Step 3: ERROR_CLUSTER — lines within 30s of any error line.
        error_indices = self._find_error_indices(lines)
        cluster_indices = self._expand_error_clusters(lines, error_indices)
        for i in cluster_indices:
            labels[i] = "ERROR_CLUSTER"

        # Step 4: RECOVERY — lines immediately after an error cluster
        # that contain recovery keywords.
        recovery_indices = self._find_recovery_lines(lines, cluster_indices)
        for i in recovery_indices:
            labels[i] = "RECOVERY"

        # Step 5: Remaining → BODY.
        for i in range(n):
            if i not in labels:
                labels[i] = "BODY"

        # Step 6–8: Merge consecutive same-type, build LogChunk objects.
        return self._merge_and_build(execution.correlation_id, lines, labels)

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _find_error_indices(lines: List[ParsedLogLine]) -> Set[int]:
        """Return indices of lines with error-level severity."""
        indices: Set[int] = set()
        for i, ln in enumerate(lines):
            try:
                if LogLevel(ln.level) in ERROR_LEVELS:
                    indices.add(i)
            except ValueError:
                pass
        return indices

    @staticmethod
    def _expand_error_clusters(
        lines: List[ParsedLogLine],
        error_indices: Set[int],
    ) -> Set[int]:
        """Expand each error index to a cluster of nearby lines.

        All lines within ``_ERROR_CLUSTER_WINDOW_SEC`` seconds of
        any error line are included.
        """
        if not error_indices:
            return set()

        cluster: Set[int] = set()
        for ei in error_indices:
            error_ts = lines[ei].parsed_timestamp
            if error_ts is None:
                cluster.add(ei)
                continue

            window_start = error_ts - timedelta(seconds=_ERROR_CLUSTER_WINDOW_SEC)
            window_end = error_ts + timedelta(seconds=_ERROR_CLUSTER_WINDOW_SEC)

            for i, ln in enumerate(lines):
                if ln.parsed_timestamp is not None:
                    if window_start <= ln.parsed_timestamp <= window_end:
                        cluster.add(i)
                elif i == ei:
                    cluster.add(i)

        return cluster

    @staticmethod
    def _find_recovery_lines(
        lines: List[ParsedLogLine],
        cluster_indices: Set[int],
    ) -> Set[int]:
        """Find lines just after an error cluster with recovery keywords."""
        recovery: Set[int] = set()
        n = len(lines)
        for i in range(n):
            if i in cluster_indices:
                continue
            # Check if the previous line was in an error cluster.
            if (i - 1) in cluster_indices and _RECOVERY_RE.search(lines[i].raw):
                recovery.add(i)
            # Also check if a nearby predecessor (within 3 lines) was a cluster.
            elif any((i - k) in cluster_indices for k in range(1, 4) if (i - k) >= 0):
                if _RECOVERY_RE.search(lines[i].raw):
                    recovery.add(i)
        return recovery

    @staticmethod
    def _merge_and_build(
        cid: str,
        lines: List[ParsedLogLine],
        labels: Dict[int, str],
    ) -> List[LogChunk]:
        """Merge consecutive same-type lines into LogChunk objects."""
        if not lines:
            return []

        from pathlib import Path

        chunks: List[LogChunk] = []
        current_type = labels[0]
        group_lines: List[ParsedLogLine] = [lines[0]]

        for i in range(1, len(lines)):
            if labels[i] == current_type:
                group_lines.append(lines[i])
            else:
                chunks.append(
                    _build_chunk(cid, len(chunks), current_type, group_lines)
                )
                current_type = labels[i]
                group_lines = [lines[i]]

        # Flush the last group.
        chunks.append(
            _build_chunk(cid, len(chunks), current_type, group_lines)
        )
        return chunks


def _build_chunk(
    cid: str,
    index: int,
    chunk_type: str,
    lines: List[ParsedLogLine],
) -> LogChunk:
    """Construct a ``LogChunk`` from a group of lines."""
    from pathlib import Path

    content = "\n".join(ln.raw for ln in lines)
    # Primary source file = most common source among lines.
    sources = [ln.source_file for ln in lines if ln.source_file]
    primary = Path(sources[0]).name if sources else "unknown"

    return LogChunk(
        correlation_id=cid,
        chunk_index=index,
        chunk_type=chunk_type,
        source_file=primary,
        start_unified_line=lines[0].unified_line_number,
        end_unified_line=lines[-1].unified_line_number,
        content=content,
        line_count=len(lines),
    )
