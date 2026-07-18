"""Metrics generation for batch executions.

Computes throughput, error rates, gap analysis, record-count
extraction, and peak-error-window detection — all from the lines
already present in a ``BatchExecution``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from app.config.constants import ERROR_LEVELS, LogLevel, WARN_LEVELS
from app.segmentation.models import BatchExecution, ParsedLogLine

# Pattern to extract record/row/item counts from log text.
_RECORD_COUNT_RE = re.compile(
    r"(\d[\d,]*)\s*(?:records?|rows?|items?|entries?)",
    re.IGNORECASE,
)


@dataclass
class BatchMetrics:
    """Computed metrics for a single batch execution.

    Attributes:
        correlation_id: The batch's CID.
        duration_seconds: Wall-clock runtime, or ``None``.
        lines_per_second: Throughput proxy, or ``None``.
        error_rate_percent: Error lines / total lines × 100.
        warn_rate_percent: Warning lines / total lines × 100.
        lines_in_first_10_percent: Startup log density.
        lines_in_last_10_percent: Shutdown log density.
        longest_gap_seconds: Longest silence between consecutive lines.
        estimated_record_count: Largest N from "processing N records".
        peak_error_window: ISO timestamp of the 60-s window with most errors.
    """

    correlation_id: str
    duration_seconds: Optional[float] = None
    lines_per_second: Optional[float] = None
    error_rate_percent: float = 0.0
    warn_rate_percent: float = 0.0
    lines_in_first_10_percent: int = 0
    lines_in_last_10_percent: int = 0
    longest_gap_seconds: Optional[float] = None
    estimated_record_count: Optional[int] = None
    peak_error_window: Optional[str] = None


class MetricsGenerator:
    """Generate runtime metrics from a ``BatchExecution``.

    Usage::

        gen = MetricsGenerator()
        metrics = gen.generate(execution)
    """

    def generate(self, execution: BatchExecution) -> BatchMetrics:
        """Compute all metrics for *execution*.

        Args:
            execution: A ``BatchExecution`` to analyse.

        Returns:
            Fully populated ``BatchMetrics``.
        """
        total = execution.total_lines or len(execution.lines)
        duration = self._calc_duration(execution)
        lps = (total / duration) if (duration and duration > 0) else None

        error_count = self._count_level(execution.lines, ERROR_LEVELS)
        warn_count = self._count_level(execution.lines, WARN_LEVELS)

        error_rate = (error_count / total * 100) if total > 0 else 0.0
        warn_rate = (warn_count / total * 100) if total > 0 else 0.0

        ten_pct = max(1, total // 10)
        first_10 = len(execution.lines[:ten_pct])
        last_10 = len(execution.lines[-ten_pct:]) if total > 0 else 0

        return BatchMetrics(
            correlation_id=execution.correlation_id,
            duration_seconds=duration,
            lines_per_second=round(lps, 2) if lps else None,
            error_rate_percent=round(error_rate, 2),
            warn_rate_percent=round(warn_rate, 2),
            lines_in_first_10_percent=first_10,
            lines_in_last_10_percent=last_10,
            longest_gap_seconds=self._longest_gap(execution.lines),
            estimated_record_count=self._extract_record_count(execution.lines),
            peak_error_window=self._peak_error_window(execution.lines),
        )

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _calc_duration(execution: BatchExecution) -> Optional[float]:
        """Return duration in seconds between start and end."""
        if execution.start_time and execution.end_time:
            return (execution.end_time - execution.start_time).total_seconds()
        return None

    @staticmethod
    def _count_level(
        lines: List[ParsedLogLine], levels: set
    ) -> int:
        """Count lines whose level matches *levels*."""
        count = 0
        for ln in lines:
            try:
                if LogLevel(ln.level) in levels:
                    count += 1
            except ValueError:
                pass
        return count

    @staticmethod
    def _longest_gap(lines: List[ParsedLogLine]) -> Optional[float]:
        """Find the longest gap in seconds between consecutive timestamped lines."""
        ts_lines = [
            ln.parsed_timestamp
            for ln in lines
            if ln.parsed_timestamp is not None
        ]
        if len(ts_lines) < 2:
            return None

        max_gap = 0.0
        for i in range(1, len(ts_lines)):
            gap = (ts_lines[i] - ts_lines[i - 1]).total_seconds()
            if gap > max_gap:
                max_gap = gap
        return max_gap if max_gap > 0 else None

    @staticmethod
    def _extract_record_count(lines: List[ParsedLogLine]) -> Optional[int]:
        """Extract the largest record/row/item count mentioned in logs.

        Handles comma-formatted numbers like ``4,200``.
        """
        largest: Optional[int] = None
        for ln in lines:
            for m in _RECORD_COUNT_RE.finditer(ln.raw):
                raw_num = m.group(1).replace(",", "")
                try:
                    num = int(raw_num)
                    if largest is None or num > largest:
                        largest = num
                except ValueError:
                    continue
        return largest

    @staticmethod
    def _peak_error_window(
        lines: List[ParsedLogLine],
    ) -> Optional[str]:
        """Find the 60-second window with the highest error density.

        Returns the start timestamp of that window as ISO string,
        or ``None`` if there are no timestamped error lines.
        """
        error_ts: List[datetime] = []
        for ln in lines:
            if ln.parsed_timestamp is None:
                continue
            try:
                if LogLevel(ln.level) in ERROR_LEVELS:
                    error_ts.append(ln.parsed_timestamp)
            except ValueError:
                pass

        if not error_ts:
            return None

        error_ts.sort()

        best_start = error_ts[0]
        best_count = 0

        for i, start in enumerate(error_ts):
            end = start + timedelta(seconds=60)
            count = sum(1 for t in error_ts[i:] if t <= end)
            if count > best_count:
                best_count = count
                best_start = start

        return best_start.isoformat()
