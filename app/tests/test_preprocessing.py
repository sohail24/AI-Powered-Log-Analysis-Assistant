"""Tests for the preprocessing layer.

Covers ErrorAggregator (categorisation, severity, unknowns),
MetricsGenerator (duration, throughput, record count, gap),
and LogChunker (HEADER, FOOTER, ERROR_CLUSTER, RECOVERY, BODY).
"""

from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import uuid4

import pytest

from app.config.constants import BatchStatus, Severity
from app.preprocessing.chunker import LogChunk, LogChunker
from app.preprocessing.error_aggregator import ErrorAggregator, ErrorRecord
from app.preprocessing.metrics_generator import BatchMetrics, MetricsGenerator
from app.segmentation.models import BatchExecution, ParsedLogLine


# ── Helpers ─────────────────────────────────────────────────────


def _line(
    raw: str,
    level: str = "INFO",
    ts: datetime | None = None,
    unified: int = 0,
) -> ParsedLogLine:
    """Create a ParsedLogLine for testing."""
    return ParsedLogLine(
        raw=raw,
        source_file="/test/app.log",
        file_line_number=unified,
        unified_line_number=unified,
        ingestion_id=str(uuid4()),
        parsed_timestamp=ts,
        level=level,
        correlation_id="CID-TEST",
        message=raw,
    )


def _exe(
    lines: List[ParsedLogLine],
    cid: str = "CID-TEST",
    status: BatchStatus = BatchStatus.SUCCESS,
    start: datetime | None = None,
    end: datetime | None = None,
    error_count: int | None = None,
    warn_count: int | None = None,
) -> BatchExecution:
    """Create a BatchExecution wrapping *lines*."""
    errs = error_count if error_count is not None else sum(
        1 for ln in lines if ln.level in {"ERROR", "FATAL", "CRITICAL"}
    )
    warns = warn_count if warn_count is not None else sum(
        1 for ln in lines if ln.level in {"WARN", "WARNING"}
    )
    return BatchExecution(
        correlation_id=cid,
        lines=lines,
        source_files=["/test/app.log"],
        start_time=start,
        end_time=end,
        status=status,
        job_name="test-job",
        error_count=errs,
        warn_count=warns,
        total_lines=len(lines),
        orphan_lines_count=0,
        has_start_marker=True,
        has_end_marker=True,
    )


# ══════════════════════════════════════════════════════════════
# ErrorAggregator Tests
# ══════════════════════════════════════════════════════════════


class TestErrorAggregatorCategories:
    """Verify keyword-based category detection."""

    def test_db_connectivity_jdbc(self) -> None:
        """JDBC error → DatabaseConnectivity."""
        lines = [_line("ERROR: JDBC connection failed", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert len(records) == 1
        assert records[0].error_category == "DatabaseConnectivity"

    def test_db_connectivity_connection_refused(self) -> None:
        """Connection refused → DatabaseConnectivity."""
        lines = [_line("ERROR: connection refused to DB host", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "DatabaseConnectivity"

    def test_db_connectivity_pool_exhausted(self) -> None:
        """Pool exhausted → DatabaseConnectivity."""
        lines = [_line("ERROR: pool exhausted, waiting for connection", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "DatabaseConnectivity"

    def test_auth_failure_401(self) -> None:
        """401 in error line → AuthenticationFailure."""
        lines = [_line("ERROR: HTTP 401 Unauthorized response", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "AuthenticationFailure"

    def test_auth_failure_permission_denied(self) -> None:
        """Permission denied → AuthenticationFailure."""
        lines = [_line("ERROR: permission denied on resource /admin", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "AuthenticationFailure"

    def test_null_pointer_exception(self) -> None:
        """NullPointerException → NullPointerException category."""
        lines = [_line("FATAL: java.lang.NullPointerException at Main.java:42", level="FATAL")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "NullPointerException"

    def test_file_not_found(self) -> None:
        """No such file → FileNotFound."""
        lines = [_line("ERROR: no such file /data/input.csv", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "FileNotFound"

    def test_network_timeout(self) -> None:
        """Read timeout → NetworkTimeout."""
        lines = [_line("ERROR: read timeout waiting for API response", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "NetworkTimeout"

    def test_out_of_memory(self) -> None:
        """OutOfMemoryError → OutOfMemory category."""
        lines = [_line("FATAL: java.lang.OutOfMemoryError: Java heap space", level="FATAL")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "OutOfMemory"

    def test_out_of_memory_severity_critical(self) -> None:
        """OOM errors are CRITICAL severity."""
        lines = [_line("FATAL: java.lang.OutOfMemoryError: heap space", level="FATAL")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].severity == Severity.CRITICAL

    def test_data_validation(self) -> None:
        """Constraint violation → DataValidation."""
        lines = [_line("ERROR: constraint violation on table users", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "DataValidation"

    def test_retry_exhausted(self) -> None:
        """Max retries → RetryExhausted."""
        lines = [_line("ERROR: max retries reached, giving up", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "RetryExhausted"

    def test_job_aborted(self) -> None:
        """SIGTERM → JobAborted."""
        lines = [_line("FATAL: Received SIGTERM, aborting job", level="FATAL")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "JobAborted"

    def test_job_aborted_severity_critical(self) -> None:
        """JobAborted is CRITICAL severity."""
        lines = [_line("FATAL: job killed by scheduler", level="FATAL")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].severity == Severity.CRITICAL

    def test_unknown_error_fallback(self) -> None:
        """Unmatched error → UnknownError."""
        lines = [_line("ERROR: something went wrong", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].error_category == "UnknownError"

    def test_unknown_error_medium_severity(self) -> None:
        """UnknownError has MEDIUM severity."""
        lines = [_line("ERROR: unexpected failure", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].severity == Severity.MEDIUM

    def test_db_connectivity_severity_high(self) -> None:
        """DatabaseConnectivity is HIGH severity."""
        lines = [_line("ERROR: JDBC connection refused", level="ERROR")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].severity == Severity.HIGH


class TestErrorAggregatorCounting:
    """Verify counts, timestamps, and sample capping."""

    def test_count_multiple_same_category(self) -> None:
        """Multiple errors of the same category increment count."""
        lines = [
            _line("ERROR: JDBC timeout #1", level="ERROR"),
            _line("ERROR: JDBC timeout #2", level="ERROR"),
            _line("ERROR: JDBC timeout #3", level="ERROR"),
        ]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].count == 3

    def test_samples_capped_at_three(self) -> None:
        """sample_lines is capped at 3 even with 10 errors."""
        lines = [_line(f"ERROR: JDBC timeout #{i}", level="ERROR") for i in range(10)]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert len(records[0].sample_lines) <= 3

    def test_sorted_by_count_descending(self) -> None:
        """Categories sorted with highest count first."""
        lines = [
            _line("ERROR: JDBC error", level="ERROR"),
            _line("ERROR: JDBC error", level="ERROR"),
            _line("ERROR: no such file", level="ERROR"),
        ]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].count >= records[-1].count

    def test_info_lines_skipped(self) -> None:
        """INFO lines are not included in error aggregation."""
        lines = [
            _line("INFO: JDBC connection OK", level="INFO"),
            _line("ERROR: JDBC connection refused", level="ERROR"),
        ]
        records = ErrorAggregator().aggregate(_exe(lines))
        # Only 1 error record.
        assert sum(r.count for r in records) == 1

    def test_empty_batch_returns_empty(self) -> None:
        """No error lines → empty result."""
        lines = [_line("INFO: all good", level="INFO")]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records == []

    def test_timestamps_tracked(self) -> None:
        """first_seen / last_seen are populated from parsed_timestamp."""
        ts1 = datetime(2025, 6, 13, 1, 0, 0)
        ts2 = datetime(2025, 6, 13, 2, 0, 0)
        lines = [
            _line("ERROR: JDBC #1", level="ERROR", ts=ts1),
            _line("ERROR: JDBC #2", level="ERROR", ts=ts2),
        ]
        records = ErrorAggregator().aggregate(_exe(lines))
        assert records[0].first_seen == ts1
        assert records[0].last_seen == ts2

    def test_multiple_categories(self) -> None:
        """Two different categories produce two records."""
        lines = [
            _line("ERROR: JDBC timeout", level="ERROR"),
            _line("FATAL: OutOfMemoryError heap space", level="FATAL"),
        ]
        records = ErrorAggregator().aggregate(_exe(lines))
        categories = {r.error_category for r in records}
        assert "DatabaseConnectivity" in categories
        assert "OutOfMemory" in categories


# ══════════════════════════════════════════════════════════════
# MetricsGenerator Tests
# ══════════════════════════════════════════════════════════════


class TestMetricsDuration:
    """Duration and throughput calculations."""

    def test_duration_seconds(self) -> None:
        """Duration is correctly computed from start/end."""
        lines = [_line("INFO: start")]
        exe = _exe(
            lines,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 5, 0),
        )
        m = MetricsGenerator().generate(exe)
        assert m.duration_seconds == 300.0

    def test_duration_none_when_missing(self) -> None:
        """duration_seconds is None when timestamps are missing."""
        m = MetricsGenerator().generate(_exe([_line("INFO: x")]))
        assert m.duration_seconds is None

    def test_lines_per_second(self) -> None:
        """lines_per_second = total_lines / duration_seconds."""
        lines = [_line("INFO: x") for _ in range(60)]
        exe = _exe(
            lines,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 1, 0),  # 60 seconds
        )
        m = MetricsGenerator().generate(exe)
        assert m.lines_per_second == pytest.approx(1.0, abs=0.01)


class TestMetricsRates:
    """Error rate and warn rate."""

    def test_error_rate_percent(self) -> None:
        """error_rate_percent = error_lines / total * 100."""
        lines = [
            _line("INFO: ok", level="INFO"),
            _line("INFO: ok", level="INFO"),
            _line("INFO: ok", level="INFO"),
            _line("ERROR: fail", level="ERROR"),
        ]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.error_rate_percent == pytest.approx(25.0, abs=0.1)

    def test_warn_rate_percent(self) -> None:
        """warn_rate_percent = warn_lines / total * 100."""
        lines = [
            _line("INFO: ok", level="INFO"),
            _line("WARN: slow", level="WARN"),
        ]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.warn_rate_percent == pytest.approx(50.0, abs=0.1)

    def test_zero_lines_no_crash(self) -> None:
        """Zero lines → 0.0 rates, no division error."""
        exe = _exe([])
        m = MetricsGenerator().generate(exe)
        assert m.error_rate_percent == 0.0
        assert m.warn_rate_percent == 0.0


class TestMetricsRecordCount:
    """Estimated record count extraction."""

    def test_extracts_comma_number(self) -> None:
        """'Processing 4,200 records' → 4200."""
        lines = [_line("INFO: Processing 4,200 records batch")]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.estimated_record_count == 4200

    def test_extracts_plain_number(self) -> None:
        """'1500 rows inserted' → 1500."""
        lines = [_line("INFO: 1500 rows inserted successfully")]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.estimated_record_count == 1500

    def test_extracts_items(self) -> None:
        """'Processing 999 items' → 999."""
        lines = [_line("INFO: Processing 999 items")]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.estimated_record_count == 999

    def test_returns_largest(self) -> None:
        """Returns the largest count across all lines."""
        lines = [
            _line("INFO: Loaded 100 records"),
            _line("INFO: Processing 5,000 records"),
            _line("INFO: Saved 4,900 rows"),
        ]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.estimated_record_count == 5000

    def test_none_when_no_match(self) -> None:
        """Returns None if no count pattern found."""
        lines = [_line("INFO: No numbers here")]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.estimated_record_count is None


class TestMetricsGapAndWindow:
    """Longest gap and peak error window."""

    def test_longest_gap_seconds(self) -> None:
        """Correctly finds the longest gap between consecutive lines."""
        lines = [
            _line("INFO: a", ts=datetime(2025, 6, 13, 2, 0, 0)),
            _line("INFO: b", ts=datetime(2025, 6, 13, 2, 0, 10)),  # 10s gap
            _line("INFO: c", ts=datetime(2025, 6, 13, 2, 1, 30)),  # 80s gap ← longest
            _line("INFO: d", ts=datetime(2025, 6, 13, 2, 1, 45)),  # 15s gap
        ]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.longest_gap_seconds == pytest.approx(80.0, abs=0.1)

    def test_longest_gap_none_when_no_timestamps(self) -> None:
        """None when fewer than 2 timestamped lines."""
        lines = [_line("INFO: x"), _line("INFO: y")]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.longest_gap_seconds is None

    def test_peak_error_window_present(self) -> None:
        """peak_error_window is an ISO timestamp string."""
        lines = [
            _line("ERROR: e1", level="ERROR", ts=datetime(2025, 6, 13, 2, 0, 1)),
            _line("ERROR: e2", level="ERROR", ts=datetime(2025, 6, 13, 2, 0, 5)),
            _line("ERROR: e3", level="ERROR", ts=datetime(2025, 6, 13, 2, 0, 9)),
        ]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.peak_error_window is not None
        assert "2025" in m.peak_error_window

    def test_peak_error_window_none_when_no_errors(self) -> None:
        """None when no error lines."""
        lines = [_line("INFO: all good", level="INFO",
                        ts=datetime(2025, 6, 13, 2, 0, 0))]
        m = MetricsGenerator().generate(_exe(lines))
        assert m.peak_error_window is None


# ══════════════════════════════════════════════════════════════
# LogChunker Tests
# ══════════════════════════════════════════════════════════════


class TestChunkerHeader:
    """HEADER chunk detection."""

    def test_header_chunk_present(self) -> None:
        """First chunk is HEADER for a normal batch."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(50)]
        chunks = LogChunker().chunk(_exe(lines))
        assert chunks[0].chunk_type == "HEADER"

    def test_header_line_count(self) -> None:
        """HEADER contains exactly header_lines lines (default 20)."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(50)]
        chunks = LogChunker().chunk(_exe(lines), header_lines=20)
        header = next(c for c in chunks if c.chunk_type == "HEADER")
        assert header.line_count == 20

    def test_header_respects_custom_size(self) -> None:
        """header_lines parameter is respected."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(30)]
        chunks = LogChunker().chunk(_exe(lines), header_lines=5)
        header = next(c for c in chunks if c.chunk_type == "HEADER")
        assert header.line_count == 5


class TestChunkerFooter:
    """FOOTER chunk detection."""

    def test_footer_chunk_present(self) -> None:
        """Last chunk is FOOTER for a normal batch."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(50)]
        chunks = LogChunker().chunk(_exe(lines))
        assert chunks[-1].chunk_type == "FOOTER"

    def test_footer_line_count(self) -> None:
        """FOOTER contains exactly footer_lines lines (default 20)."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(60)]
        chunks = LogChunker().chunk(_exe(lines), footer_lines=20)
        footer = next(c for c in chunks if c.chunk_type == "FOOTER")
        assert footer.line_count == 20


class TestChunkerErrorCluster:
    """ERROR_CLUSTER chunk detection."""

    def test_error_cluster_created(self) -> None:
        """Error line produces an ERROR_CLUSTER chunk."""
        base = datetime(2025, 6, 13, 2, 0, 0)
        lines = [
            _line(f"INFO: before {i}",
                  ts=datetime(2025, 6, 13, 2, 0, i), unified=i)
            for i in range(5)
        ] + [
            _line("ERROR: failure happened",
                  level="ERROR",
                  ts=datetime(2025, 6, 13, 2, 0, 20), unified=20)
        ] + [
            _line(f"INFO: after {i}",
                  ts=datetime(2025, 6, 13, 2, 0, 21 + i), unified=21 + i)
            for i in range(5)
        ]
        chunks = LogChunker().chunk(_exe(lines), header_lines=3, footer_lines=3)
        types = {c.chunk_type for c in chunks}
        assert "ERROR_CLUSTER" in types

    def test_error_cluster_contains_error_line(self) -> None:
        """ERROR_CLUSTER chunk contains the actual error message."""
        lines = (
            [_line(f"INFO: startup {i}", ts=datetime(2025, 6, 13, 2, 0, i), unified=i)
             for i in range(25)]
            + [_line("ERROR: big failure",
                     level="ERROR",
                     ts=datetime(2025, 6, 13, 2, 0, 30), unified=25)]
            + [_line(f"INFO: after {i}",
                     ts=datetime(2025, 6, 13, 2, 1, i), unified=26 + i)
               for i in range(25)]
        )
        chunks = LogChunker().chunk(_exe(lines), header_lines=20, footer_lines=20)
        cluster = next(c for c in chunks if c.chunk_type == "ERROR_CLUSTER")
        assert "big failure" in cluster.content


class TestChunkerRecovery:
    """RECOVERY chunk detection."""

    def test_recovery_after_error_cluster(self) -> None:
        """Recovery keyword after error cluster → RECOVERY chunk."""
        lines = (
            [_line(f"INFO: startup {i}",
                   ts=datetime(2025, 6, 13, 2, 0, i), unified=i)
             for i in range(25)]
            + [_line("ERROR: connection failed",
                     level="ERROR",
                     ts=datetime(2025, 6, 13, 2, 0, 35), unified=25)]
            # Retry line is 85 seconds after the error — outside the 30s
            # error cluster window, so it should be classified RECOVERY.
            + [_line("INFO: retrying connection...",
                     ts=datetime(2025, 6, 13, 2, 2, 0), unified=26)]
            + [_line(f"INFO: after {i}",
                     ts=datetime(2025, 6, 13, 2, 3, i), unified=27 + i)
               for i in range(25)]
        )
        chunks = LogChunker().chunk(_exe(lines), header_lines=20, footer_lines=20)
        types = {c.chunk_type for c in chunks}
        assert "RECOVERY" in types


class TestChunkerBody:
    """BODY chunk detection."""

    def test_body_chunk_present(self) -> None:
        """Mid-section lines become BODY chunks."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(60)]
        chunks = LogChunker().chunk(_exe(lines), header_lines=10, footer_lines=10)
        types = {c.chunk_type for c in chunks}
        assert "BODY" in types

    def test_empty_execution_returns_empty(self) -> None:
        """Empty execution returns empty list."""
        chunks = LogChunker().chunk(_exe([]))
        assert chunks == []

    def test_chunk_indices_sequential(self) -> None:
        """chunk_index values are sequential starting from 0."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(60)]
        chunks = LogChunker().chunk(_exe(lines))
        for expected, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected

    def test_all_lines_covered(self) -> None:
        """All lines appear in exactly one chunk."""
        lines = [_line(f"INFO: line {i}", unified=i) for i in range(50)]
        chunks = LogChunker().chunk(_exe(lines))
        total_in_chunks = sum(c.line_count for c in chunks)
        assert total_in_chunks == len(lines)

    def test_content_is_joined_text(self) -> None:
        """chunk.content joins raw text with newlines."""
        lines = [_line("INFO: line A", unified=0), _line("INFO: line B", unified=1)]
        # Force only HEADER to span both (header_lines=2 and only 2 lines).
        chunks = LogChunker().chunk(_exe(lines), header_lines=2, footer_lines=0)
        header = next(c for c in chunks if c.chunk_type == "HEADER")
        assert "line A" in header.content
        assert "line B" in header.content
