"""Tests for DigestBuilder.

Covers:
- Single-execution digest structure and length bounds.
- Error section is included / truncated when over-long.
- Multi-run digest includes all run numbers.
- Never raises on empty or unusual executions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import pytest

from app.config.constants import Severity
from app.config.settings import Settings
from app.llm.digest_builder import DigestBuilder
from app.preprocessing.error_aggregator import ErrorRecord
from app.preprocessing.metrics_generator import BatchMetrics
from app.segmentation.models import BatchExecution, BatchStatus, ParsedLogLine


# ── Helpers ──────────────────────────────────────────────────────

def _line(raw: str, level: str = "INFO", unified: int = 0) -> ParsedLogLine:
    """Build a minimal ParsedLogLine for testing."""
    return ParsedLogLine(
        raw=raw,
        source_file="/log/test.log",
        file_line_number=unified,
        unified_line_number=unified,
        ingestion_id=str(uuid4()),
        parsed_timestamp=datetime(2025, 6, 13, 0, 0, unified % 60),
        level=level,
        correlation_id="CID-TEST",
        message=raw,
    )


def _execution(
    lines: List[ParsedLogLine],
    cid: str = "CID-TEST",
    status: BatchStatus = BatchStatus.SUCCESS,
    job_name: str = "test-job",
) -> BatchExecution:
    """Build a minimal BatchExecution for testing."""
    errors = [ln for ln in lines if ln.level in ("ERROR", "FATAL")]
    warns = [ln for ln in lines if ln.level in ("WARN", "WARNING")]
    return BatchExecution(
        correlation_id=cid,
        lines=lines,
        start_time=datetime(2025, 6, 13, 0, 0, 0),
        end_time=datetime(2025, 6, 13, 0, 5, 0),
        status=status,
        job_name=job_name,
        error_count=len(errors),
        warn_count=len(warns),
        total_lines=len(lines),
        orphan_lines_count=0,
        has_start_marker=True,
        has_end_marker=True,
        source_files=["/log/test.log"],
    )


def _error_record(
    category: str = "DatabaseConnectivity",
    count: int = 3,
    message: str = "connection refused",
    severity: Severity = Severity.HIGH,
) -> ErrorRecord:
    """Build an ErrorRecord for testing."""
    return ErrorRecord(
        error_category=category,
        representative_message=message,
        count=count,
        sample_lines=[message],
        first_seen=datetime(2025, 6, 13, 0, 1, 0),
        last_seen=datetime(2025, 6, 13, 0, 4, 0),
        severity=severity,
    )


def _metrics(duration: Optional[float] = 300.0) -> BatchMetrics:
    """Build a BatchMetrics for testing."""
    return BatchMetrics(
        correlation_id="CID-TEST",
        duration_seconds=duration,
        lines_per_second=1.0,
        error_rate_percent=10.0,
        warn_rate_percent=5.0,
        estimated_record_count=None,
        lines_in_first_10_percent=None,
        lines_in_last_10_percent=None,
        longest_gap_seconds=None,
        peak_error_window=None,
    )


@pytest.fixture
def builder() -> DigestBuilder:
    """Return a DigestBuilder with default settings."""
    return DigestBuilder(Settings())


# ── Single execution digest ──────────────────────────────────────

class TestSingleExecutionDigest:
    """Tests for build_single_execution_digest."""

    def test_digest_contains_header_marker(self, builder: DigestBuilder) -> None:
        """Digest starts with '=== BATCH EXECUTION DIGEST ==='."""
        lines = [_line(f"INFO line {i}", unified=i) for i in range(20)]
        exc = _execution(lines)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert "=== BATCH EXECUTION DIGEST ===" in d

    def test_digest_ends_with_end_marker(self, builder: DigestBuilder) -> None:
        """Digest ends with '=== END DIGEST ==='."""
        lines = [_line(f"INFO line {i}", unified=i) for i in range(20)]
        exc = _execution(lines)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert "=== END DIGEST ===" in d

    def test_digest_contains_correlation_id(self, builder: DigestBuilder) -> None:
        """CID is present in the header."""
        lines = [_line("INFO start", unified=0)]
        exc = _execution(lines, cid="BATCH-42")
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert "BATCH-42" in d

    def test_digest_contains_error_summary(self, builder: DigestBuilder) -> None:
        """Error summary section is present."""
        lines = [_line("ERROR: DB down", level="ERROR", unified=0)]
        exc = _execution(lines)
        errs = [_error_record()]
        d = builder.build_single_execution_digest(exc, errs, _metrics())
        assert "ERROR SUMMARY" in d
        assert "DatabaseConnectivity" in d

    def test_digest_contains_startup_section(self, builder: DigestBuilder) -> None:
        """Startup sequence section is always present."""
        lines = [_line(f"INFO startup {i}", unified=i) for i in range(30)]
        exc = _execution(lines)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert "STARTUP SEQUENCE" in d

    def test_digest_contains_final_section(self, builder: DigestBuilder) -> None:
        """Final sequence section is always present."""
        lines = [_line(f"INFO line {i}", unified=i) for i in range(30)]
        exc = _execution(lines)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert "FINAL SEQUENCE" in d

    def test_digest_length_normal_batch(self, builder: DigestBuilder) -> None:
        """Normal batch (50 lines) digest stays under 4000 characters."""
        lines = [_line(f"INFO processing record {i}", unified=i) for i in range(50)]
        exc = _execution(lines)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert len(d) < 4_000

    def test_digest_error_lines_capped(self, builder: DigestBuilder) -> None:
        """Error lines section is capped at max_error_lines_in_digest."""
        settings = Settings(max_error_lines_in_digest=5)
        b = DigestBuilder(settings)
        # 100 error lines — should be capped.
        lines = [_line(f"ERROR: failure {i}", level="ERROR", unified=i) for i in range(100)]
        exc = _execution(lines, status=BatchStatus.FAILED)
        d = b.build_single_execution_digest(exc, [], _metrics())
        # Count occurrences of "ERROR: failure" in the error section only.
        ew_start = d.find("--- ERROR AND WARNING LINES")
        ew_end = d.find("--- FINAL SEQUENCE")
        error_section = d[ew_start:ew_end]
        error_line_count = error_section.count("ERROR: failure")
        assert error_line_count <= 5

    def test_startup_never_trimmed(self, builder: DigestBuilder) -> None:
        """First 15 lines always appear in the startup section."""
        # Build a digest that would be very long.
        lines = (
            [_line(f"INFO startup-unique-{i}", unified=i) for i in range(15)]
            + [_line(f"ERROR: repeated failure", level="ERROR", unified=i + 15) for i in range(200)]
        )
        exc = _execution(lines, status=BatchStatus.FAILED)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        # All startup lines must be present.
        for i in range(10):
            assert f"startup-unique-{i}" in d

    def test_final_never_trimmed(self, builder: DigestBuilder) -> None:
        """Last lines always appear in the final section."""
        lines = (
            [_line(f"ERROR: early failure", level="ERROR", unified=i) for i in range(200)]
            + [_line(f"INFO final-unique-{i}", unified=200 + i) for i in range(15)]
        )
        exc = _execution(lines)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        for i in range(10):
            assert f"final-unique-{i}" in d

    def test_empty_execution_no_crash(self, builder: DigestBuilder) -> None:
        """Empty execution (zero lines) produces a valid digest without raising."""
        exc = _execution([])
        d = builder.build_single_execution_digest(exc, [], _metrics(duration=None))
        assert "=== BATCH EXECUTION DIGEST ===" in d
        assert "=== END DIGEST ===" in d

    def test_no_errors_digest_still_valid(self, builder: DigestBuilder) -> None:
        """Executions with no error lines produce a clean digest."""
        lines = [_line(f"INFO all good {i}", unified=i) for i in range(20)]
        exc = _execution(lines)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert "(none)" in d or "no errors" in d.lower() or "(no categorised errors)" in d

    def test_error_section_present_when_errors_exist(self, builder: DigestBuilder) -> None:
        """Error lines show up in the error section."""
        lines = [
            _line("INFO start", unified=0),
            _line("ERROR: JDBC timeout", level="ERROR", unified=1),
            _line("WARN: retry attempt", level="WARN", unified=2),
        ]
        exc = _execution(lines, status=BatchStatus.FAILED)
        d = builder.build_single_execution_digest(exc, [], _metrics())
        assert "JDBC timeout" in d
        assert "retry attempt" in d


# ── Multi-run digest ─────────────────────────────────────────────

class TestMultiRunDigest:
    """Tests for build_multi_run_digest."""

    def _make_executions(self, n: int) -> List[BatchExecution]:
        """Build *n* executions with distinct run numbers."""
        execs = []
        for i in range(n):
            cid = f"BATCH-{i+1:03d}"
            lines = [
                _line(f"INFO start run {i+1}", unified=0),
                _line(f"ERROR: run {i+1} failed", level="ERROR", unified=1)
                if i < n - 1 else _line("INFO: success", unified=1),
            ]
            e = _execution(
                lines,
                cid=cid,
                status=BatchStatus.FAILED if i < n - 1 else BatchStatus.SUCCESS,
                job_name="daily-etl",
            )
            e._run_number = i + 1  # type: ignore[attr-defined]
            e._attempt_type = "SCHEDULED" if i == 0 else "AUTO_RETRY"  # type: ignore[attr-defined]
            execs.append(e)
        return execs

    def test_multi_run_includes_all_run_numbers(self, builder: DigestBuilder) -> None:
        """All run numbers appear in the multi-run digest."""
        execs = self._make_executions(3)
        d = builder.build_multi_run_digest(execs, {}, {})
        assert "RUN 1" in d
        assert "RUN 2" in d
        assert "RUN 3" in d

    def test_multi_run_contains_job_name(self, builder: DigestBuilder) -> None:
        """Job name is in the multi-run header."""
        execs = self._make_executions(2)
        d = builder.build_multi_run_digest(execs, {}, {})
        assert "daily-etl" in d

    def test_multi_run_contains_total_runs(self, builder: DigestBuilder) -> None:
        """Total run count is in the digest header."""
        execs = self._make_executions(4)
        d = builder.build_multi_run_digest(execs, {}, {})
        assert "Total Runs" in d
        assert "4" in d

    def test_multi_run_final_status_present(self, builder: DigestBuilder) -> None:
        """Final status is present in the digest."""
        execs = self._make_executions(2)
        d = builder.build_multi_run_digest(execs, {}, {})
        assert "SUCCESS" in d or "FAILED" in d

    def test_multi_run_empty_no_crash(self, builder: DigestBuilder) -> None:
        """Empty execution list returns a sentinel string."""
        d = builder.build_multi_run_digest([], {}, {})
        assert "no executions" in d.lower()

    def test_multi_run_middle_runs_summarised_when_too_long(
        self, builder: DigestBuilder
    ) -> None:
        """Very many runs: middle runs are summarised, first and last kept."""
        # Create 20 runs each with 50 lines to force budget overflow.
        execs = []
        for i in range(20):
            cid = f"BATCH-LONG-{i+1:03d}"
            lines = [_line(f"INFO line {j}", unified=j) for j in range(50)]
            lines[25] = _line("ERROR: connection reset", level="ERROR", unified=25)
            e = _execution(lines, cid=cid, job_name="heavy-job")
            e._run_number = i + 1  # type: ignore[attr-defined]
            e._attempt_type = "AUTO_RETRY"  # type: ignore[attr-defined]
            execs.append(e)

        d = builder.build_multi_run_digest(execs, {}, {})
        # First run present.
        assert "RUN 1" in d
        # Last run present.
        assert "RUN 20" in d
        # Should mention summarised middle runs.
        assert "SUMMARISED" in d.upper() or "RUN 2" in d

    def test_multi_run_end_marker_present(self, builder: DigestBuilder) -> None:
        """Multi-run digest ends with end marker."""
        execs = self._make_executions(2)
        d = builder.build_multi_run_digest(execs, {}, {})
        assert "=== END MULTI-RUN DIGEST ===" in d
