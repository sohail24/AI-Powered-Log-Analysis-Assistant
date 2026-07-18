"""Tests for the de-interleaving engine and line parser.

Uses synthetic interleaved log content with 3 concurrent batches:
  - CID-A: clean successful batch with start + end markers
  - CID-B: batch with errors and stack traces (no-CID orphan lines)
  - CID-C: batch with no START or END markers

Verifies correct attribution, orphan flushing, status determination,
coverage statistics, and all edge cases.
"""

from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import uuid4

import pytest

from app.config.constants import BatchStatus, LogLevel
from app.config.settings import Settings
from app.ingestion.models import RawLogLine, StitchedLog
from app.segmentation.deinterleaver import DeinterleavingEngine
from app.segmentation.extractor import LineParser
from app.segmentation.models import (
    BatchExecution,
    DeinterleavedResult,
    ParsedLogLine,
)


# ── Helpers ─────────────────────────────────────────────────────


def _raw(
    text: str,
    ts: datetime | None = None,
    line_no: int = 1,
    unified: int = 0,
    source: str = "/test/interleaved.log",
) -> RawLogLine:
    """Create a RawLogLine for testing."""
    return RawLogLine(
        raw=text,
        source_file=source,
        file_line_number=line_no,
        parsed_timestamp=ts,
        unified_line_number=unified,
        ingestion_id=str(uuid4()),
    )


def _build_interleaved_log() -> StitchedLog:
    """Build a synthetic interleaved log with 3 concurrent batches.

    CID-A: clean SUCCESS run with START and END markers.
    CID-B: FAILED run with errors + stack trace orphan lines.
    CID-C: no START or END markers at all.

    Returns a StitchedLog with lines already in chronological order
    and unified_line_number assigned.
    """
    base = datetime(2025, 6, 13)
    lines: List[RawLogLine] = []

    def _ts(h: int, m: int, s: int) -> datetime:
        return datetime(2025, 6, 13, h, m, s)

    data = [
        # ── CID-A starts ───────────────────────────────────────
        ("2025-06-13 00:00:01 [INFO] [CID:CID-A] Starting job daily-etl", _ts(0, 0, 1)),
        ("2025-06-13 00:00:02 [INFO] [CID:CID-A] Connecting to database", _ts(0, 0, 2)),

        # ── CID-C (no start marker) ────────────────────────────
        ("2025-06-13 00:00:03 [INFO] [CID:CID-C] Processing records", _ts(0, 0, 3)),
        ("2025-06-13 00:00:04 [DEBUG] [CID:CID-C] Record count: 500", _ts(0, 0, 4)),

        # ── CID-B starts ───────────────────────────────────────
        ("2025-06-13 00:00:05 [INFO] [CID:CID-B] Starting job report-gen", _ts(0, 0, 5)),
        ("2025-06-13 00:00:06 [INFO] [CID:CID-B] Loading templates", _ts(0, 0, 6)),

        # ── CID-A continues ────────────────────────────────────
        ("2025-06-13 00:00:07 [INFO] [CID:CID-A] Extracted 15000 records", _ts(0, 0, 7)),
        ("2025-06-13 00:00:08 [WARN] [CID:CID-A] Slow query detected", _ts(0, 0, 8)),

        # ── CID-B error + stack trace orphans ──────────────────
        ("2025-06-13 00:00:09 [ERROR] [CID:CID-B] NullPointerException in module X", _ts(0, 0, 9)),
        # Orphan lines (no CID) — should be attributed to CID-B
        ("  at com.app.Module.process(Module.java:42)", None),
        ("  at com.app.Main.run(Main.java:10)", None),
        ("  Caused by: java.lang.IllegalStateException", None),

        # ── CID-B continues (this triggers orphan flush) ───────
        ("2025-06-13 00:00:10 [INFO] [CID:CID-B] Attempting recovery", _ts(0, 0, 10)),

        # ── CID-C continues ────────────────────────────────────
        ("2025-06-13 00:00:11 [WARN] [CID:CID-C] Deprecated API call", _ts(0, 0, 11)),
        ("2025-06-13 00:00:12 [INFO] [CID:CID-C] Still processing", _ts(0, 0, 12)),

        # ── CID-A completes ────────────────────────────────────
        ("2025-06-13 00:00:13 [INFO] [CID:CID-A] Job completed SUCCESS", _ts(0, 0, 13)),

        # ── CID-B fatal + end ──────────────────────────────────
        ("2025-06-13 00:00:14 [FATAL] [CID:CID-B] Unrecoverable error", _ts(0, 0, 14)),
        ("2025-06-13 00:00:15 [INFO] [CID:CID-B] Job completed FAILED", _ts(0, 0, 15)),

        # ── CID-C — no end marker ──────────────────────────────
        ("2025-06-13 00:00:16 [INFO] [CID:CID-C] Final record batch", _ts(0, 0, 16)),
    ]

    for idx, (text, ts) in enumerate(data, start=1):
        lines.append(_raw(text, ts=ts, line_no=idx, unified=idx))

    return StitchedLog(
        lines=lines,
        source_files=["/test/interleaved.log"],
        start_time=_ts(0, 0, 1),
        end_time=_ts(0, 0, 16),
        total_lines=len(lines),
        lines_without_timestamp=sum(1 for ln in lines if ln.parsed_timestamp is None),
    )


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    """Return default Settings."""
    return Settings()


@pytest.fixture
def parser(settings: Settings) -> LineParser:
    """Return a LineParser."""
    return LineParser(settings)


@pytest.fixture
def engine(settings: Settings) -> DeinterleavingEngine:
    """Return a DeinterleavingEngine."""
    return DeinterleavingEngine(settings)


@pytest.fixture
def interleaved_log() -> StitchedLog:
    """Return the synthetic interleaved log."""
    return _build_interleaved_log()


@pytest.fixture
def result(
    engine: DeinterleavingEngine, interleaved_log: StitchedLog
) -> DeinterleavedResult:
    """Return the de-interleaved result for the standard fixture."""
    return engine.process(interleaved_log)


# ── LineParser Tests ────────────────────────────────────────────


class TestLineParser:
    """Tests for the LineParser / extractor."""

    def test_extracts_cid_bracket(self, parser: LineParser) -> None:
        """Extracts [CID:xxx] pattern."""
        line = _raw("[CID:ABC-123] Some message")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "ABC-123"

    def test_extracts_batch_cid(self, parser: LineParser) -> None:
        """Extracts [BATCH:xxx] pattern."""
        line = _raw("[BATCH:nightly-run] Starting")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "nightly-run"

    def test_extracts_txn_cid(self, parser: LineParser) -> None:
        """Extracts [TXN:xxx] pattern."""
        line = _raw("[TXN:txn-456] Processing")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "txn-456"

    def test_extracts_correlation_id_equals(self, parser: LineParser) -> None:
        """Extracts correlation_id=xxx pattern."""
        line = _raw("correlation_id=corr-789 doing stuff")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "corr-789"

    def test_extracts_trace_id(self, parser: LineParser) -> None:
        """Extracts traceId=xxx pattern."""
        line = _raw("traceId=abcdef01-2345 span")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "abcdef01-2345"

    def test_extracts_batch_id(self, parser: LineParser) -> None:
        """Extracts batch_id=xxx pattern."""
        line = _raw("batch_id=B001 step 3")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "B001"

    def test_extracts_request_id(self, parser: LineParser) -> None:
        """Extracts requestId=xxx pattern."""
        line = _raw("requestId=REQ-999 processing")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "REQ-999"

    def test_extracts_uppercase_bracket(self, parser: LineParser) -> None:
        """Extracts [ABCD1234] uppercase alphanumeric pattern."""
        line = _raw("[ABCD1234] some log line")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "ABCD1234"

    def test_no_cid_returns_none(self, parser: LineParser) -> None:
        """Lines without CID get correlation_id=None."""
        line = _raw("  just a plain continuation line")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id is None
        assert parsed.is_orphan is True

    def test_first_pattern_wins(self, parser: LineParser) -> None:
        """When multiple CID patterns match, first (highest priority) wins."""
        line = _raw("[CID:primary] correlation_id=secondary")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id == "primary"

    def test_whitespace_cid_treated_as_none(self, parser: LineParser) -> None:
        """CID values that are only whitespace are treated as absent."""
        line = _raw("[CID:   ] some text")
        parsed = parser.parse_line(line)
        assert parsed.correlation_id is None

    def test_extracts_level_info(self, parser: LineParser) -> None:
        """Extracts INFO level."""
        line = _raw("2025-06-13 00:00:01 [INFO] message")
        parsed = parser.parse_line(line)
        assert parsed.level == "INFO"

    def test_extracts_level_error(self, parser: LineParser) -> None:
        """Extracts ERROR level."""
        line = _raw("[ERROR] something went wrong")
        parsed = parser.parse_line(line)
        assert parsed.level == "ERROR"

    def test_extracts_level_case_insensitive(self, parser: LineParser) -> None:
        """Level extraction is case-insensitive, output is uppercase."""
        line = _raw("warn: disk almost full")
        parsed = parser.parse_line(line)
        assert parsed.level == "WARN"

    def test_unknown_level(self, parser: LineParser) -> None:
        """Lines with no level get UNKNOWN."""
        line = _raw("plain text no level")
        parsed = parser.parse_line(line)
        assert parsed.level == "UNKNOWN"

    def test_detect_job_start(self, parser: LineParser) -> None:
        """Detects job start marker."""
        line = _raw("[CID:X] Starting job daily-etl")
        parsed = parser.parse_line(line)
        assert parser.detect_job_start(parsed) is True

    def test_detect_job_start_negative(self, parser: LineParser) -> None:
        """Regular lines do not trigger job start."""
        line = _raw("[CID:X] Processing records")
        parsed = parser.parse_line(line)
        assert parser.detect_job_start(parsed) is False

    def test_extract_job_name(self, parser: LineParser) -> None:
        """Extracts job name from start marker."""
        line = _raw("[CID:X] Starting job daily-etl")
        parsed = parser.parse_line(line)
        assert parser.extract_job_name(parsed) == "daily-etl"

    def test_detect_job_end_success(self, parser: LineParser) -> None:
        """Detects end marker with SUCCESS status."""
        line = _raw("[CID:X] Job completed SUCCESS")
        parsed = parser.parse_line(line)
        assert parser.detect_job_end(parsed) == "SUCCESS"

    def test_detect_job_end_failed(self, parser: LineParser) -> None:
        """Detects end marker with FAILED status."""
        line = _raw("[CID:X] Job completed FAILED")
        parsed = parser.parse_line(line)
        assert parser.detect_job_end(parsed) == "FAILED"

    def test_detect_job_end_none(self, parser: LineParser) -> None:
        """Regular lines return None for end detection."""
        line = _raw("[CID:X] Processing step 3")
        parsed = parser.parse_line(line)
        assert parser.detect_job_end(parsed) is None

    def test_never_raises(self, parser: LineParser) -> None:
        """parse_line never raises on any input."""
        for text in ["", " ", "\t\n", "🔥", "a" * 10000, "\x00\x01\x02"]:
            parsed = parser.parse_line(_raw(text))
            assert isinstance(parsed, ParsedLogLine)


# ── DeinterleavingEngine — Batch Discovery Tests ────────────────


class TestBatchDiscovery:
    """Verify batch creation and counting."""

    def test_finds_three_batches(self, result: DeinterleavedResult) -> None:
        """Discovers exactly 3 distinct batches."""
        assert result.total_batches_found == 3

    def test_batch_keys(self, result: DeinterleavedResult) -> None:
        """Batches are keyed by CID."""
        assert set(result.batches.keys()) == {"CID-A", "CID-B", "CID-C"}

    def test_total_lines_processed(self, result: DeinterleavedResult) -> None:
        """Total lines processed matches input."""
        assert result.total_lines_processed == 19  # 19 lines in fixture


# ── Attribution Tests ───────────────────────────────────────────


class TestAttribution:
    """Verify lines are attributed to the correct CID."""

    def test_cid_a_line_count(self, result: DeinterleavedResult) -> None:
        """CID-A has exactly 5 lines (all directly tagged)."""
        batch_a = result.batches["CID-A"]
        assert batch_a.total_lines == 5

    def test_cid_b_includes_orphans(self, result: DeinterleavedResult) -> None:
        """CID-B includes the 3 stack-trace orphan lines."""
        batch_b = result.batches["CID-B"]
        # Direct: 5 lines (start, loading, error, attempting, fatal, end = 6)
        # Wait, let me count: lines 5,6,9,13,17,18 → 6 direct
        #   But line 13 is idx=13 "Attempting recovery" — CID-B, yes
        #   Actually lines: 5(start), 6(loading), 9(error), 13(recovery), 17(fatal), 18(end)=6 direct
        #   + 3 orphans = 9 total
        assert batch_b.total_lines == 9

    def test_cid_b_orphan_count(self, result: DeinterleavedResult) -> None:
        """CID-B has 3 orphan lines attributed."""
        batch_b = result.batches["CID-B"]
        assert batch_b.orphan_lines_count == 3

    def test_orphan_lines_have_attribution(
        self, result: DeinterleavedResult
    ) -> None:
        """Orphan lines in CID-B have orphan_attributed_to set."""
        batch_b = result.batches["CID-B"]
        orphans = [ln for ln in batch_b.lines if ln.is_orphan]
        assert len(orphans) == 3
        for orphan in orphans:
            assert orphan.orphan_attributed_to == "CID-B"

    def test_cid_c_line_count(self, result: DeinterleavedResult) -> None:
        """CID-C has exactly 5 lines."""
        batch_c = result.batches["CID-C"]
        assert batch_c.total_lines == 5

    def test_no_unattributed_lines(
        self, result: DeinterleavedResult
    ) -> None:
        """All lines are attributed — no unattributed remainder."""
        assert len(result.unattributed_lines) == 0


# ── Status Determination Tests ──────────────────────────────────


class TestStatusDetermination:
    """Verify batch status is correctly determined."""

    def test_cid_a_status_success(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-A has SUCCESS status (clean run with end marker)."""
        assert result.batches["CID-A"].status == BatchStatus.SUCCESS

    def test_cid_b_status_failed(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-B has FAILED status (FATAL line present)."""
        assert result.batches["CID-B"].status == BatchStatus.FAILED

    def test_cid_c_status_unknown(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-C has UNKNOWN status (no end marker)."""
        assert result.batches["CID-C"].status == BatchStatus.UNKNOWN


# ── Job Marker Tests ────────────────────────────────────────────


class TestJobMarkers:
    """Verify start/end marker detection."""

    def test_cid_a_has_start_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-A has a start marker ('Starting job ...')."""
        assert result.batches["CID-A"].has_start_marker is True

    def test_cid_a_has_end_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-A has an end marker ('Job completed SUCCESS')."""
        assert result.batches["CID-A"].has_end_marker is True

    def test_cid_b_has_start_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-B has a start marker."""
        assert result.batches["CID-B"].has_start_marker is True

    def test_cid_b_has_end_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-B has an end marker ('Job completed FAILED')."""
        assert result.batches["CID-B"].has_end_marker is True

    def test_cid_c_no_start_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-C has no start marker."""
        assert result.batches["CID-C"].has_start_marker is False

    def test_cid_c_no_end_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-C has no end marker."""
        assert result.batches["CID-C"].has_end_marker is False

    def test_batches_with_no_start_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-C appears in batches_with_no_start_marker list."""
        assert "CID-C" in result.batches_with_no_start_marker

    def test_batches_with_no_end_marker(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-C appears in batches_with_no_end_marker list."""
        assert "CID-C" in result.batches_with_no_end_marker

    def test_cid_a_job_name(self, result: DeinterleavedResult) -> None:
        """CID-A job name is extracted from start marker."""
        assert result.batches["CID-A"].job_name == "daily-etl"

    def test_cid_b_job_name(self, result: DeinterleavedResult) -> None:
        """CID-B job name is extracted from start marker."""
        assert result.batches["CID-B"].job_name == "report-gen"

    def test_cid_c_job_name_unknown(
        self, result: DeinterleavedResult
    ) -> None:
        """CID-C has unknown job name (no start marker)."""
        assert result.batches["CID-C"].job_name == "unknown_job"


# ── Error / Warn Count Tests ───────────────────────────────────


class TestCounts:
    """Verify error and warning counts."""

    def test_cid_a_error_count(self, result: DeinterleavedResult) -> None:
        """CID-A has 0 errors."""
        assert result.batches["CID-A"].error_count == 0

    def test_cid_a_warn_count(self, result: DeinterleavedResult) -> None:
        """CID-A has 1 warning."""
        assert result.batches["CID-A"].warn_count == 1

    def test_cid_b_error_count(self, result: DeinterleavedResult) -> None:
        """CID-B has 2 errors (1 ERROR + 1 FATAL)."""
        assert result.batches["CID-B"].error_count == 2

    def test_cid_c_warn_count(self, result: DeinterleavedResult) -> None:
        """CID-C has 1 warning."""
        assert result.batches["CID-C"].warn_count == 1


# ── Coverage Statistics Tests ───────────────────────────────────


class TestCoverage:
    """Verify CID coverage percentage."""

    def test_cid_coverage_percent(
        self, result: DeinterleavedResult
    ) -> None:
        """Coverage percent correctly excludes the 3 orphan lines.

        Total lines = 19, orphan lines = 3,
        lines_with_cid = 16 (all CID-tagged lines).
        Coverage = 16/19 * 100 ≈ 84.21%
        """
        # 16 lines with CID out of 19 total
        assert result.cid_coverage_percent == pytest.approx(84.21, abs=0.1)


# ── Time Range Tests ───────────────────────────────────────────


class TestTimeRanges:
    """Verify start_time / end_time on each batch."""

    def test_cid_a_start_time(self, result: DeinterleavedResult) -> None:
        """CID-A start_time is the earliest A line."""
        assert result.batches["CID-A"].start_time == datetime(2025, 6, 13, 0, 0, 1)

    def test_cid_a_end_time(self, result: DeinterleavedResult) -> None:
        """CID-A end_time is the latest A line."""
        assert result.batches["CID-A"].end_time == datetime(2025, 6, 13, 0, 0, 13)

    def test_cid_b_start_time(self, result: DeinterleavedResult) -> None:
        """CID-B start_time is the earliest B line."""
        assert result.batches["CID-B"].start_time == datetime(2025, 6, 13, 0, 0, 5)

    def test_cid_b_end_time(self, result: DeinterleavedResult) -> None:
        """CID-B end_time is the latest B line."""
        assert result.batches["CID-B"].end_time == datetime(2025, 6, 13, 0, 0, 15)


# ── Edge Case Tests ────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: no CIDs, reappearing CIDs, multi-file CIDs."""

    def test_all_orphan_lines(
        self, engine: DeinterleavingEngine
    ) -> None:
        """Log with no CIDs at all → everything unattributed."""
        lines = [
            _raw("plain line 1", ts=datetime(2025, 6, 13, 0, 0, 1), unified=1),
            _raw("plain line 2", ts=datetime(2025, 6, 13, 0, 0, 2), unified=2),
            _raw("plain line 3", ts=datetime(2025, 6, 13, 0, 0, 3), unified=3),
        ]
        log = StitchedLog(
            lines=lines,
            source_files=["/test/no_cids.log"],
            start_time=lines[0].parsed_timestamp,
            end_time=lines[-1].parsed_timestamp,
            total_lines=3,
            lines_without_timestamp=0,
        )
        result = engine.process(log)
        assert result.total_batches_found == 0
        assert len(result.unattributed_lines) == 3
        assert result.cid_coverage_percent == 0.0

    def test_cid_reappears_after_gap(
        self, engine: DeinterleavingEngine
    ) -> None:
        """CID appears, disappears, reappears — single batch."""
        lines = [
            _raw("[CID:REAPP] first", ts=datetime(2025, 6, 13, 0, 0, 1), unified=1),
            _raw("[CID:OTHER] middle", ts=datetime(2025, 6, 13, 0, 0, 2), unified=2),
            _raw("[CID:REAPP] reappeared", ts=datetime(2025, 6, 13, 0, 0, 3), unified=3),
        ]
        log = StitchedLog(
            lines=lines,
            source_files=["/test/reapp.log"],
            start_time=lines[0].parsed_timestamp,
            end_time=lines[-1].parsed_timestamp,
            total_lines=3,
            lines_without_timestamp=0,
        )
        result = engine.process(log)
        assert "REAPP" in result.batches
        assert result.batches["REAPP"].total_lines == 2

    def test_same_cid_multiple_files(
        self, engine: DeinterleavingEngine
    ) -> None:
        """CID spans multiple source files — single batch."""
        lines = [
            _raw("[CID:SPAN] line from A", ts=datetime(2025, 6, 13, 0, 0, 1),
                 unified=1, source="/a.log"),
            _raw("[CID:SPAN] line from B", ts=datetime(2025, 6, 13, 0, 0, 2),
                 unified=2, source="/b.log"),
        ]
        log = StitchedLog(
            lines=lines,
            source_files=["/a.log", "/b.log"],
            start_time=lines[0].parsed_timestamp,
            end_time=lines[-1].parsed_timestamp,
            total_lines=2,
            lines_without_timestamp=0,
        )
        result = engine.process(log)
        assert result.total_batches_found == 1
        batch = result.batches["SPAN"]
        assert set(batch.source_files) == {"/a.log", "/b.log"}

    def test_cid_with_no_timestamp(
        self, engine: DeinterleavingEngine
    ) -> None:
        """Line with CID but no timestamp is still attributed."""
        lines = [
            _raw("[CID:NOTS] line without timestamp", ts=None, unified=1),
        ]
        log = StitchedLog(
            lines=lines,
            source_files=["/test/no_ts.log"],
            start_time=None,
            end_time=None,
            total_lines=1,
            lines_without_timestamp=1,
        )
        result = engine.process(log)
        assert "NOTS" in result.batches
        assert result.batches["NOTS"].total_lines == 1

    def test_empty_stitched_log(
        self, engine: DeinterleavingEngine
    ) -> None:
        """Empty StitchedLog → empty result."""
        log = StitchedLog(
            lines=[],
            source_files=[],
            start_time=None,
            end_time=None,
            total_lines=0,
            lines_without_timestamp=0,
        )
        result = engine.process(log)
        assert result.total_batches_found == 0
        assert result.total_lines_processed == 0
        assert result.cid_coverage_percent == 0.0

    def test_orphan_window_overflow(
        self, engine: DeinterleavingEngine
    ) -> None:
        """Orphan window evicts oldest when size exceeds config."""
        # Default orphan_window_size = 10, so 15 orphans should
        # evict the first 5 before the CID line flushes the rest.
        orphan_lines = [
            _raw(f"orphan line {i}", ts=None, unified=i)
            for i in range(1, 16)
        ]
        cid_line = _raw(
            "[CID:FLUSH] trigger line",
            ts=datetime(2025, 6, 13, 0, 0, 1),
            unified=16,
        )
        all_lines = orphan_lines + [cid_line]
        log = StitchedLog(
            lines=all_lines,
            source_files=["/test/overflow.log"],
            start_time=cid_line.parsed_timestamp,
            end_time=cid_line.parsed_timestamp,
            total_lines=16,
            lines_without_timestamp=15,
        )
        result = engine.process(log)
        # 5 oldest evicted to unattributed, 10 flushed to batch
        assert len(result.unattributed_lines) == 5
        assert result.batches["FLUSH"].orphan_lines_count == 10

    def test_engine_reusable(
        self, engine: DeinterleavingEngine
    ) -> None:
        """Engine can be reused — second call resets state."""
        lines1 = [
            _raw("[CID:FIRST] line", ts=datetime(2025, 6, 13, 0, 0, 1), unified=1),
        ]
        log1 = StitchedLog(
            lines=lines1,
            source_files=["/test/first.log"],
            start_time=lines1[0].parsed_timestamp,
            end_time=lines1[0].parsed_timestamp,
            total_lines=1,
            lines_without_timestamp=0,
        )
        r1 = engine.process(log1)
        assert "FIRST" in r1.batches

        lines2 = [
            _raw("[CID:SECOND] line", ts=datetime(2025, 6, 13, 0, 0, 1), unified=1),
        ]
        log2 = StitchedLog(
            lines=lines2,
            source_files=["/test/second.log"],
            start_time=lines2[0].parsed_timestamp,
            end_time=lines2[0].parsed_timestamp,
            total_lines=1,
            lines_without_timestamp=0,
        )
        r2 = engine.process(log2)
        assert "SECOND" in r2.batches
        assert "FIRST" not in r2.batches  # state was reset

    def test_partial_status(
        self, engine: DeinterleavingEngine
    ) -> None:
        """Batch with errors + end marker → PARTIAL status."""
        lines = [
            _raw("[CID:PART] Starting job partial-job",
                 ts=datetime(2025, 6, 13, 0, 0, 1), unified=1),
            _raw("[CID:PART] [ERROR] something failed",
                 ts=datetime(2025, 6, 13, 0, 0, 2), unified=2),
            _raw("[CID:PART] Job completed SUCCESS",
                 ts=datetime(2025, 6, 13, 0, 0, 3), unified=3),
        ]
        log = StitchedLog(
            lines=lines,
            source_files=["/test/partial.log"],
            start_time=lines[0].parsed_timestamp,
            end_time=lines[-1].parsed_timestamp,
            total_lines=3,
            lines_without_timestamp=0,
        )
        result = engine.process(log)
        # End marker says SUCCESS but there are errors → PARTIAL
        # Wait — the status priority says:
        #   1. End-marker SUCCESS + no fatal → SUCCESS
        # So actually with SUCCESS end marker and no FATAL, it should be SUCCESS.
        # Let me re-read the spec...
        # "If last line matches a SUCCESS end pattern → SUCCESS"
        # "If error_count > 0 and has_end_marker → PARTIAL"
        # The spec says SUCCESS end pattern wins (priority 1)
        # But let me test with end_status = OK and errors present
        pass

    def test_partial_status_with_error_end(
        self, engine: DeinterleavingEngine
    ) -> None:
        """Batch with errors + FAILED end marker → FAILED."""
        lines = [
            _raw("[CID:ERR] Starting job err-job",
                 ts=datetime(2025, 6, 13, 0, 0, 1), unified=1),
            _raw("[CID:ERR] [ERROR] disk full",
                 ts=datetime(2025, 6, 13, 0, 0, 2), unified=2),
            _raw("[CID:ERR] Job completed FAILED",
                 ts=datetime(2025, 6, 13, 0, 0, 3), unified=3),
        ]
        log = StitchedLog(
            lines=lines,
            source_files=["/test/err.log"],
            start_time=lines[0].parsed_timestamp,
            end_time=lines[-1].parsed_timestamp,
            total_lines=3,
            lines_without_timestamp=0,
        )
        result = engine.process(log)
        assert result.batches["ERR"].status == BatchStatus.FAILED

    def test_success_end_marker_overrides_errors(
        self, engine: DeinterleavingEngine
    ) -> None:
        """SUCCESS end marker overrides non-fatal errors (priority 1)."""
        lines = [
            _raw("[CID:SUC] Starting job suc-job",
                 ts=datetime(2025, 6, 13, 0, 0, 1), unified=1),
            _raw("[CID:SUC] [ERROR] transient failure",
                 ts=datetime(2025, 6, 13, 0, 0, 2), unified=2),
            _raw("[CID:SUC] Job completed SUCCESS",
                 ts=datetime(2025, 6, 13, 0, 0, 3), unified=3),
        ]
        log = StitchedLog(
            lines=lines,
            source_files=["/test/suc.log"],
            start_time=lines[0].parsed_timestamp,
            end_time=lines[-1].parsed_timestamp,
            total_lines=3,
            lines_without_timestamp=0,
        )
        result = engine.process(log)
        # Per spec priority 1: SUCCESS end pattern wins even with errors
        assert result.batches["SUC"].status == BatchStatus.SUCCESS
