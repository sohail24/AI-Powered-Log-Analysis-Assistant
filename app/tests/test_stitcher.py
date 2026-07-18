"""Tests for the log stitcher module.

Covers file discovery, timestamp parsing (all 5 formats),
multi-file merge + sort, time-range filtering, non-timestamped line
positioning, empty directory handling, and unicode fallback.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from app.config.settings import Settings
from app.ingestion.models import RawLogLine, StitchedLog
from app.ingestion.stitcher import LogStitcher


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    """Return a default Settings instance for testing."""
    return Settings()


@pytest.fixture
def stitcher(settings: Settings) -> LogStitcher:
    """Return a LogStitcher backed by default settings."""
    return LogStitcher(settings)


@pytest.fixture
def sample_logs_dir() -> str:
    """Return absolute path to the sample_logs directory."""
    return str(
        Path(__file__).resolve().parent.parent / "data" / "sample_logs"
    )


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for creating test log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


def _write_file(
    directory: Path, name: str, lines: List[str], mtime: float | None = None
) -> Path:
    """Helper: write *lines* to *directory/name* and optionally set mtime."""
    fpath = directory / name
    fpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if mtime is not None:
        os.utime(fpath, (mtime, mtime))
    return fpath


# ── File Discovery Tests ────────────────────────────────────────


class TestDiscoverLogFiles:
    """LogStitcher.discover_log_files edge cases."""

    def test_discovers_sample_logs(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """All 3 sample log files are discovered."""
        files = stitcher.discover_log_files(sample_logs_dir)
        basenames = {Path(f).name for f in files}
        assert basenames == {"app.log", "app.log.1", "service.log"}

    def test_discovers_three_files(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """Exactly 3 files are returned."""
        files = stitcher.discover_log_files(sample_logs_dir)
        assert len(files) == 3

    def test_sorted_by_mtime(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Files are sorted oldest-first by modification time."""
        _write_file(tmp_log_dir, "old.log", ["line1"], mtime=1000.0)
        _write_file(tmp_log_dir, "new.log", ["line2"], mtime=2000.0)
        files = stitcher.discover_log_files(str(tmp_log_dir))
        assert Path(files[0]).name == "old.log"
        assert Path(files[1]).name == "new.log"

    def test_excludes_hidden_files(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Files starting with '.' are excluded."""
        _write_file(tmp_log_dir, ".hidden.log", ["secret"])
        _write_file(tmp_log_dir, "visible.log", ["visible"])
        files = stitcher.discover_log_files(str(tmp_log_dir))
        assert len(files) == 1
        assert Path(files[0]).name == "visible.log"

    def test_excludes_empty_files(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Zero-byte files are excluded."""
        empty = tmp_log_dir / "empty.log"
        empty.write_text("")
        _write_file(tmp_log_dir, "valid.log", ["data"])
        files = stitcher.discover_log_files(str(tmp_log_dir))
        assert len(files) == 1

    def test_includes_rotated_logs(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Rotated files like app.log.1, app.log.2 are included."""
        _write_file(tmp_log_dir, "app.log", ["line1"])
        _write_file(tmp_log_dir, "app.log.1", ["line2"])
        _write_file(tmp_log_dir, "app.log.2", ["line3"])
        files = stitcher.discover_log_files(str(tmp_log_dir))
        assert len(files) == 3

    def test_nonexistent_directory(self, stitcher: LogStitcher) -> None:
        """Non-existent directory returns empty list, no exception."""
        files = stitcher.discover_log_files("/no/such/path")
        assert files == []

    def test_empty_directory(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Empty directory returns empty list."""
        files = stitcher.discover_log_files(str(tmp_log_dir))
        assert files == []


# ── Timestamp Parsing Tests ─────────────────────────────────────


class TestParseTimestamp:
    """LogStitcher.parse_timestamp for all 5 formats."""

    def test_space_separated(self, stitcher: LogStitcher) -> None:
        """Parses '2025-06-13 02:00:01' format."""
        dt = stitcher.parse_timestamp(
            "2025-06-13 02:00:01 [INFO] test message"
        )
        assert dt == datetime(2025, 6, 13, 2, 0, 1)

    def test_iso8601_no_millis(self, stitcher: LogStitcher) -> None:
        """Parses '2025-06-13T02:00:01' format."""
        dt = stitcher.parse_timestamp(
            "2025-06-13T02:00:01 [DEBUG] test"
        )
        assert dt == datetime(2025, 6, 13, 2, 0, 1)

    def test_iso8601_with_millis(self, stitcher: LogStitcher) -> None:
        """Parses '2025-06-13T02:00:01.123Z' format."""
        dt = stitcher.parse_timestamp(
            "2025-06-13T02:00:01.123Z [INFO] test"
        )
        assert dt is not None
        assert dt.year == 2025
        assert dt.second == 1

    def test_apache_clf(self, stitcher: LogStitcher) -> None:
        """Parses '13/Jun/2025:02:00:01' format."""
        dt = stitcher.parse_timestamp(
            '10.0.0.1 - - [13/Jun/2025:02:00:01 +0000] "GET / HTTP/1.1" 200'
        )
        assert dt == datetime(2025, 6, 13, 2, 0, 1)

    def test_syslog(self, stitcher: LogStitcher) -> None:
        """Parses 'Jun 13 02:00:01' syslog format (assumes current year)."""
        dt = stitcher.parse_timestamp(
            "Jun 13 02:00:01 myhost sshd[1234]: Accepted publickey"
        )
        assert dt is not None
        assert dt.month == 6
        assert dt.day == 13
        assert dt.year == datetime.utcnow().year

    def test_no_timestamp(self, stitcher: LogStitcher) -> None:
        """Lines without any timestamp return None."""
        dt = stitcher.parse_timestamp("  just some random text")
        assert dt is None

    def test_malformed_timestamp(self, stitcher: LogStitcher) -> None:
        """Malformed timestamps return None, never raise."""
        dt = stitcher.parse_timestamp("9999-99-99 99:99:99 broken")
        # Either None or a parsed result — must not raise.
        assert dt is None or isinstance(dt, datetime)


# ── Read File Lines Tests ───────────────────────────────────────


class TestReadFileLines:
    """LogStitcher.read_file_lines edge cases."""

    def test_reads_sample_app_log(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """Reads app.log and returns RawLogLine objects."""
        filepath = os.path.join(sample_logs_dir, "app.log")
        lines = stitcher.read_file_lines(filepath)
        assert len(lines) > 0
        assert all(isinstance(ln, RawLogLine) for ln in lines)

    def test_file_line_numbers_are_1_indexed(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """file_line_number starts at 1."""
        filepath = os.path.join(sample_logs_dir, "app.log")
        lines = stitcher.read_file_lines(filepath)
        assert lines[0].file_line_number == 1

    def test_ingestion_ids_are_unique(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """Each line gets a unique ingestion_id."""
        filepath = os.path.join(sample_logs_dir, "app.log")
        lines = stitcher.read_file_lines(filepath)
        ids = [ln.ingestion_id for ln in lines]
        assert len(ids) == len(set(ids))

    def test_timestamps_parsed(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """Most lines in app.log have parsed timestamps."""
        filepath = os.path.join(sample_logs_dir, "app.log")
        lines = stitcher.read_file_lines(filepath)
        with_ts = [ln for ln in lines if ln.parsed_timestamp is not None]
        assert len(with_ts) > 0

    def test_nonexistent_file(self, stitcher: LogStitcher) -> None:
        """Non-existent file returns empty list."""
        lines = stitcher.read_file_lines("/no/such/file.log")
        assert lines == []

    def test_unicode_fallback(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Files with latin-1 encoding are read via fallback."""
        fpath = tmp_log_dir / "latin1.log"
        # Write bytes that are valid latin-1 but invalid UTF-8.
        fpath.write_bytes(
            b"2025-06-13 10:00:00 [INFO] caf\xe9 log entry\n"
            b"2025-06-13 10:00:01 [INFO] normal line\n"
        )
        lines = stitcher.read_file_lines(str(fpath))
        assert len(lines) == 2
        assert "caf" in lines[0].raw

    def test_empty_lines_skipped(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Completely empty lines are not included in output."""
        _write_file(
            tmp_log_dir,
            "gaps.log",
            [
                "2025-06-13 10:00:00 [INFO] first",
                "",
                "",
                "2025-06-13 10:00:01 [INFO] second",
            ],
        )
        lines = stitcher.read_file_lines(str(tmp_log_dir / "gaps.log"))
        assert len(lines) == 2


# ── Stitch Tests ────────────────────────────────────────────────


class TestStitch:
    """LogStitcher.stitch — full pipeline tests."""

    def test_stitch_sample_logs(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """Stitching sample_logs produces a valid StitchedLog."""
        result = stitcher.stitch(sample_logs_dir)
        assert isinstance(result, StitchedLog)
        assert result.total_lines > 0

    def test_source_files_populated(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """source_files lists all discovered files."""
        result = stitcher.stitch(sample_logs_dir)
        assert len(result.source_files) == 3

    def test_unified_line_numbers_sequential(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """unified_line_number is 1-indexed and sequential."""
        result = stitcher.stitch(sample_logs_dir)
        numbers = [ln.unified_line_number for ln in result.lines]
        assert numbers == list(range(1, result.total_lines + 1))

    def test_chronological_order(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """Timestamped lines appear in chronological order."""
        result = stitcher.stitch(sample_logs_dir)
        timestamps = [
            ln.parsed_timestamp
            for ln in result.lines
            if ln.parsed_timestamp is not None
        ]
        assert timestamps == sorted(timestamps)

    def test_start_and_end_time(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """start_time and end_time bracket the data."""
        result = stitcher.stitch(sample_logs_dir)
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.start_time <= result.end_time

    def test_lines_without_timestamp_counted(
        self, stitcher: LogStitcher, sample_logs_dir: str
    ) -> None:
        """lines_without_timestamp is accurate."""
        result = stitcher.stitch(sample_logs_dir)
        actual_none = sum(
            1 for ln in result.lines if ln.parsed_timestamp is None
        )
        assert result.lines_without_timestamp == actual_none

    def test_empty_directory(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Empty directory returns StitchedLog with no lines."""
        result = stitcher.stitch(str(tmp_log_dir))
        assert result.total_lines == 0
        assert result.lines == []
        assert result.start_time is None
        assert result.end_time is None

    def test_single_file(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Single file stitches correctly without merge."""
        _write_file(
            tmp_log_dir,
            "solo.log",
            [
                "2025-06-13 10:00:00 [INFO] alpha",
                "2025-06-13 10:00:01 [INFO] beta",
            ],
        )
        result = stitcher.stitch(str(tmp_log_dir))
        assert result.total_lines == 2
        assert result.lines[0].raw.endswith("alpha")
        assert result.lines[1].raw.endswith("beta")

    def test_cross_midnight_sort(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Lines spanning midnight sort correctly across day boundary."""
        _write_file(
            tmp_log_dir,
            "night.log",
            [
                "2025-06-12 23:59:59 [INFO] before midnight",
                "2025-06-13 00:00:01 [INFO] after midnight",
            ],
            mtime=1000.0,
        )
        _write_file(
            tmp_log_dir,
            "morning.log",
            [
                "2025-06-13 06:00:00 [INFO] morning line",
            ],
            mtime=2000.0,
        )
        result = stitcher.stitch(str(tmp_log_dir))
        assert result.lines[0].raw.endswith("before midnight")
        assert result.lines[1].raw.endswith("after midnight")
        assert result.lines[2].raw.endswith("morning line")

    def test_no_timestamps_fallback_order(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """All-no-timestamp files sort by mtime then line order."""
        _write_file(
            tmp_log_dir,
            "older.log",
            ["plain line A", "plain line B"],
            mtime=1000.0,
        )
        _write_file(
            tmp_log_dir,
            "newer.log",
            ["plain line C"],
            mtime=2000.0,
        )
        result = stitcher.stitch(str(tmp_log_dir))
        assert result.total_lines == 3
        raws = [ln.raw for ln in result.lines]
        assert raws == ["plain line A", "plain line B", "plain line C"]

    def test_non_timestamped_lines_follow_preceding(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Lines without timestamps stay after their preceding ts line."""
        _write_file(
            tmp_log_dir,
            "multi.log",
            [
                "2025-06-13 10:00:00 [ERROR] failure occurred",
                "  Stack trace: line 42",
                "  Caused by: NullPointer",
                "2025-06-13 10:00:05 [INFO] recovered",
            ],
        )
        result = stitcher.stitch(str(tmp_log_dir))
        assert result.total_lines == 4
        # Stack trace lines must appear between the ERROR and INFO
        assert result.lines[0].raw.endswith("failure occurred")
        assert "Stack trace" in result.lines[1].raw
        assert "Caused by" in result.lines[2].raw
        assert result.lines[3].raw.endswith("recovered")


# ── Time Range Filter Tests ─────────────────────────────────────


class TestTimeRangeFilter:
    """Stitch with start_time / end_time filtering."""

    def test_start_time_filter(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Lines before start_time are excluded."""
        _write_file(
            tmp_log_dir,
            "range.log",
            [
                "2025-06-13 08:00:00 [INFO] too early",
                "2025-06-13 10:00:00 [INFO] in range",
                "2025-06-13 12:00:00 [INFO] also in range",
            ],
        )
        result = stitcher.stitch(
            str(tmp_log_dir),
            start_time=datetime(2025, 6, 13, 9, 0, 0),
        )
        raws = [ln.raw for ln in result.lines]
        assert not any("too early" in r for r in raws)
        assert any("in range" in r for r in raws)

    def test_end_time_filter(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Lines after end_time are excluded."""
        _write_file(
            tmp_log_dir,
            "range.log",
            [
                "2025-06-13 08:00:00 [INFO] in range",
                "2025-06-13 10:00:00 [INFO] also in range",
                "2025-06-13 12:00:00 [INFO] too late",
            ],
        )
        result = stitcher.stitch(
            str(tmp_log_dir),
            end_time=datetime(2025, 6, 13, 11, 0, 0),
        )
        raws = [ln.raw for ln in result.lines]
        assert not any("too late" in r for r in raws)
        assert any("in range" in r for r in raws)

    def test_both_bounds(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Both start and end filter correctly."""
        _write_file(
            tmp_log_dir,
            "range.log",
            [
                "2025-06-13 06:00:00 [INFO] before",
                "2025-06-13 10:00:00 [INFO] inside",
                "2025-06-13 18:00:00 [INFO] after",
            ],
        )
        result = stitcher.stitch(
            str(tmp_log_dir),
            start_time=datetime(2025, 6, 13, 8, 0, 0),
            end_time=datetime(2025, 6, 13, 14, 0, 0),
        )
        assert result.total_lines == 1
        assert "inside" in result.lines[0].raw

    def test_non_timestamped_lines_kept_with_proxy(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Non-timestamped lines within range are kept via proxy timestamp."""
        _write_file(
            tmp_log_dir,
            "proxy.log",
            [
                "2025-06-13 10:00:00 [ERROR] main error",
                "  stack trace detail",
                "2025-06-13 10:00:05 [INFO] next event",
            ],
        )
        result = stitcher.stitch(
            str(tmp_log_dir),
            start_time=datetime(2025, 6, 13, 9, 0, 0),
            end_time=datetime(2025, 6, 13, 11, 0, 0),
        )
        # The stack trace line should be kept (proxy = 10:00:00 in range)
        raws = [ln.raw for ln in result.lines]
        assert any("stack trace" in r for r in raws)


# ── Save Unified Log Tests ──────────────────────────────────────


class TestSaveUnifiedLog:
    """LogStitcher.save_unified_log output."""

    def test_writes_file(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Unified log file is created on disk."""
        _write_file(
            tmp_log_dir,
            "a.log",
            ["2025-06-13 10:00:00 [INFO] hello"],
        )
        result = stitcher.stitch(str(tmp_log_dir))
        out_path = str(tmp_log_dir / "unified.log")
        returned = stitcher.save_unified_log(result, out_path)
        assert returned == out_path
        assert Path(out_path).exists()

    def test_tab_separated_format(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Each output line is tab-separated: number, basename, raw."""
        _write_file(
            tmp_log_dir,
            "fmt.log",
            ["2025-06-13 10:00:00 [INFO] test line"],
        )
        result = stitcher.stitch(str(tmp_log_dir))
        out_path = str(tmp_log_dir / "unified.log")
        stitcher.save_unified_log(result, out_path)
        content = Path(out_path).read_text(encoding="utf-8")
        parts = content.strip().split("\t")
        assert parts[0] == "1"
        assert parts[1] == "fmt.log"
        assert "test line" in parts[2]

    def test_line_count_matches(
        self, stitcher: LogStitcher, sample_logs_dir: str, tmp_log_dir: Path
    ) -> None:
        """Written file has same number of lines as StitchedLog."""
        result = stitcher.stitch(sample_logs_dir)
        out_path = str(tmp_log_dir / "unified.log")
        stitcher.save_unified_log(result, out_path)
        written_lines = Path(out_path).read_text(encoding="utf-8").strip().split("\n")
        assert len(written_lines) == result.total_lines


# ── Merge / Interleave Tests ───────────────────────────────────


class TestMergeInterleave:
    """Multi-file merge produces correct interleaving."""

    def test_interleaved_timestamps(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Lines from different files with overlapping timestamps interleave."""
        _write_file(
            tmp_log_dir,
            "file_a.log",
            [
                "2025-06-13 10:00:00 [INFO] A1",
                "2025-06-13 10:00:10 [INFO] A2",
            ],
            mtime=1000.0,
        )
        _write_file(
            tmp_log_dir,
            "file_b.log",
            [
                "2025-06-13 10:00:05 [INFO] B1",
                "2025-06-13 10:00:15 [INFO] B2",
            ],
            mtime=2000.0,
        )
        result = stitcher.stitch(str(tmp_log_dir))
        raws = [ln.raw for ln in result.lines]
        # Expected chronological order: A1 → B1 → A2 → B2
        assert "A1" in raws[0]
        assert "B1" in raws[1]
        assert "A2" in raws[2]
        assert "B2" in raws[3]

    def test_same_timestamp_across_files(
        self, stitcher: LogStitcher, tmp_log_dir: Path
    ) -> None:
        """Same timestamp in two files — both lines are preserved."""
        _write_file(
            tmp_log_dir,
            "x.log",
            ["2025-06-13 10:00:00 [INFO] from X"],
            mtime=1000.0,
        )
        _write_file(
            tmp_log_dir,
            "y.log",
            ["2025-06-13 10:00:00 [INFO] from Y"],
            mtime=2000.0,
        )
        result = stitcher.stitch(str(tmp_log_dir))
        assert result.total_lines == 2
        raws = {ln.raw for ln in result.lines}
        assert any("from X" in r for r in raws)
        assert any("from Y" in r for r in raws)
