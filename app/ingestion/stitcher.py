"""Log stitcher — merges multiple log files into a unified timeline.

Scans a directory for log files, reads them, parses timestamps,
and produces a single chronologically sorted StitchedLog. Lines
without timestamps are anchored to the nearest preceding timestamped
line from the same source file.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from app.config.log_patterns import TIMESTAMP_PATTERNS
from app.config.settings import Settings
from app.ingestion.models import RawLogLine, StitchedLog

logger = logging.getLogger("stitcher")

# File extensions considered as log files.
_LOG_EXTENSIONS = {".log", ".txt"}

# Timestamp format strings mapped from regex pattern order.
# Aligned 1-to-1 with TIMESTAMP_PATTERNS in log_patterns.py.
_TIMESTAMP_FORMATS: List[List[str]] = [
    # Pattern 0: ISO-8601 with millis ± Z  (2025-06-13T02:00:01.123Z)
    ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f"],
    # Pattern 1: ISO-8601 no millis        (2025-06-13T02:00:01)
    ["%Y-%m-%dT%H:%M:%S"],
    # Pattern 2: Space-separated           (2025-06-13 02:00:01)
    ["%Y-%m-%d %H:%M:%S"],
    # Pattern 3: Apache / CLF              (13/Jun/2025:02:00:01)
    ["%d/%b/%Y:%H:%M:%S"],
    # Pattern 4: Syslog (no year)          (Jun 13 02:00:01)
    ["%b %d %H:%M:%S"],
]


class LogStitcher:
    """Merges multiple log files from a directory into one timeline.

    Usage::

        stitcher = LogStitcher(settings)
        result = stitcher.stitch("/var/log/myapp")
        stitcher.save_unified_log(result, "/tmp/unified.log")
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the stitcher with application settings.

        Args:
            settings: Application-wide Settings instance.
        """
        self._settings = settings

    # ── Public API ──────────────────────────────────────────────

    def discover_log_files(self, directory: str) -> List[str]:
        """Scan *directory* for log files, sorted oldest-first by mtime.

        Includes ``*.log``, ``*.log.*`` (rotated), and ``*.txt`` files.
        Excludes hidden files, directories, and empty files.

        Args:
            directory: Path to the directory to scan.

        Returns:
            Sorted list of absolute file paths (oldest mtime first).
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning("Directory does not exist: %s", directory)
            return []

        found: List[str] = []
        try:
            for entry in dir_path.iterdir():
                if not entry.is_file():
                    continue
                if entry.name.startswith("."):
                    continue
                if entry.stat().st_size == 0:
                    continue
                if self._is_log_file(entry):
                    found.append(str(entry.resolve()))
        except PermissionError:
            logger.error("Permission denied reading directory: %s", directory)
            return []

        # Sort by modification time ascending (oldest first).
        found.sort(key=lambda p: os.path.getmtime(p))
        logger.info("Discovered %d log file(s) in %s", len(found), directory)
        return found

    def parse_timestamp(self, line: str) -> Optional[datetime]:
        """Extract the first matching timestamp from a log line.

        Tries each pattern in ``TIMESTAMP_PATTERNS`` (from config)
        against *line*. The first successful regex + strptime match is
        returned as a timezone-naive ``datetime`` (UTC assumed).

        Args:
            line: A single raw log line.

        Returns:
            Parsed datetime or ``None`` if no pattern matches.
        """
        for pattern, fmt_candidates in zip(TIMESTAMP_PATTERNS, _TIMESTAMP_FORMATS):
            m = pattern.search(line)
            if m is None:
                continue
            raw_ts = m.group("ts")
            for fmt in fmt_candidates:
                try:
                    dt = datetime.strptime(raw_ts, fmt)
                    # Syslog has no year — assume current year.
                    if dt.year == 1900:
                        dt = dt.replace(year=datetime.utcnow().year)
                    return dt
                except ValueError:
                    continue
        return None

    def read_file_lines(self, filepath: str) -> List[RawLogLine]:
        """Read all lines from a single log file.

        Each non-empty line is wrapped in a ``RawLogLine`` with a
        parsed timestamp and unique ingestion ID. On
        ``UnicodeDecodeError`` the file is re-read with ``latin-1``.

        Args:
            filepath: Absolute path to the file to read.

        Returns:
            List of ``RawLogLine`` (may be empty on I/O errors).
        """
        raw_lines = self._safe_read(filepath)
        if raw_lines is None:
            return []

        result: List[RawLogLine] = []
        for idx, line in enumerate(raw_lines, start=1):
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue
            result.append(
                RawLogLine(
                    raw=stripped,
                    source_file=str(Path(filepath).resolve()),
                    file_line_number=idx,
                    parsed_timestamp=self.parse_timestamp(stripped),
                    unified_line_number=0,
                    ingestion_id=str(uuid4()),
                )
            )
        return result

    def stitch(
        self,
        directory: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> StitchedLog:
        """Orchestrate the full stitching pipeline.

        1. Discover log files in *directory*.
        2. Read lines from each file.
        3. Merge and sort chronologically (timestamp-less lines are
           anchored after the last timestamped line from the same file).
        4. Optionally filter to ``[start_time, end_time]``.
        5. Assign sequential ``unified_line_number``.

        Args:
            directory: Path to the log directory.
            start_time: Optional inclusive lower bound for filtering.
            end_time: Optional inclusive upper bound for filtering.

        Returns:
            A ``StitchedLog`` with the merged, sorted, filtered lines.
        """
        files = self.discover_log_files(directory)
        if not files:
            logger.info("No log files found — returning empty StitchedLog")
            return self._empty_stitched_log(files)

        # Gather lines per file, preserving file mtime for fallback sort.
        all_lines: List[RawLogLine] = []
        file_mtimes: Dict[str, float] = {}
        for fpath in files:
            lines = self.read_file_lines(fpath)
            resolved = str(Path(fpath).resolve())
            file_mtimes[resolved] = os.path.getmtime(fpath)
            all_lines.extend(lines)

        logger.info("Total lines before filtering: %d", len(all_lines))

        # Sort: merge timestamped + non-timestamped lines correctly.
        sorted_lines = self._sort_lines(all_lines, file_mtimes)

        # Time-range filter.
        if start_time or end_time:
            sorted_lines = self._apply_time_filter(
                sorted_lines, start_time, end_time
            )
            logger.info("Total lines after filtering: %d", len(sorted_lines))

        # Assign unified line numbers.
        for idx, line in enumerate(sorted_lines, start=1):
            line.unified_line_number = idx

        no_ts_count = sum(
            1 for ln in sorted_lines if ln.parsed_timestamp is None
        )
        logger.info("Lines without timestamp: %d", no_ts_count)

        timestamps = [
            ln.parsed_timestamp
            for ln in sorted_lines
            if ln.parsed_timestamp is not None
        ]

        return StitchedLog(
            lines=sorted_lines,
            source_files=files,
            start_time=min(timestamps) if timestamps else None,
            end_time=max(timestamps) if timestamps else None,
            total_lines=len(sorted_lines),
            lines_without_timestamp=no_ts_count,
        )

    def save_unified_log(
        self, stitched: StitchedLog, output_path: str
    ) -> str:
        """Write the unified log to disk as tab-separated plain text.

        Format per line::

            {unified_line_number}\\t{source_file_basename}\\t{raw}

        Args:
            stitched: A ``StitchedLog`` produced by :meth:`stitch`.
            output_path: Destination file path.

        Returns:
            The *output_path* that was written.
        """
        out = Path(output_path)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                for line in stitched.lines:
                    basename = Path(line.source_file).name
                    fh.write(
                        f"{line.unified_line_number}\t{basename}\t{line.raw}\n"
                    )
            logger.info("Unified log written to %s (%d lines)",
                        output_path, stitched.total_lines)
        except OSError as exc:
            logger.error("Failed to write unified log to %s: %s",
                         output_path, exc)
        return output_path

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _is_log_file(entry: Path) -> bool:
        """Return True if *entry* looks like a log file.

        Matches ``*.log``, ``*.log.N`` (rotated), and ``*.txt``.
        """
        name = entry.name
        suffix = entry.suffix.lower()

        # Exact extension match.
        if suffix in _LOG_EXTENSIONS:
            return True

        # Rotated logs: app.log.1, app.log.2, etc.
        # Check if any intermediate suffix is .log
        if ".log." in name.lower():
            return True

        return False

    def _safe_read(self, filepath: str) -> Optional[List[str]]:
        """Read file lines with UTF-8, falling back to latin-1.

        Returns ``None`` on ``FileNotFoundError`` or ``PermissionError``.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                return fh.readlines()
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed for %s — retrying with latin-1",
                filepath,
            )
            try:
                with open(filepath, "r", encoding="latin-1") as fh:
                    return fh.readlines()
            except (FileNotFoundError, PermissionError) as exc:
                logger.error("Cannot read %s: %s", filepath, exc)
                return None
        except FileNotFoundError:
            logger.error("File not found: %s", filepath)
            return None
        except PermissionError:
            logger.error("Permission denied: %s", filepath)
            return None

    @staticmethod
    def _sort_lines(
        lines: List[RawLogLine],
        file_mtimes: Dict[str, float],
    ) -> List[RawLogLine]:
        """Sort lines chronologically, anchoring timestamp-less lines.

        Strategy:
        - Lines WITH a timestamp are sorted by that timestamp.
        - Lines WITHOUT a timestamp retain their relative order within
          their source file and are inserted immediately after the last
          timestamped line from the same file.
        - If ALL lines lack timestamps, fall back to file mtime then
          file line number.

        Args:
            lines: Unsorted list of RawLogLine from all files.
            file_mtimes: Map of resolved-file-path → mtime (float).

        Returns:
            A new list of RawLogLine in chronological order.
        """
        has_any_ts = any(ln.parsed_timestamp is not None for ln in lines)

        if not has_any_ts:
            # Fallback: sort by file modification time, then line order.
            return sorted(
                lines,
                key=lambda ln: (
                    file_mtimes.get(ln.source_file, 0.0),
                    ln.file_line_number,
                ),
            )

        # Build per-file groups: list of (line, last_known_ts) tuples.
        # last_known_ts is the most recent parsed_timestamp seen so far
        # within the same source file while iterating in file order.
        per_file: Dict[str, List[RawLogLine]] = {}
        for ln in lines:
            per_file.setdefault(ln.source_file, []).append(ln)

        # Within each file, sort by file_line_number to restore order
        # (should already be ordered, but be defensive).
        for flines in per_file.values():
            flines.sort(key=lambda ln: ln.file_line_number)

        # Assign a sort key to each line.
        # Timestamped lines: sort by their own timestamp.
        # Non-timestamped lines: use the last preceding timestamp from
        # the same file, with a sub-key to keep them after that line.
        annotated: List[Tuple[datetime, int, float, int, RawLogLine]] = []

        for source_file, flines in per_file.items():
            last_ts: Optional[datetime] = None
            non_ts_seq = 0  # counter for non-ts lines after a ts line
            for ln in flines:
                if ln.parsed_timestamp is not None:
                    last_ts = ln.parsed_timestamp
                    non_ts_seq = 0
                    #                 (timestamp,   sub-order, mtime,                                     file_line,            line)
                    annotated.append((last_ts,      0,         file_mtimes.get(source_file, 0.0),          ln.file_line_number,  ln))
                else:
                    non_ts_seq += 1
                    if last_ts is not None:
                        annotated.append((last_ts, non_ts_seq, file_mtimes.get(source_file, 0.0), ln.file_line_number, ln))
                    else:
                        # No ts seen yet in this file — use file mtime
                        # as epoch so they appear at the file-level position.
                        fallback = datetime.utcfromtimestamp(
                            file_mtimes.get(source_file, 0.0)
                        )
                        annotated.append((fallback, non_ts_seq, file_mtimes.get(source_file, 0.0), ln.file_line_number, ln))

        annotated.sort(key=lambda t: (t[0], t[2], t[1], t[3]))
        return [t[4] for t in annotated]

    @staticmethod
    def _apply_time_filter(
        lines: List[RawLogLine],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> List[RawLogLine]:
        """Filter lines to ``[start_time, end_time]`` inclusive.

        Lines with ``parsed_timestamp is None`` are kept if the
        previous line's timestamp (as proxy) falls within range.

        Args:
            lines: Sorted list of RawLogLine.
            start_time: Inclusive lower bound (or None for unbounded).
            end_time: Inclusive upper bound (or None for unbounded).

        Returns:
            Filtered list.
        """
        result: List[RawLogLine] = []
        last_ts: Optional[datetime] = None

        for ln in lines:
            ts = ln.parsed_timestamp if ln.parsed_timestamp is not None else last_ts
            if ln.parsed_timestamp is not None:
                last_ts = ln.parsed_timestamp

            if ts is None:
                # No timestamp context at all — keep the line.
                result.append(ln)
                continue

            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            result.append(ln)

        return result

    @staticmethod
    def _empty_stitched_log(source_files: List[str]) -> StitchedLog:
        """Return an empty StitchedLog."""
        return StitchedLog(
            lines=[],
            source_files=source_files,
            start_time=None,
            end_time=None,
            total_lines=0,
            lines_without_timestamp=0,
        )
