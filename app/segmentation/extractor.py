"""Line parser and field extractor for log segmentation.

Extracts correlation IDs, log levels, and job boundary markers
from raw log lines using the compiled pattern registry in
:mod:`app.config.log_patterns`.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config.log_patterns import (
    CORRELATION_PATTERNS,
    JOB_END_PATTERNS,
    JOB_START_PATTERNS,
    LOG_LEVEL_PATTERN,
)
from app.config.settings import Settings
from app.ingestion.models import RawLogLine
from app.segmentation.models import ParsedLogLine

logger = logging.getLogger("extractor")


class LineParser:
    """Extract structured fields from a single raw log line.

    Uses the pre-compiled pattern lists from ``app.config.log_patterns``
    to extract correlation IDs (first-match-wins), log levels, and
    job start / end markers.

    Usage::

        parser = LineParser(settings)
        parsed = parser.parse_line(raw_line)
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the parser with application settings.

        Args:
            settings: Application-wide Settings instance.
        """
        self._settings = settings

    # ── Public API ──────────────────────────────────────────────

    def parse_line(self, raw_line: RawLogLine) -> ParsedLogLine:
        """Parse a single ``RawLogLine`` into a ``ParsedLogLine``.

        Extracts:
          1. ``correlation_id`` — first matching CID pattern.
          2. ``level`` — normalised uppercase log level.
          3. Job-start / job-end markers (used downstream).

        Never raises — always returns a result, even if all fields
        are ``None`` / ``"UNKNOWN"``.

        Args:
            raw_line: The raw log line to parse.

        Returns:
            Enriched ``ParsedLogLine``.
        """
        cid = self._extract_correlation_id(raw_line.raw)
        level = self._extract_level(raw_line.raw)

        return ParsedLogLine(
            raw=raw_line.raw,
            source_file=raw_line.source_file,
            file_line_number=raw_line.file_line_number,
            unified_line_number=raw_line.unified_line_number,
            ingestion_id=raw_line.ingestion_id,
            parsed_timestamp=raw_line.parsed_timestamp,
            level=level,
            correlation_id=cid,
            message=raw_line.raw,
            is_orphan=cid is None,
            orphan_attributed_to=None,
        )

    def extract_job_name(self, line: ParsedLogLine) -> Optional[str]:
        """Try to extract a job name from a parsed line.

        Iterates ``JOB_START_PATTERNS`` against the raw text and
        returns the ``job_name`` named group from the first match.

        Args:
            line: A parsed log line.

        Returns:
            Matched job name, or ``None``.
        """
        for pattern in JOB_START_PATTERNS:
            m = pattern.search(line.raw)
            if m:
                return m.group("job_name")
        return None

    def detect_job_start(self, line: ParsedLogLine) -> bool:
        """Return ``True`` if *line* contains a job-start marker.

        Args:
            line: A parsed log line.

        Returns:
            Whether any ``JOB_START_PATTERNS`` matched.
        """
        return any(p.search(line.raw) for p in JOB_START_PATTERNS)

    def detect_job_end(self, line: ParsedLogLine) -> Optional[str]:
        """Try to detect a job-end marker and return its status.

        Args:
            line: A parsed log line.

        Returns:
            The matched status string (e.g. ``"SUCCESS"``, ``"FAILED"``),
            or ``None`` if no end marker was found.
        """
        for pattern in JOB_END_PATTERNS:
            m = pattern.search(line.raw)
            if m:
                return m.group("status").upper()
        return None

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _extract_correlation_id(text: str) -> Optional[str]:
        """Extract the first-matching correlation ID from *text*.

        Patterns are tried in priority order; first match wins.
        CID values that are empty or whitespace-only are treated
        as no match.

        Returns:
            The CID string, or ``None``.
        """
        for pattern in CORRELATION_PATTERNS:
            m = pattern.search(text)
            if m:
                cid = m.group("cid").strip()
                if cid:
                    return cid
        return None

    @staticmethod
    def _extract_level(text: str) -> str:
        """Extract the log level from *text*, normalised to uppercase.

        Returns ``"UNKNOWN"`` if no level pattern matches.
        """
        m = LOG_LEVEL_PATTERN.search(text)
        if m:
            return m.group("level").upper()
        return "UNKNOWN"
