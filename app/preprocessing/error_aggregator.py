"""Rule-based error aggregation for batch executions.

Categorises error lines using keyword matching (no LLM needed),
groups by category, counts occurrences, and assigns severity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.config.constants import ERROR_LEVELS, LogLevel, Severity
from app.segmentation.models import BatchExecution, ParsedLogLine


@dataclass
class ErrorRecord:
    """Aggregated error information for a single category.

    Attributes:
        error_category: Classification label (e.g. ``"DatabaseConnectivity"``).
        representative_message: Raw text of the first occurrence.
        count: Total number of error lines in this category.
        first_seen: Timestamp of the earliest occurrence.
        last_seen: Timestamp of the latest occurrence.
        severity: Assigned severity level.
        sample_lines: Up to 3 example raw lines.
    """

    error_category: str
    representative_message: str
    count: int
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    severity: Severity
    sample_lines: List[str] = field(default_factory=list)


# ── Category definitions ────────────────────────────────────────
# Each tuple: (category_name, compiled_regex, severity).
# Checked in priority order — first match wins.

_CATEGORIES: List[Tuple[str, re.Pattern[str], Severity]] = [
    (
        "DatabaseConnectivity",
        re.compile(
            r"(?:(?:db|database)\b.*(?:error|fail|exception))"
            r"|connection\s*refused"
            r"|(?:connection\s*)?(?:pool\s*exhausted|pool\s*timeout)"
            r"|timeout.*(?:db|database|jdbc|ora-)"
            r"|(?:jdbc|ora-)",
            re.IGNORECASE,
        ),
        Severity.HIGH,
    ),
    (
        "AuthenticationFailure",
        re.compile(
            r"\bauth\w*\s*(?:fail|error|exception)"
            r"|(?:un)?authorized"
            r"|\b40[13]\b"
            r"|permission\s*denied"
            r"|access\s*denied",
            re.IGNORECASE,
        ),
        Severity.HIGH,
    ),
    (
        "NullPointerException",
        re.compile(
            r"nullpointerexception"
            r"|null\s*ref"
            r"|object\s*reference\s*not\s*set",
            re.IGNORECASE,
        ),
        Severity.MEDIUM,
    ),
    (
        "FileNotFound",
        re.compile(
            r"filenotfound"
            r"|no\s*such\s*file"
            r"|file\s*does\s*not\s*exist",
            re.IGNORECASE,
        ),
        Severity.MEDIUM,
    ),
    (
        "NetworkTimeout",
        re.compile(
            r"(?:read|connect(?:ion)?)\s*time\s*out"
            r"|timed?\s*out"
            r"|socket\s*timeout",
            re.IGNORECASE,
        ),
        Severity.HIGH,
    ),
    (
        "OutOfMemory",
        re.compile(
            r"out\s*of\s*memory"
            r"|heap\s*space"
            r"|java\.lang\.outofmemoryerror"
            r"|oom\s*(?:kill|error)",
            re.IGNORECASE,
        ),
        Severity.CRITICAL,
    ),
    (
        "DataValidation",
        re.compile(
            r"validation\s*(?:fail|error)"
            r"|invalid\s*data"
            r"|constraint\s*violation"
            r"|duplicate\s*key",
            re.IGNORECASE,
        ),
        Severity.MEDIUM,
    ),
    (
        "RetryExhausted",
        re.compile(
            r"max\s*retr(?:y|ies)"
            r"|retry\s*limit"
            r"|retry\s*exhausted"
            r"|gave\s*up",
            re.IGNORECASE,
        ),
        Severity.HIGH,
    ),
    (
        "JobAborted",
        re.compile(
            r"\baborted\b"
            r"|\bkilled\b"
            r"|\bterminated\b"
            r"|sigterm"
            r"|sigkill",
            re.IGNORECASE,
        ),
        Severity.CRITICAL,
    ),
]

_MAX_SAMPLES = 3


class ErrorAggregator:
    """Categorise and count errors from a batch execution.

    Uses keyword / regex matching only — no LLM calls.

    Usage::

        aggregator = ErrorAggregator()
        errors = aggregator.aggregate(execution)
    """

    def aggregate(self, execution: BatchExecution) -> List[ErrorRecord]:
        """Aggregate error lines from *execution* into categorised records.

        Lines whose ``level`` matches ``ERROR_LEVELS`` are classified
        against the category list.  Results are sorted by count
        descending with up to 3 sample lines per category.

        Args:
            execution: A ``BatchExecution`` to analyse.

        Returns:
            List of ``ErrorRecord``, one per detected category.
        """
        buckets: Dict[str, _Bucket] = {}

        for line in execution.lines:
            if not self._is_error_level(line.level):
                continue

            category, severity = self._classify(line.raw)

            if category not in buckets:
                buckets[category] = _Bucket(
                    category=category,
                    severity=severity,
                    representative=line.raw,
                    first_seen=line.parsed_timestamp,
                    last_seen=line.parsed_timestamp,
                    count=0,
                    samples=[],
                )

            b = buckets[category]
            b.count += 1

            if line.parsed_timestamp is not None:
                if b.first_seen is None or line.parsed_timestamp < b.first_seen:
                    b.first_seen = line.parsed_timestamp
                if b.last_seen is None or line.parsed_timestamp > b.last_seen:
                    b.last_seen = line.parsed_timestamp

            if len(b.samples) < _MAX_SAMPLES:
                b.samples.append(line.raw)

        records = [
            ErrorRecord(
                error_category=b.category,
                representative_message=b.representative,
                count=b.count,
                first_seen=b.first_seen,
                last_seen=b.last_seen,
                severity=b.severity,
                sample_lines=list(b.samples),
            )
            for b in buckets.values()
        ]
        records.sort(key=lambda r: r.count, reverse=True)
        return records

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _is_error_level(level: str) -> bool:
        """Return ``True`` if *level* is an error-class level."""
        try:
            return LogLevel(level) in ERROR_LEVELS
        except ValueError:
            return False

    @staticmethod
    def _classify(text: str) -> Tuple[str, Severity]:
        """Return ``(category, severity)`` for *text*.

        First matching category wins; falls back to ``UnknownError``.
        """
        for category, pattern, severity in _CATEGORIES:
            if pattern.search(text):
                return category, severity
        return "UnknownError", Severity.MEDIUM


@dataclass
class _Bucket:
    """Internal accumulator for one error category."""

    category: str
    severity: Severity
    representative: str
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    count: int
    samples: List[str]
