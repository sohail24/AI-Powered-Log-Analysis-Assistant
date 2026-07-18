"""Compiled regex patterns for log parsing.

All patterns are pre-compiled at import time for performance.
Correlation patterns are ordered by specificity — first match wins.
"""

from __future__ import annotations

import re
from typing import List


# ── Correlation-ID Patterns ─────────────────────────────────────
# Each pattern MUST contain a named group ``cid``.
# Order matters: first match wins during extraction.

CORRELATION_PATTERNS: List[re.Pattern[str]] = [
    # 1. [CID:<value>]
    re.compile(r"\[CID:(?P<cid>[^\]]+)\]"),
    # 2. [BATCH:<value>]
    re.compile(r"\[BATCH:(?P<cid>[^\]]+)\]"),
    # 3. [TXN:<value>]
    re.compile(r"\[TXN:(?P<cid>[^\]]+)\]"),
    # 4. correlation_id= or correlation-id:
    re.compile(r"correlation[_-]id[=:\s]+(?P<cid>[\w\-]+)"),
    # 5. traceId=<hex-uuid>
    re.compile(r"traceId[=:\s]+(?P<cid>[a-f0-9\-]+)"),
    # 6. batch_id= or batch-id=
    re.compile(r"batch[_-]id[=:\s]+(?P<cid>[\w\-]+)"),
    # 7. requestId=<alphanum-uuid>
    re.compile(r"requestId[=:\s]+(?P<cid>[a-zA-Z0-9\-]+)"),
    # 8. [UPPERCASEID] — uppercase alphanumeric ≥8 chars in brackets
    re.compile(r"\[(?P<cid>[A-Z0-9]{8,})\]"),
]


# ── Timestamp Patterns ──────────────────────────────────────────
# Each pattern extracts a ``ts`` named group.
# Ordered from most specific (ISO-8601 with millis) to least.

TIMESTAMP_PATTERNS: List[re.Pattern[str]] = [
    # ISO-8601 with milliseconds and Z: 2025-06-13T02:00:01.123Z
    re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)"
    ),
    # ISO-8601 without millis: 2025-06-13T02:00:01
    re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    ),
    # Space-separated datetime: 2025-06-13 02:00:01
    re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})"
    ),
    # Apache / CLF: 13/Jun/2025:02:00:01
    re.compile(
        r"(?P<ts>\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2})"
    ),
    # Syslog: Jun 13 02:00:01
    re.compile(
        r"(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2})"
    ),
]


# ── Log Level Pattern ───────────────────────────────────────────
# Single compiled regex; case-insensitive.

LOG_LEVEL_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b",
    re.IGNORECASE,
)


# ── Job Start Patterns ──────────────────────────────────────────
# Each pattern has a named group ``job_name``.

JOB_START_PATTERNS: List[re.Pattern[str]] = [
    # "Starting job <name>"
    re.compile(
        r"[Ss]tarting\s+job\s+(?P<job_name>[\w\-\.]+)",
    ),
    # "BEGIN BATCH <name>"
    re.compile(
        r"BEGIN\s+BATCH\s+(?P<job_name>[\w\-\.]+)",
        re.IGNORECASE,
    ),
    # "Job started <name>"
    re.compile(
        r"[Jj]ob\s+started\s+(?P<job_name>[\w\-\.]+)",
    ),
    # "── START <name>"
    re.compile(
        r"[─\-]+\s*START\s+(?P<job_name>[\w\-\.]+)",
        re.IGNORECASE,
    ),
    # "Initiating <name>"
    re.compile(
        r"[Ii]nitiating\s+(?P<job_name>[\w\-\.]+)",
    ),
    # "Launching job <name>"
    re.compile(
        r"[Ll]aunching\s+job\s+(?P<job_name>[\w\-\.]+)",
    ),
]


# ── Job End Patterns ────────────────────────────────────────────
# Each pattern has a named group ``status`` capturing the outcome.

JOB_END_PATTERNS: List[re.Pattern[str]] = [
    # "Job completed SUCCESS|FAILED|ERROR|OK"
    re.compile(
        r"[Jj]ob\s+completed\s+(?P<status>SUCCESS|FAILED|ERROR|OK)",
        re.IGNORECASE,
    ),
    # "END BATCH <status>"
    re.compile(
        r"END\s+BATCH\s+(?P<status>SUCCESS|FAILED|ERROR|OK)",
        re.IGNORECASE,
    ),
    # "── END <status>"
    re.compile(
        r"[─\-]+\s*END\s+(?P<status>SUCCESS|FAILED|ERROR|OK)",
        re.IGNORECASE,
    ),
    # "Job finished <status>"
    re.compile(
        r"[Jj]ob\s+finished\s+(?P<status>SUCCESS|FAILED|ERROR|OK)",
        re.IGNORECASE,
    ),
    # "Batch complete <status>"
    re.compile(
        r"[Bb]atch\s+complete\s+(?P<status>SUCCESS|FAILED|ERROR|OK)",
        re.IGNORECASE,
    ),
    # "Execution finished <status>"
    re.compile(
        r"[Ee]xecution\s+finished\s+(?P<status>SUCCESS|FAILED|ERROR|OK)",
        re.IGNORECASE,
    ),
]
