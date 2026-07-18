"""Enums and constants for the batch log intelligence platform.

All status codes, log levels, call types, and severity levels are
defined here as ``str, Enum`` hybrids so they serialise naturally
to JSON and can be used as dictionary keys.
"""

from __future__ import annotations

from enum import Enum
from typing import Set


class BatchStatus(str, Enum):
    """Lifecycle status of a batch job."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class AttemptType(str, Enum):
    """Classification of a job execution attempt."""

    SCHEDULED = "SCHEDULED"
    AUTO_RETRY = "AUTO_RETRY"
    MANUAL_RETRY = "MANUAL_RETRY"
    UNKNOWN = "UNKNOWN"


class LogLevel(str, Enum):
    """Standard log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"


class CallType(str, Enum):
    """Type of LLM invocation."""

    BATCH_ANALYSIS = "BATCH_ANALYSIS"
    QUERY_RESPONSE = "QUERY_RESPONSE"


class Severity(str, Enum):
    """Alert / finding severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── Derived Sets ────────────────────────────────────────────────

ERROR_LEVELS: Set[LogLevel] = {LogLevel.ERROR, LogLevel.FATAL, LogLevel.CRITICAL}
"""Log levels that indicate an error condition."""

WARN_LEVELS: Set[LogLevel] = {LogLevel.WARN, LogLevel.WARNING}
"""Log levels that indicate a warning condition."""
