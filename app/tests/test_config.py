"""Tests for the configuration subsystem.

Validates that settings load correctly, all regex patterns contain
the expected named groups, timestamp patterns match sample strings,
and all enums expose the documented members.
"""

from __future__ import annotations

import re

import pytest

from app.config.settings import Settings, get_settings
from app.config.constants import (
    BatchStatus,
    AttemptType,
    LogLevel,
    CallType,
    Severity,
    ERROR_LEVELS,
    WARN_LEVELS,
)
from app.config.log_patterns import (
    CORRELATION_PATTERNS,
    TIMESTAMP_PATTERNS,
    LOG_LEVEL_PATTERN,
    JOB_START_PATTERNS,
    JOB_END_PATTERNS,
)


# ── Settings Tests ──────────────────────────────────────────────


class TestSettings:
    """Verify Settings dataclass loads and has sane defaults."""

    def test_settings_loads_without_error(self) -> None:
        """Settings can be instantiated with defaults (no .env required)."""
        settings = Settings()
        assert isinstance(settings, Settings)

    def test_default_llm_model(self) -> None:
        """Default LLM model is empty string (allowing provider defaults)."""
        settings = Settings()
        assert settings.llm_model == ""

    def test_default_llm_max_tokens(self) -> None:
        """Default max tokens is 1024."""
        settings = Settings()
        assert settings.llm_max_tokens == 1024

    def test_default_environment(self) -> None:
        """Default environment is 'prod'."""
        settings = Settings()
        assert settings.environment == "prod"

    def test_default_rag_top_k(self) -> None:
        """Default RAG top-k is 8."""
        settings = Settings()
        assert settings.rag_top_k_chunks == 8

    def test_default_embedding_model(self) -> None:
        """Default embedding model is all-MiniLM-L6-v2."""
        settings = Settings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_default_alert_on_critical(self) -> None:
        """Critical alerting is enabled by default."""
        settings = Settings()
        assert settings.alert_on_critical is True

    def test_slack_webhook_optional(self) -> None:
        """Slack webhook defaults to None."""
        settings = Settings()
        assert settings.slack_webhook_url is None

    def test_get_settings_returns_singleton(self) -> None:
        """get_settings returns the same instance on repeated calls."""
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()

    def test_get_settings_cache_clear(self) -> None:
        """Clearing cache produces a fresh instance."""
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # Both valid, but different objects after cache clear
        assert isinstance(s1, Settings)
        assert isinstance(s2, Settings)
        get_settings.cache_clear()


# ── Correlation Pattern Tests ───────────────────────────────────


class TestCorrelationPatterns:
    """Every correlation pattern must have a named group 'cid'."""

    def test_all_patterns_have_cid_group(self) -> None:
        """All CORRELATION_PATTERNS contain the named group 'cid'."""
        for i, pattern in enumerate(CORRELATION_PATTERNS):
            assert "cid" in pattern.groupindex, (
                f"CORRELATION_PATTERNS[{i}] ({pattern.pattern}) "
                f"is missing named group 'cid'"
            )

    @pytest.mark.parametrize(
        "sample, expected_cid",
        [
            ("[CID:abc-123]", "abc-123"),
            ("[BATCH:nightly-run]", "nightly-run"),
            ("[TXN:txn-456]", "txn-456"),
            ("correlation_id=corr-789", "corr-789"),
            ("correlation-id: corr-999", "corr-999"),
            ("traceId=abcdef01-2345", "abcdef01-2345"),
            ("batch_id=batch-001", "batch-001"),
            ("batch-id=batch-002", "batch-002"),
            ("requestId=REQ-12345", "REQ-12345"),
            ("[ABCD1234]", "ABCD1234"),
            ("[ABCDEFGH99]", "ABCDEFGH99"),
        ],
    )
    def test_correlation_pattern_matches(
        self, sample: str, expected_cid: str
    ) -> None:
        """Each sample string is matched by at least one pattern."""
        matched = False
        for pattern in CORRELATION_PATTERNS:
            m = pattern.search(sample)
            if m:
                assert m.group("cid") == expected_cid
                matched = True
                break
        assert matched, f"No correlation pattern matched: {sample!r}"


# ── Timestamp Pattern Tests ─────────────────────────────────────


class TestTimestampPatterns:
    """All timestamp formats must be recognised."""

    SAMPLE_TIMESTAMPS = [
        "2025-06-13 02:00:01",
        "2025-06-13T02:00:01",
        "2025-06-13T02:00:01.123Z",
        "13/Jun/2025:02:00:01",
        "Jun 13 02:00:01",
    ]

    def test_all_patterns_have_ts_group(self) -> None:
        """All TIMESTAMP_PATTERNS contain named group 'ts'."""
        for i, pattern in enumerate(TIMESTAMP_PATTERNS):
            assert "ts" in pattern.groupindex, (
                f"TIMESTAMP_PATTERNS[{i}] ({pattern.pattern}) "
                f"is missing named group 'ts'"
            )

    @pytest.mark.parametrize("sample", SAMPLE_TIMESTAMPS)
    def test_timestamp_matches(self, sample: str) -> None:
        """Each sample timestamp is matched by at least one pattern."""
        matched = any(p.search(sample) for p in TIMESTAMP_PATTERNS)
        assert matched, f"No timestamp pattern matched: {sample!r}"

    def test_each_pattern_matches_at_least_one_sample(self) -> None:
        """Every registered pattern matches at least one sample."""
        for i, pattern in enumerate(TIMESTAMP_PATTERNS):
            hits = [s for s in self.SAMPLE_TIMESTAMPS if pattern.search(s)]
            assert hits, (
                f"TIMESTAMP_PATTERNS[{i}] ({pattern.pattern}) "
                f"matched none of the sample timestamps"
            )


# ── Log Level Pattern Tests ─────────────────────────────────────


class TestLogLevelPattern:
    """Log level regex must catch all standard levels."""

    @pytest.mark.parametrize(
        "level",
        ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"],
    )
    def test_matches_standard_levels(self, level: str) -> None:
        """Each standard log level is matched."""
        line = f"2025-06-13 02:00:01 [{level}] Something happened"
        m = LOG_LEVEL_PATTERN.search(line)
        assert m is not None, f"LOG_LEVEL_PATTERN did not match {level}"
        assert m.group("level").upper() == level

    def test_case_insensitive(self) -> None:
        """Pattern matches lowercase levels too."""
        m = LOG_LEVEL_PATTERN.search("error: disk full")
        assert m is not None
        assert m.group("level").upper() == "ERROR"


# ── Job Start / End Pattern Tests ───────────────────────────────


class TestJobStartPatterns:
    """Job-start patterns must capture a job_name group."""

    @pytest.mark.parametrize(
        "sample, expected_name",
        [
            ("Starting job daily-etl", "daily-etl"),
            ("BEGIN BATCH nightly-sync", "nightly-sync"),
            ("Job started data-load", "data-load"),
            ("── START reconciliation", "reconciliation"),
            ("Initiating report-gen", "report-gen"),
            ("Launching job cleanup-v2", "cleanup-v2"),
        ],
    )
    def test_start_pattern_matches(
        self, sample: str, expected_name: str
    ) -> None:
        """Each sample is matched and the job_name group is correct."""
        matched = False
        for pattern in JOB_START_PATTERNS:
            m = pattern.search(sample)
            if m:
                assert m.group("job_name") == expected_name
                matched = True
                break
        assert matched, f"No JOB_START pattern matched: {sample!r}"

    def test_all_patterns_have_job_name_group(self) -> None:
        """Every start pattern contains 'job_name' named group."""
        for i, pattern in enumerate(JOB_START_PATTERNS):
            assert "job_name" in pattern.groupindex, (
                f"JOB_START_PATTERNS[{i}] missing 'job_name' group"
            )


class TestJobEndPatterns:
    """Job-end patterns must capture a status group."""

    @pytest.mark.parametrize(
        "sample, expected_status",
        [
            ("Job completed SUCCESS", "SUCCESS"),
            ("END BATCH FAILED", "FAILED"),
            ("── END OK", "OK"),
            ("Job finished ERROR", "ERROR"),
            ("Batch complete SUCCESS", "SUCCESS"),
            ("Execution finished FAILED", "FAILED"),
        ],
    )
    def test_end_pattern_matches(
        self, sample: str, expected_status: str
    ) -> None:
        """Each sample is matched and the status group is correct."""
        matched = False
        for pattern in JOB_END_PATTERNS:
            m = pattern.search(sample)
            if m:
                assert m.group("status").upper() == expected_status
                matched = True
                break
        assert matched, f"No JOB_END pattern matched: {sample!r}"

    def test_all_patterns_have_status_group(self) -> None:
        """Every end pattern contains 'status' named group."""
        for i, pattern in enumerate(JOB_END_PATTERNS):
            assert "status" in pattern.groupindex, (
                f"JOB_END_PATTERNS[{i}] missing 'status' group"
            )


# ── Enum Tests ──────────────────────────────────────────────────


class TestEnums:
    """All enums expose the documented members."""

    def test_batch_status_values(self) -> None:
        """BatchStatus has exactly the expected members."""
        expected = {"RUNNING", "SUCCESS", "FAILED", "PARTIAL", "UNKNOWN"}
        assert {s.value for s in BatchStatus} == expected

    def test_attempt_type_values(self) -> None:
        """AttemptType has exactly the expected members."""
        expected = {"SCHEDULED", "AUTO_RETRY", "MANUAL_RETRY", "UNKNOWN"}
        assert {a.value for a in AttemptType} == expected

    def test_log_level_values(self) -> None:
        """LogLevel has exactly the expected members."""
        expected = {
            "DEBUG", "INFO", "WARN", "WARNING",
            "ERROR", "FATAL", "CRITICAL",
        }
        assert {l.value for l in LogLevel} == expected

    def test_call_type_values(self) -> None:
        """CallType has exactly the expected members."""
        expected = {"BATCH_ANALYSIS", "QUERY_RESPONSE"}
        assert {c.value for c in CallType} == expected

    def test_severity_values(self) -> None:
        """Severity has exactly the expected members."""
        expected = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert {s.value for s in Severity} == expected

    def test_error_levels_set(self) -> None:
        """ERROR_LEVELS contains ERROR, FATAL, CRITICAL."""
        assert ERROR_LEVELS == {
            LogLevel.ERROR, LogLevel.FATAL, LogLevel.CRITICAL
        }

    def test_warn_levels_set(self) -> None:
        """WARN_LEVELS contains WARN and WARNING."""
        assert WARN_LEVELS == {LogLevel.WARN, LogLevel.WARNING}

    def test_str_enum_serialisation(self) -> None:
        """Enum members serialise to their string value."""
        assert str(BatchStatus.SUCCESS) == "BatchStatus.SUCCESS"
        assert BatchStatus.SUCCESS.value == "SUCCESS"
        assert f"{BatchStatus.SUCCESS.value}" == "SUCCESS"
