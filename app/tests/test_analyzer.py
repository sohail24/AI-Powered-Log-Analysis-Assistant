"""Tests for BatchAnalyzer.

Uses unittest.mock to patch the Anthropic client so no real API
calls are made.  Covers:
- Cached response returned on second call (no API call fired).
- First call stores to DB and marks execution analyzed.
- _safe_parse handles clean JSON, fenced JSON, brace-extracted JSON.
- _safe_parse returns None on completely malformed input.
- analyze_execution marks llm_analyzed = True.
- analyze_job_run_group uses the multi-run prompt.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.config.constants import Severity
from app.config.settings import Settings
from app.llm.analyzer import BatchAnalyzer
from app.llm.digest_builder import DigestBuilder
from app.llm.models import BatchAnalysisResponse
from app.preprocessing.error_aggregator import ErrorRecord
from app.preprocessing.metrics_generator import BatchMetrics
from app.segmentation.models import BatchExecution, BatchStatus, ParsedLogLine
from app.storage.database import DatabaseManager
from app.storage.repository import BatchRepository

# ── Helpers ──────────────────────────────────────────────────────

_VALID_RESPONSE = json.dumps({
    "summary": "Batch BATCH-001 failed due to JDBC pool exhaustion.",
    "root_cause": "JDBC connection pool exhausted: 10/10 connections in use.",
    "error_categories": [
        {"category": "DatabaseConnectivity", "count": 5, "severity": "HIGH"},
    ],
    "recommendations": [
        {
            "action": "Increase JDBC pool size from 10 to 20 in application.properties.",
            "priority": "HIGH",
            "rationale": "Pool was fully exhausted causing all DB calls to fail.",
        }
    ],
    "business_impact": "Daily ETL failed; downstream reports will be 8 hours late.",
    "retry_recommended": True,
    "tags": ["db", "timeout", "infra"],
    "evidence_anchors": [
        {
            "description": "Connection pool exhausted at 02:01",
            "timestamp": "2025-06-13T02:01:00",
            "keywords": ["pool", "exhausted"],
            "severity": "HIGH",
        }
    ],
})


def _line(raw: str, level: str = "INFO", unified: int = 0) -> ParsedLogLine:
    return ParsedLogLine(
        raw=raw,
        source_file="/log/test.log",
        file_line_number=unified,
        unified_line_number=unified,
        ingestion_id=str(uuid4()),
        parsed_timestamp=datetime(2025, 6, 13, 0, 0, unified % 60),
        level=level,
        correlation_id="BATCH-001",
        message=raw,
    )


def _execution(
    cid: str = "BATCH-001",
    status: BatchStatus = BatchStatus.FAILED,
    lines: Optional[List[ParsedLogLine]] = None,
) -> BatchExecution:
    if lines is None:
        lines = [
            _line("INFO start", unified=0),
            _line("ERROR: JDBC timeout", level="ERROR", unified=1),
            _line("INFO end", unified=2),
        ]
    return BatchExecution(
        correlation_id=cid,
        lines=lines,
        start_time=datetime(2025, 6, 13, 0, 0, 0),
        end_time=datetime(2025, 6, 13, 0, 5, 0),
        status=status,
        job_name="daily-etl",
        error_count=1,
        warn_count=0,
        total_lines=len(lines),
        orphan_lines_count=0,
        has_start_marker=True,
        has_end_marker=True,
        source_files=["/log/test.log"],
    )


def _error_record() -> ErrorRecord:
    return ErrorRecord(
        error_category="DatabaseConnectivity",
        representative_message="JDBC connection pool exhausted",
        count=5,
        sample_lines=["ERROR: JDBC timeout"],
        first_seen=datetime(2025, 6, 13, 0, 1, 0),
        last_seen=datetime(2025, 6, 13, 0, 4, 0),
        severity=Severity.HIGH,
    )


def _metrics() -> BatchMetrics:
    return BatchMetrics(
        correlation_id="BATCH-001",
        duration_seconds=300.0,
        lines_per_second=1.0,
        error_rate_percent=33.0,
        warn_rate_percent=0.0,
        estimated_record_count=None,
        lines_in_first_10_percent=None,
        lines_in_last_10_percent=None,
        longest_gap_seconds=None,
        peak_error_window=None,
    )


def _make_mock_llm_client(text: str) -> MagicMock:
    """Build a mock LLMClient whose call() returns (text, usage)."""
    client = MagicMock()
    client.call.return_value = (text, {"input_tokens": 500, "output_tokens": 200})
    return client


@pytest.fixture
def db() -> DatabaseManager:
    """In-memory database for isolation."""
    mgr = DatabaseManager(":memory:")
    mgr.initialize()
    return mgr


@pytest.fixture
def repo(db: DatabaseManager) -> BatchRepository:
    return BatchRepository(db)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key="sk-test",
        llm_model="claude-sonnet-4-6",
        llm_max_tokens=1024,
    )


@pytest.fixture
def analyzer(settings: Settings, repo: BatchRepository) -> BatchAnalyzer:
    """Return a BatchAnalyzer with a mock LLMClient injected."""
    a = BatchAnalyzer.__new__(BatchAnalyzer)
    a._settings = settings
    a._repo = repo
    a._digest_builder = DigestBuilder(settings)
    mock_client = _make_mock_llm_client(_VALID_RESPONSE)
    a._client = mock_client
    return a


# ── Test: _safe_parse ────────────────────────────────────────────

class TestSafeParse:
    """Unit tests for BatchAnalyzer._safe_parse (no API needed)."""

    def _make_analyzer(self) -> BatchAnalyzer:
        db = DatabaseManager(":memory:")
        db.initialize()
        repo = BatchRepository(db)
        s = Settings(anthropic_api_key="sk-test")
        a = BatchAnalyzer.__new__(BatchAnalyzer)
        a._settings = s
        a._repo = repo
        a._digest_builder = DigestBuilder(s)
        a._client = None
        return a

    def test_parses_clean_json(self) -> None:
        """Direct JSON string is parsed successfully."""
        a = self._make_analyzer()
        result = a._safe_parse(_VALID_RESPONSE)
        assert result is not None
        assert result["summary"].startswith("Batch BATCH-001")

    def test_strips_json_fence_and_parses(self) -> None:
        """Response wrapped in ```json … ``` fences is parsed correctly."""
        a = self._make_analyzer()
        fenced = f"```json\n{_VALID_RESPONSE}\n```"
        result = a._safe_parse(fenced)
        assert result is not None
        assert "summary" in result

    def test_strips_plain_fence_and_parses(self) -> None:
        """Response wrapped in plain ``` fences is parsed correctly."""
        a = self._make_analyzer()
        fenced = f"```\n{_VALID_RESPONSE}\n```"
        result = a._safe_parse(fenced)
        assert result is not None

    def test_extracts_from_prose_with_json_embedded(self) -> None:
        """JSON embedded in prose is extracted via brace detection."""
        a = self._make_analyzer()
        text = f"Here is the analysis:\n{_VALID_RESPONSE}\nEnd."
        result = a._safe_parse(text)
        assert result is not None

    def test_returns_none_on_completely_malformed(self) -> None:
        """Completely malformed response returns None (no raise)."""
        a = self._make_analyzer()
        result = a._safe_parse("This is just plain text with no JSON whatsoever.")
        assert result is None

    def test_returns_none_on_empty_string(self) -> None:
        """Empty string returns None."""
        a = self._make_analyzer()
        result = a._safe_parse("")
        assert result is None

    def test_returns_none_when_required_keys_missing(self) -> None:
        """Parsed dict missing required keys is rejected → None."""
        a = self._make_analyzer()
        incomplete = json.dumps({"some_key": "some_value"})
        result = a._safe_parse(incomplete)
        assert result is None

    def test_does_not_raise_on_truncated_json(self) -> None:
        """Truncated JSON that can't be parsed returns None (no raise)."""
        a = self._make_analyzer()
        result = a._safe_parse('{"summary": "test", "root_ca')
        assert result is None


# ── Test: analyze_execution ──────────────────────────────────────

class TestAnalyzeExecution:
    """Integration tests for analyze_execution using in-memory DB."""

    def _seed_job(self, repo: BatchRepository, exc: BatchExecution) -> int:
        """Insert a minimal job row so upsert_execution FK is satisfied."""
        from app.segmentation.job_grouper import JobRunGroup
        from datetime import date
        group = JobRunGroup(
            job_name=exc.job_name or "test-job",
            date=date(2025, 6, 13),
            environment="prod",
            executions=[exc],
        )
        return repo.upsert_job_group(group)

    def test_returns_batch_analysis_response(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """analyze_execution returns a BatchAnalysisResponse."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        result = analyzer.analyze_execution(exc, [_error_record()], _metrics())
        assert isinstance(result, BatchAnalysisResponse)

    def test_response_parse_success_on_valid_json(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """Response from valid JSON has parse_success=True."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        result = analyzer.analyze_execution(exc, [], _metrics())
        assert result.parse_success is True

    def test_llm_client_called_once_on_first_call(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """API client is called exactly once on first analysis."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        analyzer.analyze_execution(exc, [], _metrics())
        assert analyzer._client.call.call_count == 1

    def test_cached_response_returned_on_second_call(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """Second call for the same CID reads from DB — no API call."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        # First call.
        analyzer.analyze_execution(exc, [], _metrics())
        call_count_after_first = analyzer._client.call.call_count
        # Second call.
        analyzer.analyze_execution(exc, [], _metrics())
        call_count_after_second = analyzer._client.call.call_count
        # No additional API call.
        assert call_count_after_second == call_count_after_first

    def test_force_reanalyze_bypasses_cache(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """force_reanalyze=True triggers a new API call even if cached."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        analyzer.analyze_execution(exc, [], _metrics())
        analyzer.analyze_execution(exc, [], _metrics(), force_reanalyze=True)
        assert analyzer._client.call.call_count == 2

    def test_stores_response_to_db(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """LLM response is persisted to llm_inference_log."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        analyzer.analyze_execution(exc, [], _metrics())
        stored = repo.get_llm_response_for_cid("BATCH-001", "BATCH_ANALYSIS")
        assert stored is not None
        assert stored["correlation_id"] == "BATCH-001"

    def test_marks_execution_as_analyzed(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """llm_analyzed flag is set to True in batch_executions."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        analyzer.analyze_execution(exc, [], _metrics())
        row = repo.get_execution_by_cid("BATCH-001")
        assert row is not None
        assert bool(row["llm_analyzed"]) is True

    def test_stored_response_contains_raw_text(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """Stored row's response_raw matches the mocked LLM output."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        analyzer.analyze_execution(exc, [], _metrics())
        stored = repo.get_llm_response_for_cid("BATCH-001", "BATCH_ANALYSIS")
        assert "Batch BATCH-001 failed" in stored["response_raw"]

    def test_response_has_root_cause(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """BatchAnalysisResponse.root_cause is populated from parsed JSON."""
        exc = _execution()
        self._seed_job(repo, exc)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")
        result = analyzer.analyze_execution(exc, [_error_record()], _metrics())
        assert result.root_cause is not None
        assert "pool" in result.root_cause.lower()


# ── Test: analyze_execution with malformed LLM response ─────────

class TestAnalyzeExecutionMalformedResponse:
    """Verify graceful degradation when LLM returns non-JSON."""

    def _seed_job(self, repo: BatchRepository, exc: BatchExecution) -> None:
        from app.segmentation.job_grouper import JobRunGroup
        from datetime import date
        group = JobRunGroup(
            job_name=exc.job_name or "test-job",
            date=date(2025, 6, 13),
            environment="prod",
            executions=[exc],
        )
        repo.upsert_job_group(group)
        repo.upsert_execution(exc, job_id=1, run_number=1,
                              attempt_type="SCHEDULED", environment="prod")

    def test_returns_response_with_parse_success_false(self) -> None:
        """Malformed LLM output → parse_success=False, no exception."""
        db = DatabaseManager(":memory:")
        db.initialize()
        repo = BatchRepository(db)
        settings = Settings(anthropic_api_key="sk-test")
        exc = _execution()
        self._seed_job(repo, exc)

        a = BatchAnalyzer.__new__(BatchAnalyzer)
        a._settings = settings
        a._repo = repo
        a._digest_builder = DigestBuilder(settings)
        a._client = _make_mock_llm_client("I'm sorry, I cannot analyse this log.")
        result = a.analyze_execution(exc, [], _metrics())

        assert isinstance(result, BatchAnalysisResponse)
        assert result.parse_success is False

    def test_empty_string_response_no_crash(self) -> None:
        """Empty LLM response → parse_success=False, no exception."""
        db = DatabaseManager(":memory:")
        db.initialize()
        repo = BatchRepository(db)
        settings = Settings(anthropic_api_key="sk-test")
        exc = _execution()
        self._seed_job(repo, exc)

        a = BatchAnalyzer.__new__(BatchAnalyzer)
        a._settings = settings
        a._repo = repo
        a._digest_builder = DigestBuilder(settings)
        a._client = _make_mock_llm_client("")
        result = a.analyze_execution(exc, [], _metrics())

        assert isinstance(result, BatchAnalysisResponse)
        assert result.parse_success is False


# ── Test: analyze_job_run_group ──────────────────────────────────

class TestAnalyzeJobRunGroup:
    """Verify multi-run analysis uses the correct prompt and stores results."""

    def test_uses_multi_run_prompt(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """Multi-run call passes MULTI_RUN_SYSTEM_PROMPT to the API."""
        from app.llm.prompts import MULTI_RUN_SYSTEM_PROMPT
        from app.segmentation.job_grouper import JobRunGroup
        from datetime import date

        exc1 = _execution(cid="BATCH-A", status=BatchStatus.FAILED)
        exc2 = _execution(cid="BATCH-B", status=BatchStatus.SUCCESS)
        exc1._run_number = 1  # type: ignore[attr-defined]
        exc2._run_number = 2  # type: ignore[attr-defined]
        exc1._attempt_type = "SCHEDULED"  # type: ignore[attr-defined]
        exc2._attempt_type = "AUTO_RETRY"  # type: ignore[attr-defined]

        group = JobRunGroup(
            job_name="daily-etl",
            date=date(2025, 6, 13),
            environment="prod",
            executions=[exc1, exc2],
        )
        analyzer.analyze_job_run_group(group, {}, {})
        # The unified client.call(system, user_content) takes positional args.
        call_args = analyzer._client.call.call_args
        system_arg = call_args[0][0] if call_args[0] else call_args[1].get("system")
        assert system_arg == MULTI_RUN_SYSTEM_PROMPT

    def test_group_result_stored_with_group_cid(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """Multi-run result stored under last_cid + '_group'."""
        from app.segmentation.job_grouper import JobRunGroup
        from datetime import date

        exc1 = _execution(cid="BATCH-A", status=BatchStatus.FAILED)
        exc2 = _execution(cid="BATCH-B", status=BatchStatus.SUCCESS)
        exc1._run_number = 1  # type: ignore[attr-defined]
        exc2._run_number = 2  # type: ignore[attr-defined]
        exc1._attempt_type = "SCHEDULED"  # type: ignore[attr-defined]
        exc2._attempt_type = "AUTO_RETRY"  # type: ignore[attr-defined]

        group = JobRunGroup(
            job_name="daily-etl",
            date=date(2025, 6, 13),
            environment="prod",
            executions=[exc1, exc2],
        )
        analyzer.analyze_job_run_group(group, {}, {})
        stored = repo.get_llm_response_for_cid("BATCH-B_group", "BATCH_ANALYSIS")
        assert stored is not None

    def test_empty_group_returns_response_without_crash(
        self, analyzer: BatchAnalyzer, repo: BatchRepository
    ) -> None:
        """Empty group → returns a sentinel response, does not raise."""
        from app.segmentation.job_grouper import JobRunGroup
        from datetime import date

        group = JobRunGroup(
            job_name="empty-job",
            date=date(2025, 6, 13),
            environment="prod",
            executions=[],
        )
        result = analyzer.analyze_job_run_group(group, {}, {})
        assert isinstance(result, BatchAnalysisResponse)
        assert result.parse_success is False


# ── Test: LLM Client Factory and Providers ──────────────────────

class TestLLMClientFactory:
    """Verify that Settings correctly resolves LLM providers and models."""

    def test_factory_returns_none_if_no_keys(self) -> None:
        """If no keys are provided, return None (disabled)."""
        from app.llm.client import get_llm_client
        s = Settings(llm_provider="auto", anthropic_api_key="", google_api_key="")
        client = get_llm_client(s)
        assert client is None

    def test_factory_resolves_explicit_anthropic(self) -> None:
        """Explicitly request 'anthropic' provider."""
        from app.llm.client import get_llm_client, AnthropicClient
        s = Settings(
            llm_provider="anthropic",
            anthropic_api_key="sk-ant",
            google_api_key="AIza",
            llm_model="custom-claude"
        )
        with patch("anthropic.Anthropic") as MockAnthropic:
            client = get_llm_client(s)
            assert isinstance(client, AnthropicClient)
            assert client._model == "custom-claude"
            assert client._max_tokens == 1024
            MockAnthropic.assert_called_once_with(api_key="sk-ant")

    def test_factory_resolves_explicit_google(self) -> None:
        """Explicitly request 'google' provider."""
        from app.llm.client import get_llm_client, GoogleClient
        s = Settings(
            llm_provider="google",
            anthropic_api_key="sk-ant",
            google_api_key="AIza-key",
            llm_model="custom-gemini"
        )
        with patch("google.generativeai.configure") as mock_configure:
            client = get_llm_client(s)
            assert isinstance(client, GoogleClient)
            assert client._model_name == "custom-gemini"
            mock_configure.assert_called_once_with(api_key="AIza-key")

    def test_factory_auto_detect_prefers_anthropic(self) -> None:
        """Auto mode with both keys present prefers Anthropic."""
        from app.llm.client import get_llm_client, AnthropicClient
        s = Settings(
            llm_provider="auto",
            anthropic_api_key="sk-ant",
            google_api_key="AIza-key"
        )
        with patch("anthropic.Anthropic"):
            client = get_llm_client(s)
            assert isinstance(client, AnthropicClient)
            assert client._model == "claude-sonnet-4-6"  # default model resolved

    def test_factory_auto_detect_falls_back_to_google(self) -> None:
        """Auto mode with only Google key returns Google client."""
        from app.llm.client import get_llm_client, GoogleClient
        s = Settings(
            llm_provider="auto",
            anthropic_api_key="",
            google_api_key="AIza-key"
        )
        with patch("google.generativeai.configure"):
            client = get_llm_client(s)
            assert isinstance(client, GoogleClient)
            assert client._model_name == "gemini-2.0-flash"  # default model resolved

    def test_google_client_wraps_call(self) -> None:
        """GoogleClient.call correctly invokes the generativeai SDK and yields token usage."""
        from app.llm.client import GoogleClient
        with patch("google.generativeai.configure"):
            client = GoogleClient(api_key="key", model="gemini-model", max_tokens=256)

        # Mock the GenerativeModel call.
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test output"
        mock_response.usage_metadata.prompt_token_count = 120
        mock_response.usage_metadata.candidates_token_count = 50
        mock_model.generate_content.return_value = mock_response
        client._genai.GenerativeModel = MagicMock(return_value=mock_model)

        raw, usage = client.call("system-instructions", "user-content")
        assert raw == "test output"
        assert usage == {"input_tokens": 120, "output_tokens": 50}
        client._genai.GenerativeModel.assert_called_once()
        _, kwargs = client._genai.GenerativeModel.call_args
        assert kwargs["system_instruction"] == "system-instructions"
        assert kwargs["generation_config"].max_output_tokens == 256
        assert kwargs["generation_config"].temperature == 0.0
