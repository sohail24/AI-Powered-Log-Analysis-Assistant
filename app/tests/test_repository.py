"""Tests for the storage layer (DatabaseManager + BatchRepository).

All tests use in-memory SQLite (``:memory:``) for speed and isolation.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Generator

import pytest

from app.config.constants import BatchStatus
from app.segmentation.job_grouper import JobRunGroup
from app.segmentation.models import BatchExecution
from app.storage.database import DatabaseManager
from app.storage.repository import BatchRepository


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def db_manager() -> DatabaseManager:
    """Return an in-memory DatabaseManager, initialised."""
    mgr = DatabaseManager(":memory:")
    mgr.initialize()
    return mgr


@pytest.fixture
def repo(db_manager: DatabaseManager) -> BatchRepository:
    """Return a BatchRepository backed by the in-memory DB."""
    return BatchRepository(db_manager)


def _sample_group(
    job_name: str = "daily-etl",
    job_date: date | None = None,
    total_runs: int = 1,
    successful: int = 1,
    failed: int = 0,
    final_status: BatchStatus = BatchStatus.SUCCESS,
) -> JobRunGroup:
    """Create a sample JobRunGroup."""
    d = job_date or date(2025, 6, 13)
    return JobRunGroup(
        job_name=job_name,
        date=d,
        environment="prod",
        executions=[],
        total_runs=total_runs,
        successful_runs=successful,
        failed_runs=failed,
        final_status=final_status,
        first_run_time=datetime(d.year, d.month, d.day, 2, 0, 0),
        last_run_time=datetime(d.year, d.month, d.day, 2, 5, 0),
    )


def _sample_execution(
    cid: str = "CID-001",
    job_name: str = "daily-etl",
    status: BatchStatus = BatchStatus.SUCCESS,
    start: datetime | None = None,
    end: datetime | None = None,
) -> BatchExecution:
    """Create a sample BatchExecution."""
    return BatchExecution(
        correlation_id=cid,
        lines=[],
        source_files=["/test/app.log"],
        start_time=start or datetime(2025, 6, 13, 2, 0, 0),
        end_time=end or datetime(2025, 6, 13, 2, 5, 0),
        status=status,
        job_name=job_name,
        error_count=1 if status == BatchStatus.FAILED else 0,
        warn_count=0,
        total_lines=50,
        orphan_lines_count=3,
        has_start_marker=True,
        has_end_marker=True,
    )


# ── DatabaseManager Tests ──────────────────────────────────────


class TestDatabaseManager:
    """Schema creation and connection config."""

    def test_initialize_creates_tables(self, db_manager: DatabaseManager) -> None:
        """All 5 tables exist after initialization."""
        conn = db_manager.get_connection()
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            names = {row["name"] for row in tables}
            expected = {
                "batch_jobs",
                "batch_executions",
                "log_chunks",
                "error_summary",
                "llm_inference_log",
            }
            assert expected.issubset(names)
        finally:
            conn.close()

    def test_initialize_creates_indexes(self, db_manager: DatabaseManager) -> None:
        """All 7 indexes exist after initialization."""
        conn = db_manager.get_connection()
        try:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
            names = {row["name"] for row in indexes}
            expected = {
                "idx_executions_job_name",
                "idx_executions_status",
                "idx_executions_start_time",
                "idx_executions_cid",
                "idx_llm_log_cid",
                "idx_chunks_cid",
                "idx_errors_cid",
            }
            assert expected.issubset(names)
        finally:
            conn.close()

    def test_double_initialize_safe(self, db_manager: DatabaseManager) -> None:
        """Calling initialize() twice doesn't raise."""
        db_manager.initialize()  # second call

    def test_foreign_keys_enabled(self, db_manager: DatabaseManager) -> None:
        """Foreign keys pragma is ON."""
        conn = db_manager.get_connection()
        try:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()
            assert fk[0] == 1
        finally:
            conn.close()

    def test_wal_mode(self, db_manager: DatabaseManager) -> None:
        """Journal mode is WAL (or memory for in-memory DBs)."""
        conn = db_manager.get_connection()
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()
            # In-memory returns 'memory', file-based returns 'wal'.
            assert mode[0] in ("wal", "memory")
        finally:
            conn.close()


# ── Job Group Upsert Tests ─────────────────────────────────────


class TestUpsertJobGroup:
    """BatchRepository.upsert_job_group."""

    def test_insert_returns_id(self, repo: BatchRepository) -> None:
        """First insert returns a positive integer id."""
        job_id = repo.upsert_job_group(_sample_group())
        assert isinstance(job_id, int)
        assert job_id > 0

    def test_duplicate_upsert_updates(self, repo: BatchRepository) -> None:
        """Second upsert with same (name, date, env) updates the row."""
        g1 = _sample_group(total_runs=1, successful=1)
        id1 = repo.upsert_job_group(g1)

        g2 = _sample_group(total_runs=2, successful=1, failed=1)
        id2 = repo.upsert_job_group(g2)

        # Should be the same row.
        assert id1 == id2

        rows = repo.get_job_groups_by_date(date(2025, 6, 13))
        assert len(rows) == 1
        assert rows[0]["total_runs"] == 2

    def test_get_job_groups_by_date(self, repo: BatchRepository) -> None:
        """Retrieves job groups by date."""
        repo.upsert_job_group(_sample_group())
        rows = repo.get_job_groups_by_date(date(2025, 6, 13))
        assert len(rows) == 1
        assert rows[0]["job_name"] == "daily-etl"

    def test_get_job_groups_wrong_date(self, repo: BatchRepository) -> None:
        """No results for a different date."""
        repo.upsert_job_group(_sample_group())
        rows = repo.get_job_groups_by_date(date(2025, 6, 14))
        assert len(rows) == 0


# ── Execution Upsert Tests ─────────────────────────────────────


class TestUpsertExecution:
    """BatchRepository.upsert_execution."""

    def test_insert_and_retrieve(self, repo: BatchRepository) -> None:
        """Stores execution and retrieves by CID."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution()
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")

        row = repo.get_execution_by_cid("CID-001")
        assert row is not None
        assert row["correlation_id"] == "CID-001"
        assert row["job_name"] == "daily-etl"
        assert row["status"] == "SUCCESS"
        assert row["run_number"] == 1
        assert row["attempt_type"] == "SCHEDULED"

    def test_duration_calculated(self, repo: BatchRepository) -> None:
        """duration_seconds is computed from start/end times."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution(
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 5, 0),
        )
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")

        row = repo.get_execution_by_cid("CID-001")
        assert row["duration_seconds"] == 300.0

    def test_source_files_stored_as_json(self, repo: BatchRepository) -> None:
        """source_files is serialised as a JSON array."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution()
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")

        row = repo.get_execution_by_cid("CID-001")
        files = json.loads(row["source_files"])
        assert isinstance(files, list)
        assert "/test/app.log" in files

    def test_duplicate_execution_idempotent(self, repo: BatchRepository) -> None:
        """Inserting the same correlation_id twice doesn't raise."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution()
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")  # no error

        row = repo.get_execution_by_cid("CID-001")
        assert row is not None

    def test_get_execution_not_found(self, repo: BatchRepository) -> None:
        """Returns None for unknown CID."""
        row = repo.get_execution_by_cid("NONEXISTENT")
        assert row is None

    def test_get_executions_by_date(self, repo: BatchRepository) -> None:
        """Retrieves executions by date."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution()
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")

        rows = repo.get_executions_by_date(date(2025, 6, 13))
        assert len(rows) == 1

    def test_mark_execution_analyzed(self, repo: BatchRepository) -> None:
        """mark_execution_analyzed sets llm_analyzed to True."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution()
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")

        repo.mark_execution_analyzed("CID-001")
        row = repo.get_execution_by_cid("CID-001")
        assert row["llm_analyzed"] == 1


# ── Error Summary Tests ────────────────────────────────────────


class TestErrorSummary:
    """BatchRepository.store_error_summary and retrieval."""

    def test_store_and_retrieve(self, repo: BatchRepository) -> None:
        """Error summaries are stored and retrievable by CID."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution()
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")

        errors = [
            {
                "error_category": "ConnectionError",
                "error_message": "SMTP timeout",
                "count": 3,
                "first_seen": "2025-06-13T02:00:35",
                "last_seen": "2025-06-13T02:00:40",
                "severity": "HIGH",
            },
            {
                "error_category": "NullPointer",
                "error_message": "NullPointerException at line 42",
                "count": 1,
                "severity": "CRITICAL",
            },
        ]
        repo.store_error_summary("CID-001", errors)

        rows = repo.get_error_summary_for_cid("CID-001")
        assert len(rows) == 2

    def test_empty_errors(self, repo: BatchRepository) -> None:
        """No errors for unknown CID."""
        rows = repo.get_error_summary_for_cid("NONEXISTENT")
        assert len(rows) == 0


# ── Log Chunk Tests ─────────────────────────────────────────────


class TestLogChunks:
    """BatchRepository.store_log_chunk and retrieval."""

    def test_store_and_retrieve(self, repo: BatchRepository) -> None:
        """Chunks are stored and ordered by chunk_index."""
        job_id = repo.upsert_job_group(_sample_group())
        exe = _sample_execution()
        repo.upsert_execution(exe, job_id, 1, "SCHEDULED", "prod")

        repo.store_log_chunk(
            "CID-001", 0, "HEADER", "app.log", 1, 5,
            "header content here"
        )
        repo.store_log_chunk(
            "CID-001", 1, "ERROR_CLUSTER", "app.log", 6, 10,
            "error content here"
        )

        rows = repo.get_chunks_for_cid("CID-001")
        assert len(rows) == 2
        assert rows[0]["chunk_index"] == 0
        assert rows[0]["chunk_type"] == "HEADER"
        assert rows[0]["line_count"] == 5
        assert rows[1]["chunk_index"] == 1


# ── LLM Inference Log Tests ────────────────────────────────────


class TestLLMInferenceLog:
    """BatchRepository.store_llm_response and retrieval."""

    def test_store_and_retrieve(self, repo: BatchRepository) -> None:
        """LLM response is stored and retrievable."""
        record_id = repo.store_llm_response(
            correlation_id="CID-001",
            call_type="BATCH_ANALYSIS",
            request_prompt="Analyze this batch...",
            response_raw='{"summary": "All OK"}',
            response_parsed={"summary": "All OK"},
            usage={"input_tokens": 1000, "output_tokens": 500},
            prompt_version="v1.0",
        )
        assert record_id > 0

        row = repo.get_llm_response_for_cid("CID-001")
        assert row is not None
        assert row["call_type"] == "BATCH_ANALYSIS"
        assert row["input_tokens"] == 1000
        assert row["output_tokens"] == 500

    def test_cost_calculation(self, repo: BatchRepository) -> None:
        """Estimated cost is calculated correctly.

        input: 1000 tokens × $3.00/M = $0.003
        output: 500 tokens × $15.00/M = $0.0075
        total: $0.0105
        """
        repo.store_llm_response(
            correlation_id="CID-002",
            call_type="BATCH_ANALYSIS",
            request_prompt="prompt",
            response_raw="response",
            response_parsed=None,
            usage={"input_tokens": 1000, "output_tokens": 500},
            prompt_version="v1.0",
        )
        row = repo.get_llm_response_for_cid("CID-002")
        assert row["estimated_cost_usd"] == pytest.approx(0.0105, abs=1e-6)

    def test_no_response_returns_none(self, repo: BatchRepository) -> None:
        """get_llm_response_for_cid returns None if not present."""
        row = repo.get_llm_response_for_cid("NONEXISTENT")
        assert row is None

    def test_parse_success_flag(self, repo: BatchRepository) -> None:
        """parse_success is False when response_parsed is None."""
        repo.store_llm_response(
            correlation_id="CID-003",
            call_type="BATCH_ANALYSIS",
            request_prompt="prompt",
            response_raw="unparseable",
            response_parsed=None,
            usage={"input_tokens": 100, "output_tokens": 50},
            prompt_version="v1.0",
        )
        row = repo.get_llm_response_for_cid("CID-003")
        assert row["parse_success"] == 0  # False in SQLite

    def test_get_cost_summary_by_date(self, repo: BatchRepository) -> None:
        """Cost summary aggregates correctly."""
        # Insert two entries.
        repo.store_llm_response(
            correlation_id="CID-A",
            call_type="BATCH_ANALYSIS",
            request_prompt="p1",
            response_raw="r1",
            response_parsed={"ok": True},
            usage={"input_tokens": 2000, "output_tokens": 1000},
            prompt_version="v1.0",
        )
        repo.store_llm_response(
            correlation_id="CID-B",
            call_type="QUERY_RESPONSE",
            request_prompt="p2",
            response_raw="r2",
            response_parsed={"ok": True},
            usage={"input_tokens": 3000, "output_tokens": 500},
            prompt_version="v1.0",
        )

        rows = repo.get_cost_summary_by_date(days=7)
        assert len(rows) >= 1
        # Today's row should have 2 calls
        today_row = rows[0]
        assert today_row["call_count"] == 2
        assert today_row["total_input_tokens"] == 5000
        assert today_row["total_output_tokens"] == 1500


# ── Multiple Execution Types ───────────────────────────────────


class TestMultipleExecutions:
    """Multiple executions for the same job group."""

    def test_multiple_executions_stored(self, repo: BatchRepository) -> None:
        """Multiple executions with different CIDs are all stored."""
        job_id = repo.upsert_job_group(_sample_group(total_runs=3))

        for i in range(3):
            exe = _sample_execution(
                cid=f"CID-{i:03d}",
                start=datetime(2025, 6, 13, 2 + i, 0, 0),
                end=datetime(2025, 6, 13, 2 + i, 5, 0),
            )
            repo.upsert_execution(exe, job_id, i + 1, "SCHEDULED", "prod")

        rows = repo.get_executions_by_date(date(2025, 6, 13))
        assert len(rows) == 3
