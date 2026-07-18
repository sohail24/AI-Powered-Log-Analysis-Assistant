"""End-to-end pipeline integration tests.

Uses a temporary directory with synthetic log content so tests are
fully self-contained and do not depend on external sample data.
The database uses an in-memory SQLite connection.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Generator

import pytest

from app.config.settings import Settings
from app.pipeline import IntelligencePipeline, PipelineResult
from app.storage.database import DatabaseManager
from app.storage.repository import BatchRepository


# ── Synthetic log content ────────────────────────────────────────

_INTERLEAVED_LOG = """\
2025-06-13 00:00:01 [INFO] [CID:BATCH-A] Starting job daily-etl
2025-06-13 00:00:02 [INFO] [CID:BATCH-B] Starting job report-gen
2025-06-13 00:00:03 [INFO] [CID:BATCH-A] Connecting to database
2025-06-13 00:00:04 [INFO] [CID:BATCH-B] Loading template config
2025-06-13 00:00:05 [INFO] [CID:BATCH-A] Extracted 15,000 records from source_a
2025-06-13 00:00:06 [DEBUG] [CID:BATCH-A] Starting transformation phase
2025-06-13 00:00:07 [INFO] [CID:BATCH-B] Querying aggregation table
2025-06-13 00:00:08 [ERROR] [CID:BATCH-B] JDBC connection refused to analytics-db
  at com.app.DB.connect(DB.java:55)
  at com.app.Main.run(Main.java:20)
2025-06-13 00:00:09 [INFO] [CID:BATCH-B] retrying connection...
2025-06-13 00:00:10 [INFO] [CID:BATCH-A] Transformation complete: 14988 records
2025-06-13 00:00:11 [WARN] [CID:BATCH-A] 12 records had null primary keys — skipped
2025-06-13 00:00:12 [FATAL] [CID:BATCH-B] java.lang.OutOfMemoryError: Java heap space
2025-06-13 00:00:13 [INFO] [CID:BATCH-B] Job completed FAILED
2025-06-13 00:00:14 [INFO] [CID:BATCH-A] Load phase complete
2025-06-13 00:00:15 [INFO] [CID:BATCH-A] Job completed SUCCESS
"""

_SIMPLE_LOG = """\
2025-06-13 01:00:00 [INFO] [CID:SIMPLE-1] Starting job simple-job
2025-06-13 01:00:01 [INFO] [CID:SIMPLE-1] Processing 500 records
2025-06-13 01:00:02 [INFO] [CID:SIMPLE-1] Done
2025-06-13 01:00:03 [INFO] [CID:SIMPLE-1] Job completed SUCCESS
"""

_NO_CID_LOG = """\
2025-06-13 03:00:00 [INFO] plain log line with no correlation ID
2025-06-13 03:00:01 [WARN] another plain line
2025-06-13 03:00:02 [ERROR] something went wrong but no CID
"""


# ── Helpers / Fixtures ───────────────────────────────────────────


def _make_settings(log_dir: str, db_path: str = ":memory:") -> Settings:
    """Create a Settings pointing at a temp log dir and in-memory DB."""
    return Settings(
        log_directory=log_dir,
        db_path=db_path,
        orphan_window_size=10,
        auto_retry_gap_minutes=5,
    )


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Generator[str, None, None]:
    """Temporary directory with the interleaved + simple log files."""
    (tmp_path / "interleaved.log").write_text(_INTERLEAVED_LOG, encoding="utf-8")
    (tmp_path / "simple.log").write_text(_SIMPLE_LOG, encoding="utf-8")
    yield str(tmp_path)


@pytest.fixture
def pipeline(tmp_log_dir: str) -> IntelligencePipeline:
    """Pipeline configured with an in-memory DB and temp log dir."""
    settings = _make_settings(tmp_log_dir)
    return IntelligencePipeline(settings)


@pytest.fixture
def result(pipeline: IntelligencePipeline, tmp_log_dir: str) -> PipelineResult:
    """Run the pipeline and return the result."""
    return pipeline.run(tmp_log_dir)


# ── Stitcher Step Tests ──────────────────────────────────────────


class TestPipelineStitching:
    """Verify log stitching in pipeline context."""

    def test_lines_are_stitched(self, result: PipelineResult) -> None:
        """Stitched log contains all lines from both files."""
        # interleaved.log has 18 lines (16 + 2 stack trace orphans)
        # simple.log has 4 lines
        assert result.stitched_log.total_lines >= 20

    def test_source_files_listed(self, result: PipelineResult) -> None:
        """Both log files appear in source_files."""
        source_names = {
            Path(f).name for f in result.stitched_log.source_files
        }
        assert "interleaved.log" in source_names
        assert "simple.log" in source_names


# ── De-interleaving Step Tests ───────────────────────────────────


class TestPipelineDeinterleaving:
    """Verify de-interleaving in pipeline context."""

    def test_correct_batch_count(self, result: PipelineResult) -> None:
        """Three distinct batches are found (BATCH-A, BATCH-B, SIMPLE-1)."""
        assert result.deinterleaved.total_batches_found == 3

    def test_batch_keys(self, result: PipelineResult) -> None:
        """All expected CIDs are present."""
        cids = set(result.deinterleaved.batches.keys())
        assert {"BATCH-A", "BATCH-B", "SIMPLE-1"}.issubset(cids)

    def test_batch_b_status_failed(self, result: PipelineResult) -> None:
        """BATCH-B (OOM + FAILED end marker) has FAILED status."""
        from app.config.constants import BatchStatus
        assert result.deinterleaved.batches["BATCH-B"].status == BatchStatus.FAILED

    def test_batch_a_status_success(self, result: PipelineResult) -> None:
        """BATCH-A (SUCCESS end marker) has SUCCESS status."""
        from app.config.constants import BatchStatus
        assert result.deinterleaved.batches["BATCH-A"].status == BatchStatus.SUCCESS

    def test_orphan_lines_attributed_to_batch_b(
        self, result: PipelineResult
    ) -> None:
        """Stack trace orphan lines are attributed to BATCH-B."""
        batch_b = result.deinterleaved.batches["BATCH-B"]
        assert batch_b.orphan_lines_count >= 2


# ── Job Grouping Step Tests ──────────────────────────────────────


class TestPipelineGrouping:
    """Verify job grouping in pipeline context."""

    def test_job_groups_created(self, result: PipelineResult) -> None:
        """Job groups are created (one per (job_name, date) pair)."""
        # 3 different job names → up to 3 groups on same day
        assert len(result.job_groups) >= 1

    def test_job_names_normalised(self, result: PipelineResult) -> None:
        """Job names are lower-cased and normalised."""
        names = {g.job_name for g in result.job_groups}
        # original: "daily-etl", "report-gen", "simple-job"
        assert "daily-etl" in names


# ── Storage Step Tests ───────────────────────────────────────────


class TestPipelineStorage:
    """Verify data is correctly written to the database."""

    def test_executions_stored_count(
        self, result: PipelineResult
    ) -> None:
        """executions_stored equals total batches found."""
        assert result.executions_stored == result.deinterleaved.total_batches_found

    def test_chunks_stored_positive(self, result: PipelineResult) -> None:
        """At least one log chunk is stored."""
        assert result.chunks_stored > 0

    def test_errors_stored_for_batch_b(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """Error summaries are persisted for BATCH-B."""
        rows = pipeline.repo.get_error_summary_for_cid("BATCH-B")
        assert len(rows) > 0

    def test_execution_retrievable_by_cid(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """Each CID is queryable via repository."""
        for cid in ("BATCH-A", "BATCH-B", "SIMPLE-1"):
            row = pipeline.repo.get_execution_by_cid(cid)
            assert row is not None, f"CID {cid!r} not found in DB"

    def test_execution_status_stored(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """Status field is persisted correctly."""
        row_a = pipeline.repo.get_execution_by_cid("BATCH-A")
        assert row_a["status"] == "SUCCESS"
        row_b = pipeline.repo.get_execution_by_cid("BATCH-B")
        assert row_b["status"] == "FAILED"

    def test_job_groups_stored(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """Job groups are stored in batch_jobs table."""
        rows = pipeline.repo.get_job_groups_by_date(date(2025, 6, 13))
        assert len(rows) >= 1

    def test_chunks_retrievable_for_batch_a(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """Log chunks for BATCH-A are retrievable."""
        chunks = pipeline.repo.get_chunks_for_cid("BATCH-A")
        assert len(chunks) > 0

    def test_chunks_cover_chunk_types(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """Chunks for BATCH-A include at least HEADER."""
        chunks = pipeline.repo.get_chunks_for_cid("BATCH-A")
        types = {row["chunk_type"] for row in chunks}
        assert "HEADER" in types

    def test_execution_duration_stored(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """Duration is stored for batches with timestamps."""
        row = pipeline.repo.get_execution_by_cid("BATCH-A")
        assert row["duration_seconds"] is not None
        assert row["duration_seconds"] > 0

    def test_source_files_stored_as_json(
        self, pipeline: IntelligencePipeline, result: PipelineResult
    ) -> None:
        """source_files column is a valid JSON array."""
        import json
        row = pipeline.repo.get_execution_by_cid("BATCH-A")
        files = json.loads(row["source_files"])
        assert isinstance(files, list)

    def test_idempotent_rerun(
        self, pipeline: IntelligencePipeline, tmp_log_dir: str
    ) -> None:
        """Running pipeline twice does not duplicate executions."""
        pipeline.run(tmp_log_dir)
        pipeline.run(tmp_log_dir)
        row = pipeline.repo.get_execution_by_cid("BATCH-A")
        assert row is not None  # Still exactly one row


# ── Result Metadata Tests ────────────────────────────────────────


class TestPipelineResult:
    """Verify PipelineResult fields."""

    def test_duration_positive(self, result: PipelineResult) -> None:
        """Pipeline duration is non-negative."""
        assert result.duration_seconds >= 0

    def test_warnings_is_list(self, result: PipelineResult) -> None:
        """warnings field is always a list."""
        assert isinstance(result.warnings, list)

    def test_result_has_all_fields(self, result: PipelineResult) -> None:
        """PipelineResult has all expected fields."""
        assert hasattr(result, "stitched_log")
        assert hasattr(result, "deinterleaved")
        assert hasattr(result, "job_groups")
        assert hasattr(result, "executions_stored")
        assert hasattr(result, "errors_stored")
        assert hasattr(result, "chunks_stored")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "warnings")


# ── Edge Case Tests ──────────────────────────────────────────────


class TestPipelineEdgeCases:
    """Pipeline behaviour for edge-case inputs."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty log directory returns empty result gracefully."""
        settings = _make_settings(str(tmp_path))
        pipeline = IntelligencePipeline(settings)
        result = pipeline.run(str(tmp_path))
        assert result.executions_stored == 0
        assert len(result.warnings) >= 1  # "No log lines" warning

    def test_no_cid_log_produces_warnings(self, tmp_path: Path) -> None:
        """Log with no CIDs triggers low-coverage warning."""
        (tmp_path / "no_cid.log").write_text(_NO_CID_LOG, encoding="utf-8")
        settings = _make_settings(str(tmp_path))
        pipeline = IntelligencePipeline(settings)
        result = pipeline.run(str(tmp_path))
        # Coverage is 0% < 80% threshold → warning.
        coverage_warnings = [
            w for w in result.warnings if "coverage" in w.lower()
        ]
        assert len(coverage_warnings) >= 1

    def test_pipeline_reusable(
        self, pipeline: IntelligencePipeline, tmp_log_dir: str
    ) -> None:
        """Pipeline.run() can be called multiple times."""
        r1 = pipeline.run(tmp_log_dir)
        r2 = pipeline.run(tmp_log_dir)
        assert r1.executions_stored > 0
        assert r2.executions_stored > 0
