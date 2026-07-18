"""Tests for the JobGrouper.

Covers single/multiple executions, auto-retry vs manual-retry
classification based on gap, name normalisation, and multi-job
grouping.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List

import pytest

from app.config.constants import AttemptType, BatchStatus
from app.config.settings import Settings
from app.segmentation.job_grouper import JobGrouper, JobRunGroup
from app.segmentation.models import BatchExecution, DeinterleavedResult


# ── Helpers ─────────────────────────────────────────────────────


def _exe(
    cid: str,
    job_name: str,
    status: BatchStatus,
    start: datetime | None = None,
    end: datetime | None = None,
) -> BatchExecution:
    """Create a minimal BatchExecution for testing."""
    return BatchExecution(
        correlation_id=cid,
        lines=[],
        source_files=["/test.log"],
        start_time=start,
        end_time=end,
        status=status,
        job_name=job_name,
        error_count=1 if status == BatchStatus.FAILED else 0,
        warn_count=0,
        total_lines=10,
        orphan_lines_count=0,
        has_start_marker=True,
        has_end_marker=True,
    )


def _result(
    batches: List[BatchExecution],
) -> DeinterleavedResult:
    """Wrap a list of BatchExecutions into a DeinterleavedResult."""
    batches_dict = {b.correlation_id: b for b in batches}
    return DeinterleavedResult(
        batches=batches_dict,
        unattributed_lines=[],
        total_lines_processed=sum(b.total_lines for b in batches),
        total_batches_found=len(batches),
        batches_with_no_start_marker=[],
        batches_with_no_end_marker=[],
        cid_coverage_percent=100.0,
    )


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    """Default settings (auto_retry_gap_minutes = 5)."""
    return Settings()


@pytest.fixture
def grouper(settings: Settings) -> JobGrouper:
    """Return a JobGrouper."""
    return JobGrouper(settings)


# ── Single Execution Tests ──────────────────────────────────────


class TestSingleExecution:
    """A single execution maps to run_number=1, SCHEDULED."""

    def test_single_run_number(self, grouper: JobGrouper) -> None:
        """Single execution gets run_number = 1."""
        exe = _exe(
            "C1", "daily-etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 5, 0),
        )
        groups = grouper.group(_result([exe]))
        assert len(groups) == 1
        assert groups[0].total_runs == 1
        assert groups[0].executions[0]._run_number == 1  # type: ignore[attr-defined]

    def test_single_attempt_type_scheduled(self, grouper: JobGrouper) -> None:
        """Single execution is classified as SCHEDULED."""
        exe = _exe(
            "C1", "daily-etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 5, 0),
        )
        groups = grouper.group(_result([exe]))
        assert groups[0].executions[0]._attempt_type == AttemptType.SCHEDULED  # type: ignore[attr-defined]

    def test_single_final_status(self, grouper: JobGrouper) -> None:
        """Final status equals the only execution's status."""
        exe = _exe(
            "C1", "daily-etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 5, 0),
        )
        groups = grouper.group(_result([exe]))
        assert groups[0].final_status == BatchStatus.SUCCESS


# ── Auto-Retry Tests ───────────────────────────────────────────


class TestAutoRetry:
    """Two executions with gap < 5 min → second is AUTO_RETRY."""

    def test_auto_retry_classification(self, grouper: JobGrouper) -> None:
        """Gap of 2 minutes triggers AUTO_RETRY."""
        exe1 = _exe(
            "C1", "daily-etl", BatchStatus.FAILED,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 3, 0),
        )
        exe2 = _exe(
            "C2", "daily-etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 5, 0),
            end=datetime(2025, 6, 13, 2, 10, 0),
        )
        groups = grouper.group(_result([exe1, exe2]))
        assert len(groups) == 1
        execs = groups[0].executions
        assert execs[0]._attempt_type == AttemptType.SCHEDULED  # type: ignore[attr-defined]
        assert execs[1]._attempt_type == AttemptType.AUTO_RETRY  # type: ignore[attr-defined]

    def test_auto_retry_run_numbers(self, grouper: JobGrouper) -> None:
        """Run numbers are 1, 2 in order."""
        exe1 = _exe(
            "C1", "daily-etl", BatchStatus.FAILED,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 3, 0),
        )
        exe2 = _exe(
            "C2", "daily-etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 5, 0),
            end=datetime(2025, 6, 13, 2, 10, 0),
        )
        groups = grouper.group(_result([exe1, exe2]))
        assert groups[0].executions[0]._run_number == 1  # type: ignore[attr-defined]
        assert groups[0].executions[1]._run_number == 2  # type: ignore[attr-defined]


# ── Manual Retry Tests ─────────────────────────────────────────


class TestManualRetry:
    """Two executions with gap >= 5 min → second is MANUAL_RETRY."""

    def test_manual_retry_classification(self, grouper: JobGrouper) -> None:
        """Gap of 30 minutes triggers MANUAL_RETRY."""
        exe1 = _exe(
            "C1", "daily-etl", BatchStatus.FAILED,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 5, 0),
        )
        exe2 = _exe(
            "C2", "daily-etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 35, 0),
            end=datetime(2025, 6, 13, 2, 40, 0),
        )
        groups = grouper.group(_result([exe1, exe2]))
        assert groups[0].executions[1]._attempt_type == AttemptType.MANUAL_RETRY  # type: ignore[attr-defined]

    def test_missing_end_time_gives_unknown(self, grouper: JobGrouper) -> None:
        """If previous end_time is None, attempt_type is UNKNOWN."""
        exe1 = _exe(
            "C1", "daily-etl", BatchStatus.FAILED,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=None,
        )
        exe2 = _exe(
            "C2", "daily-etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 35, 0),
            end=datetime(2025, 6, 13, 2, 40, 0),
        )
        groups = grouper.group(_result([exe1, exe2]))
        assert groups[0].executions[1]._attempt_type == AttemptType.UNKNOWN  # type: ignore[attr-defined]


# ── Multi-Job Group Tests ──────────────────────────────────────


class TestMultipleJobs:
    """Different job names → separate groups."""

    def test_three_jobs_three_groups(self, grouper: JobGrouper) -> None:
        """Three different job names produce three groups."""
        exes = [
            _exe("C1", "daily-etl", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 13, 2, 0, 0),
                 end=datetime(2025, 6, 13, 2, 5, 0)),
            _exe("C2", "report-gen", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 13, 3, 0, 0),
                 end=datetime(2025, 6, 13, 3, 5, 0)),
            _exe("C3", "cleanup", BatchStatus.FAILED,
                 start=datetime(2025, 6, 13, 4, 0, 0),
                 end=datetime(2025, 6, 13, 4, 5, 0)),
        ]
        groups = grouper.group(_result(exes))
        assert len(groups) == 3

    def test_same_job_different_dates(self, grouper: JobGrouper) -> None:
        """Same job on different dates → separate groups."""
        exes = [
            _exe("C1", "daily-etl", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 13, 2, 0, 0),
                 end=datetime(2025, 6, 13, 2, 5, 0)),
            _exe("C2", "daily-etl", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 14, 2, 0, 0),
                 end=datetime(2025, 6, 14, 2, 5, 0)),
        ]
        groups = grouper.group(_result(exes))
        assert len(groups) == 2


# ── Job Name Normalisation Tests ────────────────────────────────


class TestJobNameNormalisation:
    """JobGrouper._normalize_job_name edge cases."""

    def test_lowercase(self, grouper: JobGrouper) -> None:
        """Names are lower-cased."""
        assert grouper._normalize_job_name("Daily-ETL") == "daily-etl"

    def test_strip_whitespace(self, grouper: JobGrouper) -> None:
        """Leading/trailing whitespace is stripped."""
        assert grouper._normalize_job_name("  daily-etl  ") == "daily-etl"

    def test_spaces_to_underscores(self, grouper: JobGrouper) -> None:
        """Internal spaces become underscores."""
        assert grouper._normalize_job_name("Nightly Reconciliation") == "nightly_reconciliation"

    def test_special_chars_removed(self, grouper: JobGrouper) -> None:
        """Special characters (except - and _) are removed."""
        assert grouper._normalize_job_name("job@name!v2") == "jobnamev2"

    def test_hyphens_preserved(self, grouper: JobGrouper) -> None:
        """Hyphens are preserved."""
        assert grouper._normalize_job_name("data-cleanup") == "data-cleanup"

    def test_multiple_spaces_collapse(self, grouper: JobGrouper) -> None:
        """Multiple consecutive spaces collapse to one underscore."""
        assert grouper._normalize_job_name("job   name") == "job_name"

    def test_same_normalised_name_grouped(self, grouper: JobGrouper) -> None:
        """Executions with different raw names but same normalised form are grouped."""
        exes = [
            _exe("C1", "Daily ETL", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 13, 2, 0, 0),
                 end=datetime(2025, 6, 13, 2, 5, 0)),
            _exe("C2", "daily_etl", BatchStatus.FAILED,
                 start=datetime(2025, 6, 13, 3, 0, 0),
                 end=datetime(2025, 6, 13, 3, 5, 0)),
        ]
        groups = grouper.group(_result(exes))
        assert len(groups) == 1
        assert groups[0].total_runs == 2


# ── Summary Fields Tests ───────────────────────────────────────


class TestSummaryFields:
    """Verify computed fields on JobRunGroup."""

    def test_successful_and_failed_counts(self, grouper: JobGrouper) -> None:
        """successful_runs and failed_runs counted correctly."""
        exes = [
            _exe("C1", "etl", BatchStatus.FAILED,
                 start=datetime(2025, 6, 13, 2, 0, 0),
                 end=datetime(2025, 6, 13, 2, 3, 0)),
            _exe("C2", "etl", BatchStatus.FAILED,
                 start=datetime(2025, 6, 13, 2, 4, 0),
                 end=datetime(2025, 6, 13, 2, 6, 0)),
            _exe("C3", "etl", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 13, 2, 7, 0),
                 end=datetime(2025, 6, 13, 2, 10, 0)),
        ]
        groups = grouper.group(_result(exes))
        g = groups[0]
        assert g.successful_runs == 1
        assert g.failed_runs == 2
        assert g.total_runs == 3

    def test_final_status_is_last_execution(self, grouper: JobGrouper) -> None:
        """final_status reflects the last execution."""
        exes = [
            _exe("C1", "etl", BatchStatus.FAILED,
                 start=datetime(2025, 6, 13, 2, 0, 0),
                 end=datetime(2025, 6, 13, 2, 3, 0)),
            _exe("C2", "etl", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 13, 2, 7, 0),
                 end=datetime(2025, 6, 13, 2, 10, 0)),
        ]
        groups = grouper.group(_result(exes))
        assert groups[0].final_status == BatchStatus.SUCCESS

    def test_first_and_last_run_time(self, grouper: JobGrouper) -> None:
        """first_run_time / last_run_time bracket the runs."""
        exes = [
            _exe("C1", "etl", BatchStatus.FAILED,
                 start=datetime(2025, 6, 13, 2, 0, 0),
                 end=datetime(2025, 6, 13, 2, 3, 0)),
            _exe("C2", "etl", BatchStatus.SUCCESS,
                 start=datetime(2025, 6, 13, 4, 0, 0),
                 end=datetime(2025, 6, 13, 4, 5, 0)),
        ]
        groups = grouper.group(_result(exes))
        g = groups[0]
        assert g.first_run_time == datetime(2025, 6, 13, 2, 0, 0)
        assert g.last_run_time == datetime(2025, 6, 13, 4, 0, 0)

    def test_environment_passed_through(self, grouper: JobGrouper) -> None:
        """Environment label is set on the group."""
        exe = _exe(
            "C1", "etl", BatchStatus.SUCCESS,
            start=datetime(2025, 6, 13, 2, 0, 0),
            end=datetime(2025, 6, 13, 2, 5, 0),
        )
        groups = grouper.group(_result([exe]), environment="staging")
        assert groups[0].environment == "staging"

    def test_none_start_time_uses_today(self, grouper: JobGrouper) -> None:
        """Execution with None start_time uses today's date."""
        exe = _exe("C1", "etl", BatchStatus.UNKNOWN, start=None, end=None)
        groups = grouper.group(_result([exe]))
        assert groups[0].date == date.today()


# ── Gap Computation Tests ──────────────────────────────────────


class TestGapComputation:
    """JobGrouper._compute_gap_minutes edge cases."""

    def test_gap_in_minutes(self, grouper: JobGrouper) -> None:
        """Gap is correctly computed in minutes."""
        prev = _exe("C1", "x", BatchStatus.FAILED,
                     end=datetime(2025, 6, 13, 2, 0, 0))
        curr = _exe("C2", "x", BatchStatus.SUCCESS,
                     start=datetime(2025, 6, 13, 2, 10, 0))
        assert grouper._compute_gap_minutes(prev, curr) == 10.0

    def test_gap_none_when_missing(self, grouper: JobGrouper) -> None:
        """Returns None if either timestamp is missing."""
        prev = _exe("C1", "x", BatchStatus.FAILED, end=None)
        curr = _exe("C2", "x", BatchStatus.SUCCESS,
                     start=datetime(2025, 6, 13, 2, 10, 0))
        assert grouper._compute_gap_minutes(prev, curr) is None
