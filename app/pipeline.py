"""Master pipeline orchestrator — ingestion → storage.

Wires together all Phase 1 components:
  Stitcher → DeinterleavingEngine → JobGrouper
  → ErrorAggregator / MetricsGenerator / LogChunker
  → BatchRepository (SQLite)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from app.config.settings import Settings
from app.ingestion.models import StitchedLog
from app.ingestion.stitcher import LogStitcher
from app.llm.analyzer import BatchAnalyzer
from app.preprocessing.chunker import LogChunk, LogChunker
from app.preprocessing.error_aggregator import ErrorAggregator, ErrorRecord
from app.preprocessing.metrics_generator import BatchMetrics, MetricsGenerator
from app.segmentation.deinterleaver import DeinterleavingEngine
from app.segmentation.job_grouper import JobGrouper, JobRunGroup
from app.segmentation.models import DeinterleavedResult
from app.storage.database import DatabaseManager
from app.storage.repository import BatchRepository

logger = logging.getLogger("pipeline")


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run.

    Attributes:
        stitched_log: The merged, sorted unified log.
        deinterleaved: Per-batch de-interleaving result.
        job_groups: Grouped batch executions by (job_name, date).
        executions_stored: Number of executions written to DB.
        errors_stored: Total error records written to DB.
        chunks_stored: Total log chunks written to DB.
        duration_seconds: Wall-clock time for the full pipeline.
        warnings: Non-fatal issues collected during processing.
    """

    stitched_log: StitchedLog
    deinterleaved: DeinterleavedResult
    job_groups: List[JobRunGroup]
    executions_stored: int = 0
    errors_stored: int = 0
    chunks_stored: int = 0
    analyses_triggered: int = 0
    duration_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)


class IntelligencePipeline:
    """End-to-end orchestrator for Phase 1 log processing.

    Usage::

        pipeline = IntelligencePipeline(settings)
        result = pipeline.run("./data/logs", environment="prod")
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise all pipeline components.

        Args:
            settings: Application-wide Settings instance.
        """
        self._settings = settings

        # Ingestion
        self.stitcher = LogStitcher(settings)

        # Segmentation
        self.deinterleaver = DeinterleavingEngine(settings)
        self.grouper = JobGrouper(settings)

        # Preprocessing
        self.error_aggregator = ErrorAggregator()
        self.metrics_generator = MetricsGenerator()
        self.chunker = LogChunker()

        # Storage
        self.db = DatabaseManager(settings.db_path)
        self.repo = BatchRepository(self.db)

        # Ensure schema exists on startup.
        self.db.initialize()

        # LLM analysis (initialised lazily — skipped if no API key).
        self.analyzer: Optional[BatchAnalyzer] = (
            BatchAnalyzer(settings, self.repo)
            if settings.anthropic_api_key
            else None
        )

    # ── Public API ──────────────────────────────────────────────

    def run(
        self,
        log_directory: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        environment: str = "prod",
    ) -> PipelineResult:
        """Run the full ingestion → storage pipeline.

        Steps:
          1. Stitch raw log files from *log_directory*.
          2. De-interleave into per-batch executions.
          3. Group executions by job name and date.
          4. For each execution: aggregate errors, generate metrics,
             chunk the log, persist to database.
          5. Persist job groups to database.

        Args:
            log_directory: Path to the directory containing log files.
            start_time: Optional inclusive lower bound for line filtering.
            end_time: Optional inclusive upper bound for line filtering.
            environment: Deployment environment label.

        Returns:
            A ``PipelineResult`` summarising the run.
        """
        pipeline_start = time.monotonic()
        warnings: List[str] = []
        executions_stored = 0
        errors_stored = 0
        chunks_stored = 0
        analyses_triggered = 0

        # ── Step 1: Stitch ───────────────────────────────────────
        t0 = time.monotonic()
        stitched = self.stitcher.stitch(log_directory, start_time, end_time)
        logger.info(
            "Step 1 Stitch: %d lines from %d files in %.2fs",
            stitched.total_lines,
            len(stitched.source_files),
            time.monotonic() - t0,
        )

        if stitched.total_lines == 0:
            logger.warning("No log lines found in %s — pipeline aborted early.", log_directory)
            return PipelineResult(
                stitched_log=stitched,
                deinterleaved=DeinterleavedResult(),
                job_groups=[],
                warnings=["No log lines found — directory may be empty."],
                duration_seconds=time.monotonic() - pipeline_start,
            )

        # ── Step 2: De-interleave ─────────────────────────────────
        t0 = time.monotonic()
        deinterleaved = self.deinterleaver.process(stitched)
        logger.info(
            "Step 2 De-interleave: %d batches, %.1f%% CID coverage in %.2fs",
            deinterleaved.total_batches_found,
            deinterleaved.cid_coverage_percent,
            time.monotonic() - t0,
        )

        # Collect structural warnings.
        for cid in deinterleaved.batches_with_no_start_marker:
            warnings.append(f"Batch {cid!r} has no start marker.")
        for cid in deinterleaved.batches_with_no_end_marker:
            warnings.append(f"Batch {cid!r} has no end marker.")
        if deinterleaved.cid_coverage_percent < 80.0:
            warnings.append(
                f"CID coverage is low: {deinterleaved.cid_coverage_percent:.1f}% "
                f"(< 80% threshold)."
            )

        # ── Step 3: Group ────────────────────────────────────────
        t0 = time.monotonic()
        job_groups = self.grouper.group(deinterleaved, environment=environment)
        logger.info(
            "Step 3 Group: %d job groups in %.2fs",
            len(job_groups),
            time.monotonic() - t0,
        )

        # ── Step 4: Per-execution preprocessing + storage ─────────
        t0 = time.monotonic()
        for group in job_groups:
            # Persist job group first (so we have the job_id FK).
            job_id = self.repo.upsert_job_group(group)

            for execution in group.executions:
                run_number: int = getattr(execution, "_run_number", 1)
                attempt_type: str = getattr(
                    execution, "_attempt_type", "SCHEDULED"
                )
                # AttemptType enum → value string
                if hasattr(attempt_type, "value"):
                    attempt_type = attempt_type.value

                if execution.total_lines == 0:
                    warnings.append(
                        f"Batch {execution.correlation_id!r} has no lines."
                    )

                # Timestamps warning.
                ts_lines = [
                    ln for ln in execution.lines
                    if ln.parsed_timestamp is not None
                ]
                if not ts_lines and execution.lines:
                    warnings.append(
                        f"Batch {execution.correlation_id!r} has no timestamped lines."
                    )

                # a) Check if this execution already exists (idempotency guard).
                is_new_execution = (
                    self.repo.get_execution_by_cid(execution.correlation_id) is None
                )

                # b) Persist execution FIRST (child tables FK to this row).
                self.repo.upsert_execution(
                    execution=execution,
                    job_id=job_id,
                    run_number=run_number,
                    attempt_type=attempt_type,
                    environment=environment,
                )
                executions_stored += 1

                # Only insert child records on first-time processing.
                if is_new_execution:
                    # c) Aggregate errors → store to error_summary.
                    error_records = self.error_aggregator.aggregate(execution)
                    if error_records:
                        self.repo.store_error_summary(
                            execution.correlation_id,
                            [
                                {
                                    "error_category": r.error_category,
                                    "error_message": r.representative_message,
                                    "count": r.count,
                                    "first_seen": (
                                        r.first_seen.isoformat()
                                        if r.first_seen else None
                                    ),
                                    "last_seen": (
                                        r.last_seen.isoformat()
                                        if r.last_seen else None
                                    ),
                                    "severity": r.severity.value,
                                }
                                for r in error_records
                            ],
                        )
                        errors_stored += len(error_records)

                    # d) Generate metrics (informational — logged, not persisted).
                    _metrics = self.metrics_generator.generate(execution)
                    logger.debug(
                        "Metrics for %s: duration=%.1fs, error_rate=%.1f%%",
                        execution.correlation_id,
                        _metrics.duration_seconds or 0,
                        _metrics.error_rate_percent,
                    )

                    # e) Chunk the log → store to log_chunks.
                    chunks = self.chunker.chunk(execution)
                    for chunk in chunks:
                        self.repo.store_log_chunk(
                            correlation_id=execution.correlation_id,
                            chunk_index=chunk.chunk_index,
                            chunk_type=chunk.chunk_type,
                            source_file=chunk.source_file,
                            start_line=chunk.start_unified_line,
                            end_line=chunk.end_unified_line,
                            content=chunk.content,
                        )
                        chunks_stored += 1
                else:
                    logger.debug(
                        "Execution %s already processed — skipping child inserts.",
                        execution.correlation_id,
                    )

            # ── Step 4b: LLM analysis for this group ────────────────
            if self.analyzer is not None:
                # Collect pre-computed errors and metrics for all
                # executions in this group (built in the loop above).
                group_errors: dict = {}
                group_metrics: dict = {}
                for execution in group.executions:
                    errs = self.error_aggregator.aggregate(execution)
                    met = self.metrics_generator.generate(execution)
                    group_errors[execution.correlation_id] = errs
                    group_metrics[execution.correlation_id] = met

                for execution in group.executions:
                    try:
                        self.analyzer.analyze_execution(
                            execution=execution,
                            errors=group_errors[execution.correlation_id],
                            metrics=group_metrics[execution.correlation_id],
                        )
                        analyses_triggered += 1
                    except Exception as exc:
                        logger.warning(
                            "LLM analysis failed for %s: %s",
                            execution.correlation_id,
                            exc,
                        )

                if group.total_runs > 1:
                    try:
                        self.analyzer.analyze_job_run_group(
                            group=group,
                            all_errors=group_errors,
                            all_metrics=group_metrics,
                        )
                        analyses_triggered += 1
                    except Exception as exc:
                        logger.warning(
                            "LLM group analysis failed for %s: %s",
                            group.job_name,
                            exc,
                        )

        logger.info(
            "Step 4 Preprocess+Store+Analyze: %d executions, %d errors, "
            "%d chunks, %d analyses in %.2fs",
            executions_stored,
            errors_stored,
            chunks_stored,
            analyses_triggered,
            time.monotonic() - t0,
        )

        # ── Step 5: Persist job groups summary ────────────────────
        # (already upserted in the loop above; this step is a no-op
        # but logged for clarity)
        logger.info("Step 5 Job groups: %d groups persisted.", len(job_groups))

        duration = time.monotonic() - pipeline_start
        logger.info(
            "Pipeline complete: %d batches, %d errors, %d chunks in %.2fs",
            deinterleaved.total_batches_found,
            errors_stored,
            chunks_stored,
            duration,
        )

        return PipelineResult(
            stitched_log=stitched,
            deinterleaved=deinterleaved,
            job_groups=job_groups,
            executions_stored=executions_stored,
            errors_stored=errors_stored,
            chunks_stored=chunks_stored,
            analyses_triggered=analyses_triggered,
            duration_seconds=round(duration, 3),
            warnings=warnings,
        )
