"""Repository layer — all SQLite read / write operations.

Provides a ``BatchRepository`` that wraps every database interaction
behind typed methods.  Never exposes raw SQL to callers.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from app.segmentation.job_grouper import JobRunGroup
from app.segmentation.models import BatchExecution
from app.storage.database import DatabaseManager

logger = logging.getLogger("repository")


class BatchRepository:
    """Read / write access to all batch-related tables.

    Usage::

        repo = BatchRepository(db_manager)
        job_id = repo.upsert_job_group(group)
        repo.upsert_execution(execution, job_id, 1, "SCHEDULED", "prod")
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialise with a ``DatabaseManager``.

        Args:
            db_manager: Provides database connections.
        """
        self._db = db_manager

    # ── Connection helpers ──────────────────────────────────────

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close *conn* unless it is a shared in-memory connection."""
        if self._db.db_path != ":memory:":
            conn.close()

    # ── Write Operations ────────────────────────────────────────

    def upsert_job_group(self, group: JobRunGroup) -> int:
        """Insert or replace a job-run group.

        Args:
            group: A ``JobRunGroup`` to persist.

        Returns:
            The ``batch_jobs.id`` of the upserted row.
        """
        conn = self._db.get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO batch_jobs
                    (job_name, job_date, environment, total_runs,
                     successful_runs, failed_runs, final_status,
                     first_run_time, last_run_time, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(job_name, job_date, environment)
                DO UPDATE SET
                    total_runs      = excluded.total_runs,
                    successful_runs = excluded.successful_runs,
                    failed_runs     = excluded.failed_runs,
                    final_status    = excluded.final_status,
                    first_run_time  = excluded.first_run_time,
                    last_run_time   = excluded.last_run_time,
                    updated_at      = CURRENT_TIMESTAMP
                """,
                (
                    group.job_name,
                    group.date.isoformat(),
                    group.environment,
                    group.total_runs,
                    group.successful_runs,
                    group.failed_runs,
                    group.final_status.value,
                    group.first_run_time.isoformat() if group.first_run_time else None,
                    group.last_run_time.isoformat() if group.last_run_time else None,
                ),
            )
            conn.commit()
            job_id = cursor.lastrowid

            # lastrowid is 0 on UPDATE — query to get actual id.
            if not job_id:
                row = conn.execute(
                    """
                    SELECT id FROM batch_jobs
                    WHERE job_name = ? AND job_date = ? AND environment = ?
                    """,
                    (group.job_name, group.date.isoformat(), group.environment),
                ).fetchone()
                job_id = row["id"] if row else 0

            return job_id  # type: ignore[return-value]
        except sqlite3.Error as exc:
            logger.error("upsert_job_group failed: %s", exc)
            raise
        finally:
            self._close(conn)

    def upsert_execution(
        self,
        execution: BatchExecution,
        job_id: int,
        run_number: int,
        attempt_type: str,
        environment: str,
    ) -> None:
        """Insert or ignore a batch execution.

        Duplicate ``correlation_id`` values are silently ignored for
        idempotency.

        Args:
            execution: The ``BatchExecution`` to persist.
            job_id: Foreign key to ``batch_jobs.id``.
            run_number: 1-indexed run number within the job group.
            attempt_type: SCHEDULED / AUTO_RETRY / MANUAL_RETRY.
            environment: Deployment environment label.
        """
        duration = self._calc_duration(execution.start_time, execution.end_time)
        source_files_json = json.dumps(execution.source_files)

        conn = self._db.get_connection()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO batch_executions
                    (correlation_id, job_id, job_name, run_number,
                     attempt_type, environment, status, start_time,
                     end_time, duration_seconds, total_lines,
                     error_count, warn_count, orphan_lines_count,
                     has_start_marker, has_end_marker, source_files)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution.correlation_id,
                    job_id,
                    execution.job_name,
                    run_number,
                    attempt_type,
                    environment,
                    execution.status.value,
                    execution.start_time.isoformat() if execution.start_time else None,
                    execution.end_time.isoformat() if execution.end_time else None,
                    duration,
                    execution.total_lines,
                    execution.error_count,
                    execution.warn_count,
                    execution.orphan_lines_count,
                    execution.has_start_marker,
                    execution.has_end_marker,
                    source_files_json,
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.error("upsert_execution failed: %s", exc)
            raise
        finally:
            self._close(conn)

    def store_error_summary(
        self, correlation_id: str, errors: List[Dict[str, Any]]
    ) -> None:
        """Batch-insert error summaries for a correlation ID.

        Args:
            correlation_id: The batch's CID.
            errors: List of dicts with keys: ``error_category``,
                ``error_message``, ``count``, ``first_seen``,
                ``last_seen``, ``severity``.
        """
        conn = self._db.get_connection()
        try:
            conn.executemany(
                """
                INSERT INTO error_summary
                    (correlation_id, error_category, error_message,
                     count, first_seen, last_seen, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        correlation_id,
                        e.get("error_category", "UNKNOWN"),
                        e.get("error_message", ""),
                        e.get("count", 1),
                        e.get("first_seen"),
                        e.get("last_seen"),
                        e.get("severity", "MEDIUM"),
                    )
                    for e in errors
                ],
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.error("store_error_summary failed: %s", exc)
            raise
        finally:
            self._close(conn)

    def store_log_chunk(
        self,
        correlation_id: str,
        chunk_index: int,
        chunk_type: str,
        source_file: str,
        start_line: int,
        end_line: int,
        content: str,
    ) -> None:
        """Insert a single log chunk.

        Args:
            correlation_id: The batch's CID.
            chunk_index: 0-indexed position of this chunk.
            chunk_type: HEADER / ERROR_CLUSTER / RECOVERY / FOOTER / BODY.
            source_file: Source file basename.
            start_line: Unified line number of the first line.
            end_line: Unified line number of the last line.
            content: The text content of the chunk.
        """
        conn = self._db.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO log_chunks
                    (correlation_id, chunk_index, chunk_type,
                     source_file, start_line, end_line,
                     content, line_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correlation_id,
                    chunk_index,
                    chunk_type,
                    source_file,
                    start_line,
                    end_line,
                    content,
                    end_line - start_line + 1,
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.error("store_log_chunk failed: %s", exc)
            raise
        finally:
            self._close(conn)

    def store_llm_response(
        self,
        correlation_id: str,
        call_type: str,
        request_prompt: str,
        response_raw: str,
        response_parsed: Optional[Dict[str, Any]],
        usage: Dict[str, int],
        prompt_version: str,
    ) -> int:
        """Insert an LLM inference log entry.

        Calculates estimated cost using Claude pricing:
        input $3.00/M tokens, output $15.00/M tokens.

        Args:
            correlation_id: The batch's CID.
            call_type: BATCH_ANALYSIS or QUERY_RESPONSE.
            request_prompt: The full prompt text sent to the LLM.
            response_raw: The raw LLM response text.
            response_parsed: Parsed response dict (or None).
            usage: Dict with ``input_tokens`` and ``output_tokens``.
            prompt_version: Prompt template version string.

        Returns:
            The ``llm_inference_log.id`` of the inserted row.
        """
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = (input_tokens * 3.00 / 1_000_000) + (output_tokens * 15.00 / 1_000_000)

        now = datetime.utcnow().isoformat()
        parsed_json = json.dumps(response_parsed) if response_parsed else None

        conn = self._db.get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO llm_inference_log
                    (correlation_id, call_type, request_prompt,
                     request_model, request_timestamp,
                     response_raw, response_parsed, response_timestamp,
                     input_tokens, output_tokens, estimated_cost_usd,
                     parse_success, prompt_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correlation_id,
                    call_type,
                    request_prompt,
                    "claude-sonnet-4-6",
                    now,
                    response_raw,
                    parsed_json,
                    now,
                    input_tokens,
                    output_tokens,
                    cost,
                    response_parsed is not None,
                    prompt_version,
                ),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            logger.error("store_llm_response failed: %s", exc)
            raise
        finally:
            self._close(conn)

    def mark_execution_analyzed(self, correlation_id: str) -> None:
        """Mark a batch execution as LLM-analyzed.

        Args:
            correlation_id: The batch's CID.
        """
        conn = self._db.get_connection()
        try:
            conn.execute(
                """
                UPDATE batch_executions
                SET llm_analyzed = TRUE
                WHERE correlation_id = ?
                """,
                (correlation_id,),
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.error("mark_execution_analyzed failed: %s", exc)
            raise
        finally:
            self._close(conn)

    # ── Read Operations ─────────────────────────────────────────

    def get_executions_by_date(
        self, target_date: date, environment: str = "prod"
    ) -> List[sqlite3.Row]:
        """Return all executions for a given date and environment.

        Args:
            target_date: Calendar date to query.
            environment: Deployment environment filter.

        Returns:
            List of rows from ``batch_executions``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT * FROM batch_executions
                WHERE DATE(start_time) = ? AND environment = ?
                ORDER BY start_time ASC
                """,
                (target_date.isoformat(), environment),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_execution_by_cid(
        self, correlation_id: str
    ) -> Optional[sqlite3.Row]:
        """Return a single execution by correlation ID.

        Args:
            correlation_id: The batch's CID.

        Returns:
            A row from ``batch_executions``, or ``None``.
        """
        conn = self._db.get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM batch_executions WHERE correlation_id = ?",
                (correlation_id,),
            ).fetchone()
            return row
        finally:
            self._close(conn)

    def get_job_groups_by_date(
        self, target_date: date, environment: str = "prod"
    ) -> List[sqlite3.Row]:
        """Return all job groups for a given date and environment.

        Args:
            target_date: Calendar date to query.
            environment: Deployment environment filter.

        Returns:
            List of rows from ``batch_jobs``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT * FROM batch_jobs
                WHERE job_date = ? AND environment = ?
                ORDER BY job_name ASC
                """,
                (target_date.isoformat(), environment),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_error_summary_for_cid(
        self, correlation_id: str
    ) -> List[sqlite3.Row]:
        """Return all error summaries for a correlation ID.

        Args:
            correlation_id: The batch's CID.

        Returns:
            List of rows from ``error_summary``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT * FROM error_summary
                WHERE correlation_id = ?
                ORDER BY count DESC
                """,
                (correlation_id,),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_llm_response_for_cid(
        self,
        correlation_id: str,
        call_type: str = "BATCH_ANALYSIS",
    ) -> Optional[sqlite3.Row]:
        """Return the most recent LLM response for a CID.

        Used for idempotency checks before making a new LLM call.

        Args:
            correlation_id: The batch's CID.
            call_type: BATCH_ANALYSIS or QUERY_RESPONSE.

        Returns:
            Most recent row from ``llm_inference_log``, or ``None``.
        """
        conn = self._db.get_connection()
        try:
            row = conn.execute(
                """
                SELECT * FROM llm_inference_log
                WHERE correlation_id = ? AND call_type = ?
                ORDER BY request_timestamp DESC
                LIMIT 1
                """,
                (correlation_id, call_type),
            ).fetchone()
            return row
        finally:
            self._close(conn)

    def get_cost_summary_by_date(
        self, days: int = 7
    ) -> List[sqlite3.Row]:
        """Aggregate LLM costs over the last *days* days.

        Args:
            days: Number of days to look back.

        Returns:
            List of rows with columns: ``call_date``, ``call_count``,
            ``total_input_tokens``, ``total_output_tokens``, ``total_cost``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT
                    DATE(request_timestamp) AS call_date,
                    COUNT(*)               AS call_count,
                    SUM(input_tokens)      AS total_input_tokens,
                    SUM(output_tokens)     AS total_output_tokens,
                    SUM(estimated_cost_usd) AS total_cost
                FROM llm_inference_log
                WHERE request_timestamp >= DATE('now', ?)
                GROUP BY DATE(request_timestamp)
                ORDER BY call_date DESC
                """,
                (f"-{days} days",),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_chunks_for_cid(
        self, correlation_id: str
    ) -> List[sqlite3.Row]:
        """Return all log chunks for a correlation ID.

        Args:
            correlation_id: The batch's CID.

        Returns:
            Ordered list of rows from ``log_chunks``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT * FROM log_chunks
                WHERE correlation_id = ?
                ORDER BY chunk_index ASC
                """,
                (correlation_id,),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_executions_by_date_range(
        self,
        start_date: date,
        end_date: date,
        environment: str = "prod",
    ) -> List[sqlite3.Row]:
        """Return all executions in a date range for an environment.

        Args:
            start_date: Inclusive lower bound (calendar date).
            end_date: Inclusive upper bound (calendar date).
            environment: Deployment environment filter.

        Returns:
            List of rows from ``batch_executions``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT be.*,
                       bj.job_name
                FROM batch_executions be
                LEFT JOIN batch_jobs bj ON be.job_id = bj.id
                WHERE DATE(be.start_time) BETWEEN ? AND ?
                  AND be.environment = ?
                ORDER BY be.start_time ASC
                """,
                (start_date.isoformat(), end_date.isoformat(), environment),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_executions_by_job_and_date_range(
        self,
        job_name: str,
        start_date: date,
        end_date: date,
        environment: str = "prod",
    ) -> List[sqlite3.Row]:
        """Return executions for a specific job in a date range.

        Args:
            job_name: Normalised job name.
            start_date: Inclusive lower bound.
            end_date: Inclusive upper bound.
            environment: Deployment environment filter.

        Returns:
            List of rows from ``batch_executions`` joined to
            ``batch_jobs``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT be.*,
                       bj.job_name,
                       DATE(be.start_time) AS run_date
                FROM batch_executions be
                JOIN batch_jobs bj ON be.job_id = bj.id
                WHERE bj.job_name = ?
                  AND DATE(be.start_time) BETWEEN ? AND ?
                  AND be.environment = ?
                ORDER BY be.start_time ASC
                """,
                (
                    job_name,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    environment,
                ),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_all_job_names(self) -> List[sqlite3.Row]:
        """Return distinct job names stored in the database.

        Returns:
            List of rows with column ``job_name``.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT job_name
                FROM batch_jobs
                ORDER BY job_name ASC
                """
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    def get_error_summaries_by_date_range(
        self,
        start_date: date,
        end_date: date,
        environment: str = "prod",
    ) -> List[sqlite3.Row]:
        """Return all error summaries for a date range and environment.

        Joins ``error_summary`` → ``batch_executions`` → ``batch_jobs``
        so each row includes ``job_name`` and ``run_number``.

        Args:
            start_date: Inclusive lower bound (calendar date).
            end_date: Inclusive upper bound (calendar date).
            environment: Deployment environment filter.

        Returns:
            List of joined rows.
        """
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT
                    es.*,
                    bj.job_name,
                    be.run_number
                FROM error_summary es
                JOIN batch_executions be
                  ON es.correlation_id = be.correlation_id
                JOIN batch_jobs bj
                  ON be.job_id = bj.id
                WHERE DATE(be.start_time) BETWEEN ? AND ?
                  AND be.environment = ?
                ORDER BY es.count DESC
                """,
                (start_date.isoformat(), end_date.isoformat(), environment),
            ).fetchall()
            return rows
        finally:
            self._close(conn)

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _calc_duration(
        start: Optional[datetime], end: Optional[datetime]
    ) -> Optional[float]:
        """Return duration in seconds between *start* and *end*.

        Returns ``None`` if either timestamp is missing.
        """
        if start is None or end is None:
            return None
        return (end - start).total_seconds()

