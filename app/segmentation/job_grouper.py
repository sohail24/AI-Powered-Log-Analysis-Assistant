"""Job grouper — groups multiple executions of the same job.

Takes the flat ``dict[str, BatchExecution]`` from the de-interleaver
and groups executions by ``(job_name, date)``, assigns run numbers
and attempt types (SCHEDULED / AUTO_RETRY / MANUAL_RETRY).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from app.config.constants import AttemptType, BatchStatus
from app.config.settings import Settings
from app.segmentation.models import BatchExecution, DeinterleavedResult

logger = logging.getLogger("job_grouper")


@dataclass
class JobRunGroup:
    """All executions of a single job on a single day.

    Attributes:
        job_name: Normalised job name.
        date: Calendar date of the runs.
        environment: Deployment environment (prod / staging / uat).
        executions: Chronologically sorted list of BatchExecution.
        total_runs: Number of executions in this group.
        successful_runs: Executions that ended with SUCCESS.
        failed_runs: Executions that ended with FAILED.
        final_status: Status of the *last* execution.
        first_run_time: Start time of the first execution.
        last_run_time: Start time of the last execution.
    """

    job_name: str
    date: date
    environment: str
    executions: List[BatchExecution] = field(default_factory=list)
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    final_status: BatchStatus = BatchStatus.UNKNOWN
    first_run_time: Optional[datetime] = None
    last_run_time: Optional[datetime] = None


# Pattern to strip special characters (keep alphanumerics, hyphens, underscores).
_STRIP_SPECIAL = re.compile(r"[^\w\-]", re.ASCII)


class JobGrouper:
    """Group BatchExecutions by (job_name, date) and assign run metadata.

    Usage::

        grouper = JobGrouper(settings)
        groups = grouper.group(deinterleaved_result, environment="prod")
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Application-wide Settings instance.
        """
        self._settings = settings

    # ── Public API ──────────────────────────────────────────────

    def group(
        self,
        result: DeinterleavedResult,
        environment: str = "prod",
    ) -> List[JobRunGroup]:
        """Group batch executions by normalised job name and date.

        For each group the executions are sorted chronologically,
        assigned sequential ``run_number`` values and classified as
        ``SCHEDULED``, ``AUTO_RETRY``, or ``MANUAL_RETRY``.

        Args:
            result: Output of the de-interleaving engine.
            environment: Deployment environment label.

        Returns:
            List of ``JobRunGroup``, one per (job_name, date) pair.
        """
        # Step 1–2: normalise names and bucket by (name, date).
        buckets: Dict[Tuple[str, date], List[BatchExecution]] = {}

        for batch in result.batches.values():
            norm_name = self._normalize_job_name(batch.job_name)
            run_date = (
                batch.start_time.date()
                if batch.start_time is not None
                else date.today()
            )
            key = (norm_name, run_date)
            buckets.setdefault(key, []).append(batch)

        groups: List[JobRunGroup] = []

        for (norm_name, run_date), executions in buckets.items():
            # Step 3: sort by start_time ascending.
            executions.sort(
                key=lambda b: (
                    b.start_time if b.start_time is not None else datetime.max
                )
            )

            # Step 4–5: assign run_number and attempt_type.
            for idx, exe in enumerate(executions):
                run_number = idx + 1
                exe._run_number = run_number  # type: ignore[attr-defined]

                if run_number == 1:
                    exe._attempt_type = AttemptType.SCHEDULED  # type: ignore[attr-defined]
                else:
                    prev = executions[idx - 1]
                    gap = self._compute_gap_minutes(prev, exe)
                    if gap is None:
                        exe._attempt_type = AttemptType.UNKNOWN  # type: ignore[attr-defined]
                    elif gap < self._settings.auto_retry_gap_minutes:
                        exe._attempt_type = AttemptType.AUTO_RETRY  # type: ignore[attr-defined]
                    else:
                        exe._attempt_type = AttemptType.MANUAL_RETRY  # type: ignore[attr-defined]

            # Step 6: compute summary fields.
            successful = sum(
                1 for e in executions if e.status == BatchStatus.SUCCESS
            )
            failed = sum(
                1 for e in executions if e.status == BatchStatus.FAILED
            )

            first_time = executions[0].start_time if executions else None
            last_time = executions[-1].start_time if executions else None
            final = executions[-1].status if executions else BatchStatus.UNKNOWN

            group = JobRunGroup(
                job_name=norm_name,
                date=run_date,
                environment=environment,
                executions=executions,
                total_runs=len(executions),
                successful_runs=successful,
                failed_runs=failed,
                final_status=final,
                first_run_time=first_time,
                last_run_time=last_time,
            )
            groups.append(group)

        logger.info(
            "Grouped %d executions into %d job-run groups",
            len(result.batches),
            len(groups),
        )
        return groups

    def _compute_gap_minutes(
        self,
        prev_execution: BatchExecution,
        curr_execution: BatchExecution,
    ) -> Optional[float]:
        """Return the gap in minutes between *prev* end and *curr* start.

        Args:
            prev_execution: The preceding execution.
            curr_execution: The current execution.

        Returns:
            Gap in minutes, or ``None`` if either timestamp is missing.
        """
        if prev_execution.end_time is None or curr_execution.start_time is None:
            return None
        delta = curr_execution.start_time - prev_execution.end_time
        return delta.total_seconds() / 60.0

    @staticmethod
    def _normalize_job_name(raw_name: str) -> str:
        """Normalise a raw job name.

        Lower-case, strip whitespace, collapse whitespace sequences
        to underscores, and remove special characters except hyphens
        and underscores.

        Args:
            raw_name: The raw job name string.

        Returns:
            Normalised job name.
        """
        name = raw_name.strip().lower()
        # Collapse whitespace sequences to single underscore.
        name = re.sub(r"\s+", "_", name)
        # Remove special chars (keep word chars and hyphens).
        name = _STRIP_SPECIAL.sub("", name)
        return name
