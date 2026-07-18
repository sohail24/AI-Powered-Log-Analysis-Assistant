"""De-interleaving engine for batch log segmentation.

Takes a unified ``StitchedLog`` and splits it into per-batch
``BatchExecution`` objects keyed by correlation ID. Orphan lines
(no CID) are attributed to the next CID that appears, using a
configurable rolling window.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.config.constants import (
    BatchStatus,
    ERROR_LEVELS,
    LogLevel,
    WARN_LEVELS,
)
from app.config.settings import Settings
from app.ingestion.models import StitchedLog
from app.segmentation.extractor import LineParser
from app.segmentation.models import (
    BatchExecution,
    DeinterleavedResult,
    ParsedLogLine,
)

logger = logging.getLogger("deinterleaver")


class DeinterleavingEngine:
    """Split an interleaved log into clean per-batch executions.

    Processing strategy:
      - Lines **with** a CID are added to their ``BatchExecution``.
      - Lines **without** a CID enter a rolling orphan window. When the
        next CID-bearing line arrives, the entire window is flushed
        into that batch as attributed orphans.
      - After all lines are consumed, any remaining orphans are moved
        to the ``unattributed_lines`` list.

    Usage::

        engine = DeinterleavingEngine(settings)
        result = engine.process(stitched_log)
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the engine with application settings.

        Args:
            settings: Application-wide Settings instance.
        """
        self._settings = settings
        self._parser = LineParser(settings)

        # ── Internal state (reset on every process() call) ──────
        self._batches: Dict[str, BatchExecution] = {}
        self._orphan_window: List[ParsedLogLine] = []
        self._unattributed: List[ParsedLogLine] = []

    # ── Public API ──────────────────────────────────────────────

    def process(self, stitched_log: StitchedLog) -> DeinterleavedResult:
        """De-interleave a stitched log into per-batch executions.

        Iterates every ``RawLogLine`` in order, parsing it and routing
        it to the appropriate ``BatchExecution`` or orphan buffer.

        Args:
            stitched_log: The merged, chronologically sorted log.

        Returns:
            A ``DeinterleavedResult`` with all batches, orphans, and
            coverage statistics.
        """
        # Reset state so the engine can be reused.
        self._batches = {}
        self._orphan_window = []
        self._unattributed = []

        for raw_line in stitched_log.lines:
            parsed = self._parser.parse_line(raw_line)

            if parsed.correlation_id is not None:
                cid = parsed.correlation_id
                parsed.is_orphan = False

                # Create batch on first sight of this CID.
                if cid not in self._batches:
                    self._create_batch(cid, parsed)

                # Flush any pending orphans into this batch.
                self._flush_orphans(cid)

                # Add the line itself.
                self._update_batch(self._batches[cid], parsed)
            else:
                # No CID — buffer as orphan.
                self._orphan_window.append(parsed)

                # Evict oldest if window exceeds configured size.
                if len(self._orphan_window) > self._settings.orphan_window_size:
                    evicted = self._orphan_window.pop(0)
                    self._unattributed.append(evicted)

        # End-of-stream: flush remaining orphans to unattributed.
        self._unattributed.extend(self._orphan_window)
        self._orphan_window = []

        # Finalise every batch (sort, counts, status).
        for batch in self._batches.values():
            self._finalize_batch(batch)

        return self._build_result(len(stitched_log.lines))

    # ── Private helpers ─────────────────────────────────────────

    def _flush_orphans(self, target_cid: str) -> None:
        """Move all lines in the orphan window into *target_cid* batch.

        Each orphan line is annotated with ``is_orphan = True`` and
        ``orphan_attributed_to = target_cid``.
        """
        batch = self._batches[target_cid]
        for orphan in self._orphan_window:
            orphan.is_orphan = True
            orphan.orphan_attributed_to = target_cid
            batch.lines.append(orphan)
        self._orphan_window = []

    def _create_batch(
        self, cid: str, first_line: ParsedLogLine
    ) -> BatchExecution:
        """Create a new ``BatchExecution`` for *cid*.

        Args:
            cid: Correlation ID.
            first_line: The first parsed line for this CID.

        Returns:
            The newly created ``BatchExecution``.
        """
        job_name = self._parser.extract_job_name(first_line) or "unknown_job"

        batch = BatchExecution(
            correlation_id=cid,
            lines=[],
            source_files=[],
            start_time=first_line.parsed_timestamp,
            end_time=first_line.parsed_timestamp,
            status=BatchStatus.UNKNOWN,
            job_name=job_name,
            error_count=0,
            warn_count=0,
            total_lines=0,
            orphan_lines_count=0,
            has_start_marker=False,
            has_end_marker=False,
        )
        self._batches[cid] = batch
        logger.info("New batch created: CID=%s, job=%s", cid, job_name)
        return batch

    def _update_batch(
        self, batch: BatchExecution, line: ParsedLogLine
    ) -> None:
        """Append *line* to *batch* and update running metadata.

        Updates end_time, source_files, error/warn counts, and
        start/end marker flags.
        """
        batch.lines.append(line)

        # Update end_time.
        if line.parsed_timestamp is not None:
            if batch.end_time is None or line.parsed_timestamp > batch.end_time:
                batch.end_time = line.parsed_timestamp

        # Update start_time if earlier.
        if line.parsed_timestamp is not None:
            if batch.start_time is None or line.parsed_timestamp < batch.start_time:
                batch.start_time = line.parsed_timestamp

        # Track source files.
        if line.source_file and line.source_file not in batch.source_files:
            batch.source_files.append(line.source_file)

        # Count errors and warnings using the normalised level string.
        level_str = line.level
        try:
            level_enum = LogLevel(level_str)
            if level_enum in ERROR_LEVELS:
                batch.error_count += 1
            elif level_enum in WARN_LEVELS:
                batch.warn_count += 1
        except ValueError:
            pass  # "UNKNOWN" or unrecognised level — skip.

        # Detect job-start marker.
        if self._parser.detect_job_start(line):
            batch.has_start_marker = True
            # Update job_name if still default.
            if batch.job_name == "unknown_job":
                name = self._parser.extract_job_name(line)
                if name:
                    batch.job_name = name

        # Detect job-end marker.
        end_status = self._parser.detect_job_end(line)
        if end_status is not None:
            batch.has_end_marker = True
            # Store the status hint for _finalize_batch.
            batch._status_hint = end_status  # type: ignore[attr-defined]

    def _finalize_batch(self, batch: BatchExecution) -> None:
        """Sort lines, compute final counts, and determine status.

        Status priority:
          1. End-marker status = SUCCESS → ``SUCCESS``
          2. Any FATAL / CRITICAL line → ``FAILED``
          3. End-marker status = FAILED / ERROR → ``FAILED``
          4. Errors present + end marker → ``PARTIAL``
          5. Errors present + no end marker → ``UNKNOWN``
          6. End marker + no errors → ``SUCCESS``
          7. Default → ``UNKNOWN``
        """
        # Sort lines: timestamped first (by ts), then non-timestamped.
        batch.lines.sort(
            key=lambda ln: (
                0 if ln.parsed_timestamp is not None else 1,
                ln.parsed_timestamp or "",
                ln.unified_line_number,
            )
        )

        batch.total_lines = len(batch.lines)
        batch.orphan_lines_count = sum(
            1 for ln in batch.lines if ln.is_orphan
        )

        # ── Determine status ───────────────────────────────────
        status_hint: Optional[str] = getattr(batch, "_status_hint", None)

        has_fatal = any(
            ln.level in {"FATAL", "CRITICAL"} for ln in batch.lines
        )

        if status_hint in {"SUCCESS", "OK"} and not has_fatal:
            batch.status = BatchStatus.SUCCESS
        elif has_fatal:
            batch.status = BatchStatus.FAILED
        elif status_hint in {"FAILED", "ERROR"}:
            batch.status = BatchStatus.FAILED
        elif batch.error_count > 0 and batch.has_end_marker:
            batch.status = BatchStatus.PARTIAL
        elif batch.error_count > 0 and not batch.has_end_marker:
            batch.status = BatchStatus.UNKNOWN
        elif batch.has_end_marker and batch.error_count == 0:
            batch.status = BatchStatus.SUCCESS
        else:
            batch.status = BatchStatus.UNKNOWN

    def _build_result(
        self, total_lines: int
    ) -> DeinterleavedResult:
        """Assemble the final ``DeinterleavedResult``.

        Computes coverage percentage and identifies batches missing
        start / end markers.

        Args:
            total_lines: Total lines fed into the engine.

        Returns:
            Fully populated ``DeinterleavedResult``.
        """
        lines_with_cid = sum(
            ln.total_lines - ln.orphan_lines_count
            for ln in self._batches.values()
        )
        coverage = (lines_with_cid / total_lines * 100) if total_lines else 0.0

        no_start = [
            cid for cid, b in self._batches.items() if not b.has_start_marker
        ]
        no_end = [
            cid for cid, b in self._batches.items() if not b.has_end_marker
        ]

        result = DeinterleavedResult(
            batches=dict(self._batches),
            unattributed_lines=list(self._unattributed),
            total_lines_processed=total_lines,
            total_batches_found=len(self._batches),
            batches_with_no_start_marker=no_start,
            batches_with_no_end_marker=no_end,
            cid_coverage_percent=round(coverage, 2),
        )

        logger.info(
            "De-interleaving complete: %d batches, %d unattributed lines, "
            "%.1f%% CID coverage",
            result.total_batches_found,
            len(result.unattributed_lines),
            result.cid_coverage_percent,
        )
        return result
