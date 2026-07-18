"""Token-efficient batch log digest builder.

Compresses a ``BatchExecution`` (or a list of executions for the
same job) into a structured text digest that stays under ~3000 tokens
for the LLM context window.

Key design choices:
- Always include STARTUP (first 15 lines) and FINAL (last 15 lines).
- Cap error/warn lines at ``Settings.max_error_lines_in_digest``.
- Multi-run digests summarise middle runs if total is too long.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.config.constants import ERROR_LEVELS, WARN_LEVELS, LogLevel
from app.config.settings import Settings
from app.preprocessing.error_aggregator import ErrorRecord
from app.preprocessing.metrics_generator import BatchMetrics
from app.segmentation.models import BatchExecution, ParsedLogLine

logger = logging.getLogger("digest_builder")

# Maximum characters in the error section before trimming.
_MAX_ERROR_SECTION_CHARS = 8_000

# Lines kept from startup and final sections — never trimmed.
_STARTUP_LINES = 15
_FINAL_LINES = 15

# Rough character budget for a multi-run digest.
_MULTI_RUN_MAX_CHARS = 18_000


class DigestBuilder:
    """Build token-efficient text digests from batch execution data.

    Usage::

        builder = DigestBuilder(settings)
        digest = builder.build_single_execution_digest(execution, errors, metrics)
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Used for ``max_error_lines_in_digest``.
        """
        self._settings = settings

    # ── Public API ───────────────────────────────────────────────

    def build_single_execution_digest(
        self,
        execution: BatchExecution,
        errors: List[ErrorRecord],
        metrics: BatchMetrics,
    ) -> str:
        """Build a structured digest for a single batch execution.

        Sections (in order, STARTUP and FINAL are never trimmed):
          1. Header metadata
          2. Error summary table
          3. Startup sequence (first 15 lines)
          4. Error and warning lines (capped)
          5. Final sequence (last 15 lines)

        Args:
            execution: The batch to summarise.
            errors: Pre-aggregated error records.
            metrics: Pre-computed metrics.

        Returns:
            A structured text digest string.
        """
        lines = execution.lines
        total = len(lines)

        parts: List[str] = []

        # ── Header ──────────────────────────────────────────────
        parts.append("=== BATCH EXECUTION DIGEST ===")
        parts.append(f"Correlation ID : {execution.correlation_id}")
        parts.append(f"Job Name       : {execution.job_name or 'unknown'}")
        parts.append(f"Environment    : {getattr(execution, 'environment', 'unknown')}")
        parts.append(f"Status         : {execution.status.value if hasattr(execution.status, 'value') else execution.status}")
        parts.append(f"Start Time     : {execution.start_time or 'N/A'}")
        parts.append(f"End Time       : {execution.end_time or 'N/A'}")
        parts.append(f"Duration       : {f'{metrics.duration_seconds:.1f}s' if metrics.duration_seconds is not None else 'N/A'}")
        parts.append(f"Total Lines    : {total}")
        parts.append(f"Error Count    : {execution.error_count}")
        parts.append(f"Warning Count  : {execution.warn_count}")
        if metrics.estimated_record_count is not None:
            parts.append(f"Records        : {metrics.estimated_record_count:,}")
        if metrics.error_rate_percent:
            parts.append(f"Error Rate     : {metrics.error_rate_percent:.1f}%")

        # ── Error summary table ──────────────────────────────────
        parts.append("\n--- ERROR SUMMARY ---")
        if errors:
            for r in errors:
                sev = r.severity.value if hasattr(r.severity, "value") else r.severity
                parts.append(
                    f"  [{sev}] {r.error_category}: {r.count} occurrence(s) | {r.representative_message[:120]}"
                )
        else:
            parts.append("  (no categorised errors)")

        # ── Startup sequence ─────────────────────────────────────
        parts.append("\n--- STARTUP SEQUENCE (first 15 lines) ---")
        startup = lines[:_STARTUP_LINES]
        for ln in startup:
            parts.append(f"  {ln.raw}")

        # ── Error and warning lines ──────────────────────────────
        parts.append(f"\n--- ERROR AND WARNING LINES (up to {self._settings.max_error_lines_in_digest}) ---")
        ew_lines = self._collect_error_warn_lines(lines)
        cap = self._settings.max_error_lines_in_digest
        if len(ew_lines) > cap:
            ew_lines = ew_lines[:cap]
            parts.append(f"  [truncated to {cap} lines]")
        for ln in ew_lines:
            parts.append(f"  {ln.raw}")
        if not ew_lines:
            parts.append("  (none)")

        # ── Final sequence ───────────────────────────────────────
        parts.append("\n--- FINAL SEQUENCE (last 15 lines) ---")
        final_start = max(0, total - _FINAL_LINES)
        final = lines[final_start:]
        for ln in final:
            parts.append(f"  {ln.raw}")

        parts.append("\n=== END DIGEST ===")

        digest = "\n".join(parts)

        # Trim error section if digest is excessively long.
        if len(digest) > _MAX_ERROR_SECTION_CHARS * 2:
            digest = self._trim_error_section(digest)

        return digest

    def build_multi_run_digest(
        self,
        executions: List[BatchExecution],
        all_errors: Dict[str, List[ErrorRecord]],
        all_metrics: Dict[str, BatchMetrics],
    ) -> str:
        """Build a multi-run digest for a job with repeated executions.

        Keeps first and last run fully detailed; summarises middle runs
        when the total character budget would be exceeded.

        Args:
            executions: Executions sorted by run_number ascending.
            all_errors: ``{correlation_id: [ErrorRecord, ...], ...}``.
            all_metrics: ``{correlation_id: BatchMetrics, ...}``.

        Returns:
            A structured text digest string.
        """
        if not executions:
            return "=== MULTI-RUN DIGEST: no executions provided ==="

        # Sort by run number.
        sorted_execs = sorted(
            executions,
            key=lambda e: getattr(e, "_run_number", 0),
        )
        last = sorted_execs[-1]
        last_status = last.status.value if hasattr(last.status, "value") else str(last.status)

        parts: List[str] = []
        parts.append("=== MULTI-RUN BATCH DIGEST ===")
        parts.append(f"Job Name     : {sorted_execs[0].job_name or 'unknown'}")
        parts.append(f"Total Runs   : {len(sorted_execs)}")
        parts.append(f"Final Status : {last_status}")
        parts.append("")

        # Build per-run blocks.
        run_blocks: List[str] = []
        for exec_ in sorted_execs:
            run_blocks.append(
                self._build_run_block(exec_, all_errors, all_metrics)
            )

        # Check total budget.
        total_chars = sum(len(b) for b in run_blocks)
        if total_chars <= _MULTI_RUN_MAX_CHARS or len(run_blocks) <= 2:
            parts.extend(run_blocks)
        else:
            # Keep first and last full; summarise middle.
            parts.append(run_blocks[0])
            middle = run_blocks[1:-1]
            parts.append(f"\n--- {len(middle)} MIDDLE RUN(S) SUMMARISED ---")
            for i, exec_ in enumerate(sorted_execs[1:-1]):
                run_num = getattr(exec_, "_run_number", i + 2)
                status = exec_.status.value if hasattr(exec_.status, "value") else str(exec_.status)
                errs = all_errors.get(exec_.correlation_id, [])
                err_summary = (
                    ", ".join(f"{e.error_category}×{e.count}" for e in errs[:3])
                    or "no errors"
                )
                parts.append(
                    f"  RUN {run_num}: {status} | errors: {err_summary}"
                )
            parts.append(run_blocks[-1])

        parts.append("\n=== END MULTI-RUN DIGEST ===")
        return "\n".join(parts)

    # ── Private helpers ──────────────────────────────────────────

    def _build_run_block(
        self,
        execution: BatchExecution,
        all_errors: Dict[str, List[ErrorRecord]],
        all_metrics: Dict[str, BatchMetrics],
    ) -> str:
        """Build a compact per-run block for the multi-run digest."""
        run_num = getattr(execution, "_run_number", "?")
        attempt = getattr(execution, "_attempt_type", "SCHEDULED")
        if hasattr(attempt, "value"):
            attempt = attempt.value
        status = execution.status.value if hasattr(execution.status, "value") else str(execution.status)
        m = all_metrics.get(execution.correlation_id)
        duration_str = f"{m.duration_seconds:.1f}s" if (m and m.duration_seconds is not None) else "N/A"

        lines_: List[str] = []
        lines_.append(f"\n--- RUN {run_num} [{attempt}] — {status} — {duration_str} ---")
        lines_.append(f"CID: {execution.correlation_id} | Lines: {execution.total_lines}")

        errors = all_errors.get(execution.correlation_id, [])
        if errors:
            err_oneliner = " | ".join(
                f"{e.error_category}×{e.count} [{e.severity.value if hasattr(e.severity, 'value') else e.severity}]"
                for e in errors[:4]
            )
            lines_.append(f"Errors: {err_oneliner}")
        else:
            lines_.append("Errors: none")

        # 5 error/warn lines.
        ew = self._collect_error_warn_lines(execution.lines)[:5]
        if ew:
            lines_.append("Key lines:")
            for ln in ew:
                lines_.append(f"  {ln.raw}")

        # Last 3 lines.
        if execution.lines:
            lines_.append("Final lines:")
            for ln in execution.lines[-3:]:
                lines_.append(f"  {ln.raw}")

        return "\n".join(lines_)

    @staticmethod
    def _collect_error_warn_lines(
        lines: List[ParsedLogLine],
    ) -> List[ParsedLogLine]:
        """Return only ERROR/WARN lines from *lines*."""
        result: List[ParsedLogLine] = []
        for ln in lines:
            try:
                lvl = LogLevel(ln.level)
                if lvl in ERROR_LEVELS or lvl in WARN_LEVELS:
                    result.append(ln)
            except ValueError:
                pass
        return result

    @staticmethod
    def _trim_error_section(digest: str) -> str:
        """Trim the ERROR AND WARNING LINES section to the budget."""
        start_marker = "--- ERROR AND WARNING LINES"
        end_marker = "--- FINAL SEQUENCE"
        s_idx = digest.find(start_marker)
        e_idx = digest.find(end_marker)
        if s_idx == -1 or e_idx == -1 or s_idx >= e_idx:
            return digest

        header = digest[:s_idx]
        error_block = digest[s_idx:e_idx]
        footer = digest[e_idx:]

        # Keep up to budget from the error block.
        if len(error_block) > _MAX_ERROR_SECTION_CHARS:
            error_block = error_block[:_MAX_ERROR_SECTION_CHARS]
            error_block += "\n  ... [further error lines omitted for brevity]\n"

        return header + error_block + footer
