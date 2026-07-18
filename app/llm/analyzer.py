"""LLM batch analyzer — makes Anthropic API calls and persists results.

One call per correlation ID per call_type.  Idempotency is enforced
by checking the llm_inference_log table before every call.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from app.config.constants import Severity
from app.config.settings import Settings
from app.llm.client import LLMClient, get_llm_client
from app.llm.digest_builder import DigestBuilder
from app.llm.models import (
    BatchAnalysisResponse,
    ErrorCategory,
    EvidenceAnchor,
    Recommendation,
)
from app.llm.prompts import MULTI_RUN_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.preprocessing.error_aggregator import ErrorRecord
from app.preprocessing.metrics_generator import BatchMetrics
from app.segmentation.job_grouper import JobRunGroup
from app.segmentation.models import BatchExecution
from app.storage.repository import BatchRepository

logger = logging.getLogger("analyzer")

# Regex to strip optional ```json … ``` fences from LLM output.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Keys that must be present in a valid parsed response.
_REQUIRED_KEYS = {"summary", "business_impact", "retry_recommended"}


class BatchAnalyzer:
    """Orchestrates LLM calls for batch execution analysis.

    Usage::

        analyzer = BatchAnalyzer(settings, repo)
        response = analyzer.analyze_execution(execution, errors, metrics)
    """

    def __init__(self, settings: Settings, repo: BatchRepository) -> None:
        """Initialise with application settings and repository.

        Args:
            settings: Provides API key(s), model name, and prompt version.
            repo: Used for idempotency checks and persisting results.
        """
        self._settings = settings
        self._repo = repo
        self._digest_builder = DigestBuilder(settings)
        self._client: Optional[LLMClient] = get_llm_client(settings)

    # ── Public API ───────────────────────────────────────────────

    def analyze_execution(
        self,
        execution: BatchExecution,
        errors: List[ErrorRecord],
        metrics: BatchMetrics,
        force_reanalyze: bool = False,
    ) -> BatchAnalysisResponse:
        """Analyse a single batch execution, with idempotency guard.

        Step 1: Check cache — return stored result if present.
        Step 2: Build digest.
        Step 3: Call LLM.
        Step 4: Parse response.
        Step 5: Persist to DB.
        Step 6: Return ``BatchAnalysisResponse``.

        Args:
            execution: The batch execution to analyse.
            errors: Pre-aggregated error records.
            metrics: Pre-computed metrics.
            force_reanalyze: Skip idempotency check when ``True``.

        Returns:
            Populated ``BatchAnalysisResponse``.
        """
        cid = execution.correlation_id

        # Step 1 — idempotency.
        if not force_reanalyze:
            cached = self._repo.get_llm_response_for_cid(cid, "BATCH_ANALYSIS")
            if cached is not None:
                logger.debug("Returning cached analysis for %s", cid)
                parsed = self._load_cached_parsed(cached)
                raw = cached["response_raw"] or ""
                return self._build_response(parsed, raw)

        # Step 2 — digest.
        digest = self._digest_builder.build_single_execution_digest(
            execution, errors, metrics
        )

        # Step 3 — API call.
        raw_text, usage = self._call_llm(SYSTEM_PROMPT, digest)

        # Step 4 — parse.
        parsed = self._safe_parse(raw_text)

        # Step 5 — persist.
        self._persist(
            cid=cid,
            call_type="BATCH_ANALYSIS",
            digest=digest,
            raw_text=raw_text,
            parsed=parsed,
            usage=usage,
        )
        self._repo.mark_execution_analyzed(cid)
        logger.info("Analysis complete for %s (parse_success=%s)", cid, parsed is not None)

        # Step 6 — return.
        return self._build_response(parsed, raw_text)

    def analyze_job_run_group(
        self,
        group: JobRunGroup,
        all_errors: Dict[str, List[ErrorRecord]],
        all_metrics: Dict[str, BatchMetrics],
        force_reanalyze: bool = False,
    ) -> BatchAnalysisResponse:
        """Analyse all executions in a multi-run job group.

        Only meaningful when ``group.total_runs > 1``.  Uses the
        multi-run prompt and stores with a synthetic CID derived from
        the last execution's CID.

        Args:
            group: The job run group to analyse.
            all_errors: Mapping of CID → error records.
            all_metrics: Mapping of CID → metrics.
            force_reanalyze: Skip cache when ``True``.

        Returns:
            Populated ``BatchAnalysisResponse``.
        """
        if not group.executions:
            return self._empty_response("No executions in group")

        last_cid = group.executions[-1].correlation_id
        group_cid = f"{last_cid}_group"

        # Idempotency.
        if not force_reanalyze:
            cached = self._repo.get_llm_response_for_cid(group_cid, "BATCH_ANALYSIS")
            if cached is not None:
                logger.debug("Returning cached group analysis for %s", group_cid)
                parsed = self._load_cached_parsed(cached)
                raw = cached["response_raw"] or ""
                return self._build_response(parsed, raw)

        # Build multi-run digest.
        digest = self._digest_builder.build_multi_run_digest(
            group.executions, all_errors, all_metrics
        )

        # API call.
        raw_text, usage = self._call_llm(MULTI_RUN_SYSTEM_PROMPT, digest)

        # Parse and persist.
        parsed = self._safe_parse(raw_text)
        self._persist(
            cid=group_cid,
            call_type="BATCH_ANALYSIS",
            digest=digest,
            raw_text=raw_text,
            parsed=parsed,
            usage=usage,
        )
        logger.info(
            "Group analysis complete for %s runs of %s",
            group.total_runs, group.job_name,
        )
        return self._build_response(parsed, raw_text)

    # ── Private: LLM call ────────────────────────────────────────

    def _call_llm(self, system: str, user_content: str) -> tuple[str, dict]:
        """Dispatch to the configured LLM provider.

        Returns:
            ``(raw_text, usage_dict)`` where usage_dict has
            ``input_tokens`` and ``output_tokens``.
        """
        if self._client is None:
            logger.warning("No LLM client configured — returning stub response.")
            stub = json.dumps({
                "summary": "LLM analysis skipped: no API key configured.",
                "root_cause": None,
                "error_categories": [],
                "recommendations": [],
                "business_impact": "Analysis unavailable — configure an API key to enable.",
                "retry_recommended": False,
                "tags": ["no-llm"],
                "evidence_anchors": [],
            })
            return stub, {"input_tokens": 0, "output_tokens": 0}

        try:
            return self._client.call(system, user_content)
        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            error_text = json.dumps({
                "summary": f"LLM call failed: {exc}",
                "root_cause": None,
                "error_categories": [],
                "recommendations": [],
                "business_impact": "Analysis unavailable due to API error.",
                "retry_recommended": False,
                "tags": ["api-error"],
                "evidence_anchors": [],
            })
            return error_text, {"input_tokens": 0, "output_tokens": 0}

    # ── Private: parsing ─────────────────────────────────────────

    def _safe_parse(self, raw_text: str) -> Optional[dict]:
        """Parse LLM output to a dict, never raising.

        Tries direct ``json.loads`` first; then strips markdown fences
        and retries.  Returns ``None`` on complete failure.

        Args:
            raw_text: Raw text from the LLM.

        Returns:
            Parsed dict, or ``None`` on failure.
        """
        # Attempt 1: direct parse.
        candidate = raw_text.strip()
        try:
            parsed = json.loads(candidate)
            if self._validate_parsed(parsed):
                return parsed
        except json.JSONDecodeError:
            pass

        # Attempt 2: strip markdown fences.
        m = _JSON_FENCE_RE.search(candidate)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if self._validate_parsed(parsed):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Attempt 3: find first { and last }.
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                parsed = json.loads(candidate[first : last + 1])
                if self._validate_parsed(parsed):
                    return parsed
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse LLM response as JSON (len=%d)", len(raw_text))
        return None

    @staticmethod
    def _validate_parsed(parsed: object) -> bool:
        """Return ``True`` if *parsed* is a dict with required keys."""
        if not isinstance(parsed, dict):
            return False
        return _REQUIRED_KEYS.issubset(parsed.keys())

    # ── Private: response building ───────────────────────────────

    def _build_response(
        self,
        parsed: Optional[dict],
        raw_text: str,
    ) -> BatchAnalysisResponse:
        """Map a parsed dict (or ``None``) to a ``BatchAnalysisResponse``."""
        if parsed is None:
            return self._empty_response(raw_text)

        def _sev(val: str, default: Severity = Severity.MEDIUM) -> Severity:
            try:
                return Severity(str(val).upper())
            except ValueError:
                return default

        error_categories = [
            ErrorCategory(
                category=e.get("category", "Unknown"),
                count=int(e.get("count", 0)),
                severity=_sev(e.get("severity", "MEDIUM")),
            )
            for e in (parsed.get("error_categories") or [])
            if isinstance(e, dict)
        ]

        recommendations = [
            Recommendation(
                action=r.get("action", ""),
                priority=_sev(r.get("priority", "MEDIUM")),
                rationale=r.get("rationale", ""),
            )
            for r in (parsed.get("recommendations") or [])
            if isinstance(r, dict)
        ]

        evidence_anchors = [
            EvidenceAnchor(
                description=a.get("description", ""),
                timestamp=a.get("timestamp"),
                keywords=list(a.get("keywords") or []),
                severity=_sev(a.get("severity", "MEDIUM")),
            )
            for a in (parsed.get("evidence_anchors") or [])
            if isinstance(a, dict)
        ]

        return BatchAnalysisResponse(
            summary=str(parsed.get("summary", "")),
            root_cause=parsed.get("root_cause"),
            error_categories=error_categories,
            recommendations=recommendations,
            business_impact=str(parsed.get("business_impact", "")),
            retry_recommended=bool(parsed.get("retry_recommended", False)),
            tags=list(parsed.get("tags") or []),
            evidence_anchors=evidence_anchors,
            raw_response=raw_text,
            parse_success=True,
        )

    @staticmethod
    def _empty_response(raw_text: str) -> BatchAnalysisResponse:
        """Return a minimal ``BatchAnalysisResponse`` for failure cases."""
        return BatchAnalysisResponse(
            summary="Analysis failed or unavailable.",
            root_cause=None,
            error_categories=[],
            recommendations=[],
            business_impact="Analysis unavailable.",
            retry_recommended=False,
            tags=[],
            evidence_anchors=[],
            raw_response=raw_text,
            parse_success=False,
        )

    # ── Private: persistence ─────────────────────────────────────

    def _persist(
        self,
        cid: str,
        call_type: str,
        digest: str,
        raw_text: str,
        parsed: Optional[dict],
        usage: dict,
    ) -> None:
        """Store LLM call in the inference log."""
        try:
            self._repo.store_llm_response(
                correlation_id=cid,
                call_type=call_type,
                request_prompt=digest,
                response_raw=raw_text,
                response_parsed=parsed,
                usage=usage,
                prompt_version=self._settings.llm_prompt_version,
            )
        except Exception as exc:
            logger.error("Failed to persist LLM response for %s: %s", cid, exc)

    @staticmethod
    def _load_cached_parsed(row: object) -> Optional[dict]:
        """Safely decode the ``response_parsed`` JSON column."""
        try:
            val = row["response_parsed"]  # type: ignore[index]
            if val is None:
                return None
            if isinstance(val, dict):
                return val
            return json.loads(val)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
