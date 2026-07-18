"""Data models for LLM analysis responses.

These dataclasses represent the structured output produced by the
LLM when analysing a batch execution digest.  They are the
single source of truth for the response schema throughout the
application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.config.constants import Severity


@dataclass
class EvidenceAnchor:
    """A pointer to specific evidence in the raw log.

    Used by the dashboard to link analysis prose back to
    the lines that justify a claim.

    Attributes:
        description: What this evidence shows.
        timestamp: ISO timestamp string near the evidence, or ``None``.
        keywords: 2–4 keywords that locate the relevant lines.
        severity: Assessed severity of this piece of evidence.
    """

    description: str
    timestamp: Optional[str]
    keywords: List[str]
    severity: Severity


@dataclass
class ErrorCategory:
    """An error type identified in the batch log.

    Attributes:
        category: Category label (e.g. ``"DatabaseConnectivity"``).
        count: Number of occurrences.
        severity: Assessed severity level.
    """

    category: str
    count: int
    severity: Severity


@dataclass
class Recommendation:
    """An actionable recommendation from the LLM.

    Attributes:
        action: Specific action to take.
        priority: How urgently it should be addressed.
        rationale: Why this action is recommended.
    """

    action: str
    priority: Severity
    rationale: str


@dataclass
class BatchAnalysisResponse:
    """Complete LLM analysis of a single batch execution (or group).

    Attributes:
        summary: One-sentence executive summary.
        root_cause: Specific root cause, or ``None`` if not determinable.
        error_categories: Structured list of error types found.
        recommendations: Ordered list of actionable recommendations.
        business_impact: Plain-English impact for non-technical readers.
        retry_recommended: Whether a manual retry is advised.
        tags: Short classification tags (e.g. ``["infra", "timeout"]``).
        evidence_anchors: Log anchors that support key claims.
        raw_response: Original LLM text before parsing.
        parse_success: ``True`` if the response was valid JSON.
    """

    summary: str
    root_cause: Optional[str]
    error_categories: List[ErrorCategory]
    recommendations: List[Recommendation]
    business_impact: str
    retry_recommended: bool
    tags: List[str]
    evidence_anchors: List[EvidenceAnchor]
    raw_response: str
    parse_success: bool = True

    # ── Convenience helpers ──────────────────────────────────────

    @property
    def has_critical_errors(self) -> bool:
        """Return ``True`` if any error category is CRITICAL."""
        return any(e.severity == Severity.CRITICAL for e in self.error_categories)

    @property
    def top_recommendation(self) -> Optional[Recommendation]:
        """Return the highest-priority recommendation, or ``None``."""
        if not self.recommendations:
            return None
        order = {Severity.CRITICAL: 0, Severity.HIGH: 1,
                 Severity.MEDIUM: 2, Severity.LOW: 3}
        return min(self.recommendations, key=lambda r: order.get(r.priority, 99))
