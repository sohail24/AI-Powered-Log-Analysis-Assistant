"""LLM analysis layer — digest building, prompts, and Anthropic integration.

Public API:
    DigestBuilder        — compress batch data into token-efficient text.
    BatchAnalyzer        — orchestrate LLM calls with idempotency.
    BatchAnalysisResponse — structured response dataclass.
    SYSTEM_PROMPT        — single-execution analysis prompt.
    MULTI_RUN_SYSTEM_PROMPT — multi-run analysis prompt.
"""

from app.llm.analyzer import BatchAnalyzer
from app.llm.digest_builder import DigestBuilder
from app.llm.models import (
    BatchAnalysisResponse,
    ErrorCategory,
    EvidenceAnchor,
    Recommendation,
)
from app.llm.prompts import MULTI_RUN_SYSTEM_PROMPT, SYSTEM_PROMPT

__all__ = [
    "BatchAnalyzer",
    "DigestBuilder",
    "BatchAnalysisResponse",
    "ErrorCategory",
    "EvidenceAnchor",
    "Recommendation",
    "SYSTEM_PROMPT",
    "MULTI_RUN_SYSTEM_PROMPT",
]
