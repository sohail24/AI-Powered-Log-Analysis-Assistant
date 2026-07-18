"""Central configuration for the batch log intelligence platform.

Uses Pydantic BaseSettings to load configuration from environment
variables and a .env file. Provides a cached singleton accessor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables and .env file.

    Fields are grouped by subsystem. Defaults are tuned for local
    development; override via environment variables or a .env file
    placed at the project root.
    """

    # ── Paths ───────────────────────────────────────────────────
    log_directory: str = Field(
        default="./data/logs",
        description="Directory where raw log files live.",
    )
    db_path: str = Field(
        default="./data/log_analysis.db",
        description="SQLite database file path.",
    )
    vector_db_path: str = Field(
        default="./data/vector_db",
        description="ChromaDB persist directory.",
    )
    raw_log_store_path: str = Field(
        default="./data/raw_logs",
        description="Where stitched unified logs are stored.",
    )

    # ── LLM ─────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="auto",
        description=(
            "LLM provider: 'anthropic', 'google', or 'auto' (prefer "
            "Anthropic when both keys are set)."
        ),
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key (set via ANTHROPIC_API_KEY env var).",
    )
    google_api_key: str = Field(
        default="",
        description=(
            "Google AI Studio API key for Gemini models "
            "(set via GOOGLE_API_KEY env var)."
        ),
    )
    llm_model: str = Field(
        default="",
        description=(
            "LLM model identifier. Leave empty to use the provider default "
            "(claude-sonnet-4-6 for Anthropic, gemini-2.0-flash for Google)."
        ),
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for LLM responses.",
    )
    llm_prompt_version: str = Field(
        default="v1.0",
        description="Prompt template version string.",
    )

    # ── Ingestion ───────────────────────────────────────────────
    max_error_lines_in_digest: int = Field(
        default=30,
        description="Max error lines included in a digest.",
    )
    orphan_window_size: int = Field(
        default=10,
        description="Lines to scan around an orphan for context.",
    )
    auto_retry_gap_minutes: int = Field(
        default=5,
        description="Minutes gap to classify an attempt as auto-retry.",
    )

    # ── RAG ─────────────────────────────────────────────────────
    rag_top_k_chunks: int = Field(
        default=8,
        description="Number of top chunks to retrieve for RAG.",
    )
    chunk_context_window: int = Field(
        default=10,
        description="Lines of context around an anchor line when chunking.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings.",
    )

    # ── Alerting ────────────────────────────────────────────────
    slack_webhook_url: Optional[str] = Field(
        default=None,
        description="Slack incoming-webhook URL (optional).",
    )
    alert_on_critical: bool = Field(
        default=True,
        description="Fire alerts on CRITICAL-severity events.",
    )

    # ── Environment ─────────────────────────────────────────────
    environment: str = Field(
        default="prod",
        description="Deployment environment: prod | staging | uat.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    The instance is created on first call and reused thereafter.
    To force a reload (e.g. in tests), call
    ``get_settings.cache_clear()`` first.
    """
    return Settings()
