"""Batch Log Intelligence Platform — pipeline entry point.

This module will orchestrate the full ingestion → segmentation →
preprocessing → storage → LLM analysis pipeline.  Currently a
placeholder that validates configuration loading.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Run the batch log intelligence pipeline.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    from app.config.settings import get_settings

    try:
        settings = get_settings()
    except Exception as exc:
        print(f"[FATAL] Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Environment : {settings.environment}")
    print(f"[INFO] LLM model   : {settings.llm_model}")
    print(f"[INFO] Log directory: {settings.log_directory}")
    print("[INFO] Configuration loaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
