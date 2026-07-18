"""SQLite schema and connection management.

Defines all tables, indexes, and provides a ``DatabaseManager``
that creates / migrates the schema and vends connections with
WAL mode and foreign-key enforcement.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger("database")

# ── SQL Schema ──────────────────────────────────────────────────

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS batch_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name        TEXT NOT NULL,
    job_date        DATE NOT NULL,
    environment     TEXT NOT NULL DEFAULT 'prod',
    total_runs      INTEGER DEFAULT 0,
    successful_runs INTEGER DEFAULT 0,
    failed_runs     INTEGER DEFAULT 0,
    final_status    TEXT NOT NULL,
    first_run_time  DATETIME,
    last_run_time   DATETIME,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_name, job_date, environment)
);

CREATE TABLE IF NOT EXISTS batch_executions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id      TEXT NOT NULL UNIQUE,
    job_id              INTEGER REFERENCES batch_jobs(id),
    job_name            TEXT NOT NULL,
    run_number          INTEGER NOT NULL DEFAULT 1,
    attempt_type        TEXT NOT NULL DEFAULT 'SCHEDULED',
    environment         TEXT NOT NULL DEFAULT 'prod',
    status              TEXT NOT NULL,
    start_time          DATETIME,
    end_time            DATETIME,
    duration_seconds    REAL,
    total_lines         INTEGER DEFAULT 0,
    error_count         INTEGER DEFAULT 0,
    warn_count          INTEGER DEFAULT 0,
    orphan_lines_count  INTEGER DEFAULT 0,
    has_start_marker    BOOLEAN DEFAULT FALSE,
    has_end_marker      BOOLEAN DEFAULT FALSE,
    source_files        TEXT,
    llm_analyzed        BOOLEAN DEFAULT FALSE,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS log_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id  TEXT NOT NULL REFERENCES batch_executions(correlation_id),
    chunk_index     INTEGER NOT NULL,
    chunk_type      TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    content         TEXT NOT NULL,
    line_count      INTEGER NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS error_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id  TEXT NOT NULL REFERENCES batch_executions(correlation_id),
    error_category  TEXT NOT NULL,
    error_message   TEXT NOT NULL,
    count           INTEGER DEFAULT 1,
    first_seen      DATETIME,
    last_seen       DATETIME,
    severity        TEXT NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_inference_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id      TEXT NOT NULL,
    call_type           TEXT NOT NULL,
    request_prompt      TEXT NOT NULL,
    request_model       TEXT NOT NULL,
    request_timestamp   DATETIME NOT NULL,
    response_raw        TEXT NOT NULL,
    response_parsed     TEXT,
    response_timestamp  DATETIME NOT NULL,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    estimated_cost_usd  REAL,
    parse_success       BOOLEAN DEFAULT TRUE,
    prompt_version      TEXT DEFAULT 'v1.0',
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_executions_job_name
    ON batch_executions(job_name);
CREATE INDEX IF NOT EXISTS idx_executions_status
    ON batch_executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_start_time
    ON batch_executions(start_time);
CREATE INDEX IF NOT EXISTS idx_executions_cid
    ON batch_executions(correlation_id);
CREATE INDEX IF NOT EXISTS idx_llm_log_cid
    ON llm_inference_log(correlation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_cid
    ON log_chunks(correlation_id);
CREATE INDEX IF NOT EXISTS idx_errors_cid
    ON error_summary(correlation_id);
"""


class DatabaseManager:
    """Manages the SQLite database lifecycle.

    Creates tables and indexes on first initialisation and vends
    connections configured with WAL journal mode and foreign-key
    enforcement.

    For in-memory databases (``:memory:``), a single shared connection
    is maintained so that the schema persists across calls.  File-based
    databases create fresh connections each time (the file itself is
    the shared state).

    Usage::

        db = DatabaseManager("./data/log_analysis.db")
        db.initialize()
        conn = db.get_connection()
    """

    def __init__(self, db_path: str) -> None:
        """Initialise with a database file path.

        Args:
            db_path: Path to the SQLite file.  Use ``":memory:"``
                for in-memory databases (e.g. in tests).
        """
        self._db_path = db_path
        self._shared_conn: Optional[sqlite3.Connection] = None

    @property
    def db_path(self) -> str:
        """Return the configured database path."""
        return self._db_path

    def get_connection(self) -> sqlite3.Connection:
        """Return a connection with recommended pragmas.

        For ``:memory:`` databases a shared connection is reused so
        that the schema and data persist across calls.  For file-based
        databases a fresh connection is created each time.

        Returns:
            A configured ``sqlite3.Connection``.
        """
        if self._db_path == ":memory:":
            if self._shared_conn is None:
                self._shared_conn = self._make_connection()
            return self._shared_conn

        return self._make_connection()

    def initialize(self) -> None:
        """Create all tables and indexes if they do not exist.

        Safe to call multiple times — uses ``IF NOT EXISTS``.
        """
        conn = self.get_connection()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.executescript(_INDEX_SQL)
            conn.commit()
            logger.info("Database initialised at %s", self._db_path)
        except sqlite3.Error as exc:
            logger.error("Database initialisation failed: %s", exc)
            raise
        finally:
            # Only close file-based connections; keep :memory: alive.
            if self._db_path != ":memory:":
                conn.close()

    # ── Private helpers ─────────────────────────────────────────

    def _make_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        if self._db_path != ":memory:":
            try:
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error("Cannot create DB directory: %s", exc)

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
