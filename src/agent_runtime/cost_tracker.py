"""Per-call cost logging, in the host app's own database.

Usage stays local, deliberately. There is no shared central table anywhere in this
ecosystem: each app records what its own agent spent, and anything that wants a
fleet-wide view (the spawner, eventually) asks each app for its summary rather than
reading one privileged database. That keeps every app independently cloneable and
runnable, which is the whole design constraint.

The storage discipline mirrors Amber's memory store, because that is the pattern
these apps already run: synchronous ``sqlite3`` (async callers wrap in
``asyncio.to_thread``), one connection with ``check_same_thread=False`` serialised by
a lock, schema applied as ``CREATE TABLE IF NOT EXISTS`` on open, timestamps as
sortable ISO-8601 UTC text.

Two additions specific to this package. WAL mode, because `agent-mcp-py` logs *tool*
calls into its own table in what may well be the same file, and two writers on one
database without WAL means avoidable lock contention. And a deliberate column
overlap — ``conversation_id``, ``app_name``, ``depth``, ``created_at`` — so model
spend can be joined to the tool calls that caused it.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from agent_runtime.config import Settings, get_settings
from agent_runtime.model_router import tier_for
from agent_runtime.stop_conditions import Step

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runtime_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    app_name        TEXT    NOT NULL,
    conversation_id TEXT,
    depth           INTEGER NOT NULL DEFAULT 0,
    model           TEXT    NOT NULL,
    tier            TEXT,
    step_index      INTEGER NOT NULL,
    tokens_in       INTEGER NOT NULL,
    tokens_out      INTEGER NOT NULL,
    cost_usd        REAL    NOT NULL,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runtime_usage_conv
    ON agent_runtime_usage (conversation_id);
CREATE INDEX IF NOT EXISTS idx_runtime_usage_created
    ON agent_runtime_usage (created_at);
"""


def _now() -> str:
    """Current UTC time as an ISO-8601 string (lexically sortable)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CostTracker:
    """Thin synchronous wrapper over the usage table."""

    def __init__(self, path: str = "agent_runtime.db", app_name: str = "unknown") -> None:
        self.path = path
        self.app_name = app_name
        # check_same_thread=False: the connection is reached from asyncio.to_thread
        # worker threads. Safe because every access is serialised by _lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            # WAL lets agent-mcp-py write its own table in this file concurrently.
            # In-memory databases don't support it; that's fine, nothing shares them.
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:  # pragma: no cover - platform dependent
                logger.debug("WAL unavailable for %s; continuing in default mode", path)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(
        self,
        step: Step,
        *,
        conversation_id: str | None = None,
        depth: int = 0,
    ) -> int:
        """Write one step's spend; returns the new row id."""
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO agent_runtime_usage
                    (app_name, conversation_id, depth, model, tier, step_index,
                     tokens_in, tokens_out, cost_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.app_name,
                    conversation_id,
                    depth,
                    step.model,
                    tier_for(step.model),
                    step.index,
                    step.tokens_in,
                    step.tokens_out,
                    step.cost_usd,
                    step.finished_at or _now(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def record_many(
        self,
        steps: list[Step],
        *,
        conversation_id: str | None = None,
        depth: int = 0,
    ) -> int:
        """Write a whole run's steps. Returns how many rows landed."""
        count = 0
        for step in steps:
            self.record(step, conversation_id=conversation_id, depth=depth)
            count += 1
        return count

    def summary(self, since: str | None = None) -> dict[str, Any]:
        """Aggregate spend, optionally from an ISO-8601 timestamp onwards.

        The shape an app needs to answer "what has this agent cost me" without
        anyone else's database being involved.
        """
        where, params = ("WHERE created_at >= ?", (since,)) if since else ("", ())
        with self._lock:
            totals = self._conn.execute(
                f"""
                SELECT COUNT(*)            AS calls,
                       COALESCE(SUM(tokens_in), 0)  AS tokens_in,
                       COALESCE(SUM(tokens_out), 0) AS tokens_out,
                       COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd
                FROM agent_runtime_usage {where}
                """,
                params,
            ).fetchone()
            by_model = self._conn.execute(
                f"""
                SELECT model,
                       tier,
                       COUNT(*)                     AS calls,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd
                FROM agent_runtime_usage {where}
                GROUP BY model, tier
                ORDER BY cost_usd DESC
                """,
                params,
            ).fetchall()

        result = dict(totals)
        result["since"] = since
        result["by_model"] = [dict(row) for row in by_model]
        return result


@lru_cache
def get_tracker() -> CostTracker:
    """Process-wide tracker singleton, opened at the configured path.

    Cached so the schema is applied once and the connection is reused. Tests that
    want isolation construct ``CostTracker(":memory:")`` directly and inject it.
    """
    settings: Settings = get_settings()
    return CostTracker(settings.db_path, app_name=settings.app_name)
