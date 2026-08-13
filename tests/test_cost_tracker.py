"""Cost logging into the host app's own database."""

import sqlite3

import pytest

from agent_runtime.cost_tracker import CostTracker
from agent_runtime.model_router import TIERS
from agent_runtime.stop_conditions import Step


@pytest.fixture
def tracker():
    t = CostTracker(":memory:", app_name="amber")
    yield t
    t.close()


def _step(index=0, cost=0.01, model=TIERS["balanced"], at="2026-08-13T12:00:00+00:00"):
    return Step(
        index=index,
        model=model,
        text="hi",
        tokens_in=100,
        tokens_out=50,
        cost_usd=cost,
        started_at=at,
        finished_at=at,
    )


def test_record_writes_a_row(tracker):
    row_id = tracker.record(_step(), conversation_id="conv_1", depth=2)
    assert row_id > 0

    summary = tracker.summary()
    assert summary["calls"] == 1
    assert summary["tokens_in"] == 100
    assert summary["tokens_out"] == 50
    assert summary["total_cost_usd"] == pytest.approx(0.01)


def test_tier_is_derived_from_the_model(tracker):
    tracker.record(_step())
    assert tracker.summary()["by_model"][0]["tier"] == "balanced"


def test_record_many_and_grouping(tracker):
    tracker.record_many(
        [_step(0, 0.01), _step(1, 0.02), _step(2, 0.05, model="openai/gpt-4o")],
        conversation_id="conv_1",
    )
    summary = tracker.summary()
    assert summary["calls"] == 3
    assert summary["total_cost_usd"] == pytest.approx(0.08)
    assert len(summary["by_model"]) == 2
    # Ordered by spend, so the most expensive model is first.
    assert summary["by_model"][0]["model"] == "openai/gpt-4o"


def test_summary_since_filters_by_timestamp(tracker):
    tracker.record(_step(0, 0.01, at="2026-01-01T00:00:00+00:00"))
    tracker.record(_step(1, 0.05, at="2026-08-01T00:00:00+00:00"))

    recent = tracker.summary(since="2026-06-01T00:00:00+00:00")
    assert recent["calls"] == 1
    assert recent["total_cost_usd"] == pytest.approx(0.05)
    assert recent["since"] == "2026-06-01T00:00:00+00:00"


def test_empty_summary_is_zeroed_not_null(tracker):
    summary = tracker.summary()
    assert summary["calls"] == 0
    assert summary["total_cost_usd"] == 0.0
    assert summary["by_model"] == []


def test_conversation_id_and_app_name_are_stored_for_joining(tracker):
    tracker.record(_step(), conversation_id="conv_42", depth=3)
    row = tracker._conn.execute(
        "SELECT app_name, conversation_id, depth FROM agent_runtime_usage"
    ).fetchone()
    assert (row["app_name"], row["conversation_id"], row["depth"]) == ("amber", "conv_42", 3)


def test_shares_a_database_file_with_agent_mcp(tmp_path):
    # agent-mcp-py logs tool calls into its own table in what is likely the same
    # file. Opening ours must leave theirs alone, and vice versa.
    path = str(tmp_path / "amber.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE agent_mcp_usage (id INTEGER PRIMARY KEY, conversation_id TEXT)"
    )
    conn.execute("INSERT INTO agent_mcp_usage (conversation_id) VALUES ('conv_1')")
    conn.commit()
    conn.close()

    tracker = CostTracker(path, app_name="amber")
    try:
        tracker.record(_step(), conversation_id="conv_1")
        assert tracker.summary()["calls"] == 1
    finally:
        tracker.close()

    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM agent_mcp_usage").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM agent_runtime_usage").fetchone()[0] == 1
    finally:
        conn.close()


def test_opening_twice_is_idempotent(tmp_path):
    path = str(tmp_path / "usage.db")
    first = CostTracker(path)
    first.record(_step())
    first.close()

    second = CostTracker(path)
    try:
        # Schema re-applied without wiping anything.
        assert second.summary()["calls"] == 1
    finally:
        second.close()
