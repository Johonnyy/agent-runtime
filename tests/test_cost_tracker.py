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


# --- injected settings must reach the tracker --------------------------------
#
# The other half of the get_client bug. This library is imported INTO another app's
# process, and a host that configures it by injection never sets AGENT_RUNTIME_* in
# its environment — so reading the module-level singleton here made `db_path` fall
# back to its relative default, `agent_runtime.db`.
#
# Two failure shapes, neither of which said what was wrong. Where the working
# directory was writable, cost rows landed in a stray ./agent_runtime.db while every
# other table lived in the host's real database, and the join between spend and the
# calls that caused it silently returned nothing. Where it was not writable — a
# container with WORKDIR /srv running as a non-root user — every tool in the run
# succeeded and then the final cost write raised
# `OperationalError: unable to open database file`.


def test_get_tracker_uses_the_injected_db_path(tmp_path):
    from agent_runtime.config import Settings
    from agent_runtime.cost_tracker import get_tracker

    target = tmp_path / "host.db"
    settings = Settings(_env_file=None, db_path=str(target), app_name="bloom")
    tracker = get_tracker(settings)
    assert tracker.path == str(target)
    assert tracker.app_name == "bloom"


def test_get_tracker_pools_one_connection_per_configuration(tmp_path):
    """It was an lru_cache singleton; it must still not reopen per call."""
    from agent_runtime.config import Settings
    from agent_runtime.cost_tracker import get_tracker

    settings = Settings(_env_file=None, db_path=str(tmp_path / "a.db"), app_name="x")
    assert get_tracker(settings) is get_tracker(settings)
    other = Settings(_env_file=None, db_path=str(tmp_path / "b.db"), app_name="x")
    assert get_tracker(other) is not get_tracker(settings)


def test_a_missing_parent_directory_is_created_not_fatal(tmp_path):
    """`unable to open database file` names neither the path nor the reason.

    Every host that shares its database with this library already creates the
    directory; the co-tenant was the one that did not.
    """
    from agent_runtime.cost_tracker import CostTracker

    nested = tmp_path / "does" / "not" / "exist" / "cost.db"
    tracker = CostTracker(str(nested), app_name="bloom")
    assert nested.parent.is_dir()
    tracker.close() if hasattr(tracker, "close") else None


def test_an_in_memory_database_still_works(tmp_path):
    """`:memory:` has no parent to create, and must not be treated as a path."""
    from agent_runtime.cost_tracker import CostTracker

    assert CostTracker(":memory:", app_name="x").path == ":memory:"
