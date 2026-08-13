"""The two libraries, talking to each other for real.

Every other test in this suite fakes the far end. This one stands up an actual
`agent-mcp-py` server under uvicorn and drives it with an actual `MCPClient` over
Streamable HTTP — no fake sessions, no injected resolver, the real registry.

It exists because every interop bug found between these two packages was silent.
A tool that reported `requires_confirmation=False` still worked; a failed remote
call still returned text; a request built one hop past the cap still got a
response, just an error one. Unit tests on either side passed throughout. Only
running them against each other showed it.

Skipped when `agent-mcp-py` or the `mcp` SDK is absent, since both are optional.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("agent_mcp", reason="agent-mcp-py is an optional peer")
pytest.importorskip("mcp", reason="the mcp extra is not installed")
pytest.importorskip("uvicorn")

import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402

from agent_mcp import AgentMCPServer, AgentMCPSettings  # noqa: E402
from agent_mcp.registry import PeerRecord, default_registry  # noqa: E402
from agent_mcp.usage_log import SQLiteUsageSink  # noqa: E402
from agent_runtime.mcp_client import MCPClient  # noqa: E402

TOKEN = "interop-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def peer():
    """A real finance server, registered under a bare base URL."""
    sink = SQLiteUsageSink(":memory:")
    settings = AgentMCPSettings(
        _env_file=None,
        app_name="finance",
        keys=f"amber:{TOKEN}",
        allow_anonymous=False,
        usage_enabled=True,
        sync_store_url="",
    )
    mcp = AgentMCPServer(
        app_name="finance", version="0.1.0", settings=settings, usage_sink=sink
    )

    @mcp.tool(read_only=True)
    def get_budget(quarter: str) -> str:
        """Read a budget."""
        return f"{quarter}: 4200"

    @mcp.tool(read_only=False, requires_confirmation=True)
    def move_money(amount: float) -> str:
        """Transfer funds."""
        return f"moved {amount}"

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.lifespan():
            yield

    app = Starlette(routes=mcp.routes(), lifespan=lifespan)

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "the peer server did not start"

    # Registered the way the sync store will register it: a BARE base URL. The
    # client is what appends the /mcp mount path.
    default_registry().set_static(
        {
            "finance": PeerRecord(
                name="finance", base_url=f"http://127.0.0.1:{port}", token=TOKEN
            )
        }
    )

    yield sink

    default_registry().set_static({})
    server.should_exit = True
    thread.join(timeout=10)
    sink.close()


async def test_tools_are_discovered_through_the_real_registry(peer):
    """No injected resolver: MCPClient goes through agent_mcp.registry.resolve,
    which returns a PeerRecord dataclass holding a bare base URL."""
    client = MCPClient(["finance"])
    client.bind(conversation_id="conv-interop", depth=0)
    try:
        schemas = await client.list_tools()
    finally:
        await client.aclose()

    names = sorted(s["function"]["name"] for s in schemas)
    assert names == ["finance__get_budget", "finance__move_money"]


async def test_the_flags_survive_the_round_trip(peer):
    """read_only rides in standard annotations, requires_confirmation in _meta —
    and the runtime has to end up with both."""
    client = MCPClient(["finance"])
    client.bind(conversation_id="conv-flags", depth=0)
    try:
        schemas = {s["function"]["name"]: s for s in await client.list_tools()}
    finally:
        await client.aclose()

    assert schemas["finance__get_budget"]["x_agent"] == {
        "read_only": True,
        "requires_confirmation": False,
    }
    assert schemas["finance__move_money"]["x_agent"] == {
        "read_only": False,
        "requires_confirmation": True,
    }


async def test_schemas_arrive_free_of_refs(peer):
    client = MCPClient(["finance"])
    client.bind(depth=0)
    try:
        schemas = await client.list_tools()
    finally:
        await client.aclose()
    import json

    text = json.dumps(schemas)
    assert "$ref" not in text and "$defs" not in text


async def test_a_tool_call_round_trips(peer):
    client = MCPClient(["finance"])
    client.bind(conversation_id="conv-call", depth=0)
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {"quarter": "Q3"})
    finally:
        await client.aclose()
    assert "4200" in result


async def test_the_confirmation_gate_is_enforced_across_the_wire(peer):
    """The server refuses without X-Confirmed, and the refusal must reach the model
    as readable text rather than an exception that ends the turn."""
    client = MCPClient(["finance"])
    client.bind(conversation_id="conv-gate", depth=0)
    try:
        await client.list_tools()
        result = await client.call_tool("finance__move_money", {"amount": 10.0})
    finally:
        await client.aclose()
    assert "X-Confirmed" in result or "approval" in result.lower()


async def test_confirming_lets_the_call_through(peer):
    client = MCPClient(["finance"])
    client.bind(conversation_id="conv-ok", depth=0, confirmed=True)
    try:
        await client.list_tools()
        result = await client.call_tool("finance__move_money", {"amount": 10.0})
    finally:
        await client.aclose()
    assert "moved 10.0" in result


async def test_the_depth_headers_arrive_and_are_logged_by_the_server(peer):
    sink = peer
    client = MCPClient(["finance"])
    client.bind(conversation_id="conv-depth", depth=2)  # sends depth 3
    try:
        await client.list_tools()
        await client.call_tool("finance__get_budget", {"quarter": "Q1"})
    finally:
        await client.aclose()

    row = next(r for r in sink.rows() if r["conversation_id"] == "conv-depth")
    assert row["depth"] == 3
    assert row["caller"] == "amber"  # resolved from the name:token key
    assert row["name"] == "get_budget"
    assert row["ok"] == 1


async def test_the_last_legal_hop_is_accepted_by_the_server(peer):
    """The sender's cap and the server's cap have to agree exactly, or the final
    hop is built, sent, and refused."""
    from agent_runtime.mcp_client import MAX_AGENT_DEPTH

    client = MCPClient(["finance"])
    client.bind(conversation_id="conv-edge", depth=MAX_AGENT_DEPTH - 2)
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {"quarter": "Q4"})
    finally:
        await client.aclose()
    assert "4200" in result


async def test_an_over_cap_run_is_refused_before_any_request(peer):
    from agent_runtime.mcp_client import MAX_AGENT_DEPTH, DepthExceeded

    client = MCPClient(["finance"])
    with pytest.raises(DepthExceeded):
        client.bind(conversation_id="conv-toodeep", depth=MAX_AGENT_DEPTH - 1)


async def test_a_bad_token_surfaces_as_text_not_a_crash(peer):
    """Auth failure is an HTTP 401 at the server's edge. The runtime must turn that
    into something the model can read, not an exception that ends the turn."""
    client = MCPClient(
        ["finance"],
        resolver={"finance": {"base_url": _base_url(), "token": "wrong-token"}},
    )
    client.bind(depth=0)
    try:
        schemas = await client.list_tools()
    finally:
        await client.aclose()
    # The server is unreachable *as far as this client is concerned*; losing one
    # app's tools must not fail the run.
    assert schemas == []


def _base_url() -> str:
    record = default_registry().resolve("finance")
    assert record is not None
    return record.base_url
