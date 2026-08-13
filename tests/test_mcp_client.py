"""Tool brokers.

The MCP tests never import the `mcp` SDK: `MCPClient` takes an injectable session
factory, and the fake sessions below return plain dicts. That is deliberate — the
SDK is an optional extra, so the suite has to pass without it.
"""

from contextlib import asynccontextmanager

import pytest

from agent_runtime.mcp_client import (
    MAX_AGENT_DEPTH,
    AnthropicRegistryBroker,
    CompositeBroker,
    DepthExceeded,
    LocalToolBroker,
    MCPClient,
)


# --- fakes -------------------------------------------------------------------


class FakeSession:
    def __init__(self, tools, results=None):
        self._tools = tools
        self._results = results or {}
        self.calls = []

    async def list_tools(self):
        return {"tools": self._tools}

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name not in self._results:
            return {"content": [{"text": "ok"}], "isError": False}
        result = self._results[name]
        if isinstance(result, Exception):
            raise result
        return result


class FakeFactory:
    """Hands out a session per base URL and records the headers it was opened with."""

    def __init__(self, sessions):
        self.sessions = sessions
        self.opened = []

    def __call__(self, base_url, headers):
        @asynccontextmanager
        async def _open():
            self.opened.append((base_url, dict(headers)))
            session = self.sessions.get(base_url)
            if session is None:
                raise ConnectionError(f"no server at {base_url}")
            yield session

        return _open()


REGISTRY = {
    "finance": {"base_url": "https://finance.test/mcp", "token": "tok-fin"},
    "spawner": {"base_url": "https://spawner.test/mcp"},
}

FINANCE_TOOLS = [
    {
        "name": "get_budget",
        "description": "Read a budget.",
        "inputSchema": {"type": "object", "properties": {"quarter": {"type": "string"}}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "move_money",
        "description": "Transfer funds.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"requiresConfirmation": True},
    },
]


def _client(sessions, servers=("finance",)):
    factory = FakeFactory(sessions)
    client = MCPClient(servers, resolver=REGISTRY, session_factory=factory)
    return client, factory


# --- LocalToolBroker ---------------------------------------------------------


async def test_local_broker_lists_openai_shaped_schemas():
    broker = LocalToolBroker()
    broker.register(
        "add_task",
        "Add a task.",
        {"type": "object", "properties": {"description": {"type": "string"}}},
        lambda description: f"added {description}",
    )

    (schema,) = await broker.list_tools()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add_task"
    assert schema["function"]["parameters"]["type"] == "object"


async def test_local_broker_runs_sync_and_async_tools():
    broker = LocalToolBroker()
    broker.register("sync", "", {}, lambda x: x * 2)

    async def double(x):
        return x * 2

    broker.register("async", "", {}, double)

    assert await broker.call_tool("sync", {"x": 3}) == "6"
    assert await broker.call_tool("async", {"x": 3}) == "6"


async def test_local_broker_turns_failures_into_text():
    broker = LocalToolBroker()

    def boom():
        raise RuntimeError("disk on fire")

    broker.register("boom", "", {}, boom)

    result = await broker.call_tool("boom", {})
    assert result.startswith("Error running boom:")
    assert "disk on fire" in result


async def test_local_broker_unknown_tool_is_an_error_string():
    assert "not available" in await LocalToolBroker().call_tool("nope", {})


async def test_local_broker_rejects_duplicate_registration():
    broker = LocalToolBroker()
    broker.register("x", "", {}, lambda: "1")
    with pytest.raises(ValueError):
        broker.register("x", "", {}, lambda: "2")


# --- AnthropicRegistryBroker (Amber's existing registry) ---------------------


async def test_anthropic_registry_is_converted():
    schemas = [
        {
            "name": "web_search",
            "description": "Search the web.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]

    async def dispatch(name, args):
        return f"{name}:{args['query']}"

    broker = AnthropicRegistryBroker(schemas, dispatch)

    (schema,) = await broker.list_tools()
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert await broker.call_tool("web_search", {"query": "amber"}) == "web_search:amber"


async def test_anthropic_registry_callable_is_reevaluated():
    # Amber hides tools whose config is missing, so availability changes at runtime.
    available = []

    async def dispatch(name, args):
        return "ok"

    broker = AnthropicRegistryBroker(lambda: list(available), dispatch)
    assert await broker.list_tools() == []

    available.append({"name": "later", "description": "", "input_schema": {}})
    assert len(await broker.list_tools()) == 1


# --- MCPClient ---------------------------------------------------------------


async def test_tools_are_namespaced_by_server():
    client, _ = _client({"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)})
    try:
        names = [s["function"]["name"] for s in await client.list_tools()]
    finally:
        await client.aclose()
    assert names == ["finance__get_budget", "finance__move_money"]


async def test_annotations_are_carried_through():
    client, _ = _client({"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)})
    try:
        schemas = {s["function"]["name"]: s for s in await client.list_tools()}
    finally:
        await client.aclose()
    assert schemas["finance__get_budget"]["x_agent"]["read_only"] is True
    assert schemas["finance__move_money"]["x_agent"]["requires_confirmation"] is True


async def test_call_tool_strips_the_namespace():
    session = FakeSession(FINANCE_TOOLS)
    client, _ = _client({"https://finance.test/mcp": session})
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {"quarter": "Q3"})
    finally:
        await client.aclose()
    assert session.calls == [("get_budget", {"quarter": "Q3"})]
    assert result == "ok"


async def test_depth_and_conversation_headers_are_sent():
    client, factory = _client({"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)})
    client.bind(conversation_id="conv_9", depth=1)
    try:
        await client.list_tools()
    finally:
        await client.aclose()

    _, headers = factory.opened[0]
    # This client is itself a hop, so outbound calls carry depth + 1.
    assert headers["X-Agent-Depth"] == "2"
    assert headers["X-Conversation-Id"] == "conv_9"
    assert headers["Authorization"] == "Bearer tok-fin"
    assert "X-Confirmed" not in headers


async def test_confirmed_header_only_when_confirmed():
    client, factory = _client({"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)})
    client.bind(conversation_id="c", depth=0, confirmed=True)
    try:
        await client.list_tools()
    finally:
        await client.aclose()
    assert factory.opened[0][1]["X-Confirmed"] == "true"


async def test_binding_past_the_depth_cap_fails_before_any_request():
    client, factory = _client({"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)})
    with pytest.raises(DepthExceeded):
        client.bind(depth=MAX_AGENT_DEPTH)
    assert factory.opened == []


async def test_is_error_result_becomes_readable_text():
    session = FakeSession(
        FINANCE_TOOLS,
        results={"get_budget": {"content": [{"text": "no such quarter"}], "isError": True}},
    )
    client, _ = _client({"https://finance.test/mcp": session})
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {})
    finally:
        await client.aclose()
    assert result == "Error: no such quarter"


async def test_a_failing_tool_returns_text_rather_than_raising():
    session = FakeSession(FINANCE_TOOLS, results={"get_budget": RuntimeError("upstream 500")})
    client, _ = _client({"https://finance.test/mcp": session})
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {})
    finally:
        await client.aclose()
    assert "Error running finance__get_budget" in result
    assert "upstream 500" in result


async def test_an_unreachable_server_is_skipped_not_fatal():
    client, _ = _client(
        {"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)},
        servers=("finance", "spawner"),  # spawner has no session in the factory
    )
    try:
        names = [s["function"]["name"] for s in await client.list_tools()]
    finally:
        await client.aclose()
    assert names == ["finance__get_budget", "finance__move_money"]


async def test_unknown_tool_is_an_error_string():
    client, _ = _client({"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)})
    try:
        assert "not available" in await client.call_tool("nowhere__thing", {})
    finally:
        await client.aclose()


async def test_sessions_are_reused_across_calls():
    session = FakeSession(FINANCE_TOOLS)
    client, factory = _client({"https://finance.test/mcp": session})
    try:
        await client.list_tools()
        await client.call_tool("finance__get_budget", {})
        await client.call_tool("finance__move_money", {})
    finally:
        await client.aclose()
    assert len(factory.opened) == 1


async def test_missing_registry_entry_is_reported_clearly():
    client = MCPClient(["ghost"], resolver=REGISTRY, session_factory=FakeFactory({}))
    try:
        # list_tools swallows it (one dead server must not kill the turn)...
        assert await client.list_tools() == []
        # ...but resolving directly says exactly what's wrong.
        with pytest.raises(KeyError):
            client._resolve("ghost")
    finally:
        await client.aclose()


# --- CompositeBroker ---------------------------------------------------------


async def test_composite_merges_and_routes():
    local = LocalToolBroker()
    local.register("add_task", "", {}, lambda: "added")

    session = FakeSession(FINANCE_TOOLS)
    remote, _ = _client({"https://finance.test/mcp": session})

    composite = CompositeBroker([local, remote])
    try:
        names = [s["function"]["name"] for s in await composite.list_tools()]
        assert names == ["add_task", "finance__get_budget", "finance__move_money"]

        assert await composite.call_tool("add_task", {}) == "added"
        assert await composite.call_tool("finance__get_budget", {"quarter": "Q3"}) == "ok"
    finally:
        await composite.aclose()

    assert session.calls == [("get_budget", {"quarter": "Q3"})]


async def test_composite_keeps_the_first_broker_on_a_collision():
    first = LocalToolBroker()
    first.register("shared", "", {}, lambda: "from first")
    second = LocalToolBroker()
    second.register("shared", "", {}, lambda: "from second")

    composite = CompositeBroker([first, second])
    assert len(await composite.list_tools()) == 1
    assert await composite.call_tool("shared", {}) == "from first"


async def test_composite_binds_every_broker_that_can_be_bound():
    local = LocalToolBroker()  # no bind() at all — must not blow up
    remote, factory = _client({"https://finance.test/mcp": FakeSession(FINANCE_TOOLS)})

    composite = CompositeBroker([local, remote])
    composite.bind(conversation_id="conv_1", depth=0)
    try:
        await composite.list_tools()
    finally:
        await composite.aclose()

    assert factory.opened[0][1]["X-Conversation-Id"] == "conv_1"
