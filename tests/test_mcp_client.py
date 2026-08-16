"""Tool brokers.

The MCP tests never import the `mcp` SDK: `MCPClient` takes an injectable session
factory, and the fake sessions below return plain dicts. That is deliberate — the
SDK is an optional extra, so the suite has to pass without it.
"""

import asyncio
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


# Registry records hold a *bare* base URL — the ecosystem's fixed /mcp mount path is
# appended by the client, so the sync store never has to store it. agent-mcp-py's
# PeerRecord works the same way.
REGISTRY = {
    "finance": {"base_url": "https://finance.test", "token": "tok-fin"},
    "spawner": {"base_url": "https://spawner.test"},
}
# What _endpoint() resolves those to. The trailing slash is load-bearing: a mounted
# MCP app answers the bare path with a 307 the client does not follow.
FINANCE_URL = "https://finance.test/mcp/"
SPAWNER_URL = "https://spawner.test/mcp/"

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
    client, _ = _client({FINANCE_URL: FakeSession(FINANCE_TOOLS)})
    try:
        names = [s["function"]["name"] for s in await client.list_tools()]
    finally:
        await client.aclose()
    assert names == ["finance__get_budget", "finance__move_money"]


async def test_annotations_are_carried_through():
    client, _ = _client({FINANCE_URL: FakeSession(FINANCE_TOOLS)})
    try:
        schemas = {s["function"]["name"]: s for s in await client.list_tools()}
    finally:
        await client.aclose()
    assert schemas["finance__get_budget"]["x_agent"]["read_only"] is True
    assert schemas["finance__move_money"]["x_agent"]["requires_confirmation"] is True


async def test_call_tool_strips_the_namespace():
    session = FakeSession(FINANCE_TOOLS)
    client, _ = _client({FINANCE_URL: session})
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {"quarter": "Q3"})
    finally:
        await client.aclose()
    assert session.calls == [("get_budget", {"quarter": "Q3"})]
    assert result == "ok"


async def test_depth_and_conversation_headers_are_sent():
    client, factory = _client({FINANCE_URL: FakeSession(FINANCE_TOOLS)})
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
    client, factory = _client({FINANCE_URL: FakeSession(FINANCE_TOOLS)})
    client.bind(conversation_id="c", depth=0, confirmed=True)
    try:
        await client.list_tools()
    finally:
        await client.aclose()
    assert factory.opened[0][1]["X-Confirmed"] == "true"


async def test_binding_past_the_depth_cap_fails_before_any_request():
    client, factory = _client({FINANCE_URL: FakeSession(FINANCE_TOOLS)})
    with pytest.raises(DepthExceeded):
        client.bind(depth=MAX_AGENT_DEPTH)
    assert factory.opened == []


async def test_is_error_result_becomes_readable_text():
    session = FakeSession(
        FINANCE_TOOLS,
        results={"get_budget": {"content": [{"text": "no such quarter"}], "isError": True}},
    )
    client, _ = _client({FINANCE_URL: session})
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {})
    finally:
        await client.aclose()
    assert result == "Error: no such quarter"


async def test_a_failing_tool_returns_text_rather_than_raising():
    session = FakeSession(FINANCE_TOOLS, results={"get_budget": RuntimeError("upstream 500")})
    client, _ = _client({FINANCE_URL: session})
    try:
        await client.list_tools()
        result = await client.call_tool("finance__get_budget", {})
    finally:
        await client.aclose()
    assert "Error running finance__get_budget" in result
    assert "upstream 500" in result


async def test_an_unreachable_server_is_skipped_not_fatal():
    client, _ = _client(
        {FINANCE_URL: FakeSession(FINANCE_TOOLS)},
        servers=("finance", "spawner"),  # spawner has no session in the factory
    )
    try:
        names = [s["function"]["name"] for s in await client.list_tools()]
    finally:
        await client.aclose()
    assert names == ["finance__get_budget", "finance__move_money"]


async def test_unknown_tool_is_an_error_string():
    client, _ = _client({FINANCE_URL: FakeSession(FINANCE_TOOLS)})
    try:
        assert "not available" in await client.call_tool("nowhere__thing", {})
    finally:
        await client.aclose()


async def test_sessions_are_reused_across_calls():
    session = FakeSession(FINANCE_TOOLS)
    client, factory = _client({FINANCE_URL: session})
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
    remote, _ = _client({FINANCE_URL: session})

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
    remote, factory = _client({FINANCE_URL: FakeSession(FINANCE_TOOLS)})

    composite = CompositeBroker([local, remote])
    composite.bind(conversation_id="conv_1", depth=0)
    try:
        await composite.list_tools()
    finally:
        await composite.aclose()

    assert factory.opened[0][1]["X-Conversation-Id"] == "conv_1"


# --- interop with agent-mcp-py ------------------------------------------------
#
# Each test below pins a collision that was real: the two libraries disagreed, and
# every one of these failures was silent — a tool that looked safe, an error that
# looked like success, a request built one hop past what the receiver accepts.


def test_the_depth_fallback_matches_agent_mcps_signature_exactly():
    """The call sites are shared between both branches, so a fallback whose
    constructor differs turns a depth refusal into a TypeError *only* where
    agent-mcp-py is installed — i.e. only in production."""
    import inspect

    from agent_runtime.mcp_client import DepthExceeded, check_depth

    exc = DepthExceeded(7, 5, "conv-1")
    assert (exc.depth, exc.limit, exc.conversation_id) == (7, 5, "conv-1")
    assert isinstance(exc, Exception)
    params = list(inspect.signature(check_depth).parameters)
    assert params[0] == "depth"


def test_the_sender_refuses_exactly_what_the_receiver_would_refuse():
    """bind() checks the depth it is about to SEND, not the one it received.
    Checking the received depth builds a request one hop past the cap that every
    agent-mcp-py server then rejects — a wasted round trip surfacing as a remote
    protocol error."""
    from agent_runtime.mcp_client import check_depth

    for depth in range(0, MAX_AGENT_DEPTH + 2):
        try:
            MCPClient([]).bind(depth=depth)
            sender_allows = True
        except DepthExceeded:
            sender_allows = False
        try:
            check_depth(depth + 1, limit=MAX_AGENT_DEPTH)
            receiver_accepts = True
        except DepthExceeded:
            receiver_accepts = False
        assert sender_allows == receiver_accepts, f"disagreement at depth {depth}"


def test_requires_confirmation_is_read_from_meta_not_annotations():
    """MCP has no standard confirmation flag, and the SDK's ToolAnnotations model is
    extra="ignore" — a server putting requiresConfirmation there has it silently
    dropped. agent-mcp-py publishes it in _meta instead. Reading only annotations
    fails open: every gated tool looks callable without approval."""
    from agent_runtime.mcp_client import _tool_flags

    tool = {
        "name": "create_invoice",
        "annotations": {"readOnlyHint": False},
        "_meta": {
            "dev.johnny.agent-mcp/requiresConfirmation": True,
            "dev.johnny.agent-mcp/readOnly": False,
        },
    }
    assert _tool_flags(tool) == (False, True)


def test_flags_fall_back_to_annotations_for_servers_that_are_not_ours():
    from agent_runtime.mcp_client import _tool_flags

    assert _tool_flags({"annotations": {"readOnlyHint": True}}) == (True, False)
    assert _tool_flags({}) == (False, False)


def test_fields_are_read_in_either_naming_style():
    """MCP's wire format is camelCase but mcp v2's Python attributes are snake_case.
    Reading only camelCase off a real SDK object returns the default every time."""
    from agent_runtime.mcp_client import _field

    class SnakeCaseObject:
        is_error = True
        input_schema = {"type": "object"}

    obj = SnakeCaseObject()
    assert _field(obj, "isError") is True
    assert _field(obj, "inputSchema") == {"type": "object"}
    assert _field({"is_error": True}, "isError") is True
    assert _field({"isError": True}, "isError") is True
    assert _field(obj, "somethingElse", "default") == "default"


def test_a_failed_remote_tool_is_never_reported_as_success():
    from agent_runtime.mcp_client import _result_text

    class SnakeCaseResult:
        content = [{"text": "boom"}]
        is_error = True

    assert _result_text(SnakeCaseResult()).startswith("Error:")


def test_a_peer_record_dataclass_is_normalised():
    """agent-mcp-py's registry returns a typed PeerRecord, not a mapping;
    dict(record) raises 'not iterable'."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FakePeerRecord:
        name: str
        base_url: str
        token: str = ""

        def as_dict(self):
            return {
                "name": self.name,
                "base_url": self.base_url,
                "mcp_url": self.base_url + "/mcp/",
                "token": self.token,
            }

    record = MCPClient._as_record(FakePeerRecord("finance", "https://f", "t"), "finance")
    assert record["base_url"] == "https://f"
    assert record["token"] == "t"


def test_a_plain_dataclass_without_as_dict_still_works():
    from dataclasses import dataclass

    @dataclass
    class Bare:
        name: str
        base_url: str

    assert MCPClient._as_record(Bare("x", "https://x"), "x")["base_url"] == "https://x"


def test_an_unknown_peer_gives_a_real_error_not_a_nonetype_traceback():
    with pytest.raises(KeyError, match="missing"):
        MCPClient._as_record(None, "missing")


def test_the_endpoint_appends_the_mount_path_to_a_bare_base_url():
    from agent_runtime.mcp_client import MCP_MOUNT_PATH

    assert MCPClient._endpoint({"base_url": "https://f"}, "f") == f"https://f{MCP_MOUNT_PATH}/"
    assert MCPClient._endpoint({"base_url": "https://f/"}, "f") == f"https://f{MCP_MOUNT_PATH}/"


def test_the_endpoint_does_not_double_the_mount_path():
    """The contract says a record holds a bare base URL, but a hand-written resolver
    naturally writes the endpoint it actually curled. /mcp/mcp/ would 404 in a way
    that looks like a server fault."""
    assert MCPClient._endpoint({"base_url": "https://f/mcp"}, "f") == "https://f/mcp/"


def test_an_explicit_mcp_url_wins_over_the_base_url():
    record = {"base_url": "https://f", "mcp_url": "https://elsewhere/custom/"}
    assert MCPClient._endpoint(record, "f") == "https://elsewhere/custom/"


def test_a_record_with_no_url_at_all_is_a_clear_error():
    with pytest.raises(KeyError, match="base_url"):
        MCPClient._endpoint({"token": "t"}, "f")


# --- slow peers, and the two ways they used to take down a turn ---------------
#
# A peer tool is not an HTTP fetch: it is another agent, which makes at least one
# model call before answering. Every one of these covers a failure that looked like
# the far end being broken and was actually the client giving up on it.


def test_a_peer_is_given_a_real_deadline_rather_than_the_http_default():
    """`httpx` defaults to five seconds, which is shorter than almost any real tool.

    The symptom was a peer whose `list_tools` worked and whose every actual call
    failed — a delegated task, or anything that thinks first, never fits in five
    seconds. The number below is not sacred; being far larger than a model call is.
    """
    from agent_runtime.mcp_client import DEFAULT_TIMEOUT_S

    assert DEFAULT_TIMEOUT_S >= 300
    assert MCPClient(["finance"], resolver=REGISTRY).timeout_s == DEFAULT_TIMEOUT_S


def test_the_deadline_is_bound_into_the_default_factory_not_the_call():
    """A caller's own session factory keeps the signature it was written against."""
    factory = FakeFactory({})
    client = MCPClient(["finance"], resolver=REGISTRY, session_factory=factory, timeout_s=1.0)
    # Untouched: a fake that takes (base_url, headers) is still called that way.
    assert client._session_factory is factory


async def test_a_tool_that_outlasts_the_deadline_is_reported_not_raised():
    """The whole point of the timeout being ours.

    anyio implements a transport timeout by cancelling *this* task, so it surfaces
    as a bare `CancelledError` that `Task.cancelling()` cannot tell from a barge-in.
    Owning the deadline with `asyncio.timeout` turns it into a `TimeoutError`, which
    is unambiguous — and lets the model be told, instead of the turn dying.
    """

    class Slow:
        async def list_tools(self):
            return {"tools": FINANCE_TOOLS}

        async def call_tool(self, name, args):
            await asyncio.sleep(5)
            return {"content": [{"text": "too late"}]}

    client = MCPClient(
        ["finance"],
        resolver=REGISTRY,
        session_factory=FakeFactory({FINANCE_URL: Slow()}),
        timeout_s=0.05,
    )
    await client.list_tools()
    result = await client.call_tool("finance__get_budget", {})
    assert "did not answer within" in result
    assert result.startswith("Error running finance__get_budget")


async def test_a_real_cancellation_still_unwinds_the_turn():
    """The other half. Swallow this one and barge-in stops working."""

    class Blocking:
        async def list_tools(self):
            return {"tools": FINANCE_TOOLS}

        async def call_tool(self, name, args):
            await asyncio.sleep(60)

    client = MCPClient(
        ["finance"],
        resolver=REGISTRY,
        session_factory=FakeFactory({FINANCE_URL: Blocking()}),
        timeout_s=60.0,
    )
    await client.list_tools()
    task = asyncio.create_task(client.call_tool("finance__get_budget", {}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_transport_failure_arriving_in_a_group_is_still_a_string():
    """The MCP SDK runs its transport under a task group, so failures arrive wrapped.

    `BaseExceptionGroup` is not an `Exception`, so it walked straight past the
    catch-all and out of the broker — one dead peer, whole turn gone.
    """

    class Grouped:
        async def list_tools(self):
            return {"tools": FINANCE_TOOLS}

        async def call_tool(self, name, args):
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [OSError("boom")])

    client = MCPClient(
        ["finance"], resolver=REGISTRY, session_factory=FakeFactory({FINANCE_URL: Grouped()})
    )
    await client.list_tools()
    result = await client.call_tool("finance__get_budget", {})
    # And the leaf is what gets reported: "unhandled errors in a TaskGroup" tells a
    # model nothing it can act on.
    assert "OSError: boom" in result


async def test_a_server_that_fails_discovery_in_a_group_is_skipped_not_fatal():
    """Same wrapping, on the other method. One bad peer must not hide the good one."""

    class Grouped:
        async def list_tools(self):
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [OSError("down")])

    sessions = {FINANCE_URL: FakeSession(FINANCE_TOOLS), SPAWNER_URL: Grouped()}
    client = MCPClient(
        ["finance", "spawner"], resolver=REGISTRY, session_factory=FakeFactory(sessions)
    )
    names = [s["function"]["name"] for s in await client.list_tools()]
    assert names == ["finance__get_budget", "finance__move_money"]


async def test_teardown_never_replaces_the_runs_real_outcome():
    """A session whose transport already failed throws again on the way out.

    The caller is in a `finally` with an outcome to report, and this is not it.
    """

    class Exploding:
        async def list_tools(self):
            return {"tools": FINANCE_TOOLS}

    class ExplodingFactory(FakeFactory):
        def __call__(self, base_url, headers):
            @asynccontextmanager
            async def _open():
                self.opened.append((base_url, dict(headers)))
                try:
                    yield Exploding()
                finally:
                    raise OSError("the socket was already gone")

            return _open()

    client = MCPClient(
        ["finance"], resolver=REGISTRY, session_factory=ExplodingFactory({FINANCE_URL: None})
    )
    await client.list_tools()
    await client.aclose()  # must not raise
