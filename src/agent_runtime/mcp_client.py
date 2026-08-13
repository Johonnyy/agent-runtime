"""Where tools come from.

The runner does not know or care whether a tool is a Python function three frames
up the stack or an HTTP round trip to another app. It talks to a `ToolBroker`: two
async methods, ``list_tools()`` and ``call_tool()``. Three implementations ship
here.

`LocalToolBroker` wraps in-process callables. This matters more than it looks:
Amber's tools (`add_task`, `web_search`, …) are ordinary Python functions, and
making her make an HTTP call to herself in order to add a task would be absurd. An
MCP-only runtime would have shipped Amber with no tools at all.

`MCPClient` is the remote case — other apps' MCP servers, reached over Streamable
HTTP, discovered through the sync store. It threads the ecosystem's depth-guard
headers on every call so agent-to-agent chains can't loop forever.

`CompositeBroker` merges them, which is the normal end state: an app's own tools
plus everything it is allowed to reach.

On the dependency direction: `agent-mcp-py` owns the depth-guard convention and the
registry, and must never depend on this package. So the constants are imported from
it when it is installed and fall back to identical local copies when it isn't —
this package stays usable with neither `agent_mcp` nor the `mcp` SDK present, which
is also what lets the whole test suite run without either.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# --- The depth-guard contract, borrowed from agent-mcp-py when available ---
try:  # pragma: no cover - exercised by whichever branch the environment provides
    from agent_mcp.depth import (  # type: ignore[import-not-found]
        HEADER_AGENT_DEPTH,
        HEADER_CONFIRMED,
        HEADER_CONVERSATION_ID,
        MAX_AGENT_DEPTH,
        DepthExceeded,
    )

    _DEPTH_SOURCE = "agent_mcp"
except Exception:  # noqa: BLE001 — agent-mcp-py is optional, by design
    HEADER_CONVERSATION_ID = "X-Conversation-Id"
    HEADER_AGENT_DEPTH = "X-Agent-Depth"
    HEADER_CONFIRMED = "X-Confirmed"
    MAX_AGENT_DEPTH = 5

    class DepthExceeded(RuntimeError):
        """Raised when a call would exceed the ecosystem's agent-hop cap."""

    _DEPTH_SOURCE = "fallback"

# Function names the OpenAI tool-calling schema will accept. Anything outside this
# is rejected by the provider, so it is worth catching at discovery time rather than
# mid-turn.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Separator for namespacing a remote tool by the server that owns it. Double
# underscore because MCP tool names are snake_case and must not contain one.
NAMESPACE_SEP = "__"


@runtime_checkable
class ToolBroker(Protocol):
    """Anything the runner can get tools from."""

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-shaped tool schemas: ``{"type": "function", "function": {...}}``."""

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return its result as text."""


# --------------------------------------------------------------------------- #
# In-process tools
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalTool:
    """One in-process tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Awaitable[str] | str]
    read_only: bool = False
    requires_confirmation: bool = False


def _openai_schema(
    name: str,
    description: str,
    parameters: dict[str, Any] | None,
    *,
    read_only: bool = False,
    requires_confirmation: bool = False,
) -> dict[str, Any]:
    """Build one entry of the OpenAI ``tools=[...]`` parameter.

    The ``x_agent`` block is ours, not the provider's — it rides along so the runner
    can see read-only/confirmation hints without a second lookup, and providers
    ignore unknown keys at the top level of a function entry.
    """
    if not _TOOL_NAME_RE.match(name):
        logger.warning(
            "Tool name %r does not match %s; the provider will likely reject it",
            name,
            _TOOL_NAME_RE.pattern,
        )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": parameters or {"type": "object", "properties": {}},
        },
        "x_agent": {
            "read_only": read_only,
            "requires_confirmation": requires_confirmation,
        },
    }


class LocalToolBroker:
    """Tools that are plain Python callables in this process.

    Errors become result *strings*, never exceptions: a broken tool must not take
    down the turn, and the model can often recover from being told what went wrong
    within the same conversation.
    """

    def __init__(self, tools: Iterable[LocalTool] = ()) -> None:
        self._tools: dict[str, LocalTool] = {t.name: t for t in tools}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Awaitable[str] | str],
        *,
        read_only: bool = False,
        requires_confirmation: bool = False,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name!r}")
        self._tools[name] = LocalTool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            read_only=read_only,
            requires_confirmation=requires_confirmation,
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            _openai_schema(
                t.name,
                t.description,
                t.parameters,
                read_only=t.read_only,
                requires_confirmation=t.requires_confirmation,
            )
            for t in self._tools.values()
        ]

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("Tool unknown: %s", name)
            return f"Error: tool {name!r} is not available."
        try:
            result = tool.func(**(args or {}))
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except asyncio.CancelledError:
            raise  # an interrupt must unwind the turn
        except Exception as exc:  # noqa: BLE001 — a tool must not crash the turn
            logger.exception("Tool %s failed", name)
            return f"Error running {name}: {exc}"


class AnthropicRegistryBroker:
    """Adapter for an existing Anthropic-shaped tool registry.

    Amber already has one — ``get_tool_schemas()`` returning
    ``{"name", "description", "input_schema"}`` and an async ``run_tool(name, input)``
    that never raises. This converts that surface to a `ToolBroker` so her refactor
    is two lines rather than a rewrite of every tool.

    ``schemas`` may be a list or a zero-argument callable; a callable is re-invoked
    on every run, which preserves registries whose tool availability changes with
    configuration at runtime.
    """

    def __init__(
        self,
        schemas: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
        dispatch: Callable[[str, dict[str, Any]], Awaitable[str]],
    ) -> None:
        self._schemas = schemas
        self._dispatch = dispatch

    async def list_tools(self) -> list[dict[str, Any]]:
        raw = self._schemas() if callable(self._schemas) else self._schemas
        return [
            _openai_schema(
                s["name"],
                s.get("description", ""),
                s.get("input_schema") or s.get("parameters"),
                read_only=bool(s.get("read_only", False)),
                requires_confirmation=bool(s.get("requires_confirmation", False)),
            )
            for s in raw
        ]

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            return str(await self._dispatch(name, args))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s failed", name)
            return f"Error running {name}: {exc}"


# --------------------------------------------------------------------------- #
# Remote tools, over MCP
# --------------------------------------------------------------------------- #


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an object or a mapping.

    The MCP SDK returns pydantic models; tests hand in dicts. Supporting both is
    what keeps the test suite free of the SDK entirely.
    """
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _result_text(result: Any) -> str:
    """Flatten an MCP tool result into text the model can read.

    An ``isError`` result is returned as text too, not raised. The model handles
    "that lookup failed because X" perfectly well; an exception here would end the
    turn instead.
    """
    content = _field(result, "content") or []
    parts: list[str] = []
    for block in content:
        text = _field(block, "text")
        if text:
            parts.append(str(text))
            continue
        data = _field(block, "data")
        if data:
            parts.append(str(data))
    text = "\n".join(parts).strip()
    if _field(result, "isError", False):
        return f"Error: {text or 'the tool reported a failure with no detail.'}"
    return text


@asynccontextmanager
async def _default_session_factory(base_url: str, headers: dict[str, str]):
    """Open a real MCP Streamable HTTP session.

    Imported lazily so the `mcp` extra is genuinely optional — a host app using only
    in-process tools never needs the SDK installed.
    """
    try:
        from mcp import ClientSession  # type: ignore[import-not-found]
        from mcp.client.streamable_http import (  # type: ignore[import-not-found]
            streamablehttp_client,
        )
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise RuntimeError(
            "Remote MCP tools need the 'mcp' extra: pip install 'agent-runtime[mcp]'"
        ) from exc

    async with streamablehttp_client(base_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@dataclass
class _ServerTools:
    server: str
    # namespaced name -> original tool name on that server
    names: dict[str, str] = field(default_factory=dict)


class MCPClient:
    """Talks to other apps' MCP servers as a client.

    Sessions are opened lazily and cached for the life of the client, so a run that
    calls three tools on one server pays for one connection. `aclose()` tears them
    down; the runner calls it when it owns the client.
    """

    def __init__(
        self,
        servers: Iterable[str],
        *,
        resolver: Callable[[str], dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.servers = list(servers)
        self._resolver = resolver
        self._session_factory = session_factory or _default_session_factory
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}
        self._owners: dict[str, tuple[str, str]] = {}
        self._conversation_id: str | None = None
        self._depth = 0
        self._confirmed = False

    # --- run binding -------------------------------------------------------

    def bind(
        self,
        *,
        conversation_id: str | None = None,
        depth: int = 0,
        confirmed: bool = False,
    ) -> None:
        """Attach this run's conversation id and depth to every outbound call.

        The check is done here rather than per call so an over-deep chain fails
        before any request is paid for. Outbound calls carry ``depth + 1``: this
        client is itself one hop.
        """
        if depth + 1 > MAX_AGENT_DEPTH:
            raise DepthExceeded(
                f"Agent depth {depth + 1} exceeds the cap of {MAX_AGENT_DEPTH}."
            )
        self._conversation_id = conversation_id
        self._depth = depth
        self._confirmed = confirmed

    def _headers(self) -> dict[str, str]:
        headers = {HEADER_AGENT_DEPTH: str(self._depth + 1)}
        if self._conversation_id:
            headers[HEADER_CONVERSATION_ID] = self._conversation_id
        if self._confirmed:
            headers[HEADER_CONFIRMED] = "true"
        return headers

    # --- discovery ---------------------------------------------------------

    def _resolve(self, server: str) -> dict[str, Any]:
        """Find a server's base URL and token.

        Order: an explicitly injected resolver first, then `agent-mcp-py`'s registry
        client if it is installed. The registry import is deferred to here because
        that package is optional and this is the only place that needs it.
        """
        if isinstance(self._resolver, Mapping):
            record = self._resolver.get(server)
            if record is None:
                raise KeyError(f"No MCP server registered as {server!r}")
            return dict(record)
        if callable(self._resolver):
            return dict(self._resolver(server))

        try:
            from agent_mcp.registry import resolve as registry_resolve  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cannot resolve MCP server {server!r}: pass a resolver, or install "
                "agent-mcp-py so the sync-store registry is available."
            ) from exc
        return dict(registry_resolve(server))

    async def _session(self, server: str) -> Any:
        if server in self._sessions:
            return self._sessions[server]

        record = self._resolve(server)
        base_url = record.get("base_url") or record.get("url")
        if not base_url:
            raise KeyError(f"Registry record for {server!r} has no base_url")

        headers = self._headers()
        token = record.get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        if self._stack is None:
            self._stack = AsyncExitStack()
        session = await self._stack.enter_async_context(
            self._session_factory(base_url, headers)
        )
        self._sessions[server] = session
        return session

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover every reachable tool, namespaced by the server that owns it.

        A server that can't be reached is logged and skipped rather than failing the
        whole run — losing one app's tools is better than losing the turn.
        """
        schemas: list[dict[str, Any]] = []
        self._owners.clear()

        for server in self.servers:
            try:
                session = await self._session(server)
                listing = await session.list_tools()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("MCP server %s unreachable: %s", server, exc)
                continue

            for tool in _field(listing, "tools") or []:
                original = _field(tool, "name")
                if not original:
                    continue
                namespaced = f"{server}{NAMESPACE_SEP}{original}"
                if namespaced in self._owners:
                    logger.warning("Duplicate tool name %s; keeping the first", namespaced)
                    continue
                self._owners[namespaced] = (server, original)

                annotations = _field(tool, "annotations") or {}
                schemas.append(
                    _openai_schema(
                        namespaced,
                        _field(tool, "description", "") or "",
                        _field(tool, "inputSchema") or _field(tool, "input_schema"),
                        read_only=bool(_field(annotations, "readOnlyHint", False)),
                        requires_confirmation=bool(
                            _field(annotations, "requiresConfirmation", False)
                        ),
                    )
                )
        return schemas

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        owner = self._owners.get(name)
        if owner is None:
            # The model may echo an un-namespaced name; try to place it.
            server, _, original = name.partition(NAMESPACE_SEP)
            if not original or server not in self.servers:
                return f"Error: tool {name!r} is not available."
            owner = (server, original)

        server, original = owner
        try:
            session = await self._session(server)
            result = await session.call_tool(original, args or {})
            return _result_text(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a tool must not crash the turn
            logger.exception("MCP tool %s failed", name)
            return f"Error running {name}: {exc}"

    async def aclose(self) -> None:
        """Close every cached session."""
        stack, self._stack = self._stack, None
        self._sessions.clear()
        if stack is not None:
            await stack.aclose()


# --------------------------------------------------------------------------- #
# Both at once
# --------------------------------------------------------------------------- #


class CompositeBroker:
    """Presents several brokers to the runner as one.

    The normal shape for a real agent: its own in-process tools plus every remote
    app it's allowed to reach. On a name collision the first broker wins and the
    clash is logged — silently shadowing a tool would be a genuinely nasty bug to
    track down later.
    """

    def __init__(self, brokers: Iterable[ToolBroker]) -> None:
        self.brokers = list(brokers)
        self._owners: dict[str, ToolBroker] = {}

    async def list_tools(self) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        self._owners.clear()
        for broker in self.brokers:
            for schema in await broker.list_tools():
                name = schema["function"]["name"]
                if name in self._owners:
                    logger.warning(
                        "Tool name collision on %r; keeping the first broker's version",
                        name,
                    )
                    continue
                self._owners[name] = broker
                merged.append(schema)
        return merged

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        broker = self._owners.get(name)
        if broker is None:
            return f"Error: tool {name!r} is not available."
        return await broker.call_tool(name, args)

    def bind(self, **kwargs: Any) -> None:
        for broker in self.brokers:
            bind = getattr(broker, "bind", None)
            if bind is not None:
                bind(**kwargs)

    async def aclose(self) -> None:
        for broker in self.brokers:
            close = getattr(broker, "aclose", None)
            if close is not None:
                await close()
