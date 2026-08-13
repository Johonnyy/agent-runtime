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
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# --- The depth-guard contract, borrowed from agent-mcp-py when available ---
#
# The fallback is not merely "some constants that look similar": it must be
# *signature compatible* with the real thing, because the call sites below are
# shared between both branches. An exception class whose constructor differs would
# turn a depth refusal into a TypeError only in the branch where agent-mcp-py is
# installed — i.e. only in production.
try:  # pragma: no cover - exercised by whichever branch the environment provides
    from agent_mcp.depth import (  # type: ignore[import-not-found]
        HEADER_AGENT_DEPTH,
        HEADER_CONFIRMED,
        HEADER_CONVERSATION_ID,
        MAX_AGENT_DEPTH,
        DepthExceeded,
        check_depth,
    )

    _DEPTH_SOURCE = "agent_mcp"
except Exception:  # noqa: BLE001 — agent-mcp-py is optional, by design
    HEADER_CONVERSATION_ID = "X-Conversation-Id"
    HEADER_AGENT_DEPTH = "X-Agent-Depth"
    HEADER_CONFIRMED = "X-Confirmed"
    MAX_AGENT_DEPTH = 5

    class DepthExceeded(Exception):
        """Raised when a call would exceed the ecosystem's agent-hop cap.

        Mirrors ``agent_mcp.depth.DepthExceeded`` exactly, including the base class
        and the constructor signature.
        """

        def __init__(
            self, depth: int, limit: int, conversation_id: str | None = None
        ) -> None:
            self.depth = depth
            self.limit = limit
            self.conversation_id = conversation_id
            super().__init__(
                f"agent depth {depth} exceeds the limit of {limit}"
                + (f" (conversation {conversation_id})" if conversation_id else "")
            )

    def check_depth(
        depth: int,
        *,
        limit: int = MAX_AGENT_DEPTH,
        conversation_id: str | None = None,
    ) -> None:
        """Refuse a depth at or past the cap.

        At-or-past, not merely past: a call arriving *at* the limit has no hops
        left, so admitting it would let it make one more. Identical to
        ``agent_mcp.depth.check_depth`` — the two must agree exactly or a sender
        will build requests the receiver rejects.
        """
        if depth >= limit:
            raise DepthExceeded(depth, limit, conversation_id)

    _DEPTH_SOURCE = "fallback"

# Where every server in the ecosystem mounts MCP. Same source as the depth
# constants, same reason: one definition, so a registry full of base URLs resolves
# to the same endpoint on both sides.
try:  # pragma: no cover - exercised by whichever branch the environment provides
    from agent_mcp.registry import MCP_MOUNT_PATH  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    MCP_MOUNT_PATH = "/mcp"

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


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an object or a mapping, in either naming style.

    The MCP SDK returns pydantic models; tests hand in dicts. Supporting both is
    what keeps the test suite free of the SDK entirely.

    The casing fallback is not cosmetic. MCP's *wire* format is camelCase
    (``isError``, ``readOnlyHint``, ``inputSchema``) but mcp v2's Python attributes
    are snake_case (``is_error``, ``read_only_hint``, ``input_schema``). A plain
    ``getattr(result, "isError")`` therefore returns the default against a real SDK
    object — which for ``isError`` means every failed remote tool is silently
    reported to the model as a success, and for ``readOnlyHint`` means every tool
    looks unrestricted. Both are quiet, wrong, and only happen against real
    objects, so tests using dicts would never catch them.
    """
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        alt = _camel_to_snake(name)
        return obj.get(alt, default)
    sentinel = object()
    value = getattr(obj, name, sentinel)
    if value is not sentinel:
        return value
    value = getattr(obj, _camel_to_snake(name), sentinel)
    return default if value is sentinel else value


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

    Written against the **mcp v2** API, which is not a cosmetic change from v1:
    ``streamablehttp_client`` (one word) no longer exists, the replacement
    ``streamable_http_client`` takes neither ``headers=`` nor ``timeout=``, and it
    yields a transport rather than a ``(read, write, _)`` triple. Custom headers —
    which is the entire depth-guard mechanism — are now set by owning the underlying
    ``httpx2.AsyncClient`` and handing it in.
    """
    try:
        import httpx2  # type: ignore[import-not-found]
        from mcp import Client  # type: ignore[import-not-found]
        from mcp.client.streamable_http import (  # type: ignore[import-not-found]
            streamable_http_client,
        )
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise RuntimeError(
            "Remote MCP tools need the 'mcp' extra: pip install 'agent-runtime[mcp]'. "
            "Note this package targets mcp v2 (>=2.0); the v1 client API is gone."
        ) from exc

    async with httpx2.AsyncClient(headers=headers) as http_client:
        transport = streamable_http_client(base_url, http_client=http_client)
        async with Client(transport) as client:
            yield client


#: Namespace `agent-mcp-py` publishes its flags under, in a tool's ``_meta``.
AGENT_MCP_META_PREFIX = "dev.johnny.agent-mcp"


def _tool_flags(tool: Any) -> tuple[bool, bool]:
    """Read ``(read_only, requires_confirmation)`` off a discovered tool.

    ``readOnlyHint`` is a standard MCP annotation, but **there is no standard slot
    for a confirmation flag** — and the SDK's ``ToolAnnotations`` model is
    ``extra="ignore"``, so a server that puts ``requiresConfirmation`` there has it
    silently dropped before it ever reaches the wire. `agent-mcp-py` therefore
    publishes both flags in ``_meta``, which is ``extra="allow"``.

    ``_meta`` wins where present; annotations are the fallback for servers that are
    not ours. Getting this wrong fails *open* — every tool would look safe to call
    without approval — so it is worth the two lookups.
    """
    meta = _field(tool, "_meta") or _field(tool, "meta") or {}
    annotations = _field(tool, "annotations") or {}

    def flag(meta_key: str, annotation_key: str) -> bool:
        value = _field(meta, f"{AGENT_MCP_META_PREFIX}/{meta_key}")
        if value is None:
            value = _field(annotations, annotation_key, False)
        return bool(value)

    return flag("readOnly", "readOnlyHint"), flag(
        "requiresConfirmation", "requiresConfirmation"
    )


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

        What is checked is the depth we are about to **send**, not the one we
        received. Checking the received depth would let this client build a request
        one hop past the cap, which every agent-mcp-py server then refuses — the
        refusal would cost a round trip and arrive as a remote protocol error
        instead of a local one.
        """
        check_depth(depth + 1, limit=MAX_AGENT_DEPTH, conversation_id=conversation_id)
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

    @staticmethod
    def _as_record(value: Any, server: str) -> dict[str, Any]:
        """Normalise whatever a resolver handed back into a plain dict.

        `agent-mcp-py` returns a typed ``PeerRecord`` dataclass (with ``as_dict()``),
        tests and hand-rolled resolvers return mappings, and an unknown name comes
        back as ``None``. Passing ``None`` to ``dict()`` raises "NoneType is not
        iterable", which tells nobody anything — so unknown names get a real message.
        """
        if value is None:
            raise KeyError(f"No MCP server registered as {server!r}")
        as_dict = getattr(value, "as_dict", None)
        if callable(as_dict):
            return dict(as_dict())
        if is_dataclass(value) and not isinstance(value, type):
            return dict(asdict(value))
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(
            f"Resolver for {server!r} returned {type(value).__name__}; expected a "
            "mapping, a dataclass, or an object with as_dict()"
        )

    def _resolve(self, server: str) -> dict[str, Any]:
        """Find a server's base URL and token.

        Order: an explicitly injected resolver first, then `agent-mcp-py`'s registry
        client if it is installed. The registry import is deferred to here because
        that package is optional and this is the only place that needs it.
        """
        if isinstance(self._resolver, Mapping):
            return self._as_record(self._resolver.get(server), server)
        if callable(self._resolver):
            return self._as_record(self._resolver(server), server)

        try:
            from agent_mcp.registry import resolve as registry_resolve  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cannot resolve MCP server {server!r}: pass a resolver, or install "
                "agent-mcp-py so the sync-store registry is available."
            ) from exc
        return self._as_record(registry_resolve(server), server)

    @staticmethod
    def _endpoint(record: Mapping[str, Any], server: str) -> str:
        """The URL to actually open a session against.

        A registry record stores a *base* URL (``https://finance.johnny.dev``); the
        MCP endpoint is that plus the ecosystem's fixed mount path. Connecting to
        the base URL would 404. ``agent-mcp-py`` hands us a precomputed ``mcp_url``
        when it is the resolver; otherwise the path is appended here, with the
        trailing slash that avoids the mount's 307 redirect (which the MCP client
        does not follow).
        """
        explicit = record.get("mcp_url") or record.get("url")
        if explicit:
            return str(explicit)
        base = record.get("base_url")
        if not base:
            raise KeyError(f"Registry record for {server!r} has no base_url")
        trimmed = str(base).rstrip("/")
        # Idempotent: the contract says a record holds a bare base URL, but a
        # hand-written resolver very naturally writes the endpoint it actually
        # curled. Appending blindly would produce /mcp/mcp/ and a 404 that looks
        # like a server problem.
        if not trimmed.endswith(MCP_MOUNT_PATH):
            trimmed += MCP_MOUNT_PATH
        # The trailing slash matters: a mounted MCP app answers the bare path with a
        # 307 that the MCP client does not follow.
        return trimmed + "/"

    async def _session(self, server: str) -> Any:
        if server in self._sessions:
            return self._sessions[server]

        record = self._resolve(server)
        base_url = self._endpoint(record, server)

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

                read_only, requires_confirmation = _tool_flags(tool)
                schemas.append(
                    _openai_schema(
                        namespaced,
                        _field(tool, "description", "") or "",
                        _field(tool, "inputSchema") or _field(tool, "input_schema"),
                        read_only=read_only,
                        requires_confirmation=requires_confirmation,
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
