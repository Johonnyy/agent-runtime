"""The loop.

Call the model, watch the stream for tool calls, execute them, feed the results
back, repeat until the model answers in plain text or a stop condition fires. That
is the entire idea, and it is the same idea every agent in this ecosystem needs —
which is exactly why it lives in one library instead of being reimplemented per app.

Two entry points, deliberately:

``stream()`` is the primitive. It yields text deltas and nothing else — an
``AsyncIterator[str]`` — which is the contract Amber's voice pipeline is already
built around. Tool use is invisible downstream: the model may make five round trips
before answering, and the consumer just sees text arriving.

``run()`` wraps it, driving the sentence splitter and returning a `RunResult` with
the text, the spend, and the steps. Its ``on_sentence`` callback is the seam that
matters for a voice agent: the first sentence is handed over for synthesis while
the rest of the answer is still generating.

The provider is OpenRouter, reached with the standard `openai` client because
OpenRouter speaks the OpenAI wire protocol. One consequence shapes this file more
than anything else: on that protocol, streamed tool calls arrive in *fragments*.
The id and function name appear once, the JSON arguments dribble in across many
chunks, and it is on us to reassemble them. (The Anthropic SDK this loop was
originally written against handed back finished tool blocks, which hid the problem.)
Malformed reassembled JSON is reported to the model as a tool error rather than
raised — the model can usually fix its own call on the next step.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from agent_runtime.config import Settings, get_settings
from agent_runtime.cost_tracker import CostTracker, get_tracker
from agent_runtime.mcp_client import MCPClient, ToolBroker
from agent_runtime.model_router import estimate_cost, resolve
from agent_runtime.stop_conditions import Step, StopOnSteps, first_triggered
from agent_runtime.streaming import stream_to_sentences

logger = logging.getLogger(__name__)

Messages = Sequence[dict[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@lru_cache
def _build_client(
    api_key: str, base_url: str, timeout: float, referer: str, title: str
) -> Any:
    """One client per distinct configuration, so the pool is still shared."""
    from openai import AsyncOpenAI

    headers: dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        default_headers=headers or None,
    )


def get_client(settings: Settings | None = None) -> Any:
    """The OpenRouter client for `settings`, pooled per configuration.

    Takes the settings rather than reading `get_settings()` unconditionally. This
    library is imported *into* another app's process, and a host that configures
    it by injection — ``AgentRunner(settings=Settings(_env_file=None, ...))``, the
    documented pattern — never sets ``AGENT_RUNTIME_*`` in the environment. Reading
    the module-level singleton here meant the key the host injected was used for
    everything except the one thing that needs it: the HTTP request.

    A missing key raises rather than defaulting to a placeholder. The placeholder
    reached the provider as a real ``Authorization`` header and came back as a bare
    401 from OpenRouter — a misconfiguration that reads like an outage.
    """
    settings = settings or get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "No OpenRouter API key configured. Set AGENT_RUNTIME_OPENROUTER_API_KEY, "
            "or pass the key through to AgentRunner(settings=Settings(...)) if the "
            "host app configures this library by injection."
        )
    return _build_client(
        settings.openrouter_api_key,
        settings.openrouter_base_url,
        settings.request_timeout_s,
        settings.referer,
        settings.title,
    )


@dataclass
class RunResult:
    """Everything a caller might want to know after `AgentRunner.run`."""

    text: str
    total_cost: float = 0.0
    steps: list[Step] = field(default_factory=list)
    # Which stop condition ended the run, if one did. ``None`` means the model
    # simply finished answering.
    stopped_by: str | None = None


@dataclass
class _RunState:
    """Mutable scratch shared between the generator and its `run()` wrapper."""

    steps: list[Step] = field(default_factory=list)
    stopped_by: str | None = None


@asynccontextmanager
async def _closing(stream: Any):
    """Close a stream on the way out, however it spells 'close'.

    A `finally` rather than `async with` because the fakes in the test suite are
    plain async iterators, and requiring them to implement the SDK's context
    manager protocol would be testing the fake instead of the loop.
    """
    try:
        yield stream
    finally:
        close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
        if close is not None:
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 — closing must not mask the real error
                logger.debug("Ignoring error while closing the model stream", exc_info=True)


class AgentRunner:
    """Runs an agentic loop against OpenRouter, with tools and guardrails.

    Everything expensive or stateful is injectable — the client, the tool broker,
    the cost tracker, the settings — so the whole loop is testable with no network,
    no API key, and no MCP SDK.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        mcp_servers: Iterable[str] | None = None,
        broker: ToolBroker | None = None,
        system_prompt: str | None = None,
        stop_conditions: list[Any] | None = None,
        max_tokens: int | None = None,
        client: Any | None = None,
        tracker: CostTracker | None = None,
        settings: Settings | None = None,
        flush_on_tool_call: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.default_tier
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens or self.settings.max_tokens
        # Emit a newline when the model stops speaking to call a tool. On by
        # default because the first consumer of this library is a voice loop and
        # the alternative is audible dead air; it is whitespace, which a sentence
        # splitter consumes. Pass False for a consumer that wants the model's text
        # byte-for-byte.
        self.flush_on_tool_call = flush_on_tool_call

        self._client = client
        self._tracker = tracker
        self._owns_broker = False

        if broker is None and mcp_servers:
            broker = MCPClient(mcp_servers)
            self._owns_broker = True
        self.broker = broker

        # `max_steps` and `StopOnSteps` are two names for one control. The setting
        # is the default condition when none is given, *and* an unconditional
        # backstop — otherwise `stop_conditions=[]` would be a licence to spin.
        self.stop_conditions = (
            list(stop_conditions)
            if stop_conditions is not None
            else [StopOnSteps(self.settings.max_steps)]
        )
        declared = [getattr(c, "max_steps", 0) for c in self.stop_conditions]
        self._hard_cap = max(self.settings.max_steps, *declared) if declared else self.settings.max_steps

    # --- public API --------------------------------------------------------

    def stream(
        self,
        prompt_or_messages: str | Messages,
        *,
        system: str | None = None,
        conversation_id: str | None = None,
        depth: int = 0,
    ) -> AsyncIterator[str]:
        """Stream the answer as text deltas.

        A drop-in for any existing ``AsyncIterator[str]`` brain: tool round trips
        happen inside, and only spoken text comes out.
        """
        return self._iterate(
            prompt_or_messages,
            system=system,
            conversation_id=conversation_id,
            depth=depth,
            state=_RunState(),
        )

    async def run(
        self,
        prompt_or_messages: str | Messages,
        *,
        system: str | None = None,
        conversation_id: str | None = None,
        depth: int = 0,
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
    ) -> RunResult:
        """Run to completion, returning text, spend, and the steps taken.

        ``on_sentence`` receives each sentence the moment it completes — pass a
        text-to-speech function here and the answer starts playing before it has
        finished generating.
        """
        state = _RunState()
        tokens = self._iterate(
            prompt_or_messages,
            system=system,
            conversation_id=conversation_id,
            depth=depth,
            state=state,
        )
        text = await stream_to_sentences(tokens, on_sentence)

        # Cost logging is off the latency path on purpose: the caller already has
        # every sentence by now, so a slow disk delays nobody.
        await self._record(state.steps, conversation_id=conversation_id, depth=depth)

        return RunResult(
            text=text,
            total_cost=sum(s.cost_usd for s in state.steps),
            steps=state.steps,
            stopped_by=state.stopped_by,
        )

    # --- the loop ----------------------------------------------------------

    async def _iterate(
        self,
        prompt_or_messages: str | Messages,
        *,
        system: str | None,
        conversation_id: str | None,
        depth: int,
        state: _RunState,
    ) -> AsyncIterator[str]:
        working = self._build_messages(prompt_or_messages, system)
        model_id = resolve(self.model)

        broker = self.broker
        if broker is not None:
            bind = getattr(broker, "bind", None)
            if bind is not None:
                bind(conversation_id=conversation_id, depth=depth)

        tools = await self._tool_schemas(broker)

        try:
            for index in range(self._hard_cap):
                started = _now()
                text_parts: list[str] = []
                calls: dict[int, dict[str, Any]] = {}
                usage: Any = None

                stream = await self._create(model_id, working, tools)
                async with _closing(stream):
                    async for chunk in stream:
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            usage = chunk_usage
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            # The usage-bearing final chunk carries no choices.
                            continue
                        delta = getattr(choices[0], "delta", None)
                        if delta is None:
                            continue
                        content = getattr(delta, "content", None)
                        if content:
                            text_parts.append(content)
                            yield content
                        self._accumulate_tool_calls(calls, getattr(delta, "tool_calls", None))

                tokens_in, tokens_out, cost = self._usage(usage, model_id)
                ordered = [calls[i] for i in sorted(calls)]
                state.steps.append(
                    Step(
                        index=index,
                        model=model_id,
                        text="".join(text_parts),
                        tool_calls=ordered,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=cost,
                        started_at=started,
                        finished_at=_now(),
                    )
                )

                if not ordered:
                    return  # the model answered; we're done

                if self.flush_on_tool_call and any(text_parts):
                    # A flush hint for a downstream sentence splitter. When the model
                    # speaks before calling a tool ("let me check that"), those words
                    # are a complete spoken unit and should reach TTS *before* the
                    # tool runs. Without this the splitter holds the last sentence
                    # waiting for trailing whitespace that won't arrive until the
                    # tool returns — for a web search, several seconds of dead air
                    # followed by the preamble and the answer at once.
                    yield "\n"

                working.append(self._assistant_turn(state.steps[-1]))
                for call in ordered:
                    working.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"] or call["name"],
                            "content": await self._execute(broker, call),
                        }
                    )

                triggered = first_triggered(self.stop_conditions, state.steps)
                if triggered is not None:
                    state.stopped_by = repr(triggered)
                    logger.info("Stopping: %s", state.stopped_by)
                    if getattr(triggered, "allows_final_answer", True):
                        async for text in self._final_answer(model_id, working):
                            yield text
                    return

            # Hard cap reached while still calling tools. Answer with what we have,
            # tools off, so the caller never gets silence.
            logger.warning(
                "Tool loop hit the hard cap of %d steps; forcing a final answer",
                self._hard_cap,
            )
            state.stopped_by = f"hard_cap({self._hard_cap})"
            async for text in self._final_answer(model_id, working):
                yield text
        finally:
            if self._owns_broker and broker is not None:
                close = getattr(broker, "aclose", None)
                if close is not None:
                    await close()

    async def _final_answer(self, model_id: str, working: list[dict[str, Any]]) -> AsyncIterator[str]:
        """One closing completion with no tools offered."""
        stream = await self._create(model_id, working, [])
        async with _closing(stream):
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    yield content

    # --- helpers -----------------------------------------------------------

    def _build_messages(
        self, prompt_or_messages: str | Messages, system: str | None
    ) -> list[dict[str, Any]]:
        """Normalise the input and prepend the system prompt.

        Accepts a bare prompt string or a full message history. The copy is what
        keeps a caller's conversation clean: the loop appends tool plumbing to
        `working`, and the caller only ever records the spoken text.
        """
        if isinstance(prompt_or_messages, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = list(prompt_or_messages)

        prompt = system if system is not None else self.system_prompt
        if prompt and not (messages and messages[0].get("role") == "system"):
            return [{"role": "system", "content": prompt}, *messages]
        return messages

    async def _tool_schemas(self, broker: ToolBroker | None) -> list[dict[str, Any]]:
        if broker is None:
            return []
        try:
            return await broker.list_tools()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — no tools beats no answer
            logger.exception("Tool discovery failed; continuing without tools")
            return []

    async def _create(
        self, model_id: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        client = self._client or get_client(self.settings)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
            # Without this the streamed response carries no usage at all, and
            # `total_cost` would be permanently zero.
            "stream_options": {"include_usage": True},
            # OpenRouter-specific: return the credits actually charged, so cost is
            # measured rather than estimated from a price table that will drift.
            "extra_body": {"usage": {"include": True}},
        }
        if tools:
            kwargs["tools"] = [self._api_tool(t) for t in tools]
        return await client.chat.completions.create(**kwargs)

    @staticmethod
    def _api_tool(schema: dict[str, Any]) -> dict[str, Any]:
        """Strip our private annotations before the schema goes over the wire."""
        return {"type": schema.get("type", "function"), "function": schema["function"]}

    @staticmethod
    def _accumulate_tool_calls(
        calls: dict[int, dict[str, Any]], deltas: Any
    ) -> None:
        """Reassemble streamed tool calls.

        Each delta carries an index; ``id`` and ``function.name`` show up once,
        ``function.arguments`` arrives in fragments that must be concatenated in
        order. This is the part the OpenAI-compatible protocol makes the caller's
        problem.
        """
        for delta in deltas or []:
            index = getattr(delta, "index", 0) or 0
            call = calls.setdefault(index, {"id": None, "name": None, "arguments": ""})
            call_id = getattr(delta, "id", None)
            if call_id:
                call["id"] = call_id
            function = getattr(delta, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if name:
                call["name"] = name
            arguments = getattr(function, "arguments", None)
            if arguments:
                call["arguments"] += arguments

    @staticmethod
    def _assistant_turn(step: Step) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": step.text or None,
            "tool_calls": [
                {
                    "id": call["id"] or call["name"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"] or "{}",
                    },
                }
                for call in step.tool_calls
            ],
        }

    async def _execute(self, broker: ToolBroker | None, call: dict[str, Any]) -> str:
        name = call.get("name")
        if not name:
            return "Error: the model requested a tool with no name."
        if broker is None:
            return f"Error: tool {name!r} is not available."

        raw = (call.get("arguments") or "").strip()
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            logger.warning("Unparseable arguments for %s: %s", name, raw)
            return f"Error: could not parse the arguments for {name} ({exc})."
        if not isinstance(args, dict):
            return f"Error: the arguments for {name} must be a JSON object."

        logger.info("Tool call: %s(%s)", name, args)
        try:
            return str(await broker.call_tool(name, args))
        except asyncio.CancelledError:
            raise  # interrupt/barge-in — let the turn unwind
        except Exception as exc:  # noqa: BLE001 — a tool must not crash the turn
            logger.exception("Tool %s failed", name)
            return f"Error running {name}: {exc}"

    @staticmethod
    def _usage(usage: Any, model_id: str) -> tuple[int, int, float]:
        """Pull tokens and USD out of a usage payload.

        OpenRouter returns the real charge as a non-standard ``cost`` field, which
        survives because the `openai` SDK's models allow extra keys. When it is
        missing — a different provider, an older response — fall back to the price
        table, which is an estimate and labelled as one.
        """
        if usage is None:
            return 0, 0, 0.0
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = getattr(usage, "cost", None)
        if cost is None:
            return tokens_in, tokens_out, estimate_cost(model_id, tokens_in, tokens_out)
        return tokens_in, tokens_out, float(cost)

    async def _record(
        self, steps: list[Step], *, conversation_id: str | None, depth: int
    ) -> None:
        if not steps or not self.settings.feature_cost_tracking:
            return
        try:
            # Inside the try, not above it. Opening the database is the part most
            # likely to fail — a missing directory, a path the process cannot write —
            # and it used to sit outside this block, so the one line that raised was
            # the one line not covered by a guard whose whole purpose is the sentence
            # below. A completed run, every tool in it successful, failed at the last
            # step with `OperationalError: unable to open database file`.
            tracker = self._tracker or get_tracker(self.settings)
            await asyncio.to_thread(
                tracker.record_many, steps, conversation_id=conversation_id, depth=depth
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — bookkeeping must not fail a completed run
            logger.exception("Could not record usage")
