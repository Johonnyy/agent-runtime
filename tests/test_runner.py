"""The loop itself, driven entirely by fakes.

No network, no API key, no MCP SDK. The fake OpenRouter client below mimics the one
thing about the OpenAI-compatible streaming protocol that actually makes this loop
hard: tool calls arrive in fragments and have to be reassembled.
"""

import asyncio

import pytest

from agent_runtime.config import Settings
from agent_runtime.model_router import TIERS
from agent_runtime.runner import AgentRunner
from agent_runtime.stop_conditions import StopOnCost, StopOnSteps

MODEL = TIERS["balanced"]


# --- fake OpenRouter ---------------------------------------------------------


class _FnDelta:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCallDelta:
    def __init__(self, index=0, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.type = "function"
        self.function = _FnDelta(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    """OpenRouter's usage payload. `cost` only exists when it reports one."""

    def __init__(self, prompt_tokens=0, completion_tokens=0, cost=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if cost is not None:
            self.cost = cost


class _Chunk:
    def __init__(self, choices=(), usage=None):
        self.choices = list(choices)
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()

    async def close(self):
        self.closed = True


class _FakeCompletions:
    def __init__(self, streams):
        self._pending = list(streams)
        self.calls = []
        self.streams = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        chunks = self._pending.pop(0) if self._pending else text_chunks("")
        stream = _FakeStream(chunks)
        self.streams.append(stream)
        return stream


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, *streams):
        self.chat = _FakeChat(_FakeCompletions(streams))

    @property
    def calls(self):
        return self.chat.completions.calls

    @property
    def streams(self):
        return self.chat.completions.streams


def text_chunks(*parts, usage=None):
    """A plain text completion, optionally with a trailing usage chunk."""
    chunks = [_Chunk([_Choice(_Delta(content=p))]) for p in parts]
    chunks.append(_Chunk([_Choice(_Delta(), finish_reason="stop")]))
    if usage is not None:
        # The usage-bearing final chunk carries no choices at all.
        chunks.append(_Chunk([], usage=usage))
    return chunks


def tool_chunks(name, *arg_fragments, call_id="call_1", text=None, usage=None):
    """A completion that requests one tool, with arguments split across chunks."""
    chunks = []
    if text:
        chunks.append(_Chunk([_Choice(_Delta(content=text))]))
    chunks.append(
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, id=call_id, name=name)]))])
    )
    for fragment in arg_fragments:
        chunks.append(
            _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, arguments=fragment)]))])
        )
    chunks.append(_Chunk([_Choice(_Delta(), finish_reason="tool_calls")]))
    if usage is not None:
        chunks.append(_Chunk([], usage=usage))
    return chunks


class FakeBroker:
    def __init__(self, results=None, tools=None):
        self._results = results or {}
        self._tools = tools if tools is not None else [_schema("get_budget")]
        self.calls = []
        self.bound = None
        self.closed = False

    def bind(self, **kwargs):
        self.bound = kwargs

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        result = self._results.get(name, "ok")
        if isinstance(result, BaseException):
            raise result
        return result

    async def aclose(self):
        self.closed = True


def _schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
        "x_agent": {"read_only": True, "requires_confirmation": False},
    }


def settings(**overrides):
    # Cost tracking off by default so tests never touch the disk.
    overrides.setdefault("feature_cost_tracking", False)
    return Settings(_env_file=None, **overrides)


def runner(client, **kwargs):
    kwargs.setdefault("settings", settings())
    return AgentRunner(client=client, **kwargs)


async def collect(stream):
    return [token async for token in stream]


# --- streaming, no tools -----------------------------------------------------


async def test_streams_text_deltas():
    client = FakeClient(text_chunks("Hello ", "there."))
    out = await collect(runner(client).stream("hi"))
    assert out == ["Hello ", "there."]


async def test_no_broker_means_no_tools_parameter():
    client = FakeClient(text_chunks("hi"))
    await collect(runner(client).stream("hi"))
    assert "tools" not in client.calls[0]


async def test_bare_prompt_becomes_a_user_message_under_the_system_prompt():
    client = FakeClient(text_chunks("hi"))
    await collect(runner(client, system_prompt="You are Amber.").stream("what's up"))

    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "You are Amber."},
        {"role": "user", "content": "what's up"},
    ]


async def test_per_call_system_overrides_the_default():
    client = FakeClient(text_chunks("hi"))
    await collect(runner(client, system_prompt="default").stream("x", system="per-turn"))
    assert client.calls[0]["messages"][0]["content"] == "per-turn"


async def test_existing_message_history_is_accepted_unchanged():
    client = FakeClient(text_chunks("hi"))
    history = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
    await collect(runner(client).stream(history))
    assert client.calls[0]["messages"] == history


async def test_usage_is_requested_so_cost_can_be_measured():
    client = FakeClient(text_chunks("hi"))
    await collect(runner(client).stream("hi"))
    assert client.calls[0]["stream_options"] == {"include_usage": True}
    assert client.calls[0]["extra_body"] == {"usage": {"include": True}}


async def test_streams_are_closed():
    client = FakeClient(text_chunks("hi"))
    await collect(runner(client).stream("hi"))
    assert all(s.closed for s in client.streams)


# --- the tool loop -----------------------------------------------------------


async def test_tool_loop_executes_then_answers():
    client = FakeClient(
        tool_chunks("get_budget", '{"quarter": "Q3"}'),
        text_chunks("You have $10 left."),
    )
    broker = FakeBroker(results={"get_budget": "$10"})

    out = await collect(runner(client, broker=broker).stream("budget?"))

    assert out == ["You have $10 left."]
    assert broker.calls == [("get_budget", {"quarter": "Q3"})]

    # The tool result went back as its own message, keyed to the call id.
    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_budget"
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "$10"}


async def test_tool_arguments_are_reassembled_across_chunks():
    # The whole reason this loop is harder than the Anthropic one it replaces.
    client = FakeClient(
        tool_chunks("get_budget", '{"quar', 'ter": ', '"Q3", "year": 2026}'),
        text_chunks("done"),
    )
    broker = FakeBroker()

    await collect(runner(client, broker=broker).stream("budget?"))

    assert broker.calls == [("get_budget", {"quarter": "Q3", "year": 2026})]


async def test_text_spoken_before_a_tool_call_still_streams():
    client = FakeClient(
        tool_chunks("get_budget", "{}", text="Let me check. "),
        text_chunks("It's $10."),
    )
    out = await collect(runner(client, broker=FakeBroker()).stream("budget?"))
    # The "\n" is the tool-boundary flush hint, on by default since the first
    # consumer of this library is a voice loop — see the dedicated tests below for
    # why, and `flush_on_tool_call=False` to stream the model's text verbatim.
    assert out == ["Let me check. ", "\n", "It's $10."]


async def test_malformed_arguments_become_an_error_result():
    client = FakeClient(tool_chunks("get_budget", '{"quarter": '), text_chunks("sorry"))
    broker = FakeBroker()

    await collect(runner(client, broker=broker).stream("budget?"))

    assert broker.calls == []  # never dispatched
    tool_message = client.calls[1]["messages"][-1]
    assert "could not parse the arguments" in tool_message["content"]


async def test_a_failing_tool_does_not_crash_the_turn():
    client = FakeClient(
        tool_chunks("get_budget", "{}"),
        text_chunks("Something went wrong."),
    )
    broker = FakeBroker(results={"get_budget": RuntimeError("upstream is down")})

    out = await collect(runner(client, broker=broker).stream("budget?"))

    assert out == ["Something went wrong."]
    assert "upstream is down" in client.calls[1]["messages"][-1]["content"]


async def test_caller_history_is_not_mutated():
    client = FakeClient(tool_chunks("get_budget", "{}"), text_chunks("done"))
    history = [{"role": "user", "content": "budget?"}]
    before = [dict(m) for m in history]

    await collect(runner(client, broker=FakeBroker()).stream(history))

    assert history == before


async def test_private_annotations_are_stripped_before_the_wire():
    client = FakeClient(text_chunks("hi"))
    await collect(runner(client, broker=FakeBroker()).stream("hi"))

    (tool,) = client.calls[0]["tools"]
    assert set(tool) == {"type", "function"}  # no x_agent


async def test_broker_is_bound_with_the_conversation_and_depth():
    client = FakeClient(text_chunks("hi"))
    broker = FakeBroker()
    await collect(runner(client, broker=broker).stream("hi", conversation_id="c1", depth=2))
    assert broker.bound == {"conversation_id": "c1", "depth": 2}


async def test_tool_discovery_failure_degrades_to_no_tools():
    class Broken(FakeBroker):
        async def list_tools(self):
            raise RuntimeError("registry down")

    client = FakeClient(text_chunks("hi"))
    out = await collect(runner(client, broker=Broken()).stream("hi"))

    assert out == ["hi"]
    assert "tools" not in client.calls[0]


# --- guardrails --------------------------------------------------------------


async def test_hard_cap_forces_a_final_tools_off_answer():
    client = FakeClient(
        tool_chunks("get_budget", "{}", call_id="a"),
        tool_chunks("get_budget", "{}", call_id="b"),
        text_chunks("Here's what I found."),
    )
    run = runner(client, broker=FakeBroker(), stop_conditions=[], settings=settings(max_steps=2))

    out = await collect(run.stream("budget?"))

    assert out == ["Here's what I found."]
    assert len(client.calls) == 3
    assert "tools" not in client.calls[2]  # the closing answer offers no tools


async def test_stop_condition_ends_the_run_with_a_final_answer():
    client = FakeClient(tool_chunks("get_budget", "{}"), text_chunks("Best I can do."))
    run = runner(client, broker=FakeBroker(), stop_conditions=[StopOnSteps(1)])

    result = await run.run("budget?")

    assert result.text == "Best I can do."
    assert result.stopped_by == "StopOnSteps(1)"
    assert len(client.calls) == 2


async def test_cost_cap_stops_dead_without_spending_again():
    client = FakeClient(
        tool_chunks("get_budget", "{}", usage=_Usage(100, 50, cost=1.0)),
        text_chunks("never reached"),
    )
    run = runner(client, broker=FakeBroker(), stop_conditions=[StopOnCost(0.5)])

    result = await run.run("budget?")

    assert result.stopped_by == "StopOnCost(0.5)"
    assert len(client.calls) == 1
    assert result.total_cost == pytest.approx(1.0)


async def test_cancellation_unwinds_the_turn():
    client = FakeClient(tool_chunks("get_budget", "{}"), text_chunks("unreachable"))
    broker = FakeBroker(results={"get_budget": asyncio.CancelledError()})

    with pytest.raises(asyncio.CancelledError):
        await collect(runner(client, broker=broker).stream("budget?"))


# --- run() -------------------------------------------------------------------


async def test_run_returns_text_cost_and_steps():
    client = FakeClient(
        tool_chunks("get_budget", "{}", usage=_Usage(100, 20, cost=0.002)),
        text_chunks("You have $10.", usage=_Usage(150, 10, cost=0.003)),
    )

    result = await runner(client, broker=FakeBroker()).run("budget?", conversation_id="c1")

    assert result.text == "You have $10."
    assert result.total_cost == pytest.approx(0.005)
    assert [s.index for s in result.steps] == [0, 1]
    assert result.steps[0].tool_calls[0]["name"] == "get_budget"
    assert result.steps[1].tokens_in == 150
    assert result.stopped_by is None


async def test_cost_falls_back_to_the_price_table_when_none_is_reported():
    # 1000 in + 1000 out on the balanced tier, priced from model_router.
    client = FakeClient(text_chunks("hi", usage=_Usage(1000, 1000)))

    result = await runner(client).run("hi")

    from agent_runtime.model_router import estimate_cost

    assert result.total_cost == pytest.approx(estimate_cost(MODEL, 1000, 1000))
    assert result.total_cost > 0


async def test_on_sentence_fires_as_each_sentence_completes():
    client = FakeClient(text_chunks("Hi there. ", "How are ", "you?"))
    spoken = []

    async def speak(sentence):
        spoken.append(sentence)

    result = await runner(client).run("hi", on_sentence=speak)

    assert spoken == ["Hi there.", "How are you?"]
    assert result.text == "Hi there. How are you?"


async def test_run_records_cost_when_tracking_is_on(tmp_path):
    from agent_runtime.cost_tracker import CostTracker

    tracker = CostTracker(str(tmp_path / "usage.db"), app_name="amber")
    client = FakeClient(text_chunks("hi", usage=_Usage(10, 5, cost=0.001)))
    run = AgentRunner(
        client=client,
        tracker=tracker,
        settings=settings(feature_cost_tracking=True),
    )
    try:
        await run.run("hi", conversation_id="c1", depth=1)
        summary = tracker.summary()
    finally:
        tracker.close()

    assert summary["calls"] == 1
    assert summary["total_cost_usd"] == pytest.approx(0.001)


async def test_model_tier_is_resolved_before_the_call():
    client = FakeClient(text_chunks("hi"))
    await collect(runner(client, model="strong").stream("hi"))
    assert client.calls[0]["model"] == TIERS["strong"]


# --- the tool-boundary flush hint ---


async def test_speech_before_a_tool_call_is_flushed_for_the_splitter():
    """A voice agent's sentence splitter holds the last sentence until trailing
    whitespace arrives. Without a hint at the tool boundary that whitespace only
    comes after the tool returns — several seconds of dead air for a web search,
    then the preamble and the answer at once."""
    client = FakeClient(
        tool_chunks("get_budget", '{"quarter": "Q3"}', text="Let me check that"),
        text_chunks(" You have $10 left."),
    )
    broker = FakeBroker(results={"get_budget": "$10"})

    out = await collect(runner(client, broker=broker).stream("budget?"))

    assert out == ["Let me check that", "\n", " You have $10 left."]


async def test_the_flush_hint_can_be_turned_off():
    client = FakeClient(
        tool_chunks("get_budget", "{}", text="Checking"),
        text_chunks(" done."),
    )
    broker = FakeBroker(results={"get_budget": "$10"})
    r = runner(client, broker=broker)
    r.flush_on_tool_call = False

    assert await collect(r.stream("budget?")) == ["Checking", " done."]


async def test_a_silent_tool_call_emits_no_stray_newline():
    """Nothing was spoken before the tool, so there is nothing to flush — a bare
    newline would be a spurious empty unit for the splitter."""
    client = FakeClient(
        tool_chunks("get_budget", "{}"),  # no text at all
        text_chunks("You have $10 left."),
    )
    broker = FakeBroker(results={"get_budget": "$10"})

    assert await collect(runner(client, broker=broker).stream("budget?")) == [
        "You have $10 left."
    ]
