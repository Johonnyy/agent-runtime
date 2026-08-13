# agent-runtime

The shared agentic loop: call the model, execute the tools it asks for, feed the
results back, repeat until it answers — with streaming and cost tracking.

Every agent in this ecosystem needs the same loop. This is that loop, in one place,
imported in-process by Amber, the spawner, and any app that wants agent behaviour
without reimplementing token-level tool-call reassembly.

Built on OpenRouter's OpenAI-compatible endpoint via the standard `openai` client.

## Install

Consumers pin it by git tag:

```toml
agent-runtime = { git = "https://github.com/Johonnyy/agent-runtime", tag = "v0.1.0" }
```

Locally:

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"       # add ".[dev,mcp]" for remote MCP tools
pytest                        # 87 tests, no network, no API key
```

## Quickstart

```python
from agent_runtime import AgentRunner, StopOnSteps, StopOnCost

runner = AgentRunner(
    model="balanced",                      # named tier, resolved via model_router
    mcp_servers=["finance", "spawner"],    # resolved via the sync-store registry
    system_prompt="You are Johnny's finance assistant...",
    stop_conditions=[StopOnSteps(6), StopOnCost(0.10)],
)

result = await runner.run(
    "What's my remaining Q3 marketing budget?",
    conversation_id="conv_123",
    depth=0,
    on_sentence=None,                      # a voice agent passes its TTS function here
)

print(result.text, result.total_cost, len(result.steps))
```

### Two entry points

`stream()` is the primitive — an `AsyncIterator[str]` of text deltas, and nothing
else. Tool round trips happen inside; consumers just see text arriving. This is a
drop-in for any existing brain with that contract.

```python
async for token in runner.stream(conversation.messages, system=system_prompt):
    ...
```

`run()` wraps it, driving the sentence splitter and returning a `RunResult`
(`text`, `total_cost`, `steps`, `stopped_by`). Its `on_sentence` callback fires the
moment each sentence completes, so a voice agent starts speaking before the answer
has finished generating. That seam is the whole reason the callback exists.

Both accept either a bare prompt string or a full OpenAI-shaped message list, and
neither ever mutates the list you pass — tool plumbing goes on an internal copy.

## Model tiers

Apps ask for a capability level, never a model string:

```python
TIERS = {
    "cheap":    "meta-llama/llama-3.1-8b-instruct",
    "balanced": "anthropic/claude-haiku-4.5",
    "strong":   "anthropic/claude-sonnet-4.6",
}
```

Edit [`model_router.py`](src/agent_runtime/model_router.py) when better or cheaper
models appear and every app calling `AgentRunner(model="balanced")` moves with it.
This table is meant to be refined from real cost and quality data, not fixed
upfront. A literal model id containing `/` passes through untouched as an escape
hatch.

## Where tools come from

The runner talks to a `ToolBroker` — two async methods, `list_tools()` and
`call_tool()`. Three implementations ship:

| Broker | For |
|---|---|
| `LocalToolBroker` | In-process Python callables |
| `AnthropicRegistryBroker` | An existing Anthropic-shaped registry (Amber's `app/tools`) |
| `MCPClient` | Other apps' MCP servers over Streamable HTTP |
| `CompositeBroker` | All of the above at once |

`MCPClient` namespaces remote tools as `<server>__<tool>`, converts MCP schemas to
the OpenAI function shape, reads `readOnlyHint` / `requiresConfirmation`
annotations, and threads the ecosystem's depth-guard headers
(`X-Conversation-Id`, `X-Agent-Depth`, cap 5) on every call. An unreachable server
is logged and skipped rather than failing the turn, and a failing tool comes back
as text the model can react to — a bad tool never crashes a turn.

## Stop conditions

Any object with `should_stop(steps) -> bool`. No base class, no registration.
Checked at the step boundary — after tool results are appended, before the next
model call — because text already streamed to a caller can never be unsent.

`StopOnSteps` and `StopOnCost` ship. A condition may set
`allows_final_answer = False` to mean "stop dead, spend nothing more";
`StopOnCost` does, since a budget cap that then makes one more paid call is not a
budget cap. Everything else gets a closing tools-off completion so the caller
receives an answer rather than silence.

## Cost tracking

Usage stays in each app's own database — no shared central table anywhere. Point
`AGENT_RUNTIME_DB_PATH` at the database the host app already owns; the
`agent_runtime_usage` table is prefixed and WAL-enabled so it can share a file with
`agent-mcp-py`'s `agent_mcp_usage`. The columns `conversation_id`, `app_name`,
`depth`, and `created_at` are deliberately common to both, so model spend can be
joined to the tool calls that caused it.

Cost is *measured*, not estimated: the runner asks OpenRouter for the credits
actually charged (`usage: {include: true}`) and records those. The price table in
`model_router` is only a fallback for providers that don't report a charge.

`CostTracker.summary(since=...)` gives an app what it needs to answer "what has
this agent cost me" without anyone else reading its database.

## Configuration

Everything is `AGENT_RUNTIME_`-prefixed (see [.env.example](.env.example)) so it
cannot collide with a host app's own settings — this library is instantiated inside
Amber's process, where `AMBER_*` and `AGENT_MCP_*` variables are already present.
`extra="ignore"` on the settings model is load-bearing for exactly that reason.

## Testing

87 tests, zero network, zero API keys, and no MCP SDK required. The fake OpenRouter
client in [`tests/test_runner.py`](tests/test_runner.py) reproduces the one genuinely
awkward part of the OpenAI-compatible protocol: streamed tool calls arrive in
fragments — the id and name once, the JSON arguments dribbling across many chunks —
and reassembling them correctly is the loop's real work.

```bash
pytest
pytest tests/test_runner.py::test_tool_arguments_are_reassembled_across_chunks
```

## Relationship to agent-mcp-py

The dependency runs one way: `agent-mcp-py` (the server-side convention layer) never
depends on this package. This package optionally consumes `agent_mcp`'s leaf
modules — the depth constants and the registry resolver — and falls back to
identical local copies when it isn't installed, which is what lets the whole suite
run with neither `agent_mcp` nor `mcp` present.
