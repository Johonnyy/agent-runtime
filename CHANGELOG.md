# Changelog

All notable changes to `agent-runtime` are recorded here. Because every consumer
pins this package by git tag (`agent_runtime = {git = "...", tag = "v0.1.0"}`), a
release is only real once it is tagged.

## v0.1.0 — unreleased

Initial release. Extracted from Amber's hand-rolled Anthropic tool loop and
rebuilt against OpenRouter's OpenAI-compatible endpoint.

- `AgentRunner` with both a `stream()` primitive (`AsyncIterator[str]`, a drop-in
  for Amber's `brain.think`) and a `run()` wrapper returning a `RunResult`.
- `model_router` — named tiers (`cheap` / `balanced` / `strong`) resolved in one
  place, so bumping a model is a library change, not an app change.
- `ToolBroker` protocol with three implementations: `LocalToolBroker` (in-process
  callables), `MCPClient` (remote MCP servers), `CompositeBroker` (both at once).
- `stop_conditions` — `StopOnSteps`, `StopOnCost`, and any duck-typed
  `should_stop(steps)`.
- `streaming` — the sentence splitter, moved here verbatim from Amber so every
  agent in the ecosystem gets identical streaming behaviour.
- `cost_tracker` — per-call cost logging to the host app's own SQLite database.

### Interop pass against `agent-mcp-py`

Both packages were written to the same contract independently, then run against
each other for the first time. Eight mismatches surfaced, every one of them
silent — unit suites on both sides were green throughout. Fixed here:

- **The `mcp` extra now requires v2** (`mcp>=2.0,<3`, was `>=1.2`). The v1 client
  API it was written against no longer exists: `streamablehttp_client` is gone,
  the replacement accepts no `headers=`, and it yields a transport rather than a
  `(read, write, _)` triple. Depth-guard headers now travel via a caller-owned
  `httpx2.AsyncClient`. `>=1.2` would have resolved happily and failed at the
  first remote call.
- **Field reads accept either naming style.** MCP's wire format is camelCase but
  v2's Python attributes are snake_case, so `getattr(result, "isError")` returned
  the default against every real SDK object — meaning a failed remote tool was
  reported to the model as a *success*, and every tool looked read-only-unaware.
- **`requires_confirmation` is read from `_meta`.** `ToolAnnotations` is
  `extra="ignore"`, so the flag never survives to the wire as an annotation. This
  failed open: every confirmation-gated tool looked callable without approval.
- **The depth cap is checked on the outgoing hop**, and `DepthExceeded` /
  `check_depth` are imported from `agent_mcp` when present. The old code raised
  `DepthExceeded(str)` against a shared class whose signature is
  `(depth, limit, conversation_id)` — a `TypeError` at the exact moment a depth
  refusal mattered, and only where `agent-mcp-py` was installed. The off-by-one
  also meant the sender would build a hop the receiver always rejected.
- **Peer records are normalised.** `agent_mcp.registry.resolve()` returns a typed
  `PeerRecord`, and `dict(record)` raises "not iterable"; an unknown name returned
  `None`, giving "NoneType is not iterable" instead of a usable error.
- **The MCP mount path is appended to a record's bare base URL**, with the
  trailing slash that avoids a 307 the client does not follow. Connecting to the
  base URL 404s. Appending is idempotent.

`tests/test_interop_agent_mcp.py` covers all of it against a live server, and the
unit suite gained a regression test per mismatch.
