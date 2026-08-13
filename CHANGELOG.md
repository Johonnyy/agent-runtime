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
