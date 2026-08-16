"""agent-runtime — the shared agentic loop.

Call the model, execute the tools it asks for, feed the results back, repeat until
it answers. Streaming and cost tracking come along for the ride.

Typical use::

    from agent_runtime import AgentRunner, StopOnSteps, StopOnCost

    runner = AgentRunner(
        model="balanced",                  # named tier, resolved by model_router
        mcp_servers=["finance", "spawner"],
        system_prompt="You are Johnny's finance assistant...",
        stop_conditions=[StopOnSteps(6), StopOnCost(0.10)],
    )

    result = await runner.run("What's my remaining Q3 marketing budget?",
                              conversation_id="conv_123")
    print(result.text, result.total_cost, len(result.steps))

For a voice agent, ``runner.stream(...)`` yields text deltas directly, and
``runner.run(..., on_sentence=tts.speak)`` hands each sentence over as it completes.
"""

from __future__ import annotations

from agent_runtime.config import Settings, get_settings
from agent_runtime.cost_tracker import CostTracker, get_tracker
from agent_runtime.mcp_client import (
    MAX_AGENT_DEPTH,
    AnthropicRegistryBroker,
    CompositeBroker,
    DepthExceeded,
    LocalTool,
    LocalToolBroker,
    MCPClient,
    ToolBroker,
)
from agent_runtime.model_router import TIERS, UnknownTier, estimate_cost, resolve, tier_for
from agent_runtime.runner import AgentRunner, RunResult, RunState, get_client
from agent_runtime.stop_conditions import (
    Step,
    StopCondition,
    StopOnCost,
    StopOnSteps,
)
from agent_runtime.streaming import (
    SentenceSplitter,
    split_complete,
    stream_sentences,
    stream_to_sentences,
)

__version__ = "0.1.0"

__all__ = [
    "AgentRunner",
    "RunResult",
    "RunState",
    "Step",
    "StopCondition",
    "StopOnCost",
    "StopOnSteps",
    "ToolBroker",
    "LocalTool",
    "LocalToolBroker",
    "AnthropicRegistryBroker",
    "MCPClient",
    "CompositeBroker",
    "DepthExceeded",
    "MAX_AGENT_DEPTH",
    "SentenceSplitter",
    "split_complete",
    "stream_sentences",
    "stream_to_sentences",
    "TIERS",
    "UnknownTier",
    "resolve",
    "tier_for",
    "estimate_cost",
    "CostTracker",
    "get_tracker",
    "Settings",
    "get_settings",
    "get_client",
    "__version__",
]
