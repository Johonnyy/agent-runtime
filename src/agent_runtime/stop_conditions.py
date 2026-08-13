"""When to stop looping.

An agentic loop without a ceiling is an open-ended bill. Stop conditions are the
ceiling, and they are deliberately plain objects rather than a framework: anything
with a ``should_stop(steps) -> bool`` method qualifies. No base class to inherit, no
registry to register with.

They are evaluated at the *step boundary* — after a step's tool results have been
appended and before the next model call is issued. That is the only honest place to
check: text already streamed to a caller (and, in a voice loop, already spoken) can
never be unsent, so a condition can end a run but can never retract one.

A condition may also declare ``allows_final_answer = False`` to mean "stop dead, do
not spend another token". `StopOnCost` does, because a budget cap that then makes
one more paid call is not a budget cap. Everything else defaults to allowing the
runner's closing tools-off completion, so the caller always gets an answer built
from whatever was gathered rather than silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Step:
    """One trip through the loop: a model call plus any tools it then requested."""

    index: int
    model: str
    text: str = ""
    # Accumulated tool calls, each ``{"id", "name", "arguments"}`` with arguments
    # still a raw JSON string — exactly as the model emitted it.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    # Real charge when the provider reported one, otherwise a `model_router` estimate.
    cost_usd: float = 0.0
    started_at: str = ""
    finished_at: str = ""


@runtime_checkable
class StopCondition(Protocol):
    """Structural type for stop conditions. Implement the method, that's all."""

    def should_stop(self, steps: list[Step]) -> bool: ...


class StopOnSteps:
    """Stop once the loop has completed ``max_steps`` model calls."""

    allows_final_answer = True

    def __init__(self, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.max_steps = max_steps

    def should_stop(self, steps: list[Step]) -> bool:
        return len(steps) >= self.max_steps

    def __repr__(self) -> str:
        return f"StopOnSteps({self.max_steps})"


class StopOnCost:
    """Stop once accumulated spend reaches ``max_cost_usd``.

    Hard stop: no closing completion, because spending more to announce that the
    budget is gone defeats the purpose.
    """

    allows_final_answer = False

    def __init__(self, max_cost_usd: float) -> None:
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        self.max_cost_usd = max_cost_usd

    def should_stop(self, steps: list[Step]) -> bool:
        return sum(s.cost_usd for s in steps) >= self.max_cost_usd

    def __repr__(self) -> str:
        return f"StopOnCost({self.max_cost_usd})"


def first_triggered(
    conditions: list[Any], steps: list[Step]
) -> Any | None:
    """Return the first condition that fires for ``steps``, or None.

    A condition that raises is treated as not firing and is left to the caller to
    log — a broken guardrail must not take the whole turn down with it.
    """
    for condition in conditions:
        try:
            if condition.should_stop(steps):
                return condition
        except Exception:  # noqa: BLE001 — a bad condition must not crash a turn
            continue
    return None
