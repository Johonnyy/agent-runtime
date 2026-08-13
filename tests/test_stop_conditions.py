"""Stop conditions. Plain objects, checked at the step boundary."""

import pytest

from agent_runtime.stop_conditions import (
    Step,
    StopCondition,
    StopOnCost,
    StopOnSteps,
    first_triggered,
)


def _steps(n, cost=0.0):
    return [Step(index=i, model="m", cost_usd=cost) for i in range(n)]


def test_stop_on_steps_fires_at_the_limit():
    condition = StopOnSteps(3)
    assert condition.should_stop(_steps(2)) is False
    assert condition.should_stop(_steps(3)) is True
    assert condition.should_stop(_steps(4)) is True


def test_stop_on_steps_rejects_a_useless_limit():
    with pytest.raises(ValueError):
        StopOnSteps(0)


def test_stop_on_cost_sums_across_steps():
    condition = StopOnCost(0.10)
    assert condition.should_stop(_steps(3, cost=0.03)) is False  # 0.09
    assert condition.should_stop(_steps(4, cost=0.03)) is True  # 0.12


def test_stop_on_cost_rejects_a_useless_limit():
    with pytest.raises(ValueError):
        StopOnCost(0)


def test_cost_cap_forbids_a_final_answer_but_step_cap_allows_one():
    # Spending more to announce the budget is gone would defeat the budget.
    assert StopOnCost(1.0).allows_final_answer is False
    assert StopOnSteps(1).allows_final_answer is True


def test_a_custom_condition_needs_no_base_class():
    class StopWhenToolWasCalled:
        def should_stop(self, steps):
            return any(s.tool_calls for s in steps)

    condition = StopWhenToolWasCalled()
    assert isinstance(condition, StopCondition)
    assert condition.should_stop(_steps(2)) is False

    called = [Step(index=0, model="m", tool_calls=[{"name": "x"}])]
    assert condition.should_stop(called) is True


def test_first_triggered_returns_the_first_match():
    never = StopOnSteps(99)
    always = StopOnSteps(1)
    assert first_triggered([never, always], _steps(1)) is always
    assert first_triggered([never], _steps(1)) is None
    assert first_triggered([], _steps(1)) is None


def test_a_broken_condition_does_not_take_down_the_turn():
    class Broken:
        def should_stop(self, steps):
            raise RuntimeError("boom")

    assert first_triggered([Broken()], _steps(1)) is None
    # ...and a working condition after it is still consulted.
    assert first_triggered([Broken(), StopOnSteps(1)], _steps(1)) is not None
