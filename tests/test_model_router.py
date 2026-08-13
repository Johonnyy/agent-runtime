"""Tier resolution. Pure functions, nothing to fake."""

import pytest

from agent_runtime.model_router import (
    FALLBACK_PRICES,
    TIERS,
    UnknownTier,
    estimate_cost,
    resolve,
    tier_for,
)


def test_every_tier_resolves():
    for tier, model in TIERS.items():
        assert resolve(tier) == model


def test_three_named_tiers_exist():
    assert set(TIERS) == {"cheap", "balanced", "strong"}


def test_literal_model_id_passes_through():
    assert resolve("openai/gpt-4o-mini") == "openai/gpt-4o-mini"


def test_unknown_tier_raises_with_a_useful_message():
    with pytest.raises(UnknownTier) as exc:
        resolve("medium")
    assert "balanced" in str(exc.value)


def test_empty_raises():
    with pytest.raises(UnknownTier):
        resolve("")


def test_tier_for_is_the_reverse_lookup():
    assert tier_for(TIERS["strong"]) == "strong"
    assert tier_for("some/unknown-model") is None


def test_every_tier_has_a_fallback_price():
    for model in TIERS.values():
        assert model in FALLBACK_PRICES


def test_estimate_cost_uses_the_price_table():
    model = TIERS["balanced"]
    price_in, price_out = FALLBACK_PRICES[model]
    expected = (1_000_000 * price_in + 1_000_000 * price_out) / 1_000_000
    assert estimate_cost(model, 1_000_000, 1_000_000) == pytest.approx(expected)


def test_estimate_cost_of_an_unpriced_model_is_zero():
    # An unknown price must never inflate a cost cap into stopping a run early.
    assert estimate_cost("nobody/unknown", 10_000, 10_000) == 0.0
