"""Named model tiers, resolved in exactly one place.

Apps ask for a *capability level* — ``cheap``, ``balanced``, ``strong`` — never a
model string. That indirection is the whole point: when a better or cheaper model
appears, this table changes and every app calling ``AgentRunner(model="balanced")``
picks it up on its next dependency bump, with no code edits anywhere else.

The table starts as a reasonable guess. It is meant to be refined from the real
cost and quality data the usage logs collect, not fixed upfront.

A literal model id still works as an escape hatch — anything containing a ``/`` is
passed through untouched — so pinning one call site to a specific model never
requires inventing a new tier.
"""

from __future__ import annotations

# --- The table. This is the file you edit when models change. ---
TIERS: dict[str, str] = {
    "cheap": "meta-llama/llama-3.1-8b-instruct",
    "balanced": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6",
}

# Fallback pricing, USD per 1M tokens as ``(input, output)``.
#
# This is *only* consulted when OpenRouter omits the actual charge from its usage
# payload. Normally the runner asks for real credits back (``usage: {include: true}``)
# and records what was truly billed, which is always more accurate than a table that
# drifts. Treat these as order-of-magnitude guards for cost-based stop conditions,
# not as accounting.
FALLBACK_PRICES: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.03),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
}


class UnknownTier(ValueError):
    """Raised when a bare name is neither a known tier nor a model id."""


def resolve(tier_or_model: str) -> str:
    """Return the OpenRouter model id for a tier name (or pass a model id through).

    ``resolve("balanced")`` → the balanced tier's model. ``resolve("openai/gpt-4o")``
    → itself, because a ``/`` means the caller has already named a specific model.
    """
    if not tier_or_model:
        raise UnknownTier("No model or tier given.")
    if "/" in tier_or_model:
        return tier_or_model
    try:
        return TIERS[tier_or_model]
    except KeyError:
        raise UnknownTier(
            f"Unknown tier {tier_or_model!r}. Known tiers: {', '.join(sorted(TIERS))}. "
            "Pass a literal model id (containing '/') to bypass the tier table."
        ) from None


def tier_for(model_id: str) -> str | None:
    """Reverse lookup: which tier resolves to this model id, if any.

    Used for logging and usage rows, so a cost report can group by tier even though
    the wire only ever carries a model id.
    """
    for tier, model in TIERS.items():
        if model == model_id:
            return tier
    return None


def estimate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate USD spend from token counts using `FALLBACK_PRICES`.

    Returns ``0.0`` for a model with no entry — an unknown price must never inflate
    a cost cap into stopping a run early, and an under-estimate is visible in the
    logs whereas a fabricated one is not.
    """
    prices = FALLBACK_PRICES.get(model_id)
    if prices is None:
        return 0.0
    price_in, price_out = prices
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
