"""Central configuration for the agent runtime.

Every model choice, key, and cap flows through here so a host app can retune the
loop without touching call sites. Nothing in this package hardcodes a model string
or a price — import `get_settings()` instead.

Values come from environment variables (prefix ``AGENT_RUNTIME_``) and an optional
`.env` file. The prefix matters more than usual: this library is imported *into*
another app's process — Amber's, the spawner's — where ``AMBER_*`` and
``AGENT_MCP_*`` variables are already in the environment. ``extra="ignore"`` is
therefore load-bearing, not decoration: without it, instantiating `Settings` inside
Amber would raise on the first unrelated variable it happened to see.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_RUNTIME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenRouter ---
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key. Empty is fine for tests, which never call out.",
    )
    # Overridable so a proxy — or a local OpenAI-compatible server — can stand in
    # without any code change.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # OpenRouter's attribution headers (HTTP-Referer / X-Title). Optional; they only
    # affect how the app is credited on OpenRouter's public leaderboards.
    referer: str = ""
    title: str = ""

    # --- Identity ---
    # Stamped into every usage row. When several apps share one database file, this
    # is what keeps spend attributable.
    app_name: str = "unknown"

    # --- Model defaults ---
    # Named tier, resolved by `model_router`. Never a literal model id here: the
    # whole point of the tier table is that one edit moves every app at once.
    default_tier: str = "balanced"
    # Cap on a single completion. Modest by default because the first consumer is a
    # voice loop, where a runaway generation would stream for minutes.
    max_tokens: int = 1024
    # Backstop on tool round trips per run. A model that loops on tools burns money
    # silently, so the loop always has a ceiling even with `stop_conditions=[]`.
    max_steps: int = 5
    # Per-request timeout. Generous enough for a slow tool-heavy step, short enough
    # that a hung provider doesn't hold a socket open forever.
    request_timeout_s: float = 60.0

    # --- Cost tracking ---
    # Deliberately a *local* path: usage stays in each app's own database, never a
    # shared central table. Host apps point this at the DB they already own (Amber
    # sets it to `amber.db`); table names are prefixed so co-tenancy is safe.
    db_path: str = "agent_runtime.db"
    # When false, no rows are written and no database file is created. Lets the loop
    # run with no disk at all.
    feature_cost_tracking: bool = True


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the `.env` file is parsed once. Tests clear it with
    ``get_settings.cache_clear()``, or sidestep it entirely by constructing
    ``Settings(_env_file=None, **overrides)`` and injecting the result.
    """
    return Settings()
