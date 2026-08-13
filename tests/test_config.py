"""Settings, and the one property that really matters: not exploding in someone
else's process.
"""

from agent_runtime.config import Settings, get_settings


def test_defaults_are_sane():
    settings = Settings(_env_file=None)
    assert settings.default_tier == "balanced"
    assert settings.openrouter_base_url.endswith("/api/v1")
    assert settings.max_steps >= 1
    assert settings.db_path.endswith(".db")


def test_env_prefix_is_honoured(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_MAX_STEPS", "9")
    monkeypatch.setenv("AGENT_RUNTIME_DEFAULT_TIER", "strong")
    settings = Settings(_env_file=None)
    assert settings.max_steps == 9
    assert settings.default_tier == "strong"


def test_foreign_env_vars_are_ignored(monkeypatch):
    # This library is instantiated inside Amber's process, where AMBER_* and
    # AGENT_MCP_* variables are everywhere. extra="ignore" is what stops that from
    # raising on import.
    monkeypatch.setenv("AMBER_OPENAI_API_KEY", "sk-not-ours")
    monkeypatch.setenv("AMBER_FEATURE_TOOLS", "false")
    monkeypatch.setenv("AGENT_MCP_AUTH_SECRET", "also-not-ours")
    settings = Settings(_env_file=None)
    assert settings.openrouter_api_key == ""


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
