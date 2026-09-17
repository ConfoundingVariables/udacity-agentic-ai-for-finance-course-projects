"""
Configuration settings for the SWIFT processing system.

All OpenAI-related settings are loaded from a local ``.env`` file (falling back
to real environment variables) so that secrets never live in source control.
See ``.env.template`` for the required keys.
"""

import os
from typing import Dict, Any

try:
    # Load variables from a local .env file if python-dotenv is available.
    from dotenv import load_dotenv

    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_ENV_PATH)
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


class Config:
    """Configuration class for the SWIFT processing system."""

    # --- System settings -------------------------------------------------
    MESSAGE_COUNT = 10
    BANK_COUNT = 5

    # --- Processing settings ---------------------------------------------
    MAX_WORKERS = 8
    BATCH_SIZE = 50

    # --- OpenAI / LLM settings -------------------------------------------
    # Loaded from the environment (.env). OPENAI_BASE_URL supports the
    # Vocareum-hosted OpenAI proxy used by the course.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Not every model/endpoint supports the JSON response_format parameter.
    # The Vocareum-hosted plain "gpt-4" rejects it, so we only request JSON mode
    # for models known to support it and otherwise rely on prompt instructions
    # plus robust parsing. (llm_client also auto-disables this at runtime if the
    # endpoint reports the parameter as unsupported.)
    LLM_JSON_MODE = any(
        tag in os.getenv("OPENAI_MODEL", "gpt-4o").lower()
        for tag in ("gpt-4o", "gpt-4.1", "turbo", "gpt-5", "o1", "o3")
    )

    # --- Reliability / performance settings ------------------------------
    # Retry strategy for transient LLM failures (rate limits, timeouts).
    LLM_MAX_RETRIES = 3
    LLM_RETRY_BASE_DELAY = 1.0  # seconds; grows exponentially per attempt
    LLM_REQUEST_TIMEOUT = 60    # seconds per request
    LLM_TEMPERATURE = 0.1       # low temperature for deterministic analysis

    # On-disk cache for identical LLM prompts (performance optimization).
    LLM_CACHE_ENABLED = True
    LLM_CACHE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".llm_cache"
    )

    # When True the system degrades gracefully (heuristic fallbacks) instead of
    # crashing when the OpenAI API is unreachable or no key is configured.
    ALLOW_OFFLINE_FALLBACK = True

    @classmethod
    def has_api_key(cls) -> bool:
        """Return True when a usable (non-placeholder) API key is configured."""
        key = (cls.OPENAI_API_KEY or "").strip()
        return bool(key) and key.lower() not in {"real-api-key", "your-api-key-here"}

    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary (secrets redacted)."""
        settings = {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith("_") and not callable(getattr(cls, attr))
        }
        if settings.get("OPENAI_API_KEY"):
            settings["OPENAI_API_KEY"] = "***redacted***"
        return settings
