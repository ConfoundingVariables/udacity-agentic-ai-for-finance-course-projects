"""
Shared LLM client utilities.

This module centralises OpenAI access for every agent pattern so that the
base URL (Vocareum proxy), API key, model selection, retry strategy, response
caching and offline fallback logic all live in ONE place.

Design goals
------------
* Performance   -> on-disk response cache keyed by the full prompt.
* Reliability   -> automatic retry with exponential backoff for transient
                   failures (rate limits, timeouts, network blips).
* Resilience    -> optional offline fallback so the pipeline still completes
                   (and still produces reports) when the API is unreachable or
                   no API key is configured.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

# A single shared client instance (created lazily).
_CLIENT: Optional[OpenAI] = None

# Whether the endpoint accepts response_format={"type": "json_object"}.
# Seeded from Config and auto-disabled at runtime if the API rejects it.
_JSON_MODE_SUPPORTED: bool = Config.LLM_JSON_MODE


def _parse_json_content(content: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from an LLM response.

    Handles clean JSON, markdown-fenced JSON (```json ... ```), and responses
    with leading/trailing prose by extracting the outermost brace-delimited
    object.
    """
    text = (content or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present.
    if "```" in text:
        fenced = text.split("```")
        for chunk in fenced:
            chunk = chunk.lstrip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:]
            chunk = chunk.strip()
            if chunk.startswith("{"):
                text = chunk
                break

    # Extract the outermost {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _is_json_format_error(exc: Exception) -> bool:
    """True when the endpoint rejected the response_format parameter."""
    msg = str(exc).lower()
    return "response_format" in msg and ("not supported" in msg or "unsupported" in msg)


def _is_non_transient(exc: Exception) -> bool:
    """True for client errors that will not succeed on retry (4xx except 429)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        # Fall back to string inspection for wrapped errors.
        text = str(exc)
        if "429" in text:
            return False
        return any(code in text for code in ("400", "401", "403", "404", "422"))
    if status == 429:
        return False
    return 400 <= status < 500


def get_client() -> OpenAI:
    """Return a lazily-initialised, shared OpenAI client.

    Honours ``OPENAI_BASE_URL`` (Vocareum) and ``OPENAI_API_KEY`` from Config.
    """
    global _CLIENT
    if _CLIENT is None:
        kwargs: Dict[str, Any] = {}
        if Config.OPENAI_API_KEY:
            kwargs["api_key"] = Config.OPENAI_API_KEY
        if Config.OPENAI_BASE_URL:
            kwargs["base_url"] = Config.OPENAI_BASE_URL
        kwargs["timeout"] = Config.LLM_REQUEST_TIMEOUT
        kwargs["max_retries"] = 0  # we handle retries ourselves
        _CLIENT = OpenAI(**kwargs)
    return _CLIENT


def _cache_key(model: str, system_prompt: str, user_prompt: str,
               temperature: float) -> str:
    """Build a stable cache key from the request parameters."""
    raw = json.dumps(
        {"model": model, "system": system_prompt, "user": user_prompt,
         "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_read(key: str) -> Optional[Dict[str, Any]]:
    if not Config.LLM_CACHE_ENABLED:
        return None
    path = os.path.join(Config.LLM_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_write(key: str, value: Dict[str, Any]) -> None:
    if not Config.LLM_CACHE_ENABLED:
        return
    try:
        os.makedirs(Config.LLM_CACHE_DIR, exist_ok=True)
        path = os.path.join(Config.LLM_CACHE_DIR, f"{key}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh)
    except OSError as exc:  # pragma: no cover - cache is best-effort
        logger.debug("Could not write LLM cache: %s", exc)


def chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    fallback: Optional[Dict[str, Any]] = None,
    fallback_factory: Optional[Callable[[], Dict[str, Any]]] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Call the chat completion API and return parsed JSON.

    Combines caching, retry-with-backoff and offline fallback.

    Args:
        system_prompt: The system role content.
        user_prompt: The user role content.
        model: Override model (defaults to ``Config.OPENAI_MODEL``).
        temperature: Override temperature (defaults to ``Config.LLM_TEMPERATURE``).
        fallback: Value returned if the call ultimately fails and offline
            fallback is allowed. Defaults to ``{}``.
        fallback_factory: Callable producing the fallback value (takes
            precedence over ``fallback``); useful for heuristic offline results.
        use_cache: Whether to consult / populate the on-disk cache.

    Returns:
        The parsed JSON response as a dict (never raises for API errors when
        ``ALLOW_OFFLINE_FALLBACK`` is enabled).
    """
    model = model or Config.OPENAI_MODEL
    temperature = Config.LLM_TEMPERATURE if temperature is None else temperature

    def _build_fallback() -> Dict[str, Any]:
        if fallback_factory is not None:
            return fallback_factory()
        return {} if fallback is None else fallback

    # No usable key -> go straight to fallback (offline mode).
    if not Config.has_api_key():
        if Config.ALLOW_OFFLINE_FALLBACK:
            logger.warning("No API key configured; using offline fallback.")
            return _build_fallback()
        raise RuntimeError("No OpenAI API key configured and offline fallback disabled.")

    key = _cache_key(model, system_prompt, user_prompt, temperature)
    if use_cache:
        cached = _cache_read(key)
        if cached is not None:
            logger.debug("LLM cache hit for %s", key[:12])
            return cached

    global _JSON_MODE_SUPPORTED
    client = get_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, Config.LLM_MAX_RETRIES + 1):
        try:
            request: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            # Only request JSON mode when the endpoint supports it.
            if _JSON_MODE_SUPPORTED:
                request["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content or "{}"
            result = _parse_json_content(content)
            if use_cache:
                _cache_write(key, result)
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc

            # The endpoint rejects response_format -> disable it and retry now
            # (this is not a transient failure, so don't back off).
            if _JSON_MODE_SUPPORTED and _is_json_format_error(exc):
                logger.info("Endpoint rejects JSON response_format; disabling it.")
                _JSON_MODE_SUPPORTED = False
                continue

            # Other client-side (4xx, non-429) errors won't recover on retry.
            if _is_non_transient(exc):
                logger.error("Non-transient LLM error: %s", exc)
                break

            delay = Config.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt, Config.LLM_MAX_RETRIES, exc, delay,
            )
            if attempt < Config.LLM_MAX_RETRIES:
                time.sleep(delay)

    # All retries exhausted.
    logger.error("LLM call failed after %d attempts: %s",
                 Config.LLM_MAX_RETRIES, last_error)
    if Config.ALLOW_OFFLINE_FALLBACK:
        return _build_fallback()
    raise RuntimeError(f"LLM call failed after retries: {last_error}")
