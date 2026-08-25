"""Core modules for the AML/SAR processing system."""

from __future__ import annotations

import os

__version__ = "1.0.0"


def create_vocareum_openai_client():
    """Create the configured OpenAI client without logging credentials."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required; install the project requirements"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    return OpenAI(base_url=base_url, api_key=api_key)


__all__ = ["create_vocareum_openai_client"]
