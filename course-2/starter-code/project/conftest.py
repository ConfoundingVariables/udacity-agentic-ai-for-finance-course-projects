"""
Pytest configuration.

* Ensures the project root is importable (so ``import config`` etc. work).
* Forces OFFLINE mode for the whole test session so no test makes a real
  network / OpenAI call — the deterministic fallbacks are exercised instead.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _force_offline():
    """Make every test run offline (no live API key)."""
    from config import Config
    original = Config.OPENAI_API_KEY
    # A placeholder value is treated as "no usable key" by Config.has_api_key().
    Config.OPENAI_API_KEY = "real-api-key"
    Config.ALLOW_OFFLINE_FALLBACK = True
    yield
    Config.OPENAI_API_KEY = original
