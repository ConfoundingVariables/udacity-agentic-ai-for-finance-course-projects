"""
Model configuration shared by all modules: config.toml loading, provider selection
with automatic fallback (Vocareum -> OpenRouter), LlamaIndex + DSPy setup, and token counting.
"""

import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import openai
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # course-3/.env for local runs
os.environ.setdefault("LITELLM_LOG", "ERROR")  # DSPy's LiteLLM otherwise logs every call at INFO

CONFIG = tomllib.loads((Path(__file__).resolve().parent.parent / "config.toml").read_text())
TOKEN_COUNTER = TokenCountingHandler()  # cumulative LlamaIndex LLM + embedding tokens


class AdaptiveEmbedding(OpenAIEmbedding):
    """Embedding batch limits differ per provider, so a batch rejected as too large is split in half and retried."""

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            return super()._get_text_embeddings(texts)
        except openai.BadRequestError:
            if len(texts) == 1:
                raise
            half = len(texts) // 2
            return self._get_text_embeddings(texts[:half]) + self._get_text_embeddings(texts[half:])


@dataclass(frozen=True)
class Provider:
    name: str
    api_base: str
    api_key: str
    llm: str
    embedding: str


def _provider(name: str) -> Provider | None:
    p = CONFIG["providers"][name]
    key = os.getenv(p["api_key_env"])
    if not key:
        return None
    api_base = os.getenv(p.get("api_base_env", ""), p["default_api_base"])
    return Provider(name, api_base, key, p["llm"], p["embedding"])


def _responds(p: Provider) -> bool:
    try:
        openai.OpenAI(base_url=p.api_base, api_key=p.api_key, timeout=15, max_retries=0).chat.completions.create(
            model=p.llm, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
        return True
    except openai.OpenAIError:
        return False


@cache
def active_provider() -> Provider:
    """First provider in provider_order with a key that answers a 1-token probe."""
    candidates = [p for name in CONFIG["models"]["provider_order"] if (p := _provider(name))]
    if not candidates:
        raise RuntimeError("No LLM provider key set (OPENAI_API_KEY or OPENROUTER_API_KEY)")
    return next((p for p in candidates if _responds(p)), candidates[0])


def configure_models() -> OpenAILike:
    """Point LlamaIndex (and DSPy) at the active provider via api_base; return the LLM.

    OpenAILike is LlamaIndex's OpenAI client for OpenAI-compatible endpoints; unlike OpenAI
    it accepts provider-prefixed model names such as OpenRouter's "openai/gpt-3.5-turbo".
    """
    p, m = active_provider(), CONFIG["models"]
    callbacks = Settings.callback_manager = CallbackManager([TOKEN_COUNTER])
    llm = OpenAILike(model=p.llm, api_base=p.api_base, api_key=p.api_key, temperature=m["temperature"],
                     is_chat_model=True, context_window=m["context_window"], callback_manager=callbacks)
    Settings.llm = llm
    Settings.embed_model = AdaptiveEmbedding(model_name=p.embedding, api_base=p.api_base, api_key=p.api_key,
                                           embed_batch_size=m["embed_batch_size"],
                                           callback_manager=callbacks)
    import dspy  # imported late: dspy's lazy loader breaks `openai` if it is imported first

    dspy.configure(lm=dspy.LM(f"openai/{p.llm}", api_base=p.api_base, api_key=p.api_key,
                              temperature=m["temperature"], cache=False, num_retries=5), track_usage=True)
    return llm
