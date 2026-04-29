"""Ollama detection + env-shim helpers shared by every other engine adapter.

Why this module exists:
  * Mem0, Cognee, Letta and Zep all hard-code an OpenAI client somewhere
    in their stack. Ollama (https://ollama.com) ships an OpenAI-compatible
    HTTP API at ``http://localhost:11434/v1`` so we can redirect those
    clients to a local model just by twiddling a handful of env vars
    BEFORE the other engine library imports its provider.
  * Doing the env shim once, in one helper, keeps the per-adapter changes
    tiny (a single ``os.environ.update(...)`` call gated on
    ``MEMOIRS_USE_OLLAMA``).

Public surface:
  * ``ollama_is_up()``      — cheap probe of ``/api/tags`` (≤2 s timeout).
  * ``ollama_list_models()``— names of the locally pulled models.
  * ``ollama_env_for_openai_clients(model)`` — env-var dict for OpenAI-
    compatible callers (Mem0, Cognee, Letta).
  * ``use_ollama_requested()``  — checks ``MEMOIRS_USE_OLLAMA`` /
    ``MEMOIRS_OLLAMA`` flags so adapters share one rule.
  * ``ollama_install_hint()`` — single source of truth for the install
    instructions surfaced by the bench when Ollama is down.

The helpers are intentionally side-effect-free except for
``apply_ollama_env``: callers decide whether they want to mutate the
process env or just read the dict.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_OPENAI_BASE_URL = f"{OLLAMA_HOST}/v1"
DEFAULT_LLM_MODEL = "qwen2.5:3b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Env vars that flip Ollama wiring on for an adapter. We accept BOTH spellings
# because the rest of the codebase / docs are inconsistent and the cost of
# tolerance here is one extra ``or`` per call.
_FLAG_VARS = ("MEMOIRS_USE_OLLAMA", "MEMOIRS_OLLAMA")
_TRUTHY = {"1", "on", "true", "yes", "y"}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def ollama_is_up(*, timeout: float = 2.0, host: str = OLLAMA_HOST) -> bool:
    """Return True iff Ollama responds at ``{host}/api/tags`` within timeout.

    We hit ``/api/tags`` rather than ``/`` because the root returns a
    plain "Ollama is running" string while ``/api/tags`` is a stable JSON
    endpoint we also reuse for ``ollama_list_models``.
    """
    try:
        import urllib.request

        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def ollama_list_models(*, timeout: float = 2.0, host: str = OLLAMA_HOST) -> list[str]:
    """Return names of locally available Ollama models, [] when down.

    Used by callers that want to verify a specific model is present
    before pointing other engines at it. Returns an empty list on any error so
    the caller can pair it with ``ollama_is_up`` for a "down" message.
    """
    try:
        import urllib.request

        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for item in (payload.get("models") or []):
        name = item.get("name") or item.get("model")
        if name:
            out.append(str(name))
    return out


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------


def ollama_env_for_openai_clients(
    model: str = DEFAULT_LLM_MODEL,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    base_url: str = OLLAMA_OPENAI_BASE_URL,
) -> dict[str, str]:
    """Return env vars that redirect OpenAI-compatible clients to Ollama.

    The dict covers the three vocabularies we encounter:

    * **OpenAI SDK** — ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL`` +
      ``OPENAI_MODEL``. Mem0 and Letta read these (Letta also accepts
      ``OPENAI_API_BASE``, which we set as a synonym).
    * **Cognee LiteLLM-style** — ``LLM_PROVIDER``, ``LLM_API_KEY``,
      ``LLM_ENDPOINT``, ``LLM_MODEL`` plus the ``EMBEDDING_*`` parallel
      set for the embedding side.
    * **Zep / Graphiti** — same OpenAI keys.

    A dummy ``OPENAI_API_KEY=ollama`` value satisfies the SDK's "key
    must be present" assertion without us having to plumb a real key.
    """
    env: dict[str, str] = {
        # OpenAI SDK shape (Mem0, Letta, Zep, anything that imports openai).
        "OPENAI_API_KEY": "ollama",
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_BASE": base_url,        # legacy SDK attr
        "OPENAI_MODEL": model,
        # Cognee / LiteLLM shape.
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "ollama",
        "LLM_ENDPOINT": base_url,
        "LLM_MODEL": model,
        # Embeddings (Cognee splits LLM vs embed config).
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_API_KEY": "ollama",
        "EMBEDDING_ENDPOINT": base_url,
        "EMBEDDING_MODEL": embed_model,
    }
    return env


def use_ollama_requested(env: Optional[dict[str, str]] = None) -> bool:
    """True when any of the supported feature flags is set to a truthy value.

    Adapters call this once at the top of ``__init__`` so the bench can
    flip Ollama on without each adapter knowing about both spellings.
    """
    src = env if env is not None else os.environ
    for var in _FLAG_VARS:
        val = (src.get(var) or "").strip().lower()
        if val in _TRUTHY:
            return True
    return False


def apply_ollama_env(
    *,
    model: str = DEFAULT_LLM_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    extra_keys: Iterable[str] = (),
) -> dict[str, str]:
    """Mutate ``os.environ`` with the Ollama shim and return the diff.

    Adapters MUST call this BEFORE importing their backing library — the
    OpenAI SDK reads its env at module import time, so a late update is
    silently ignored. Returning the diff lets tests assert the exact
    vars that were set.
    """
    diff = ollama_env_for_openai_clients(model=model, embed_model=embed_model)
    for k in extra_keys:
        diff.setdefault(k, "ollama")
    os.environ.update(diff)
    return diff


# ---------------------------------------------------------------------------
# Operator-facing hint
# ---------------------------------------------------------------------------


def ollama_install_hint(*, model: str = DEFAULT_LLM_MODEL,
                         embed_model: str = DEFAULT_EMBED_MODEL) -> str:
    """One-liner the bench prints when ``--ollama`` is set but Ollama is down."""
    return (
        "Ollama not up at " + OLLAMA_HOST + " — install via: "
        "curl -fsSL https://ollama.com/install.sh | sh "
        f"&& ollama pull {model} && ollama pull {embed_model}"
    )


__all__ = [
    "OLLAMA_HOST",
    "OLLAMA_OPENAI_BASE_URL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBED_MODEL",
    "ollama_is_up",
    "ollama_list_models",
    "ollama_env_for_openai_clients",
    "use_ollama_requested",
    "apply_ollama_env",
    "ollama_install_hint",
]
