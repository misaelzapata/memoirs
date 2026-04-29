"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
from pathlib import Path

from memoirs.db import MemoirsDB


@pytest.fixture
def tmp_db(tmp_path: Path) -> MemoirsDB:
    db = MemoirsDB(tmp_path / "memoirs.sqlite")
    db.init()
    yield db
    db.close()


@pytest.fixture(autouse=True)
def _reset_global_state(monkeypatch):
    """Keep tests deterministic regardless of the host environment.

    Two sources of cross-test contamination motivated this fixture:

    1. **LLM curator presence.** ``MEMOIRS_GEMMA_CURATOR`` defaults to
       ``auto`` in production, so when Qwen3 / Phi / Gemma GGUF weights are
       installed on the developer machine ``decide_memory_action`` silently
       calls into the LLM. The model's verdict is non-deterministic from
       the unit-test POV (e.g. it can return ``UPDATE`` where the rules
       return ``CONTRADICTION``), causing flakes that depend purely on
       whether the host has the weights.

       We neuter the LLM path by monkey-patching ``_have_gemma`` to
       ``False`` and disabling the retrieval-side flag. Any test that wants
       to exercise the curator does so by mocking ``_have_gemma`` /
       ``_get_llm`` directly (the test's ``monkeypatch.setattr`` runs
       *after* this fixture and overrides our patch — pytest-monkeypatch
       semantics).

    2. **LRU embed cache.** ``embed_text_cached`` wraps a process-wide
       ``functools.lru_cache``. Tests that ``monkeypatch.setattr(emb,
       "embed_text", ...)`` need a clean cache, otherwise a previously
       cached real embedding leaks across tests.
    """
    # Default the retrieval-side flag OFF too — it has the same default-auto
    # surprise as the curator. Tests that need it on opt in explicitly. We
    # set both the new (``MEMOIRS_RETRIEVAL_CURATOR``) and legacy
    # (``MEMOIRS_RETRIEVAL_GEMMA``) names so either lookup wins.
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_CURATOR", "off")
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_GEMMA", "off")

    # Force the LLM availability check to ``False`` so neither
    # ``decide_memory_action`` (auto/on mode) nor ``curator_consolidate`` /
    # ``curator_resolve_conflict`` actually load the model. We patch
    # ``_have_curator`` on the new ``curator`` module AND ``_have_gemma`` on
    # the legacy ``gemma`` shim so callers that look up either name see the
    # override.
    try:
        from memoirs.engine import curator as _curator
        monkeypatch.setattr(_curator, "_have_curator", lambda: False)
    except Exception:
        # curator module not importable — fine, nothing to patch.
        pass
    try:
        from memoirs.engine import gemma as _gemma
        monkeypatch.setattr(_gemma, "_have_gemma", lambda: False)
    except Exception:
        # gemma shim not importable — fine, nothing to patch.
        pass

    yield

    try:
        from memoirs.engine.embeddings import clear_embed_cache
        clear_embed_cache()
    except Exception:
        # Embeddings module not importable in this environment — fine.
        pass
