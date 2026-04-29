"""Tests for the query-embedding LRU cache + fastembed opt-in (P-perf).

Covers:
  * Cache hit returns the same object for repeat queries (no re-embed).
  * ``clear_embed_cache`` invalidates everything.
  * LRU evicts the oldest entry past ``maxsize``.
  * ``should_use_dense`` heuristic (trivial → False, content-bearing → True).
  * ``MEMOIRS_EMBED_BACKEND=fastembed`` falls back to sentence-transformers
    with a warning when fastembed is not installed.

The first three tests monkey-patch ``embed_text`` so they run in milliseconds
without touching the heavy sentence-transformers model.
"""
from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import patch

import pytest

from memoirs.engine import embeddings as emb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the LRU between tests so order doesn't leak state."""
    emb.clear_embed_cache()
    yield
    emb.clear_embed_cache()


class _CountingEmbedder:
    """Drop-in replacement for ``emb.embed_text`` that records call count."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        # Deterministic, cheap "embedding" — content-dependent so different
        # queries produce different vectors.
        return [float(len(text)) + 0.001 * self.calls, 0.5, 0.25]


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_cache_hit_skips_reembed(monkeypatch):
    counter = _CountingEmbedder()
    monkeypatch.setattr(emb, "embed_text", counter)

    first = emb.embed_text_cached("memoirs MCP server architecture")
    second = emb.embed_text_cached("memoirs MCP server architecture")

    assert first == second
    assert counter.calls == 1, f"expected 1 underlying embed call, got {counter.calls}"
    info = emb.embed_cache_info()
    assert info.hits >= 1
    assert info.misses == 1


def test_clear_embed_cache_invalidates(monkeypatch):
    counter = _CountingEmbedder()
    monkeypatch.setattr(emb, "embed_text", counter)

    emb.embed_text_cached("hello world from memoirs")
    assert counter.calls == 1
    emb.embed_text_cached("hello world from memoirs")
    assert counter.calls == 1  # cache hit

    emb.clear_embed_cache()
    emb.embed_text_cached("hello world from memoirs")
    assert counter.calls == 2  # forced re-embed after clear


def test_lru_evicts_oldest_past_maxsize(monkeypatch):
    counter = _CountingEmbedder()
    monkeypatch.setattr(emb, "embed_text", counter)

    # Patch the internal LRU with a tiny maxsize=2 wrapper so we can
    # observe eviction without filling 1024 slots in a test.
    import functools

    @functools.lru_cache(maxsize=2)
    def _tiny(text: str, model_key: str) -> tuple[float, ...]:
        return tuple(emb.embed_text(text))

    monkeypatch.setattr(emb, "_embed_cached_inner", _tiny)

    emb.embed_text_cached("alpha bravo charlie")  # miss → calls=1
    emb.embed_text_cached("delta echo foxtrot")   # miss → calls=2
    emb.embed_text_cached("alpha bravo charlie")  # hit → calls=2
    assert counter.calls == 2

    # Insert a third — pushes "delta…" out (LRU = "delta…" since alpha was
    # just touched). Note functools.lru_cache evicts the LEAST RECENTLY USED.
    emb.embed_text_cached("golf hotel india")     # miss → calls=3
    assert counter.calls == 3

    # "delta…" should now be evicted → re-embed required.
    emb.embed_text_cached("delta echo foxtrot")
    assert counter.calls == 4


# ---------------------------------------------------------------------------
# should_use_dense heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected",
    [
        ("AI", False),                                  # 1 token, short
        ("the", False),                                 # stopword only
        ("the cat", False),                             # 2 tokens, mostly stopword
        ("", False),                                    # empty
        ("   ", False),                                 # whitespace only
        ("memoirs", False),                             # 1 token, only 7 chars → trivial
        ("retrieval", True),                            # 1 token, 9 chars → meaty enough
        ("memoirs MCP server architecture", True),      # 4 tokens
        ("how does retrieval rank memories?", True),    # 5 tokens
    ],
)
def test_should_use_dense(query, expected):
    assert emb.should_use_dense(query) is expected, query


# ---------------------------------------------------------------------------
# fastembed fallback
# ---------------------------------------------------------------------------


def test_fastembed_unavailable_falls_back_with_warning(monkeypatch, caplog):
    """If fastembed isn't installed, embed_text logs a warning and uses ST."""
    # Force the env flag.
    monkeypatch.setenv("MEMOIRS_EMBED_BACKEND", "fastembed")
    # Ensure no cached singleton + warned flag from prior runs.
    monkeypatch.setattr(emb, "_FASTEMBED_SINGLETON", None)
    monkeypatch.setattr(emb, "_FASTEMBED_WARNED", False)

    # Block any real fastembed import — even if it ends up installed in CI.
    monkeypatch.setitem(sys.modules, "fastembed", None)

    # Replace the sentence-transformers fallback so we don't actually load
    # the model — we only care that the fallback path is taken.
    sentinel: list[str] = []

    def _fake_st_embed():
        class _M:
            def encode(self, texts, normalize_embeddings=True):
                sentinel.append(texts[0])
                return [[0.1, 0.2, 0.3]]
        return _M()

    monkeypatch.setattr(emb, "_require_embedder", _fake_st_embed)

    caplog.set_level(logging.WARNING, logger="memoirs.embeddings")
    out = emb.embed_text("hello world")

    assert out == [0.1, 0.2, 0.3]
    assert sentinel == ["hello world"]
    assert any("fastembed" in rec.message and "fall" in rec.message.lower()
               for rec in caplog.records), \
        "expected warning about fastembed fallback"


def test_resolve_backend_default_when_unset(monkeypatch):
    """Default backend depends on what's installed.

    Fix #4 (P-perf-throughput) made fastembed the auto-default when its
    ONNX runtime is importable, because it bypasses the GIL and gives
    sustained throughput a 3-5× boost. Sentence-transformers is the
    fallback only when fastembed isn't on the path.
    """
    monkeypatch.delenv("MEMOIRS_EMBED_BACKEND", raising=False)
    try:
        import fastembed  # noqa: F401
        expected = "fastembed"
    except ImportError:
        expected = "sentence-transformers"
    assert emb._resolve_backend() == expected


def test_resolve_backend_unknown_warns(monkeypatch, caplog):
    monkeypatch.setenv("MEMOIRS_EMBED_BACKEND", "vodoo-magic")
    caplog.set_level(logging.WARNING, logger="memoirs.embeddings")
    assert emb._resolve_backend() == "sentence-transformers"
    assert any("unknown MEMOIRS_EMBED_BACKEND" in rec.message for rec in caplog.records)


def test_cache_key_includes_model_so_swap_invalidates(monkeypatch):
    """Changing backend/model should naturally miss the cache because the key changes."""
    counter = _CountingEmbedder()
    monkeypatch.setattr(emb, "embed_text", counter)

    monkeypatch.setenv("MEMOIRS_EMBED_BACKEND", "sentence-transformers")
    emb.embed_text_cached("project status update memo")
    assert counter.calls == 1

    # Same query, but pretend the backend flipped — different key, miss again.
    monkeypatch.setenv("MEMOIRS_EMBED_BACKEND", "fastembed")
    emb.embed_text_cached("project status update memo")
    assert counter.calls == 2
