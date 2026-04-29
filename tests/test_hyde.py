"""Tests for engine/hyde.py — P2-3 HyDE / query expansion.

Gemma is mocked at the ``_gemma_hypothetical`` boundary so the suite stays
offline and fast.
"""
from __future__ import annotations

import pytest

from memoirs.engine import hyde


# ---------------------------------------------------------------------------
# Activation gate
# ---------------------------------------------------------------------------


def test_is_enabled_off_by_default(monkeypatch):
    monkeypatch.delenv(hyde.ENV_HYDE, raising=False)
    assert hyde.is_enabled() is False


def test_is_enabled_on(monkeypatch):
    monkeypatch.setenv(hyde.ENV_HYDE, "on")
    assert hyde.is_enabled() is True


def test_is_enabled_truthy_aliases(monkeypatch):
    for v in ("1", "true", "yes", "ON", "True"):
        monkeypatch.setenv(hyde.ENV_HYDE, v)
        assert hyde.is_enabled() is True


# ---------------------------------------------------------------------------
# Keyword backend (no LLM)
# ---------------------------------------------------------------------------


def test_expand_query_keyword_backend_basic(monkeypatch):
    # Force keyword backend; ensure no Gemma path runs even if installed.
    monkeypatch.setattr(hyde, "_gemma_hypothetical", lambda q: "")
    out = hyde.expand_query("How do I install memoirs?", backend="keyword")
    assert out.original == "How do I install memoirs?"
    assert out.hypothetical_doc == ""
    # Stop-words ("how", "do", "i") and short words must be filtered out.
    assert "install" in out.keywords
    assert "memoirs" in out.keywords
    assert "how" not in out.keywords
    assert "i" not in out.keywords
    assert out.combined.startswith("How do I install memoirs?")
    # combined contains the keywords joined.
    assert "install" in out.combined and "memoirs" in out.combined
    assert out.backend == "keyword"


def test_expand_query_caps_keywords_at_5():
    q = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    out = hyde.expand_query(q, backend="keyword")
    assert len(out.keywords) <= 5
    assert out.keywords[0] == "alpha"


def test_expand_query_empty_input_round_trips():
    out = hyde.expand_query("", backend="auto")
    assert out.original == ""
    assert out.combined == ""
    assert out.hypothetical_doc == ""
    assert out.keywords == []


# ---------------------------------------------------------------------------
# Gemma backend (mocked LLM)
# ---------------------------------------------------------------------------


def test_expand_query_gemma_backend_with_mock(monkeypatch):
    monkeypatch.setattr(
        hyde,
        "_gemma_hypothetical",
        lambda q: "Memoirs is a local-first memory engine you install via pip.",
    )
    out = hyde.expand_query("How do I install memoirs?", backend="gemma")
    assert out.hypothetical_doc.startswith("Memoirs is a local-first")
    assert len(out.hypothetical_doc) > 20
    assert out.backend == "gemma"
    # Combined must include both original + hypothetical.
    assert "How do I install memoirs?" in out.combined
    assert "Memoirs is a local-first" in out.combined


def test_expand_query_auto_falls_back_to_keyword_when_gemma_empty(monkeypatch):
    monkeypatch.setattr(hyde, "_gemma_hypothetical", lambda q: "")
    out = hyde.expand_query("memoirs MCP server architecture", backend="auto")
    assert out.backend == "keyword"
    assert out.hypothetical_doc == ""
    assert "memoirs" in out.keywords


def test_expand_query_auto_uses_gemma_when_available(monkeypatch):
    monkeypatch.setattr(hyde, "_gemma_hypothetical", lambda q: "Imagined answer.")
    out = hyde.expand_query("What is memoirs?", backend="auto")
    assert out.backend == "gemma"
    assert "Imagined answer" in out.combined


def test_expand_query_unknown_backend_falls_back_to_auto(monkeypatch, caplog):
    monkeypatch.setattr(hyde, "_gemma_hypothetical", lambda q: "")
    with caplog.at_level("WARNING", logger="memoirs.hyde"):
        out = hyde.expand_query("test query content", backend="banana")
    assert out.backend == "keyword"


# ---------------------------------------------------------------------------
# Pipeline-level activation gate (env off → expand_query NOT called)
# ---------------------------------------------------------------------------


def test_pipeline_skips_expand_when_env_off(monkeypatch):
    """When ``MEMOIRS_HYDE`` is off the engine helper must NOT invoke
    ``expand_query`` at all (saves the keyword-extract cost too).
    """
    from memoirs.engine import memory_engine

    monkeypatch.delenv(hyde.ENV_HYDE, raising=False)

    calls = {"n": 0}

    def _spy(query, backend="auto"):
        calls["n"] += 1
        return hyde.ExpandedQuery(original=query, combined=query)

    monkeypatch.setattr(hyde, "expand_query", _spy)
    q, info = memory_engine._apply_hyde("anything")
    assert q == "anything"
    assert info is None
    assert calls["n"] == 0


def test_pipeline_calls_expand_when_env_on(monkeypatch):
    from memoirs.engine import memory_engine

    monkeypatch.setenv(hyde.ENV_HYDE, "on")

    def _spy(query, backend="auto"):
        return hyde.ExpandedQuery(
            original=query,
            hypothetical_doc="hypothetical",
            keywords=["foo", "bar"],
            combined=f"{query}. hypothetical. foo bar",
            backend="gemma",
        )

    monkeypatch.setattr(hyde, "expand_query", _spy)
    q, info = memory_engine._apply_hyde("foo bar baz")
    assert q.startswith("foo bar baz")
    assert info is not None
    assert info["backend"] == "gemma"


def test_combined_contains_original_and_expansion(monkeypatch):
    monkeypatch.setattr(hyde, "_gemma_hypothetical", lambda q: "Brief plausible answer.")
    out = hyde.expand_query("install memoirs locally", backend="gemma")
    assert "install memoirs locally" in out.combined
    assert "Brief plausible answer" in out.combined
    # Keywords should also be included.
    for kw in out.keywords:
        assert kw in out.combined


def test_extract_keywords_dedups_case_insensitive():
    out = hyde.expand_query("Memoirs memoirs MEMOIRS install", backend="keyword")
    # Only one occurrence of the lowercased token.
    assert out.keywords.count("memoirs") == 1
