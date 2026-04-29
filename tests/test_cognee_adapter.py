"""Unit tests for the ``CogneeAdapter`` ID-mapping fix.

Cognee's default ``search`` mode collapses the ranked memorias into a
single LLM-generated answer string, which made the rival bench
report MRR=0.0 even though the engine was ingesting and searching.

The fix in ``scripts/adapters/cognee_adapter.py``:

  * Drives ``cognee.search`` with ``SearchType.CHUNKS`` so each result
    keeps its raw chunk text.
  * Tags each ingested memory with a ``[bench_id=<id>]`` prefix so the
    adapter can recover the bench ID from the chunk text.
  * Falls back to substring-matching the original content when the
    prefix is missing (rare, but defensive).

This file pins those behaviors with hermetic tests — no live cognee,
no LLM. The end-to-end smoke test against a real cognee install lives
in ``scripts/probe_cognee.py`` (manual, requires OPENAI_API_KEY).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Resolve the project root so `scripts.*` works under pytest.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 1. Tag/extract round-trip
# ---------------------------------------------------------------------------


def test_tag_with_bench_id_round_trips_through_extract():
    """The tag is stable: tagging then extracting gives the original ID."""
    pytest.importorskip("cognee")
    from scripts.adapters.cognee_adapter import (
        _extract_bench_id,
        _tag_with_bench_id,
    )

    tagged = _tag_with_bench_id("mem_sh_lasagna",
                                "user's favorite recipe is lasagna")
    assert tagged.startswith("[bench_id=mem_sh_lasagna] ")
    assert _extract_bench_id(tagged) == "mem_sh_lasagna"


def test_extract_bench_id_finds_marker_anywhere_in_text():
    """Marker survives ANY surrounding context — chunkers may prepend
    or append metadata, so we search the whole string."""
    pytest.importorskip("cognee")
    from scripts.adapters.cognee_adapter import _extract_bench_id

    assert _extract_bench_id(
        "USER: hi\n[bench_id=conv_42] some text"
    ) == "conv_42"
    # Underscores, dots, dashes, colons all valid in IDs.
    assert _extract_bench_id("[bench_id=mem_t_drink_old]") == "mem_t_drink_old"
    assert _extract_bench_id("[bench_id=conv_pref_python::3] x") == "conv_pref_python::3"


def test_extract_bench_id_returns_none_when_marker_absent():
    """No marker ⇒ ``None`` so the adapter can fall through to the
    substring match path."""
    pytest.importorskip("cognee")
    from scripts.adapters.cognee_adapter import _extract_bench_id

    assert _extract_bench_id("just some text without a marker") is None


# ---------------------------------------------------------------------------
# 2. Search-result flattener handles every shape cognee returns
# ---------------------------------------------------------------------------


def _adapter():
    """Build a CogneeAdapter without running its ``__init__`` env work.

    We only need the bound method ``_flatten_search_result``; bypassing
    ``__init__`` keeps the test hermetic (no real cognee module touch).
    """
    from scripts.adapters.cognee_adapter import CogneeAdapter
    return CogneeAdapter.__new__(CogneeAdapter)


def test_flatten_handles_chunks_envelope():
    """``SearchType.CHUNKS`` returns the canonical envelope; the
    flattener must extract each chunk's ``text`` field in order."""
    pytest.importorskip("cognee")
    adapter = _adapter()
    raw = [
        {
            "dataset_id": "abc",
            "dataset_name": "memoirs_bench",
            "search_result": [
                {"id": "1", "text": "[bench_id=A] alpha content"},
                {"id": "2", "text": "[bench_id=B] beta content"},
            ],
        }
    ]
    flat = adapter._flatten_search_result(raw)
    assert flat == [
        "[bench_id=A] alpha content",
        "[bench_id=B] beta content",
    ]


def test_flatten_handles_rag_completion_string():
    """``RAG_COMPLETION`` returns a string answer per dataset envelope.

    The flattener still passes it through so the substring-match
    fallback can operate on it (even though the bench adapter never
    triggers RAG_COMPLETION itself).
    """
    pytest.importorskip("cognee")
    adapter = _adapter()
    raw = [{
        "dataset_id": "abc",
        "search_result": ["Lasagna bolognese with bechamel."],
    }]
    flat = adapter._flatten_search_result(raw)
    assert flat == ["Lasagna bolognese with bechamel."]


def test_flatten_handles_dict_with_content_or_name_fallback():
    """Dicts that lack ``text`` should fall back to ``content`` or
    ``name`` (cognee ships several DataPoint shapes)."""
    pytest.importorskip("cognee")
    adapter = _adapter()
    raw = [{
        "search_result": [
            {"id": "1", "content": "[bench_id=A] alpha"},
            {"id": "2", "name": "[bench_id=B] beta"},
            {"id": "3"},  # no readable field -> repr fallback
        ],
    }]
    flat = adapter._flatten_search_result(raw)
    assert flat[0] == "[bench_id=A] alpha"
    assert flat[1] == "[bench_id=B] beta"
    # Last entry stringifies the dict -- safe so substring fallback
    # still has a shot.
    assert "3" in flat[2]


def test_flatten_handles_empty_and_none():
    """Empty / None must not crash and must yield an empty list."""
    pytest.importorskip("cognee")
    adapter = _adapter()
    assert adapter._flatten_search_result(None) == []
    assert adapter._flatten_search_result([]) == []
    assert adapter._flatten_search_result([{"search_result": []}]) == []


# ---------------------------------------------------------------------------
# 3. End-to-end query path with mocked cognee
# ---------------------------------------------------------------------------


def test_query_recovers_bench_ids_from_chunks(monkeypatch):
    """The full ``add_memories`` + ``query`` path must:

      * tag each memory with the ``[bench_id=...]`` prefix on add
      * pass the tagged text through ``cognee.add``
      * decode the bench ID from each chunk on query, preserving rank.

    We mock ``cognee.add``/``cognify``/``search`` so the test stays
    hermetic. The shapes are taken straight from the live probe in
    ``scripts/probe_cognee.py`` so the mock matches reality.
    """
    pytest.importorskip("cognee")
    import cognee

    from scripts.adapters.cognee_adapter import CogneeAdapter
    from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery

    add_calls: list[str] = []
    cognify_calls: list[dict] = []

    async def fake_add(text, **kwargs):
        # Capture both text and dataset_name so we can assert the
        # adapter scopes its writes.
        add_calls.append(text)

    async def fake_cognify(*_a, **kwargs):
        cognify_calls.append(kwargs)

    async def fake_search(query_text, query_type=None, datasets=None,
                          top_k=10, **_kwargs):
        # Mirror the live probe's CHUNKS shape: an envelope list with a
        # ``search_result`` list of chunk dicts. Vector rank decides
        # order; we pretend the second memory is the closest match.
        return [{
            "dataset_id": "fake-uuid",
            "dataset_name": "memoirs_bench",
            "search_result": [
                {"id": "c2", "text": "[bench_id=mem_b] beta content"},
                {"id": "c1", "text": "[bench_id=mem_a] alpha content"},
            ],
        }]

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", fake_cognify)
    monkeypatch.setattr(cognee, "search", fake_search)

    adapter = CogneeAdapter()
    if not adapter.status.ok:
        pytest.skip(f"cognee unavailable: {adapter.status.reason}")

    adapter.add_memories([
        BenchMemory(id="mem_a", type="fact", content="alpha content"),
        BenchMemory(id="mem_b", type="fact", content="beta content"),
    ])
    # ``add`` called once per memory and the tag is present.
    assert len(add_calls) == 2
    assert add_calls[0].startswith("[bench_id=mem_a] ")
    assert add_calls[1].startswith("[bench_id=mem_b] ")
    # Cognify ran once after ingestion.
    assert len(cognify_calls) == 1

    # Query — IDs come back in the order cognee ranked them.
    ids = adapter.query(BenchQuery(query="anything", gold_memory_ids=["mem_b"]))
    assert ids == ["mem_b", "mem_a"]


def test_query_falls_back_to_content_substring_when_marker_missing(monkeypatch):
    """If chunking strips the prefix, the adapter must still recover
    the bench ID via a substring match against the content cache.

    This is the safety net that keeps the bench numeric even when
    cognee changes its chunking heuristics.
    """
    pytest.importorskip("cognee")
    import cognee

    from scripts.adapters.cognee_adapter import CogneeAdapter
    from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery

    async def noop_add(text, **kwargs): return None
    async def noop_cognify(*_a, **_k): return None

    async def fake_search(*_a, **_kw):
        # Marker is GONE — only the raw content survives.
        return [{
            "search_result": [
                {"text": "alpha content here"},  # matches mem_a content
            ],
        }]

    monkeypatch.setattr(cognee, "add", noop_add)
    monkeypatch.setattr(cognee, "cognify", noop_cognify)
    monkeypatch.setattr(cognee, "search", fake_search)

    adapter = CogneeAdapter()
    if not adapter.status.ok:
        pytest.skip(f"cognee unavailable: {adapter.status.reason}")

    adapter.add_memories([
        BenchMemory(id="mem_a", type="fact", content="alpha content"),
    ])
    ids = adapter.query(BenchQuery(query="anything", gold_memory_ids=["mem_a"]))
    assert ids == ["mem_a"]


def test_query_returns_empty_on_search_failure(monkeypatch):
    """A cognee runtime error must NOT crash the bench — the adapter
    swallows it and returns an empty list (counted as a miss)."""
    pytest.importorskip("cognee")
    import cognee

    from scripts.adapters.cognee_adapter import CogneeAdapter
    from scripts.bench_vs_others_dataset import BenchQuery

    async def boom(*_a, **_k):
        raise RuntimeError("cognee blew up")

    monkeypatch.setattr(cognee, "search", boom)

    adapter = CogneeAdapter()
    if not adapter.status.ok:
        pytest.skip(f"cognee unavailable: {adapter.status.reason}")

    ids = adapter.query(BenchQuery(query="anything", gold_memory_ids=["mem_a"]))
    assert ids == []


# ---------------------------------------------------------------------------
# 4. Init guard surfaces missing API key as SKIP, not crash
# ---------------------------------------------------------------------------


def test_init_skips_clean_when_no_api_key(monkeypatch):
    """No OPENAI_API_KEY ⇒ adapter status is SKIP with a clear reason.

    The bench treats this as "engine unavailable" rather than crashing.
    We delete every plausible LLM-key env var so the guard fires even
    on a developer machine that has e.g. ``LLM_API_KEY`` set.
    """
    pytest.importorskip("cognee")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    from scripts.adapters.cognee_adapter import CogneeAdapter

    adapter = CogneeAdapter()
    assert adapter.status.ok is False
    # Reason should reference cognee + the canonical OpenAI key.
    assert "OPENAI_API_KEY" in adapter.status.reason
