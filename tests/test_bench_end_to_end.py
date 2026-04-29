"""Tests for the *end-to-end* other engine bench suite.

Where `test_bench_vs_others.py` exercises the retrieval-only bench
(pre-curated `BenchMemory` records → engine ranks them), this suite
covers the new pipeline-cost path: raw conversations → engine extracts
its OWN memories → query → ranking. The bench measures total cost
(latency, tokens, RAM) across the full pipeline so a memoirs-vs-mem0
comparison stops being unfair (mem0 was paying LLM cost the old bench
couldn't see).

What this suite proves:
  1. `BenchConversation` / `BenchSuite` shapes are well-formed.
  2. `MemoirsAdapter.ingest_conversation` runs Gemma extract +
     consolidate against a tempdir DB and stamps `bench_conv_id` so
     `resolve_conv_id` can route memories back to gold.
  3. `Mem0Adapter.ingest_conversation` calls `Memory.add(messages=…)`
     with the OpenAI client mocked so we never hit the network.
  4. `CogneeAdapter.ingest_conversation` joins messages and runs
     cognify (mocked) — the conv_id resolves through the content map.
  5. The default fallback in `EngineAdapter.ingest_conversation` works
     for any adapter that doesn't override it (e.g. zep / letta in the
     bench): one memory per message, IDs `{conv}::{idx}`.
  6. Tokens are counted (tiktoken when available, char-heuristic
     otherwise).
  7. The end-to-end runner with all-mocked adapters completes well
     under 30s.
  8. The CLI smoke `--suite end-to-end --engines memoirs` exits 0,
     writes a parseable JSON artifact, and prints an end-to-end table
     with the new `mode` / `tokens` columns.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Sequence
from unittest import mock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 1. Suite shape
# ---------------------------------------------------------------------------


def test_end_to_end_suite_shape_is_well_formed():
    """Suite must contract: ≥8 conversations, ≥20 queries, every gold ID
    points at a real conversation, and every conv has ≥2 expected
    memories embedded in its messages."""
    from scripts.bench_vs_others_dataset import (
        BenchConversation,
        BenchSuite,
        build_end_to_end_suite,
    )

    suite = build_end_to_end_suite()
    assert isinstance(suite, BenchSuite)
    assert len(suite.conversations) >= 8
    assert len(suite.queries) >= 20

    # Every conversation has stable id + at least 2 expected memories.
    ids = [c.id for c in suite.conversations]
    assert len(set(ids)) == len(ids), "conversation IDs must be unique"
    for c in suite.conversations:
        assert isinstance(c, BenchConversation)
        assert c.messages, f"{c.id} has no messages"
        assert all(m.get("role") in {"user", "assistant"} for m in c.messages)
        assert len(c.expected_memories) >= 2, (
            f"{c.id} should embed ≥2 durable memories, got "
            f"{len(c.expected_memories)}"
        )

    # Every query gold ID maps to a real conversation.
    known = set(ids)
    for q in suite.queries:
        assert q.gold_memory_ids
        for gid in q.gold_memory_ids:
            assert gid in known, f"unknown gold conv id {gid!r}"


def test_bench_conversation_dataclass_is_frozen():
    """BenchConversation must be immutable so accidental in-place
    mutation can't poison the corpus across runs."""
    from dataclasses import FrozenInstanceError

    from scripts.bench_vs_others_dataset import BenchConversation

    c = BenchConversation(id="x", messages=[{"role": "user", "content": "hi"}],
                          expected_memories=["x"])
    with pytest.raises(FrozenInstanceError):
        c.id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. MemoirsAdapter ingest_conversation
# ---------------------------------------------------------------------------


def test_memoirs_adapter_ingest_conversation_runs_extract_and_consolidate(
    tmp_path, monkeypatch,
):
    """`MemoirsAdapter.ingest_conversation` must (a) persist messages,
    (b) call extract_memory_candidates, (c) call consolidate_pending,
    and (d) tag persisted memories with `bench_conv_id` so
    `resolve_conv_id` returns the conv id.

    We mock Gemma extraction with a stub that returns one Candidate so
    the test runs in <1s without invoking the real model.
    """
    from scripts.adapters.memoirs_adapter import MemoirsAdapter
    from scripts.bench_vs_others_dataset import BenchConversation

    from memoirs.engine.gemma import Candidate

    # Stub out gemma_extract so we don't need a GGUF model.
    def fake_gemma(messages):
        return [Candidate(
            type="preference",
            content="user prefers Python over Go",
            importance=4,
            confidence=0.9,
            source_message_ids=[m.get("id") for m in messages if m.get("id")],
            extractor="gemma-test-stub",
        )]

    # Force the cascade to use our stub by stubbing _have_gemma → True
    # AND replacing gemma_extract.
    import memoirs.engine.curator as curator_mod
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)
    monkeypatch.setattr(curator_mod, "curator_extract", fake_gemma)

    adapter = MemoirsAdapter(db_dir=tmp_path / "memoirs_e2e")
    try:
        assert adapter.status.ok, adapter.status.reason
        assert adapter.supports_native_ingest is True

        conv = BenchConversation(
            id="conv_test_python",
            messages=[
                {"role": "user", "content": "hey, working on a microservice"},
                {"role": "user", "content": "I prefer Python over Go for backend"},
            ],
            expected_memories=["user prefers Python over Go"],
        )
        adapter.ingest_conversation(conv)

        # The stamping step must produce at least one memory tagged
        # with our bench_conv_id.
        rows = adapter._db.conn.execute(
            "SELECT id, json_extract(metadata_json, '$.bench_conv_id') AS cid "
            "FROM memories"
        ).fetchall()
        cids = [r["cid"] for r in rows]
        assert "conv_test_python" in cids, (
            f"expected conv id stamped on memories, got {cids!r}"
        )
        # resolve_conv_id round-trips for at least one of those memories.
        tagged = next(r for r in rows if r["cid"] == "conv_test_python")
        assert adapter.resolve_conv_id(tagged["id"]) == "conv_test_python"
    finally:
        adapter.shutdown()


def test_memoirs_adapter_sets_curator_env_default_to_auto(tmp_path, monkeypatch):
    """When the user hasn't pinned MEMOIRS_GEMMA_CURATOR, the adapter
    flips it to 'auto' on first ingest_conversation so consolidation
    actually exercises Qwen. If the user already set it, we honor."""
    from scripts.adapters.memoirs_adapter import MemoirsAdapter
    from scripts.bench_vs_others_dataset import BenchConversation
    import memoirs.engine.curator as curator_mod

    monkeypatch.delenv("MEMOIRS_GEMMA_CURATOR", raising=False)
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)

    adapter = MemoirsAdapter(db_dir=tmp_path / "memoirs_curator")
    try:
        adapter.ingest_conversation(BenchConversation(
            id="c1",
            messages=[{"role": "user", "content": "small note"}],
            expected_memories=["x"],
        ))
        import os as _os
        assert _os.environ.get("MEMOIRS_GEMMA_CURATOR") == "auto"
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# 3. Mem0 adapter ingest_conversation (mocked)
# ---------------------------------------------------------------------------


def test_mem0_adapter_ingest_conversation_passes_messages_list(monkeypatch):
    """Mem0 should receive the conversation as `messages=[...]` so it
    triggers its native extraction. We mock the `Memory` class so no
    OpenAI call goes out, then assert the call shape and confirm the
    returned event ID maps to the conv id via `resolve_conv_id`."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    from scripts.adapters import mem0_adapter as ma

    # Build a fake Memory whose `add` records the call kwargs.
    captured: dict = {}

    class FakeMemory:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def add(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {"results": [{"id": "mem0-evt-1",
                                 "memory": "user prefers Python"}]}

        def search(self, *_a, **_k):
            return []

    fake_mod = mock.MagicMock()
    fake_mod.Memory = FakeMemory
    monkeypatch.setitem(sys.modules, "mem0", fake_mod)

    adapter = ma.Mem0Adapter()
    try:
        assert adapter.status.ok, adapter.status.reason
        assert adapter.supports_native_ingest is True
        from scripts.bench_vs_others_dataset import BenchConversation
        conv = BenchConversation(
            id="conv_xyz",
            messages=[
                {"role": "user", "content": "I prefer Python"},
                {"role": "assistant", "content": "noted"},
            ],
            expected_memories=["user prefers Python"],
        )
        adapter.ingest_conversation(conv)

        # `add` must have been called with `messages=` kwarg containing
        # both turns (Mem0's native API contract).
        kwargs = captured.get("kwargs") or {}
        assert "messages" in kwargs
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["content"] == "I prefer Python"
        # The event id should resolve back to the conv id.
        assert adapter.resolve_conv_id("mem0-evt-1") == "conv_xyz"
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# 4. Cognee adapter ingest_conversation (mocked async)
# ---------------------------------------------------------------------------


def test_cognee_adapter_ingest_conversation_joins_messages(monkeypatch):
    """Cognee gets the full conversation as one document. We mock both
    `cognee.add` and `cognee.cognify` to verify the join + that the
    content map is populated for `resolve_conv_id` later."""
    pytest.importorskip("cognee")

    from scripts.adapters.cognee_adapter import CogneeAdapter
    from scripts.bench_vs_others_dataset import BenchConversation

    add_calls: list[str] = []

    async def fake_add(text):
        add_calls.append(text)

    async def fake_cognify(*_a, **_k):
        return None

    import cognee
    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", fake_cognify)

    adapter = CogneeAdapter()
    try:
        if not adapter.status.ok:
            pytest.skip(f"cognee unavailable: {adapter.status.reason}")
        assert adapter.supports_native_ingest is True
        conv = BenchConversation(
            id="conv_multi",
            messages=[
                {"role": "user", "content": "alpha"},
                {"role": "assistant", "content": "beta"},
                {"role": "user", "content": "gamma"},
            ],
            expected_memories=["alpha", "gamma"],
        )
        adapter.ingest_conversation(conv)
        assert len(add_calls) == 1
        joined = add_calls[0]
        assert "alpha" in joined and "beta" in joined and "gamma" in joined
        # Cognee resolves conv via its content_to_id map.
        # Per-message keys are populated so a substring search returns it.
        assert adapter._content_to_id.get("alpha") == "conv_multi"
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# 5. Default fallback for adapters that don't override ingest_conversation
# ---------------------------------------------------------------------------


def test_default_fallback_creates_one_memory_per_message():
    """An adapter that only implements `add_memories` must still get a
    working e2e path: the base class synthesizes one memory per turn
    and registers `{conv}::{idx}` → conv.id in the lookup."""
    from scripts.adapters.base import EngineAdapter
    from scripts.bench_vs_others_dataset import (
        BenchConversation, BenchMemory, BenchQuery,
    )

    class StubAdapter(EngineAdapter):
        name = "stub"

        def __init__(self):
            super().__init__()
            self.added: list[BenchMemory] = []

        def add_memories(self, memories: Sequence[BenchMemory]) -> None:
            self.added.extend(memories)

        def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
            return [m.id for m in self.added[:top_k]]

    a = StubAdapter()
    conv = BenchConversation(
        id="conv_fb", messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": ""},  # empty — should be skipped
            {"role": "user", "content": "third"},
        ],
        expected_memories=["first", "third"],
    )
    a.ingest_conversation(conv)
    # Empty message dropped → 3 synthetic memories.
    assert len(a.added) == 3
    assert a.added[0].id == "conv_fb::0"
    assert a.added[2].id == "conv_fb::3"  # idx preserved across skip
    assert a.resolve_conv_id("conv_fb::0") == "conv_fb"
    assert a.resolve_conv_id("nonexistent") is None


# ---------------------------------------------------------------------------
# 6. Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_returns_positive_for_nonempty_text():
    """Token counter must return >0 for non-empty text and 0 for empty,
    regardless of whether tiktoken is installed."""
    from scripts.bench_vs_others import _estimate_tokens

    assert _estimate_tokens("") == 0
    n = _estimate_tokens("hello world this is a sentence")
    assert n > 0
    # The fallback heuristic returns ~len/4 — make sure that's also a
    # sensible floor when tiktoken is missing.
    with mock.patch.dict(sys.modules, {"tiktoken": None}):
        approx = _estimate_tokens("a" * 40)
        assert 1 <= approx <= 40


# ---------------------------------------------------------------------------
# 7. End-to-end runner with mocked adapters completes fast
# ---------------------------------------------------------------------------


def test_run_engine_end_to_end_with_mocked_adapter_finishes_fast():
    """The full e2e runner must complete in well under 30s when adapters
    don't actually call any LLM (which is the harness's realistic upper
    bound for CI). We use a stub that just counts conversations + makes
    every query a hit so the metric path also runs."""
    from scripts.adapters.base import EngineAdapter
    from scripts.bench_vs_others import run_engine_end_to_end
    from scripts.bench_vs_others_dataset import (
        BenchMemory, BenchQuery, build_end_to_end_suite,
    )

    class PerfectStub(EngineAdapter):
        name = "perfect-stub"
        supports_native_ingest = True

        def __init__(self):
            super().__init__()
            self._conv_for_query: dict[str, list[str]] = {}

        def add_memories(self, memories: Sequence[BenchMemory]) -> None:
            return None

        def ingest_conversation(self, conv) -> None:
            # Index every word from expected_memories under conv.id so
            # the `query` step can answer perfectly.
            self._register_conv_link(conv.id, conv.id)

        def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
            # Return the gold conv IDs verbatim — perfect retrieval.
            return list(q.gold_memory_ids)[:top_k]

    suite = build_end_to_end_suite()
    a = PerfectStub()
    t0 = time.perf_counter()
    rep = run_engine_end_to_end(a, suite, top_k=10)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"e2e runner took {elapsed:.1f}s, want <30s"
    assert rep.status == "OK"
    assert rep.mode == "e2e-native"
    # Tokens were counted across all conversations (>0).
    assert rep.tokens_used > 0
    # The stub returns gold ids → metrics should be perfect.
    assert rep.hit_at_1 == pytest.approx(1.0)
    assert rep.mrr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. CLI smoke for --suite end-to-end
# ---------------------------------------------------------------------------


def test_cli_runs_with_suite_end_to_end_and_emits_extended_table(
    tmp_path, capsys, monkeypatch,
):
    """`bench_vs_others --suite end-to-end --engines memoirs` must:
      - exit 0
      - write a JSON artifact whose `suite` field is "end-to-end"
      - print a markdown table with the `mode` / `tokens` columns
      - the memoirs row records `mode` starting with `e2e-`
    """
    # Avoid touching real Gemma / OpenAI during the smoke run by
    # forcing the spaCy fallback off and the curator off as well —
    # the adapter still records ingest latency + tokens even when the
    # extractor produced 0 candidates.
    import memoirs.engine.curator as curator_mod
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)
    import memoirs.engine.extract_spacy as spacy_mod
    monkeypatch.setattr(spacy_mod, "is_available", lambda: False)

    from scripts.bench_vs_others import main

    out_json = tmp_path / "report_e2e.json"
    rc = main([
        "--suite", "end-to-end",
        "--engines", "memoirs",
        "--top-k", "5",
        "--out", str(out_json),
        "--quiet",
    ])
    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert data["suite"] == "end-to-end"
    assert data["n_engines"] == 1
    rec = data["engines"][0]
    assert rec["engine"] == "memoirs"
    # mode is set by the runner to e2e-native (memoirs supports native).
    assert rec["mode"].startswith("e2e-"), f"unexpected mode {rec['mode']!r}"
    # Tokens accounted from input messages — must be a positive integer.
    assert isinstance(rec["tokens_used"], int) and rec["tokens_used"] > 0

    out = capsys.readouterr().out
    assert "end-to-end" in out
    assert "mode" in out
    assert "tokens" in out
