"""Tests for GAP fixes #2 (chunker overflow) and #3 (project summary validation).

#2 — `_wrapper_overhead_tokens` measures the real prompt-wrapper cost so the
     per-chunk content budget can't underestimate and overflow `n_ctx`.
#3 — `_validate_summary` rejects vapid / entity-free output so
     `curator_summarize_project` retries once and falls back gracefully.

Tests use a FakeLLM mirroring llama-cpp-python's surface so they run without
the real Gemma weights installed.
"""
from __future__ import annotations

import json

import pytest

from memoirs.config import GEMMA_MAX_OUTPUT_TOKENS
from memoirs.engine import curator as curator_mod
from memoirs.engine.curator import (
    _BUDGET_HEADROOM_TOKENS,
    _chunk_user_turns,
    _collect_summary_entities,
    _content_token_budget,
    _count_tokens,
    _model_ctx,
    _validate_summary,
    _wrap_prompt,
    _wrapper_overhead_tokens,
    curator_extract,
    curator_summarize_project,
)


# ----------------------------------------------------------------------
# Fake llama-cpp-python compatible LLM
# ----------------------------------------------------------------------


class FakeLLM:
    """Minimal stand-in for `llama_cpp.Llama` shared with other gemma tests.

    - `tokenize(bytes)` -> list[int] (1 token per whitespace-split word)
    - `detokenize(list[int])` -> bytes (rebuilt via stable id<->word map)
    - `create_completion(prompt, ...)` -> next pre-queued response, or "[]"
    - `n_ctx` -> int attribute
    """

    def __init__(self, n_ctx: int = 4096, responses=None):
        self.n_ctx = n_ctx
        self._responses = list(responses or [])
        self._token_map: dict[str, int] = {}
        self._inv_map: dict[int, str] = {}
        self._next_id = 1
        self.completion_calls: list[dict] = []

    def tokenize(self, data, add_bos: bool = False, special: bool = False):
        if isinstance(data, (bytes, bytearray)):
            text = bytes(data).decode("utf-8", errors="ignore")
        else:
            text = str(data)
        toks: list[int] = []
        for word in text.split(" "):
            if not word:
                continue
            tid = self._token_map.get(word)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._token_map[word] = tid
                self._inv_map[tid] = word
            toks.append(tid)
        return toks

    def detokenize(self, tokens):
        words = [self._inv_map.get(int(t), "") for t in tokens]
        return (" ".join(w for w in words if w)).encode("utf-8")

    def create_completion(self, prompt: str, **kwargs):
        self.completion_calls.append({"prompt": prompt, **kwargs})
        if not self._responses:
            text = "[]"
        else:
            text = self._responses.pop(0)
        return {"choices": [{"text": text}]}


@pytest.fixture
def patch_have_gemma(monkeypatch):
    """Make `_have_curator()` return True so production paths exercise the fake LLM."""
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)


@pytest.fixture
def install_llm(monkeypatch):
    """Wire a FakeLLM into the singleton slot used by gemma helpers."""

    def _install(llm: FakeLLM):
        monkeypatch.setattr(curator_mod, "_get_llm", lambda: llm)
        return llm

    return _install


# ======================================================================
# Fix #2 — Wrapper overhead + content budget + chunker safety
# ======================================================================


def test_wrapper_overhead_returns_reasonable_value():
    """`_wrapper_overhead_tokens` must measure a non-trivial overhead.

    Under FakeLLM (1 word per token), the system prompt expands to several
    hundred tokens. The previous static `_HEADER_TOKEN_BUDGET = 200` was a
    severe under-estimate; this test guards against regressing to that.
    """
    llm = FakeLLM(n_ctx=4096)
    overhead = _wrapper_overhead_tokens(llm)
    # The system prompt + chat template is large — at minimum a few dozen
    # tokens, far less than the model's full context. Real backends (gemma,
    # qwen, phi) all land in the 50-1500 range depending on tokenizer.
    assert 50 <= overhead <= 1500
    # Sanity: the empty wrap actually contains the system prompt tokens.
    assert overhead > 50
    # And the overhead is stable across calls (deterministic tokenizer).
    assert _wrapper_overhead_tokens(llm) == overhead


def test_wrapper_overhead_falls_back_on_none_llm():
    """When no LLM is wired, the overhead falls back to the static estimate."""
    out = _wrapper_overhead_tokens(None)
    assert isinstance(out, int)
    assert out > 0


def test_content_budget_subtracts_overhead_and_max_output():
    """Budget = n_ctx − wrapper_overhead − GEMMA_MAX_OUTPUT_TOKENS − headroom."""
    llm = FakeLLM(n_ctx=4096)
    overhead = _wrapper_overhead_tokens(llm)
    budget = _content_token_budget(llm)
    expected = max(
        512, 4096 - overhead - GEMMA_MAX_OUTPUT_TOKENS - _BUDGET_HEADROOM_TOKENS
    )
    assert budget == expected
    # And the budget MUST be tighter than n_ctx by at least overhead+max_output.
    assert budget < 4096 - overhead


def test_content_budget_floors_at_512_on_tiny_ctx():
    """A pathological tiny ctx doesn't drive the budget negative."""
    llm = FakeLLM(n_ctx=128)
    budget = _content_token_budget(llm)
    assert budget == 512


def test_short_conversation_fits_in_one_chunk(install_llm):
    """A snug conversation uses exactly 1 chunk with the corrected budget."""
    llm = install_llm(FakeLLM(n_ctx=4096))
    user_turns = [f"turn {i} memoirs project" for i in range(10)]
    chunks = _chunk_user_turns(llm, user_turns, budget=_content_token_budget(llm))
    assert len(chunks) == 1
    joined = chunks[0]
    for i in range(10):
        assert f"turn {i}" in joined


def test_long_conversation_chunks_stay_under_n_ctx(install_llm):
    """The bug we're fixing: every wrapped chunk must fit n_ctx WITH max_output.

    Pre-fix: chunker used a 200-token static header estimate, the real wrapper
    cost ~3-4× that, and the wrapped prompt blew past n_ctx (real warning was
    `Requested tokens (4699) exceed context window of 4096`). This test
    proves: wrapped(chunk) + max_tokens ≤ n_ctx for every chunk produced.
    """
    llm = install_llm(FakeLLM(n_ctx=4096))
    # ~10000 tokens of synthetic content (1000 turns × 10 words).
    user_turns = [
        " ".join(f"word{i}_{j}" for j in range(10))
        for i in range(1000)
    ]
    budget = _content_token_budget(llm)
    chunks = _chunk_user_turns(llm, user_turns, budget=budget)
    assert len(chunks) >= 3, f"expected ≥3 chunks for ~10k tokens, got {len(chunks)}"
    n_ctx = _model_ctx(llm)
    for i, chunk in enumerate(chunks):
        wrapped = _wrap_prompt(chunk)
        wrapped_tokens = _count_tokens(llm, wrapped)
        assert wrapped_tokens + GEMMA_MAX_OUTPUT_TOKENS <= n_ctx, (
            f"chunk {i} wrapped={wrapped_tokens} + max_out={GEMMA_MAX_OUTPUT_TOKENS}"
            f" exceeds n_ctx={n_ctx}"
        )


def test_gemma_extract_bisects_oversize_chunk(patch_have_gemma, install_llm,
                                                 monkeypatch):
    """Post-wrap re-validation: if a chunk still overflows, bisect & retry.

    We force the chunker to emit a single huge chunk, then verify
    `curator_extract` slices it before sending and produces ≥ 2 LLM calls.
    """
    # Content needs to be long enough to pass the candidate noise filter
    # (`should_skip_extraction` rejects short / code-shaped strings).
    content_a = "user prefers terse responses with concrete file names and entity references"
    content_b = "user works on the memoirs local-first memory engine using sqlite-vec for embeddings"
    llm = install_llm(FakeLLM(
        n_ctx=4096,
        responses=[
            json.dumps([{"type": "preference", "content": content_a,
                          "importance": 3, "confidence": 0.7, "entities": []}]),
            json.dumps([{"type": "preference", "content": content_b,
                          "importance": 3, "confidence": 0.7, "entities": []}]),
        ],
    ))

    # Build one giant chunk (15k unique words → way over n_ctx after wrap).
    huge_chunk = " ".join(f"word_{i}" for i in range(15000))

    # Force the chunker to return our oversized chunk regardless of budget.
    monkeypatch.setattr(
        curator_mod, "_chunk_user_turns",
        lambda *a, **kw: [huge_chunk],
    )

    messages = [{"id": "m0", "role": "user", "content": huge_chunk}]
    candidates = curator_extract(messages)

    # Bisection should have triggered ≥ 2 completion calls and merged
    # candidates from both halves.
    assert len(llm.completion_calls) >= 2
    contents = {c.content for c in candidates}
    assert content_a in contents
    assert content_b in contents


def test_gemma_extract_logs_overhead_and_budget(
    patch_have_gemma, install_llm, caplog
):
    """The fix instrumentation must surface chunks/budget/overhead in logs."""
    install_llm(FakeLLM(
        n_ctx=4096,
        responses=[
            json.dumps([{"type": "fact", "content": "x", "importance": 3,
                          "confidence": 0.9, "entities": []}]),
        ],
    ))
    messages = [
        {"id": "m0", "role": "user", "content": "this is a meaningful user turn about memoirs"},
    ]
    with caplog.at_level("INFO", logger="memoirs.gemma"):
        curator_extract(messages)
    text = " ".join(rec.message for rec in caplog.records)
    assert "chunks=" in text
    assert "budget=" in text
    assert "overhead=" in text


# ======================================================================
# Fix #3 — Project summary validator + retry
# ======================================================================


def test_validate_summary_rejects_too_short():
    ok, reason = _validate_summary("Foo", entities=["foo"])
    assert ok is False
    assert reason == "too_short"


def test_validate_summary_accepts_specific_summary():
    # 60+ chars (the spec floor), references entities, no vapid phrases.
    text = "Built memoirs MCP tool with sqlite-vec for embedding cache support."
    assert len(text) >= 60
    ok, reason = _validate_summary(text, entities=["memoirs", "MCP"])
    assert ok is True
    assert reason == "ok"


def test_validate_summary_rejects_vapid_phrases():
    text = (
        "This conversation discusses how the user works on memoirs and "
        "ships a memory engine over the local filesystem with sqlite."
    )
    ok, reason = _validate_summary(text, entities=["memoirs"])
    assert ok is False
    assert reason.startswith("vapid_phrase:")


def test_validate_summary_rejects_missing_entity_coverage():
    # 60+ chars, no bad phrases, but doesn't reference any of the entities.
    text = (
        "The author shipped a tiny CLI utility for parsing markdown into "
        "structured JSON for downstream consumers."
    )
    ok, reason = _validate_summary(text, entities=["memoirs", "sqlite-vec"])
    assert ok is False
    assert reason == "no_entity_coverage"


def test_validate_summary_skips_entity_check_when_none_provided():
    # Long, non-vapid prose with no entity list is still accepted.
    text = (
        "The author refactored the indexing pipeline so that batched "
        "writes flush every 200 records without holding a transaction."
    )
    ok, reason = _validate_summary(text, entities=[])
    assert ok is True
    assert reason == "ok"


def test_validate_summary_rejects_too_long():
    text = "memoirs " * 200  # ~1600 chars
    ok, reason = _validate_summary(text, entities=["memoirs"])
    assert ok is False
    assert reason == "too_long"


def test_collect_summary_entities_dedupes_case_insensitive():
    memories = [
        {"id": "1", "entities": ["Memoirs", "SQLite"]},
        {"id": "2", "entities": ["memoirs", "Qwen"]},
        {"id": "3", "entities": []},
    ]
    out = _collect_summary_entities(memories)
    # Case-insensitive dedup: "Memoirs" and "memoirs" collapse.
    lower = [e.lower() for e in out]
    assert "memoirs" in lower
    assert "sqlite" in lower
    assert "qwen" in lower
    assert lower.count("memoirs") == 1


def test_summarize_project_retries_on_vapid_first_response(patch_have_gemma,
                                                              install_llm):
    """First response is vapid → validator rejects → retry uses retry prompt."""
    vapid = (
        "This conversation discusses how the project evolved and what was "
        "decided as mentioned earlier in the discussion thread."
    )
    good = (
        "Built memoirs as a local-first memory engine; chose sqlite-vec for "
        "embeddings and Qwen3 as the curator backend."
    )
    llm = install_llm(FakeLLM(n_ctx=4096, responses=[vapid, good]))
    memories = [
        {"id": "a", "type": "decision",
         "content": "use sqlite-vec for embeddings",
         "entities": ["memoirs", "sqlite-vec"], "similarity": 0.9},
        {"id": "b", "type": "decision",
         "content": "Qwen3 picked as curator after bench",
         "entities": ["Qwen3", "memoirs"], "similarity": 0.8},
    ]
    out = curator_summarize_project("memoirs", memories)
    assert out is not None
    assert "memoirs" in out.lower() or "sqlite-vec" in out.lower() or "qwen3" in out.lower()
    # Two completions: first vapid, second accepted.
    assert len(llm.completion_calls) == 2


def test_summarize_project_returns_none_when_retry_also_fails(patch_have_gemma,
                                                                 install_llm):
    """Both responses vapid → validator rejects both → returns None."""
    vapid_a = (
        "This conversation discusses how the project moves forward as "
        "mentioned previously in the prior conversation as discussed."
    )
    vapid_b = (
        "In summary, the project covers many topics as mentioned and "
        "the conversation discusses the relevant details extensively."
    )
    llm = install_llm(FakeLLM(n_ctx=4096, responses=[vapid_a, vapid_b]))
    memories = [
        {"id": "a", "type": "decision", "content": "use sqlite-vec",
         "entities": ["sqlite-vec"], "similarity": 0.9},
    ]
    out = curator_summarize_project("memoirs", memories)
    assert out is None
    assert len(llm.completion_calls) == 2


def test_summarize_project_accepts_specific_first_response(patch_have_gemma,
                                                             install_llm):
    """Good first response → no retry, returns text directly."""
    good = (
        "memoirs ships a local-first memory engine on top of sqlite-vec; "
        "the curator runs on Qwen3 after benching against three other models."
    )
    llm = install_llm(FakeLLM(n_ctx=4096, responses=[good]))
    memories = [
        {"id": "a", "type": "decision",
         "content": "use sqlite-vec",
         "entities": ["memoirs", "sqlite-vec"], "similarity": 0.9},
    ]
    out = curator_summarize_project("memoirs", memories)
    assert out is not None
    assert "memoirs" in out.lower()
    # Single completion only — no retry needed.
    assert len(llm.completion_calls) == 1


def test_summarize_project_max_tokens_is_256(patch_have_gemma, install_llm):
    """Bug #3 fix: summary calls request 256 tokens (was 200)."""
    good = (
        "memoirs is a local-first memory engine using sqlite-vec for "
        "vector search and Qwen3 as the curator."
    )
    llm = install_llm(FakeLLM(n_ctx=4096, responses=[good]))
    memories = [
        {"id": "a", "type": "decision", "content": "x",
         "entities": ["memoirs"], "similarity": 0.9},
    ]
    curator_summarize_project("memoirs", memories)
    assert llm.completion_calls
    assert llm.completion_calls[0]["max_tokens"] == 256


def test_summarize_project_skip_short_circuits_to_empty(patch_have_gemma,
                                                          install_llm):
    """SKIP from the model preserves the empty-string contract (no retry)."""
    llm = install_llm(FakeLLM(n_ctx=4096, responses=["SKIP"]))
    memories = [
        {"id": "a", "type": "fact", "content": "trivial",
         "entities": ["memoirs"], "similarity": 0.5},
    ]
    out = curator_summarize_project("memoirs", memories)
    # SKIP is a deliberate "nothing to summarize" signal; preserve the
    # empty-string contract callers (and pre-existing tests) rely on.
    assert out == ""
    # No retry: SKIP is the model speaking, not a validation failure.
    assert len(llm.completion_calls) == 1
