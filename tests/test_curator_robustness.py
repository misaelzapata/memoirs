"""Tests for Gemma robustness: token-budget chunking + tolerant JSON parser.

These tests use a fake `llm` object that mimics llama-cpp-python's surface
(tokenize / detokenize / create_completion / n_ctx) so they run WITHOUT the
real Gemma model installed.
"""
from __future__ import annotations

import json

import pytest

from memoirs.engine import curator as curator_mod
from memoirs.engine.curator import (
    _candidate_dedup_key,
    _chunk_user_turns,
    _content_token_budget,
    _count_tokens,
    curator_extract,
    curator_resolve_conflict,
    parse_conflict_response,
)


# ----------------------------------------------------------------------
# Fake llama-cpp-python compatible LLM
# ----------------------------------------------------------------------


class FakeLLM:
    """Minimal stand-in for llama_cpp.Llama used by Gemma helpers.

    - `tokenize(bytes)` -> list of fake token ints (1 per word, deterministic)
    - `detokenize(list[int])` -> bytes (rebuilt from token-id → word map)
    - `create_completion(prompt, max_tokens, ...)` -> next pre-queued response
    - `n_ctx` -> attribute (matches llama-cpp-python public surface)
    """

    def __init__(self, n_ctx: int = 4096, responses: list[str] | None = None):
        self.n_ctx = n_ctx
        self._responses = list(responses or [])
        self._token_map: dict[str, int] = {}
        self._inv_map: dict[int, str] = {}
        self._next_id = 1
        self.completion_calls: list[dict] = []

    # llama_cpp.Llama exposes a tokenize that takes bytes -> list[int]
    def tokenize(self, data, add_bos: bool = False, special: bool = False):
        if isinstance(data, (bytes, bytearray)):
            text = bytes(data).decode("utf-8", errors="ignore")
        else:
            text = str(data)
        toks: list[int] = []
        # Approximate tokenization: split on whitespace; punctuation stays glued.
        # Each unique word maps to one stable token id.
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
    """Force `_have_curator()` to True so production paths execute under mock."""
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)


@pytest.fixture
def patch_llm(monkeypatch):
    """Install a FakeLLM as the gemma singleton; tests can configure responses."""
    holder: dict = {}

    def install(llm: FakeLLM):
        holder["llm"] = llm
        monkeypatch.setattr(curator_mod, "_get_llm", lambda: llm)
        return llm

    return install


# ----------------------------------------------------------------------
# Bug 1 — Token budgeting + chunking
# ----------------------------------------------------------------------


def test_count_tokens_uses_llm_tokenize_not_chars_over_4(patch_llm):
    """Token counting MUST go through llm.tokenize, not len(text)//4."""
    llm = FakeLLM(n_ctx=4096)
    patch_llm(llm)

    text = "alpha beta gamma delta epsilon"
    tok_count = _count_tokens(llm, text)

    # Char heuristic would give len("alpha beta gamma delta epsilon")//4 = 7.
    # Real (fake) tokenizer gives one token per word = 5. They MUST differ
    # to prove we don't use the heuristic.
    assert tok_count == 5
    assert tok_count != len(text) // 4

    # Sanity: re-tokenizing same text returns same count (stable).
    assert _count_tokens(llm, text) == 5


def test_content_budget_subtracts_header_and_output(patch_llm):
    """Bug #2 fix: budget = n_ctx − measured_wrapper_overhead − output − headroom.

    The wrapper overhead is now measured with the real tokenizer instead of
    a hard-coded estimate; under FakeLLM (1 word ≈ 1 token) the overhead is
    ~600 tokens, so the budget shrinks accordingly. We assert the floor
    instead of pinning the exact number to avoid coupling to the system
    prompt's exact wording.
    """
    from memoirs.engine.curator import _wrapper_overhead_tokens
    from memoirs.config import GEMMA_MAX_OUTPUT_TOKENS

    llm = FakeLLM(n_ctx=4096)
    patch_llm(llm)
    overhead = _wrapper_overhead_tokens(llm)
    budget = _content_token_budget(llm)
    assert budget == max(512, 4096 - overhead - GEMMA_MAX_OUTPUT_TOKENS - 64)
    # Sanity: even with a chunky overhead, the budget must leave meaningful
    # room for content (≥ 1 KB worth of tokens under FakeLLM).
    assert budget >= 512


def test_short_conversation_fits_in_one_chunk(patch_llm):
    """Tight-fit conversation produces exactly 1 chunk, no over-truncation."""
    llm = FakeLLM(n_ctx=4096)
    patch_llm(llm)

    # 10 short turns -> well under 3196 token budget.
    user_turns = [f"turn number {i} about memoirs project" for i in range(10)]
    chunks = _chunk_user_turns(llm, user_turns, budget=_content_token_budget(llm))
    assert len(chunks) == 1
    # All original words must be retained somewhere in the single chunk.
    joined = chunks[0]
    for i in range(10):
        assert f"turn number {i}" in joined


def test_long_conversation_splits_into_multiple_chunks(patch_llm):
    """A ~10k token conversation must produce N>=3 chunks."""
    llm = FakeLLM(n_ctx=4096)
    patch_llm(llm)

    # ~10000 tokens via 1000 turns of 10 unique words each (FakeLLM = 1 word -> 1 tok)
    user_turns = [
        " ".join(f"word{i}_{j}" for j in range(10))
        for i in range(1000)
    ]
    chunks = _chunk_user_turns(llm, user_turns, budget=_content_token_budget(llm))
    assert len(chunks) >= 3, f"expected ≥3 chunks for ~10k tokens, got {len(chunks)}"

    # No chunk may exceed the budget.
    budget = _content_token_budget(llm)
    for c in chunks:
        assert _count_tokens(llm, c) <= budget, "chunk exceeded token budget"


def test_gemma_extract_processes_all_chunks(patch_have_gemma, patch_llm):
    """curator_extract must round-trip through every chunk and merge candidates."""
    cand_a = json.dumps([
        {"type": "preference", "content": "prefers Python over Go",
         "importance": 4, "confidence": 0.9, "entities": ["Python", "Go"]},
    ])
    cand_b = json.dumps([
        {"type": "decision", "content": "will use SQLite + sqlite-vec",
         "importance": 4, "confidence": 0.85, "entities": ["SQLite"]},
    ])
    cand_c = json.dumps([
        # Duplicate of cand_a -> must be deduped.
        {"type": "preference", "content": "prefers Python over Go",
         "importance": 4, "confidence": 0.9, "entities": []},
        # Padded over 20 chars so should_skip_extraction's "too short" rule
        # doesn't fire — the test is about chunk dedup, not the noise filter.
        {"type": "project", "content": "currently working on memoirs",
         "importance": 5, "confidence": 0.95, "entities": ["memoirs"]},
    ])
    llm = FakeLLM(n_ctx=4096, responses=[cand_a, cand_b, cand_c])
    patch_llm(llm)

    user_turns = [
        " ".join(f"word{i}_{j}" for j in range(10))
        for i in range(1000)
    ]
    messages = [
        {"id": f"m{i}", "role": "user", "content": t}
        for i, t in enumerate(user_turns)
    ]

    candidates = curator_extract(messages)

    # FakeLLM was called once per chunk; we queued exactly 3 responses.
    assert len(llm.completion_calls) >= 3
    # Dedup: prefers Python appears twice in raw output, once in result.
    contents = [c.content for c in candidates]
    assert contents.count("prefers Python over Go") == 1
    # Other candidates merged in.
    assert "will use SQLite + sqlite-vec" in contents
    assert "currently working on memoirs" in contents


def test_candidate_dedup_key_is_stable_across_normalization():
    from memoirs.engine.curator import Candidate
    a = Candidate(type="preference", content="Prefers   Python  over Go")
    b = Candidate(type="preference", content="prefers python over go")
    assert _candidate_dedup_key(a) == _candidate_dedup_key(b)


# ----------------------------------------------------------------------
# Bug 2 — Tolerant JSON parser for curator_resolve_conflict
# ----------------------------------------------------------------------


def test_parse_conflict_bare_contradictory_string():
    out = parse_conflict_response('"contradictory"')
    assert out == {"action": "MARK_CONFLICT", "reason": "contradictory"}


def test_parse_conflict_bare_conflict_string_alias():
    out = parse_conflict_response('"conflict"')
    assert out["action"] == "MARK_CONFLICT"


def test_parse_conflict_bare_merge_string():
    out = parse_conflict_response('"merge"')
    assert out == {"action": "MERGE"}


def test_parse_conflict_markdown_fenced_merge():
    out = parse_conflict_response('```json\n"merge"\n```')
    assert out == {"action": "MERGE"}


def test_parse_conflict_bare_keep_and_noop():
    assert parse_conflict_response('"keep"')["action"] == "NOOP"
    assert parse_conflict_response('"noop"')["action"] == "NOOP"
    assert parse_conflict_response('"none"')["action"] == "NOOP"


def test_parse_conflict_empty_input_returns_none():
    assert parse_conflict_response("") is None
    assert parse_conflict_response("   ") is None
    assert parse_conflict_response(None) is None  # type: ignore[arg-type]


def test_parse_conflict_unknown_bare_string_returns_none():
    assert parse_conflict_response('"hodor"') is None


def test_parse_conflict_valid_json_object_passthrough():
    obj = {"contradictory": True, "winner": "A", "reason": "newer"}
    out = parse_conflict_response(json.dumps(obj))
    assert out is not None
    assert out["contradictory"] is True
    assert out["winner"] == "A"
    assert out["action"] == "MARK_CONFLICT"


def test_parse_conflict_valid_json_object_compatible():
    obj = {"contradictory": False, "winner": None, "reason": "compatible"}
    out = parse_conflict_response(json.dumps(obj))
    assert out is not None
    assert out["contradictory"] is False
    assert out["action"] == "NOOP"


def test_parse_conflict_strips_bom_and_whitespace():
    out = parse_conflict_response("﻿  \"merge\"  ")
    assert out == {"action": "MERGE"}


def test_parse_conflict_recovers_object_from_chatty_output():
    text = 'Sure! Here is your verdict:\n{"contradictory": true, "winner": "B", "reason": "specific"}\nDone.'
    out = parse_conflict_response(text)
    assert out is not None
    assert out["contradictory"] is True
    assert out["winner"] == "B"


def test_gemma_resolve_conflict_handles_bare_contradictory(patch_have_gemma, patch_llm):
    """End-to-end: model returns a bare string and we still produce a sane verdict."""
    llm = FakeLLM(n_ctx=4096, responses=['"contradictory"'])
    patch_llm(llm)
    verdict = curator_resolve_conflict("user prefers Python", "user prefers Go")
    assert verdict["action"] == "MARK_CONFLICT"
    assert verdict["contradictory"] is True
    assert verdict["winner"] is None
    assert "reason" in verdict


def test_gemma_resolve_conflict_handles_fenced_merge(patch_have_gemma, patch_llm):
    llm = FakeLLM(n_ctx=4096, responses=['```json\n"merge"\n```'])
    patch_llm(llm)
    verdict = curator_resolve_conflict("a", "b")
    assert verdict["action"] == "MERGE"
    assert verdict["contradictory"] is False


def test_gemma_resolve_conflict_handles_valid_json(patch_have_gemma, patch_llm):
    obj = {"contradictory": True, "winner": "A", "reason": "more specific"}
    llm = FakeLLM(n_ctx=4096, responses=[json.dumps(obj)])
    patch_llm(llm)
    verdict = curator_resolve_conflict("foo", "bar")
    assert verdict["contradictory"] is True
    assert verdict["winner"] == "A"
    assert verdict["action"] == "MARK_CONFLICT"


def test_gemma_resolve_conflict_unparseable_falls_back(patch_have_gemma, patch_llm):
    llm = FakeLLM(n_ctx=4096, responses=["lorem ipsum dolor sit amet"])
    patch_llm(llm)
    verdict = curator_resolve_conflict("a", "b")
    # Falls back to NOOP shape; never raises.
    assert verdict["contradictory"] is False
    assert verdict["action"] == "NOOP"
    assert verdict["winner"] is None
