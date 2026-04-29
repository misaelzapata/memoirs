"""Tests for the Gemma-driven curator (P1-11): curator_consolidate,
curator_detect_contradiction, curator_summarize_project, plus the
`decide_memory_action` integration & MEMOIRS_GEMMA_CURATOR env switch.

All tests use a `FakeLLM` mirroring llama-cpp-python's surface so they run
without the real Gemma model installed.
"""
from __future__ import annotations

import json
import os

import pytest

from memoirs.engine import curator as curator_mod
from memoirs.engine import memory_engine
from memoirs.engine.curator import (
    Candidate,
    curator_consolidate,
    curator_detect_contradiction,
    curator_summarize_project,
    parse_consolidation_response,
)


# ----------------------------------------------------------------------
# Fake llama-cpp-python compatible LLM (mirrors test_gemma_robustness.FakeLLM)
# ----------------------------------------------------------------------


class FakeLLM:
    """Minimal stand-in for llama_cpp.Llama used by Gemma helpers."""

    def __init__(self, n_ctx: int = 4096, responses: list[str] | None = None):
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
            text = "{}"
        else:
            text = self._responses.pop(0)
        return {"choices": [{"text": text}]}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@pytest.fixture
def force_curator(monkeypatch):
    """Pretend the Gemma model is installed."""
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)


@pytest.fixture
def install_llm(monkeypatch):
    """Install a FakeLLM in place of the real singleton."""

    def _install(llm: FakeLLM):
        monkeypatch.setattr(curator_mod, "_get_llm", lambda: llm)
        return llm

    return _install


@pytest.fixture(autouse=True)
def reset_curator_env(monkeypatch):
    """Don't let env settings leak between tests."""
    monkeypatch.delenv("MEMOIRS_CURATOR_ENABLED", raising=False)
    monkeypatch.delenv("MEMOIRS_GEMMA_CURATOR", raising=False)


def _candidate(content="prefers Python", ctype="preference", importance=4):
    return Candidate(type=ctype, content=content, importance=importance, confidence=0.8)


# ======================================================================
# 1. parse_consolidation_response — pure parser
# ======================================================================


def test_parse_well_formed_json_action_merge():
    out = parse_consolidation_response(
        '{"action":"MERGE","target_id":"mem_xyz","reason":"semantic dup"}'
    )
    assert out is not None
    assert out["action"] == "MERGE"
    assert out["target_id"] == "mem_xyz"
    assert out["reason"] == "semantic dup"


def test_parse_bare_string_merge():
    out = parse_consolidation_response('"merge"')
    assert out == {"action": "MERGE"}


def test_parse_bare_string_unquoted_ignore():
    # Gemma sometimes emits the bare word without JSON quoting; we still try.
    out = parse_consolidation_response('"keep"')
    assert out == {"action": "IGNORE"}


def test_parse_garbage_returns_none():
    assert parse_consolidation_response("not json at all !!!") is None
    assert parse_consolidation_response("") is None


def test_parse_markdown_fenced_json():
    txt = "```json\n{\"action\":\"ADD\",\"target_id\":null,\"reason\":\"new\"}\n```"
    out = parse_consolidation_response(txt)
    assert out is not None
    assert out["action"] == "ADD"


# ======================================================================
# 2. curator_consolidate — the main entry point
# ======================================================================


def test_consolidate_returns_merge_with_target(force_curator, install_llm):
    llm = install_llm(FakeLLM(responses=[
        '{"action":"MERGE","target_id":"mem_xyz","reason":"semantic duplicate"}'
    ]))
    cand = {"type": "preference", "content": "prefers Python", "importance": 4}
    neighbors = [
        {"id": "mem_xyz", "type": "preference",
         "content": "user prefers Python over Go", "similarity": 0.93},
    ]
    out = curator_consolidate(cand, neighbors)
    assert out["action"] == "MERGE"
    assert out["target_id"] == "mem_xyz"
    assert out["source"] == "gemma"
    assert "semantic" in out["reason"]
    # Verify the prompt was actually shipped to the LLM.
    assert len(llm.completion_calls) == 1
    sent = llm.completion_calls[0]["prompt"]
    assert "prefers Python" in sent
    assert "mem_xyz" in sent


def test_consolidate_tolerates_bare_string(force_curator, install_llm):
    install_llm(FakeLLM(responses=['"merge"']))
    cand = {"type": "preference", "content": "prefers Python", "importance": 4}
    neighbors = [
        {"id": "mem_xyz", "type": "preference", "content": "prefers Python", "similarity": 0.95},
    ]
    out = curator_consolidate(cand, neighbors)
    # Bare "merge" gets mapped → MERGE, target inferred from neighbors[0].
    assert out["action"] == "MERGE"
    assert out["target_id"] == "mem_xyz"
    assert out["source"] == "gemma"


def test_consolidate_garbage_returns_parse_error(force_curator, install_llm):
    install_llm(FakeLLM(responses=["this is not JSON or any valid action"]))
    cand = {"type": "fact", "content": "irrelevant", "importance": 2}
    out = curator_consolidate(cand, [])
    assert out["action"] is None
    assert out["source"] == "gemma_parse_error"


def test_consolidate_when_gemma_unavailable(monkeypatch):
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)
    out = curator_consolidate({"type": "fact", "content": "x"}, [])
    assert out["action"] is None
    assert out["source"] == "gemma_unavailable"


def test_consolidate_explicit_llm_none_skips_load(monkeypatch):
    # If caller passes llm=None and gemma isn't installed, source == unavailable
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)
    out = curator_consolidate({"type": "fact", "content": "x"}, [], llm=None)
    assert out["action"] is None
    assert out["source"] == "gemma_unavailable"


def test_consolidate_truncates_huge_neighbors(force_curator, install_llm):
    """30 neighbors with long content should not blow context budget; the
    prompt sent must include some neighbors but not all 30 (truncated)."""
    llm = install_llm(FakeLLM(
        n_ctx=1024,  # tight ctx so truncation kicks in.
        responses=['{"action":"ADD","target_id":null,"reason":"new memory"}'],
    ))
    cand = {"type": "preference", "content": "prefers terse responses", "importance": 4}
    neighbors = [
        {
            "id": f"mem_{i:03d}",
            "type": "preference",
            "content": ("blah " * 40).strip() + f" item {i}",
            "similarity": 0.5 + i * 0.01,
        }
        for i in range(30)
    ]
    out = curator_consolidate(cand, neighbors)
    assert out["action"] == "ADD"
    sent = llm.completion_calls[0]["prompt"]
    # We expect either an explicit truncation marker OR fewer than all 30 ids.
    assert ("truncated" in sent) or sum(1 for i in range(30) if f"mem_{i:03d}" in sent) < 30


def test_consolidate_update_without_target_falls_back_to_first_neighbor(
    force_curator, install_llm,
):
    install_llm(FakeLLM(responses=['{"action":"UPDATE","target_id":null,"reason":"refines"}']))
    cand = {"type": "preference", "content": "prefers Python 3.13", "importance": 4}
    neighbors = [
        {"id": "mem_first", "type": "preference",
         "content": "prefers Python 3.12", "similarity": 0.91},
    ]
    out = curator_consolidate(cand, neighbors)
    assert out["action"] == "UPDATE"
    assert out["target_id"] == "mem_first"


def test_consolidate_update_no_target_no_neighbors_downgrades_to_add(
    force_curator, install_llm,
):
    install_llm(FakeLLM(responses=['{"action":"UPDATE","target_id":null,"reason":"x"}']))
    out = curator_consolidate({"type": "fact", "content": "anything"}, [])
    # No target + no neighbors → can't UPDATE, downgrade to ADD.
    assert out["action"] == "ADD"
    assert out["target_id"] is None


# ======================================================================
# 3. curator_detect_contradiction
# ======================================================================


def test_detect_contradiction_winner_a(force_curator, install_llm):
    install_llm(FakeLLM(responses=[
        '{"contradictory": true, "winner": "a", "reason": "A is later"}'
    ]))
    a = {"content": "user switched to Rust"}
    b = {"content": "user uses Python"}
    out = curator_detect_contradiction(a, b)
    assert out["contradictory"] is True
    assert out["winner"] == "a"
    assert out["source"] == "gemma"


def test_detect_contradiction_no_conflict(force_curator, install_llm):
    install_llm(FakeLLM(responses=[
        '{"contradictory": false, "winner": null, "reason": "different aspects"}'
    ]))
    out = curator_detect_contradiction({"content": "uses sqlite"}, {"content": "lives in BA"})
    assert out["contradictory"] is False
    assert out["winner"] is None


def test_detect_contradiction_unavailable(monkeypatch):
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)
    out = curator_detect_contradiction({"content": "a"}, {"content": "b"})
    assert out["contradictory"] is False
    assert out["source"] == "gemma_unavailable"


def test_detect_contradiction_garbage(force_curator, install_llm):
    install_llm(FakeLLM(responses=["definitely not JSON"]))
    out = curator_detect_contradiction({"content": "a"}, {"content": "b"})
    assert out["contradictory"] is False
    assert out["source"] == "gemma_parse_error"


# ======================================================================
# 4. curator_summarize_project
# ======================================================================


def test_summarize_project_trims_to_max_chars(force_curator, install_llm):
    long_text = "x" * 1500
    install_llm(FakeLLM(responses=[long_text]))
    memories = [
        {"id": "mem_a", "type": "decision", "content": "use sqlite-vec", "similarity": 0.9},
        {"id": "mem_b", "type": "task", "content": "add tests for X", "similarity": 0.8},
    ]
    out = curator_summarize_project("memoirs", memories, max_chars=500)
    assert isinstance(out, str)
    assert 0 < len(out) <= 510  # +trailing ellipsis tolerated
    assert out.endswith("…") or len(out) <= 500


def test_summarize_project_empty_memories():
    # No need for fake llm: empty memories -> empty string short-circuit.
    out = curator_summarize_project("memoirs", [])
    assert out == ""


def test_summarize_project_skip_response(force_curator, install_llm):
    install_llm(FakeLLM(responses=["SKIP"]))
    memories = [{"id": "x", "type": "fact", "content": "trivial", "similarity": 0.5}]
    out = curator_summarize_project("p", memories)
    assert out == ""


# ======================================================================
# 5. decide_memory_action — env-flag integration
# ======================================================================


def _add_memory(db, content, mtype="preference"):
    """Insert a memoria the cheap way for testing duplicate detection."""
    from memoirs.db import content_hash, stable_id, utc_now
    mid = stable_id("mem", mtype, content)
    h = content_hash(content)
    now = utc_now()
    db.conn.execute(
        """
        INSERT INTO memories (id, type, content, content_hash, importance, confidence,
                              score, usage_count, user_signal, valid_from, metadata_json,
                              created_at, updated_at)
        VALUES (?, ?, ?, ?, 4, 0.8, 0, 0, 0, ?, '{}', ?, ?)
        """,
        (mid, mtype, content, h, now, now, now),
    )
    db.conn.commit()
    return mid


def test_decide_off_uses_heuristic(tmp_db, monkeypatch):
    """With curator=off, Gemma should not be consulted at all."""
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "off")
    # Spy on curator_consolidate to ensure it's NEVER called.
    calls: list = []

    def boom(*a, **kw):  # pragma: no cover — should not be reached
        calls.append((a, kw))
        raise AssertionError("curator_consolidate must not be called when curator=off")

    monkeypatch.setattr(curator_mod, "curator_consolidate", boom)

    cand = _candidate("brand new memory under off mode")
    decision = memory_engine.decide_memory_action(tmp_db, cand)
    assert decision.action in {"ADD", "UPDATE", "MERGE", "CONTRADICTION", "IGNORE"}
    assert calls == []


def test_decide_auto_uses_gemma_when_valid(tmp_db, monkeypatch):
    """auto mode + valid Gemma response → action sourced from Gemma."""
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "auto")

    def fake_consolidate(candidate, neighbors, *, llm=None):
        return {
            "action": "IGNORE",
            "target_id": None,
            "reason": "low value",
            "source": "gemma",
        }

    monkeypatch.setattr(curator_mod, "curator_consolidate", fake_consolidate)

    cand = _candidate("another memory under auto mode")
    decision = memory_engine.decide_memory_action(tmp_db, cand)
    assert decision.action == "IGNORE"
    assert "gemma" in decision.reason


def test_decide_falls_back_when_gemma_returns_none(tmp_db, monkeypatch):
    """When Gemma replies with action=None (parse error / unavailable), the
    heuristic must take over and produce a valid action."""
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "auto")

    def fake_consolidate(candidate, neighbors, *, llm=None):
        return {"action": None, "source": "gemma_parse_error", "reason": "garbage"}

    monkeypatch.setattr(curator_mod, "curator_consolidate", fake_consolidate)

    cand = _candidate("yet another memory for fallback")
    decision = memory_engine.decide_memory_action(tmp_db, cand)
    # No existing memory in DB → heuristic returns ADD.
    assert decision.action == "ADD"


def test_decide_exact_dup_short_circuits_before_gemma(tmp_db, monkeypatch):
    """Exact-content match should not even reach Gemma — saves a model call."""
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "auto")
    existing_id = _add_memory(tmp_db, "exact identical content")
    boom: list = []

    def spy(*a, **kw):
        boom.append((a, kw))
        return {"action": "ADD", "source": "gemma"}

    monkeypatch.setattr(curator_mod, "curator_consolidate", spy)

    cand = _candidate("exact identical content")
    decision = memory_engine.decide_memory_action(tmp_db, cand)
    assert decision.action == "UPDATE"
    assert decision.target_memory_id == existing_id
    assert boom == []  # Gemma was bypassed


def test_decide_on_mode_falls_back_when_gemma_fails(tmp_db, monkeypatch, caplog):
    """`on` still falls back to heuristic when Gemma fails — we never block
    consolidation. But a warning is logged."""
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "on")

    def fake_consolidate(candidate, neighbors, *, llm=None):
        return {"action": None, "source": "gemma_unavailable", "reason": "no model"}

    monkeypatch.setattr(curator_mod, "curator_consolidate", fake_consolidate)

    cand = _candidate("on-mode memory should still go through")
    with caplog.at_level("WARNING", logger="memoirs.engine"):
        decision = memory_engine.decide_memory_action(tmp_db, cand)
    assert decision.action == "ADD"
    # Make sure we logged the fallback (so operator can debug).
    assert any("MEMOIRS_CURATOR_ENABLED=on" in r.message for r in caplog.records)
