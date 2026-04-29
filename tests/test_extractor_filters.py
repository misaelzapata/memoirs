"""Tests for Fix #1 of the GAP audit Fase 5C.

Covers:
  - ``core.normalize.should_skip_extraction`` filter rules.
  - ``core.normalize.canonicalize_for_dedup`` invariance.
  - Integration: ``gemma._candidates_from_text`` round-trip + the
    ``gemma_extract`` dedup loop hook drops noise but keeps signal.
  - ``memory_engine.detect_exact_duplicate`` cross-type, canonical match.
  - ``memory_engine.detect_semantic_duplicate`` threshold lowered to 0.85.
  - ``memory_engine.decide_memory_action`` issues REJECT for noise.
  - ``memory_engine.consolidate_candidate`` persists ``status='rejected'`` +
    ``rejection_reason`` for REJECT decisions.
  - Cross-type EXACT-dup MERGE promotes the type to whichever side has the
    higher importance.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from memoirs.core.normalize import canonicalize_for_dedup, should_skip_extraction
from memoirs.core.ids import content_hash, utc_now
from memoirs.db import MemoirsDB
from memoirs.engine import curator as curator_mod
from memoirs.engine import memory_engine as me
from memoirs.engine.gemma import Candidate


# ---------------------------------------------------------------------------
# should_skip_extraction — table-driven coverage of every rule
# ---------------------------------------------------------------------------


def test_skip_code_snippet_multiline():
    skip, reason = should_skip_extraction("def foo():\n    return 1")
    assert skip is True
    assert reason == "code snippet"


def test_skip_path_prefix_tmp():
    # Padded so the length filter doesn't trip first; the rule under test
    # is the path prefix.
    skip, reason = should_skip_extraction("/tmp/foo/bar/baz/output.log")
    assert skip is True
    assert reason == "path"


def test_skip_path_prefix_home():
    skip, reason = should_skip_extraction("/home/misael/projects/memoirs/CLAUDE.md")
    assert skip is True
    assert reason == "path"


def test_skip_url_prefix():
    skip, reason = should_skip_extraction("https://example.com/foo/bar/baz")
    assert skip is True
    assert reason == "path"


def test_skip_stack_trace_python():
    skip, reason = should_skip_extraction(
        'Traceback (most recent call last):\nFile "foo.py", line 5'
    )
    assert skip is True
    # Tool-output prefix wins because the head matches "Traceback (most recent".
    assert reason in {"tool output", "stack trace"}


def test_skip_stack_trace_inline_marker():
    skip, reason = should_skip_extraction(
        'somewhere File "/x/y.py", line 42 we crashed'
    )
    assert skip is True
    assert reason == "stack trace"


def test_skip_tool_output_marker():
    skip, reason = should_skip_extraction("[tool_use:bash] ls -la /tmp")
    assert skip is True
    assert reason == "tool output"


def test_skip_tool_result_marker():
    skip, reason = should_skip_extraction("[tool_result] some output content")
    assert skip is True
    assert reason == "tool output"


def test_skip_too_short():
    skip, reason = should_skip_extraction("hi")
    assert skip is True
    assert reason == "too short"


def test_skip_too_long():
    skip, reason = should_skip_extraction("a" * 3000)
    assert skip is True
    assert reason == "too long; needs summarization first"


def test_skip_hex_dump():
    skip, reason = should_skip_extraction(
        "blob: deadbeef" + "0123456789abcdef" * 3
    )
    assert skip is True
    assert reason == "hex dump"


def test_keep_durable_user_preference():
    skip, reason = should_skip_extraction("user prefers dark mode in IDE")
    assert skip is False
    assert reason == ""


def test_keep_short_but_durable_preferences():
    # Regression: the old 20-char threshold rejected these legit short prefs.
    for content in (
        "user prefers Python",   # 19 chars
        "I am Misael",            # 11 chars
        "use snake_case",         # 14 chars
        "Python > JS",            # 11 chars
        "I love coffee",          # 13 chars
    ):
        skip, reason = should_skip_extraction(content)
        assert skip is False, f"falsely skipped {content!r} ({reason})"


def test_skip_single_token_noise():
    # Single tokens / courtesy turns should still be rejected.
    for content in ("hi", "yes", "okay", "thanks", "no"):
        skip, reason = should_skip_extraction(content)
        assert skip is True, f"failed to skip noise {content!r}"
        assert reason == "too short"


def test_keep_decision_with_path_substring():
    # A path inside a meaningful sentence must NOT be flagged.
    skip, reason = should_skip_extraction(
        "decided to relocate the database from /tmp to a persistent volume"
    )
    assert skip is False, f"unexpected skip ({reason}) for benign sentence"


# ---------------------------------------------------------------------------
# canonicalize_for_dedup — invariance properties
# ---------------------------------------------------------------------------


def test_canonical_case_and_whitespace_invariant():
    assert canonicalize_for_dedup("HELLO WORLD") == canonicalize_for_dedup(" hello world. ")


def test_canonical_url_dropped():
    a = "loves the docs at https://example.com/foo for context"
    b = "loves the docs at  for context"
    assert canonicalize_for_dedup(a) == canonicalize_for_dedup(b)


def test_canonical_punctuation_strip():
    assert (
        canonicalize_for_dedup("user wants tests!!!")
        == canonicalize_for_dedup("USER WANTS TESTS")
    )


# ---------------------------------------------------------------------------
# Integration: gemma_extract drop-on-noise via _candidates_from_text
# ---------------------------------------------------------------------------


def test_candidates_from_text_pure_signal_kept():
    """The parser stage doesn't drop noise — that happens in the
    `gemma_extract` dedup loop. This test pins down the parsing contract.
    """
    payload = json.dumps([
        {"type": "preference", "content": "user prefers Python over Go",
         "importance": 4, "confidence": 0.9},
    ])
    cands = curator_mod._candidates_from_text(payload)
    assert len(cands) == 1
    assert cands[0].type == "preference"


def test_extract_loop_drops_noise_and_keeps_signal(monkeypatch):
    """`gemma_extract` must drop code/path/tool-output candidates emitted
    by the model while retaining durable signal. We stub the LLM so the
    test is deterministic.
    """
    fake_payload = json.dumps([
        {"type": "preference", "content": "user prefers concise responses",
         "importance": 4, "confidence": 0.9},
        {"type": "fact", "content": "def foo():\n    return 1",
         "importance": 2, "confidence": 0.9},
        {"type": "fact", "content": "/tmp/claude-1000/something/path/here",
         "importance": 2, "confidence": 0.7},
        {"type": "fact", "content": "[tool_use:bash] ls",
         "importance": 1, "confidence": 0.5},
    ])

    class _FakeLLM:
        # Minimal surface gemma_extract uses.
        def tokenize(self, data, add_bos=False, special=False):
            # 1 token per byte is a fine approximation for a deterministic stub.
            return list(data)

        def detokenize(self, toks):
            return bytes(toks)

        def n_ctx(self):
            return 32_000

        def create_completion(self, *, prompt, max_tokens, temperature, stop):
            return {"choices": [{"text": fake_payload}]}

    monkeypatch.setattr(curator_mod, "_get_llm", lambda: _FakeLLM())
    # Skip the post-wrap re-tokenize bisection by ensuring n_ctx is huge.
    msgs = [{"role": "user", "content": "real durable user message about prefs"}]
    cands = curator_mod.curator_extract(msgs)
    contents = {c.content for c in cands}
    assert "user prefers concise responses" in contents
    # All three noise candidates must be dropped.
    assert "def foo():\n    return 1" not in contents
    assert "/tmp/claude-1000/something/path/here" not in contents
    assert "[tool_use:bash] ls" not in contents


# ---------------------------------------------------------------------------
# Cross-type exact-dup MERGE promotes type to higher-importance side
# ---------------------------------------------------------------------------


def _seed_memory(db: MemoirsDB, *, mid: str, mtype: str, content: str,
                 importance: int) -> None:
    h = content_hash(content)
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, "
        "metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (mid, mtype, content, h, importance, utc_now(), utc_now(), utc_now()),
    )
    db.conn.commit()


def test_detect_exact_duplicate_canonical_match(tmp_db):
    _seed_memory(tmp_db, mid="m_canon", mtype="fact",
                 content="user wants tests", importance=3)
    hit = me.detect_exact_duplicate(tmp_db, "  USER wants tests!!!  ")
    assert hit is not None
    assert hit["id"] == "m_canon"


def test_cross_type_exact_dup_merges_at_higher_importance(tmp_db, monkeypatch):
    # Disable network/embedding side-effects.
    monkeypatch.setattr(me, "_maybe_link_memory", lambda *a, **k: None)
    _seed_memory(tmp_db, mid="m_target", mtype="task",
                 content="ship the curator quality fix", importance=2)

    cand = Candidate(
        type="fact",
        content="ship the curator quality fix",
        importance=5,
        confidence=0.9,
    )
    decision = me.decide_memory_action(tmp_db, cand)
    assert decision.action == "MERGE"
    assert decision.target_memory_id == "m_target"

    me.apply_decision(tmp_db, cand, decision)

    row = tmp_db.conn.execute(
        "SELECT type, importance FROM memories WHERE id = 'm_target'"
    ).fetchone()
    # Candidate had importance=5 > 2, so type promotes to 'fact'.
    assert row["type"] == "fact"
    assert row["importance"] == 5


# ---------------------------------------------------------------------------
# Threshold 0.85 — the new default
# ---------------------------------------------------------------------------


def test_semantic_dup_default_threshold_is_0_85(monkeypatch):
    captured: dict = {}

    def fake_find(_db, _content, *, threshold):
        captured["threshold"] = threshold
        return [{"id": "m_sim", "type": "fact",
                 "content": "x", "similarity": 0.86}]

    monkeypatch.setattr(me.emb, "find_semantic_duplicates", fake_find)
    hit = me.detect_semantic_duplicate(None, "anything")  # type: ignore[arg-type]
    assert captured["threshold"] == 0.85
    assert hit is not None
    assert hit["similarity"] == 0.86


def test_semantic_dup_threshold_above_old_threshold(monkeypatch):
    """A neighbor at sim=0.86 (which the old 0.92 default would have kept
    as a separate memoria) is now flagged as duplicate.
    """
    monkeypatch.setattr(
        me.emb,
        "find_semantic_duplicates",
        lambda _db, _content, *, threshold: (
            [{"id": "m_x", "type": "fact", "content": "y", "similarity": 0.86}]
            if threshold <= 0.86 else []
        ),
    )
    hit = me.detect_semantic_duplicate(None, "anything")  # type: ignore[arg-type]
    assert hit is not None and hit["id"] == "m_x"


# ---------------------------------------------------------------------------
# REJECT path — defense in depth + persistence
# ---------------------------------------------------------------------------


def test_decide_action_rejects_noise(tmp_db):
    cand = Candidate(
        type="fact",
        content="def foo():\n    return 1",
        importance=3,
        confidence=0.5,
    )
    decision = me.decide_memory_action(tmp_db, cand)
    assert decision.action == "REJECT"
    assert "noise" in decision.reason.lower()


def test_apply_decision_reject_does_not_create_memory(tmp_db):
    cand = Candidate(
        type="fact",
        content="def foo():\n    return 1",
        importance=3,
        confidence=0.5,
    )
    decision = me.Decision("REJECT", reason="noise: code snippet")
    result = me.apply_decision(tmp_db, cand, decision)
    assert result["action"] == "REJECT"
    assert result["rejection_reason"] == "noise: code snippet"
    rows = tmp_db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    assert rows[0] == 0


def test_consolidate_candidate_persists_rejection(tmp_db):
    """A pending memory_candidates row carrying obvious noise must be
    promoted to status='rejected' with the reason recorded.
    """
    cid = "cand_noise_001"
    tmp_db.conn.execute(
        "INSERT INTO memory_candidates (id, conversation_id, source_message_ids, "
        "type, content, importance, confidence, entities, status, extractor, "
        "raw_json, created_at, updated_at) "
        "VALUES (?, NULL, '[]', 'fact', ?, 2, 0.7, '[]', 'pending', "
        "'gemma-2-2b', '{}', ?, ?)",
        (cid, "def foo():\n    return 1", utc_now(), utc_now()),
    )
    tmp_db.conn.commit()
    row = tmp_db.conn.execute(
        "SELECT * FROM memory_candidates WHERE id = ?", (cid,)
    ).fetchone()
    me.consolidate_candidate(tmp_db, dict(row))

    after = tmp_db.conn.execute(
        "SELECT status, rejection_reason, promoted_memory_id "
        "FROM memory_candidates WHERE id = ?",
        (cid,),
    ).fetchone()
    assert after["status"] == "rejected"
    assert "noise" in (after["rejection_reason"] or "").lower()
    assert after["promoted_memory_id"] is None


# ---------------------------------------------------------------------------
# Audit-style report: how many existing memorias would be rejected if we
# replayed the new filter? This is read-only; it does NOT mutate the corpus.
# ---------------------------------------------------------------------------


def test_filter_replay_on_synthetic_corpus_drops_known_bad_rows(tmp_db):
    """Pin down the contract: replaying ``should_skip_extraction`` over a
    synthetic corpus modeled after the audit's bottom-20 (paths, code,
    tool output) flags every bad row and keeps every durable one.
    """
    seeds = [
        ("good_pref", "user prefers dark mode in IDE", False),
        ("good_decision", "decided to switch from Postgres to SQLite for memoirs", False),
        ("bad_path", "/tmp/claude-1000/conversation_log.jsonl", True),
        ("bad_code", "def launch_goroutine():\n    return go()", True),
        ("bad_tool", "[tool_use:bash] ls /tmp", True),
        ("bad_short", "ok", True),
    ]
    for mid, content, _ in seeds:
        _seed_memory(tmp_db, mid=mid, mtype="fact", content=content, importance=3)

    rejected = 0
    kept = 0
    for mid, content, expected_skip in seeds:
        skip, _ = should_skip_extraction(content)
        if skip:
            rejected += 1
        else:
            kept += 1
        assert skip is expected_skip, f"{mid!r} skip={skip} expected={expected_skip}"

    assert rejected == 4
    assert kept == 2
