"""Fase 5A — wiring tests for the ACL hot path.

These tests verify that ``engine.acl`` is now actually consulted by:

* :func:`memory_engine._retrieve_candidates` (read side — used by
  ``mcp_get_context`` / ``assemble_context`` / ``assemble_context_stream``).
* :func:`memory_engine.apply_decision` (write side — used by every
  consolidation action that touches ``target_memory_id``).

Plus the new CLI subcommands (``scope``, ``share``, ``unshare``).

The previous ``engine/acl.py`` already had the predicates implemented; the
gap was that nobody was calling them, so a multi-tenant install would still
leak ``user_id="bob"`` private memories to a query running under
``user_id="alice"``. Each test below codifies one slice of that contract.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pytest

from memoirs.engine import acl as _acl
from memoirs.engine import memory_engine as me
from memoirs.engine.gemma import Candidate
from memoirs.models import Scope, ScopeFilter


NOW = "2026-04-27T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_memory(
    db,
    memory_id: str,
    content: str,
    *,
    user_id: str = "local",
    visibility: str = "private",
    mem_type: str = "fact",
) -> None:
    """Insert a memory with explicit scope columns (post-008 schema)."""
    db.conn.execute(
        "INSERT INTO memories ("
        "  id, type, content, content_hash, importance, confidence,"
        "  score, usage_count, user_signal, valid_from, metadata_json,"
        "  user_id, agent_id, run_id, namespace, visibility,"
        "  created_at, updated_at"
        ") VALUES (?, ?, ?, 'h_'||?, 3, 0.7, 0.6, 0, 0, ?, '{}',"
        "          ?, NULL, NULL, NULL, ?, ?, ?)",
        (memory_id, mem_type, content, memory_id, NOW,
         user_id, visibility, NOW, NOW),
    )
    # FTS5 mirror so BM25 retrieval can find the row.
    try:
        db.conn.execute(
            "INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)",
            (memory_id, content),
        )
    except Exception:
        pass
    db.conn.commit()


def _candidates_for_user(db, query: str, user_id: str) -> list[dict]:
    """Drive ``_retrieve_candidates`` with a Scope-style requester. Mirrors
    what ``assemble_context_stream`` does internally.
    """
    return me._retrieve_candidates(
        db,
        query,
        top_k=20,
        as_of=None,
        mode="bm25",  # deterministic, no embeddings needed
        scope_filter=None,
        scope=Scope(user_id=user_id),
    )


# ---------------------------------------------------------------------------
# Read-side ACL — _retrieve_candidates / assemble_context
# ---------------------------------------------------------------------------


def test_alice_sees_only_her_private_plus_bob_public(tmp_db):
    _seed_memory(tmp_db, "m_alice_priv", "alice loves typescript",
                 user_id="alice", visibility="private")
    _seed_memory(tmp_db, "m_alice_pub", "alice prefers tabs",
                 user_id="alice", visibility="public")
    _seed_memory(tmp_db, "m_bob_priv", "bob hates typescript",
                 user_id="bob", visibility="private")
    _seed_memory(tmp_db, "m_bob_pub", "bob prefers spaces",
                 user_id="bob", visibility="public")

    rows = _candidates_for_user(tmp_db, "typescript prefers tabs spaces", "alice")
    ids = {r["id"] for r in rows}
    assert "m_bob_priv" not in ids, "private bob memory leaked to alice"
    # alice's own + bob's public are visible
    assert "m_alice_priv" in ids
    assert {"m_alice_pub", "m_bob_pub"} <= ids


def test_share_grants_read_then_unshare_revokes(tmp_db):
    _seed_memory(tmp_db, "m_bob_secret", "bob's private launch plans",
                 user_id="bob", visibility="private")

    # Pre-share: alice cannot see it.
    rows = _candidates_for_user(tmp_db, "launch plans", "alice")
    assert all(r["id"] != "m_bob_secret" for r in rows)

    # Share → alice now sees it.
    res = _acl.share_memory(tmp_db, "m_bob_secret", "alice")
    assert res["ok"] is True and res["shared"] is True
    # acl.share_memory does NOT change visibility — but for a 'shared'
    # visibility row, alice would be allowed via the share table. The seed
    # above is 'private', so flip it to 'shared' which is the supported
    # visibility for the share table to apply.
    tmp_db.conn.execute(
        "UPDATE memories SET visibility = 'shared' WHERE id = ?",
        ("m_bob_secret",),
    )
    tmp_db.conn.commit()
    rows = _candidates_for_user(tmp_db, "launch plans", "alice")
    assert any(r["id"] == "m_bob_secret" for r in rows)

    # Unshare → revoked.
    res = _acl.unshare_memory(tmp_db, "m_bob_secret", "alice")
    assert res["ok"] is True and res["removed"] == 1
    rows = _candidates_for_user(tmp_db, "launch plans", "alice")
    assert all(r["id"] != "m_bob_secret" for r in rows)


def test_visibility_public_is_always_readable(tmp_db):
    _seed_memory(tmp_db, "m_pub", "public service announcement",
                 user_id="bob", visibility="public")
    rows = _candidates_for_user(tmp_db, "public service announcement", "alice")
    assert any(r["id"] == "m_pub" for r in rows)
    rows = _candidates_for_user(tmp_db, "public service announcement", "carol")
    assert any(r["id"] == "m_pub" for r in rows)


def test_visibility_org_readable_across_users(tmp_db):
    # Placeholder semantics per acl.py: visibility=org is readable by any
    # user_id (future iterations may verify org membership).
    _seed_memory(tmp_db, "m_org", "org-wide standup notes",
                 user_id="bob", visibility="org")
    rows = _candidates_for_user(tmp_db, "org-wide standup notes", "alice")
    assert any(r["id"] == "m_org" for r in rows)


# ---------------------------------------------------------------------------
# Write-side ACL — apply_decision
# ---------------------------------------------------------------------------


def test_apply_decision_update_rejected_for_non_owner(tmp_db):
    _seed_memory(tmp_db, "m_bob_pref", "bob likes vim",
                 user_id="bob", visibility="private")
    cand = Candidate(
        type="preference", content="bob likes vim (reconfirmed)",
        importance=3, confidence=0.8, entities=[], source_message_ids=[],
    )
    decision = me.Decision(
        action="UPDATE", target_memory_id="m_bob_pref",
        reason="reconfirmation",
    )
    result = me.apply_decision(
        tmp_db, cand, decision, scope=Scope(user_id="alice"),
    )
    assert result["action"] == "REJECTED"
    assert "ACL" in result["reason"]
    # Ensure no mutation happened: confidence still 0.7
    row = tmp_db.conn.execute(
        "SELECT confidence FROM memories WHERE id = ?", ("m_bob_pref",),
    ).fetchone()
    assert abs(float(row["confidence"]) - 0.7) < 1e-6


def test_apply_decision_update_allowed_for_owner(tmp_db):
    _seed_memory(tmp_db, "m_alice_pref", "alice likes vim",
                 user_id="alice", visibility="private")
    cand = Candidate(
        type="preference", content="alice likes vim (reconfirmed)",
        importance=4, confidence=0.9, entities=[], source_message_ids=[],
    )
    decision = me.Decision(
        action="UPDATE", target_memory_id="m_alice_pref",
        reason="reconfirmation",
    )
    result = me.apply_decision(
        tmp_db, cand, decision, scope=Scope(user_id="alice"),
    )
    assert result["action"] == "UPDATE"
    assert result.get("memory_id") == "m_alice_pref"
    row = tmp_db.conn.execute(
        "SELECT confidence, importance FROM memories WHERE id = ?",
        ("m_alice_pref",),
    ).fetchone()
    assert float(row["confidence"]) > 0.7  # bumped
    assert int(row["importance"]) >= 4


def test_apply_decision_expire_rejected_for_non_owner(tmp_db):
    _seed_memory(tmp_db, "m_bob_old", "bob's stale fact",
                 user_id="bob", visibility="private")
    cand = Candidate(
        type="fact", content="bob's fresh fact",
        importance=3, confidence=0.8, entities=[], source_message_ids=[],
    )
    decision = me.Decision(
        action="EXPIRE", target_memory_id="m_bob_old",
        reason="superseded",
    )
    result = me.apply_decision(
        tmp_db, cand, decision, scope=Scope(user_id="alice"),
    )
    assert result["action"] == "REJECTED"
    row = tmp_db.conn.execute(
        "SELECT valid_to FROM memories WHERE id = ?", ("m_bob_old",),
    ).fetchone()
    assert row["valid_to"] is None  # untouched


def test_apply_decision_archive_rejected_for_non_owner(tmp_db):
    _seed_memory(tmp_db, "m_bob_stale", "bob's legacy note",
                 user_id="bob", visibility="private")
    cand = Candidate(
        type="fact", content="(archive trigger)",
        importance=1, confidence=0.5, entities=[], source_message_ids=[],
    )
    decision = me.Decision(
        action="ARCHIVE", target_memory_id="m_bob_stale",
        reason="too stale",
    )
    result = me.apply_decision(
        tmp_db, cand, decision, scope=Scope(user_id="alice"),
    )
    assert result["action"] == "REJECTED"
    row = tmp_db.conn.execute(
        "SELECT archived_at FROM memories WHERE id = ?", ("m_bob_stale",),
    ).fetchone()
    assert row["archived_at"] is None


# ---------------------------------------------------------------------------
# Fast-path: no overhead in single-user mode
# ---------------------------------------------------------------------------


def test_fast_path_skips_acl_when_all_local(tmp_db, monkeypatch):
    """When every memory is owned by ``"local"`` (the legacy single-user
    default), the ACL layer must not import :mod:`engine.acl` nor perform
    per-row predicates. We prove this by patching ``acl.can_read`` to raise
    — if the fast-path engages, the test passes; otherwise it explodes.
    """
    for i in range(20):
        _seed_memory(tmp_db, f"m_local_{i}", f"local memory number {i}",
                     user_id="local", visibility="private")

    def _boom(*a, **kw):
        raise AssertionError("can_read should not run on the local fast-path")
    monkeypatch.setattr(_acl, "can_read", _boom)

    rows = _candidates_for_user(tmp_db, "local memory number", "local")
    assert len(rows) > 0


def test_microbenchmark_fast_path_overhead_reasonable(tmp_db):
    """Single-user retrieval should not regress measurably. We compare a
    cohort of ``"local"`` rows (fast-path engaged) against the multi-tenant
    path (one bob row mixed in). Threshold is generous (5x) to avoid
    flakiness on noisy CI; the goal here is to detect order-of-magnitude
    regressions, not micro-optimization.
    """
    for i in range(50):
        _seed_memory(tmp_db, f"m_local_{i}", f"benchmark word {i}",
                     user_id="local", visibility="private")

    # Warm up.
    me._retrieve_candidates(tmp_db, "benchmark word", top_k=20, as_of=None,
                            mode="bm25", scope_filter=None,
                            scope=Scope(user_id="local"))

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        me._retrieve_candidates(tmp_db, "benchmark word", top_k=20, as_of=None,
                                mode="bm25", scope_filter=None,
                                scope=Scope(user_id="local"))
    fast_ns = (time.perf_counter() - t0)

    # Now mix in a non-local row → forces per-row can_read.
    _seed_memory(tmp_db, "m_bob_mix", "benchmark word from bob",
                 user_id="bob", visibility="private")

    t0 = time.perf_counter()
    for _ in range(iters):
        me._retrieve_candidates(tmp_db, "benchmark word", top_k=20, as_of=None,
                                mode="bm25", scope_filter=None,
                                scope=Scope(user_id="local"))
    slow_ns = (time.perf_counter() - t0)

    # Fast path must not be dramatically slower than the slow path. We
    # assert fast_ns is in the same order of magnitude (and ideally lower).
    # 5x slack absorbs CI jitter; failure here would mean the fast-path
    # actually has *more* overhead than the full ACL pass — a real bug.
    assert fast_ns <= slow_ns * 5.0, (
        f"single-user fast-path slower than ACL path: "
        f"fast={fast_ns:.4f}s slow={slow_ns:.4f}s"
    )


# ---------------------------------------------------------------------------
# CLI surface — scope / share / unshare
# ---------------------------------------------------------------------------


def test_cli_scope_show_default(tmp_path, monkeypatch):
    """``memoirs scope show`` reports ``user_id='local'`` by default."""
    from memoirs import cli as _cli
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(_cli, "_SCOPE_CONFIG_PATH", fake_home / ".memoirs" / "scope.json")
    monkeypatch.delenv("MEMOIRS_USER_ID", raising=False)
    monkeypatch.delenv("MEMOIRS_AGENT_ID", raising=False)
    monkeypatch.delenv("MEMOIRS_NAMESPACE", raising=False)
    monkeypatch.delenv("MEMOIRS_VISIBILITY", raising=False)
    monkeypatch.delenv("MEMOIRS_RUN_ID", raising=False)

    eff = _cli._effective_scope_dict()
    assert eff["user_id"] == "local"
    assert eff["visibility"] == "private"


def test_cli_scope_set_then_show_roundtrip(tmp_path, monkeypatch, capsys):
    from memoirs import cli as _cli
    cfg_path = tmp_path / ".memoirs" / "scope.json"
    monkeypatch.setattr(_cli, "_SCOPE_CONFIG_PATH", cfg_path)
    monkeypatch.delenv("MEMOIRS_USER_ID", raising=False)
    monkeypatch.delenv("MEMOIRS_AGENT_ID", raising=False)
    monkeypatch.delenv("MEMOIRS_NAMESPACE", raising=False)
    monkeypatch.delenv("MEMOIRS_VISIBILITY", raising=False)

    class _Args:
        scope_cmd = "set"
        user_id = "alice"
        agent_id = "claude-code"
        namespace = "work"
        visibility = "shared"

    rc = _cli._cmd_scope(_Args())
    assert rc == 0
    assert cfg_path.exists()
    eff = _cli._effective_scope_dict()
    assert eff["user_id"] == "alice"
    assert eff["agent_id"] == "claude-code"
    assert eff["namespace"] == "work"
    assert eff["visibility"] == "shared"

    # Env beats config.
    monkeypatch.setenv("MEMOIRS_USER_ID", "bob")
    eff = _cli._effective_scope_dict()
    assert eff["user_id"] == "bob"


def test_cli_share_unshare_roundtrip(tmp_db, capsys):
    from memoirs import cli as _cli
    _seed_memory(tmp_db, "m_share_target", "shared content",
                 user_id="bob", visibility="shared")

    class _Args:
        memory_id = "m_share_targ"  # prefix
        target_user = "alice"

    # share via CLI helper
    rc = _cli._cmd_share(tmp_db, _Args())
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["shared"] is True
    assert payload["shared_with_user_id"] == "alice"

    # alice can read via _retrieve_candidates now
    rows = _candidates_for_user(tmp_db, "shared content", "alice")
    assert any(r["id"] == "m_share_target" for r in rows)

    # unshare
    rc = _cli._cmd_unshare(tmp_db, _Args())
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["removed"] == 1

    rows = _candidates_for_user(tmp_db, "shared content", "alice")
    assert all(r["id"] != "m_share_target" for r in rows)
