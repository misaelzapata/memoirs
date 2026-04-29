"""Targeted coverage tests for memoirs/mcp/tools.py.

The dispatcher (``call_tool``) and each ``_h_*`` handler are exercised here.
Heavyweight handlers that depend on Gemma / spaCy are tested with the
upstream module monkeypatched so we exercise the dispatch glue without paying
the model-load cost.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb
from memoirs.mcp import tools as mt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(angle_rad: float) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _seed(db, *, memory_id: str, mem_type: str = "fact", content: str = "x",
          score: float = 0.5, importance: int = 3, angle_rad: float | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, 'h_'||?, ?, 0.5, ?, 0, 0, ?, '{}', ?, ?)",
        (memory_id, mem_type, content, memory_id, importance, score, now, now, now),
    )
    if angle_rad is not None:
        emb._require_vec(db)
        blob = emb._pack(_unit_vec(angle_rad))
        db.conn.execute(
            "INSERT INTO memory_embeddings (memory_id, dim, embedding, model, created_at) "
            "VALUES (?, ?, ?, 'test', ?)",
            (memory_id, EMBEDDING_DIM, blob, now),
        )
        db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
        db.conn.execute(
            "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
            (memory_id, blob),
        )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_call_tool_unknown_raises(tmp_db):
    with pytest.raises(ValueError, match="unknown tool"):
        mt.call_tool(tmp_db, "no_such_tool", {})


def test_tool_schemas_have_required_fields():
    """Every schema must expose name + inputSchema."""
    assert len(mt.TOOL_SCHEMAS) > 0
    for s in mt.TOOL_SCHEMAS:
        assert "name" in s
        assert "inputSchema" in s


# ---------------------------------------------------------------------------
# Layer 1 — ingest + status
# ---------------------------------------------------------------------------


def test_h_ingest_event_via_call_tool(tmp_db):
    out = mt.call_tool(tmp_db, "mcp_ingest_event", {
        "type": "chat_message",
        "source": "test",
        "conversation_id": "c1",
        "message_id": "m1",
        "role": "user",
        "content": "hello",
    })
    assert out["ok"] is True
    assert out["conversation_id"]
    assert out["message_id"]


def test_h_ingest_event_unwraps_event_kwarg(tmp_db):
    """If args contain an `event` dict, it gets unwrapped before ingest."""
    out = mt.call_tool(tmp_db, "mcp_ingest_event", {"event": {
        "type": "chat_message",
        "source": "wrapped",
        "conversation_id": "c2",
        "message_id": "m2",
        "role": "user",
        "content": "wrapped event",
    }})
    assert out["ok"] is True


def test_h_ingest_conversation(tmp_db):
    out = mt.call_tool(tmp_db, "mcp_ingest_conversation", {
        "source": "test",
        "conversation_id": "cv1",
        "messages": [
            {"role": "user", "content": "a", "message_id": "u1"},
            {"role": "assistant", "content": "b", "message_id": "a1"},
        ],
    })
    assert out["ok"] is True
    assert out["message_count"] == 2


def test_h_status_returns_counts(tmp_db):
    out = mt.call_tool(tmp_db, "mcp_status", {})
    for k in ("sources", "conversations", "active_messages", "messages_total"):
        assert k in out


# ---------------------------------------------------------------------------
# Layer 2/5 — extract / consolidate / audit / maintenance
#   (mocked at the module boundary so we don't load Gemma)
# ---------------------------------------------------------------------------


def test_h_extract_pending_dispatches(monkeypatch, tmp_db):
    from memoirs.engine import curator as _curator
    called = {}

    def fake_extract(db, *, limit, reprocess_with_gemma):
        called["args"] = (limit, reprocess_with_gemma)
        return {"processed": 0}

    monkeypatch.setattr(_curator, "extract_pending", fake_extract)
    out = mt.call_tool(tmp_db, "mcp_extract_pending", {"limit": 7, "reprocess_with_gemma": True})
    assert out == {"processed": 0}
    assert called["args"] == (7, True)


def test_h_consolidate_pending_dispatches(monkeypatch, tmp_db):
    from memoirs.engine import memory_engine
    monkeypatch.setattr(memory_engine, "consolidate_pending",
                        lambda db, *, limit: {"processed": 0, "by_action": {}, "limit": limit})
    out = mt.call_tool(tmp_db, "mcp_consolidate_pending", {"limit": 25})
    assert out["limit"] == 25


def test_h_audit_corpus_dispatches(monkeypatch, tmp_db):
    import memoirs.engine.audit as audit_mod
    monkeypatch.setattr(audit_mod, "audit_corpus",
                        lambda db, *, limit, type_filter, apply: {"limit": limit, "applied": apply})
    out = mt.call_tool(tmp_db, "mcp_audit_corpus", {"limit": 5, "type": "fact", "apply": True})
    assert out["limit"] == 5
    assert out["applied"] is True


def test_h_run_maintenance_dispatches(monkeypatch, tmp_db):
    from memoirs.engine import memory_engine
    monkeypatch.setattr(memory_engine, "run_daily_maintenance", lambda db: {"ok": 1})
    out = mt.call_tool(tmp_db, "mcp_run_maintenance", {})
    assert out == {"ok": 1}


# ---------------------------------------------------------------------------
# Retrieval / context tools
# ---------------------------------------------------------------------------


def test_h_get_context_requires_query(tmp_db):
    with pytest.raises(ValueError, match="query is required"):
        mt.call_tool(tmp_db, "mcp_get_context", {"query": "  "})


def test_h_get_context_dispatches(monkeypatch, tmp_db):
    from memoirs.engine import memory_engine
    captured = {}

    def fake_assemble(db, query, **kwargs):
        captured.update({"query": query, **kwargs})
        return {"context": [], "memories": []}

    monkeypatch.setattr(memory_engine, "assemble_context", fake_assemble)
    out = mt.call_tool(tmp_db, "mcp_get_context", {
        "query": "alpha", "top_k": 8, "max_lines": 4, "as_of": "2026-04-01T00:00:00+00:00",
    })
    assert "context" in out
    assert captured["query"] == "alpha"
    assert captured["top_k"] == 8
    assert captured["max_lines"] == 4
    assert captured["as_of"] == "2026-04-01T00:00:00+00:00"


def test_h_summarize_thread_requires_id(tmp_db):
    with pytest.raises(ValueError, match="conversation_id is required"):
        mt.call_tool(tmp_db, "mcp_summarize_thread", {})


def test_h_summarize_thread_dispatches(monkeypatch, tmp_db):
    from memoirs.engine import curator as _curator
    monkeypatch.setattr(_curator, "summarize_conversation", lambda db, cid: {"cid": cid})
    out = mt.call_tool(tmp_db, "mcp_summarize_thread", {"conversation_id": "c1"})
    assert out == {"cid": "c1"}


def test_h_search_memory_requires_query(tmp_db):
    with pytest.raises(ValueError, match="query is required"):
        mt.call_tool(tmp_db, "mcp_search_memory", {"query": ""})


def test_h_search_memory_returns_results(monkeypatch, tmp_db):
    monkeypatch.setattr(emb, "search_similar_memories",
                        lambda db, q, top_k: [{"id": "m1", "similarity": 0.9}])
    out = mt.call_tool(tmp_db, "mcp_search_memory", {"query": "alpha", "limit": 3})
    assert out == {"results": [{"id": "m1", "similarity": 0.9}]}


# ---------------------------------------------------------------------------
# Memory CRUD tools
# ---------------------------------------------------------------------------


def test_h_add_memory_validates_content(tmp_db):
    with pytest.raises(ValueError, match="content is required"):
        mt.call_tool(tmp_db, "mcp_add_memory", {"type": "fact", "content": ""})


def test_h_add_memory_validates_type(tmp_db):
    with pytest.raises(ValueError, match="invalid memory type"):
        mt.call_tool(tmp_db, "mcp_add_memory", {"type": "bogus", "content": "hi"})


def test_h_add_memory_happy_path(monkeypatch, tmp_db):
    from memoirs.engine import memory_engine
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    monkeypatch.setattr(memory_engine, "_maybe_link_memory", lambda *a, **k: None)
    out = mt.call_tool(tmp_db, "mcp_add_memory", {
        # Padded over 20 chars so should_skip_extraction's "too short" rule
        # does not trip — the test exercises the ADD happy path.
        "type": "fact", "content": "user prefers Python over JavaScript",
        "importance": 4, "confidence": 0.9,
    })
    assert out["action"] == "ADD"
    assert "memory_id" in out


def test_h_update_memory_validates(tmp_db):
    with pytest.raises(ValueError, match="memory_id and content"):
        mt.call_tool(tmp_db, "mcp_update_memory", {"memory_id": "", "content": "x"})
    with pytest.raises(ValueError, match="memory_id and content"):
        mt.call_tool(tmp_db, "mcp_update_memory", {"memory_id": "m1", "content": "  "})


def test_h_update_memory_dispatches(monkeypatch, tmp_db):
    from memoirs.engine import memory_engine
    monkeypatch.setattr(memory_engine, "create_memory_version",
                        lambda db, mid, content: "new_id_42")
    out = mt.call_tool(tmp_db, "mcp_update_memory", {"memory_id": "old", "content": "v2"})
    assert out == {"ok": True, "old_memory_id": "old", "new_memory_id": "new_id_42"}


def test_h_score_feedback_validates(tmp_db):
    with pytest.raises(ValueError, match="memory_id is required"):
        mt.call_tool(tmp_db, "mcp_score_feedback", {"memory_id": "", "useful": True})


def test_h_score_feedback_missing_memory_raises(tmp_db):
    with pytest.raises(ValueError, match="not found"):
        mt.call_tool(tmp_db, "mcp_score_feedback", {"memory_id": "nope", "useful": True})


def test_h_score_feedback_bumps_signal(tmp_db):
    _seed(tmp_db, memory_id="m1")
    out = mt.call_tool(tmp_db, "mcp_score_feedback", {"memory_id": "m1", "useful": True})
    assert out["ok"] is True
    assert out["new_user_signal"] >= 0.2


def test_h_score_feedback_negative_useful(tmp_db):
    _seed(tmp_db, memory_id="m1")
    # First push it up so subtraction has somewhere to go (clamp at 0).
    mt.call_tool(tmp_db, "mcp_score_feedback", {"memory_id": "m1", "useful": True})
    out = mt.call_tool(tmp_db, "mcp_score_feedback", {"memory_id": "m1", "useful": False})
    assert out["ok"] is True


def test_h_explain_context_requires_query(tmp_db):
    with pytest.raises(ValueError, match="query is required"):
        mt.call_tool(tmp_db, "mcp_explain_context", {"query": "  "})


def test_h_explain_context_returns_rationales(monkeypatch, tmp_db):
    monkeypatch.setattr(emb, "search_similar_memories", lambda db, q, top_k: [
        {"id": "m1", "type": "fact", "content": "alpha bravo charlie",
         "similarity": 0.9, "score": 0.7, "importance": 4, "confidence": 0.8,
         "usage_count": 3},
    ])
    out = mt.call_tool(tmp_db, "mcp_explain_context", {"query": "x", "top_k": 1})
    assert out["query"] == "x"
    assert len(out["results"]) == 1
    assert "rationale" in out["results"][0]


def test_h_forget_memory_validates(tmp_db):
    with pytest.raises(ValueError, match="memory_id is required"):
        mt.call_tool(tmp_db, "mcp_forget_memory", {})


def test_h_forget_memory_archives(tmp_db):
    _seed(tmp_db, memory_id="m1")
    out = mt.call_tool(tmp_db, "mcp_forget_memory", {"memory_id": "m1"})
    assert out["ok"] is True
    archived = tmp_db.conn.execute(
        "SELECT archived_at, archive_reason FROM memories WHERE id='m1'"
    ).fetchone()
    assert archived["archived_at"] is not None
    assert archived["archive_reason"] == "user requested forget"


def test_h_list_memories_no_filter(tmp_db):
    _seed(tmp_db, memory_id="m1", score=0.9)
    _seed(tmp_db, memory_id="m2", score=0.1)
    out = mt.call_tool(tmp_db, "mcp_list_memories", {"limit": 10})
    ids = [m["id"] for m in out["memories"]]
    # Score-DESC order: m1 first.
    assert ids == ["m1", "m2"]


def test_h_list_memories_with_type_filter(tmp_db):
    _seed(tmp_db, memory_id="m1", mem_type="fact")
    _seed(tmp_db, memory_id="m2", mem_type="preference")
    out = mt.call_tool(tmp_db, "mcp_list_memories", {"type": "preference"})
    ids = [m["id"] for m in out["memories"]]
    assert ids == ["m2"]


# ---------------------------------------------------------------------------
# Graph / project tools
# ---------------------------------------------------------------------------


def test_h_index_entities_dispatches(monkeypatch, tmp_db):
    import memoirs.engine.graph as graph
    monkeypatch.setattr(graph, "index_memory_entities",
                        lambda db, *, limit: {"indexed": 0, "limit": limit})
    out = mt.call_tool(tmp_db, "mcp_index_entities", {"limit": 11})
    assert out["limit"] == 11


def test_h_get_project_context_validates(tmp_db):
    with pytest.raises(ValueError, match="project is required"):
        mt.call_tool(tmp_db, "mcp_get_project_context", {})


def test_h_get_project_context_dispatches(monkeypatch, tmp_db):
    import memoirs.engine.graph as graph
    monkeypatch.setattr(graph, "get_project_context",
                        lambda db, project, *, limit: {"project": project, "limit": limit})
    out = mt.call_tool(tmp_db, "mcp_get_project_context", {"project": "memoirs", "limit": 9})
    assert out == {"project": "memoirs", "limit": 9}


def test_h_list_projects(monkeypatch, tmp_db):
    import memoirs.engine.graph as graph
    monkeypatch.setattr(graph, "refresh_projects_from_conversations", lambda db: None)
    out = mt.call_tool(tmp_db, "mcp_list_projects", {})
    assert "projects" in out
    assert isinstance(out["projects"], list)
