"""Tests for the `procedural` memory type.

A procedural memory is an agent-policy instruction ("when X happens, do Y").
We segregate it from the fact list and inject it via the
``system_instructions`` field of ``assemble_context``, regardless of
whether the query lexically matched it. This means an agent's persistent
policies survive even when the user's question is unrelated.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from memoirs.core.ids import content_hash, utc_now
from memoirs.db import MemoirsDB
from memoirs.engine import embeddings as emb
from memoirs.engine import hybrid_retrieval as hr
from memoirs.engine import memory_engine as me


def _seed(db, mid, mtype, content, importance=3):
    now = utc_now()
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, "
        "metadata_json, created_at, updated_at) VALUES "
        "(?, ?, ?, ?, ?, 0.9, 0.5, 0, 0, ?, '{}', ?, ?)",
        (mid, mtype, content, content_hash(content + mid),
         importance, now, now, now),
    )
    db.conn.commit()
    emb.upsert_memory_embedding(db, mid, content)


@pytest.fixture
def db_with_one_procedural():
    with tempfile.TemporaryDirectory() as td:
        db = MemoirsDB(Path(td) / "p.sqlite")
        db.init()
        hr.ensure_fts_schema(db.conn)
        _seed(db, "m_fact", "fact",
              "user lives in Sweden", importance=3)
        _seed(db, "m_proc", "procedural",
              "when user asks for code, prefer minimal diffs", importance=5)
        yield db
        db.close()


def test_procedural_memory_type_accepted(db_with_one_procedural):
    db = db_with_one_procedural
    row = db.conn.execute(
        "SELECT type FROM memories WHERE id = 'm_proc'").fetchone()
    assert row["type"] == "procedural"


def test_procedural_segregated_from_facts(db_with_one_procedural):
    db = db_with_one_procedural
    out = me.assemble_context(
        db, "where does the user live",
        top_k=10, max_lines=5, retrieval_mode="hybrid",
    )
    fact_ids = [m["id"] for m in out["memories"]]
    # Procedural memory MUST NOT appear in the fact list (would pollute the
    # "context" lines an LLM consumes as factual context).
    assert "m_proc" not in fact_ids
    assert "m_fact" in fact_ids


def test_procedural_surfaces_in_system_instructions_even_without_lexical_match(
    db_with_one_procedural,
):
    db = db_with_one_procedural
    out = me.assemble_context(
        db, "totally unrelated query about weather in Tokyo",
        top_k=10, max_lines=5, retrieval_mode="hybrid",
    )
    si = out.get("system_instructions") or []
    # Even when the query is unrelated, the procedural policy must be
    # surfaced — that's the point of "procedural" (always-on instructions).
    assert any(s["id"] == "m_proc" for s in si)


def test_no_system_instructions_field_when_no_procedural(tmp_db):
    # tmp_db is the project-wide fixture; no procedural memories exist.
    out = me.assemble_context(
        tmp_db, "anything", top_k=5, max_lines=3, retrieval_mode="hybrid",
    )
    assert "system_instructions" not in out
