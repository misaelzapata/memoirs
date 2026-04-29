"""Targeted coverage tests for memoirs/engine/graph.py.

Heuristic entity extraction + entity/relationship persistence + project
inference are exercised here. spaCy is NOT required: we mock its availability
to OFF so the spaCy branch is also covered cleanly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memoirs.engine import graph as gr


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_strips():
    assert gr._normalize("  Memoirs  ") == "memoirs"


def test_is_camel_case_examples():
    assert gr._is_camel_case("ReactNative") is True
    assert gr._is_camel_case("FastAPI") is True
    assert gr._is_camel_case("plainword") is False
    assert gr._is_camel_case("ALLCAPS") is False
    assert gr._is_camel_case("ab") is False  # too short


def test_classify_entity_tool_project_concept_other():
    assert gr._classify_entity("python") == "tool"
    assert gr._classify_entity("Memoirs") == "project"
    assert gr._classify_entity("ReactNative") == "concept"
    assert gr._classify_entity("plainword") == "other"


# ---------------------------------------------------------------------------
# extract_entities (heuristic-only branches)
# ---------------------------------------------------------------------------


def test_extract_entities_picks_up_tool_hint(monkeypatch):
    # Force spaCy off so we exercise the vocabulary branch.
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    out = gr.extract_entities("we used python and sqlite to build it")
    names = {n for n, _ in out}
    assert "python" in names
    assert "sqlite" in names
    types = {t for _, t in out}
    assert "tool" in types


def test_extract_entities_backtick_spans(monkeypatch):
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    out = gr.extract_entities("see `MyHelper` and `OtherThing` in the code")
    names = {n for n, _ in out}
    assert "MyHelper" in names or "OtherThing" in names


def test_extract_entities_camel_case_token(monkeypatch):
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    out = gr.extract_entities("the FastAPI endpoint returns JSON")
    names = {n for n, _ in out}
    assert "FastAPI" in names


def test_extract_entities_skips_stoplist(monkeypatch):
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    out = gr.extract_entities("REMINDER: TODO list — note this fixme")
    names = {n.lower() for n, _ in out}
    # Stop-listed words should NOT appear.
    assert "reminder" not in names
    assert "todo" not in names


# ---------------------------------------------------------------------------
# Entity persistence
# ---------------------------------------------------------------------------


def test_upsert_entity_idempotent(tmp_db):
    a = gr.upsert_entity(tmp_db, "memoirs", "project")
    b = gr.upsert_entity(tmp_db, "memoirs", "project")
    assert a == b
    rows = tmp_db.conn.execute(
        "SELECT COUNT(*) FROM entities WHERE normalized_name='memoirs' AND type='project'"
    ).fetchone()[0]
    assert rows == 1


def test_upsert_entity_classifies_when_etype_missing(tmp_db):
    eid = gr.upsert_entity(tmp_db, "python")
    row = tmp_db.conn.execute(
        "SELECT type FROM entities WHERE id = ?", (eid,)
    ).fetchone()
    assert row["type"] == "tool"


def test_link_memory_to_entities_counts_inserts(tmp_db):
    # Seed a memory + two entities.
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp_db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) VALUES "
        "('m1', 'fact', 'x', 'h_x', 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (now, now, now),
    )
    e1 = gr.upsert_entity(tmp_db, "python")
    e2 = gr.upsert_entity(tmp_db, "memoirs", "project")
    n = gr.link_memory_to_entities(tmp_db, "m1", [e1, e2])
    assert n == 2
    # Re-linking is idempotent — INSERT OR IGNORE returns 0.
    n2 = gr.link_memory_to_entities(tmp_db, "m1", [e1, e2])
    assert n2 == 0


def test_create_relationship_inserts_edge(tmp_db):
    rid = gr.create_relationship(tmp_db, "memoirs", "uses", "python")
    row = tmp_db.conn.execute(
        "SELECT relation FROM relationships WHERE id = ?", (rid,)
    ).fetchone()
    assert row["relation"] == "uses"


# ---------------------------------------------------------------------------
# Project inference
# ---------------------------------------------------------------------------


def test_refresh_projects_from_cwd(tmp_db):
    """A conversation with metadata.cwd should yield a project entity."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Seed source first.
    tmp_db.conn.execute(
        "INSERT INTO sources (uri, kind, name, content_hash, mtime_ns, size_bytes, "
        "created_at, updated_at) VALUES ('s://1','test','test',NULL,NULL,NULL,?,?)",
        (now, now),
    )
    src_id = tmp_db.conn.execute("SELECT id FROM sources WHERE uri='s://1'").fetchone()["id"]
    md = json.dumps({"cwd": "/home/user/Desktop/projects/myproj"})
    tmp_db.conn.execute(
        "INSERT INTO conversations (id, source_id, external_id, title, created_at, "
        "updated_at, message_count, metadata_json) "
        "VALUES ('c1', ?, 'ext1', 't', ?, ?, 0, ?)",
        (src_id, now, now, md),
    )
    tmp_db.conn.commit()
    out = gr.refresh_projects_from_conversations(tmp_db)
    assert out["projects"] >= 1
    assert "myproj" in out["names"]


def test_get_project_context_unknown_returns_empty(tmp_db):
    out = gr.get_project_context(tmp_db, "no-such-project")
    assert out["memories"] == []
    assert out["related_entities"] == []


# ---------------------------------------------------------------------------
# build_relationships — co-occurrence + decision_link branches
# ---------------------------------------------------------------------------


def _seed_mem_with_entities(db, *, memory_id: str, mem_type: str, entity_ids: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) VALUES "
        "(?, ?, 'x', 'h_'||?, 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (memory_id, mem_type, memory_id, now, now, now),
    )
    for eid in entity_ids:
        db.conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (memory_id, eid),
        )
    db.conn.commit()


def test_build_relationships_co_occurrence(tmp_db, monkeypatch):
    """Pairs of entities sharing ≥2 memorias get a co_occurs_in edge."""
    e1 = gr.upsert_entity(tmp_db, "python")
    e2 = gr.upsert_entity(tmp_db, "memoirs", "project")
    # Two memories that share both entities → count = 2
    _seed_mem_with_entities(tmp_db, memory_id="m1", mem_type="fact", entity_ids=[e1, e2])
    _seed_mem_with_entities(tmp_db, memory_id="m2", mem_type="fact", entity_ids=[e1, e2])
    # Disable spaCy to skip the uses pass.
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    out = gr.build_relationships(tmp_db, min_co_occurrence=2)
    assert out["co_occurs_in"] >= 1


def test_build_relationships_decision_link(tmp_db, monkeypatch):
    """A decision-type memory linking 2+ entities → decision_link edge."""
    e1 = gr.upsert_entity(tmp_db, "python")
    e2 = gr.upsert_entity(tmp_db, "memoirs", "project")
    _seed_mem_with_entities(tmp_db, memory_id="d1", mem_type="decision", entity_ids=[e1, e2])
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    out = gr.build_relationships(tmp_db, min_co_occurrence=99)  # disable co-occurrence
    assert out["decision_link"] >= 1


def test_build_relationships_empty_graph_returns_zero(tmp_db, monkeypatch):
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    out = gr.build_relationships(tmp_db, min_co_occurrence=2)
    assert out["total"] == 0


def test_index_memory_entities_ignores_archived(tmp_db, monkeypatch):
    """Archived memories shouldn't be picked up by the indexer."""
    from memoirs.engine import extract_spacy
    monkeypatch.setattr(extract_spacy, "is_available", lambda: False)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp_db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, archived_at, "
        "metadata_json, created_at, updated_at) VALUES "
        "('arch', 'fact', 'using python and sqlite', 'h_arch', 3, 0.5, 0.5, 0, 0, ?, ?, '{}', ?, ?)",
        (now, now, now, now),
    )
    tmp_db.conn.commit()
    out = gr.index_memory_entities(tmp_db)
    assert out["memories_processed"] == 0
