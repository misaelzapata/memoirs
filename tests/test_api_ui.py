"""Tests for the local web inspector (P5-1 + P5-3).

The UI ships server-rendered HTML (HTMX + Tailwind via CDN) plus a couple of
JSON endpoints (timeline, graph) for the front-end to consume. These tests
exercise the routes directly via ``TestClient`` so we don't pay the cost of
booting uvicorn.

Coverage map (from the GAP.md acceptance list):
  * GET /ui/memories returns 200 + an HTML table
  * Type filter narrows the result set
  * GET /ui/memories/<missing> returns 404 with HTML
  * POST /ui/memories/<id>/pin sets user_signal=1.0 and returns a fragment
  * GET /ui/timeline returns >= 1 event (JSON)
  * GET /ui/graph?seed=<id>&depth=1 returns valid nodes/edges JSON

A handful of bonus assertions (forget, patch, links fragment, provenance
fragment) keep the regressions tighter than the strict acceptance bar.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from memoirs.api.server import _build_app
from memoirs.core.ids import content_hash, utc_now
from memoirs.db import MemoirsDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db_path(tmp_path: Path) -> Path:
    """A real SQLite DB seeded with three memorias of different types.

    We also add one ``memory_candidate`` and a couple of synthetic source +
    conversation + message rows so the provenance trail has something to
    walk through. Embeddings are skipped — none of the UI routes require
    them and the tests stay fast.
    """
    p = tmp_path / "memoirs.sqlite"
    db = MemoirsDB(p)
    db.init()
    now = utc_now()
    rows = [
        ("mem_alpha",  "preference", "prefers Python over Go for prototyping",       0.81, 0.92),
        ("mem_beta",   "fact",        "memoirs uses local-first memory engine",        0.69, 0.78),
        ("mem_gamma",  "decision",    "use SQLite + sqlite-vec instead of Postgres",   0.74, 0.88),
    ]
    with db.conn:
        for mid, mtype, content, score, conf in rows:
            db.conn.execute(
                """
                INSERT INTO memories (
                    id, type, content, content_hash, importance, confidence,
                    score, usage_count, user_signal, valid_from, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 4, ?, ?, 0, 0, ?, '{}', ?, ?)
                """,
                (mid, mtype, content, content_hash(content), conf, score, now, now, now),
            )
        # Provenance scaffolding: source -> conversation -> message -> candidate
        db.conn.execute(
            """
            INSERT INTO sources (uri, kind, name, content_hash, mtime_ns, size_bytes,
                                 created_at, updated_at)
            VALUES ('file:///fake/chat.md', 'markdown', 'fake-chat', NULL, NULL, NULL,
                    ?, ?)
            """,
            (now, now),
        )
        src_id = db.conn.execute(
            "SELECT id FROM sources WHERE uri = 'file:///fake/chat.md'"
        ).fetchone()[0]
        db.conn.execute(
            """
            INSERT INTO conversations (id, source_id, external_id, title,
                                       created_at, updated_at, message_count, metadata_json)
            VALUES ('conv_test', ?, 'ext_conv', 'Test chat', ?, ?, 1, '{}')
            """,
            (src_id, now, now),
        )
        db.conn.execute(
            """
            INSERT INTO messages (id, conversation_id, external_id, role, content,
                                  ordinal, created_at, content_hash, raw_json,
                                  metadata_json, is_active, first_seen_at, updated_at)
            VALUES ('msg_test', 'conv_test', 'ext_msg', 'user',
                    'I prefer Python over Go for prototyping',
                    0, ?, ?, '{}', '{}', 1, ?, ?)
            """,
            (now, content_hash("I prefer Python over Go for prototyping"), now, now),
        )
        db.conn.execute(
            """
            INSERT INTO memory_candidates (id, conversation_id, source_message_ids,
                                           type, content, importance, confidence,
                                           entities, status, extractor, raw_json,
                                           created_at, updated_at, promoted_memory_id)
            VALUES ('cand_alpha', 'conv_test', '["msg_test"]',
                    'preference', 'prefers Python over Go for prototyping',
                    4, 0.92, '[]', 'accepted', 'gemma-2-2b', '{}',
                    ?, ?, 'mem_alpha')
            """,
            (now, now),
        )
        # A couple of memory_links to exercise the graph endpoint.
        db.conn.execute(
            """
            INSERT INTO memory_links (source_memory_id, target_memory_id, similarity,
                                      reason, created_at)
            VALUES ('mem_alpha', 'mem_beta',  0.74, 'semantic', ?),
                   ('mem_beta',  'mem_alpha', 0.74, 'semantic', ?),
                   ('mem_alpha', 'mem_gamma', 0.61, 'semantic', ?),
                   ('mem_gamma', 'mem_alpha', 0.61, 'semantic', ?)
            """,
            (now, now, now, now),
        )
    db.close()
    return p


@pytest.fixture
def client(seeded_db_path: Path) -> TestClient:
    return TestClient(_build_app(seeded_db_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_returns_html_with_table(client: TestClient):
    r = client.get("/ui/memories")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "<table" in r.text
    # All three seeded memorias should show up.
    assert "prefers Python over Go" in r.text
    assert "use SQLite" in r.text


def test_list_filter_by_type(client: TestClient):
    r = client.get("/ui/memories", params={"type": "preference"})
    assert r.status_code == 200
    assert "prefers Python over Go" in r.text
    # The fact and decision rows must NOT appear when filtering preference.
    assert "use SQLite + sqlite-vec" not in r.text
    assert "memoirs uses local-first" not in r.text


def test_detail_404_for_missing_memory(client: TestClient):
    r = client.get("/ui/memories/mem_does_not_exist")
    assert r.status_code == 404
    # We render an HTML error page, not raw JSON.
    assert "text/html" in r.headers.get("content-type", "")
    assert "memory not found" in r.text.lower()


def test_detail_renders_for_existing_memory(client: TestClient):
    r = client.get("/ui/memories/mem_alpha")
    assert r.status_code == 200
    assert "<h2" in r.text
    # Sanity-check that the controls the user expects are wired up.
    assert "/ui/memories/mem_alpha/pin" in r.text
    assert "/ui/memories/mem_alpha/forget" in r.text
    assert "/ui/memories/mem_alpha/links" in r.text
    assert "/ui/memories/mem_alpha/provenance" in r.text


def test_pin_updates_user_signal_and_returns_fragment(
    client: TestClient, seeded_db_path: Path,
):
    r = client.post("/ui/memories/mem_beta/pin")
    assert r.status_code == 200
    # The returned fragment is HTML, NOT a full page (HTMX swap target).
    assert "<html" not in r.text.lower()
    assert "pinned" in r.text.lower()

    # The DB really did get updated — user_signal=1.0 and score recomputed.
    db = MemoirsDB(seeded_db_path)
    db.init()
    try:
        row = db.conn.execute(
            "SELECT user_signal, score FROM memories WHERE id = 'mem_beta'"
        ).fetchone()
    finally:
        db.close()
    assert row is not None
    assert row["user_signal"] == pytest.approx(1.0, abs=1e-6)
    assert row["score"] > 0.0  # recomputed via calculate_memory_score


def test_pin_404_for_missing(client: TestClient):
    r = client.post("/ui/memories/mem_does_not_exist/pin")
    assert r.status_code == 404


def test_forget_archives_memory(client: TestClient, seeded_db_path: Path):
    r = client.post("/ui/memories/mem_gamma/forget")
    assert r.status_code == 200
    assert "forgotten" in r.text.lower() or "archived" in r.text.lower()

    db = MemoirsDB(seeded_db_path)
    db.init()
    try:
        row = db.conn.execute(
            "SELECT archived_at FROM memories WHERE id = 'mem_gamma'"
        ).fetchone()
    finally:
        db.close()
    assert row is not None
    assert row["archived_at"] is not None


def test_patch_updates_content(client: TestClient, seeded_db_path: Path):
    r = client.patch(
        "/ui/memories/mem_beta",
        data={"content": "memoirs is a local-first SQLite memory engine"},
    )
    assert r.status_code == 200
    assert "saved" in r.text.lower()
    db = MemoirsDB(seeded_db_path)
    db.init()
    try:
        row = db.conn.execute(
            "SELECT content FROM memories WHERE id = 'mem_beta'"
        ).fetchone()
    finally:
        db.close()
    assert "local-first SQLite memory engine" in row["content"]


def test_links_fragment(client: TestClient):
    r = client.get("/ui/memories/mem_alpha/links")
    assert r.status_code == 200
    # The fallback path (no sqlite-vec needed) walks ``memory_links`` and we
    # seeded 2 outgoing edges from mem_alpha. Either the recursive
    # ``zettelkasten.get_neighbors`` or the fallback path must return at
    # least one neighbor.
    assert "mem_beta" in r.text or "mem_gamma" in r.text


def test_provenance_fragment_renders_chain(client: TestClient):
    r = client.get("/ui/memories/mem_alpha/provenance")
    assert r.status_code == 200
    # candidate -> messages -> conversation -> source — all should be visible.
    assert "cand_alpha" in r.text
    assert "msg_test" in r.text or "I prefer Python" in r.text
    assert "Test chat" in r.text or "conv_test" in r.text
    assert "fake-chat" in r.text or "file:///fake/chat.md" in r.text


def test_timeline_returns_json_events(client: TestClient):
    r = client.get("/ui/timeline")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("application/json")
    body = r.json()
    assert "events" in body
    assert body["count"] >= 1
    # Each event has the documented shape.
    for ev in body["events"]:
        assert {"id", "type", "content", "score", "start", "end"} <= set(ev.keys())


def test_graph_seed_returns_nodes_and_edges(client: TestClient):
    r = client.get("/ui/graph", params={"seed": "mem_alpha", "depth": 1})
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body and "edges" in body
    node_ids = {n["data"]["id"] for n in body["nodes"]}
    # Seed must always be present; at least one neighbor reachable in 1 hop.
    assert "mem_alpha" in node_ids
    assert {"mem_beta", "mem_gamma"} & node_ids
    # Each edge must reference real nodes and carry a numeric weight.
    for edge in body["edges"]:
        d = edge["data"]
        assert {"source", "target", "weight"} <= set(d.keys())
        assert d["source"] in node_ids
        assert d["target"] in node_ids
        assert isinstance(d["weight"], (int, float))


def test_graph_no_seed_returns_top_n(client: TestClient):
    r = client.get("/ui/graph", params={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    # Three seeded memorias, all active, all returned.
    ids = {n["data"]["id"] for n in body["nodes"]}
    assert {"mem_alpha", "mem_beta", "mem_gamma"} <= ids


def test_index_serves_dashboard(client: TestClient):
    """``/ui`` (and ``/``) now render the dashboard directly — no redirect.

    The dashboard exposes corpus stats, the doughnut chart of memory types,
    procedural memories and the most recent items.
    """
    r = client.get("/ui", follow_redirects=False)
    assert r.status_code == 200
    body = r.text
    # Must be the dashboard, not memories list.
    assert "Dashboard" in body
    assert "Active memories" in body
    assert "Memory types" in body


def test_search_page_renders(client: TestClient):
    r = client.get("/ui/search")
    assert r.status_code == 200
    assert "/context/stream" in r.text  # the SSE endpoint we consume
    assert "<form" in r.text
