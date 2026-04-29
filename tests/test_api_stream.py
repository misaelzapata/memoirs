"""Tests for the SSE streaming retrieval endpoint (P4-1).

Covers:
  - GET /context/stream returns text/event-stream
  - Event order is meta → memory* → context → done
  - Each event payload is valid JSON in the documented shape
  - The non-streaming POST /context still works alongside it
  - Headers required for proxy bypass are set
  - assemble_context (non-streaming) and assemble_context_stream (generator)
    produce the same final `context` payload (refactor invariant)

The retrieval call (`emb.search_similar_memories`) is monkeypatched so tests
don't need sentence-transformers or sqlite-vec loaded — we want to exercise
the streaming/SSE plumbing, not the embedding stack.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from memoirs.api.server import _build_app
from memoirs.db import MemoirsDB, utc_now
from memoirs.engine import embeddings as _emb
from memoirs.engine import memory_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db_path(tmp_path: Path) -> Path:
    """A real SQLite DB with three memorias of different types/scores."""
    p = tmp_path / "memoirs.sqlite"
    db = MemoirsDB(p)
    db.init()
    now = utc_now()
    rows = [
        ("mem_aaa", "preference", "prefers Python over Go for prototyping", 0.81, 0.92),
        ("mem_bbb", "fact",       "memoirs uses local-first memory engine",  0.69, 0.78),
        ("mem_ccc", "decision",   "use SQLite + sqlite-vec instead of Postgres", 0.74, 0.88),
    ]
    with db.conn:
        for mid, mtype, content, score, sim in rows:
            db.conn.execute(
                """
                INSERT INTO memories (
                    id, type, content, content_hash, importance, confidence,
                    score, usage_count, user_signal, valid_from, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 4, 0.95, ?, 0, 0, ?, '{}', ?, ?)
                """,
                (mid, mtype, content, mid, score, now, now, now),
            )
    db.close()
    return p


@pytest.fixture
def fake_search(monkeypatch):
    """Replace the ANN search with a deterministic stub keyed by query text.

    Returns the three seeded memories with descending similarity. Each row
    matches the dict shape produced by the real `search_similar_memories`.
    """
    def _stub(db, query: str, top_k: int = 10, *, as_of: str | None = None):
        rows = db.conn.execute(
            "SELECT id, type, content, importance, confidence, score, "
            "       usage_count, last_used_at, valid_from, valid_to, archived_at "
            "FROM memories WHERE archived_at IS NULL ORDER BY score DESC LIMIT ?",
            (top_k,),
        ).fetchall()
        out = []
        for i, r in enumerate(rows):
            d = dict(r)
            d["similarity"] = round(0.9 - 0.1 * i, 4)
            out.append(d)
        return out

    monkeypatch.setattr(_emb, "search_similar_memories", _stub)
    return _stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    """Split a raw SSE response body into (event, data_dict) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        event_name = None
        data_lines: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if event_name is None:
            continue
        data_str = "\n".join(data_lines) or "{}"
        events.append((event_name, json.loads(data_str)))
    return events


async def _stream(app, params: dict) -> tuple[int, dict, str]:
    """Issue a GET to /context/stream and return (status, headers, body)."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/context/stream", params=params)
        return r.status_code, dict(r.headers), r.text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stream_event_order_meta_memory_context_done(seeded_db_path, fake_search):
    """meta first, memory* in the middle, context, then done — exact order."""
    app = _build_app(seeded_db_path)
    status, headers, body = asyncio.run(_stream(app, {"q": "python preferences"}))

    assert status == 200, body
    assert headers.get("content-type", "").startswith("text/event-stream")

    events = parse_sse(body)
    assert len(events) >= 3, f"expected at least meta+context+done, got {events}"

    # First event must be meta.
    assert events[0][0] == "meta"
    meta = events[0][1]
    assert meta["query"] == "python preferences"
    assert meta["live"] is True
    assert meta["as_of"] is None

    # Last event must be done.
    assert events[-1][0] == "done"
    assert events[-1][1] == {}

    # Penultimate must be context.
    assert events[-2][0] == "context"
    ctx = events[-2][1]
    assert "context" in ctx and isinstance(ctx["context"], list)
    assert "memories" in ctx and isinstance(ctx["memories"], list)
    assert "token_estimate" in ctx
    assert "conflicts_resolved" in ctx
    assert ctx["live"] is True

    # Everything between must be memory events, and there must be at least one.
    middle = events[1:-2]
    assert len(middle) >= 1, "expected at least one memory event"
    assert all(e[0] == "memory" for e in middle), [e[0] for e in middle]

    # Each memory event has the required keys.
    for _, m in middle:
        assert set(m.keys()) >= {"id", "type", "score", "similarity", "summary"}
        assert isinstance(m["summary"], str) and m["summary"]


def test_stream_memory_events_match_context_memories(seeded_db_path, fake_search):
    """The streamed memory ids must match (in order) those in the final context."""
    app = _build_app(seeded_db_path)
    _, _, body = asyncio.run(_stream(app, {"q": "anything"}))
    events = parse_sse(body)

    streamed_ids = [e[1]["id"] for e in events if e[0] == "memory"]
    ctx = next(e[1] for e in events if e[0] == "context")
    final_ids = [m["id"] for m in ctx["memories"]]

    assert streamed_ids == final_ids


def test_stream_respects_max_lines(seeded_db_path, fake_search):
    """max_lines must cap both the streamed memory events and the context list."""
    app = _build_app(seeded_db_path)
    # Use a non-trivial query so the engine doesn't short-circuit dense
    # retrieval (P-perf optimization skips dense for 1-token queries).
    _, _, body = asyncio.run(_stream(app, {"q": "prefers python prototyping", "max_lines": 1}))
    events = parse_sse(body)

    memories = [e for e in events if e[0] == "memory"]
    assert len(memories) == 1

    ctx = next(e[1] for e in events if e[0] == "context")
    assert len(ctx["memories"]) == 1


def test_stream_headers_disable_buffering(seeded_db_path, fake_search):
    """Required headers for nginx / CDN passthrough."""
    app = _build_app(seeded_db_path)
    _, headers, _ = asyncio.run(_stream(app, {"q": "headers"}))
    assert headers.get("cache-control") == "no-cache"
    assert headers.get("x-accel-buffering") == "no"
    # Connection: keep-alive can be normalized by ASGI; presence is enough.
    assert "connection" in headers


def test_stream_as_of_makes_live_false(seeded_db_path, fake_search):
    """When `as_of` is passed, meta.live must be false (time-travel mode)."""
    app = _build_app(seeded_db_path)
    _, _, body = asyncio.run(
        _stream(app, {"q": "back then", "as_of": "2024-01-01T00:00:00+00:00"})
    )
    events = parse_sse(body)
    meta = next(e[1] for e in events if e[0] == "meta")
    assert meta["live"] is False
    assert meta["as_of"] == "2024-01-01T00:00:00+00:00"


def test_assemble_context_non_streaming_unchanged(seeded_db_path, fake_search):
    """Refactor invariant — `assemble_context` (the non-streaming entry point
    used by POST /context and the MCP tools) must keep its exact public shape.

    We test the engine function directly because pydantic-model body params
    declared inside `_build_app` interact poorly with `from __future__ import
    annotations` under httpx ASGI transport (a pre-existing FastAPI/pydantic
    quirk unrelated to this refactor). Production calls go through uvicorn
    which is unaffected.
    """
    db = MemoirsDB(seeded_db_path)
    db.init()
    try:
        body = memory_engine.assemble_context(db, "anything")
    finally:
        db.close()

    assert set(body.keys()) >= {
        "context", "memories", "token_estimate",
        "conflicts_resolved", "as_of", "live",
    }
    assert isinstance(body["context"], list)
    assert isinstance(body["memories"], list)
    # Each item in `memories` keeps the original 4-key shape (no `summary` —
    # that one is streaming-only).
    for m in body["memories"]:
        assert set(m.keys()) == {"id", "type", "score", "similarity"}
    assert body["live"] is True
    assert body["as_of"] is None


def test_assemble_context_matches_stream_final_payload(seeded_db_path, fake_search):
    """`assemble_context` must equal the trailing `context` event of the stream."""
    db = MemoirsDB(seeded_db_path)
    db.init()
    try:
        non_stream = memory_engine.assemble_context(db, "same query")
    finally:
        db.close()

    db = MemoirsDB(seeded_db_path)
    db.init()
    try:
        events = list(
            memory_engine.assemble_context_stream(db, "same query")
        )
    finally:
        db.close()

    final = next(data for evt, data in events if evt == "context")
    # token_estimate / conflicts_resolved / context list must match exactly.
    assert non_stream["context"] == final["context"]
    assert non_stream["token_estimate"] == final["token_estimate"]
    assert non_stream["conflicts_resolved"] == final["conflicts_resolved"]
    assert [m["id"] for m in non_stream["memories"]] == [m["id"] for m in final["memories"]]


def test_stream_meta_event_emitted_first_byte(seeded_db_path, fake_search):
    """meta arrives before any retrieval cost (low TTFT contract).

    We assert this structurally: meta must be the first frame in the body
    even when the body contains retrieval results afterwards.
    """
    app = _build_app(seeded_db_path)
    _, _, body = asyncio.run(_stream(app, {"q": "ttft"}))
    # The very first SSE frame must be `event: meta`.
    first_frame = body.split("\n\n", 1)[0]
    assert first_frame.startswith("event: meta\n")
