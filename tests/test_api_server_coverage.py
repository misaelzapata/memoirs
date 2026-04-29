"""Targeted coverage tests for memoirs/api/server.py.

The streaming endpoint already has its own test file (test_api_stream.py).
We focus here on the simple JSON endpoints (CRUD-ish + healthz/metrics).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from memoirs.api.server import _build_app, _sse_pack
from memoirs.db import MemoirsDB


@pytest.fixture
def client(tmp_path: Path):
    """A FastAPI TestClient bound to a fresh DB."""
    from fastapi.testclient import TestClient
    db_path = tmp_path / "memoirs.sqlite"
    # Init the DB so endpoints don't trip on missing tables.
    db = MemoirsDB(db_path)
    db.init()
    db.close()
    app = _build_app(db_path)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sse_pack_format():
    out = _sse_pack("meta", {"x": 1})
    s = out.decode()
    assert s.startswith("event: meta\n")
    assert '"x":' in s or '"x": 1' in s


# ---------------------------------------------------------------------------
# /healthz + /metrics
# ---------------------------------------------------------------------------


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_returns_counts(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "memories" in body
    assert "embeddings" in body


# ---------------------------------------------------------------------------
# /memories CRUD
# ---------------------------------------------------------------------------


def test_list_memories_empty(client):
    r = client.get("/memories")
    assert r.status_code == 200
    assert r.json() == []


def test_list_memories_filter_by_type(client):
    r = client.get("/memories", params={"type": "preference", "limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_memory_404(client):
    r = client.get("/memories/no-such-id")
    assert r.status_code == 404


def test_forget_memory_returns_ok(client, tmp_path):
    # Pre-seed a memory directly (avoids needing an embedder).
    db_path = tmp_path / "memoirs.sqlite"
    db = MemoirsDB(db_path)
    db.init()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) VALUES "
        "('m1', 'fact', 'x', 'h_m1', 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (now, now, now),
    )
    db.conn.commit()
    db.close()
    r = client.delete("/memories/m1")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["memory_id"] == "m1"


# Note: POST endpoints whose body is a pydantic model defined inside
# `_build_app` (FeedbackBody, MemoryCreate, SearchBody, ContextBody) are NOT
# tested via TestClient here — `from __future__ import annotations` plus the
# closure-defined models trip a known FastAPI/pydantic v2 forward-ref quirk.
# The same limitation is documented in tests/test_api_stream.py and that
# suite tests the engine layer directly. We rely on those tests to cover
# the body-handling logic; endpoints below only need _get_db wiring to be
# correct, which the GET tests already verify.


# ---------------------------------------------------------------------------
# /projects
# ---------------------------------------------------------------------------


def test_list_projects_empty(client):
    r = client.get("/projects")
    assert r.status_code == 200
    assert r.json() == []


def test_project_context_unknown(client):
    r = client.get("/projects/no-such/context")
    assert r.status_code == 200
    body = r.json()
    assert body["memories"] == []
