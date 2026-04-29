"""Local web inspector for memoirs (P5-1 + P5-3).

A FastAPI ``APIRouter`` that ships server-rendered HTML (Jinja2) wired up
with HTMX + Tailwind via CDN — no build step required. This is the UI that
lets the user *see* the memory engine: list, search, edit, pin/forget, plus
a provenance trail showing source -> messages -> candidate -> memory.

Public surface
~~~~~~~~~~~~~~
``build_router(db_path)`` returns the router for mounting under ``/ui``.
``mount_ui(app, db_path)`` is the convenience wrapper used by
``memoirs.api.server`` and the ``memoirs ui`` CLI.

Notes for maintainers
~~~~~~~~~~~~~~~~~~~~~
* All route handlers are declared at module top-level (not nested inside a
  factory) to side-step the FastAPI/pydantic quirk where parameters
  declared inside a closure with ``from __future__ import annotations``
  break under ``httpx`` ASGI transport in tests.
* The DB path is captured per-router via a module-level state dict so each
  mount is independent. Tests and the CLI mount the router fresh, so each
  call to :func:`mount_ui` overrides the path — that's intentional.
"""
from __future__ import annotations

import json as _json
import sqlite3 as _sqlite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ...engine import snapshots as _snap

# Import ``Request`` at module level so ``typing.get_type_hints`` can resolve
# the forward references used by route handlers when this module is loaded
# with ``from __future__ import annotations`` (PEP 563). FastAPI calls
# ``get_type_hints`` to introspect handler signatures.
try:
    from fastapi import Request  # noqa: F401
except ImportError:  # pragma: no cover -- fastapi extra not installed
    if TYPE_CHECKING:
        from fastapi import Request  # noqa: F401


_state: dict[str, Path] = {}


def _templates():
    """Lazy-create the Jinja2Templates singleton.

    Lazy so importing this module doesn't require Jinja2 unless the UI is
    actually mounted.
    """
    from fastapi.templating import Jinja2Templates

    here = Path(__file__).parent / "templates"
    return Jinja2Templates(directory=str(here))


def _get_db():
    """Open a fresh ``MemoirsDB`` against the configured path.

    A new connection per request keeps things simple and avoids cross-thread
    issues with SQLite under uvicorn's threadpool.
    """
    from ...db import MemoirsDB

    path = _state.get("db_path")
    if path is None:  # pragma: no cover -- guarded by mount_ui
        raise RuntimeError("UI router not initialised; call mount_ui first")
    db = MemoirsDB(path)
    db.init()
    return db


def _short(text: str, n: int = 90) -> str:
    """Single-line truncation for table cells."""
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_router(db_path: Path):
    """Construct the ``APIRouter`` and bind the DB path."""
    from fastapi import APIRouter

    _state["db_path"] = Path(db_path)
    router = APIRouter(tags=["ui"])

    router.get("/ui", include_in_schema=False)(_ui_index)
    router.get("/ui/", include_in_schema=False)(_ui_index)
    router.get("/ui/memories")(_ui_memories_list)
    router.get("/ui/memories/{memory_id}")(_ui_memory_detail)
    router.get("/ui/memories/{memory_id}/links")(_ui_memory_links)
    router.get("/ui/memories/{memory_id}/provenance")(_ui_memory_provenance)
    router.get("/ui/memories/{memory_id}/why")(_ui_memory_why)
    router.post("/ui/memories/{memory_id}/pin")(_ui_memory_pin)
    router.post("/ui/memories/{memory_id}/forget")(_ui_memory_forget)
    router.patch("/ui/memories/{memory_id}")(_ui_memory_patch)
    router.get("/ui/timeline")(_ui_timeline)
    router.get("/ui/graph")(_ui_graph)
    router.get("/ui/search")(_ui_search)
    router.get("/ui/conflicts")(_ui_conflicts_list)
    router.get("/ui/conflicts/{conflict_id}")(_ui_conflict_detail)
    router.post("/ui/conflicts/{conflict_id}/resolve")(_ui_conflict_resolve)
    router.get("/ui/snapshots")(_ui_snapshots_list)
    router.post("/ui/snapshots/create")(_ui_snapshots_create)
    router.get("/ui/snapshots/diff")(_ui_snapshots_diff)
    router.post("/ui/snapshots/restore")(_ui_snapshots_restore)
    return router


def mount_ui(app, db_path: Path) -> None:
    """Mount the UI router (and static files) onto an existing FastAPI app."""
    from fastapi.staticfiles import StaticFiles

    router = build_router(db_path)
    app.include_router(router)
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount(
        "/ui/static",
        StaticFiles(directory=str(static_dir)),
        name="ui-static",
    )


# ============================================================================
# Route handlers — top-level functions; FastAPI will inject ``Request``.
# ============================================================================


def _ui_index(request: "Request"):  # type: ignore[name-defined]
    """Dashboard: corpus stats + new-feature pointers (procedural,
    attribution, point-in-time snapshots)."""
    from collections import Counter

    db = _get_db()
    try:
        rows = db.conn.execute(
            "SELECT type, COUNT(*) AS n FROM memories WHERE archived_at IS NULL "
            "GROUP BY type ORDER BY n DESC"
        ).fetchall()
        type_counts = [(r["type"], r["n"]) for r in rows]
        total_active = sum(n for _, n in type_counts)
        archived_n = db.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived_at IS NOT NULL"
        ).fetchone()[0]
        provenance_filled = db.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE provenance_json != '{}' "
            "AND archived_at IS NULL"
        ).fetchone()[0]
        actor_rows = db.conn.execute(
            "SELECT json_extract(provenance_json, '$.actor') AS actor, COUNT(*) AS n "
            "FROM memories WHERE provenance_json != '{}' AND archived_at IS NULL "
            "GROUP BY actor ORDER BY n DESC"
        ).fetchall()
        actor_counts = [(r["actor"] or "unknown", r["n"]) for r in actor_rows]
        proc_rows = db.conn.execute(
            "SELECT id, content, importance FROM memories "
            "WHERE type='procedural' AND archived_at IS NULL "
            "ORDER BY importance DESC, score DESC LIMIT 10"
        ).fetchall()
        recent = db.conn.execute(
            "SELECT id, type, content, created_at FROM memories "
            "WHERE archived_at IS NULL ORDER BY created_at DESC LIMIT 8"
        ).fetchall()
    finally:
        db.close()

    from .. import ui as _ui_pkg  # noqa: F401 (just to keep templates loader local)
    from ..ui import _templates as _t

    # Snapshots — best effort.
    try:
        from ...engine import snapshots as _snap
        snaps = _snap.list_snapshots(_state["db_path"])
    except Exception:
        snaps = []

    return _t().TemplateResponse(
        request, "index.html",
        {
            "total_active": total_active,
            "archived_n": archived_n,
            "type_counts": type_counts,
            "provenance_filled": provenance_filled,
            "actor_counts": actor_counts,
            "procedural_top": [dict(r) for r in proc_rows],
            "recent": [dict(r) for r in recent],
            "snapshots": snaps[:5],
            "db_path": str(_state["db_path"]),
        },
    )


def _ui_memories_list(
    request: "Request",  # type: ignore[name-defined]  -- forward ref, FastAPI resolves
    type: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
):
    """Paginated list of memorias.

    ``q`` does a substring match against ``content`` so the search box works
    without depending on FTS being built. Type filter is exact.
    """
    page = max(1, int(page or 1))
    page_size = max(1, min(200, int(page_size or 25)))
    offset = (page - 1) * page_size

    where = ["archived_at IS NULL"]
    params: list[Any] = []
    if type:
        where.append("type = ?")
        params.append(type)
    if q:
        where.append("content LIKE ?")
        params.append(f"%{q}%")
    where_sql = " AND ".join(where)

    db = _get_db()
    try:
        rows = db.conn.execute(
            f"""
            SELECT id, type, content, score, importance, confidence,
                   user_signal, created_at, updated_at
            FROM memories
            WHERE {where_sql}
            ORDER BY score DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, page_size, offset),
        ).fetchall()
        total = db.conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {where_sql}", params,
        ).fetchone()[0]
        types = [
            r[0] for r in db.conn.execute(
                "SELECT DISTINCT type FROM memories "
                "WHERE archived_at IS NULL ORDER BY type"
            ).fetchall()
        ]
    finally:
        db.close()

    items = [
        {
            "id": r["id"],
            "type": r["type"],
            "content": r["content"],
            "summary": _short(r["content"], 120),
            "score": r["score"],
            "importance": r["importance"],
            "confidence": r["confidence"],
            "user_signal": r["user_signal"] or 0.0,
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "pinned": (r["user_signal"] or 0.0) >= 0.999,
        }
        for r in rows
    ]
    pages = max(1, (total + page_size - 1) // page_size)
    is_htmx = request.headers.get("hx-request", "").lower() == "true"
    template_name = "_list_table.html" if is_htmx else "list.html"
    return _templates().TemplateResponse(
        request, template_name,
        {
            "items": items,
            "total": total,
            "page": page,
            "pages": pages,
            "page_size": page_size,
            "type": type or "",
            "q": q or "",
            "types": types,
            "title": "Memorias",
        },
    )


def _ui_memory_detail(request: "Request", memory_id: str):  # type: ignore[name-defined]
    from fastapi.responses import HTMLResponse

    db = _get_db()
    try:
        row = db.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            html = _templates().get_template("not_found.html").render(
                memory_id=memory_id
            )
            return HTMLResponse(html, status_code=404)
        memory = dict(row)
        entities = [
            dict(e)
            for e in db.conn.execute(
                """
                SELECT e.id, e.name, e.type
                FROM memory_entities me
                JOIN entities e ON e.id = me.entity_id
                WHERE me.memory_id = ?
                ORDER BY e.name
                """,
                (memory_id,),
            ).fetchall()
        ]
    finally:
        db.close()
    memory["pinned"] = (memory.get("user_signal") or 0.0) >= 0.999
    return _templates().TemplateResponse(
        request, "detail.html",
        {
            "memory": memory,
            "entities": entities,
            "title": f"Memory {memory_id[:14]}",
        },
    )


def _ui_memory_links(request: "Request", memory_id: str, depth: int = 1):  # type: ignore[name-defined]
    """HTML fragment with the Zettelkasten neighbors of a memory."""
    depth = max(1, min(3, int(depth or 1)))
    db = _get_db()
    try:
        try:
            from ...engine import zettelkasten as _zk
            neighbors = _zk.get_neighbors(
                db, memory_id, max_depth=depth, min_similarity=0.0,
            )
            links = [
                {
                    "id": n.memory_id,
                    "type": n.type,
                    "summary": _short(n.content or "", 110),
                    "similarity": n.similarity,
                    "depth": n.depth,
                    "reason": n.reason,
                }
                for n in neighbors
            ]
        except Exception:
            # Fall back to a direct memory_links query if the zettelkasten
            # module can't load (e.g. sqlite-vec not present).
            try:
                rows = db.conn.execute(
                    """
                    SELECT ml.target_memory_id AS id, ml.similarity, ml.reason,
                           m.type, m.content
                    FROM memory_links ml
                    JOIN memories m ON m.id = ml.target_memory_id
                    WHERE ml.source_memory_id = ?
                      AND m.archived_at IS NULL
                    ORDER BY ml.similarity DESC
                    LIMIT 25
                    """,
                    (memory_id,),
                ).fetchall()
            except Exception:
                rows = []
            links = [
                {
                    "id": r["id"],
                    "type": r["type"],
                    "summary": _short(r["content"] or "", 110),
                    "similarity": r["similarity"],
                    "depth": 1,
                    "reason": r["reason"],
                }
                for r in rows
            ]
    finally:
        db.close()
    return _templates().TemplateResponse(
        request, "_links.html",
        {"links": links, "memory_id": memory_id},
    )


def _ui_memory_provenance(request: "Request", memory_id: str):  # type: ignore[name-defined]
    """HTML fragment: source -> conversation -> messages -> candidate chain.

    Best-effort. Anything we can't reconstruct shows up as "unknown" rather
    than 500'ing — the UI is a debugging tool, not an authoritative API.
    """
    from fastapi.responses import HTMLResponse

    db = _get_db()
    try:
        memory = db.conn.execute(
            "SELECT id, type, content, created_at, provenance_json "
            "FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if not memory:
            return HTMLResponse(
                "<p class='text-red-600'>memory not found</p>",
                status_code=404,
            )
        # Migration 012 — surface the structured attribution stamp.
        try:
            provenance = _json.loads(memory["provenance_json"] or "{}")
        except Exception:
            provenance = {}
        # 1. Find candidates that promoted to this memory (if any).
        candidates = db.conn.execute(
            """
            SELECT id, conversation_id, source_message_ids, type, content,
                   importance, confidence, status, extractor, created_at
            FROM memory_candidates
            WHERE promoted_memory_id = ?
            ORDER BY created_at
            """,
            (memory_id,),
        ).fetchall()
        chain = []
        for cand in candidates:
            cand_d = dict(cand)
            try:
                msg_ids = list(_json.loads(cand_d.get("source_message_ids") or "[]"))
            except Exception:
                msg_ids = []
            messages: list[dict] = []
            if msg_ids:
                placeholders = ",".join("?" * len(msg_ids))
                rows = db.conn.execute(
                    f"""
                    SELECT id, conversation_id, role, ordinal, content, created_at
                    FROM messages WHERE id IN ({placeholders})
                    ORDER BY conversation_id, ordinal
                    """,
                    msg_ids,
                ).fetchall()
                messages = [
                    {**dict(r), "summary": _short(r["content"], 160)}
                    for r in rows
                ]
            conv_d: dict | None = None
            source_d: dict | None = None
            conv_id = cand_d.get("conversation_id")
            if conv_id:
                conv = db.conn.execute(
                    """
                    SELECT c.id, c.title, c.created_at, c.message_count,
                           s.id AS source_id, s.uri, s.kind, s.name
                    FROM conversations c
                    LEFT JOIN sources s ON s.id = c.source_id
                    WHERE c.id = ?
                    """,
                    (conv_id,),
                ).fetchone()
                if conv:
                    conv_d = {
                        "id": conv["id"],
                        "title": conv["title"],
                        "created_at": conv["created_at"],
                        "message_count": conv["message_count"],
                    }
                    if conv["source_id"]:
                        source_d = {
                            "id": conv["source_id"],
                            "uri": conv["uri"],
                            "kind": conv["kind"],
                            "name": conv["name"],
                        }
            chain.append({
                "candidate": cand_d,
                "messages": messages,
                "conversation": conv_d,
                "source": source_d,
            })
    finally:
        db.close()
    return _templates().TemplateResponse(
        request, "_provenance.html",
        {"memory": dict(memory), "chain": chain, "provenance": provenance},
    )


def _ui_memory_why(  # type: ignore[name-defined]
    request: "Request",
    memory_id: str,
    q: Optional[str] = None,
):
    """HTMX fragment: chain-of-memory provenance for ``memory_id`` given ``q``.

    Walks the joint memory↔entity graph (P1-2 / P1-3) and renders the
    shortest path from query-seed entities to the target memory. Falls
    back to a ``semantic_match`` step when no graph path exists.
    """
    from ...engine import explain as explain_mod

    query = (q or "").strip()
    db = _get_db()
    try:
        chain = explain_mod.build_provenance_chain(
            db, query, None, memory_id, max_hops=3,
        )
    finally:
        db.close()
    return _templates().TemplateResponse(
        request, "_why.html",
        {"chain": chain, "memory_id": memory_id, "query": query},
    )


def _ui_memory_pin(request: "Request", memory_id: str):  # type: ignore[name-defined]
    """Mark a memory as pinned: max user_signal, recompute score."""
    from fastapi.responses import HTMLResponse

    db = _get_db()
    try:
        from ...core.ids import utc_now
        from ...engine.memory_engine import calculate_memory_score

        row = db.conn.execute(
            "SELECT * FROM memories WHERE id = ? AND archived_at IS NULL",
            (memory_id,),
        ).fetchone()
        if not row:
            return HTMLResponse(
                "<p class='text-red-600'>memory not found</p>", status_code=404,
            )
        new_signal = 1.0
        d = dict(row)
        d["user_signal"] = new_signal
        new_score = calculate_memory_score(d)
        db.conn.execute(
            "UPDATE memories SET user_signal = ?, score = ?, updated_at = ? "
            "WHERE id = ?",
            (new_signal, new_score, utc_now(), memory_id),
        )
        db.conn.commit()
    finally:
        db.close()
    return _templates().TemplateResponse(
        request, "_pinned.html",
        {"memory_id": memory_id, "score": new_score},
    )


def _ui_memory_forget(request: "Request", memory_id: str):  # type: ignore[name-defined]
    """Soft-delete via ``archived_at = now``."""
    from fastapi.responses import HTMLResponse

    db = _get_db()
    try:
        from ...core.ids import utc_now
        cursor = db.conn.execute(
            "UPDATE memories SET archived_at = ?, "
            "  archive_reason = 'UI forget' WHERE id = ? AND archived_at IS NULL",
            (utc_now(), memory_id),
        )
        db.conn.commit()
        affected = cursor.rowcount
    finally:
        db.close()
    if not affected:
        return HTMLResponse(
            "<p class='text-red-600'>memory not found</p>", status_code=404,
        )
    return _templates().TemplateResponse(
        request, "_forgotten.html",
        {"memory_id": memory_id},
    )


async def _ui_memory_patch(request: "Request", memory_id: str):  # type: ignore[name-defined]
    """Edit a memory's ``content`` (HTMX form-encoded body).

    Versioning is intentionally light: in-place update + hash bump. A full
    bitemporal split (close old row with ``valid_to``, open new row with
    ``superseded_by``) is the next iteration.
    """
    from fastapi.responses import HTMLResponse

    new_content = ""
    body = await request.body()
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            data = _json.loads(body or b"{}")
            new_content = (data.get("content") or "").strip()
        except Exception:
            new_content = ""
    elif "application/x-www-form-urlencoded" in ctype or (body and b"=" in body and b"{" not in body):
        # Parse the urlencoded body directly so we don't depend on
        # python-multipart at runtime (it's an optional FastAPI extra).
        from urllib.parse import parse_qs

        try:
            parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
            vals = parsed.get("content") or []
            new_content = (vals[0] if vals else "").strip()
        except Exception:
            new_content = ""
    else:
        # Last-ditch: try to read it as JSON.
        try:
            data = _json.loads(body or b"{}")
            new_content = (data.get("content") or "").strip()
        except Exception:
            new_content = ""

    if not new_content:
        return HTMLResponse(
            "<p class='text-red-600'>content is required</p>", status_code=400,
        )
    db = _get_db()
    try:
        from ...core.ids import content_hash, utc_now

        row = db.conn.execute(
            "SELECT id FROM memories WHERE id = ? AND archived_at IS NULL",
            (memory_id,),
        ).fetchone()
        if not row:
            return HTMLResponse(
                "<p class='text-red-600'>memory not found</p>", status_code=404,
            )
        db.conn.execute(
            "UPDATE memories SET content = ?, content_hash = ?, updated_at = ? "
            "WHERE id = ?",
            (new_content, content_hash(new_content), utc_now(), memory_id),
        )
        db.conn.commit()
    finally:
        db.close()
    return HTMLResponse("<div class='text-emerald-700'>saved</div>")


def _ui_timeline(
    request: "Request",  # type: ignore[name-defined]
    type: Optional[str] = None,
    to: Optional[str] = None,
):
    """Chronological events.

    Content-negotiated: returns the ``timeline.html`` page when the request
    accepts HTML (e.g. clicking the sidebar link in a browser), and JSON
    otherwise (used by the page's own ``fetch`` calls and external clients).

    ``from`` is a Python keyword so we read it from the query string
    directly via ``request.query_params``.
    """
    from_value: Optional[str] = request.query_params.get("from") if request else None
    accept = request.headers.get("accept", "") if request else ""
    wants_html = (
        "text/html" in accept
        and "application/json" not in accept
        and request.headers.get("hx-request", "").lower() != "true"
    )

    if wants_html:
        return _templates().TemplateResponse(
            request, "timeline.html", {"title": "Timeline"},
        )

    db = _get_db()
    try:
        where = ["archived_at IS NULL"]
        params: list[Any] = []
        if type:
            where.append("type = ?")
            params.append(type)
        if from_value:
            where.append("created_at >= ?")
            params.append(from_value)
        if to:
            where.append("created_at <= ?")
            params.append(to)
        where_sql = " AND ".join(where)
        rows = db.conn.execute(
            f"""
            SELECT id, type, content, score, created_at, updated_at
            FROM memories WHERE {where_sql}
            ORDER BY created_at ASC
            LIMIT 500
            """,
            params,
        ).fetchall()
    finally:
        db.close()
    events = [
        {
            "id": r["id"],
            "type": r["type"],
            "content": _short(r["content"], 120),
            "score": r["score"],
            "start": r["created_at"],
            "end": r["updated_at"],
        }
        for r in rows
    ]
    return {"events": events, "count": len(events)}


def _ui_graph(
    request: "Request",  # type: ignore[name-defined]
    seed: Optional[str] = None,
    depth: int = 2,
    limit: int = 50,
):
    """Cytoscape-friendly JSON for a memory neighborhood (or top-N corpus).

    Content-negotiated: serves the ``graph.html`` page to browsers and JSON
    to API clients (the page consumes the same endpoint via ``fetch``).
    """
    accept = request.headers.get("accept", "") if request else ""
    wants_html = (
        "text/html" in accept
        and "application/json" not in accept
        and request.headers.get("hx-request", "").lower() != "true"
    )
    if wants_html and not seed:
        # Only redirect bare GET /ui/graph (browser sidebar nav) to the page.
        # When ``seed`` is supplied we always return JSON so the page's
        # ``fetch`` works regardless of accept headers.
        return _templates().TemplateResponse(
            request, "graph.html", {"title": "Graph"},
        )
    depth = max(1, min(3, int(depth or 1)))
    limit = max(1, min(200, int(limit or 50)))
    db = _get_db()
    try:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        def _add_node(mid: str, mtype: str, content: str, score: float | None = None):
            if mid in nodes:
                return
            nodes[mid] = {
                "data": {
                    "id": mid,
                    "label": _short(content or "", 40),
                    "type": mtype or "memory",
                    "score": score or 0.0,
                }
            }

        if seed:
            seed_row = db.conn.execute(
                "SELECT id, type, content, score FROM memories WHERE id = ?",
                (seed,),
            ).fetchone()
            if not seed_row:
                return {"nodes": [], "edges": [], "error": "seed not found"}
            _add_node(
                seed_row["id"], seed_row["type"], seed_row["content"],
                seed_row["score"],
            )
            # BFS over memory_links so we can return edges as well as nodes
            # (zettelkasten.get_neighbors only returns the neighbor set).
            frontier = [(seed, 0)]
            visited = {seed}
            while frontier:
                current, d = frontier.pop(0)
                if d >= depth:
                    continue
                rows = db.conn.execute(
                    """
                    SELECT ml.target_memory_id AS tid, ml.similarity, ml.reason,
                           m.type, m.content, m.score
                    FROM memory_links ml
                    JOIN memories m ON m.id = ml.target_memory_id
                    WHERE ml.source_memory_id = ?
                      AND m.archived_at IS NULL
                    """,
                    (current,),
                ).fetchall()
                for r in rows:
                    tid = r["tid"]
                    _add_node(tid, r["type"], r["content"], r["score"])
                    edges.append({
                        "data": {
                            "source": current,
                            "target": tid,
                            "weight": r["similarity"] or 0.0,
                            "reason": r["reason"] or "semantic",
                        }
                    })
                    if tid not in visited:
                        visited.add(tid)
                        frontier.append((tid, d + 1))
        else:
            rows = db.conn.execute(
                """
                SELECT id, type, content, score
                FROM memories WHERE archived_at IS NULL
                ORDER BY score DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for r in rows:
                _add_node(r["id"], r["type"], r["content"], r["score"])
            ids = list(nodes.keys())
            if ids:
                placeholders = ",".join("?" * len(ids))
                try:
                    link_rows = db.conn.execute(
                        f"""
                        SELECT source_memory_id, target_memory_id, similarity, reason
                        FROM memory_links
                        WHERE source_memory_id IN ({placeholders})
                          AND target_memory_id IN ({placeholders})
                        """,
                        (*ids, *ids),
                    ).fetchall()
                    for r in link_rows:
                        edges.append({
                            "data": {
                                "source": r["source_memory_id"],
                                "target": r["target_memory_id"],
                                "weight": r["similarity"] or 0.0,
                                "reason": r["reason"] or "semantic",
                            }
                        })
                except Exception:
                    # memory_links table may not exist on a fresh DB pre-002
                    pass
    finally:
        db.close()
    return {"nodes": list(nodes.values()), "edges": edges}


def _ui_search(request: "Request", q: Optional[str] = None):  # type: ignore[name-defined]
    """Search box page; results stream from ``/context/stream`` via SSE."""
    return _templates().TemplateResponse(
        request, "search.html",
        {"q": q or "", "title": "Search"},
    )


# ============================================================================
# Conflict resolution (P5-2)
# ============================================================================

#: Status filter values exposed in the UI dropdown. The "all" sentinel
#: triggers a status-less query in the engine helper.
_CONFLICT_STATUS_FILTERS = (
    "pending",
    "resolved_keep_a",
    "resolved_keep_b",
    "resolved_keep_both",
    "resolved_merge",
    "dismissed",
    "all",
)


def _ui_conflicts_list(
    request: "Request",  # type: ignore[name-defined]
    status: Optional[str] = "pending",
    limit: int = 50,
):
    """List of contradictions detected by the sleep-time job.

    The page renders a small filter form + table. Each row links to the
    detail view where the user picks an action.
    """
    from ...engine.conflicts import list_conflicts

    status = status or "pending"
    if status not in _CONFLICT_STATUS_FILTERS:
        status = "pending"
    effective_status: Optional[str] = None if status == "all" else status
    limit = max(1, min(200, int(limit or 50)))

    db = _get_db()
    try:
        rows = list_conflicts(db, status=effective_status, limit=limit)
    finally:
        db.close()

    items = [
        {
            "id": r["id"],
            "memory_a_id": r["memory_a_id"],
            "memory_b_id": r["memory_b_id"],
            "similarity": r["similarity"],
            "detected_at": r["detected_at"],
            "detector": r["detector"],
            "reason": _short(r["reason"] or "", 100),
            "status": r["status"],
            "a_summary": _short(r["a_content"] or "", 90),
            "b_summary": _short(r["b_content"] or "", 90),
            "a_type": r["a_type"],
            "b_type": r["b_type"],
        }
        for r in rows
    ]

    return _templates().TemplateResponse(
        request, "conflicts.html",
        {
            "items": items,
            "total": len(items),
            "status": status,
            "statuses": _CONFLICT_STATUS_FILTERS,
            "title": "Conflicts",
        },
    )


def _ui_conflict_detail(request: "Request", conflict_id: int):  # type: ignore[name-defined]
    """Side-by-side diff for a single conflict + resolution form."""
    from fastapi.responses import HTMLResponse
    from ...engine.conflicts import RESOLVE_ACTIONS, get_conflict

    db = _get_db()
    try:
        row = get_conflict(db, int(conflict_id))
    finally:
        db.close()
    if not row:
        return HTMLResponse(
            "<p class='text-red-600'>conflict not found</p>", status_code=404,
        )

    diff_html = _render_diff(row.get("a_content") or "", row.get("b_content") or "")

    return _templates().TemplateResponse(
        request, "_conflict_diff.html",
        {
            "conflict": row,
            "diff_html": diff_html,
            "actions": RESOLVE_ACTIONS,
            "title": f"Conflict #{row['id']}",
        },
    )


async def _ui_conflict_resolve(request: "Request", conflict_id: int):  # type: ignore[name-defined]
    """Apply a resolution action submitted via HTMX form post."""
    from fastapi.responses import HTMLResponse
    from urllib.parse import parse_qs

    from ...engine.conflicts import RESOLVE_ACTIONS, resolve_conflict

    body = await request.body()
    action = ""
    notes = ""
    ctype = request.headers.get("content-type", "")
    try:
        if "application/json" in ctype:
            data = _json.loads(body or b"{}")
            action = (data.get("action") or "").strip()
            notes = (data.get("notes") or "").strip()
        else:
            parsed = parse_qs((body or b"").decode("utf-8"), keep_blank_values=True)
            action = (parsed.get("action", [""])[0] or "").strip()
            notes = (parsed.get("notes", [""])[0] or "").strip()
    except Exception:
        action, notes = "", ""

    if action not in RESOLVE_ACTIONS:
        return HTMLResponse(
            f"<p class='text-red-600'>invalid action {action!r}</p>", status_code=400,
        )

    db = _get_db()
    try:
        try:
            report = resolve_conflict(
                db, int(conflict_id), action=action, notes=notes or None,
            )
        except ValueError as e:
            return HTMLResponse(
                f"<p class='text-red-600'>{e}</p>", status_code=400,
            )
    finally:
        db.close()

    archived_str = ", ".join(a[:14] for a in report["archived"]) or "—"
    new_str = (report["new_memory_id"] or "—")[:14]
    return HTMLResponse(
        "<div class='text-emerald-700 text-sm'>"
        f"resolved as <strong>{report['status']}</strong>"
        f" · archived: <code>{archived_str}</code>"
        f" · new memory: <code>{new_str}</code>"
        "</div>"
    )


def _render_diff(a: str, b: str) -> str:
    """Render a tailwind-friendly side-by-side diff for two text blobs.

    Uses :class:`difflib.HtmlDiff` and rewrites the inline styles into
    minimal class hooks so the table inherits the surrounding tailwind
    typography.
    """
    import difflib

    a_lines = (a or "").splitlines() or [""]
    b_lines = (b or "").splitlines() or [""]
    diff = difflib.HtmlDiff(wrapcolumn=72)
    table = diff.make_table(
        a_lines, b_lines,
        fromdesc="Memory A", todesc="Memory B",
        context=False,
    )
    table = table.replace('nowrap="nowrap"', "")
    return table

def _ui_snapshots_list(request: "Request"):  # type: ignore[name-defined]
    """HTMX page: list snapshots + create form."""
    import os as _os
    snaps = _snap.list_snapshots(_state["db_path"])
    return _templates().TemplateResponse(
        request, "snapshots.html",
        {
            "snapshots": snaps,
            "db_path": str(_state["db_path"]),
            "auto_cadence": _os.environ.get("MEMOIRS_AUTO_SNAPSHOT", "off"),
            "keep": _os.environ.get("MEMOIRS_SNAPSHOT_KEEP", "10"),
        },
    )


def _ui_snapshots_create(request: "Request", name: Optional[str] = None):  # type: ignore[name-defined]
    """HTMX endpoint: create a snapshot, return updated table body fragment."""
    _snap.create(_state["db_path"], name=name)
    snaps = _snap.list_snapshots(_state["db_path"])
    return _templates().TemplateResponse(
        request, "_snap_table.html",
        {"snapshots": snaps},
    )


def _ui_snapshots_diff(request: "Request",  # type: ignore[name-defined]
                        a: str, b: str = "live"):
    """Side-by-side rich diff: counts + content of added/removed/changed rows.

    HTMX requests get the bare fragment (so it slots into the snapshots page
    without flashing the full layout). Direct browser navigation gets the
    fragment wrapped in the standard sidebar layout.
    """
    db_p = _state["db_path"]
    a_path = db_p if a == "live" else Path(a)
    b_path = db_p if b == "live" else Path(b)
    d = _snap.diff(a_path, b_path)

    SHOW = 20
    added_rows   = _fetch_memorias_by_id(b_path, d["added"][:SHOW])
    removed_rows = _fetch_memorias_by_id(a_path, d["removed"][:SHOW])
    changed_a    = _fetch_memorias_by_id(a_path, d["changed"][:SHOW])
    changed_b    = _fetch_memorias_by_id(b_path, d["changed"][:SHOW])

    is_htmx = request.headers.get("hx-request", "").lower() == "true"
    template = "_snap_diff.html" if is_htmx else "snapshots_diff_page.html"

    return _templates().TemplateResponse(
        request, template,
        {
            "a": a, "b": b,
            "a_count": d["a_count"], "b_count": d["b_count"],
            "added":   list(added_rows.values()),
            "removed": list(removed_rows.values()),
            "changed": [
                {"a": changed_a.get(k), "b": changed_b.get(k)}
                for k in d["changed"][:SHOW]
            ],
            "n_added": len(d["added"]), "n_removed": len(d["removed"]),
            "n_changed": len(d["changed"]),
            "shown_cap": SHOW,
        },
    )


def _ui_snapshots_restore(request: "Request", path: str):  # type: ignore[name-defined]
    from fastapi.responses import HTMLResponse
    info = _snap.restore(path, _state["db_path"])
    return HTMLResponse(
        f"<p class='text-emerald-700'>restored from <code>{info.path}</code> "
        f"(safety snapshot was created first)</p>"
    )


def _fetch_memorias_by_id(path: Path, ids: list[str]) -> dict[str, dict]:
    """Pull the full row for a list of memoria ids out of any snapshot DB."""
    if not ids:
        return {}
    c = _sqlite.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = _sqlite.Row
    try:
        placeholders = ",".join("?" * len(ids))
        rows = c.execute(
            f"SELECT id, type, content, importance, created_at "
            f"FROM memories WHERE id IN ({placeholders})", ids,
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}
    finally:
        c.close()


def _render_snapshot_row(s) -> str:
    return (
        f"<tr class=\"border-t border-slate-200\">"
        f"<td class=\"px-3 py-2 text-slate-600\">{s.created_at}</td>"
        f"<td class=\"px-3 py-2\">{s.name}</td>"
        f"<td class=\"px-3 py-2 text-right\">{s.memory_count}</td>"
        f"<td class=\"px-3 py-2 text-right\">{s.size_bytes:,}</td>"
        f"<td class=\"px-3 py-2 space-x-2\">"
        f"<a class=\"text-indigo-600 hover:underline\" "
        f"hx-get=\"/ui/snapshots/diff?a={s.path}&b=live\" hx-target=\"#snap-diff\">diff vs live</a>"
        f"<a class=\"text-rose-600 hover:underline\" "
        f"hx-post=\"/ui/snapshots/restore?path={s.path}\" hx-target=\"#snap-msg\" "
        f"hx-confirm=\"Restore live DB from {s.name}? A safety snapshot will be created first.\">"
        f"restore</a>"
        f"</td></tr>"
    )
