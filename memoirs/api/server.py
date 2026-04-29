"""REST API for memoirs — FastAPI + auto-generated OpenAPI.

Endpoints:
  POST /memories                    — add a memory (manual)
  GET  /memories?type=&limit=       — list memorias
  GET  /memories/{id}               — single memory
  DELETE /memories/{id}             — archive (soft delete)
  POST /memories/{id}/feedback      — score feedback (useful=bool)

  POST /search                      — similarity search
  POST /context                     — assemble compact context for query
  GET  /context/stream              — SSE streaming variant (meta/memory/context/done)
  POST /summarize/{conv_id}         — Gemma-summarize a conversation

  GET  /projects                    — list project entities
  GET  /projects/{name}/context     — entities + memorias for a project

  POST /maintenance                 — run lifecycle pass
  POST /audit                       — Gemma curation pass
  POST /cleanup                     — auto-merge near-duplicates

  GET  /healthz                     — liveness check
  GET  /metrics                     — counts + version

  GET  /docs                        — Swagger UI (auto-generated)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..config import DEFAULT_DB_PATH
from ..db import MemoirsDB


log = logging.getLogger("memoirs.api")


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
        import pydantic  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "FastAPI not installed. Run: pip install -e '.[api]'"
        ) from e


def _sse_pack(event: str, data: dict) -> bytes:
    """Encode an SSE frame: `event: <name>\\ndata: <json>\\n\\n`."""
    import json as _json
    payload = _json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _build_app(db_path: Path):
    _require_fastapi()
    from fastapi import Body, FastAPI, HTTPException, Path as PathParam, Query
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="Memoirs",
        version=__version__,
        description="Local-first memory engine for AI agents. "
                    "MCP-compatible. Use /docs for the Swagger UI.",
    )
    state = {"db_path": Path(db_path)}

    def _get_db() -> MemoirsDB:
        db = MemoirsDB(state["db_path"])
        db.init()
        return db

    # ----- pydantic models ---------------------------------------------------

    class MemoryCreate(BaseModel):
        type: str = Field(..., description="One of preference/fact/project/task/decision/style/credential_pointer/procedural")
        content: str
        importance: int = Field(4, ge=1, le=5)
        confidence: float = Field(0.95, ge=0.0, le=1.0)

    class FeedbackBody(BaseModel):
        useful: bool

    class SearchBody(BaseModel):
        query: str
        limit: int = 10
        as_of: Optional[str] = None

    class ContextBody(BaseModel):
        query: str
        top_k: int = 20
        max_lines: int = 15
        as_of: Optional[str] = None

    # ----- raw layer ---------------------------------------------------------

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/metrics")
    def metrics():
        db = _get_db()
        try:
            return {
                "version": __version__,
                "counts": db.status(),
                "memories": db.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE archived_at IS NULL"
                ).fetchone()[0],
                "embeddings": db.conn.execute(
                    "SELECT COUNT(*) FROM memory_embeddings"
                ).fetchone()[0],
                "entities": db.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
                "relationships": db.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0],
            }
        finally:
            db.close()

    # ----- memorias ----------------------------------------------------------

    @app.post("/memories", status_code=201)
    def add_memory(body: MemoryCreate):
        from ..engine import memory_engine
        from ..engine.curator import Candidate, validate_allowed_memory_type
        cand = Candidate(type=body.type, content=body.content,
                         importance=body.importance, confidence=body.confidence)
        if not validate_allowed_memory_type(cand):
            raise HTTPException(400, f"invalid type: {body.type}")
        db = _get_db()
        try:
            decision = memory_engine.decide_memory_action(db, cand)
            return memory_engine.apply_decision(db, cand, decision)
        finally:
            db.close()

    @app.get("/memories")
    def list_memories(type: Optional[str] = None, limit: int = 50):
        db = _get_db()
        try:
            if type:
                rows = db.conn.execute(
                    "SELECT id, type, content, score, importance, confidence "
                    "FROM memories WHERE archived_at IS NULL AND type = ? "
                    "ORDER BY score DESC LIMIT ?", (type, limit),
                ).fetchall()
            else:
                rows = db.conn.execute(
                    "SELECT id, type, content, score, importance, confidence "
                    "FROM memories WHERE archived_at IS NULL "
                    "ORDER BY score DESC LIMIT ?", (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            db.close()

    @app.get("/memories/{memory_id}")
    def get_memory(memory_id: str = PathParam(..., description="Memory id")):
        db = _get_db()
        try:
            row = db.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not row:
                raise HTTPException(404, "memory not found")
            return dict(row)
        finally:
            db.close()

    @app.delete("/memories/{memory_id}")
    def forget_memory(memory_id: str):
        from ..core.ids import utc_now
        db = _get_db()
        try:
            db.conn.execute(
                "UPDATE memories SET archived_at = ?, "
                "  archive_reason = 'API forget' WHERE id = ?",
                (utc_now(), memory_id),
            )
            db.conn.commit()
            return {"ok": True, "memory_id": memory_id}
        finally:
            db.close()

    @app.post("/memories/{memory_id}/feedback")
    def memory_feedback(memory_id: str, body: FeedbackBody):
        from ..core.ids import utc_now
        from ..engine.memory_engine import calculate_memory_score
        delta = 0.2 if body.useful else -0.2
        db = _get_db()
        try:
            db.conn.execute(
                "UPDATE memories SET "
                "  user_signal = MAX(0.0, MIN(1.0, COALESCE(user_signal,0) + ?)), "
                "  updated_at = ? WHERE id = ? AND archived_at IS NULL",
                (delta, utc_now(), memory_id),
            )
            row = db.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not row:
                raise HTTPException(404, "memory not found")
            new_score = calculate_memory_score(dict(row))
            db.conn.execute(
                "UPDATE memories SET score = ? WHERE id = ?", (new_score, memory_id),
            )
            db.conn.commit()
            return {"memory_id": memory_id, "new_score": new_score, "user_signal": row["user_signal"]}
        finally:
            db.close()

    # ----- retrieval ---------------------------------------------------------

    @app.post("/search")
    def search(body: SearchBody):
        from ..engine.embeddings import search_similar_memories
        db = _get_db()
        try:
            return search_similar_memories(db, body.query, top_k=body.limit, as_of=body.as_of)
        finally:
            db.close()

    @app.post("/context")
    def context(body: ContextBody):
        from ..engine.memory_engine import assemble_context
        db = _get_db()
        try:
            return assemble_context(db, body.query, top_k=body.top_k,
                                     max_lines=body.max_lines, as_of=body.as_of)
        finally:
            db.close()

    @app.get("/context/stream")
    async def context_stream(
        q: str = Query(..., description="Query text"),
        top_k: int = Query(20, ge=1, le=100),
        max_lines: int = Query(15, ge=1, le=100),
        as_of: Optional[str] = Query(None, description="ISO timestamp for time-travel"),
    ):
        """SSE stream of `assemble_context`.

        Event order on the wire:
          event: meta     — emitted immediately (TTFT < 100ms typical)
          event: memory   — once per ranked memory, in final order
          event: context  — full payload (lines + memories + token_estimate)
          event: done     — terminator; clients can close

        Headers disable buffering at common reverse proxies (nginx,
        Cloudflare) and CDNs so events flush as soon as they're yielded.

        Cancellation: Starlette closes this async generator (raises
        GeneratorExit) when the client disconnects, which unwinds the
        try/finally below — the underlying generator stops on the next
        iteration and `db.close()` runs. No manual disconnect polling
        needed.
        """
        from ..engine.memory_engine import assemble_context_stream

        async def event_source():
            import anyio
            db = _get_db()
            try:
                gen = assemble_context_stream(
                    db, q, top_k=top_k, max_lines=max_lines, as_of=as_of,
                )
                for event_name, data in gen:
                    yield _sse_pack(event_name, data)
                    # Cooperative checkpoint: forces a transport flush so
                    # `meta` lands on the wire before retrieval finishes,
                    # and lets Starlette deliver disconnect → GeneratorExit.
                    await anyio.sleep(0)
                yield _sse_pack("done", {})
            finally:
                db.close()

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.post("/summarize/{conversation_id}")
    def summarize(conversation_id: str):
        from ..engine.curator import summarize_conversation
        db = _get_db()
        try:
            return summarize_conversation(db, conversation_id)
        finally:
            db.close()

    # ----- projects + KG -----------------------------------------------------

    @app.get("/projects")
    def list_projects():
        db = _get_db()
        try:
            rows = db.conn.execute(
                "SELECT id, name FROM entities WHERE type = 'project' ORDER BY name"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            db.close()

    @app.get("/projects/{name}/context")
    def project_context(name: str, limit: int = 20):
        from ..engine.graph import get_project_context
        db = _get_db()
        try:
            return get_project_context(db, name, limit=limit)
        finally:
            db.close()

    # ----- maintenance / curation -------------------------------------------

    @app.post("/maintenance")
    def maintenance():
        from ..engine.memory_engine import run_daily_maintenance
        db = _get_db()
        try:
            return run_daily_maintenance(db)
        finally:
            db.close()

    @app.post("/audit")
    def audit(limit: int = Query(30), apply: bool = Query(False), type: Optional[str] = Query(None)):
        from ..engine.audit import audit_corpus
        db = _get_db()
        try:
            return audit_corpus(db, limit=limit, apply=apply, type_filter=type)
        finally:
            db.close()

    @app.post("/cleanup")
    def cleanup(threshold: float = Query(0.92), dry_run: bool = Query(True)):
        from ..engine.lifecycle import auto_merge_near_duplicates
        db = _get_db()
        try:
            return auto_merge_near_duplicates(db, threshold=threshold, dry_run=dry_run)
        finally:
            db.close()

    # ----- /ui — local web inspector (P5-1 + P5-3) ---------------------------
    # Mounted as a sibling router; no impact on /context, /context/stream, or
    # any other existing endpoint. Failures during mount are logged but never
    # break the API.
    try:
        from .ui import mount_ui
        mount_ui(app, state["db_path"])
    except Exception as ui_err:  # pragma: no cover -- defensive
        log.warning("UI router not mounted: %s", ui_err)

    # ``/`` serves the dashboard directly (the API has no root handler so
    # there's no collision). All other UI pages stay under ``/ui/*`` to
    # avoid colliding with the JSON REST endpoints (``/memories``,
    # ``/search``, ``/context``, etc).
    from .ui import _ui_index
    app.get("/", include_in_schema=False)(_ui_index)

    # ----- trace-id middleware (P0-6) ----------------------------------------
    # Honors an inbound `X-Trace-Id` header (so callers can correlate across
    # services) and generates a fresh id when absent. The id is bound to a
    # contextvar via `with_trace_context`, so any logger inside the request
    # automatically tags its records. The same id is echoed back on the
    # response so clients can grep their own logs.
    from ..observability import with_trace_context as _with_trace_ctx

    @app.middleware("http")
    async def _trace_id_middleware(request, call_next):  # type: ignore[no-redef]
        inbound = request.headers.get("x-trace-id")
        with _with_trace_ctx(trace_id=inbound) as (tid, _sid):
            response = await call_next(request)
            response.headers["X-Trace-Id"] = tid
            return response

    return app


# Module-level app for `uvicorn memoirs.api:app`
app = None


def run(db_path: Path | str = DEFAULT_DB_PATH, host: str = "127.0.0.1", port: int = 8283):
    global app
    _require_fastapi()
    import uvicorn
    app = _build_app(Path(db_path))
    uvicorn.run(app, host=host, port=port, log_level="info")
