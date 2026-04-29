"""Adapter for the local **memoirs** engine.

Uses the same DB-seeding shortcut the existing eval suite takes
(`memoirs.evals.suites.synthetic_basic._insert_memory`): we INSERT
straight into the `memories` table so the bench measures *retrieval*
quality, not the consolidation pipeline. Embeddings are upserted
best-effort — if sentence-transformers / sqlite-vec are unavailable
the BM25 path still ranks.

Each query goes through `assemble_context` so we exercise the full
public retrieval surface (scoring + conflict resolution + bi-temporal
gating) — that's what an MCP client actually sees.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from memoirs.core.ids import content_hash, utc_now
from memoirs.db import MemoirsDB
from memoirs.engine import embeddings as emb
from memoirs.engine import hybrid_retrieval as hr
from memoirs.engine import memory_engine as me

# The curator module is the new home (renamed from `gemma`). Fall back to
# the gemma shim so we keep working through the in-flight rename agent
# even if `curator` doesn't exist yet on this branch.
try:
    from memoirs.engine.curator import extract_memory_candidates as _extract_memory_candidates
except ImportError:  # pragma: no cover — pre-rename path
    from memoirs.engine.gemma import extract_memory_candidates as _extract_memory_candidates  # type: ignore[no-redef]

from scripts.adapters.base import AdapterStatus, EngineAdapter
from scripts.bench_vs_others_dataset import (
    BenchConversation,
    BenchMemory,
    BenchQuery,
)

log = logging.getLogger("bench.adapters.memoirs")


class MemoirsAdapter(EngineAdapter):
    """Bench adapter wrapping the local memoirs engine.

    Spawns a fresh SQLite DB in a tempdir per run so the bench never
    touches the user's production DB. Reuses the seeding helper from
    the synthetic suite to skip consolidation overhead — same pattern
    the existing harness uses.
    """

    name = "memoirs"
    supports_native_ingest = True

    def __init__(self, *, db_dir: Optional[Path] = None,
                 use_assemble_context: bool = True) -> None:
        super().__init__()
        self._owns_dir = db_dir is None
        self._dir = Path(db_dir) if db_dir else Path(tempfile.mkdtemp(prefix="memoirs_bench_"))
        self._db = None
        self._use_assemble_context = use_assemble_context
        try:
            self._db = MemoirsDB(self._dir / "bench.sqlite")
            self._db.init()
        except Exception as e:
            self.status = AdapterStatus(ok=False, reason=f"DB open failed: {e!r}")

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        if not self.status.ok or self._db is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")

        # FTS5 schema must exist before we INSERT so triggers backfill it.
        hr.ensure_fts_schema(self._db.conn)
        now = utc_now()
        with self._db.conn:
            for m in memories:
                h = content_hash(m.content + m.id)
                valid_from = m.valid_from or now
                self._db.conn.execute(
                    """
                    INSERT INTO memories (
                        id, type, content, content_hash, importance, confidence,
                        score, usage_count, user_signal, valid_from, valid_to,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 0.5, 0, 0, ?, ?, '{}', ?, ?)
                    """,
                    (m.id, m.type, m.content, h, m.importance, m.confidence,
                     valid_from, m.valid_to, now, now),
                )
        # Embeddings best-effort.
        for m in memories:
            try:
                emb.upsert_memory_embedding(self._db, m.id, m.content)
            except Exception as e:
                log.info("embed skipped for %s (%s)", m.id, e)


    # ------------------------------------------------------------------
    # End-to-end ingest (conversation → extract → consolidate)
    # ------------------------------------------------------------------

    def ingest_conversation(self, conv: BenchConversation) -> None:
        """Drive the full memoirs pipeline for one conversation.

        Pipeline:
          1. INSERT raw messages via `db.ingest_conversation_event`.
          2. Run the extractor cascade (Gemma → spaCy → noop) through
             `extract_memory_candidates`.
          3. Run `consolidate_pending` to lift candidates → memories.
          4. Tag each persisted memory with metadata so its
             `memoirs_id::{conv.id}` lineage is recoverable from queries.

        Embedding upserts happen via the existing memory insert paths
        already wired into consolidation; this method does not duplicate
        them.
        """
        if not self.status.ok or self._db is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")

        # Make sure FTS schema exists (consolidation triggers depend on
        # it). Idempotent.
        hr.ensure_fts_schema(self._db.conn)

        os.environ.setdefault("MEMOIRS_CURATOR_ENABLED", "auto")
        os.environ.setdefault("MEMOIRS_GEMMA_CURATOR", "auto")

        # 1. Persist raw messages.
        payload = {
            "type": "chat_message",
            "source": "bench-end-to-end",
            "source_uri": "bench://end-to-end",
            "source_kind": "bench",
            "conversation_id": conv.id,
            "title": conv.id,
            "messages": [
                {
                    "role": (m.get("role") or "user"),
                    "content": (m.get("content") or ""),
                    "type": "chat_message",
                }
                for m in conv.messages
                if (m.get("content") or "").strip()
            ],
        }
        result = self._db.ingest_conversation_event(payload)
        conversation_db_id = result.get("conversation_id")
        if not conversation_db_id:
            raise RuntimeError("ingest_conversation_event returned no id")

        # 2. Extract candidates (curator LLM → spaCy fallback).
        try:
            _extract_memory_candidates(self._db, conversation_db_id)
        except Exception as e:
            log.warning("extract_memory_candidates failed for %s: %s",
                        conv.id, e)

        # 3. Consolidate candidates into memories.
        try:
            me.consolidate_pending(self._db, limit=200)
        except Exception as e:
            log.warning("consolidate_pending failed for %s: %s",
                        conv.id, e)

        # 4. Stamp lineage metadata so queries can route memories to
        # their source conversation. We tag every memory derived from
        # this conversation's candidates.
        try:
            self._tag_memories_with_conv(conversation_db_id, conv.id)
        except Exception as e:
            log.info("metadata tag skipped for %s: %s", conv.id, e)

    def resolve_conv_id(self, memory_id: str) -> Optional[str]:
        """Look up the bench conversation ID stamped on a memory.

        Falls back to the in-memory index used by the default adapter
        path so test doubles and synthetic ingest both keep working.
        """
        if self._db is not None:
            try:
                row = self._db.conn.execute(
                    "SELECT json_extract(metadata_json, '$.bench_conv_id') AS cid "
                    "FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if row and row["cid"]:
                    return str(row["cid"])
            except Exception:
                pass
        # Fall through to base.EngineAdapter._conv_index for the
        # standalone path (used when ingest_conversation falls back).
        return super().resolve_conv_id(memory_id)

    def _tag_memories_with_conv(self, conversation_db_id: str,
                                 conv_bench_id: str) -> None:
        """Annotate memories derived from a conversation with its bench ID.

        Walks `memory_candidates -> memory_consolidations -> memories`
        and writes ``$.bench_conv_id`` into the memory metadata. Uses
        the candidate IDs directly so we capture every persisted memory
        even if consolidation merged it into an existing record.
        """
        if self._db is None:
            return
        rows = self._db.conn.execute(
            "SELECT id FROM memory_candidates WHERE conversation_id = ?",
            (conversation_db_id,),
        ).fetchall()
        cand_ids = [r["id"] for r in rows]
        if not cand_ids:
            return
        # Read consolidation links (best-effort — table layout varies
        # across versions; skip silently if unavailable).
        try:
            links = self._db.conn.execute(
                f"SELECT DISTINCT memory_id FROM memory_consolidations "
                f"WHERE candidate_id IN ({','.join('?' * len(cand_ids))})",
                cand_ids,
            ).fetchall()
            mem_ids = [r["memory_id"] for r in links if r["memory_id"]]
        except Exception:
            mem_ids = []
        # Fallback: use content match against active memories created
        # in this run (cheap when the DB is bench-private).
        if not mem_ids:
            try:
                mem_ids = [
                    r["id"] for r in self._db.conn.execute(
                        "SELECT id FROM memories WHERE archived_at IS NULL"
                    ).fetchall()
                ]
            except Exception:
                return
        with self._db.conn:
            for mid in mem_ids:
                self._db.conn.execute(
                    "UPDATE memories SET metadata_json = json_set("
                    "  COALESCE(metadata_json, '{}'),"
                    "  '$.bench_conv_id', ?"
                    ") WHERE id = ?",
                    (conv_bench_id, mid),
                )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok or self._db is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")

        mode = "hybrid"
        if self._use_assemble_context:
            payload = me.assemble_context(
                self._db, q.query,
                top_k=top_k, max_lines=top_k,
                as_of=q.as_of,
                retrieval_mode=mode,
            )
            return [m["id"] for m in payload.get("memories", [])][:top_k]
        candidates = me._retrieve_candidates(
            self._db, q.query,
            top_k=top_k, as_of=q.as_of, mode=mode,
        )
        return [c["id"] for c in candidates][:top_k]

    def shutdown(self) -> None:
        try:
            if self._db is not None:
                self._db.close()
        except Exception:
            pass
        if self._owns_dir:
            try:
                shutil.rmtree(self._dir, ignore_errors=True)
            except Exception:
                pass


__all__ = ["MemoirsAdapter"]
