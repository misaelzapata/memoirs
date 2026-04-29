"""Common interface for every engine adapter used in `bench_vs_others`.

Each other engine memory engine speaks a slightly different language (HTTP,
Python SDK, gRPC, …). The adapter layer normalizes them down to four
methods so the bench runner doesn't grow per-engine branches.

Returned IDs MUST be the same string IDs the dataset uses — adapters
are responsible for mapping engine-internal IDs back when the engine
generates its own.

Two ingestion paths are supported:

1. ``add_memories(memories)`` — pre-curated memory records. Used by the
   default *retrieval-only* bench suite. Adapters that don't extract
   their own memories simply INSERT each record.
2. ``ingest_conversation(conv)`` — raw chat messages. Used by the
   *end-to-end* suite that measures the FULL pipeline cost (extract +
   consolidate + retrieve). Default implementation falls back to the
   memory path so adapters that don't yet implement extraction still
   produce numbers (with a clearly marked ``mode="standalone"``).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Sequence

from scripts.bench_vs_others_dataset import (
    BenchConversation,
    BenchMemory,
    BenchQuery,
)


@dataclass
class AdapterStatus:
    """Outcome of an adapter's lifecycle for one bench run.

    `ok` flips to False whenever something prevented the engine from
    serving the queries (Docker missing, package missing, API timeout,
    etc.). `reason` is surfaced verbatim in the markdown table so a
    skipped row tells the reader exactly *why* it skipped.
    """

    ok: bool = True
    reason: str = ""
    # Optional process metadata (PID, container ID, etc.) — adapters can
    # populate this for diagnostics but the runner doesn't depend on it.
    meta: dict = field(default_factory=dict)


class EngineAdapter(abc.ABC):
    """Minimal contract every adapter implements.

    Lifecycle:
      1. ``__init__`` decides whether the engine can run at all (e.g.
         checks for Docker, env vars, importable Python packages).
         Failures should populate ``self.status`` rather than raising —
         the runner uses ``status.ok`` to decide whether to call
         ``add_memories`` / ``query``.
      2. ``add_memories(memories)`` ingests the full corpus. Should be
         idempotent if possible; the runner only calls it once.
      3. ``query(q, top_k)`` returns ranked memory IDs. Each call is
         timed by the runner; adapters should NOT re-ingest inside
         query.
      4. ``shutdown()`` releases resources (closes DB, stops container).
         Always invoked, even after add/query failures.
    """

    name: str = "engine"

    #: True when the adapter implements its own ``ingest_conversation``
    #: (i.e. runs the engine's extract+consolidate pipeline). Adapters
    #: that fall back to the memory-standalone path leave this as False
    #: so the bench can mark the row ``mode="standalone"``.
    supports_native_ingest: bool = False

    def __init__(self) -> None:
        self.status = AdapterStatus(ok=True, reason="")
        # End-to-end metrics — populated by the runner.
        self.tokens_used: int = 0
        self.total_ingest_seconds: float = 0.0
        self.peak_ram_mb: float = 0.0

    @abc.abstractmethod
    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        """Ingest the full corpus. May raise; runner catches and skips."""

    @abc.abstractmethod
    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        """Run one query and return the top-`k` ranked memory IDs."""

    def ingest_conversation(self, conv: BenchConversation) -> None:
        """Ingest a raw conversation through the engine's own pipeline.

        Default implementation: synthesize a `BenchMemory` per message
        and reuse `add_memories`. Adapters that implement extraction
        natively (memoirs, mem0, cognee) override this to drive their
        real pipeline and should set ``supports_native_ingest = True``.

        The synthetic IDs use ``{conv.id}::{idx}`` so the runner can
        attribute retrieved memories back to the source conversation
        through `resolve_conv_id`.
        """
        synthetic: list[BenchMemory] = []
        for idx, msg in enumerate(conv.messages):
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            mem_id = f"{conv.id}::{idx}"
            synthetic.append(BenchMemory(
                id=mem_id,
                type="event",
                content=content,
            ))
            # Default fallback path: register the synthetic ID with the
            # conv lookup so `resolve_conv_id` works for free.
            self._register_conv_link(mem_id, conv.id)
        if synthetic:
            self.add_memories(synthetic)

    # ------------------------------------------------------------------
    # End-to-end ID lookup
    #
    # Each adapter maintains a tiny lookup ``memory_id -> conv_id`` so
    # the bench runner can translate retrieved memory IDs back to the
    # gold conversation IDs in the end-to-end suite.
    # ------------------------------------------------------------------

    def _register_conv_link(self, memory_id: str, conv_id: str) -> None:
        if not hasattr(self, "_conv_index"):
            self._conv_index: dict[str, str] = {}
        self._conv_index[memory_id] = conv_id

    def resolve_conv_id(self, memory_id: str) -> Optional[str]:
        """Return the gold conversation ID for a retrieved memory ID.

        ``None`` when the memory cannot be traced — the runner counts
        unresolved IDs as misses.
        """
        index = getattr(self, "_conv_index", None)
        if not index:
            return None
        return index.get(memory_id)

    def shutdown(self) -> None:
        """Tear down resources. Default: no-op."""
        return None


__all__ = ["AdapterStatus", "EngineAdapter"]
