"""Adapter for **Mem0** (external memory engine) via the official Python library (``pip install mem0ai``).

Replaces the previous Docker-based adapter. The ``mem0ai/mem0:latest``
public image was renamed/removed (`pull access denied`), so we drive
Mem0 directly through ``from mem0 import Memory``. Same OPENAI_API_KEY
contract — Mem0 needs OpenAI for extraction + embedding.

For Ollama mode (``MEMOIRS_USE_OLLAMA=on``) we configure Mem0's LLM and
embedder dicts to point at the local Ollama daemon's OpenAI-compatible
endpoint instead of OpenAI cloud.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Sequence

from scripts.adapters._ollama import (
    OLLAMA_OPENAI_BASE_URL,
    apply_ollama_env,
    ollama_install_hint,
    ollama_is_up,
    use_ollama_requested,
)
from scripts.adapters.base import AdapterStatus, EngineAdapter
from scripts.bench_vs_others_dataset import (
    BenchConversation,
    BenchMemory,
    BenchQuery,
)

log = logging.getLogger("bench.adapters.mem0")


class Mem0Adapter(EngineAdapter):
    name = "mem0"
    supports_native_ingest = True

    def __init__(self, *, user_id: str = "bench-user", **_unused) -> None:
        super().__init__()
        self._user_id = user_id
        self._memory = None
        self._id_map: dict[str, str] = {}
        self._reverse_map: dict[str, str] = {}
        self._use_ollama = use_ollama_requested()

        if self._use_ollama:
            if not ollama_is_up():
                self.status = AdapterStatus(ok=False, reason=ollama_install_hint())
                return
            apply_ollama_env()
        elif not os.environ.get("OPENAI_API_KEY"):
            self.status = AdapterStatus(ok=False, reason="OPENAI_API_KEY missing")
            return

        try:
            from mem0 import Memory  # type: ignore
        except ImportError as e:
            self.status = AdapterStatus(
                ok=False, reason=f"mem0ai not installed (pip install mem0ai): {e!r}",
            )
            return

        try:
            config = self._build_config()
            self._memory = Memory.from_config(config) if config else Memory()
        except Exception as e:
            self.status = AdapterStatus(ok=False, reason=f"Memory() init failed: {e!r}")
            self._memory = None
            return

        self.status.meta["backend"] = "mem0ai-python"
        self.status.meta["use_ollama"] = self._use_ollama

    def _build_config(self) -> dict | None:
        """Build a Mem0 config dict.

        Mem0 2.0.1 picks a recent OpenAI model by default that REJECTS
        ``max_tokens`` (it wants ``max_completion_tokens``), so the
        extraction silently fails and ``add()`` returns ``{"results": []}``.
        We pin ``gpt-4o-mini`` which still accepts the legacy field.
        Embedder uses ``text-embedding-3-small`` at its native 1536 dims.
        """
        if self._use_ollama:
            return {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": os.environ.get("OPENAI_MODEL", "qwen2.5:3b"),
                        "openai_base_url": OLLAMA_OPENAI_BASE_URL,
                        "api_key": "ollama",
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
                        "openai_base_url": OLLAMA_OPENAI_BASE_URL,
                        "api_key": "ollama",
                    },
                },
            }
        # Plain OpenAI cloud — pin a model with the legacy max_tokens API.
        return {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "embedding_dims": 1536,
                },
            },
        }

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        if not self.status.ok or self._memory is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        for m in memories:
            try:
                result = self._memory.add(
                    m.content,
                    user_id=self._user_id,
                    metadata={"bench_id": m.id, "type": m.type},
                )
                # Mem0's `.add` may return a dict or a list. Pull the first
                # event ID so we can map the engine's view back to bench IDs.
                eid = self._extract_id(result)
                if eid:
                    self._id_map[m.id] = eid
                    self._reverse_map[eid] = m.id
            except Exception as e:
                log.warning("mem0 add failed for %s: %s", m.id, e)

    def ingest_conversation(self, conv: BenchConversation) -> None:
        """Drive Mem0's native extraction by passing the full message list.

        Mem0's `Memory.add(messages=[...])` triggers its internal
        OpenAI-powered extraction. The returned event IDs are stored so
        we can map results back to the source conversation.
        """
        if not self.status.ok or self._memory is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        # Mem0 expects a list of {role, content} dicts.
        msgs = [
            {"role": (m.get("role") or "user"),
             "content": (m.get("content") or "")}
            for m in conv.messages
            if (m.get("content") or "").strip()
        ]
        if not msgs:
            return
        try:
            result = self._memory.add(
                messages=msgs,
                user_id=self._user_id,
                metadata={"bench_conv_id": conv.id},
            )
        except Exception as e:
            log.warning("mem0 ingest_conversation failed for %s: %s",
                        conv.id, e)
            return
        # Mem0 emits one or more events; index them all under conv.id so
        # the runner can resolve them back to gold.
        events: list[dict] = []
        if isinstance(result, dict):
            events = result.get("results") or result.get("memories") or []
        elif isinstance(result, list):
            events = result
        for ev in events:
            if not isinstance(ev, dict):
                continue
            eid = str(ev.get("id") or "")
            if eid:
                self._reverse_map[eid] = conv.id
                self._register_conv_link(eid, conv.id)

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok or self._memory is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        # Mem0 ≥ 2.0 moved scope kwargs into a `filters` dict; older
        # versions accept `user_id=` directly. Try the new API first.
        try:
            results = self._memory.search(
                query=q.query, filters={"user_id": self._user_id}, limit=top_k,
            )
        except (TypeError, ValueError):
            try:
                results = self._memory.search(
                    query=q.query, user_id=self._user_id, limit=top_k,
                )
            except Exception as e:
                log.warning("mem0 search failed for %r: %s", q.query, e)
                return []
        except Exception as e:
            log.warning("mem0 search failed for %r: %s", q.query, e)
            return []

        # Mem0 returns either a list[dict] or {"results": [...]}.
        items = results if isinstance(results, list) else (
            results.get("results", []) if isinstance(results, dict) else []
        )
        out: list[str] = []
        for item in items[:top_k]:
            md = (item.get("metadata") or {}) if isinstance(item, dict) else {}
            bid = md.get("bench_id")
            if bid:
                out.append(str(bid))
                continue
            eid = str(item.get("id", "")) if isinstance(item, dict) else ""
            # End-to-end suite: return the engine ID; the runner resolves
            # it via `resolve_conv_id` (populated during
            # ingest_conversation).
            if eid:
                if eid in self._reverse_map:
                    out.append(self._reverse_map[eid])
                else:
                    out.append(eid)
        return out

    def shutdown(self) -> None:
        # Memory() has no explicit close; best-effort: drop reference to free
        # the qdrant in-memory backend.
        self._memory = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_id(result) -> str | None:
        """Pull the first memory ID Mem0 emitted for an `add` call."""
        if isinstance(result, dict):
            results = result.get("results") or result.get("memories") or []
            if isinstance(results, list) and results:
                return str(results[0].get("id", "")) or None
            return str(result.get("id", "")) or None
        if isinstance(result, list) and result:
            return str(result[0].get("id", "")) or None
        return None


__all__ = ["Mem0Adapter"]
