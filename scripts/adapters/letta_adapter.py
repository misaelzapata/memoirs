"""Adapter for **Letta** (formerly MemGPT) self-hosted via Docker.

Letta's official image (`letta/letta:latest`) needs an LLM provider
configured (OpenAI / Anthropic / Ollama) to do anything meaningful —
it's an agent runtime, not a pure memory store. Standing it up for a
retrieval bench is heavyweight, so this adapter follows the same
"skip cleanly" pattern as Zep:

  * If `LETTA_BASE_URL` (or an explicit `base_url`) is set, hit it.
  * Otherwise mark the adapter as SKIP with a reason and let the
    runner record a `-` row in the table.

When wired up, we use Letta's archival_memory subsystem because it's
the closest analog to a flat retrieval index.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Sequence

from scripts.adapters._ollama import (
    apply_ollama_env,
    ollama_install_hint,
    ollama_is_up,
    use_ollama_requested,
)
from scripts.adapters.base import AdapterStatus, EngineAdapter
from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery

log = logging.getLogger("bench.adapters.letta")


class LettaAdapter(EngineAdapter):
    name = "letta"

    def __init__(self, *, base_url: Optional[str] = None,
                 timeout: float = 60.0) -> None:
        super().__init__()
        # Always initialise mutable attrs first so shutdown() is safe
        # even if we abort the constructor early.
        self._agent_id: Optional[str] = None
        self._content_to_id: dict[str, str] = {}
        self._base_url: Optional[str] = None
        # Configure the OpenAI-compatible client BEFORE any LETTA HTTP
        # call: when MEMOIRS_USE_OLLAMA is on we need the SDK env in
        # place so any Letta SDK helpers in this process pick it up.
        if use_ollama_requested():
            if not ollama_is_up():
                self.status = AdapterStatus(ok=False, reason=ollama_install_hint())
                return
            apply_ollama_env()
        url = base_url or os.environ.get("LETTA_BASE_URL")
        if not url:
            self.status = AdapterStatus(
                ok=False,
                reason=("letta needs an LLM-configured runtime; "
                        "set LETTA_BASE_URL to use an existing instance"),
            )
            return
        try:
            import requests  # noqa: F401
        except Exception as e:
            self.status = AdapterStatus(ok=False, reason=f"requests not installed: {e!r}")
            return
        self._base_url = url.rstrip("/")
        if not self._wait_healthy(timeout=timeout):
            self.status = AdapterStatus(
                ok=False,
                reason=f"letta at {self._base_url} not healthy in {timeout:.0f}s",
            )

    def _wait_healthy(self, *, timeout: float) -> bool:
        import requests
        deadline = time.monotonic() + timeout
        url = f"{self._base_url}/v1/health/"
        while time.monotonic() < deadline:
            try:
                r = requests.get(url, timeout=2.0)
                if r.status_code < 500:
                    return True
            except Exception:
                pass
            time.sleep(1.0)
        return False

    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        if not self.status.ok:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        import requests

        # Create a transient agent.
        try:
            r = requests.post(
                f"{self._base_url}/v1/agents/",
                json={"name": "bench-agent"}, timeout=30.0,
            )
            r.raise_for_status()
            self._agent_id = r.json().get("id")
        except Exception as e:
            self.status = AdapterStatus(ok=False, reason=f"letta agent create failed: {e!r}")
            return

        for m in memories:
            self._content_to_id[m.content] = m.id
            try:
                r = requests.post(
                    f"{self._base_url}/v1/agents/{self._agent_id}/archival-memory/",
                    json={"text": m.content}, timeout=30.0,
                )
                r.raise_for_status()
            except Exception as e:
                log.warning("letta add failed for %s: %s", m.id, e)

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok or not self._agent_id:
            return []
        import requests
        try:
            r = requests.get(
                f"{self._base_url}/v1/agents/{self._agent_id}/archival-memory/",
                params={"query": q.query, "limit": top_k}, timeout=30.0,
            )
            r.raise_for_status()
            items = r.json()
        except Exception as e:
            log.warning("letta search failed: %s", e)
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in (items or [])[:top_k]:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            for content, mid in self._content_to_id.items():
                if mid in seen:
                    continue
                if content in text or text in content:
                    out.append(mid)
                    seen.add(mid)
                    break
        return out

    def shutdown(self) -> None:
        if self._agent_id:
            try:
                import requests
                requests.delete(
                    f"{self._base_url}/v1/agents/{self._agent_id}", timeout=10.0,
                )
            except Exception:
                pass


__all__ = ["LettaAdapter"]
