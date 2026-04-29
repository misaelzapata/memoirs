"""Adapter for **Zep / Graphiti** self-hosted via Docker.

Zep / Graphiti requires a Neo4j backend, which means the official
``getzep/graphiti`` repo ships a docker-compose file rather than a
single image. Standing it up reliably from a CI sandbox is fragile
(needs cloning the repo, downloading neo4j, etc.), so this adapter
defaults to **skip with a clean reason** unless an existing Zep
endpoint is reachable on ``http://localhost:8000``.

Pass ``base_url=...`` (or set ``ZEP_BASE_URL``) to point at a running
instance — then the adapter speaks Zep's HTTP API.
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

log = logging.getLogger("bench.adapters.zep")


_DEFAULT_TIMEOUT = 60.0


class ZepAdapter(EngineAdapter):
    name = "zep"

    def __init__(self, *, base_url: Optional[str] = None,
                 timeout: float = _DEFAULT_TIMEOUT) -> None:
        super().__init__()
        # Mirror the Ollama shim used by the other other engines so a single
        # ``--ollama`` flag at the bench level is enough; Zep itself
        # delegates LLM calls to the embedded OpenAI client.
        if use_ollama_requested():
            if not ollama_is_up():
                self.status = AdapterStatus(ok=False, reason=ollama_install_hint())
                return
            apply_ollama_env()
        url = base_url or os.environ.get("ZEP_BASE_URL")
        if not url:
            self.status = AdapterStatus(
                ok=False,
                reason=("zep self-host requires Neo4j docker-compose stack; "
                        "set ZEP_BASE_URL to use an existing instance"),
            )
            return
        try:
            import requests  # noqa: F401
        except Exception as e:
            self.status = AdapterStatus(ok=False, reason=f"requests not installed: {e!r}")
            return
        self._base_url = url.rstrip("/")
        self._session = "bench-session"
        if not self._wait_healthy(timeout=timeout):
            self.status = AdapterStatus(
                ok=False,
                reason=f"zep at {self._base_url} not healthy in {timeout:.0f}s",
            )

    def _wait_healthy(self, *, timeout: float) -> bool:
        import requests
        deadline = time.monotonic() + timeout
        # Zep exposes /healthz; if absent, fall back to /docs.
        for path in ("/healthz", "/docs"):
            url = f"{self._base_url}{path}"
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

        # Zep groups memorias under sessions; we encode the bench ID in
        # `metadata` so search results can map back.
        for m in memories:
            payload = {
                "messages": [
                    {"role": "user", "content": m.content,
                     "metadata": {"bench_id": m.id}},
                ],
            }
            try:
                r = requests.post(
                    f"{self._base_url}/api/v1/sessions/{self._session}/memory",
                    json=payload, timeout=30.0,
                )
                r.raise_for_status()
            except Exception as e:
                log.warning("zep add failed for %s: %s", m.id, e)

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok:
            return []
        import requests

        try:
            r = requests.post(
                f"{self._base_url}/api/v1/sessions/{self._session}/search",
                json={"text": q.query, "limit": top_k}, timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("zep search failed: %s", e)
            return []
        items = data.get("results", []) if isinstance(data, dict) else data
        out: list[str] = []
        for item in items[:top_k]:
            md = (item.get("message", {}).get("metadata", {})
                  if isinstance(item, dict) else {}) or {}
            bid = md.get("bench_id")
            if bid:
                out.append(str(bid))
        return out


__all__ = ["ZepAdapter"]
